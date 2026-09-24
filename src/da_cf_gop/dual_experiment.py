"""Frozen, speaker-disjoint dual-graph comparison on existing CTC outputs.

This experiment regenerates labels from an independent acceptable policy and
therefore recomputes comparator scores on exactly the same new token keys.
Historical v1/Parikh metrics are not reused as comparable headline numbers.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from collections import Counter, defaultdict
from functools import lru_cache
from itertools import product
import json
from pathlib import Path
import platform
import time

import numpy as np
from scipy.special import expit, logsumexp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve, f1_score

from .baselines import conventional_forced_alignment_gop, ctc_sf_sd_norm
from .ctc import ctc_log_probabilities_batch
from .dual_decode import decode
from .dual_lexicon import build_acceptable_graph, evaluate_alignment, load_dictionary
from .dual_prior import generic_prior, estimate_error_prior, serializable_prior
from .metrics import exact_sign_flip_test, paired_speaker_bootstrap
from .parikh_experiment import read_jsonl, write_json, write_jsonl, index_logits, score_event
from .phonology import PHONE_TO_CTC_ID, CTC_ID_TO_PHONE


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "dual_graph_v1.json"
EVENT_NAMES = ("match", "substitution", "deletion")
GRAPH_METHODS = ("M2_generic_viterbi", "M3_learned_viterbi", "M4_dual_graph",
                 "M4_generic_posterior", "M4_generic_tuned", "M4_single_reference")
BASELINE_METHODS = ("conventional_gop_raw", "ctc_sf_sd_norm_raw", "ppaf_rps_raw", "ppaf_ups_raw", "ppaf_acceptable_adapter_raw",
                    "M0_greedy_single", "M1_greedy_acceptable")


def json_safe(value):
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def make_folds(config, cohort):
    speakers = sorted(config["cohorts"][cohort])
    return [{"fold": f"{cohort}_test_{test}", "test_patient": test,
             "development_patient": speakers[(i + 1) % len(speakers)],
             "training_patients": [s for s in speakers if s not in (test, speakers[(i + 1) % len(speakers)])],
             "healthy_train_reserved": config["healthy_train_reserved"],
             "healthy_development": config["healthy_development"], "healthy_test": config["healthy_test"]}
            for i, test in enumerate(speakers)]


def prepare_records(artifacts, cohort, config, output):
    patients = set(config["cohorts"][cohort])
    speakers = patients | set(config["healthy_development"]) | set(config["healthy_test"])
    rows = [r for r in read_jsonl(artifacts / "manifests" / f"{cohort}.jsonl.gz") if r["speaker_id"] in speakers]
    audits = {r["reading_event_id"]: r for r in read_jsonl(
        artifacts / "evaluation" / "dual_graph" / "data_audit" / f"{cohort}.records.jsonl.gz")}
    dictionary, graphs, records = load_dictionary(), {}, []
    for row in rows:
        prompt = row["prompt"]
        if prompt not in graphs:
            graphs[prompt] = build_acceptable_graph(prompt, row["canonical_phones"], dictionary)
        graph = graphs[prompt]
        event = row["reading_event_id"]
        exclusion = audits[event]["quality_exclusion"]
        labels = evaluate_alignment(graph["acceptable_phones"], row["phn_model_phones"], graph["diagnostic_mask"])
        records.append({"event": event, "speaker": row["speaker_id"], "prompt": prompt,
                        "audio_microphone": row["audio_microphone"], "phn_microphone": row["phn_microphone"],
                        "duration_s": row["duration_s"], "canonical_phones": graph["canonical_phones"],
                        "acceptable_phones": graph["acceptable_phones"], "diagnostic_mask": graph["diagnostic_mask"],
                        "labels": labels, "quality_exclusion": exclusion})
    if len({r["event"] for r in records}) != len(records):
        raise ValueError("duplicate reading events")
    sources = index_logits(artifacts / "cache" / "logits", {r["event"] for r in records})
    write_json(output / "acceptable_graphs.json", graphs)
    write_jsonl(output / "evaluation_labels.jsonl.gz", records)
    write_jsonl(output / "acoustic_inputs.jsonl.gz", [acoustic_record(r) for r in records])
    coverage = []
    for speaker in sorted(speakers):
        own = [r for r in records if r["speaker"] == speaker]
        usable = [r for r in own if r["quality_exclusion"] is None]
        tokens = [t for r in usable for t in r["labels"]["tokens"]]
        coverage.append({"speaker": speaker, "utterances": len(own), "quality_usable_utterances": len(usable),
                         "canonical_tokens": sum(len(r["canonical_phones"]) for r in own),
                         "detection_tokens": sum(t["stable_error"] and t["diagnostic"] for t in tokens),
                         "exact_event_tokens": sum(t["stable_event"] and t["diagnostic"] for t in tokens),
                         "errors": sum(t["is_error"] is True and t["stable_error"] and t["diagnostic"] for t in tokens),
                         "variant_positions": sum(sum(len(a) > 1 for a in r["acceptable_phones"]) for r in usable)})
    write_json(output / "coverage.json", coverage)
    return records, sources, coverage


def acoustic_record(row):
    return {k: row[k] for k in ("event", "speaker", "prompt", "canonical_phones", "acceptable_phones")}


@lru_cache(maxsize=4096)
def log_probs_for(path):
    with np.load(path, allow_pickle=False) as data:
        logits = data["logits"].astype(np.float64)
    return logits - logsumexp(logits, axis=1, keepdims=True)


def phone_ids(phones):
    return [PHONE_TO_CTC_ID[p] for p in phones]


def greedy_phones(log_probs):
    labels = np.argmax(log_probs, axis=1)
    return [CTC_ID_TO_PHONE[int(p)] for i, p in enumerate(labels) if p != 0 and (i == 0 or p != labels[i - 1])]


def neutral_predictions(acoustic, log_probs, acceptable):
    alignment = evaluate_alignment(acceptable, greedy_phones(log_probs))
    result = []
    for t in alignment["tokens"]:
        result.append({"phone_index": t["phone_index"], "target": acoustic["canonical_phones"][t["phone_index"]],
                       "gop": 1.0 - 2.0 * float(t["is_error"]) if t["stable_error"] else 0.0,
                       "candidate_event": t["event_type"] if t["stable_event"] else "unspecified",
                       "candidate_phone": t["realized_phone"] if t["event_type"] == "substitution" else None,
                       "candidate_confidence": 1.0 if t["stable_event"] else 0.0,
                       "candidate_margin": 1.0 if t["stable_event"] else 0.0})
    gaps = []
    for gap in alignment["gaps"]:
        slots = {o["ordinal"]: o for o in gap["ordinals"]}
        for ordinal in range(2):
            o = slots.get(ordinal, {"stable": True, "present": False, "phone": None})
            gaps.append({"gap_index": gap["gap_index"], "ordinal": ordinal,
                         "insertion_probability": float(o["present"]) if o["stable"] else 0.5,
                         "candidate_phone": o["phone"], "candidate_confidence": 1.0 if o["stable"] else 0.0})
    return result, gaps


def baseline_predictions(acoustic, log_probs):
    canonical = phone_ids(acoustic["canonical_phones"])
    forced = conventional_forced_alignment_gop(log_probs, canonical)
    sf = ctc_sf_sd_norm(log_probs, canonical)
    ppaf = score_event(log_probs, canonical, device="cuda")
    methods, gap_methods = {}, {}
    for name, values in (("conventional_gop_raw", [v.score for v in forced]),
                         ("ctc_sf_sd_norm_raw", [v.normalized_score for v in sf])):
        methods[name] = [{"phone_index": i, "target": acoustic["canonical_phones"][i], "gop": float(score),
                          "candidate_event": "unspecified", "candidate_phone": None,
                          "candidate_confidence": 0.0, "candidate_margin": 0.0} for i, score in enumerate(values)]
    for mode, tokens in ppaf.items():
        methods[f"ppaf_{mode}_raw"] = []
        for row in tokens:
            alternatives = [(p, lp) for p, lp in row["candidate_log_probabilities"].items() if lp is not None]
            probs = np.exp(np.array([lp for _, lp in alternatives]) - logsumexp([lp for _, lp in alternatives]))
            # Relative candidate support is a diagnostic adapter, not an error probability.
            order = np.argsort(-probs, kind="stable")
            best = alternatives[int(order[0])][0]
            methods[f"ppaf_{mode}_raw"].append({
                "phone_index": row["phone_index"], "target": row["target"], "gop": row["gop"],
                "candidate_event": "deletion" if best == "<DEL>" else "substitution",
                "candidate_phone": None if best == "<DEL>" else best,
                "candidate_confidence": float(probs[order[0]]),
                "candidate_margin": float(probs[order[0]] - (probs[order[1]] if len(order) > 1 else 0))})
    for name, acc in (("M0_greedy_single", [[p] for p in acoustic["canonical_phones"]]),
                      ("M1_greedy_acceptable", acoustic["acceptable_phones"])):
        methods[name], gap_methods[name] = neutral_predictions(acoustic, log_probs, acc)
    methods["ppaf_acceptable_adapter_raw"] = acceptable_ppaf_adapter(acoustic, log_probs, methods["ppaf_ups_raw"])
    return methods, gap_methods


def acceptable_ppaf_adapter(acoustic, log_probs, single_reference_rows):
    """Explicit common normal-variant adapter; other positions stay acceptable.

    Marginalize deduplicated, uniform acceptable pronunciations. For each
    replacement/deletion, marginalize normal variants elsewhere and then take
    the strongest candidate as in PP-AF. This does not permit other errors.
    """
    choices = [phone_ids(a) for a in acoustic["acceptable_phones"]]
    if all(len(a) == 1 for a in choices):
        return [dict(r) for r in single_reference_rows]
    normal = list(product(*choices))
    normal_lp = ctc_log_probabilities_batch(log_probs, normal, device="cuda")
    correct = float(logsumexp(normal_lp) - np.log(len(normal)))
    output = []
    for i, acceptable in enumerate(choices):
        candidates = [q for q in range(1,40) if q not in acceptable] + [None]
        seqs, lengths = [], []
        for q in candidates:
            perturbed = sorted(set(tuple(v[:i]) + (() if q is None else (q,)) + tuple(v[i+1:]) for v in normal))
            seqs.extend(perturbed)
            lengths.append(len(perturbed))
        lp = ctc_log_probabilities_batch(log_probs, seqs, device="cuda")
        candidate_scores, offset = [], 0
        for length in lengths:
            candidate_scores.append(logsumexp(lp[offset:offset+length])-np.log(length))
            offset += length
        probabilities = np.exp(candidate_scores-logsumexp(candidate_scores))
        order = np.argsort(-probabilities,kind="stable")
        q = candidates[int(order[0])]
        output.append({"phone_index": i, "target": acoustic["canonical_phones"][i],
                       "gop": correct-float(max(candidate_scores)),
                       "candidate_event": "deletion" if q is None else "substitution",
                       "candidate_phone": CTC_ID_TO_PHONE.get(q), "candidate_confidence": float(probabilities[order[0]]),
                       "candidate_margin": float(probabilities[order[0]]-probabilities[order[1]])})
    return output


def graph_predictions(acoustic, log_probs, prior, edit_scale, *, single=False):
    started = time.perf_counter()
    acceptable = [[p] for p in acoustic["canonical_phones"]] if single else acoustic["acceptable_phones"]
    output = decode(log_probs, [phone_ids(p) for p in acceptable], prior,
                    canonical_ids=phone_ids(acoustic["canonical_phones"]), max_insertions=2,
                    edit_scale=edit_scale, variant_scale=1.0, include_viterbi=True)
    output["decode_seconds"] = time.perf_counter() - started
    tokens = []
    for i, posterior in enumerate(output["token_posteriors"]):
        errors = np.r_[output["substitution_posteriors"][i, 1:], posterior[2]]
        conditional = errors / max(float(errors.sum()), 1e-300)
        order = np.argsort(-conditional, kind="stable")
        best = int(order[0])
        tokens.append({"phone_index": i, "target": acoustic["canonical_phones"][i],
                       "gop": float(output["token_log_odds"][i]),
                       "graph_posterior_error": float(posterior[1] + posterior[2]),
                       "event_posteriors": posterior.tolist(),
                       "candidate_event": "deletion" if best == 39 else "substitution",
                       "candidate_phone": None if best == 39 else CTC_ID_TO_PHONE[best + 1],
                       "candidate_confidence": float(conditional[best]),
                       "candidate_margin": float(conditional[order[0]] - conditional[order[1]]),
                       "substitution_posteriors": output["substitution_posteriors"][i].tolist()})
    gaps = []
    for i, slots in enumerate(output["insertion_posteriors"]):
        for ordinal, posterior in enumerate(slots):
            probability = float(posterior.sum())
            best = int(np.argmax(posterior[1:])) + 1
            gaps.append({"gap_index": i, "ordinal": ordinal, "insertion_probability": probability,
                         "candidate_phone": CTC_ID_TO_PHONE[best],
                         "candidate_confidence": float(posterior[best] / max(probability, 1e-300))})
    return tokens, gaps, output


def viterbi_predictions(acoustic, decoded):
    # The decoder's max marginals are continuous constrained-best-path scores,
    # not path-summed posteriors. Hard paths are retained separately by decoder.
    rows = []
    scores = decoded.get("max_marginal_token_log_odds")
    events = decoded["viterbi_token_events"]
    phones = decoded["viterbi_token_phones"]
    for i, event in enumerate(events):
        event = int(event)
        score = float(scores[i]) if scores is not None else (1.0 if event == 0 else -1.0)
        rows.append({"phone_index": i, "target": acoustic["canonical_phones"][i], "gop": score,
                     "candidate_event": EVENT_NAMES[event],
                     "candidate_phone": CTC_ID_TO_PHONE.get(int(phones[i])) if event == 1 else None,
                     "candidate_confidence": 1.0, "candidate_margin": 1.0})
    return rows


def viterbi_gap_predictions(decoded):
    return [{"gap_index": i, "ordinal": k, "insertion_probability": float(phone > 0),
             "candidate_phone": CTC_ID_TO_PHONE.get(int(phone)), "candidate_confidence": float(phone > 0)}
            for i, slots in enumerate(decoded["viterbi_insertions"]) for k, phone in enumerate(slots)]


def tagged(rows, acoustic, method, fold):
    return [{"method": method, "fold": fold, "event": acoustic["event"], "speaker": acoustic["speaker"], **r} for r in rows]


def add_gold(predictions, records, *, exact=False):
    lookup = {r["event"]: r for r in records}
    result = []
    for row in predictions:
        source = lookup[row["event"]]
        if source["quality_exclusion"] is not None:
            continue
        gold = source["labels"]["tokens"][row["phone_index"]]
        if not gold["diagnostic"] or not gold["stable_error"]:
            continue
        result.append({**row, "gold_error": bool(gold["is_error"]),
                       "gold_event": gold["event_type"], "gold_realized": gold["realized_phone"],
                       "stable_event": gold["stable_event"]})
    return result


def macro_auprc(rows, speaker_filter=None):
    values = []
    speakers = sorted({r["speaker"] for r in rows} if speaker_filter is None else speaker_filter)
    for speaker in speakers:
        own = [r for r in rows if r["speaker"] == speaker]
        y = [r["gold_error"] for r in own]
        if own and len(set(y)) == 2:
            values.append(average_precision_score(y, [-r["gop"] for r in own]))
    if not values:
        raise ValueError("development population has no identifiable positive/negative class")
    return float(np.mean(values))


def fit_calibration(rows, config):
    counts = Counter(r["speaker"] for r in rows)
    x = np.array([-r["gop"] for r in rows], dtype=float)
    y = np.array([r["gold_error"] for r in rows], dtype=int)
    weight = np.array([len(rows) / (len(counts) * counts[r["speaker"]]) for r in rows])
    if not np.isfinite(x).all() or len(np.unique(y)) != 2:
        raise ValueError("calibration requires finite scores and both classes")
    model = LogisticRegression(C=1.0, class_weight=None, solver="lbfgs", max_iter=1000)
    model.fit(x[:, None], y, sample_weight=weight)
    slope, intercept = float(model.coef_[0, 0]), float(model.intercept_[0])
    probability = expit(slope * x + intercept)
    thresholds = []
    for threshold in config["decision_threshold_grid"]:
        score = np.mean([f1_score(y[[r["speaker"] == s for r in rows]],
                                 probability[[r["speaker"] == s for r in rows]] >= threshold,
                                 zero_division=0) for s in counts])
        thresholds.append((float(score), threshold))
    _, threshold = max(thresholds, key=lambda p: (p[0], -abs(p[1] - 0.5), p[1]))
    gate = None
    eligible = [i for i, r in enumerate(rows) if r["stable_event"] and r["gold_error"]]
    # Precision includes every proposed specific error, including false alarms.
    for confidence in config["specific_confidence_grid"]:
        selected = [i for i, r in enumerate(rows) if r["stable_event"] and probability[i] >= threshold
                    and r["candidate_confidence"] >= confidence and r["candidate_margin"] >= config["specific_margin_floor"]
                    and r["candidate_event"] in ("substitution", "deletion")]
        if not selected:
            continue
        correct = sum(rows[i]["gold_error"] and rows[i]["candidate_event"] == rows[i]["gold_event"]
                      and (rows[i]["candidate_event"] == "deletion" or rows[i]["candidate_phone"] == rows[i]["gold_realized"])
                      for i in selected)
        precision = correct / len(selected)
        coverage = sum(rows[i]["gold_error"] for i in selected) / max(len(eligible), 1)
        if precision >= config["specific_precision_floor"] and coverage >= config["specific_coverage_floor"]:
            gate = {"confidence_threshold": confidence, "development_precision": precision, "development_coverage": coverage}
            break
    return {"slope": slope, "intercept": intercept, "threshold": threshold, "specific_gate": gate,
            "training_speakers": sorted(counts), "n_development_tokens": len(rows), "C": 1.0,
            "class_weight": None, "threshold_scores": thresholds}


def apply_calibration(rows, calibration, config):
    result = []
    for row in rows:
        p = float(expit(calibration["slope"] * -row["gop"] + calibration["intercept"]))
        error = p >= calibration["threshold"]
        gate = calibration["specific_gate"]
        specific = bool(error and gate and row["candidate_event"] in ("substitution", "deletion")
                        and row["candidate_confidence"] >= gate["confidence_threshold"]
                        and row["candidate_margin"] >= config["specific_margin_floor"])
        result.append({**row, "p_error": p, "predicted_error": error,
                       "specific_authorized": specific,
                       "status": row["candidate_event"] if specific else ("suspected_error" if error else "acceptable")})
    return result


def save_baseline_cache(records, sources, directory, *, existing=None):
    """Recompute from audio outputs and text; no selection by any gold score."""
    output, gaps = {}, {}
    for index, row in enumerate(records):
        acoustic = acoustic_record(row)
        old = (existing or {}).get(row["event"])
        if old is not None and all([r["target"] for r in values] == row["canonical_phones"]
                                   for values in old["methods"].values()):
            predictions, insertions = old["methods"], old["gap_methods"]
            predictions["ppaf_acceptable_adapter_raw"] = acceptable_ppaf_adapter(
                acoustic, log_probs_for(sources[row["event"]]), predictions["ppaf_ups_raw"])
        else:
            predictions, insertions = baseline_predictions(acoustic, log_probs_for(sources[row["event"]]))
        output[row["event"]], gaps[row["event"]] = predictions, insertions
        if (index + 1) % 200 == 0:
            print(f"baselines: {index+1}/{len(records)} utterances", flush=True)
    write_jsonl(directory / "baseline_predictions_without_gold.jsonl.gz", [
        {"event": event, "methods": predictions, "gap_methods": gaps[event]} for event, predictions in output.items()])
    return output, gaps


def load_baseline_cache(records, sources, directory):
    path = directory / "baseline_predictions_without_gold.jsonl.gz"
    cached = {r["event"]: r for r in read_jsonl(path)}
    if set(cached) != {r["event"] for r in records}:
        raise ValueError("explicit baseline reuse requires identical event population")
    output, gaps = {}, {}
    for row in records:
        old = cached[row["event"]]
        for name, scores in old["methods"].items():
            if [r["target"] for r in scores] != row["canonical_phones"]:
                raise ValueError("cached comparator has a different canonical reference; recompute")
        output[row["event"]] = old["methods"]
        # Always regenerate the graph adapter because acceptable policy is a
        # separate input from the comparator's single canonical sequence.
        output[row["event"]]["ppaf_acceptable_adapter_raw"] = acceptable_ppaf_adapter(
            acoustic_record(row), log_probs_for(sources[row["event"]]), old["methods"]["ppaf_ups_raw"])
        gaps[row["event"]] = old["gap_methods"]
        for method, acc in (("M0_greedy_single", [[p] for p in row["canonical_phones"]]),
                            ("M1_greedy_acceptable", row["acceptable_phones"])):
            output[row["event"]][method], gaps[row["event"]][method] = neutral_predictions(
                acoustic_record(row), log_probs_for(sources[row["event"]]), acc)
    return output, gaps


def _decode_worker(job):
    acoustic, path, prior, scale, single = job
    return graph_predictions(acoustic, log_probs_for(path), prior, scale, single=single)


def _worker_init():
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)


def run_fold(fold, records, sources, baseline_cache, baseline_gaps, config, directory, *, generic_cache=None, pool=None):
    print(f"{fold['fold']}: train={fold['training_patients']}, dev={fold['development_patient']}", flush=True)
    train = [r for r in records if r["speaker"] in fold["training_patients"] and r["quality_exclusion"] is None]
    dev = [r for r in records if r["speaker"] in [fold["development_patient"], *fold["healthy_development"]]
           and r["quality_exclusion"] is None]
    test = [r for r in records if r["speaker"] in [fold["test_patient"], *fold["healthy_test"]]]
    assert not set(fold["training_patients"]) & {fold["test_patient"], fold["development_patient"]}
    prior_by_alpha = {a: estimate_error_prior(train, training_speakers=fold["training_patients"], alpha=a,
                                            effective_tokens=config["speaker_effective_tokens"],
                                            min_phone_tokens=config["minimum_phone_tokens"],
                                            min_phone_speakers=config["minimum_phone_speakers"]) for a in config["alpha_grid"]}
    generic = generic_prior()
    if generic_cache is None:
        generic_cache = {}
    local_cache = {}

    def infer_many(rows, prior, scale, single=False):
        cache = generic_cache if prior["kind"] == "generic" else local_cache
        keys, missing, jobs = [], [], []
        for row in rows:
            acoustic = acoustic_record(row)
            effective_single = single and any(len(a) > 1 for a in acoustic["acceptable_phones"])
            key = (acoustic["event"], tuple(map(tuple, acoustic["acceptable_phones"])),
                   prior.get("alpha"), scale, effective_single)
            keys.append(key)
            if key not in cache:
                missing.append(key)
                jobs.append((acoustic,sources[row["event"]],prior,scale,effective_single))
        values = map(_decode_worker,jobs) if pool is None else pool.map(_decode_worker,jobs,chunksize=1)
        for count, (key,value) in enumerate(zip(missing,values), 1):
            cache[key] = value
            if count % 200 == 0:
                print(f"    graph decode {count}/{len(jobs)}",flush=True)
        return [cache[key] for key in keys]

    def dev_columns(rows, values):
        predictions, bestpath = [], []
        for row, (scores, _, decoded) in zip(rows,values):
            acoustic = acoustic_record(row)
            predictions.extend(tagged(scores, acoustic, "search", fold["fold"]))
            bestpath.extend(tagged(viterbi_predictions(acoustic, decoded), acoustic, "search", fold["fold"]))
        return predictions,bestpath

    search, development_runs = [], {}
    patient_dev = [r for r in dev if r["speaker"] == fold["development_patient"]]
    for alpha, prior in [(None, generic), *prior_by_alpha.items()]:
        for scale in config["edit_scale_grid"]:
            dev_predictions,bestpath = dev_columns(patient_dev,infer_many(patient_dev,prior,scale))
            evaluated = add_gold(dev_predictions, patient_dev)
            metric = macro_auprc(evaluated, [fold["development_patient"]])
            search.append({"alpha": alpha, "edit_scale": scale, "development_auprc": metric})
            development_runs[(alpha, scale)] = (dev_predictions, bestpath)
            print(f"  dev alpha={alpha}, scale={scale}: AUPRC={metric:.4f}", flush=True)
    candidates = [s for s in search if s["alpha"] is not None]
    best = max(candidates, key=lambda s: (s["development_auprc"], s["alpha"], -s["edit_scale"]))
    best_generic = max([s for s in search if s["alpha"] is None], key=lambda s: (s["development_auprc"], -s["edit_scale"]))
    alpha, scale = best["alpha"], best["edit_scale"]
    selected_prior = prior_by_alpha[alpha]
    for a, weight, prior in ((alpha,scale,selected_prior), (None,scale,generic),
                            (None,best_generic["edit_scale"],generic)):
        development_runs[(a,weight)] = dev_columns(dev,infer_many(dev,prior,weight))
    selections = {
        "M4_dual_graph": development_runs[(alpha, scale)][0],
        "M3_learned_viterbi": development_runs[(alpha, scale)][1],
        "M4_generic_posterior": development_runs[(None, scale)][0],
        "M2_generic_viterbi": development_runs[(None, scale)][1],
        "M4_generic_tuned": development_runs[(None, best_generic["edit_scale"])][0],
    }
    single,_ = dev_columns(dev,infer_many(dev,selected_prior,scale,single=True))
    selections["M4_single_reference"] = single
    for method in BASELINE_METHODS:
        selections[method] = [item for row in dev for item in tagged(
            baseline_cache[row["event"]][method], acoustic_record(row), method, fold["fold"])]
    calibrators = {method: fit_calibration(add_gold(predictions, dev), config) for method, predictions in selections.items()}
    write_json(directory / f"{fold['fold']}.fit.json", json_safe({
        "fold": fold, "selection": best, "generic_selection": best_generic, "development_search": search,
        "prior": serializable_prior(selected_prior), "calibrators": calibrators,
        "development_insertion_lengths": dict(Counter(len(g["inserted_phones"]) for r in dev
            for g in r["labels"]["gaps"] if g["stable"] and g["diagnostic"]))}))
    predictions, gap_predictions = [], []
    learned_outputs = infer_many(test,selected_prior,scale)
    generic_outputs = infer_many(test,generic,scale)
    single_outputs = infer_many(test,selected_prior,scale,single=True)
    tuned_outputs = infer_many(test,generic,best_generic["edit_scale"])
    for index, (row, learned_output, generic_output, single_output, tuned_output) in enumerate(zip(
            test,learned_outputs,generic_outputs,single_outputs,tuned_outputs)):
        acoustic = acoustic_record(row)
        learned, learned_gaps, decoded = learned_output
        common, common_gaps, generic_decoded = generic_output
        one, one_gaps, _ = single_output
        tuned,tuned_gaps,_ = tuned_output
        methods = {"M4_dual_graph": learned, "M3_learned_viterbi": viterbi_predictions(acoustic, decoded),
                   "M4_generic_posterior": common, "M2_generic_viterbi": viterbi_predictions(acoustic, generic_decoded),
                   "M4_single_reference": one, "M4_generic_tuned": tuned, **baseline_cache[row["event"]]}
        for method, values in methods.items():
            predictions.extend(apply_calibration(tagged(values, acoustic, method, fold["fold"]), calibrators[method], config))
        for method, values in (("M4_dual_graph", learned_gaps), ("M4_generic_posterior", common_gaps),
                               ("M4_generic_tuned", tuned_gaps), ("M4_single_reference", one_gaps),
                               ("M3_learned_viterbi", viterbi_gap_predictions(decoded)),
                               ("M2_generic_viterbi", viterbi_gap_predictions(generic_decoded))):
            gap_predictions.extend(tagged(values, acoustic, method, fold["fold"]))
        for method, values in baseline_gaps[row["event"]].items():
            gap_predictions.extend(tagged(values, acoustic, method, fold["fold"]))
        if (index+1) % 100 == 0:
            print(f"  test {index+1}/{len(test)}", flush=True)
    return predictions, gap_predictions, {"fold": fold, "selection": best, "generic_selection": best_generic,
                                         "development_search": search, "calibrators": calibrators}


def token_metrics(rows):
    y = np.array([r["gold_error"] for r in rows], dtype=int)
    scores = np.array([-r["gop"] for r in rows])
    p = np.clip([r["p_error"] for r in rows], 1e-12, 1-1e-12)
    pred = np.array([r["predicted_error"] for r in rows])
    if not np.isfinite(scores).all():
        raise ValueError("nonfinite scored token")
    fpr_at_recall = None
    if len(set(y)) == 2:
        fpr, tpr, _ = roc_curve(y, scores)
        fpr_at_recall = float(min(fpr[tpr >= 0.8]))
    return {"n": len(rows), "n_error": int(y.sum()), "fpr_at_80pct_recall": fpr_at_recall,
            "auprc": float(average_precision_score(y, scores)) if len(set(y)) == 2 else None,
            "auroc": float(roc_auc_score(y, scores)) if len(set(y)) == 2 else None,
            "f1": float(f1_score(y, pred, zero_division=0)),
            "brier": float(np.mean((p-y)**2)), "nll": float(np.mean(-y*np.log(p)-(1-y)*np.log1p(-p))),
            "false_positive_rate": float(pred[y == 0].mean()) if (y == 0).any() else None,
            "false_negative_rate": float((~pred[y == 1]).mean()) if (y == 1).any() else None}


def diagnosis_metrics(rows):
    exact = [r for r in rows if r["stable_event"]]
    output = {}
    for event in ("substitution", "deletion"):
        gold = sum(r["gold_event"] == event for r in exact)
        proposed = [r for r in exact if r["predicted_error"] and r["candidate_event"] == event]
        correct = sum(r["gold_event"] == event and (event == "deletion" or r["candidate_phone"] == r["gold_realized"])
                      for r in proposed)
        precision, recall = correct / max(len(proposed), 1), correct / max(gold, 1)
        output[event] = {"gold": gold, "predicted": len(proposed), "true_positive": correct,
                         "precision": precision, "recall": recall, "f1": 2*precision*recall/max(precision+recall, 1e-12)}
    specific = [r for r in exact if r["specific_authorized"]]
    correct = sum(r["gold_event"] == r["candidate_event"] and
                  (r["candidate_event"] == "deletion" or r["candidate_phone"] == r["gold_realized"]) for r in specific)
    output["selective_specific"] = {"coverage_of_exact_tokens": len(specific)/max(len(exact), 1),
                                    "precision": correct/len(specific) if specific else None, "n": len(specific)}
    risk = []
    ordered = sorted(rows, key=lambda r: -(r["p_error"] if r["predicted_error"] else 1-r["p_error"]))
    for coverage in (0.1, 0.25, 0.5, 0.75, 1.0):
        keep = ordered[:max(1, int(len(ordered)*coverage))]
        risk.append({"coverage": coverage, "binary_risk": float(np.mean([
            r["predicted_error"] != r["gold_error"] for r in keep]))})
    output["risk_coverage"] = risk
    cumulative_risk = np.cumsum([r["predicted_error"] != r["gold_error"] for r in ordered]) / np.arange(1,len(ordered)+1)
    output["risk_coverage_auc"] = float(np.trapz(np.r_[0., cumulative_risk], np.linspace(0,1,len(ordered)+1)))
    return output


def evaluate_results(predictions, records, cohort, config):
    evaluated = add_gold(predictions, records)
    patients = set(config["cohorts"][cohort])
    groups = defaultdict(list)
    for row in evaluated:
        groups[row["method"]].append(row)
    all_methods = set(BASELINE_METHODS) | set(GRAPH_METHODS)
    if set(groups) != all_methods:
        raise ValueError("missing experiment method")
    reference_keys = None
    for method, rows in groups.items():
        keys = [(r["fold"], r["event"], r["phone_index"], r["target"]) for r in rows]
        if len(keys) != len(set(keys)):
            raise ValueError(f"duplicate token in {method}")
        if reference_keys is not None and set(keys) != reference_keys:
            raise ValueError("methods do not share identical scored token keys")
        reference_keys = set(keys)
    results = []
    for method, rows in sorted(groups.items()):
        patient_rows = [r for r in rows if r["speaker"] in patients]
        per_speaker = [{"speaker": s, **token_metrics([r for r in patient_rows if r["speaker"] == s])} for s in sorted(patients)]
        macro = {key: float(np.mean([r[key] for r in per_speaker])) for key in (
            "auprc", "auroc", "f1", "brier", "nll", "false_positive_rate", "false_negative_rate", "fpr_at_80pct_recall")}
        healthy = [{"fold": f, **token_metrics([r for r in rows if r["fold"] == f and r["speaker"] not in patients])}
                   for f in sorted({r["fold"] for r in rows})]
        diagnosis_supported = method not in ("conventional_gop_raw", "ctc_sf_sd_norm_raw")
        results.append({"method": method, "macro": macro, "per_speaker": per_speaker,
                        "pooled": token_metrics(patient_rows),
                        "diagnosis": diagnosis_metrics(patient_rows) if diagnosis_supported else None,
                        "diagnosis_capability": ("paper score plus explicit maximum-alternative adapter" if method.startswith("ppaf")
                            else "native graph/sequence candidates" if diagnosis_supported else "unsupported: scalar detection score only"),
                        "healthy_test_per_patient_model": healthy,
                        "healthy_note": "same held-out MC04 evaluated by each patient-fold model; not independent extra speakers"})
    by_method = {r["method"]: {s["speaker"]: s["auprc"] for s in r["per_speaker"]} for r in results}
    comparisons = []
    for baseline in sorted(all_methods - {"M4_dual_graph"}):
        a, b = by_method["M4_dual_graph"], by_method[baseline]
        differences = {s: a[s]-b[s] for s in a}
        comparisons.append({"a": "M4_dual_graph", "b": baseline,
                            "auprc": paired_speaker_bootstrap(a,b,n_bootstrap=config["bootstrap_replicates"],seed=config["seed"]),
                            "exact_sign_flip": exact_sign_flip_test(differences),
                            "speakers_improved": sum(x > 0 for x in differences.values())})
    return evaluated, {"cohort": cohort, "results": results, "paired_comparisons": comparisons,
                       "same_token_keys": True, "acoustic_track": config["acoustic_track"],
                       "gold_policy": "independent all-optimal unit-cost alignment to frozen acceptable graph",
                       "statistical_scope": "exploratory new protocol; small patients; comparisons unadjusted; sensitivity7 overlaps primary5"}


def evaluate_insertions(predictions, records, patients):
    lookup = {r["event"]: r for r in records}
    groups = defaultdict(list)
    for row in predictions:
        source = lookup[row["event"]]
        if row["speaker"] not in patients or source["quality_exclusion"] is not None:
            continue
        gap = source["labels"]["gaps"][row["gap_index"]]
        if not gap["diagnostic"] or not gap["stable"]:
            continue
        gold = gap["inserted_phones"]
        if len(gold) > 2:
            continue
        q = gold[row["ordinal"]] if row["ordinal"] < len(gold) else None
        groups[row["method"]].append({**row, "gold_phone": q})
    results = []
    for method, rows in sorted(groups.items()):
        gold = sum(r["gold_phone"] is not None for r in rows)
        positive = [r for r in rows if r["insertion_probability"] >= 0.5]
        correct = sum(r["gold_phone"] is not None and r["gold_phone"] == r["candidate_phone"] for r in positive)
        precision, recall = correct/max(len(positive),1), correct/max(gold,1)
        results.append({"method": method, "n_gap_slots": len(rows), "gold_insertions": gold,
                        "precision": precision, "recall": recall,
                        "f1": 2*precision*recall/max(precision+recall,1e-12),
                        "cap_hit_rate": (float(np.mean([r["insertion_probability"] >= 0.5 for r in rows if r["ordinal"] == 1]))
                                         if any(r["ordinal"] == 1 for r in rows) else None),
                        "threshold": 0.5, "threshold_note": "fixed exploratory graph posterior threshold, not calibrated"})
    return results


def run(artifacts=ROOT/"artifacts", *, cohort="primary5", config_path=DEFAULT_CONFIG, reuse_baselines=False, workers=6):
    import torch
    from threadpoolctl import threadpool_limits
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    artifacts = Path(artifacts)
    output = artifacts / "evaluation" / "dual_graph" / config["version"] / cohort
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    torch.set_num_threads(4)
    write_json(output / "config.json", config)
    folds = make_folds(config, cohort)
    write_json(output / "folds.json", folds)
    records, sources, coverage = prepare_records(artifacts, cohort, config, output)
    print(f"{cohort}: {len(records)} acoustic events; new labels/policy frozen", flush=True)
    with threadpool_limits(limits=4):
        baseline, baseline_gaps = (load_baseline_cache(records,sources,output) if reuse_baselines
                                   else save_baseline_cache(records, sources, output))
        predictions, gaps, fitted, generic_cache = [], [], [], {}
        with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as pool:
            for fold in folds:
                p, g, state = run_fold(fold, records, sources, baseline, baseline_gaps, config, output / "folds",
                                      generic_cache=generic_cache, pool=pool)
                predictions.extend(p)
                gaps.extend(g)
                fitted.append(state)
                write_jsonl(output / "folds" / f"{fold['fold']}.predictions_without_gold.jsonl.gz", p)
    write_jsonl(output / "predictions_without_gold.jsonl.gz", predictions)
    write_jsonl(output / "gap_predictions_without_gold.jsonl.gz", gaps)
    evaluated, report = evaluate_results(predictions, records, cohort, config)
    report["insertion_diagnostics"] = evaluate_insertions(gaps, records, set(config["cohorts"][cohort]))
    report.update({"coverage": coverage, "folds": fitted, "elapsed_seconds": round(time.monotonic()-started,2),
                   "hardware": platform.platform(), "acoustic_inference_time": "not remeasured: pre-existing raw logit cache",
                   "candidate_scope": "one-to-one dictionary-backed weak-vowel variants; unsupported word blocks non-diagnostic"})
    write_jsonl(output / "phone_predictions.jsonl.gz", evaluated)
    write_json(output / "metrics.json", json_safe(report))
    from .parikh_experiment import save_tables
    save_tables(output, report)
    for result in report["results"]:
        print(f"{result['method']}: macro AUPRC={result['macro']['auprc']:.6f}; F1={result['macro']['f1']:.6f}")
    print(f"Results: {output}")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=("primary5", "sensitivity7"), default="primary5")
    parser.add_argument("--artifacts", type=Path, default=ROOT/"artifacts")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--reuse-baselines", action="store_true", help="explicitly reuse this run's fixed raw-backend baseline scores")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args(argv)
    run(args.artifacts, cohort=args.cohort, config_path=args.config, reuse_baselines=args.reuse_baselines, workers=args.workers)


if __name__ == "__main__":
    main()
