"""Retrospective, speaker-disjoint DA-CF evidence adaptation.

`prepare` extracts only fixed acoustic/graph evidence. `predict` fits on train
and development speakers and writes every outer prediction without gold.
`evaluate` is a separate invocation; it never fits or selects a model.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .dual_experiment import add_gold, diagnosis_metrics, token_metrics, json_safe
from .dual_prior import estimate_error_prior, generic_prior, serializable_prior
from .metrics import paired_speaker_bootstrap, exact_sign_flip_test
from .parikh_experiment import read_jsonl, write_json, write_jsonl
from .phonology import PHONE_TO_CTC_ID, CTC_ID_TO_PHONE

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "artifacts/evaluation/dual_graph/dual_graph_v1"
OUT = ROOT / "artifacts/evaluation/final_v2"
CONFIG = ROOT / "configs/final_v2.json"
RAW = ("conventional_gop_raw", "ctc_sf_sd_norm_raw", "ppaf_rps_raw", "ppaf_ups_raw")
ADAPTED = ("sf_da", "ppaf_da", "isolated_da", "graph_da", "graph_cf_da", "da_cf_gop")
DIRECT = (*RAW, "ctc_greedy", "graph_generic", "graph_cf")
METHODS = (*DIRECT, *ADAPTED)


def key(row):
    return row["event"], row["phone_index"]


def prepare():
    """Reuse generic *raw* evidence, checking settings rather than method names."""
    directory = OLD / "sensitivity7"
    config = json.loads((directory / "config.json").read_text())
    assert config["generic_operation_probabilities"] == [0.90, 0.07, 0.03]
    assert config["max_insertions_per_gap"] == 2
    patients = set(config["cohorts"]["sensitivity7"])
    baseline = {r["event"]: r["methods"] for r in read_jsonl(
        directory / "baseline_predictions_without_gold.jsonl.gz") if r["event"].split("_")[0] in patients}
    evidence, provenance = {}, []
    for fold in read_jsonl_folds(directory):
        fitpath = directory / "folds" / (fold["fold"] + ".fit.json")
        fit = json.loads(fitpath.read_text())
        # The historical generic ablation inherits this scale. This experiment
        # fixes 1.0 in advance and uses neither the selection rule nor its scores.
        assert fit["selection"]["edit_scale"] == 1.0
        path = directory / "folds" / (fold["fold"] + ".predictions_without_gold.jsonl.gz")
        for row in read_jsonl(path):
            if row["speaker"] != fold["test_patient"]:
                continue
            if row["method"] not in ("M4_generic_posterior", "M2_generic_viterbi"):
                continue
            k = key(row)
            if k not in evidence:
                b = baseline[row["event"]]
                i = row["phone_index"]
                evidence[k] = {field: row[field] for field in ("event", "speaker", "phone_index", "target")}
                evidence[k]["baseline"] = {name: b[name][i]["gop"] for name in RAW}
                evidence[k]["greedy"] = {field: b["M1_greedy_acceptable"][i][field] for field in (
                    "gop", "candidate_event", "candidate_phone", "candidate_confidence")}
            if row["method"] == "M4_generic_posterior":
                evidence[k].update({field: row[field] for field in (
                    "gop", "event_posteriors", "substitution_posteriors")})
            else:
                evidence[k]["viterbi_gop"] = row["gop"]
        provenance.append({"file": str(path.relative_to(ROOT)), "raw_graph_scale": 1.0,
                           "speaker": fold["test_patient"], "prior": "generic",
                           "discarded": ["historical p_error", "threshold", "calibration", "learned graph"]})
    records = {r["event"]: r for r in read_jsonl(directory / "evaluation_labels.jsonl.gz")}
    for row in evidence.values():
        # Acceptable phones are dictionary/text inputs. No evaluation mask or
        # actual realization is retained in the evidence artifact.
        row["acceptable"] = records[row["event"]]["acceptable_phones"][row["phone_index"]]
        assert "event_posteriors" in row and "viterbi_gop" in row
    OUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT / "fixed_evidence_without_gold.jsonl.gz", sorted(evidence.values(), key=key))
    write_json(OUT / "fixed_evidence_provenance.json", {
        "settings": {"edit_scale": 1.0, "operation": [.9, .07, .03], "insertions_per_gap": 2},
        "sources": provenance, "n_tokens": len(evidence), "all_phones": 39,
        "meaning": "Deterministic generic graph evidence; historical fitted probabilities are discarded."})
    print(f"Prepared {len(evidence)} raw patient tokens", flush=True)


def read_jsonl_folds(directory):
    return json.loads((directory / "folds.json").read_text())


def prior_for(records, speakers, config):
    selected = [r for r in records if r["speaker"] in speakers and r["quality_exclusion"] is None]
    return estimate_error_prior(selected, training_speakers=speakers,
        alpha=config["cf_alpha"], effective_tokens=config["cf_effective_tokens"],
        min_phone_tokens=config["cf_min_phone_tokens"], min_phone_speakers=config["cf_min_phone_speakers"])


def corrected_events(row, prior):
    """Local prior replacement; NOT globally reweighted graph inference.

    If every learned prior equals the generic prior this is the identity,
    including positions with more than one acceptable realization.
    """
    p = PHONE_TO_CTC_ID[row["target"]]
    allowed = [PHONE_TO_CTC_ID[q] for q in row["acceptable"]]
    q = np.asarray(prior["sub_probs"][p], dtype=float).copy()
    q[0] = 0
    q[allowed] = 0
    q /= q.sum()
    mass = np.r_[row["event_posteriors"][0], row["substitution_posteriors"][1:], row["event_posteriors"][2]]
    ratio = np.r_[prior["op_probs"][p, 0] / .9,
                  prior["op_probs"][p, 1] * q[1:] / (.07 / (39 - len(allowed))),
                  prior["op_probs"][p, 2] / .03]
    adjusted = mass * ratio
    return adjusted / adjusted.sum()


def log_ratio(a, b):
    return np.log(np.maximum(a, 1e-300)) - np.log(np.maximum(b, 1e-300))


def continuous_features(rows, priors, method):
    data, corrected = [], []
    for row in rows:
        prior = priors[row["speaker"]] if isinstance(priors, dict) and "op_probs" not in priors else priors
        cf = corrected_events(row, prior)
        corrected.append(cf)
        b = row["baseline"]
        isolated = [-b[name] for name in RAW[1:]]
        ok, sub, dele = row["event_posteriors"]
        graph = [-row["gop"], -row["viterbi_gop"], log_ratio(sub, ok), log_ratio(dele, ok)]
        correction = [log_ratio(cf[1:-1].sum(), cf[0]), log_ratio(cf[-1], cf[0])]
        if method == "sf_da":
            features = [isolated[0]]
        elif method == "ppaf_da":
            features = [isolated[1]]
        elif method == "isolated_da":
            features = isolated
        elif method == "graph_da":
            features = graph
        elif method == "graph_cf_da":
            features = graph + correction
        elif method == "da_cf_gop":
            features = isolated + graph + correction
        else:
            raise ValueError(method)
        data.append(features)
    # Fixed monotone compression handles the scale of complete-sequence CTC
    # evidence without selecting caps from held-out labels.
    x = np.asarray(data, dtype=float)
    return np.sign(x) * np.log1p(np.abs(x)), np.asarray(corrected)


def phone_design(rows, numeric, *, interactions):
    onehot = np.eye(39)[[PHONE_TO_CTC_ID[r["target"]] - 1 for r in rows]]
    if interactions:
        return np.c_[numeric, onehot, (numeric[:, :, None] * onehot[:, None, :]).reshape(len(rows), -1)]
    return np.c_[numeric, onehot]


def speaker_weights(rows):
    counts = Counter(r["speaker"] for r in rows)
    return np.asarray([len(rows) / len(counts) / counts[r["speaker"]] for r in rows])


def fit_predictor(train, dev, xtrain, xdev, candidate):
    scaler = StandardScaler().fit(xtrain)
    logistic = candidate["family"] == "logistic"
    xt = phone_design(train, scaler.transform(xtrain), interactions=logistic)
    xd = phone_design(dev, scaler.transform(xdev), interactions=logistic)
    if logistic:
        model = LogisticRegression(C=candidate["C"], max_iter=1500, solver="lbfgs")
    else:
        model = HistGradientBoostingClassifier(max_iter=candidate["iterations"], max_leaf_nodes=candidate["leaves"],
            min_samples_leaf=40, l2_regularization=10., learning_rate=.05, early_stopping=False, random_state=20260905)
    model.fit(xt, [r["gold_error"] for r in train], sample_weight=speaker_weights(train))
    score = model.decision_function(xd)
    return {"scaler": scaler, "model": model, "interactions": logistic}, score


def model_score(fitted, rows, x):
    return fitted["model"].decision_function(phone_design(rows, fitted["scaler"].transform(x),
        interactions=fitted["interactions"]))


def calibration(score, labels):
    """Nondecreasing Platt calibration with a development-only F1 threshold."""
    score, labels = np.asarray(score), np.asarray(labels, dtype=float)
    def loss(params):
        z = params[0] * score + params[1]
        return np.mean(np.logaddexp(0, z) - labels * z) + .0001 * params[0] ** 2
    prevalence = np.clip(labels.mean(), 1e-6, 1 - 1e-6)
    optimum = minimize(loss, [1., float(logit(prevalence))], method="L-BFGS-B", bounds=[(0., None), (None, None)])
    slope, intercept = optimum.x
    p = expit(slope * score + intercept)
    thresholds = [(float(f1_score(labels, p >= t, zero_division=0)), float(t)) for t in np.arange(1, 100) / 100]
    best = max(thresholds, key=lambda pair: (pair[0], -abs(pair[1] - .5), pair[1]))
    return {"slope": float(slope), "intercept": float(intercept), "threshold": best[1], "development_f1": best[0]}


def event_model(train, x):
    mask = np.asarray([r["stable_event"] and r["gold_error"] for r in train])
    scaler = StandardScaler().fit(x[mask])
    model = LogisticRegression(C=.1, max_iter=1000)
    model.fit(scaler.transform(x[mask]), [r["gold_event"] == "substitution" for r in np.asarray(train)[mask]],
        sample_weight=speaker_weights(list(np.asarray(train)[mask])))
    return {"scaler": scaler, "model": model, "training_error_tokens": int(mask.sum())}


def diagnostic_values(event_fit, x, cf):
    psub = event_fit["model"].predict_proba(event_fit["scaler"].transform(x))[:, 1]
    q = cf[:, 1:-1]
    conditional = q / q.sum(axis=1, keepdims=True)
    best = np.argmax(conditional, axis=1)
    return psub, best, conditional[np.arange(len(cf)), best]


def choose_type_threshold(dev, p_error, calibration_fit, psub, phones):
    # Fix event/phone output using only the development patient's exact labels.
    options = []
    for threshold in (.25, .5, .75):
        selected = p_error >= calibration_fit["threshold"]
        scores = []
        for kind in ("substitution", "deletion"):
            truth = np.asarray([r["gold_event"] == kind and r["stable_event"] for r in dev])
            proposed = selected & ((psub >= threshold) if kind == "substitution" else (psub < threshold))
            proposed &= [r["stable_event"] for r in dev]
            correct_phone = np.asarray([CTC_ID_TO_PHONE[int(q) + 1] == r["gold_realized"] for q, r in zip(phones, dev)])
            tp = int(np.sum(truth & proposed & (correct_phone if kind == "substitution" else True)))
            scores.append(2 * tp / max(int(truth.sum()) + int(proposed.sum()), 1))
        options.append((float(np.mean(scores)), threshold))
    return max(options, key=lambda pair: (pair[0], -abs(pair[1] - .5)))[1]


def output_rows(raw, method, fold, score, cal, *, cf=None, diagnostic=None, type_threshold=.5, specific_gate=None):
    p = expit(cal["slope"] * np.asarray(score) + cal["intercept"])
    output = []
    for i, r in enumerate(raw):
        if method == "ctc_greedy":
            event, phone = r["greedy"]["candidate_event"], r["greedy"]["candidate_phone"]
            conf = r["greedy"]["candidate_confidence"]
        elif diagnostic is not None:
            subprob, phones, phoneprob = diagnostic
            issub = subprob[i] >= type_threshold
            event = "substitution" if issub else "deletion"
            phone = CTC_ID_TO_PHONE[int(phones[i]) + 1] if issub else None
            conf = float(subprob[i] * phoneprob[i] if issub else 1 - subprob[i])
        elif cf is not None:
            winner = int(np.argmax(cf[i, 1:]))
            event = "deletion" if winner == 39 else "substitution"
            phone = None if winner == 39 else CTC_ID_TO_PHONE[winner + 1]
            conf = float(cf[i, winner + 1] / cf[i, 1:].sum())
        else:
            event, phone, conf = "unspecified", None, 0.
        error = bool(p[i] >= cal["threshold"])
        joint = float(p[i] * conf)
        output.append({**{field: r[field] for field in ("event", "speaker", "phone_index", "target")},
            "method": method, "fold": fold, "gop": -float(score[i]), "p_error": float(p[i]),
            "predicted_error": error, "candidate_event": event, "candidate_phone": phone,
            "candidate_confidence": conf, "specific_confidence": joint,
            "specific_authorized": bool(error and specific_gate is not None and joint >= specific_gate)})
    return output


def select_specific_gate(dev_predictions, dev):
    lookup = {key(r): r for r in dev}
    for gate in (.5, .6, .7, .8, .9, .95):
        selected = [r for r in dev_predictions if r["predicted_error"] and r["specific_confidence"] >= gate
            and lookup[key(r)]["stable_event"] and r["candidate_event"] in ("substitution", "deletion")]
        if len(selected) < 5:
            continue
        correct = sum(r["candidate_event"] == lookup[key(r)]["gold_event"] and
            (r["candidate_event"] == "deletion" or r["candidate_phone"] == lookup[key(r)]["gold_realized"]) for r in selected)
        if correct / len(selected) >= .8:
            return gate
    return None


def retained_rows(rows, config):
    """Apply documented input-quality exclusions before splitting or fitting."""
    excluded = set(config.get("excluded_events", []))
    return [row for row in rows if row["event"] not in excluded]


def predict(cohort, output_root=OUT, config_path=CONFIG):
    config_path = Path(config_path)
    config = json.loads(config_path.read_text())
    directory = Path(output_root) / cohort
    directory.mkdir(parents=True, exist_ok=True)
    assert not (directory / "metrics.json").exists(), "Do not refit a run whose outer results have been evaluated."
    write_json(directory / "config.json", config)
    write_json(directory / "run_identity.json", {"config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    speakers = set(config["cohorts"][cohort])
    evidence = [r for r in retained_rows(read_jsonl(OUT / "fixed_evidence_without_gold.jsonl.gz"), config)
                if r["speaker"] in speakers]
    records = [r for r in retained_rows(read_jsonl(OLD / cohort / "evaluation_labels.jsonl.gz"), config)
               if r["speaker"] in speakers]
    folds = read_jsonl_folds(OLD / cohort)
    write_json(directory / "folds.json", folds)
    started, all_predictions = time.perf_counter(), []
    with threadpool_limits(limits=4):
        for fold in folds:
            train_speakers, dev_speaker, test_speaker = fold["training_patients"], fold["development_patient"], fold["test_patient"]
            assert not set(train_speakers) & {dev_speaker, test_speaker}
            train = add_gold([r for r in evidence if r["speaker"] in train_speakers],
                             [r for r in records if r["speaker"] in train_speakers])
            dev = add_gold([r for r in evidence if r["speaker"] == dev_speaker],
                           [r for r in records if r["speaker"] == dev_speaker])
            test = [r for r in evidence if r["speaker"] == test_speaker]
            prior = prior_for(records, train_speakers, config)
            crossfit = {s: prior_for(records, [other for other in train_speakers if other != s], config) for s in train_speakers}
            fitted = {"fold": fold, "cf_prior": serializable_prior(prior),
                "training_crossfit_speakers": {s: p["training_speakers"] for s, p in crossfit.items()},
                "n_train": len(train), "n_development": len(dev), "n_test_predictions": len(test), "methods": {}}
            predictions = []
            print(f"{cohort}/{test_speaker}: train {train_speakers}, dev {dev_speaker}", flush=True)
            for method in METHODS:
                xt, cft = continuous_features(train, crossfit, method if method in ADAPTED else "da_cf_gop")
                xd, cfd = continuous_features(dev, prior, method if method in ADAPTED else "da_cf_gop")
                xx, cfx = continuous_features(test, prior, method if method in ADAPTED else "da_cf_gop")
                search = []
                diagnostic_dev = diagnostic_test = None
                type_threshold = .5
                if method in ADAPTED:
                    candidates = []
                    for candidate in config["classifier_candidates"]:
                        model, score_dev = fit_predictor(train, dev, xt, xd, candidate)
                        ap = float(average_precision_score([r["gold_error"] for r in dev], score_dev))
                        search.append({"parameters": candidate, "development_ap": ap})
                        candidates.append((ap, model, score_dev))
                    index = max(range(len(candidates)), key=lambda i: (candidates[i][0], -i))
                    _, model, score_dev = candidates[index]
                    score_test = model_score(model, test, xx)
                    diagnosis = None
                    if method in ("graph_da", "graph_cf_da", "da_cf_gop"):
                        diagnosis = event_model(train, xt)
                        if method == "graph_da":
                            generic = generic_prior()
                            cfd = np.asarray([corrected_events(r, generic) for r in dev])
                            cfx = np.asarray([corrected_events(r, generic) for r in test])
                        diagnostic_dev = diagnostic_values(diagnosis, xd, cfd)
                        diagnostic_test = diagnostic_values(diagnosis, xx, cfx)
                    model_path = directory / "models" / f"{test_speaker}.{method}.joblib"
                    model_path.parent.mkdir(exist_ok=True)
                    joblib.dump({"binary": model, "event": diagnosis, "feature_method": method}, model_path)
                    selected = search[index]
                else:
                    def raw_score(rows, cf):
                        if method in RAW:
                            return np.asarray([-r["baseline"][method] for r in rows])
                        if method == "ctc_greedy":
                            return np.asarray([-r["greedy"]["gop"] for r in rows])
                        if method == "graph_generic":
                            return np.asarray([-r["gop"] for r in rows])
                        return log_ratio(cf[:, 1:].sum(axis=1), cf[:, 0])
                    score_dev, score_test = raw_score(dev, cfd), raw_score(test, cfx)
                    selected = None
                cal = calibration(score_dev, [r["gold_error"] for r in dev])
                if diagnostic_dev is not None:
                    type_threshold = choose_type_threshold(dev, expit(cal["slope"] * score_dev + cal["intercept"]),
                        cal, diagnostic_dev[0], diagnostic_dev[1])
                diagnostic_supported = method in ("graph_da", "graph_cf_da", "da_cf_gop", "graph_generic", "graph_cf")
                if method == "graph_generic":
                    generic = generic_prior()
                    cfd = np.asarray([corrected_events(r, generic) for r in dev])
                    cfx = np.asarray([corrected_events(r, generic) for r in test])
                dev_predictions = output_rows(dev, method, fold["fold"], score_dev, cal,
                    cf=cfd if diagnostic_supported else None, diagnostic=diagnostic_dev, type_threshold=type_threshold)
                gate = select_specific_gate(dev_predictions, dev)
                predictions.extend(output_rows(test, method, fold["fold"], score_test, cal,
                    cf=cfx if diagnostic_supported else None, diagnostic=diagnostic_test,
                    type_threshold=type_threshold, specific_gate=gate))
                fitted["methods"][method] = {"search": search, "selected": selected, "calibration": cal,
                    "specific_gate": gate, "type_threshold": type_threshold,
                    "supervised_fit_speakers": train_speakers if method in ADAPTED else [],
                    "selection_calibration_speakers": [dev_speaker]}
                print(f"  {method}: selected on dev only", flush=True)
            write_json(directory / f"{test_speaker}.fit.json", json_safe(fitted))
            write_jsonl(directory / f"{test_speaker}.predictions_without_gold.jsonl.gz", predictions)
            all_predictions.extend(predictions)
    write_jsonl(directory / "predictions_without_gold.jsonl.gz", all_predictions)
    write_json(directory / "prediction_complete.json", {"n_predictions": len(all_predictions),
        "elapsed_seconds": time.perf_counter() - started, "outer_evaluation_performed": False})
    print(f"All predictions written for {cohort}; evaluation is a separate command.", flush=True)


def risk_metrics(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[r["p_error"] if r["predicted_error"] else 1 - r["p_error"]].append(r["predicted_error"] != r["gold_error"])
    mistakes = np.asarray([np.mean(groups[k]) for k in sorted(groups, reverse=True) for _ in groups[k]])
    risk = np.cumsum(mistakes) / np.arange(1, len(mistakes) + 1)
    return {"aurc": float(np.trapz(np.r_[0., risk], np.arange(len(risk) + 1) / len(risk))),
        "curve": [{"coverage": c, "risk": float(risk[max(1, int(len(risk) * c)) - 1])}
            for c in (.1, .25, .5, .75, 1.)]}


def evaluate(cohort, output_root=OUT):
    directory = Path(output_root) / cohort
    config = json.loads((directory / "config.json").read_text())
    assert (directory / "prediction_complete.json").exists()
    predictions = list(read_jsonl(directory / "predictions_without_gold.jsonl.gz"))
    assert not any(any(k.startswith("gold") for k in r) for r in predictions)
    records = retained_rows(read_jsonl(OLD / cohort / "evaluation_labels.jsonl.gz"), config)
    assert not set(config.get("excluded_events", [])) & {r["event"] for r in predictions}
    evaluated = add_gold(predictions, records)
    results = []
    keys = None
    for method in METHODS:
        rows = [r for r in evaluated if r["method"] == method]
        current = {key(r) for r in rows}
        assert len(current) == len(rows)
        assert keys is None or keys == current
        keys = current
        per_speaker = []
        for speaker in sorted(config["cohorts"][cohort]):
            own = [r for r in rows if r["speaker"] == speaker]
            per_speaker.append({"speaker": speaker, **token_metrics(own), "risk": risk_metrics(own),
                "diagnosis": diagnosis_metrics(own)})
        macro = {k: float(np.mean([r[k] for r in per_speaker])) for k in (
            "auprc", "auroc", "f1", "brier", "nll", "false_positive_rate", "false_negative_rate")}
        macro["aurc"] = float(np.mean([r["risk"]["aurc"] for r in per_speaker]))
        macro["sub_f1"] = float(np.mean([r["diagnosis"]["substitution"]["f1"] for r in per_speaker]))
        macro["del_f1"] = float(np.mean([r["diagnosis"]["deletion"]["f1"] for r in per_speaker]))
        truth_subs = [r for r in rows if r["stable_event"] and r["gold_event"] == "substitution"]
        results.append({"method": method, "macro": macro, "per_speaker": per_speaker,
            "pooled": token_metrics(rows), "diagnosis": diagnosis_metrics(rows),
            "substitution_top1_all_gold": float(np.mean([r["candidate_event"] == "substitution" and r["candidate_phone"] == r["gold_realized"] for r in truth_subs])),
            "score_note": "AP not reported as a continuous confidence metric for discrete CTC greedy" if method == "ctc_greedy" else "higher negative gop means error"})
    by_method = {r["method"]: {s["speaker"]: s["auprc"] for s in r["per_speaker"]} for r in results}
    comparisons = []
    for method in METHODS:
        if method == "da_cf_gop":
            continue
        a, b = by_method["da_cf_gop"], by_method[method]
        comparisons.append({"a": "da_cf_gop", "b": method,
            "bootstrap": paired_speaker_bootstrap(a, b, n_bootstrap=config["bootstrap_replicates"], seed=config["seed"]),
            "sign_flip": exact_sign_flip_test({s: a[s] - b[s] for s in a})})
    report = {"cohort": cohort, "study_type": config["study_type"], "results": results,
        "paired_comparisons": comparisons, "same_token_keys": True,
        "n_detection_tokens": len(keys), "statistical_note": "Small speaker samples; overlapping cohorts; comparisons unadjusted; retrospective development."}
    write_json(directory / "metrics.json", json_safe(report))
    write_jsonl(directory / "phone_predictions.jsonl.gz", evaluated)
    import csv
    with (directory / "method_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method", *results[0]["macro"]])
        writer.writeheader()
        writer.writerows({"method": r["method"], **r["macro"]} for r in results)
    for r in results:
        print(r["method"], {k: round(r["macro"][k], 5) for k in ("auprc", "f1", "sub_f1", "del_f1")})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "predict", "evaluate"))
    parser.add_argument("--cohort", choices=("primary5", "sensitivity7"), default="primary5")
    parser.add_argument("--output", type=Path, default=OUT, help="Separate output directory for an unchanged reproduction")
    parser.add_argument("--config", type=Path, default=CONFIG, help="Prediction protocol; evaluation uses the saved configuration")
    args = parser.parse_args(argv)
    if args.stage == "prepare":
        prepare()
    elif args.stage == "predict":
        predict(args.cohort, args.output, args.config)
    else:
        evaluate(args.cohort, args.output)


if __name__ == "__main__":
    main()
