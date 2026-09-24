"""Run Parikh (2025) PP-AF on the existing frozen TORGO cohorts.

No fitting or test-set threshold selection is performed. The uncalibrated
paper score and the same score after the existing speaker-disjoint sequence
recalibrator are evaluated separately. PHN labels enter only in evaluate().
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_fscore_support

from .calibration import CTCSequenceRecalibrator
from .ctc import counterfactual_log_probability_matrix
from .metrics import exact_sign_flip_test, paired_speaker_bootstrap
from .phonology import PHONE_TO_CTC_ID
from .parikh2025 import candidate_mask, ppaf_scores


DEFAULT_ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"
ID_TO_PHONE = {value: key for key, value in PHONE_TO_CTC_ID.items()}
METHODS = tuple(
    f"parikh2025_ppaf_{mode}_{backend}"
    for backend in ("raw", "sequence_recalibrated")
    for mode in ("rps", "ups")
)
COMPARATORS = (
    "conventional_forced_alignment_gop",
    "ctc_sf_sd_norm",
    "ctc_sf_sd_norm_sequence_recalibrated",
    "all_phone_cf_uniform_raw",
    "all_phone_cf_calibrated",
    "code9_legacy_hard_graph",
    "da_cf_adapted",
)


def read_jsonl(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def acoustic_manifest(path, patients):
    """Project only inference inputs; PHN, stability and severity are ignored."""
    result = {}
    for row in read_jsonl(path):
        if row["speaker_id"] in patients:
            event = row["reading_event_id"]
            if event in result:
                raise ValueError(f"duplicate reading event: {event}")
            result[event] = {
                "speaker": row["speaker_id"],
                "phones": tuple(row["canonical_phones"]),
            }
    return result


def index_logits(directory, events):
    result = {}
    for path in sorted(directory.glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            event = str(data["utterance_id"].item())
        if event in events:
            if event in result:
                raise ValueError(f"multiple raw logit files for {event}")
            result[event] = path
    missing = set(events) - set(result)
    if missing:
        raise ValueError(f"missing raw logits for {len(missing)} events: {sorted(missing)[:3]}")
    return result


def load_recalibrator(artifacts, cohort, fold, recorded_fold):
    """Select the fitted model named by the completed run, never by its score."""
    name = recorded_fold["fit_hash"]  # Existing filename; no digest is computed.
    prefix = artifacts / "cache" / "folds" / cohort / name
    state = json.loads(prefix.with_suffix(".fit.json").read_text(encoding="utf-8"))
    expected = set(fold["training_patients"]) | set(fold["healthy_references"])
    held_out = fold["held_out_patient"]
    if state["fold_id"] != fold["fold_id"] or state["held_out_patient"] != held_out:
        raise ValueError("recalibrator belongs to a different outer fold")
    if held_out in state["training_speakers"] or set(state["training_speakers"]) != expected:
        raise ValueError("recalibrator training speakers violate the frozen split")
    recalibrator = CTCSequenceRecalibrator.load(prefix.with_suffix(".recalibrator.npz"))
    if set(recalibrator.training_summary["training_speakers"]) != expected:
        raise ValueError("recalibrator array metadata violates the frozen split")
    return recalibrator, {
        "fold": fold["fold_id"], "held_out": held_out,
        "training_speakers": sorted(expected),
        "recalibrator_path": str(prefix.with_suffix(".recalibrator.npz")),
    }


def score_event(logits, canonical_ids, *, device="cpu"):
    """Pure inference: audio-model logits and prompt phones, with no labels."""
    log_probs = np.asarray(logits, dtype=np.float64)
    log_probs = log_probs - logsumexp(log_probs, axis=1, keepdims=True)
    matrix = counterfactual_log_probability_matrix(
        log_probs, canonical_ids, phone_ids=sorted(ID_TO_PHONE), blank_id=0,
        include_deletion=True, topology_correction=False, device=device,
    )
    result = {}
    for mode in ("rps", "ups"):
        restricted = mode == "rps"
        values = ppaf_scores(matrix, canonical_ids, ID_TO_PHONE, restricted=restricted)
        if not np.isfinite(values).all():
            raise ValueError("non-finite PP-AF scores; evaluation cannot silently drop tokens")
        mask = candidate_mask(matrix.candidate_ids, canonical_ids, ID_TO_PHONE, restricted=restricted)
        records = []
        for position, score in enumerate(values):
            candidates = [
                (ID_TO_PHONE.get(int(candidate), "<DEL>"), float(likelihood))
                for candidate, likelihood, keep in zip(
                    matrix.candidate_ids[position], matrix.log_probabilities[position], mask[position]
                ) if keep
            ]
            candidates.sort(key=lambda pair: (-pair[1], pair[0]))
            records.append({
                "phone_index": position, "target": ID_TO_PHONE[int(canonical_ids[position])],
                "gop": float(score), "error_score": float(-score),
                "best_alternative": candidates[0][0],
                "candidate_count": len(candidates),
                "canonical_log_probability": float(matrix.canonical_log_probability),
                "candidate_log_probabilities": {p: v if np.isfinite(v) else None for p, v in candidates},
            })
        result[mode] = records
    return result


def predict(artifacts, cohort, *, device="cuda", progress=print):
    folds = json.loads((artifacts / "manifests" / f"{cohort}.folds.json").read_text())["folds"]
    patients = {fold["held_out_patient"] for fold in folds}
    events = acoustic_manifest(artifacts / "manifests" / f"{cohort}.jsonl.gz", patients)
    progress(f"{cohort}: indexing cached logits for {len(events)} patient utterances", flush=True)
    sources = index_logits(artifacts / "cache" / "logits", events)
    summary = json.loads((artifacts / "summaries" / f"run-phone-loso.{cohort}.json").read_text())
    output, fold_info = [], []
    completed = 0
    for fold in folds:
        recalibrator, info = load_recalibrator(artifacts, cohort, fold, summary["details"]["folds"][fold["fold_id"]])
        fold_info.append(info)
        for event, acoustic in sorted(events.items()):
            if acoustic["speaker"] != fold["held_out_patient"]:
                continue
            canonical_ids = tuple(PHONE_TO_CTC_ID[phone] for phone in acoustic["phones"])
            with np.load(sources[event], allow_pickle=False) as data:
                logits = data["logits"].astype(np.float64)
                if tuple(data["canonical_ids"].tolist()) != canonical_ids:
                    raise ValueError(f"cached prompt phones disagree: {event}")
            if logits.shape[1] != 40:
                raise ValueError("this comparison requires the existing 40-class CTC backend")
            for backend, values in (("raw", logits), ("sequence_recalibrated", recalibrator.apply(logits))):
                for mode, tokens in score_event(values, canonical_ids, device=device).items():
                    for token in tokens:
                        output.append({
                            "method": f"parikh2025_ppaf_{mode}_{backend}", "cohort": cohort,
                            "fold": fold["fold_id"], "speaker": acoustic["speaker"], "event": event,
                            **token,
                        })
            completed += 1
            if completed % 100 == 0 or completed == len(events):
                progress(f"{cohort}: scored {completed}/{len(events)} utterances", flush=True)
    return output, fold_info


def token_key(row):
    return (row["speaker"], row["event"], int(row["phone_index"]), row["target"])


def ranking_metrics(rows):
    truth = np.asarray([row["gold_event"] != "correct" for row in rows], dtype=int)
    scores = np.asarray([row["error_score"] for row in rows])
    if not np.isfinite(scores).all() or len(np.unique(truth)) != 2:
        raise ValueError("ranking metrics require finite scores and both gold classes")
    return {"n": len(rows), "n_error": int(truth.sum()), "prevalence": float(truth.mean()),
            "auprc": float(average_precision_score(truth, scores)),
            "auroc": float(roc_auc_score(truth, scores))}


def evaluate(predictions, baseline_path):
    """Join the frozen gold only after every patient prediction is complete."""
    gold, groups = {}, defaultdict(list)
    for row in read_jsonl(baseline_path):
        if row["method"] not in COMPARATORS:
            continue
        key = token_key(row)
        label = {"gold_event": row["gold_event"], "gold_realized": row["gold_realized"], "stable": row["stable"]}
        if key in gold and gold[key] != label:
            raise ValueError("baseline methods disagree on frozen gold")
        gold[key] = label
        groups[row["method"]].append({
            **{field: row[field] for field in ("method", "speaker", "event", "phone_index", "target", "gop")},
            **label, "error_score": -float(row["gop"]),
        })
    evaluated = []
    for row in predictions:
        key = token_key(row)
        if key in gold:
            joined = {**row, **gold[key]}
            groups[row["method"]].append(joined)
            evaluated.append(joined)
    if not gold or (set(METHODS) | set(COMPARATORS)) - set(groups):
        raise ValueError("missing frozen gold, PP-AF method or comparator")
    for name, rows in groups.items():
        keys = [token_key(row) for row in rows]
        if len(keys) != len(set(keys)) or set(keys) != set(gold):
            raise ValueError(f"duplicate/missing canonical tokens in {name}")
    results = []
    for name, rows in sorted(groups.items()):
        speakers = sorted({row["speaker"] for row in rows})
        per_speaker = [{"speaker": speaker, **ranking_metrics([r for r in rows if r["speaker"] == speaker])} for speaker in speakers]
        result = {"method": name, "per_speaker": per_speaker, "pooled": ranking_metrics(rows),
                  "macro": {metric: float(np.mean([s[metric] for s in per_speaker])) for metric in ("auprc", "auroc")}}
        if name in METHODS:
            truth = [row["gold_event"] != "correct" for row in rows]
            pred = [row["gop"] < 0 for row in rows]
            precision, recall, f1, _ = precision_recall_fscore_support(truth, pred, average="binary", zero_division=0)
            result["pooled_paper_zero_threshold"] = {"precision": float(precision), "recall": float(recall), "f1": float(f1), "gop_cutoff": 0.0}
        results.append(result)
    by_name = {r["method"]: {s["speaker"]: s["auprc"] for s in r["per_speaker"]} for r in results}
    comparisons = []
    for method in METHODS:
        for comparator in ("da_cf_adapted", "ctc_sf_sd_norm_sequence_recalibrated", "all_phone_cf_calibrated"):
            a, b = by_name[method], by_name[comparator]
            differences = {speaker: a[speaker] - b[speaker] for speaker in a}
            comparisons.append({"a": method, "b": comparator,
                                "auprc": paired_speaker_bootstrap(a, b, n_bootstrap=10000, seed=20260829),
                                "exact_sign_flip": exact_sign_flip_test(differences),
                                "speakers_improved": sum(value > 0 for value in differences.values())})
    return evaluated, {"results": results, "paired_comparisons": comparisons, "n_stable_tokens_per_method": len(gold),
                       "metrics_use": "continuous error_score=-gop for every method; speaker macro average",
                       "zero_threshold": "PP-AF equation natural cutoff only; no test tuning and not a calibrated operating point",
                       "probability_metrics": "not reported: the paper score is not p_error"}


def save_tables(directory, report):
    for filename, rows in (
        ("method_metrics.csv", [{"method": r["method"], **r["macro"]} for r in report["results"]]),
        ("speaker_metrics.csv", [{"method": r["method"], **s} for r in report["results"] for s in r["per_speaker"]]),
    ):
        with (directory / filename).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def run(artifacts=DEFAULT_ARTIFACTS, *, cohort="primary5", device="cuda"):
    import torch
    from threadpoolctl import threadpool_limits
    from .parikh2025 import graph_audit

    artifacts = Path(artifacts).resolve()
    output = artifacts / "evaluation" / "parikh2025" / cohort
    started = time.monotonic()
    torch.set_num_threads(4)
    with threadpool_limits(limits=4):
        predictions, folds = predict(artifacts, cohort, device=device)
    # Save all canonical-token predictions before opening any evaluation labels.
    write_jsonl(output / "predictions_without_gold.jsonl.gz", predictions)
    evaluated, report = evaluate(predictions, artifacts / "evaluation" / cohort / "phone_predictions.jsonl.gz")
    report.update({"cohort": cohort, "methods": list(METHODS), "folds": folds,
                   "track": "paper PP-AF equation; official confusion map projected to existing ARPABET backend",
                   "graph": graph_audit(), "n_all_canonical_tokens": len(predictions) // len(METHODS),
                   "topology_correction": False, "deletion_penalty": False,
                   "include_single_phone_deletion": True,
                   "singleton_deletion_note": "Paper-consistent empty CTC target; public scripts instead require N>1.",
                   "n_single_phone_utterances": sum(
                       len(rows) == 1 for rows in _event_rows(predictions, METHODS[0]).values()
                   ),
                   "threshold_fitting": False, "gold_join_after_predictions": True,
                   "elapsed_seconds": round(time.monotonic() - started, 2)})
    write_jsonl(output / "phone_predictions.jsonl.gz", evaluated)
    write_json(output / "metrics.json", report)
    save_tables(output, report)
    for result in report["results"]:
        print(f"{result['method']}: AUPRC={result['macro']['auprc']:.6f}, AUROC={result['macro']['auroc']:.6f}")
    print(f"Results: {output}")
    return report


def _event_rows(predictions, method):
    groups = defaultdict(list)
    for row in predictions:
        if row["method"] == method:
            groups[row["event"]].append(row)
    return groups


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=("primary5", "sensitivity7"), default="primary5")
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args(argv)
    run(args.artifacts, cohort=args.cohort, device=args.device)


if __name__ == "__main__":
    main()
