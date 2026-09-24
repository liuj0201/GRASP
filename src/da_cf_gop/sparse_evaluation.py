"""Evaluate frozen sparse-graph predictions without fitting or selection.

Usage: python -m da_cf_gop.sparse_evaluation --output <run directory>
All cohorts' predictions must be complete before this command reads test gold.
Candidate omissions remain in the detection denominator.
"""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from .dual_experiment import add_gold, token_metrics
from .final_experiment import OLD
from .metrics import exact_sign_flip_test, paired_speaker_bootstrap
from .parikh_experiment import read_jsonl, write_json, write_jsonl
from .phonology import CTC_ID_TO_PHONE


EXPECTED_TOKENS = {"primary5": 6820, "sensitivity7": 8353}
PAIRED_METRICS = ("auprc", "f1", "false_negative_rate")


def _key(row):
    return row["speaker"], row["event"], row["phone_index"], row["target"]


def _check_methods_and_keys(rows, methods):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)
    if set(grouped) != set(methods):
        raise ValueError("prediction methods differ from the registered strategies")
    reference = None
    for method in methods:
        keys = {_key(row) for row in grouped[method]}
        if len(keys) != len(grouped[method]):
            raise ValueError(f"duplicate prediction token in {method}")
        if reference is not None and keys != reference:
            raise ValueError("methods do not share identical token keys")
        reference = keys
    return grouped, reference


def _retention_metrics(rows, records_by_event):
    substitutions = [row for row in rows if row["stable_event"] and row["gold_event"] == "substitution"]
    missing = []
    for row in substitutions:
        allowed = {CTC_ID_TO_PHONE[phone] for phone in row["allowed_substitution_phone_ids"]}
        accepted = records_by_event[row["event"]]["acceptable_phones"][row["phone_index"]]
        if row["gold_realized"] not in allowed | set(accepted):
            missing.append({**row, "acceptable_phones": accepted,
                            "binary_detection_error": row["predicted_error"] != row["gold_error"]})
    n, missed = len(substitutions), len(missing)
    errors = sum(row["binary_detection_error"] for row in missing)
    return {
        "n_stable_substitutions": n,
        "retained": n - missed,
        "missing": missed,
        "retention_rate": (n - missed) / n if n else None,
        "missing_candidate_detection": {
            "n": missed, "n_detection_errors": errors,
            "detection_error_rate": errors / missed if missed else None,
            "false_negative_rate": sum(not row["predicted_error"] for row in missing) / missed if missed else None,
        },
    }, missing


def evaluate_predictions(predictions, records, *, cohort, speakers, methods,
                         expected_count, bootstrap_replicates=10000, seed=20260909):
    """Pure evaluation helper; predictions must already be frozen by the caller."""
    if "full_graph_da" not in methods:
        raise ValueError("full_graph_da is required as the registered paired comparator")
    if any(any(field.startswith("gold") for field in row) for row in predictions):
        raise ValueError("predictions_without_gold contains gold fields")
    if {row["speaker"] for row in predictions} != set(speakers):
        raise ValueError("prediction speakers differ from the registered cohort")
    _check_methods_and_keys(predictions, methods)
    evaluated = add_gold(predictions, records)
    grouped, keys = _check_methods_and_keys(evaluated, methods)
    if len(keys) != expected_count:
        raise ValueError(f"{cohort} has {len(keys)} detection tokens; expected {expected_count}")
    record_lookup = {row["event"]: row for row in records}
    results, all_misses = [], []
    for method in methods:
        rows = grouped[method]
        per_speaker = []
        for speaker in sorted(speakers):
            own = [row for row in rows if row["speaker"] == speaker]
            retention, _ = _retention_metrics(own, record_lookup)
            per_speaker.append({"speaker": speaker, **token_metrics(own),
                                "candidate_retention": retention})
        macro = {}
        for metric in (*PAIRED_METRICS, "auroc", "brier", "nll", "false_positive_rate"):
            values = [row[metric] for row in per_speaker]
            macro[metric] = float(np.mean(values)) if all(value is not None for value in values) else None
        retention, misses = _retention_metrics(rows, record_lookup)
        speaker_rates = [row["candidate_retention"]["retention_rate"] for row in per_speaker
                         if row["candidate_retention"]["retention_rate"] is not None]
        retention["speaker_macro_retention_rate"] = float(np.mean(speaker_rates)) if speaker_rates else None
        retention["speakers_with_stable_substitutions"] = len(speaker_rates)
        results.append({"method": method, "macro": macro, "per_speaker": per_speaker,
                        "pooled": token_metrics(rows), "candidate_retention": retention})
        all_misses.extend(misses)

    by_method = {row["method"]: row for row in results}
    comparisons = []
    for method in methods:
        if method == "full_graph_da":
            continue
        comparison = {"a": method, "b": "full_graph_da"}
        for metric in PAIRED_METRICS:
            a = {row["speaker"]: row[metric] for row in by_method[method]["per_speaker"]}
            b = {row["speaker"]: row[metric] for row in by_method["full_graph_da"]["per_speaker"]}
            if any(value is None for value in (*a.values(), *b.values())):
                comparison[metric] = None
                continue
            comparison[metric] = paired_speaker_bootstrap(a, b, n_bootstrap=bootstrap_replicates, seed=seed)
            if metric == "auprc":
                comparison["auprc_sign_flip"] = exact_sign_flip_test({speaker: a[speaker] - b[speaker] for speaker in a})
        comparisons.append(comparison)
    return {
        "cohort": cohort, "results": results, "paired_comparisons": comparisons,
        "same_token_keys": True, "n_detection_tokens": len(keys),
        "n_input_tokens_per_method": len(predictions) // len(methods),
        "candidate_omissions_excluded_from_detection": False,
        "candidate_retention_definition": "gold realized phone in retained SUB candidates or acceptable phones; stable gold substitutions only",
        "metric_units": "fractions; AP and F1 higher is better; FNR lower is better",
        "statistical_note": "Paired speaker bootstrap conditional on fitted models; overlapping cohorts; unadjusted comparisons; retrospective study.",
    }, all_misses


def evaluate_run(output):
    output = Path(output)
    marker = output / "predictions_complete.json"
    if not marker.is_file():
        raise RuntimeError("all-cohort predictions_complete.json is required before reading test labels")
    config_path = output / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    methods = [strategy["name"] for strategy in config["strategies"]]
    for cohort, speakers in config["cohorts"].items():
        directory = output / cohort
        prediction_path = directory / "predictions_without_gold.jsonl.gz"
        predictions = list(read_jsonl(prediction_path))
        records = list(read_jsonl(OLD / cohort / "evaluation_labels.jsonl.gz"))
        report, misses = evaluate_predictions(
            predictions, records, cohort=cohort, speakers=speakers, methods=methods,
            expected_count=EXPECTED_TOKENS[cohort],
            bootstrap_replicates=config.get("bootstrap_replicates", 10000),
            seed=config.get("seed", 20260909),
        )
        report["provenance"] = {
            "prediction_sha256": hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "completion_marker": json.loads(marker.read_text(encoding="utf-8")),
        }
        write_json(directory / "metrics.json", report)
        write_jsonl(directory / "candidate_misses.jsonl.gz", misses)
        print(f"Evaluated {cohort}: {report['n_detection_tokens']} common detection positions", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    evaluate_run(args.output)


if __name__ == "__main__":
    main()
