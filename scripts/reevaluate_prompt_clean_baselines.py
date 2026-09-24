"""Recalibrate fixed transferred scores after a documented prompt exclusion.

The malformed event is removed before development calibration, not merely from
test metrics. Healthy training, acoustic features and raw scores are unchanged.
Historical artifacts and the default final_baselines runner remain untouched.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from da_cf_gop.dual_experiment import add_gold, apply_calibration, make_folds, token_metrics
from da_cf_gop.final_baselines import ROOT, OUT as SOURCE, DUAL, fit_development_calibration
from da_cf_gop.parikh_experiment import read_jsonl, write_json, write_jsonl


OUT = ROOT / "artifacts/evaluation/final_baselines_prompt_clean"
FINAL = ROOT / "artifacts/evaluation/final_v3_prompt_clean"
EXCLUDED = frozenset({"F04_Session2_0009"})
COHORTS = ("primary5", "sensitivity7")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prompt_audit():
    """Inspect all frozen prompt texts and gold anchors independently of scores."""
    report = {"scope": "All frozen manifest recordings and dual-graph label records",
              "selection_uses_predictions_or_error_rates": False, "sources": {}}
    sources = {"manifest_sensitivity7": ROOT / "artifacts/manifests/sensitivity7.jsonl.gz",
               **{cohort: DUAL / cohort / "evaluation_labels.jsonl.gz" for cohort in COHORTS}}
    for name, path in sources.items():
        rows = list(read_jsonl(path))
        controls, nonascii, anchor_mismatch = [], [], []
        for row in rows:
            event = row.get("event", row.get("reading_event_id"))
            prompt = row["prompt"]
            bad_controls = sorted({f"U+{ord(c):04X}" for c in prompt
                                   if (ord(c) < 32 and c not in "\t\r\n") or ord(c) == 127})
            if bad_controls:
                controls.append({"event": event, "characters": bad_controls,
                                 "prompt_escaped": repr(prompt)})
            if any(ord(c) > 127 for c in prompt):
                nonascii.append({"event": event, "prompt_escaped": repr(prompt)})
            if "labels" in row:
                n = len(row["canonical_phones"])
                tokens = row["labels"]["tokens"]
                if not (len(row["acceptable_phones"]) == len(row["diagnostic_mask"]) == len(tokens) == n
                        and [t["phone_index"] for t in tokens] == list(range(n))
                        and all(t["canonical_phone"] == row["canonical_phones"][i]
                                and t["acceptable_phones"] == row["acceptable_phones"][i]
                                and bool(t["diagnostic"]) == bool(row["diagnostic_mask"][i])
                                for i, t in enumerate(tokens))):
                    anchor_mismatch.append(event)
        report["sources"][name] = {
            "path": str(path), "sha256": sha256(path), "records": len(rows),
            "unique_events": len({r.get("event", r.get("reading_event_id")) for r in rows}),
            "control_issues": controls, "non_ascii_issues": nonascii,
            "gold_anchor_inconsistencies": anchor_mismatch,
        }
    report["exclusion"] = {"events": sorted(EXCLUDED),
                           "reason": "Terminal cursor escape codes became spurious C/D letter-name target words",
                           "action": "Exclude complete reading event from every role; do not reconstruct intended text from audio or PHN"}
    write_json(OUT / "prompt_audit.json", report)
    return report


def baselines():
    OUT.mkdir(parents=True, exist_ok=True)
    threadpool_limits(limits=2)
    prompt_audit()
    config = json.loads((ROOT / "configs/dual_graph_v1.json").read_text())
    paths = [SOURCE / "dense_predictions_without_gold.jsonl.gz",
             SOURCE / "mixgop_predictions_without_gold.jsonl.gz"]
    original = [r for path in paths for r in read_jsonl(path)]
    raw = [r for r in original if r["event"] not in EXCLUDED]
    audit = {"excluded_events": sorted(EXCLUDED),
             "exclusion_before_label_join_and_development_calibration": True,
             "raw_acoustic_scores_refitted": False, "healthy_training_refitted": False,
             "source_files": [{"path": str(p), "sha256": sha256(p)} for p in paths],
             "removed_raw_rows_by_method": dict(Counter(r["method"] for r in original if r["event"] in EXCLUDED)),
             "cohorts": {}}
    for cohort in COHORTS:
        records = [r for r in read_jsonl(DUAL / cohort / "evaluation_labels.jsonl.gz")
                   if r["event"] not in EXCLUDED]
        evaluated = add_gold([r for r in raw if r["speaker"] in config["cohorts"][cohort]], records)
        results, predictions, fits = [], [], []
        old_fits = {(r["fold"]["fold"], r["method"]): r["calibrator"]
                    for r in json.loads((SOURCE / cohort / "fits.json").read_text())}
        changes = []
        for method in sorted({r["method"] for r in evaluated}):
            own = [r for r in evaluated if r["method"] == method]
            per = []
            for fold in make_folds(config, cohort):
                dev = [r for r in own if r["speaker"] == fold["development_patient"]]
                test = [{**r, "fold": fold["fold"]} for r in own if r["speaker"] == fold["test_patient"]]
                assert not EXCLUDED.intersection(r["event"] for r in dev + test)
                calibrator = fit_development_calibration(dev, config)
                out = apply_calibration(test, calibrator, config)
                per.append({"speaker": fold["test_patient"], **token_metrics(out)})
                predictions.extend(out)
                fits.append({"fold": fold, "method": method, "calibrator": calibrator,
                             "development_tokens": len(dev), "development_events": len({r["event"] for r in dev}),
                             "excluded_events": sorted(EXCLUDED)})
                changed = json.loads(json.dumps(calibrator)) != old_fits[(fold["fold"], method)]
                if fold["development_patient"] != "F04":
                    assert not changed, "Unaffected development calibrator changed"
                changes.append({"fold": fold["fold"], "method": method,
                                "development_patient": fold["development_patient"],
                                "calibrator_changed": changed, "development_tokens": len(dev)})
            macro = {key: float(np.mean([r[key] for r in per]))
                     for key in ("auprc", "auroc", "f1", "brier", "nll")}
            results.append({"method": method, "macro": macro, "per_speaker": per})
        dest = OUT / cohort
        write_json(dest / "metrics.json", {"cohort": cohort, "results": results,
                                           "excluded_events": sorted(EXCLUDED)})
        write_json(dest / "fits.json", fits)
        write_jsonl(dest / "phone_predictions.jsonl.gz", predictions)
        historical = {(r["event"], r["phone_index"], r["method"]): r
                      for r in read_jsonl(SOURCE / cohort / "phone_predictions.jsonl.gz")
                      if r["event"] not in EXCLUDED}
        assert len(historical) == len(predictions)
        probability_changes = Counter()
        for row in predictions:
            old = historical[(row["event"], row["phone_index"], row["method"])]
            assert row["gop"] == old["gop"], "Recalibration changed a raw score"
            assert row["gold_error"] == old["gold_error"], "Retained gold label changed"
            if row["p_error"] != old["p_error"]:
                probability_changes[row["speaker"]] += 1
                assert row["speaker"] == "F03", "Unexpected changed development speaker"
        audit["cohorts"][cohort] = {"n_predictions": len(predictions), "n_methods": len(results),
                                     "n_eligible_events": len({r["event"] for r in predictions}),
                                     "calibration_checks": changes,
                                     "retained_raw_scores_and_gold_identical": True,
                                     "probability_changes_by_test_speaker": dict(probability_changes),
                                     "excluded_event_present": any(r["event"] in EXCLUDED for r in predictions)}
        print(cohort, [(r["method"], round(r["macro"]["auprc"], 6), round(r["macro"]["f1"], 6))
                       for r in results if r["method"] in ("UQ_maxlogit_prior", "CaGOP_cagop_full", "MixGoP_XLSR19_C32")], flush=True)
    write_json(OUT / "recalibration_audit.json", audit)


def shared_subset():
    """Use cleaned transfer/final predictions, never historical dual-model fits."""
    config = json.loads((ROOT / "configs/dual_graph_v1.json").read_text())
    for cohort in COHORTS:
        transfer_path = OUT / cohort / "phone_predictions.jsonl.gz"
        final_path = FINAL / cohort / "phone_predictions.jsonl.gz"
        predictions = list(read_jsonl(transfer_path))
        newest = list(read_jsonl(final_path))
        assert not EXCLUDED.intersection(r["event"] for r in predictions + newest)
        mix = [r for r in predictions if r["method"] == "MixGoP_XLSR19_C32"]
        reference = {(r["event"], r["phone_index"], r["target"]) for r in mix}
        assert len(reference) == len(mix)
        groups = defaultdict(list)
        for row in predictions + newest:
            if (row["event"], row["phone_index"], row["target"]) in reference:
                groups[row["method"]].append(row)
        subset = []
        for name, values in sorted(groups.items()):
            keys = [(r["event"], r["phone_index"], r["target"]) for r in values]
            assert len(keys) == len(reference) and set(keys) == reference, name
            per = [{"speaker": speaker, **token_metrics([r for r in values if r["speaker"] == speaker])}
                   for speaker in config["cohorts"][cohort]]
            subset.append({"method": name, "n": len(reference), "per_speaker": per,
                           "macro": {key: float(np.mean([r[key] for r in per]))
                                     for key in ("auprc", "auroc", "f1", "brier", "nll")}})
        write_json(OUT / cohort / "mixgop32_shared_metrics.json", {
            "results": subset, "same_token_keys": True, "n": len(reference),
            "n_errors": sum(r["gold_error"] for r in mix), "excluded_events": sorted(EXCLUDED),
            "final_v3_source": str(final_path), "final_v3_sha256": sha256(final_path),
            "transfer_source": str(transfer_path), "transfer_sha256": sha256(transfer_path),
            "note": "Supplementary frozen 32-phone inventory; cleaned calibrators; 39-phone headline remains separate"})
        print(cohort, "shared", len(reference), [(r["method"], round(r["macro"]["auprc"], 6)) for r in subset
              if r["method"] in ("da_cf_gop", "ppaf_rps_raw", "MixGoP_XLSR19_C32")], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("baselines", "shared"))
    args = parser.parse_args()
    {"baselines": baselines, "shared": shared_subset}[args.stage]()
