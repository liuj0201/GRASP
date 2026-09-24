"""Independent read-only checks of prepared/final dual-graph artifacts.

No scoring, fitting, decoder, or experiment-module code is imported.  No hashes
are computed.  Use --prepared while a run is incomplete; rerun without that
flag after metrics.json appears.  The only optional write is a new audit report.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


METHODS = {
    "M2_generic_viterbi", "M3_learned_viterbi", "M4_dual_graph",
    "M4_generic_posterior", "M4_generic_tuned", "M4_single_reference",
    "conventional_gop_raw", "ctc_sf_sd_norm_raw", "ppaf_rps_raw", "ppaf_ups_raw", "ppaf_acceptable_adapter_raw",
    "M0_greedy_single", "M1_greedy_acceptable",
}
PHONES = ("AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH",
          "EH", "ER", "EY", "F", "G", "HH", "IH", "IY", "JH", "K",
          "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH",
          "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH")
PHONE_ID = {phone: i + 1 for i, phone in enumerate(PHONES)}


def jsonl(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def close(actual, expected, name, tolerance=1e-10):
    if not np.allclose(actual, expected, atol=tolerance, rtol=tolerance):
        raise AssertionError(f"{name}: {actual!r} != {expected!r}")


def probability(value, name):
    array = np.asarray(value, dtype=float)
    assert np.isfinite(array).all(), f"nonfinite {name}"
    assert np.all(array >= -1e-8) and np.all(array <= 1 + 1e-8), f"out-of-range {name}"


def prepared_audit(directory):
    config = read_json(directory / "config.json")
    cohort = directory.name
    patients = set(config["cohorts"][cohort])
    folds = read_json(directory / "folds.json")
    rows = list(jsonl(directory / "evaluation_labels.jsonl.gz"))
    records = {row["event"]: row for row in rows}
    assert len(rows) == len(records), "duplicate label event"
    assert len({f["test_patient"] for f in folds}) == len(patients)
    for fold in folds:
        test, dev = fold["test_patient"], fold["development_patient"]
        train = set(fold["training_patients"])
        assert test != dev and not train & {test, dev}
        assert train | {test, dev} == patients
        assert not set(fold["healthy_development"]) & set(fold["healthy_test"])
        assert not set(fold["healthy_train_reserved"]) & set(fold["healthy_development"] + fold["healthy_test"])
    acoustic = list(jsonl(directory / "acoustic_inputs.jsonl.gz"))
    assert len(acoustic) == len(records)
    for row in acoustic:
        assert set(row) == {"event", "speaker", "prompt", "canonical_phones", "acceptable_phones"}, "non-acoustic field leaked"
        assert all(row[key] == records[row["event"]][key] for key in row)
    coverage = read_json(directory / "coverage.json")
    gaps_by_speaker = {}
    for claimed in coverage:
        speaker = claimed["speaker"]
        own = [r for r in rows if r["speaker"] == speaker]
        usable = [r for r in own if r["quality_exclusion"] is None]
        tokens = [t for r in usable for t in r["labels"]["tokens"]]
        actual = {
            "utterances": len(own), "quality_usable_utterances": len(usable),
            "canonical_tokens": sum(len(r["canonical_phones"]) for r in own),
            "detection_tokens": sum(t["diagnostic"] and t["stable_error"] for t in tokens),
            "exact_event_tokens": sum(t["diagnostic"] and t["stable_event"] for t in tokens),
            "errors": sum(t["diagnostic"] and t["stable_error"] and t["is_error"] is True for t in tokens),
            "variant_positions": sum(sum(len(a) > 1 for a in r["acceptable_phones"]) for r in usable),
        }
        for key, value in actual.items():
            assert claimed[key] == value, f"coverage mismatch {speaker} {key}"
        gap_counts = Counter()
        for row in own:
            for gap in row["labels"]["gaps"]:
                gap_counts["all_gaps"] += 1
                if row["quality_exclusion"] is not None:
                    gap_counts["quality_excluded_gaps"] += 1
                elif not gap["diagnostic"]:
                    gap_counts["nondiagnostic_gaps"] += 1
                elif not gap["stable"]:
                    gap_counts["alignment_uncertain_gaps"] += 1
                else:
                    length = len(gap["inserted_phones"])
                    gap_counts[f"stable_length_{length}"] += 1
                    gap_counts["stable_inserted_phones"] += length
                    gap_counts["over_cap_gaps"] += length > config["max_insertions_per_gap"]
                    gap_counts["over_cap_extra_phones"] += max(0, length-config["max_insertions_per_gap"])
        gaps_by_speaker[speaker] = dict(gap_counts)
    fit_summaries = []
    for fold in folds:
        path = directory / "folds" / f"{fold['fold']}.fit.json"
        if not path.exists():
            continue
        fit = read_json(path)
        assert fit["fold"] == fold, "fit fold differs from frozen declaration"
        expected_train = sorted(fold["training_patients"])
        expected_dev = sorted([fold["development_patient"], *fold["healthy_development"]])
        assert fit["prior"]["training_speakers"] == expected_train
        train_rows = [r for r in rows if r["speaker"] in expected_train and r["quality_exclusion"] is None]
        raw = np.zeros(40)
        for speaker in expected_train:
            local = [t for r in train_rows if r["speaker"] == speaker for t in r["labels"]["tokens"]
                     if t["stable_event"] and t["diagnostic"]]
            assert fit["prior"]["contribution"][speaker]["stable_event_tokens"] == len(local)
            close(fit["prior"]["contribution"][speaker]["effective_token_weight"],
                  config["speaker_effective_tokens"] / len(expected_train), "speaker prior weight")
            for token in local:
                raw[PHONE_ID[token["canonical_phone"]]] += 1
        close(fit["prior"]["raw_phone_counts"], raw, "prior training-only source counts")
        op = np.asarray(fit["prior"]["op_probs"])
        sub = np.asarray(fit["prior"]["sub_probs"])
        probability(op, "operation prior")
        probability(sub, "substitution prior")
        close(op.sum(axis=1), 1, "operation normalization")
        close(sub.sum(axis=1), 1, "substitution normalization")
        assert np.all(op[1:] > 0)
        assert np.all(sub[1:, 1:][~np.eye(39, dtype=bool)] > 0)
        assert np.all(sub[:, 0] == 0) and np.all(np.diag(sub) == 0)
        dev_rows = [r for r in rows if r["speaker"] in expected_dev and r["quality_exclusion"] is None]
        expected_n = sum(t["stable_error"] and t["diagnostic"] for r in dev_rows for t in r["labels"]["tokens"])
        assert set(fit["calibrators"]) == METHODS
        for method, calibration in fit["calibrators"].items():
            assert calibration["training_speakers"] == expected_dev, f"test speaker in {method} calibration"
            assert calibration["n_development_tokens"] == expected_n
            assert np.isfinite([calibration["slope"], calibration["intercept"], calibration["threshold"]]).all()
        actual_lengths = dict(Counter(str(len(g["inserted_phones"])) for r in dev_rows for g in r["labels"]["gaps"]
                                     if g["diagnostic"] and g["stable"]))
        assert fit["development_insertion_lengths"] == actual_lengths
        fit_summaries.append({"fold": fold["fold"], "train": expected_train, "calibration": expected_dev,
                              "n_development_tokens": expected_n})
    summary = {
        "status": "PREPARED_PASS", "cohort": cohort, "n_acoustic_events": len(rows),
        "n_fit_files_checked": len(fit_summaries), "n_expected_folds": len(folds),
        "patient_detection_tokens": sum(r["detection_tokens"] for r in coverage if r["speaker"] in patients),
        "patient_canonical_tokens": sum(r["canonical_tokens"] for r in coverage if r["speaker"] in patients),
        "patient_errors": sum(r["errors"] for r in coverage if r["speaker"] in patients),
        "coverage_recomputed": coverage, "gap_coverage_recomputed": gaps_by_speaker, "fit_checks": fit_summaries,
    }
    return summary, records, folds, config


def prediction_key(row):
    return row["method"], row["fold"], row["event"], row["phone_index"], row["target"]


def fold_prediction_audit(directory, fold_id):
    """Check saved acoustic predictions only; never compute test performance."""
    summary, records, folds, config = prepared_audit(directory)
    fold = next(f for f in folds if f["fold"] == fold_id)
    path = directory / "folds" / f"{fold_id}.predictions_without_gold.jsonl.gz"
    keys = defaultdict(set)
    speakers = Counter()
    posterior_rows = 0
    max_normalization_error = 0.0
    forbidden = {"labels", "gold_error", "gold_event", "gold_realized", "phn_path", "severity", "phn_model_phones"}
    for row in jsonl(path):
        assert not forbidden & set(row), "gold leaked into no-gold prediction"
        assert row["fold"] == fold_id
        assert row["speaker"] in [fold["test_patient"], *fold["healthy_test"]]
        source = records[row["event"]]
        assert row["speaker"] == source["speaker"]
        assert row["target"] == source["canonical_phones"][row["phone_index"]]
        key = (row["event"], row["phone_index"], row["target"])
        assert key not in keys[row["method"]], "duplicate no-gold token"
        keys[row["method"]].add(key)
        speakers[row["speaker"]] += 1
        assert np.isfinite(row["gop"]), "nonfinite gop"
        for field in ("p_error", "candidate_confidence", "candidate_margin"):
            probability(row[field], field)
        if "event_posteriors" in row:
            event, sub = np.array(row["event_posteriors"]), np.array(row["substitution_posteriors"])
            probability(event, "event posterior")
            probability(sub, "substitution posterior")
            max_normalization_error = max(max_normalization_error, abs(event.sum()-1), abs(sub.sum()-event[1]))
            close(event.sum(), 1, "local event normalization", 1e-7)
            close(sub.sum(), event[1], "SUB marginal normalization", 1e-7)
            close(row["graph_posterior_error"], event[1]+event[2], "error marginal", 1e-7)
            assert sub[0] == 0
            acceptable = [row["target"]] if row["method"] == "M4_single_reference" else source["acceptable_phones"][row["phone_index"]]
            assert all(abs(sub[PHONE_ID[phone]]) < 1e-10 for phone in acceptable), "accepted candidate also marked SUB"
            if event[1] + event[2] > 1e-12:
                conditional = np.r_[sub[1:], event[2]] / (event[1] + event[2])
                close(conditional.sum(), 1, "error-conditional candidate normalization", 1e-7)
                close(row["candidate_confidence"], conditional.max(), "top candidate conditional support", 1e-7)
            posterior_rows += 1
    expected = {(r["event"], i, phone) for r in records.values()
                if r["speaker"] in [fold["test_patient"], *fold["healthy_test"]]
                for i, phone in enumerate(r["canonical_phones"])}
    assert set(keys) == METHODS
    assert all(k == expected for k in keys.values()), "methods differ on all acoustic token keys"
    return {"status": "FOLD_PREDICTIONS_PASS", "fold": fold_id, "n_methods": len(keys),
            "n_predictions": sum(speakers.values()), "n_acoustic_tokens_per_method": len(expected),
            "speaker_prediction_counts": dict(speakers), "n_posterior_rows_checked": posterior_rows,
            "max_normalization_error": max_normalization_error,
            "test_performance_computed": False, "same_all_acoustic_token_keys": True,
            "single_reference_ablation_accepts_only_canonical": True}


def complete_audit(directory):
    summary, records, folds, config = prepared_audit(directory)
    assert summary["n_fit_files_checked"] == len(folds), "missing fold fit"
    report = read_json(directory / "metrics.json")
    patients = set(config["cohorts"][directory.name])
    fold_lookup = {f["fold"]: f for f in folds}
    predicted_values = {}
    forbidden = {"labels", "gold_error", "gold_event", "gold_realized", "phn_path", "severity", "phn_model_phones"}
    for row in jsonl(directory / "predictions_without_gold.jsonl.gz"):
        assert not forbidden & set(row), "gold leaked to before-gold prediction artifact"
        key = prediction_key(row)
        assert key not in predicted_values, "duplicate acoustic prediction"
        source, fold = records[row["event"]], fold_lookup[row["fold"]]
        assert row["speaker"] == source["speaker"]
        assert row["speaker"] in [fold["test_patient"], *fold["healthy_test"]]
        assert row["target"] == source["canonical_phones"][row["phone_index"]]
        assert np.isfinite(row["gop"])
        for field in ("p_error", "candidate_confidence", "candidate_margin"):
            probability(row[field], field)
        if "event_posteriors" in row:
            event, sub = np.array(row["event_posteriors"]), np.array(row["substitution_posteriors"])
            probability(event, "event posterior")
            probability(sub, "substitution posterior")
            close(event.sum(), 1, "local event normalization", 1e-7)
            close(sub.sum(), event[1], "SUB marginal sum", 1e-7)
            close(row["graph_posterior_error"], event[1]+event[2], "error marginal sum", 1e-7)
            assert sub[0] == 0
            acceptable = [row["target"]] if row["method"] == "M4_single_reference" else source["acceptable_phones"][row["phone_index"]]
            for phone in acceptable:
                assert abs(sub[PHONE_ID[phone]]) < 1e-10, "acceptable candidate also marked SUB"
        predicted_values[key] = (row["gop"], row["p_error"])
    keys_by_method = defaultdict(set)
    groups = defaultdict(list)
    for row in jsonl(directory / "phone_predictions.jsonl.gz"):
        key = prediction_key(row)
        assert key in predicted_values
        assert predicted_values[key] == (row["gop"], row["p_error"]), "gold join modified prediction"
        source = records[row["event"]]
        gold = source["labels"]["tokens"][row["phone_index"]]
        assert source["quality_exclusion"] is None and gold["diagnostic"] and gold["stable_error"]
        assert row["gold_error"] == gold["is_error"] and row["gold_event"] == gold["event_type"]
        assert row["gold_realized"] == gold["realized_phone"] and row["stable_event"] == gold["stable_event"]
        reduced_key = key[1:]
        assert reduced_key not in keys_by_method[row["method"]], "duplicate evaluated token"
        keys_by_method[row["method"]].add(reduced_key)
        if row["speaker"] in patients:
            groups[row["method"], row["speaker"]].append((bool(row["gold_error"]), -row["gop"], row["p_error"]))
    assert set(keys_by_method) == METHODS
    expected_keys = set()
    for fold in folds:
        for source in records.values():
            if source["speaker"] not in [fold["test_patient"], *fold["healthy_test"]] or source["quality_exclusion"] is not None:
                continue
            for token in source["labels"]["tokens"]:
                if token["stable_error"] and token["diagnostic"]:
                    expected_keys.add((fold["fold"], source["event"], token["phone_index"], token["canonical_phone"]))
    assert all(keys == expected_keys for keys in keys_by_method.values()), "method coverage mismatch"
    macro = {}
    for claimed in report["results"]:
        method = claimed["method"]
        speaker_values = []
        for speaker in sorted(patients):
            data = np.asarray(groups[method, speaker], dtype=float)
            y, scores, p = data.T
            values = {"auprc": float(average_precision_score(y, scores)),
                      "auroc": float(roc_auc_score(y, scores)), "brier": float(np.mean((p-y)**2))}
            recorded = next(s for s in claimed["per_speaker"] if s["speaker"] == speaker)
            assert recorded["n"] == len(data) and recorded["n_error"] == int(y.sum())
            for metric, value in values.items():
                close(recorded[metric], value, f"{method} {speaker} {metric}")
            speaker_values.append(values)
        macro[method] = {metric: float(np.mean([s[metric] for s in speaker_values])) for metric in speaker_values[0]}
        for metric, value in macro[method].items():
            close(claimed["macro"][metric], value, f"macro {method} {metric}")
        if method in ("conventional_gop_raw", "ctc_sf_sd_norm_raw"):
            assert claimed["diagnosis"] is None, "native scalar GOP diagnosis fabricated"
    insertion_groups = defaultdict(list)
    gap_keys = defaultdict(set)
    for row in jsonl(directory / "gap_predictions_without_gold.jsonl.gz"):
        probability(row["insertion_probability"], "insertion probability")
        probability(row["candidate_confidence"], "insertion candidate confidence")
        source = records[row["event"]]
        if row["speaker"] not in patients or source["quality_exclusion"] is not None:
            continue
        gold = source["labels"]["gaps"][row["gap_index"]]
        if not gold["diagnostic"] or not gold["stable"] or len(gold["inserted_phones"]) > 2:
            continue
        key = (row["fold"], row["event"], row["gap_index"], row["ordinal"])
        assert key not in gap_keys[row["method"]], "duplicate insertion slot"
        gap_keys[row["method"]].add(key)
        phone = gold["inserted_phones"][row["ordinal"]] if row["ordinal"] < len(gold["inserted_phones"]) else None
        insertion_groups[row["method"]].append((row, phone))
    assert len({frozenset(keys) for keys in gap_keys.values()}) == 1, "insertion method coverage mismatch"
    for claimed in report["insertion_diagnostics"]:
        rows = insertion_groups[claimed["method"]]
        positive = [(r, p) for r, p in rows if r["insertion_probability"] >= .5]
        tp = sum(p is not None and r["candidate_phone"] == p for r, p in positive)
        gold_n = sum(p is not None for _, p in rows)
        precision, recall = tp/max(len(positive), 1), tp/max(gold_n, 1)
        assert claimed["n_gap_slots"] == len(rows) and claimed["gold_insertions"] == gold_n
        close(claimed["precision"], precision, "insertion precision")
        close(claimed["recall"], recall, "insertion recall")
        close(claimed["f1"], 2*precision*recall/max(precision+recall, 1e-12), "insertion F1")
        if "cap_hit_rate" in claimed:
            close(claimed["cap_hit_rate"], np.mean([r["insertion_probability"] >= .5 for r, _ in rows if r["ordinal"] == 1]), "cap hit rate")
    summary.update(status="PASS", methods=sorted(METHODS), same_token_keys=True,
                   n_keys_per_method=len(expected_keys), independent_macro_metrics=macro,
                   n_without_gold_predictions=len(predicted_values),
                   probability_and_candidate_normalization="PASS", insertion_recomputation="PASS",
                   training_and_calibration_roles="PASS")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--prepared", action="store_true")
    parser.add_argument("--fold", help="audit one saved no-gold fold without computing test metrics")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = (fold_prediction_audit(args.directory, args.fold) if args.fold else
              prepared_audit(args.directory)[0] if args.prepared else complete_audit(args.directory))
    payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
