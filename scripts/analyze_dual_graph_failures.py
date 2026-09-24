"""Post-hoc descriptive analysis only; never changes predictions or settings."""
import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


PHONES = ("AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH", "EH", "ER", "EY", "F", "G", "HH", "IH", "IY", "JH", "K", "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH", "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH")


def quantiles(values):
    return {key: float(value) for key, value in zip(("min", "q25", "median", "q75", "max"), np.quantile(values, (0, .25, .5, .75, 1)))} if values else None


def detection(rows):
    y, score = [r["gold_error"] for r in rows], [-r["gop"] for r in rows]
    return {"n": len(rows), "n_error": sum(y),
            "ap": float(average_precision_score(y, score)) if len(set(y)) == 2 else None,
            "auroc": float(roc_auc_score(y, score)) if len(set(y)) == 2 else None}


def m4_decomposition(rows):
    exact_sub = [r for r in rows if r["stable_event"] and r["gold_event"] == "substitution"]
    correct_candidate = [r for r in exact_sub if r["candidate_event"] == "substitution" and r["candidate_phone"] == r["gold_realized"]]
    over_threshold = [r for r in correct_candidate if r["predicted_error"]]
    counts = Counter()
    composition = Counter()
    conditional = defaultdict(list)
    index_checks = 0
    for row in rows:
        group = row["gold_event"] if row["stable_event"] else "ambiguous_error" if row["gold_error"] else "ambiguous_correct"
        composition[group] += 1
        counts[group + "__candidate_" + row["candidate_event"]] += 1
        counts[group + "__predicted_error"] += row["predicted_error"]
        if "event_posteriors" in row:
            for event, value in zip(("OK", "SUB", "DEL"), row["event_posteriors"]):
                conditional[group + "__posterior_" + event].append(value)
            candidates = np.r_[np.asarray(row["substitution_posteriors"])[1:], row["event_posteriors"][2]]
            best = int(np.argmax(candidates))
            assert row["candidate_event"] == ("deletion" if best == 39 else "substitution")
            assert row["candidate_phone"] == (None if best == 39 else PHONES[best])
            assert row["gold_realized"] is None or row["gold_realized"] in PHONES
            index_checks += 1
    top_sub_correct = sum(r["gold_realized"] == PHONES[int(np.argmax(np.asarray(r["substitution_posteriors"])[1:]))]
                          for r in exact_sub)
    return {
        "gold_composition": dict(composition), "candidate_and_threshold_counts": dict(counts),
        "mean_event_posteriors_by_gold_group": {key: float(np.mean(values)) for key, values in conditional.items()},
        "exact_sub_gold": len(exact_sub), "correct_SUB_candidate_before_error_threshold": len(correct_candidate),
        "correct_SUB_candidate_after_error_threshold": len(over_threshold),
        "correct_SUB_candidate_below_error_threshold": len(correct_candidate)-len(over_threshold),
        "correct_SUB_candidate_after_specific_gate": sum(r["specific_authorized"] for r in correct_candidate),
        "top_phone_within_SUB_only_correct": top_sub_correct,
        "correct_SUB_candidate_p_error": quantiles([r["p_error"] for r in correct_candidate]),
        "all_exact_SUB_p_error": quantiles([r["p_error"] for r in exact_sub]),
        "candidate_index_to_phone_string_checks": index_checks,
        "identity_checks": "PASS",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    directory = args.directory
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    patients = set(config["cohorts"][directory.name])
    methods = ("M4_dual_graph", "ppaf_rps_raw", "M3_learned_viterbi")
    grouped = defaultdict(list)
    with gzip.open(directory / "phone_predictions.jsonl.gz", "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["speaker"] in patients and row["method"] in methods:
                grouped[row["method"], row["speaker"]].append(row)
    decomposition = {}
    descriptive_subtasks = {}
    for speaker in sorted(patients):
        decomposition[speaker] = m4_decomposition(grouped["M4_dual_graph", speaker])
        fit = json.loads((directory / "folds" / f"{directory.name}_test_{speaker}.fit.json").read_text(encoding="utf-8"))
        decomposition[speaker]["frozen_calibration"] = fit["calibrators"]["M4_dual_graph"]
        descriptive_subtasks[speaker] = {}
        for method in methods:
            rows = grouped[method, speaker]
            descriptive_subtasks[speaker][method] = {
                "headline_population": detection(rows),
                "match_plus_exact_SUB_new_denominator": detection([r for r in rows if r["stable_event"] and r["gold_event"] in ("match", "substitution")]),
                "match_plus_exact_DEL_new_denominator": detection([r for r in rows if r["stable_event"] and r["gold_event"] in ("match", "deletion")]),
                "match_plus_ambiguous_error_new_denominator": detection([r for r in rows if r["gold_event"] == "match" or not r["stable_event"] and r["gold_error"]]),
            }
    combined = m4_decomposition([r for s in sorted(patients) for r in grouped["M4_dual_graph", s]])
    result = {"analysis_scope": "post-hoc descriptive analysis; not a new headline endpoint or parameter-selection evidence",
              "production_settings_changed": False, "subtask_denominators_differ_from_headline": True,
              "all_patients_M4": combined, "per_patient_M4": decomposition,
              "descriptive_subtask_rankings": descriptive_subtasks,
              "warning": "Groupwise associations do not establish clinical or pathological causes; no new tuning is performed."}
    path = directory / "failure_analysis.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(path), "all_patient_counts": combined,
                      "M05": {"decomposition": decomposition["M05"], "subtasks": descriptive_subtasks["M05"]}}, indent=2))


if __name__ == "__main__":
    main()
