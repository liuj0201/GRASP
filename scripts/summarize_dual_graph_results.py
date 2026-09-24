"""Read-only-to-production summary of the frozen dual-graph experiment.

Recomputes ranking, calibration and tie-aware selective risk from evaluated
prediction artifacts. It never fits a parameter or changes a prediction.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve, f1_score


COVERAGES = (0.1, 0.25, 0.5, 0.75, 1.0)
MATCHED_FPRS = (0.01, 0.05, 0.10)
SCALAR_DIAGNOSIS_UNSUPPORTED = {"conventional_gop_raw", "ctc_sf_sd_norm_raw"}
LIMITATIONS = [
    "The new CMUdict acceptable policy and neutral labels differ from legacy DA-CF/Parikh experiments; old headline numbers are not directly comparable.",
    "Only explicitly attested, one-position AH0/IH0 normal variants are supported. Context-ambiguous and length-changing normal word blocks are non-diagnostic, not full accepted span graphs.",
    "Corpus-level human pronunciation-error annotation is supported by source publications, but current files were not independently re-adjudicated and audio/text/PHN listening spot-checks were not performed this run.",
    "The frozen 39-phone model projection loses contrasts including AX/AH and DX/T. This is projected segmental evaluation, not a complete clinical distortion assessment.",
    "M4_single_reference, M3 and M2 inherit full-M4 selected development parameters; they are controlled ablations, not independently optimized systems. M4_generic_tuned is the independently scale-tuned generic comparator.",
    "FPR at at-least 80% recall is a descriptive held-out ROC statistic, not a threshold selected for deployment or proof that development-fixed thresholds achieve 80% recall on a new patient.",
    "Original diagnostic F1 uses each method's development-fixed threshold. The separate matched-FPR diagnosis rows are descriptive test-ROC functionals with randomized boundary ties, not fitted or deployable thresholds.",
    "Calibration and threshold selection reuse the development population. Held-out evaluation is independent, but development 80% selective precision is not an externally guaranteed precision.",
    "M3/M4 selective-risk comparison includes each method's own development calibration; it does not alone isolate every calibration effect from path summation.",
    "Independent acoustic OOD/low-support rejection is not implemented. Non-diagnostic raw scoring rows are internal predictions, not a validated patient-facing diagnostic interface.",
    "Insertions are capped at two per gap; evaluation uses whole-gap consensus and excludes longer or ambiguous gaps. Fixed insertion posterior threshold 0.5 is exploratory and uncalibrated.",
    "MC04 is the same held-out healthy speaker evaluated by multiple patient-fold models; those fold results are not independent healthy participants.",
    "Only five or seven patients, overlapping cohorts and unadjusted multiple comparisons; speaker bootstrap and sign-flip results should remain exploratory.",
    "Total elapsed experiment time is not per-utterance deployment latency. Per-stage timing, peak RAM/VRAM, acoustic parameter count/inference timing and representative graph-size resource tables were not fully measured.",
]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def tie_aware_risk(rows, coverages=COVERAGES):
    """Expected risk under uniform random order within exactly tied confidence.

    At rank k inside an m-item tie with b mistakes, expected added mistakes are
    k*b/m. Confidence is the probability of the development-threshold-selected
    class, NOT abs(p_error-0.5). No random seed or arbitrary file order is used.
    """
    if not rows:
        return {"n": 0, "aurc": None, "curve": [], "confidence_tie_groups": 0}
    groups = defaultdict(list)
    for row in rows:
        probability = float(row["p_error"])
        confidence = probability if row["predicted_error"] else 1 - probability
        groups[confidence].append(int(row["predicted_error"] != row["gold_error"]))
    expected_errors = []
    for confidence in sorted(groups, reverse=True):
        errors = groups[confidence]
        expected_errors.extend([float(np.mean(errors))] * len(errors))
    n = len(expected_errors)
    risks = np.cumsum(expected_errors) / np.arange(1, n + 1)
    curve = []
    for coverage in coverages:
        k = max(1, int(n * coverage))
        curve.append({"coverage": coverage, "effective_coverage": k / n,
                      "n_selected": k, "expected_binary_risk": float(risks[k - 1])})
    return {"n": n, "aurc": float(np.trapz(np.r_[0.0, risks], np.arange(n + 1) / n)),
            "curve": curve, "confidence_tie_groups": sum(len(v) > 1 for v in groups.values()),
            "aurc_convention": "trapezoidal rank grid including the conventional (coverage=0,risk=0) endpoint"}


def error_metrics(rows, probability_key="p_error"):
    y = np.array([r["gold_error"] for r in rows], dtype=int)
    p = np.clip([r[probability_key] for r in rows], 1e-12, 1 - 1e-12)
    return {"brier": float(np.mean((p - y) ** 2)),
            "nll": float(np.mean(-y * np.log(p) - (1 - y) * np.log1p(-p)))}


def independent_metrics(rows):
    y = np.array([r["gold_error"] for r in rows], dtype=int)
    score = np.array([-r["gop"] for r in rows], dtype=float)
    prediction = np.array([r["predicted_error"] for r in rows], dtype=bool)
    if not np.isfinite(score).all():
        raise ValueError("nonfinite prediction score in final artifact")
    binary = len(set(y)) == 2
    fpr80 = None
    if binary:
        fpr, tpr, _ = roc_curve(y, score)
        fpr80 = float(min(fpr[tpr >= 0.8]))
    return {"n": len(rows), "n_error": int(y.sum()),
            "auprc": float(average_precision_score(y, score)) if binary else None,
            "auroc": float(roc_auc_score(y, score)) if binary else None,
            "f1": float(f1_score(y, prediction, zero_division=0)),
            "false_positive_rate": float(prediction[y == 0].mean()) if (y == 0).any() else None,
            "false_negative_rate": float((~prediction[y == 1]).mean()) if (y == 1).any() else None,
            "fpr_at_80pct_recall": fpr80, **error_metrics(rows)}


def macro(values, fields):
    result = {}
    for field in fields:
        numbers = [v[field] for v in values if v.get(field) is not None]
        result[field] = float(np.mean(numbers)) if numbers else None
    return result


def macro_risk(per_speaker):
    curves = [row["risk"] for row in per_speaker]
    return {"aurc": float(np.mean([r["aurc"] for r in curves])),
            "curve": [{"coverage": coverage,
                       "expected_binary_risk": float(np.mean([r["curve"][i]["expected_binary_risk"] for r in curves]))}
                      for i, coverage in enumerate(COVERAGES)],
            "speaker_count": len(curves), "averaging": "equal speaker, tie-aware within speaker"}


def matched_fpr_weights(rows, target_fpr):
    """Expected score-threshold selection weights with exactly matched FPR.

    The cutoff is a descriptive function of ALL normal detection tokens for
    one held-out speaker. Strictly higher scores have weight 1; equal scores
    share one random-selection probability. It is never used to alter the
    development-calibrated predictions in the experiment artifacts.
    """
    score = np.array([-r["gop"] for r in rows], dtype=float)
    normal = np.array([not r["gold_error"] for r in rows], dtype=bool)
    if not normal.any() or not np.isfinite(score).all() or not 0 <= target_fpr <= 1:
        raise ValueError("matched FPR requires normal tokens, finite scores and FPR in [0,1]")
    budget = target_fpr * int(normal.sum())
    if target_fpr == 1:
        weights = np.ones(len(rows))
        threshold, boundary_probability = None, 1.0
    else:
        above = 0
        for threshold in sorted(set(score[normal]), reverse=True):
            count = int(np.sum(normal & (score == threshold)))
            if above + count >= budget:
                boundary_probability = (budget - above) / count
                weights = np.where(score > threshold, 1.0,
                                   np.where(score == threshold, boundary_probability, 0.0))
                break
            above += count
    return weights, {"target_fpr": target_fpr, "achieved_expected_fpr": float(weights[normal].sum() / normal.sum()),
                     "normal_detection_tokens": int(normal.sum()), "expected_false_alarms": float(weights[normal].sum()),
                     "descriptive_score_cutoff": float(threshold) if threshold is not None else None,
                     "boundary_selection_probability": float(boundary_probability)}


def exact_diagnosis_at_fpr(rows, target_fpr):
    """Fractionally weighted exact diagnosis; no new candidate is fabricated.

    Exact precision/recall use the fixed stable_event subset, consistent with
    the existing diagnostic table. Specific proposals on binary-stable but
    exact-ambiguous tokens are reported separately, not silently scored TP/FP.
    Rates are ratios of expected counts under the randomized boundary rule.
    """
    weights, cutoff = matched_fpr_weights(rows, target_fpr)
    exact = np.array([r["stable_event"] for r in rows], dtype=bool)
    specific = np.array([r.get("candidate_event") in ("substitution", "deletion") for r in rows])
    events = {}
    for event in ("substitution", "deletion", "combined_SUB_DEL"):
        event_set = {"substitution", "deletion"} if event == "combined_SUB_DEL" else {event}
        gold = np.array([r["gold_event"] in event_set for r in rows]) & exact
        proposed = np.array([r.get("candidate_event") in event_set for r in rows]) & exact
        correct = np.array([
            r.get("candidate_event") == r["gold_event"]
            and r["gold_event"] in event_set
            and (r["gold_event"] == "deletion" or r.get("candidate_phone") == r["gold_realized"])
            for r in rows]) & exact
        tp = float(weights[correct].sum())
        proposed_mass = float(weights[proposed].sum())
        gold_count = int(gold.sum())
        precision = tp / proposed_mass if proposed_mass else 0.0
        recall = tp / gold_count if gold_count else 0.0
        events[event] = {"gold_exact_events": gold_count, "expected_specific_proposals": proposed_mass,
                         "expected_true_positives": tp, "precision": precision, "recall": recall,
                         "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}
    return {**cutoff, "events": events, "exact_token_count": int(exact.sum()),
            "exact_unverifiable_specific_proposal_mass": float(weights[specific & ~exact].sum()),
            "exact_unverifiable_token_count": int((~exact).sum()),
            "exact_unverifiable_specific_proposal_coverage": (
                float(weights[specific & ~exact].sum() / (~exact).sum()) if (~exact).any() else 0.0),
            "selected_without_specific_SUB_DEL_candidate_mass": float(weights[~specific].sum()),
            "expected_selected_token_mass": float(weights.sum()),
            "precision_scope": "fixed stable_event subset; separately counted unverifiable proposals excluded from both TP and proposal denominator"}


def macro_matched_fpr_diagnosis(rows, targets=MATCHED_FPRS):
    speakers = sorted({r["speaker"] for r in rows})
    points = []
    for target in targets:
        per_speaker = [{"speaker": s, **exact_diagnosis_at_fpr([r for r in rows if r["speaker"] == s], target)}
                       for s in speakers]
        events = {}
        for event in ("substitution", "deletion", "combined_SUB_DEL"):
            values = [point["events"][event] for point in per_speaker]
            events[event] = {**macro(values, ("precision", "recall", "f1")),
                             "speakers_with_gold_events": sum(v["gold_exact_events"] > 0 for v in values),
                             "gold_exact_events_total": sum(v["gold_exact_events"] for v in values)}
        points.append({"target_fpr": target,
                       "macro_achieved_expected_fpr": float(np.mean([r["achieved_expected_fpr"] for r in per_speaker])),
                       "macro_events": events, "per_speaker": per_speaker})
    return {"supported": True, "points": points, "speaker_count": len(speakers),
            "scope": "Descriptive held-out ROC statistic only; test-derived cutoff is NOT a deployable threshold, development selection, or mutation of saved predictions",
            "score": "-gop, uncalibrated continuous error ranking",
            "tie_rule": "same boundary probability for every token with equal score, yielding target FPR in expectation on all normal detection keys",
            "macro_rule": "equal-speaker average of ratios of expected exact confusion counts; zero division returns zero"}


def summarize_cohort(directory):
    directory = Path(directory)
    report, config = read_json(directory / "metrics.json"), read_json(directory / "config.json")
    cohort = report["cohort"]
    patients = set(config["cohorts"][cohort])
    rows = read_jsonl(directory / "phone_predictions.jsonl.gz")
    labels = {r["event"]: r for r in read_jsonl(directory / "evaluation_labels.jsonl.gz")}
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)
    stored = {r["method"]: r for r in report["results"]}
    results = {}
    reference_keys = None
    for method, all_rows in sorted(grouped.items()):
        own = [r for r in all_rows if r["speaker"] in patients]
        keys = [(r["speaker"], r["event"], r["phone_index"], r["target"]) for r in own]
        if len(keys) != len(set(keys)) or (reference_keys is not None and set(keys) != reference_keys):
            raise ValueError(f"duplicate or mismatched evaluated keys for {method}")
        reference_keys = set(keys)
        per_speaker = []
        for speaker in sorted(patients):
            subset = [r for r in own if r["speaker"] == speaker]
            per_speaker.append({"speaker": speaker, **independent_metrics(subset), "risk": tie_aware_risk(subset)})
        fields = ("auprc", "auroc", "f1", "false_positive_rate", "false_negative_rate", "fpr_at_80pct_recall", "brier", "nll")
        averages = macro(per_speaker, fields)
        for field in fields:
            if not np.isclose(averages[field], stored[method]["macro"][field], atol=1e-10, rtol=1e-10):
                raise ValueError(f"independent {method}/{field} disagrees with final metrics artifact")
        healthy_rows = [r for r in all_rows if r["speaker"] not in patients]
        healthy = [{"fold": fold, **independent_metrics([r for r in healthy_rows if r["fold"] == fold])}
                   for fold in sorted({r["fold"] for r in healthy_rows})]
        normal_variant = []
        for row in own:
            source = labels[row["event"]]
            if row["stable_event"] and row["gold_event"] == "match" and row["gold_realized"] != row["target"]:
                if row["gold_realized"] not in source["acceptable_phones"][row["phone_index"]]:
                    raise ValueError("gold match does not satisfy the fixed acceptable policy")
                normal_variant.append(row)
        calibration = None
        if own and all("graph_posterior_error" in r for r in own):
            before = [error_metrics([r for r in own if r["speaker"] == s], "graph_posterior_error") for s in sorted(patients)]
            after = [error_metrics([r for r in own if r["speaker"] == s]) for s in sorted(patients)]
            calibration = {"raw_graph_posterior": macro(before, ("brier", "nll")),
                           "development_calibrated": macro(after, ("brier", "nll")),
                           "scope": "same held-out tokens; changes in probability accuracy, not acoustic recognition accuracy"}
        results[method] = {"macro": averages, "per_speaker": per_speaker,
                           "speaker_macro_tie_aware_risk": macro_risk(per_speaker),
                           "healthy_test_by_fold": healthy,
                           "healthy_descriptive_mean_fpr_across_models": float(np.mean([r["false_positive_rate"] for r in healthy])),
                           "actually_accepted_noncanonical_positions": {
                               "n": len(normal_variant),
                               "false_alarms": sum(r["predicted_error"] for r in normal_variant),
                               "pooled_false_positive_rate": float(np.mean([r["predicted_error"] for r in normal_variant])) if normal_variant else None,
                               "note": "small subset; method-specific development thresholds, not matched-recall comparison"},
                           "calibration_before_after": calibration,
                           "matched_fpr_exact_diagnosis": (
                               {"supported": False, "reason": "scalar detection baseline; no native or declared adapter candidate"}
                               if method in SCALAR_DIAGNOSIS_UNSUPPORTED else macro_matched_fpr_diagnosis(own)),
                           "diagnosis_from_frozen_report": stored[method].get("diagnosis")}

    pairs = {
        "H1_normal_variants_graph": ("M4_dual_graph", "M4_single_reference"),
        "H1_normal_variants_greedy": ("M1_greedy_acceptable", "M0_greedy_single"),
        "H2_learned_prior_same_scale": ("M4_dual_graph", "M4_generic_posterior"),
        "H2_learned_prior_vs_independently_tuned_generic": ("M4_dual_graph", "M4_generic_tuned"),
        "H3_path_sum_vs_best_path_shared_parameters": ("M4_dual_graph", "M3_learned_viterbi"),
    }
    comparisons = {}
    for hypothesis, (a, b) in pairs.items():
        delta = {field: results[a]["macro"][field] - results[b]["macro"][field] for field in fields}
        comparisons[hypothesis] = {"a": a, "b": b, "macro_delta_a_minus_b": delta,
            "tie_aware_speaker_macro_aurc_delta_a_minus_b": results[a]["speaker_macro_tie_aware_risk"]["aurc"] - results[b]["speaker_macro_tie_aware_risk"]["aurc"],
            "interpretation": "Higher AUPRC/AUROC/F1 is better; lower FPR/FNR/Brier/NLL/AURC is better"}
        matched_a = results[a]["matched_fpr_exact_diagnosis"]
        matched_b = results[b]["matched_fpr_exact_diagnosis"]
        if matched_a["supported"] and matched_b["supported"]:
            comparisons[hypothesis]["matched_fpr_exact_diagnosis_macro_delta_a_minus_b"] = [
                {"target_fpr": pa["target_fpr"], "events": {
                    event: {metric: pa["macro_events"][event][metric] - pb["macro_events"][event][metric]
                            for metric in ("precision", "recall", "f1")}
                    for event in ("substitution", "deletion", "combined_SUB_DEL")}}
                for pa, pb in zip(matched_a["points"], matched_b["points"])]
    all_gaps = [g for r in labels.values() if r["speaker"] in patients and r["quality_exclusion"] is None
                for g in r["labels"]["gaps"] if g["diagnostic"] and g["stable"]]
    return {"cohort": cohort, "independent_metric_recomputation": "PASS", "results": results,
            "hypothesis_comparisons": comparisons, "coverage_from_frozen_report": report["coverage"],
            "patient_stable_diagnostic_insertion_gaps": len(all_gaps),
            "patient_insertion_gaps_longer_than_cap": sum(len(g["inserted_phones"]) > 2 for g in all_gaps),
            "paired_speaker_comparisons_from_frozen_report": report["paired_comparisons"],
            "insertion_diagnostics_from_frozen_report": report["insertion_diagnostics"],
            "source_directory": str(directory.resolve())}


def markdown_summary(summary):
    lines = ["# Dual-graph independent results summary", "",
             "No fitting or tuning is performed by this report. Ranking and probability metrics are independently checked against the frozen token artifact.", ""]
    for cohort, report in summary["cohorts"].items():
        lines.extend([f"## {cohort}", "", "| Method | Macro AUPRC | Macro F1 | Tie-aware speaker-macro AURC |", "|---|---:|---:|---:|"])
        for method, result in sorted(report["results"].items(), key=lambda x: -x[1]["macro"]["auprc"]):
            lines.append(f"| {method} | {result['macro']['auprc']:.6f} | {result['macro']['f1']:.6f} | {result['speaker_macro_tie_aware_risk']['aurc']:.6f} |")
        lines.extend(["", "| Hypothesis comparison (A − B) | Δ AUPRC | Δ FPR at ≥80% recall | Δ AURC |", "|---|---:|---:|---:|"])
        for hypothesis, comparison in report["hypothesis_comparisons"].items():
            delta = comparison["macro_delta_a_minus_b"]
            lines.append(f"| {hypothesis} | {delta['auprc']:+.6f} | {delta['fpr_at_80pct_recall']:+.6f} | {comparison['tie_aware_speaker_macro_aurc_delta_a_minus_b']:+.6f} |")
        lines.extend(["", "### Descriptive matched-FPR exact diagnosis", "",
                      "Each point is a held-out ROC statistic with randomized boundary ties, not a deployment threshold. Precision/recall/F1 below use the same fixed exact-event subset; cutoffs use every normal detection token. No original prediction is changed.", "",
                      "| Method | Expected normal FPR | SUB macro F1 | DEL macro F1 | Combined exact macro precision | Combined exact macro recall | Combined exact macro F1 |",
                      "|---|---:|---:|---:|---:|---:|---:|"])
        for method, result in sorted(report["results"].items()):
            matched = result["matched_fpr_exact_diagnosis"]
            if not matched["supported"]:
                continue
            for point in matched["points"]:
                events = point["macro_events"]
                combined = events["combined_SUB_DEL"]
                lines.append(f"| {method} | {point['target_fpr']:.0%} | {events['substitution']['f1']:.6f} | {events['deletion']['f1']:.6f} | {combined['precision']:.6f} | {combined['recall']:.6f} | {combined['f1']:.6f} |")
        lines.extend(["", f"Stable diagnostic insertion gaps beyond the two-phone cap: {report['patient_insertion_gaps_longer_than_cap']} / {report['patient_stable_diagnostic_insertion_gaps']}.", ""])
    lines.extend(["## Limitations", "", *[f"- {item}" for item in summary["limitations"]], ""])
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path,
                        default=Path(__file__).resolve().parents[1] / "artifacts" / "evaluation" / "dual_graph" / "dual_graph_v1")
    parser.add_argument("--cohorts", nargs="+", choices=("primary5", "sensitivity7"), default=["primary5", "sensitivity7"])
    parser.add_argument("--output", type=Path, help="New summary output directory; defaults to run-root/independent_summary")
    args = parser.parse_args(argv)
    summary = {"cohorts": {cohort: summarize_cohort(args.run_root / cohort) for cohort in args.cohorts},
               "limitations": LIMITATIONS,
               "risk_definition": "speaker macro expected selective binary risk under random ordering within equal chosen-label-confidence ties"}
    output = args.output or args.run_root / "independent_summary"
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "summary.md").write_text(markdown_summary(summary), encoding="utf-8")
    print(f"Independent summary PASS: {output.resolve()}")


if __name__ == "__main__":
    main()
