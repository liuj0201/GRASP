import importlib.util
import gzip
import json
from itertools import permutations
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "summarize_dual_graph_results.py"
spec = importlib.util.spec_from_file_location("dual_summary", SCRIPT)
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def rows_for(errors, probability=0.8):
    return [{"p_error": probability, "predicted_error": True, "gold_error": not error} for error in errors]


def test_ties_independent_of_input_order():
    rows = rows_for([1, 1, 0, 0])
    result = summary.tie_aware_risk(rows)
    assert result == summary.tie_aware_risk(list(reversed(rows)))
    assert all(point["expected_binary_risk"] == 0.5 for point in result["curve"])
    assert np.isclose(result["aurc"], 0.4375)


def test_tie_expectation_matches_average_of_all_permutations():
    exact = []
    for error_order in set(permutations([1, 1, 0, 0])):
        risk = np.cumsum(error_order) / np.arange(1, 5)
        exact.append(np.trapz(np.r_[0, risk], np.arange(5) / 4))
    result = summary.tie_aware_risk(rows_for([1, 1, 0, 0]))
    assert np.isclose(result["aurc"], np.mean(exact))


def test_chosen_class_confidence_not_distance_from_half():
    # First is an error prediction under development threshold 0.1, but has
    # only 0.11 confidence in that chosen class. Second must rank ahead of it.
    rows = [{"p_error": 0.11, "predicted_error": True, "gold_error": False},
            {"p_error": 0.2, "predicted_error": False, "gold_error": False}]
    result = summary.tie_aware_risk(rows, (0.5, 1.0))
    assert result["curve"][0]["expected_binary_risk"] == 0
    assert result["curve"][1]["expected_binary_risk"] == 0.5


def test_untied_risk_matches_direct_curve():
    rows = rows_for([0, 1, 1])
    rows[0]["p_error"], rows[1]["p_error"], rows[2]["p_error"] = 0.9, 0.8, 0.7
    actual = summary.tie_aware_risk(rows)
    expected = np.trapz([0, 0, 0.5, 2/3], np.arange(4) / 3)
    assert np.isclose(actual["aurc"], expected)


def test_speaker_macro_not_token_weighted():
    per_speaker = [{"risk": summary.tie_aware_risk(rows_for([1] * 100))},
                   {"risk": summary.tie_aware_risk(rows_for([0] * 10))}]
    result = summary.macro_risk(per_speaker)
    assert result["curve"][-1]["expected_binary_risk"] == 0.5
    assert np.isclose(result["aurc"], 0.4975)


def test_no_parameters_are_fitted_for_before_after_calibration():
    rows = [{"gold_error": True, "p_error": 0.8, "graph_posterior_error": 0.6},
            {"gold_error": False, "p_error": 0.2, "graph_posterior_error": 0.4}]
    assert np.isclose(summary.error_metrics(rows)["brier"], 0.04)
    assert np.isclose(summary.error_metrics(rows, "graph_posterior_error")["brier"], 0.16)


def test_full_artifact_summary_independently_checks_metrics(tmp_path):
    methods = ("M4_dual_graph", "M4_single_reference", "M1_greedy_acceptable", "M0_greedy_single",
               "M4_generic_posterior", "M4_generic_tuned", "M3_learned_viterbi")
    patients = ("F03", "F04")
    rows, labels = [], []
    for speaker in (*patients, "MC04"):
        for error in (False, True):
            event = f"{speaker}_{error}"
            labels.append({"event": event, "speaker": speaker, "quality_exclusion": None,
                           "acceptable_phones": [["T"]], "labels": {"gaps": []}})
            for method in methods:
                rows.append({"method": method, "speaker": speaker, "fold": "fold_1", "event": event,
                             "phone_index": 0, "target": "T", "gop": -1 if error else 1,
                             "p_error": 0.8 if error else 0.2, "predicted_error": error,
                             "gold_error": error, "gold_event": "substitution" if error else "match",
                             "gold_realized": "D" if error else "T", "stable_event": True,
                             "graph_posterior_error": 0.7 if error else 0.3})
    stored = []
    for method in methods:
        scores = summary.independent_metrics([r for r in rows if r["method"] == method and r["speaker"] == "F03"])
        stored.append({"method": method, "macro": scores, "diagnosis": None})
    report = {"cohort": "primary5", "results": stored, "coverage": [], "paired_comparisons": [],
              "insertion_diagnostics": []}
    (tmp_path / "metrics.json").write_text(json.dumps(report), encoding="utf-8")
    (tmp_path / "config.json").write_text(json.dumps({"cohorts": {"primary5": patients}}), encoding="utf-8")
    for name, values in (("phone_predictions.jsonl.gz", rows), ("evaluation_labels.jsonl.gz", labels)):
        with gzip.open(tmp_path / name, "wt", encoding="utf-8") as handle:
            for row in values:
                handle.write(json.dumps(row) + "\n")
    output = summary.summarize_cohort(tmp_path)
    assert output["independent_metric_recomputation"] == "PASS"
    assert output["results"]["M4_dual_graph"]["macro"]["auprc"] == 1.0
    assert output["hypothesis_comparisons"]["H3_path_sum_vs_best_path_shared_parameters"]["macro_delta_a_minus_b"]["auprc"] == 0
    report["results"][0]["macro"]["auprc"] = 0.5
    (tmp_path / "metrics.json").write_text(json.dumps(report), encoding="utf-8")
    import pytest
    with pytest.raises(ValueError, match="disagrees with final metrics"):
        summary.summarize_cohort(tmp_path)


def diagnostic_row(score, gold_event, candidate_event, *, stable=True, candidate_phone="D", speaker="F03"):
    return {"gop": -score, "gold_error": gold_event != "match", "stable_event": stable,
            "gold_event": gold_event, "gold_realized": "D" if gold_event == "substitution" else None,
            "candidate_event": candidate_event, "candidate_phone": candidate_phone, "speaker": speaker}


def test_matched_fpr_boundary_weights_exactly_match_expected_false_alarms():
    rows = [diagnostic_row(3, "match", "substitution") for _ in range(2)]
    rows += [diagnostic_row(1, "match", "match") for _ in range(18)]
    rows += [diagnostic_row(3, "substitution", "substitution")]
    weights, point = summary.matched_fpr_weights(rows, 0.05)
    assert point["boundary_selection_probability"] == 0.5
    assert point["achieved_expected_fpr"] == 0.05
    assert np.all(weights[:2] == 0.5)
    assert weights[-1] == 0.5
    assert not weights[2:20].any()


def test_matched_fpr_never_fabricates_candidates_or_labels_uncertain_proposals():
    rows = [diagnostic_row(2, "match", "substitution"),
            diagnostic_row(2, "match", "match"),
            diagnostic_row(2, "substitution", "substitution"),
            diagnostic_row(3, "deletion", "match"),
            diagnostic_row(2, "alignment_uncertain", "substitution", stable=False)]
    point = summary.exact_diagnosis_at_fpr(rows, 0.25)
    assert point["achieved_expected_fpr"] == 0.25
    sub = point["events"]["substitution"]
    assert sub["gold_exact_events"] == 1
    assert sub["expected_specific_proposals"] == 0.5
    assert sub["expected_true_positives"] == 0.25
    assert sub["precision"] == 0.5
    assert sub["recall"] == 0.25
    assert point["events"]["deletion"]["gold_exact_events"] == 1
    assert point["events"]["deletion"]["expected_specific_proposals"] == 0
    assert point["exact_unverifiable_specific_proposal_mass"] == 0.25
    assert point["exact_unverifiable_specific_proposal_coverage"] == 0.25
    assert point == summary.exact_diagnosis_at_fpr(list(reversed(rows)), 0.25)


def test_matched_fpr_is_identical_across_methods_and_speakers_with_different_ties():
    rows = []
    for speaker, normal_count in (("F03", 20), ("F04", 100)):
        rows.extend(diagnostic_row(i % 3, "match", "substitution", speaker=speaker) for i in range(normal_count))
        rows.append(diagnostic_row(5, "substitution", "substitution", speaker=speaker))
    other_method = [{**r, "gop": r["gop"] * 3 + 7} for r in rows]
    a, b = summary.macro_matched_fpr_diagnosis(rows), summary.macro_matched_fpr_diagnosis(other_method)
    for point_a, point_b in zip(a["points"], b["points"]):
        assert np.isclose(point_a["macro_achieved_expected_fpr"], point_a["target_fpr"])
        assert np.isclose(point_b["macro_achieved_expected_fpr"], point_a["target_fpr"])
        for sa, sb in zip(point_a["per_speaker"], point_b["per_speaker"]):
            assert np.isclose(sa["achieved_expected_fpr"], point_a["target_fpr"])
            assert sa["events"] == sb["events"]


def test_matched_fpr_diagnostic_true_positive_requires_exact_phone_identity():
    rows = [diagnostic_row(0, "match", "match"),
            diagnostic_row(1, "substitution", "substitution", candidate_phone="T")]
    point = summary.exact_diagnosis_at_fpr(rows, 0.05)
    assert point["events"]["substitution"]["expected_specific_proposals"] == 1
    assert point["events"]["substitution"]["expected_true_positives"] == 0
