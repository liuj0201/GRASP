from copy import deepcopy

import pytest

from da_cf_gop.phonology import PHONE_TO_CTC_ID
from da_cf_gop.sparse_evaluation import evaluate_predictions, evaluate_run


def _fixture():
    k, t, d = (PHONE_TO_CTC_ID[phone] for phone in ("K", "T", "D"))
    records, predictions = [], []
    for speaker, actual, sparse_score, sparse_detect in (("A", "T", .8, True), ("B", "D", .1, False)):
        records.append({
            "event": speaker, "speaker": speaker, "quality_exclusion": None,
            "acceptable_phones": [["K"], ["K"]],
            "labels": {"tokens": [
                {"diagnostic": True, "stable_error": True, "is_error": True,
                 "stable_event": True, "event_type": "substitution", "realized_phone": actual},
                {"diagnostic": True, "stable_error": True, "is_error": False,
                 "stable_event": True, "event_type": "match", "realized_phone": "K"},
            ]},
        })
        for method in ("full_graph_da", "sparse"):
            for position in (0, 1):
                error_score = .9 if method == "full_graph_da" else sparse_score
                error_detect = True if method == "full_graph_da" else sparse_detect
                score = error_score if position == 0 else .2
                predictions.append({
                    "event": speaker, "speaker": speaker, "fold": speaker, "phone_index": position,
                    "target": "K", "method": method, "gop": -score, "p_error": score,
                    "predicted_error": error_detect if position == 0 else False,
                    "allowed_substitution_phone_ids": [phone for phone in range(1, 40) if phone != k]
                    if method == "full_graph_da" else [d if speaker == "A" else t],
                })
    return predictions, records


def _evaluate(predictions, records):
    return evaluate_predictions(predictions, records, cohort="synthetic", speakers=["A", "B"],
                                methods=["full_graph_da", "sparse"], expected_count=4,
                                bootstrap_replicates=100, seed=1)


def test_candidate_omissions_stay_in_detection_metrics_even_if_detector_flags_them():
    predictions, records = _fixture()
    report, misses = _evaluate(predictions, records)
    methods = {row["method"]: row for row in report["results"]}
    assert methods["full_graph_da"]["candidate_retention"]["retention_rate"] == 1
    assert methods["sparse"]["candidate_retention"]["retention_rate"] == 0
    assert methods["sparse"]["pooled"]["n"] == 4
    assert methods["sparse"]["macro"]["auprc"] == pytest.approx(.75)
    assert methods["sparse"]["macro"]["f1"] == pytest.approx(.5)
    assert methods["sparse"]["macro"]["false_negative_rate"] == pytest.approx(.5)
    assert methods["sparse"]["candidate_retention"]["missing_candidate_detection"]["detection_error_rate"] == .5
    assert len(misses) == 2 and sum(row["binary_detection_error"] for row in misses) == 1
    comparison = report["paired_comparisons"][0]
    assert comparison["auprc"]["difference_a_minus_b"] == pytest.approx(-.25)
    assert comparison["false_negative_rate"]["difference_a_minus_b"] == pytest.approx(.5)
    assert all(not any(field.startswith("gold") for field in row) for row in predictions)


def test_uncertain_event_is_not_a_substitution_retention_case_but_stays_in_detection():
    predictions, records = _fixture()
    records[0]["labels"]["tokens"][0]["stable_event"] = False
    report, misses = _evaluate(predictions, records)
    sparse = report["results"][1]
    assert sparse["pooled"]["n"] == 4
    assert sparse["candidate_retention"]["n_stable_substitutions"] == 1
    assert len(misses) == 1 and misses[0]["speaker"] == "B"


def test_methods_must_share_keys_and_expected_coverage():
    predictions, records = _fixture()
    with pytest.raises(ValueError, match="identical token keys"):
        _evaluate(predictions[:-1], records)
    with pytest.raises(ValueError, match="duplicate"):
        _evaluate(predictions + [deepcopy(predictions[0])], records)
    records[0]["labels"]["tokens"][0]["diagnostic"] = False
    with pytest.raises(ValueError, match="expected 4"):
        _evaluate(predictions, records)


def test_missing_completion_marker_blocks_before_any_configuration_or_gold_read(tmp_path):
    with pytest.raises(RuntimeError, match="predictions_complete"):
        evaluate_run(tmp_path)
