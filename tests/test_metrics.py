from __future__ import annotations

import math

import pytest

from da_cf_gop.llm_schema import ARPABET_39
from da_cf_gop.metrics import (
    DELETE,
    MetricError,
    binary_metrics,
    classification_precision_at_coverages,
    deletion_metrics,
    evaluate_predictions,
    exact_sign_flip_test,
    paired_speaker_bootstrap,
    risk_coverage_curve,
    specific_output_metrics,
    substitution_rank_metrics,
    validate_prediction_row,
)


def prediction(
    *,
    speaker: str = "F03",
    event: str = "e1",
    index: int = 0,
    method: str = "DA-CF adapted",
    cohort: str = "primary5",
    target: str = "T",
    gold_event: str = "correct",
    realized: str | None = "T",
    p_error: float = 0.1,
    top_alt: str = "D",
    stable: bool = True,
) -> dict:
    candidates = sorted((ARPABET_39 - {target}) | {DELETE})
    second = next(phone for phone in candidates if phone != top_alt)
    probabilities = {phone: 0.0 for phone in candidates}
    probabilities[top_alt] = 0.50
    probabilities[second] = 0.20
    remainder = [phone for phone in candidates if phone not in {top_alt, second}]
    for phone in remainder:
        probabilities[phone] = 0.30 / len(remainder)
    return {
        "method": method,
        "cohort": cohort,
        "fold": f"outer_{speaker}",
        "speaker": speaker,
        "event": event,
        "phone_index": index,
        "target": target,
        "gold_event": gold_event,
        "gold_realized": realized,
        "stable": stable,
        "gop": 1.0 - p_error,
        "p_error": p_error,
        "top_alt": top_alt,
        "p_alt_given_error": 0.50,
        "candidate_probabilities": probabilities,
        "top2_margin": 0.30,
    }


def test_binary_metrics_are_calibrated_and_no_nan() -> None:
    result = binary_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert result["auprc"] == 1.0
    assert result["auroc"] == 1.0
    assert result["brier"] == pytest.approx(0.025)
    assert math.isfinite(result["nll"])
    one_class = binary_metrics([0, 0], [0.1, 0.2])
    assert one_class["auprc"] is None and one_class["auroc"] is None


def test_prediction_contract_checks_candidate_posterior() -> None:
    row = prediction()
    validate_prediction_row(row)
    reordered = dict(reversed(list(row["candidate_probabilities"].items())))
    row["candidate_probabilities"] = reordered
    validate_prediction_row(row)
    row["top2_margin"] = 0.2
    with pytest.raises(MetricError, match="top2_margin"):
        validate_prediction_row(row)
    row = prediction()
    row["candidate_log_scores"] = {
        phone: (None if value == 0.0 else math.log(value))
        for phone, value in row["candidate_probabilities"].items()
    }
    validate_prediction_row(row)
    row["candidate_log_scores"]["D"] = float("inf")
    with pytest.raises(MetricError, match="candidate_log_scores"):
        validate_prediction_row(row)


def test_substitution_topk_mrr_and_deletion() -> None:
    substitution = prediction(
        event="sub", gold_event="substitution", realized="D", p_error=0.9, top_alt="D"
    )
    deletion = prediction(
        event="del", index=1, gold_event="deletion", realized=None,
        p_error=0.8, top_alt=DELETE,
    )
    correct = prediction(event="ok", index=2, p_error=0.1, top_alt=DELETE)
    rank = substitution_rank_metrics([substitution, deletion, correct])
    assert rank == {"n": 1, "top1": 1.0, "top3": 1.0, "mrr": 1.0}
    delete = deletion_metrics([substitution, deletion, correct])
    assert delete["precision"] == 1.0
    assert delete["recall"] == 1.0
    assert delete["f1"] == 1.0


def test_specific_output_metrics_distinguish_abstention() -> None:
    specific = prediction(
        event="sub", gold_event="substitution", realized="D", p_error=0.9, top_alt="D"
    )
    wrong = prediction(
        event="del", index=1, gold_event="deletion", realized=None,
        p_error=0.8, top_alt="D",
    )
    assert specific_output_metrics([specific, wrong])["available"] is False
    specific["specific_output_authorized"] = True
    wrong["specific_output_authorized"] = False
    selected = specific_output_metrics([specific, wrong])
    assert selected["coverage"] == 0.5
    assert selected["precision"] == 1.0


def test_risk_coverage_and_macro_speaker_evaluation() -> None:
    curve = risk_coverage_curve([0, 1, 0, 1], [0.01, 0.99, 0.49, 0.51])
    assert curve["risk"][0] == 0.0
    assert curve["risk"][-1] == 0.0
    precision = classification_precision_at_coverages(
        [0, 1, 0, 1], [0.01, 0.99, 0.49, 0.51]
    )
    assert precision["0.10"]["precision"] == 1.0
    assert precision["1.00"]["precision"] == 1.0

    rows = []
    for speaker in ("F03", "F04"):
        rows.extend([
            prediction(speaker=speaker, event="ok", index=0, p_error=0.1),
            prediction(
                speaker=speaker, event="sub", index=1, p_error=0.9,
                gold_event="substitution", realized="D", top_alt="D",
            ),
        ])
    report = evaluate_predictions(rows)
    result = report["results"][0]
    assert result["macro"]["auprc"] == {"value": 1.0, "n_valid_speakers": 2}
    assert result["macro"]["auroc"] == {"value": 1.0, "n_valid_speakers": 2}
    assert result["macro"]["risk_coverage_auc"] == {
        "value": 0.0,
        "n_valid_speakers": 2,
    }
    assert result["macro"]["precision_at_coverage_0.10"] == {
        "value": 1.0,
        "n_valid_speakers": 2,
    }
    assert set(result["pooled_precision_at_coverage"]) == {
        "0.10", "0.25", "0.50", "0.75", "1.00"
    }
    assert [row["stratum"] for row in result["supplemental_strata"]["canonical_phone"]] == ["T"]
    assert result["supplemental_strata"]["audio_microphone"] == []


def test_paired_speaker_statistics_are_exact_and_deterministic() -> None:
    a = {"F03": 0.9, "F04": 0.8, "M01": 0.7}
    b = {"F03": 0.7, "F04": 0.7, "M01": 0.6}
    first = paired_speaker_bootstrap(a, b, n_bootstrap=1000)
    second = paired_speaker_bootstrap(a, b, n_bootstrap=1000)
    assert first == second
    assert first["difference_a_minus_b"] == pytest.approx(0.1333333333)
    exact = exact_sign_flip_test({speaker: a[speaker] - b[speaker] for speaker in a})
    assert exact["n_permutations"] == 8
    assert 0.0 <= exact["p_two_sided"] <= 1.0
