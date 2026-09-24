from __future__ import annotations

import math

import numpy as np
import pytest

from da_cf_gop.calibration import DeletionCalibrator, SpeakerWeightedPlattCalibrator
from da_cf_gop.ctc import CounterfactualMatrix
from da_cf_gop.priors import DELETION, fixed_soft_priors
from da_cf_gop.scoring import (
    apply_platt_calibration,
    conditional_alternative_probabilities,
    gop_log_likelihood_ratio,
    score_counterfactual_matrix,
    score_phone,
)


def test_llr_matches_manual_mixture() -> None:
    alternatives = {"B": -2.0, DELETION: -4.0}
    prior = {"B": 0.75, DELETION: 0.25}
    expected = -1.0 - math.log(0.75 * math.exp(-2.0) + 0.25 * math.exp(-4.0))
    assert gop_log_likelihood_ratio(-1.0, alternatives, prior) == pytest.approx(expected)


def test_conditional_probabilities_are_order_invariant() -> None:
    first = conditional_alternative_probabilities(
        {"B": -1.0, "K": -2.0, DELETION: -3.0},
        {"B": 0.5, "K": 0.3, DELETION: 0.2},
    )
    second = conditional_alternative_probabilities(
        {DELETION: -3.0, "K": -2.0, "B": -1.0},
        {"K": 0.3, "B": 0.5, DELETION: 0.2},
    )
    assert first == pytest.approx(second)
    assert sum(first.values()) == pytest.approx(1.0)


def test_deletion_calibration_changes_both_llr_and_conditional_channel() -> None:
    kwargs = {
        "position": 0,
        "target_phone": "T",
        "canonical_log_probability": -2.0,
        "alternative_log_probabilities": {"D": -3.0, DELETION: -1.0},
        "prior": {"D": 0.5, DELETION: 0.5},
    }
    raw = score_phone(**kwargs)
    calibrated = score_phone(
        **kwargs,
        deletion_calibrator=DeletionCalibrator({"T": 2.0}, 2.0),
    )
    assert calibrated.gop > raw.gop
    assert calibrated.candidate_probabilities[DELETION] < raw.candidate_probabilities[DELETION]


def test_official_inventory_score_exposes_39_error_probabilities_and_margin() -> None:
    prior = fixed_soft_priors()["T"]
    likelihoods = {candidate: -float(index % 7) for index, candidate in enumerate(prior)}
    row = score_phone(
        position=3,
        target_phone="T",
        canonical_log_probability=-4.0,
        alternative_log_probabilities=likelihoods,
        prior=prior,
    )
    assert len(row.candidate_probabilities) == 39
    assert sum(row.candidate_probabilities.values()) == pytest.approx(1.0)
    ranked = sorted(row.candidate_probabilities.values(), reverse=True)
    assert row.top2_margin == pytest.approx(ranked[0] - ranked[1])


def test_matrix_scoring_and_platt_attachment() -> None:
    candidate_ids = np.asarray([[2, 3, -1], [1, 3, -1]])
    likelihoods = np.asarray([[-2.0, -3.0, -4.0], [-2.5, -3.5, -4.5]])
    matrix = CounterfactualMatrix(
        -1.0,
        1.0,
        -2.0,
        candidate_ids,
        likelihoods,
        np.ones_like(likelihoods),
        likelihoods,
    )
    priors = {
        "AA": {"AE": 0.4, "AH": 0.4, DELETION: 0.2},
        "AE": {"AA": 0.4, "AH": 0.4, DELETION: 0.2},
    }
    rows = score_counterfactual_matrix(
        matrix, ["AA", "AE"], {0: "<pad>", 1: "AA", 2: "AE", 3: "AH"}, priors
    )
    assert len(rows) == 2
    platt = SpeakerWeightedPlattCalibrator(1.0, 0.0)
    calibrated = apply_platt_calibration(rows, platt)
    assert all(row.error_probability is not None for row in calibrated)
    assert calibrated[0].candidate_probabilities == rows[0].candidate_probabilities

