from __future__ import annotations

import numpy as np
import pytest

from da_cf_gop.priors import (
    DELETION,
    LAMBDA_GRID,
    empirical_confusion_priors,
    feature_distance,
    fixed_soft_priors,
    literature_multiplier,
    mix_soft_priors,
    select_lambda_by_macro_auprc,
)


def test_default_prior_is_all_phone_strict_simplex() -> None:
    priors = fixed_soft_priors()
    assert len(priors) == 39
    for target, weights in priors.items():
        assert len(weights) == 39  # 38 substitutions + <DEL>
        assert target not in weights
        assert DELETION in weights
        assert min(weights.values()) > 0
        assert sum(weights.values()) == pytest.approx(1.0)
    assert feature_distance("P", "B") < feature_distance("P", "IY")
    assert priors["P"]["B"] > priors["P"]["IY"]
    assert literature_multiplier("P", DELETION) == 2.0
    assert literature_multiplier("IY", DELETION) == 1.1
    assert literature_multiplier("P", "IY") == 0.7


def test_empirical_prior_is_stable_only_and_speaker_equal() -> None:
    inventory = ("AA", "AE", "AH")
    fixed = fixed_soft_priors(inventory)
    records = []
    # D1 has many duplicated AA->AE errors; D2 has AA->AH.  Both speakers get
    # equal group weight, rather than D1 receiving 100x influence.
    records += [
        {"speaker_id": "D1", "group": "dysarthric", "target": "AA", "realized": "AE", "stable": True}
        for _ in range(100)
    ]
    records += [
        {"speaker_id": "D2", "group": "dysarthric", "target": "AA", "realized": "AH", "stable": True}
        for _ in range(5)
    ]
    records += [
        {"speaker_id": "H1", "group": "healthy", "target": "AA", "realized": "AA", "stable": True}
        for _ in range(20)
    ]
    # This large unstable block must have no effect.
    records += [
        {"speaker_id": "D3", "group": "dysarthric", "target": "AA", "realized": DELETION, "stable": False}
        for _ in range(1000)
    ]
    empirical = empirical_confusion_priors(
        records, inventory, fixed=fixed, pseudocount=20.0, enrichment_cap=4.0
    )
    assert empirical["AA"]["AE"] > fixed["AA"]["AE"]
    assert empirical["AA"][DELETION] < fixed["AA"][DELETION]
    assert sum(empirical["AA"].values()) == pytest.approx(1.0)


def test_epsilon_mixture_keeps_every_candidate_and_obeys_lambda() -> None:
    inventory = ("AA", "AE", "AH")
    fixed = fixed_soft_priors(inventory)
    empirical = {target: dict(weights) for target, weights in fixed.items()}
    empirical["AA"] = {"AE": 0.8, "AH": 0.1, DELETION: 0.1}
    mixed = mix_soft_priors(fixed, empirical, 1.0, uniform_epsilon=0.05)
    assert mixed["AA"]["AE"] == pytest.approx(0.05 / 3 + 0.95 * 0.8)
    assert min(mixed["AA"].values()) >= 0.05 / 3
    with pytest.raises(ValueError, match="lambda"):
        mix_soft_priors(fixed, empirical, 0.4)


def test_lambda_selection_uses_macro_speaker_auprc_and_tie_break() -> None:
    labels = [0, 1, 0, 1, 0, 1, 0, 1]
    speakers = ["A"] * 4 + ["B"] * 4
    perfect = np.asarray([0.1, 0.9, 0.2, 0.8, 0.1, 0.9, 0.2, 0.8])
    poor = 1.0 - perfect
    scores = {value: poor for value in LAMBDA_GRID}
    scores[0.25] = perfect
    scores[0.5] = perfect.copy()
    selected = select_lambda_by_macro_auprc(scores, labels, speakers)
    assert selected.lambda_value == 0.25
    assert selected.macro_auprc == pytest.approx(1.0)
