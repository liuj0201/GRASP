from __future__ import annotations

import numpy as np
import pytest

from da_cf_gop.baselines import (
    CONVENTIONAL_METHOD,
    CTC_SA_METHOD,
    CTC_SF_SD_METHOD,
    CTC_SF_SD_NORM_METHOD,
    SCORE_DIRECTION,
    conventional_forced_alignment_gop,
    legacy_hard_candidate_graph,
    project_priors_to_hard_graph,
    same_backend_scalar_baselines,
)
from da_cf_gop.ctc_sf import log_probs_from_logits
from da_cf_gop.priors import DELETION, DEFAULT_PHONE_INVENTORY, fixed_soft_priors


def _easy_timeline() -> np.ndarray:
    # blank=0, target phones=1 and 2, distractor=3
    return log_probs_from_logits(
        np.asarray(
            [
                [7.0, 1.0, -4.0, -3.0],
                [0.0, 8.0, -4.0, -2.0],
                [0.0, 8.0, -4.0, -2.0],
                [7.0, 0.0, 0.0, -3.0],
                [0.0, -4.0, 8.0, -2.0],
                [7.0, -3.0, 1.0, -4.0],
            ],
            dtype=np.float64,
        )
    )


def test_forced_alignment_baseline_has_auditable_segments() -> None:
    scores = conventional_forced_alignment_gop(
        _easy_timeline(), (1, 2), blank_id=0, phone_ids=(1, 2, 3)
    )
    assert len(scores) == 2
    assert [row.position for row in scores] == [0, 1]
    assert [row.phone_id for row in scores] == [1, 2]
    assert all(row.frame_count == row.end_frame - row.start_frame for row in scores)
    assert scores[0].end_frame <= scores[1].start_frame
    assert all(row.score > row.top_competitor_score for row in scores)
    assert all(row.posterior_margin > 0.0 for row in scores)
    assert all(row.score_direction == SCORE_DIRECTION for row in scores)


def test_forced_alignment_score_is_log_mean_target_posterior() -> None:
    values = _easy_timeline()
    row = conventional_forced_alignment_gop(
        values, (1, 2), blank_id=0, phone_ids=(1, 2, 3)
    )[0]
    expected = np.log(np.exp(values[row.start_frame : row.end_frame, 1]).mean())
    assert row.score == pytest.approx(expected, abs=1e-12)


def test_same_backend_suite_is_complete_and_position_aligned() -> None:
    result = same_backend_scalar_baselines(
        _easy_timeline(), (1, 2), blank_id=0, phone_ids=(1, 2, 3)
    )
    assert set(result) == {
        CONVENTIONAL_METHOD,
        CTC_SA_METHOD,
        CTC_SF_SD_METHOD,
        CTC_SF_SD_NORM_METHOD,
    }
    assert all(len(values) == 2 for values in result.values())
    assert all(np.isfinite(values).all() for values in result.values())
    assert all(value <= 1e-10 for value in result[CTC_SF_SD_METHOD])


def test_legacy_hard_graph_is_exact_true_pruning_and_deterministic() -> None:
    first = legacy_hard_candidate_graph()
    second = legacy_hard_candidate_graph()
    assert first == second
    assert set(first) == set(DEFAULT_PHONE_INVENTORY)
    assert set(first["P"]) == {"B", "F", "M", DELETION}
    assert "T" not in first["P"]
    assert len(first["P"]) < len(DEFAULT_PHONE_INVENTORY)
    assert sum(first["P"].values()) == pytest.approx(1.0)
    assert all(value > 0.0 for row in first.values() for value in row.values())


def test_full_prior_projection_retains_only_hard_candidates() -> None:
    full = fixed_soft_priors()
    hard = legacy_hard_candidate_graph()
    projected = project_priors_to_hard_graph(full, graph=hard)
    for target in DEFAULT_PHONE_INVENTORY:
        assert set(projected[target]) == set(hard[target])
        assert sum(projected[target].values()) == pytest.approx(1.0)
        for candidate in projected[target]:
            ratio = projected[target][candidate] / full[target][candidate]
            assert ratio == pytest.approx(
                1.0 / sum(full[target][key] for key in hard[target])
            )


def test_subset_graph_filters_unavailable_phones() -> None:
    graph = legacy_hard_candidate_graph(("P", "B"))
    assert set(graph["P"]) == {"B", DELETION}
    assert set(graph["B"]) == {"P", DELETION}

