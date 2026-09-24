from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

import da_cf_gop.ctc_sf as ctc_sf_module

from da_cf_gop.ctc_sf import (
    ctc_log_probability,
    log_probs_from_logits,
    segmentation_free_score,
    segmentation_free_scores,
)


def _collapse(path: tuple[int, ...], blank: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    labels: list[int] = []
    runs: list[int] = []
    previous = blank
    for token in path:
        if token == blank:
            previous = blank
        elif token == previous:
            runs[-1] += 1
        else:
            labels.append(token)
            runs.append(1)
            previous = token
    return tuple(labels), tuple(runs)


def _path_probability(log_probs: np.ndarray, path: tuple[int, ...]) -> float:
    return math.exp(sum(log_probs[frame, token] for frame, token in enumerate(path)))


def _brute_ctc(log_probs: np.ndarray, target: tuple[int, ...], blank: int) -> float:
    return sum(
        _path_probability(log_probs, path)
        for path in itertools.product(range(log_probs.shape[1]), repeat=log_probs.shape[0])
        if _collapse(path, blank)[0] == target
    )


def _brute_sd(
    log_probs: np.ndarray,
    canonical: tuple[int, ...],
    position: int,
    blank: int,
) -> tuple[float, float]:
    prefix = canonical[:position]
    suffix = canonical[position + 1 :]
    total = 0.0
    weighted_occurrence = 0.0
    for path in itertools.product(range(log_probs.shape[1]), repeat=log_probs.shape[0]):
        collapsed, runs = _collapse(path, blank)
        if len(collapsed) < len(prefix) + len(suffix):
            continue
        if collapsed[: len(prefix)] != prefix:
            continue
        if suffix and collapsed[-len(suffix) :] != suffix:
            continue
        middle_length = len(collapsed) - len(prefix) - len(suffix)
        if middle_length not in (0, 1):
            continue
        probability = _path_probability(log_probs, path)
        occurrence = sum(runs[len(prefix) : len(prefix) + middle_length])
        total += probability
        weighted_occurrence += probability * occurrence
    return total, weighted_occurrence / total


@pytest.fixture
def tiny_log_probs() -> np.ndarray:
    return log_probs_from_logits(
        np.asarray(
            [
                [1.4, 0.2, -0.7],
                [-0.3, 1.2, 0.1],
                [0.8, -0.2, 1.1],
                [0.0, 0.7, 1.3],
                [1.0, 0.4, -0.4],
            ],
            dtype=np.float64,
        )
    )


def test_ctc_forward_matches_exhaustive_paths(tiny_log_probs: np.ndarray) -> None:
    for target in ((), (1,), (1, 2), (1, 1)):
        expected = _brute_ctc(tiny_log_probs, target, 0)
        actual = math.exp(ctc_log_probability(tiny_log_probs, target, blank_id=0))
        assert actual == pytest.approx(expected, abs=1e-12)


def test_sd_language_matches_exhaustive_paths(tiny_log_probs: np.ndarray) -> None:
    canonical = (1, 2)
    for position in range(len(canonical)):
        expected_probability, expected_occurrence = _brute_sd(
            tiny_log_probs, canonical, position, 0
        )
        result = segmentation_free_score(
            tiny_log_probs, canonical, position, blank_id=0
        )
        assert math.exp(result.denominator_log_probability) == pytest.approx(
            expected_probability, abs=1e-12
        )
        assert result.posterior_occurrence_diagnostic == pytest.approx(
            expected_occurrence, abs=1e-12
        )
        assert result.score <= 1e-12
        assert result.normalized_score == pytest.approx(
            result.score / max(1.0, result.occurrence)
        )


def test_sd_retains_repeated_neighbour_cases(tiny_log_probs: np.ndarray) -> None:
    for canonical in ((1, 1), (1, 2, 1)):
        for position in range(len(canonical)):
            expected, _ = _brute_sd(tiny_log_probs, canonical, position, 0)
            result = segmentation_free_score(
                tiny_log_probs, canonical, position, blank_id=0
            )
            assert math.exp(result.denominator_log_probability) == pytest.approx(
                expected, abs=1e-12
            )


def test_position_batching_matches_single_position(tiny_log_probs: np.ndarray) -> None:
    batched = segmentation_free_scores(
        tiny_log_probs,
        (1, 2),
        blank_id=0,
        compute_posterior_diagnostic=True,
    )
    for position, batch_result in enumerate(batched):
        single = segmentation_free_score(tiny_log_probs, (1, 2), position, blank_id=0)
        assert batch_result.denominator_log_probability == pytest.approx(
            single.denominator_log_probability, abs=1e-12
        )
        assert batch_result.occurrence == pytest.approx(single.occurrence, abs=1e-12)
        assert batch_result.posterior_occurrence_diagnostic == pytest.approx(
            single.posterior_occurrence_diagnostic, abs=1e-12
        )


def test_work_guard_fails_closed(tiny_log_probs: np.ndarray) -> None:
    with pytest.raises(RuntimeError, match="rather than silently truncating"):
        segmentation_free_scores(
            tiny_log_probs, (1, 2), blank_id=0, max_work=1
        )


def test_compact_sd_graph_matches_audited_reference_randomized() -> None:
    rng = np.random.default_rng(20260829)
    for vocabulary in (3, 5, 8):
        phones = tuple(range(1, vocabulary))
        for length in range(1, 6):
            canonical = tuple(rng.integers(1, vocabulary, size=length))
            log_probs = log_probs_from_logits(rng.normal(size=(7, vocabulary)))
            languages = [
                ctc_sf_module._SDLanguage(
                    canonical[:position], canonical[position + 1 :], phones
                )
                for position in range(length)
            ]
            compact = ctc_sf_module._wildcard_forward_many(
                log_probs,
                languages,
                blank_id=0,
                phone_ids=phones,
                max_work=None,
                compute_posterior_occurrence=True,
            )
            reference = ctc_sf_module._wildcard_forward_many_reference(
                log_probs,
                languages,
                blank_id=0,
                phone_ids=phones,
                max_work=None,
                compute_posterior_occurrence=True,
            )
            for actual, expected in zip(compact, reference):
                assert actual.log_probability == pytest.approx(
                    expected.log_probability, abs=1.0e-12
                )
                assert actual.paper_forward_occurrence == pytest.approx(
                    expected.paper_forward_occurrence, abs=1.0e-12
                )
                assert actual.posterior_occurrence == pytest.approx(
                    expected.posterior_occurrence, abs=1.0e-12
                )
