from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from da_cf_gop.ctc import (
    DELETION_ID,
    counterfactual_log_probability_matrix,
    ctc_log_probability_numpy,
    ctc_log_probabilities_batch,
    log_softmax,
    topology_corrected_log_probability,
    uniform_ctc_log_path_count,
)


def _collapse(path: tuple[int, ...], blank: int) -> list[int]:
    output: list[int] = []
    previous = None
    for value in path:
        if value != previous and value != blank:
            output.append(value)
        previous = value
    return output


@pytest.mark.parametrize("blank", [0, 3])
@pytest.mark.parametrize("labels", [[], [1], [1, 2], [1, 1]])
def test_numpy_and_torch_match_bruteforce(blank: int, labels: list[int]) -> None:
    torch = pytest.importorskip("torch")
    del torch
    vocabulary = 4
    labels = [value if value != blank else (value + 1) % vocabulary for value in labels]
    frames = max(4, len(labels) + sum(a == b for a, b in zip(labels, labels[1:])))
    rng = np.random.default_rng(42 + blank + len(labels))
    probabilities = np.exp(log_softmax(rng.normal(size=(frames, vocabulary))))
    brute = sum(
        float(np.prod([probabilities[index, value] for index, value in enumerate(path)]))
        for path in itertools.product(range(vocabulary), repeat=frames)
        if _collapse(path, blank) == labels
    )
    log_probs = np.log(probabilities)
    oracle = ctc_log_probability_numpy(log_probs, labels, blank)
    batched = ctc_log_probabilities_batch(log_probs, [labels], blank, device="cpu")[0]
    assert math.exp(oracle) == pytest.approx(brute, abs=1e-12)
    assert batched == pytest.approx(oracle, abs=1e-10)


def test_path_count_handles_repeats_substitution_and_deletion() -> None:
    assert math.exp(uniform_ctc_log_path_count(2, [1])) == pytest.approx(3.0)
    assert math.exp(uniform_ctc_log_path_count(3, [1, 2])) == pytest.approx(5.0)
    assert math.exp(uniform_ctc_log_path_count(3, [1, 1])) == pytest.approx(1.0)
    assert uniform_ctc_log_path_count(2, [1, 1]) == -np.inf
    raw = math.log(5.0) - 3 * math.log(4.0)
    assert topology_corrected_log_probability(raw, 3, [1, 2]) == pytest.approx(-3 * math.log(4.0))


def test_full_counterfactual_matrix_contract() -> None:
    rng = np.random.default_rng(8)
    log_probs = log_softmax(rng.normal(size=(7, 5)))
    result = counterfactual_log_probability_matrix(
        log_probs, [1, 2, 2], blank_id=0, device="cpu"
    )
    # Four nonblank phones: three substitutions plus deletion per target.
    assert result.candidate_ids.shape == (3, 4)
    assert result.log_probabilities.shape == (3, 4)
    assert np.all(result.candidate_ids[:, -1] == DELETION_ID)
    assert all(target not in row for target, row in zip([1, 2, 2], result.candidate_ids))
    deletion_expected = ctc_log_probability_numpy(log_probs, [2, 2], blank_id=0)
    assert result.log_probabilities[0, -1] == pytest.approx(deletion_expected, abs=1e-10)
    assert result.canonical_corrected_log_probability == pytest.approx(
        result.canonical_log_probability - result.canonical_log_path_count
    )


def test_invalid_and_impossible_sequences_fail_explicitly() -> None:
    log_probs = log_softmax(np.zeros((2, 4)))
    assert ctc_log_probability_numpy(log_probs, [1, 1], 0) == -np.inf
    assert ctc_log_probabilities_batch(log_probs, [[1, 1]], 0, device="cpu")[0] == -np.inf
    with pytest.raises(ValueError, match="blank"):
        ctc_log_probability_numpy(log_probs, [0], 0)
    with pytest.raises(ValueError, match="outside"):
        ctc_log_probability_numpy(log_probs, [4], 0)
    with pytest.raises(ValueError, match="NaN"):
        log_softmax(np.asarray([[0.0, np.nan]]))


def test_factorized_counterfactuals_match_numpy_oracle_with_repeats() -> None:
    """Every compact CPU entry is still a complete standard CTC likelihood."""

    rng = np.random.default_rng(20260829)
    for vocabulary in (3, 5, 7):
        for length in range(1, 6):
            for frames in (1, 3, 7):
                canonical = rng.integers(1, vocabulary, size=length).tolist()
                log_probs = log_softmax(rng.normal(size=(frames, vocabulary)))
                result = counterfactual_log_probability_matrix(
                    log_probs, canonical, blank_id=0, device="cpu"
                )
                expected_canonical = ctc_log_probability_numpy(
                    log_probs, canonical, blank_id=0
                )
                assert result.canonical_log_probability == pytest.approx(
                    expected_canonical, abs=1.0e-12
                )
                for position, candidates in enumerate(result.candidate_ids):
                    for column, candidate in enumerate(candidates):
                        hypothesis = (
                            canonical[:position] + canonical[position + 1 :]
                            if candidate == DELETION_ID
                            else canonical[:position]
                            + [int(candidate)]
                            + canonical[position + 1 :]
                        )
                        expected = ctc_log_probability_numpy(
                            log_probs, hypothesis, blank_id=0
                        )
                        actual = result.log_probabilities[position, column]
                        if np.isneginf(expected):
                            assert np.isneginf(actual)
                        else:
                            assert actual == pytest.approx(expected, abs=1.0e-12)
                        assert result.log_path_counts[position, column] == pytest.approx(
                            uniform_ctc_log_path_count(frames, hypothesis), abs=1.0e-12
                        )


def test_cuda_padded_counterfactuals_match_general_batch() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    rng = np.random.default_rng(81)
    canonical = [1, 1, 2, 3]
    log_probs = log_softmax(rng.normal(size=(9, 6)))
    result = counterfactual_log_probability_matrix(
        log_probs, canonical, blank_id=0, device="cuda", batch_size=2
    )
    hypotheses: list[list[int]] = [canonical]
    for position, candidates in enumerate(result.candidate_ids):
        for candidate in candidates:
            hypotheses.append(
                canonical[:position] + canonical[position + 1 :]
                if candidate == DELETION_ID
                else canonical[:position]
                + [int(candidate)]
                + canonical[position + 1 :]
            )
    reference = ctc_log_probabilities_batch(
        log_probs, hypotheses, blank_id=0, device="cuda", batch_size=2
    )
    actual = np.concatenate(
        ([result.canonical_log_probability], result.log_probabilities.ravel())
    )
    np.testing.assert_array_equal(actual, reference)
