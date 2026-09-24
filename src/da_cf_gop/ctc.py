"""Exact, forced-alignment-free CTC sequence scoring utilities."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np


NEG_INF = -float("inf")
DELETION_ID = -1


def logsumexp(values: Sequence[float] | np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return NEG_INF
    maximum = float(np.max(array))
    if not np.isfinite(maximum):
        return maximum
    return float(maximum + np.log(np.exp(array - maximum).sum()))


def log_softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable log-softmax which permits masked ``-inf`` logits."""

    values = np.asarray(logits, dtype=np.float64)
    if values.size == 0 or np.isnan(values).any() or np.isposinf(values).any():
        raise ValueError("logits must be non-empty and contain neither NaN nor +inf")
    maximum = np.max(values, axis=axis, keepdims=True)
    if not np.isfinite(maximum).all():
        raise ValueError("each softmax slice must contain at least one finite value")
    shifted = values - maximum
    return shifted - np.log(np.exp(shifted).sum(axis=axis, keepdims=True))


def _validate_log_probs(log_probs: np.ndarray, blank_id: int) -> np.ndarray:
    values = np.asarray(log_probs, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("log_probs must be a [frames, non-empty vocabulary] matrix")
    if np.isnan(values).any() or np.isposinf(values).any():
        raise ValueError("log_probs may contain -inf but not NaN or +inf")
    if values.shape[0] and not np.isfinite(values).any(axis=1).all():
        raise ValueError("every frame must contain at least one finite log-probability")
    if not 0 <= int(blank_id) < values.shape[1]:
        raise ValueError("blank_id is outside the CTC vocabulary")
    return values


def _validate_sequence(sequence: Sequence[int], vocabulary: int, blank_id: int) -> list[int]:
    labels = [int(value) for value in sequence]
    if blank_id in labels:
        raise ValueError("CTC target sequence contains the blank id")
    if any(value < 0 or value >= vocabulary for value in labels):
        raise ValueError("CTC target label is outside the vocabulary")
    return labels


def ctc_minimum_frames(labels: Sequence[int]) -> int:
    sequence = list(labels)
    return len(sequence) + sum(left == right for left, right in zip(sequence, sequence[1:]))


def ctc_log_probability_numpy(
    log_probs: np.ndarray,
    labels: Sequence[int],
    blank_id: int = 0,
) -> float:
    """NumPy dynamic-programming oracle for one standard CTC likelihood."""

    values = _validate_log_probs(log_probs, blank_id)
    labels = _validate_sequence(labels, values.shape[1], blank_id)
    frames = values.shape[0]
    if frames == 0:
        return 0.0 if not labels else NEG_INF
    if frames < ctc_minimum_frames(labels):
        return NEG_INF
    states = np.full(2 * len(labels) + 1, int(blank_id), dtype=np.int64)
    states[1::2] = labels
    previous = np.full(len(states), NEG_INF, dtype=np.float64)
    previous[0] = values[0, blank_id]
    if labels:
        previous[1] = values[0, labels[0]]
    for time in range(1, frames):
        current = np.full_like(previous, NEG_INF)
        for state, label in enumerate(states):
            incoming = [previous[state]]
            if state >= 1:
                incoming.append(previous[state - 1])
            if state >= 2 and label != blank_id and label != states[state - 2]:
                incoming.append(previous[state - 2])
            current[state] = logsumexp(incoming) + values[time, label]
        previous = current
    finals = [previous[-1]]
    if labels:
        finals.append(previous[-2])
    return logsumexp(finals)


# A short public alias is convenient for code which does not need to select an
# implementation.  The NumPy routine is also the independent numerical oracle.
ctc_log_probability = ctc_log_probability_numpy


def ctc_log_probabilities_batch(
    log_probs: np.ndarray,
    sequences: Sequence[Sequence[int]],
    blank_id: int = 0,
    *,
    batch_size: int = 256,
    device: str = "auto",
) -> np.ndarray:
    """Evaluate exact complete-sequence CTC likelihoods with Torch.

    Impossible sequences are retained as ``-inf``.  ``zero_infinity`` is
    deliberately false, so callers must explicitly exclude impossible training
    examples instead of silently converting their losses to zero.
    """

    if not sequences:
        return np.empty(0, dtype=np.float64)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    values = _validate_log_probs(log_probs, blank_id)
    checked = [
        _validate_sequence(sequence, values.shape[1], blank_id) for sequence in sequences
    ]
    if values.shape[0] == 0:
        return np.asarray([0.0 if not sequence else NEG_INF for sequence in checked])

    import torch

    selected_device = (
        "cuda"
        if device == "auto" and torch.cuda.is_available()
        else ("cpu" if device == "auto" else device)
    )
    torch_device = torch.device(selected_device)
    # Native CUDA CTC is most portable in float32; retain float64 on CPU for
    # oracle-grade unit-test precision.
    numeric_dtype = np.float32 if torch_device.type == "cuda" else np.float64
    matrix = torch.from_numpy(np.ascontiguousarray(values, dtype=numeric_dtype)).to(torch_device)
    output: list[np.ndarray] = []
    for offset in range(0, len(checked), batch_size):
        subset = checked[offset : offset + batch_size]
        count = len(subset)
        expanded = matrix.unsqueeze(1).expand(-1, count, -1)
        input_lengths = torch.full(
            (count,), values.shape[0], dtype=torch.long, device=torch_device
        )
        target_lengths = torch.tensor(
            [len(sequence) for sequence in subset], dtype=torch.long, device=torch_device
        )
        targets = torch.tensor(
            [label for sequence in subset for label in sequence],
            dtype=torch.long,
            device=torch_device,
        )
        losses = torch.nn.functional.ctc_loss(
            expanded,
            targets,
            input_lengths,
            target_lengths,
            blank=int(blank_id),
            reduction="none",
            zero_infinity=False,
        )
        output.append((-losses).detach().cpu().numpy().astype(np.float64))
    return np.concatenate(output)


def _ctc_log_probabilities_padded_cuda(
    log_probs: np.ndarray,
    targets: np.ndarray,
    target_lengths: np.ndarray,
    *,
    blank_id: int,
    batch_size: int,
    device: str,
) -> np.ndarray:
    """CUDA CTC for an already validated, padded target matrix."""

    import torch

    values = np.asarray(log_probs)
    target_matrix = np.asarray(targets, dtype=np.int64)
    lengths = np.asarray(target_lengths, dtype=np.int64)
    torch_device = torch.device(device)
    matrix = torch.from_numpy(
        np.ascontiguousarray(values, dtype=np.float32)
    ).to(torch_device)
    output: list[np.ndarray] = []
    for offset in range(0, len(lengths), batch_size):
        stop = min(offset + batch_size, len(lengths))
        count = stop - offset
        losses = torch.nn.functional.ctc_loss(
            matrix.unsqueeze(1).expand(-1, count, -1),
            torch.from_numpy(np.ascontiguousarray(target_matrix[offset:stop])).to(
                torch_device
            ),
            torch.full(
                (count,), values.shape[0], dtype=torch.long, device=torch_device
            ),
            torch.from_numpy(np.ascontiguousarray(lengths[offset:stop])).to(
                torch_device
            ),
            blank=int(blank_id),
            reduction="none",
            zero_infinity=False,
        )
        output.append((-losses).detach().cpu().numpy().astype(np.float64))
    return np.concatenate(output)


# Backwards-friendly spelling used by the historical experiment reports.
ctc_logprobs_batch = ctc_log_probabilities_batch


def uniform_ctc_log_path_count(frames: int, labels: Sequence[int]) -> float:
    """Log number of valid length-``frames`` CTC paths under uniform emissions."""

    if isinstance(frames, bool) or int(frames) != frames or frames < 0:
        raise ValueError("frames must be a non-negative integer")
    sequence = list(labels)
    repeats = sum(left == right for left, right in zip(sequence, sequence[1:]))
    minimum = len(sequence) + repeats
    if frames < minimum:
        return NEG_INF
    # Number of monotone state paths through the blank-interleaved CTC graph.
    upper = int(frames + len(sequence) - repeats)
    lower = int(2 * len(sequence))
    return float(
        math.lgamma(upper + 1)
        - math.lgamma(lower + 1)
        - math.lgamma(upper - lower + 1)
    )


def topology_corrected_log_probability(
    log_probability: float,
    frames: int,
    labels: Sequence[int],
) -> float:
    """Remove the uniform-emission CTC path-count advantage from a score."""

    count = uniform_ctc_log_path_count(frames, labels)
    if not np.isfinite(log_probability) or not np.isfinite(count):
        return NEG_INF
    return float(log_probability - count)


def _ctc_forward_backward(
    log_probs: np.ndarray,
    labels: Sequence[int],
    blank_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return exact log-semiring messages for one CTC linear graph.

    The one-token counterfactual factorization below shares canonical prefix
    and suffix messages across every position and replacement phone.
    """

    values = np.asarray(log_probs, dtype=np.float64)
    sequence = np.asarray(labels, dtype=np.int64)
    frames = int(values.shape[0])
    states = np.full(2 * len(sequence) + 1, int(blank_id), dtype=np.int64)
    states[1::2] = sequence
    alpha = np.full((frames, len(states)), NEG_INF, dtype=np.float64)
    beta = np.full_like(alpha, NEG_INF)
    if frames == 0:
        return states, alpha, beta

    alpha[0, 0] = values[0, blank_id]
    if len(sequence):
        alpha[0, 1] = values[0, sequence[0]]
    skip_destination = np.zeros(len(states), dtype=bool)
    if len(states) > 2:
        skip_destination[2:] = (
            (states[2:] != blank_id) & (states[2:] != states[:-2])
        )
    for frame in range(1, frames):
        previous = alpha[frame - 1]
        incoming = previous.copy()
        incoming[1:] = np.logaddexp(incoming[1:], previous[:-1])
        if len(states) > 2:
            eligible = np.flatnonzero(skip_destination[2:]) + 2
            incoming[eligible] = np.logaddexp(
                incoming[eligible], previous[eligible - 2]
            )
        alpha[frame] = incoming + values[frame, states]

    beta[-1, -1] = 0.0
    if len(sequence):
        beta[-1, -2] = 0.0
    skip_source = np.zeros(len(states), dtype=bool)
    if len(states) > 2:
        skip_source[:-2] = (
            (states[2:] != blank_id) & (states[2:] != states[:-2])
        )
    for frame in range(frames - 2, -1, -1):
        following = beta[frame + 1]
        outgoing = following + values[frame + 1, states]
        outgoing[:-1] = np.logaddexp(
            outgoing[:-1], following[1:] + values[frame + 1, states[1:]]
        )
        if len(states) > 2:
            eligible = np.flatnonzero(skip_source[:-2])
            outgoing[eligible] = np.logaddexp(
                outgoing[eligible],
                following[eligible + 2] + values[frame + 1, states[eligible + 2]],
            )
        beta[frame] = outgoing
    return states, alpha, beta


def _single_edit_ctc_log_probabilities(
    log_probs: np.ndarray,
    canonical: Sequence[int],
    phone_ids: Sequence[int],
    blank_id: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Score every one-phone replacement and deletion with shared messages.

    A replacement changes only one label-emitting state and the two adjacent
    CTC skip arcs. Propagating that state for all ``[position, phone]`` pairs
    at once sums exactly the same complete paths as independent CTC calls.
    """

    values = np.asarray(log_probs, dtype=np.float64)
    labels = np.asarray(canonical, dtype=np.int64)
    phones = np.asarray(phone_ids, dtype=np.int64)
    frames = int(values.shape[0])
    length = int(labels.size)
    if length == 0:
        canonical_lp = float(values[:, blank_id].sum()) if frames else 0.0
        return (
            canonical_lp,
            np.empty((0, len(phones)), dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )
    if frames == 0:
        return (
            NEG_INF,
            np.full((length, len(phones)), NEG_INF, dtype=np.float64),
            np.full(length, NEG_INF, dtype=np.float64),
        )

    _states, alpha, beta = _ctc_forward_backward(values, labels, blank_id)
    canonical_lp = logsumexp((alpha[-1, -1], alpha[-1, -2]))

    # q[position, phone] is the forward mass which has reached the changed
    # label but has not yet left it. Exits join an invariant canonical suffix.
    q = np.full((length, len(phones)), NEG_INF, dtype=np.float64)
    q[0] = values[0, phones]
    substitutions = np.full_like(q, NEG_INF)
    position_index = np.arange(length, dtype=np.int64)
    blank_after = 2 * position_index + 2
    has_next = position_index + 1 < length
    for frame in range(1, frames):
        blank_exit = q + values[frame, blank_id] + beta[frame, blank_after, None]
        substitutions = np.logaddexp(substitutions, blank_exit)
        if np.any(has_next):
            rows = position_index[has_next]
            next_labels = labels[rows + 1]
            direct = (
                q[rows]
                + values[frame, next_labels, None]
                + beta[frame, 2 * rows + 3, None]
            )
            direct = np.where(
                phones[None, :] != next_labels[:, None], direct, NEG_INF
            )
            substitutions[rows] = np.logaddexp(substitutions[rows], direct)

        incoming = np.logaddexp(q, alpha[frame - 1, 2 * position_index, None])
        if length > 1:
            rows = position_index[1:]
            previous_labels = labels[rows - 1]
            previous_mass = alpha[frame - 1, 2 * rows - 1, None]
            previous_mass = np.where(
                phones[None, :] != previous_labels[:, None],
                previous_mass,
                NEG_INF,
            )
            incoming[rows] = np.logaddexp(incoming[rows], previous_mass)
        q = incoming + values[frame, phones][None, :]
    substitutions[-1] = np.logaddexp(substitutions[-1], q[-1])

    deletions = np.full(length, NEG_INF, dtype=np.float64)
    # Deleting the final label leaves the canonical prefix, whose two CTC final
    # states are already present in alpha.
    deletions[-1] = alpha[-1, 2 * (length - 1)]
    if length > 1:
        deletions[-1] = np.logaddexp(
            deletions[-1], alpha[-1, 2 * (length - 1) - 1]
        )
        rows = np.arange(length - 1, dtype=np.int64)
        next_labels = labels[rows + 1]
        next_states = 2 * rows + 3
        terms = (
            alpha[:-1, 2 * rows]
            + values[1:, next_labels]
            + beta[1:, next_states]
        )
        if frames > 1:
            deletions[rows] = np.logaddexp.reduce(terms, axis=0)
        if length > 2:
            nonfirst = rows[1:]
            allowed = labels[nonfirst - 1] != labels[nonfirst + 1]
            if np.any(allowed):
                selected = nonfirst[allowed]
                skip_terms = (
                    alpha[:-1, 2 * selected - 1]
                    + values[1:, labels[selected + 1]]
                    + beta[1:, 2 * selected + 3]
                )
                deletions[selected] = np.logaddexp(
                    deletions[selected], np.logaddexp.reduce(skip_terms, axis=0)
                )
        # A deletion at position zero may start directly in the first suffix.
        deletions[0] = np.logaddexp(
            deletions[0], values[0, labels[1]] + beta[0, 3]
        )
    return float(canonical_lp), substitutions, deletions


@dataclass(frozen=True)
class CounterfactualMatrix:
    """Canonical and every one-position substitution/deletion likelihood.

    Candidate id ``-1`` denotes ``<DEL>``.  For a 40-class vocabulary with one
    blank, each row contains 38 phone substitutions plus deletion: 39 entries.
    """

    canonical_log_probability: float
    canonical_log_path_count: float
    canonical_corrected_log_probability: float
    candidate_ids: np.ndarray
    log_probabilities: np.ndarray
    log_path_counts: np.ndarray
    corrected_log_probabilities: np.ndarray

    def __post_init__(self) -> None:
        shapes = {
            tuple(np.asarray(self.candidate_ids).shape),
            tuple(np.asarray(self.log_probabilities).shape),
            tuple(np.asarray(self.log_path_counts).shape),
            tuple(np.asarray(self.corrected_log_probabilities).shape),
        }
        if len(shapes) != 1 or len(next(iter(shapes))) != 2:
            raise ValueError("counterfactual arrays must share one two-dimensional shape")


def counterfactual_log_probability_matrix(
    log_probs: np.ndarray,
    canonical_ids: Sequence[int],
    *,
    blank_id: int = 0,
    phone_ids: Iterable[int] | None = None,
    include_deletion: bool = True,
    topology_correction: bool = True,
    batch_size: int = 256,
    device: str = "auto",
) -> CounterfactualMatrix:
    """Score all-phone substitution and deletion hypotheses at every position."""

    values = _validate_log_probs(log_probs, blank_id)
    canonical = _validate_sequence(canonical_ids, values.shape[1], blank_id)
    inventory = (
        [index for index in range(values.shape[1]) if index != blank_id]
        if phone_ids is None
        else [int(index) for index in phone_ids]
    )
    if len(inventory) != len(set(inventory)):
        raise ValueError("phone_ids must be unique")
    if blank_id in inventory or any(index < 0 or index >= values.shape[1] for index in inventory):
        raise ValueError("phone_ids must be non-blank vocabulary ids")
    if any(label not in inventory for label in canonical):
        raise ValueError("every canonical id must occur in phone_ids")

    width = len(inventory) - 1 + int(include_deletion)
    candidate_ids = np.empty((len(canonical), width), dtype=np.int64)
    for position, target in enumerate(canonical):
        alternatives = [index for index in inventory if index != target]
        if include_deletion:
            alternatives.append(DELETION_ID)
        candidate_ids[position] = alternatives
    selected_device = device
    if device == "auto":
        import torch

        selected_device = "cuda" if torch.cuda.is_available() else "cpu"
    if str(selected_device).startswith("cuda"):
        # CUDA's native CTC kernel is faster for the full 40-phone experiment.
        # Amortize launches over a larger safe batch than the conservative
        # training default; this changes no targets or reduction semantics.
        sequence_count = 1 + len(canonical) * width
        padded_targets = np.broadcast_to(
            np.asarray(canonical, dtype=np.int64),
            (sequence_count, len(canonical)),
        ).copy()
        target_lengths = np.full(sequence_count, len(canonical), dtype=np.int64)
        cursor = 1
        for position, alternatives in enumerate(candidate_ids):
            substitutions = alternatives[alternatives != DELETION_ID]
            rows = np.arange(cursor, cursor + len(substitutions), dtype=np.int64)
            padded_targets[rows, position] = substitutions
            cursor += len(substitutions)
            if include_deletion:
                if position + 1 < len(canonical):
                    padded_targets[cursor, position : len(canonical) - 1] = canonical[
                        position + 1 :
                    ]
                target_lengths[cursor] -= 1
                cursor += 1
        cuda_batch_size = max(batch_size, min(sequence_count, 1024))
        all_log_probabilities = _ctc_log_probabilities_padded_cuda(
            values,
            padded_targets,
            target_lengths,
            blank_id=blank_id,
            batch_size=cuda_batch_size,
            device=str(selected_device),
        )
        canonical_lp = float(all_log_probabilities[0])
        matrix = all_log_probabilities[1:].reshape(len(canonical), width)
    else:
        # On CPU, sharing canonical prefix/suffix messages avoids evaluating a
        # complete trellis for each candidate. Arithmetic remains float64.
        canonical_lp, substitutions, deletions = _single_edit_ctc_log_probabilities(
            values, canonical, inventory, blank_id
        )
        matrix = np.empty((len(canonical), width), dtype=np.float64)
        inventory_array = np.asarray(inventory, dtype=np.int64)
        for position, target in enumerate(canonical):
            keep = inventory_array != target
            substitution_count = int(np.sum(keep))
            matrix[position, :substitution_count] = substitutions[position, keep]
            if include_deletion:
                matrix[position, -1] = deletions[position]
    canonical_count = uniform_ctc_log_path_count(values.shape[0], canonical)
    counts = np.empty_like(matrix)
    canonical_repeats = sum(
        left == right for left, right in zip(canonical, canonical[1:])
    )
    count_cache: dict[tuple[int, int], float] = {
        (len(canonical), canonical_repeats): canonical_count
    }

    def count_for(length: int, repeats: int) -> float:
        key = (int(length), int(repeats))
        if key not in count_cache:
            minimum = key[0] + key[1]
            if values.shape[0] < minimum:
                count_cache[key] = NEG_INF
            else:
                upper = values.shape[0] + key[0] - key[1]
                lower = 2 * key[0]
                count_cache[key] = float(
                    math.lgamma(upper + 1)
                    - math.lgamma(lower + 1)
                    - math.lgamma(upper - lower + 1)
                )
        return count_cache[key]

    for position, alternatives in enumerate(candidate_ids):
        target = canonical[position]
        removed = int(position > 0 and canonical[position - 1] == target) + int(
            position + 1 < len(canonical) and target == canonical[position + 1]
        )
        for column, alternative in enumerate(alternatives):
            if alternative == DELETION_ID:
                joined = int(
                    position > 0
                    and position + 1 < len(canonical)
                    and canonical[position - 1] == canonical[position + 1]
                )
                repeats = canonical_repeats - removed + joined
                counts[position, column] = count_for(len(canonical) - 1, repeats)
            else:
                added = int(
                    position > 0 and canonical[position - 1] == int(alternative)
                ) + int(
                    position + 1 < len(canonical)
                    and int(alternative) == canonical[position + 1]
                )
                counts[position, column] = count_for(
                    len(canonical), canonical_repeats - removed + added
                )
    if topology_correction:
        canonical_corrected = topology_corrected_log_probability(
            canonical_lp, values.shape[0], canonical
        )
        corrected = np.full_like(matrix, NEG_INF)
        finite = np.isfinite(matrix) & np.isfinite(counts)
        corrected[finite] = matrix[finite] - counts[finite]
    else:
        canonical_corrected = canonical_lp
        corrected = matrix.copy()
    return CounterfactualMatrix(
        canonical_lp,
        canonical_count,
        canonical_corrected,
        candidate_ids,
        matrix,
        counts,
        corrected,
    )


# Singular spelling for callers that read the operation as one matrix.
counterfactual_log_probabilities = counterfactual_log_probability_matrix
