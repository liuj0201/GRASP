"""Exact CTC-SF SD scoring migrated into the standalone DA-CF-GoP package.

This is a self-contained implementation of the audited code9 recurrence.  It
implements the literal substitution-or-deletion (``SD``) wildcard language,
including repeated neighbours, in stable log space.  The public scorer accepts
only a blank-inclusive CTC timeline and a canonical phone sequence; external
phone boundaries cannot be supplied.

Score direction is ``higher_is_better``.  Both the raw score and SD-Norm score
are log posteriors and are therefore non-positive (up to floating-point error).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite, log1p
from typing import Iterable, Sequence

import numpy as np


NEG_INF = float("-inf")
_NON_WILDCARD = 0
_WILDCARD = 1


def _logsumexp(values: Iterable[float] | np.ndarray) -> float:
    array = np.asarray(
        list(values) if not isinstance(values, np.ndarray) else values,
        dtype=np.float64,
    ).ravel()
    if array.size == 0:
        return NEG_INF
    maximum = float(np.max(array))
    if maximum == NEG_INF:
        return NEG_INF
    return float(maximum + np.log(np.exp(array - maximum).sum()))


def _logsubexp_array(total: np.ndarray, part: np.ndarray) -> np.ndarray:
    """Vectorized ``log(exp(total)-exp(part))`` for subset measures."""

    total_values = np.broadcast_to(total, part.shape)
    result = np.full(part.shape, NEG_INF, dtype=np.float64)
    empty = np.isneginf(part)
    result[empty] = total_values[empty]
    usable = np.isfinite(total_values) & np.isfinite(part) & ~empty
    delta = np.zeros(part.shape, dtype=np.float64)
    delta[usable] = part[usable] - total_values[usable]
    if np.any(delta[usable] > 1.0e-9):
        raise ArithmeticError("subset log-mass exceeds total log-mass")
    strict = usable & (delta < -1.0e-14)
    result[strict] = total_values[strict] + np.log1p(-np.exp(delta[strict]))
    return result


def log_probs_from_logits(
    logits: np.ndarray | Sequence[Sequence[float]],
) -> np.ndarray:
    """Return row-normalized log probabilities in float64."""

    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] < 2:
        raise ValueError("logits must have shape [frames, classes]")
    if not np.all(np.isfinite(values)):
        raise ValueError("logits must be finite")
    shifted = values - values.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def _validate_log_probs(
    log_probs: np.ndarray | Sequence[Sequence[float]],
    *,
    check_normalized: bool = True,
) -> np.ndarray:
    values = np.asarray(log_probs, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] < 2:
        raise ValueError("log_probs must have shape [frames, classes]")
    if np.any(np.isnan(values)) or np.any(np.isposinf(values)):
        raise ValueError("log_probs may contain -inf but not NaN or +inf")
    if check_normalized:
        maxima = np.max(values, axis=1, keepdims=True)
        norms = maxima[:, 0] + np.log(np.exp(values - maxima).sum(axis=1))
        if not np.allclose(norms, 0.0, atol=1.0e-7, rtol=0.0):
            raise ValueError("rows are not normalized log probabilities")
    return values


def _validate_target(
    target: Sequence[int], num_classes: int, blank_id: int
) -> tuple[int, ...]:
    if not 0 <= int(blank_id) < num_classes:
        raise ValueError("blank_id is outside the model vocabulary")
    labels = tuple(int(label) for label in target)
    for label in labels:
        if not 0 <= label < num_classes:
            raise ValueError(f"target label {label} is outside the vocabulary")
        if label == blank_id:
            raise ValueError("canonical sequence may not contain CTC blank")
    return labels


def _validate_phone_ids(
    phone_ids: Sequence[int] | None,
    num_classes: int,
    blank_id: int,
    canonical: Sequence[int],
) -> tuple[int, ...]:
    phones = (
        tuple(index for index in range(num_classes) if index != blank_id)
        if phone_ids is None
        else tuple(int(index) for index in phone_ids)
    )
    if not phones or len(phones) != len(set(phones)):
        raise ValueError("phone_ids must be nonempty and unique")
    if any(phone == blank_id or not 0 <= phone < num_classes for phone in phones):
        raise ValueError("phone_ids must be valid nonblank model outputs")
    missing = sorted(set(canonical) - set(phones))
    if missing:
        raise ValueError(f"canonical labels are missing from phone_ids: {missing}")
    return phones


def ctc_log_probability(
    log_probs: np.ndarray | Sequence[Sequence[float]],
    target: Sequence[int],
    *,
    blank_id: int = 0,
    check_normalized: bool = True,
) -> float:
    """Exact standard CTC sequence probability in log space."""

    values = _validate_log_probs(log_probs, check_normalized=check_normalized)
    labels = _validate_target(target, values.shape[1], blank_id)
    if not labels:
        return float(values[:, blank_id].sum())
    states = np.full(2 * len(labels) + 1, blank_id, dtype=np.int64)
    states[1::2] = labels
    previous = np.full(states.size, NEG_INF, dtype=np.float64)
    previous[0] = values[0, blank_id]
    previous[1] = values[0, labels[0]]
    skip = np.zeros(states.size, dtype=bool)
    skip[2:] = (states[2:] != blank_id) & (states[2:] != states[:-2])
    for frame in range(1, values.shape[0]):
        incoming = previous.copy()
        incoming[1:] = np.logaddexp(incoming[1:], previous[:-1])
        incoming[skip] = np.logaddexp(incoming[skip], previous[np.flatnonzero(skip) - 2])
        previous = incoming + values[frame, states]
    return _logsumexp((previous[-1], previous[-2]))


def ctc_viterbi_state_path(
    log_probs: np.ndarray | Sequence[Sequence[float]],
    target: Sequence[int],
    *,
    blank_id: int = 0,
    check_normalized: bool = True,
) -> tuple[np.ndarray, float]:
    """Best state path through the canonical blank-interleaved CTC trellis."""

    values = _validate_log_probs(log_probs, check_normalized=check_normalized)
    labels = _validate_target(target, values.shape[1], blank_id)
    if not labels:
        raise ValueError("Viterbi alignment requires a nonempty target")
    states = np.full(2 * len(labels) + 1, blank_id, dtype=np.int64)
    states[1::2] = labels
    scores = np.full((values.shape[0], states.size), NEG_INF, dtype=np.float64)
    back = np.full((values.shape[0], states.size), -1, dtype=np.int64)
    scores[0, 0] = values[0, blank_id]
    scores[0, 1] = values[0, labels[0]]
    state_index = np.arange(states.size, dtype=np.int64)
    skip = np.zeros(states.size, dtype=bool)
    skip[2:] = (states[2:] != blank_id) & (states[2:] != states[:-2])
    for frame in range(1, values.shape[0]):
        candidates = np.full((3, states.size), NEG_INF, dtype=np.float64)
        candidates[0] = scores[frame - 1]
        candidates[1, 1:] = scores[frame - 1, :-1]
        candidates[2, skip] = scores[frame - 1, np.flatnonzero(skip) - 2]
        winner = np.argmax(candidates, axis=0)
        back[frame] = state_index - winner
        scores[frame] = candidates[winner, state_index] + values[frame, states]
    final_states = (states.size - 1, states.size - 2)
    final_values = [scores[-1, state] for state in final_states]
    state = final_states[int(np.argmax(final_values))]
    best_score = float(scores[-1, state])
    if best_score == NEG_INF:
        raise ValueError("canonical sequence has no valid CTC path")
    path = np.empty(values.shape[0], dtype=np.int64)
    path[-1] = state
    for frame in range(values.shape[0] - 1, 0, -1):
        state = int(back[frame, state])
        if state < 0:
            raise RuntimeError("invalid Viterbi backpointer")
        path[frame - 1] = state
    return path, best_score


def self_alignment_scores(
    log_probs: np.ndarray | Sequence[Sequence[float]],
    canonical: Sequence[int],
    *,
    blank_id: int = 0,
    check_normalized: bool = True,
    state_path: Sequence[int] | np.ndarray | None = None,
) -> list[float]:
    """CTC-SA: mean log target posterior on Viterbi label-state frames."""

    values = _validate_log_probs(log_probs, check_normalized=check_normalized)
    labels = _validate_target(canonical, values.shape[1], blank_id)
    if state_path is None:
        resolved_path, _ = ctc_viterbi_state_path(
            values, labels, blank_id=blank_id, check_normalized=False
        )
    else:
        resolved_path = np.asarray(state_path, dtype=np.int64)
        if (
            resolved_path.shape != (values.shape[0],)
            or np.any(resolved_path < 0)
            or np.any(resolved_path >= 2 * len(labels) + 1)
        ):
            raise ValueError("state_path is incompatible with this utterance")
    output: list[float] = []
    for position, label in enumerate(labels):
        frames = np.flatnonzero(resolved_path == 2 * position + 1)
        if frames.size == 0:
            raise RuntimeError("valid CTC Viterbi path skipped a label state")
        output.append(float(values[frames, label].mean()))
    return output


State = tuple[str, int]


class _SDLanguage:
    """Unambiguous NFA for ``prefix + optional-one-phone + suffix``."""

    def __init__(
        self,
        prefix: Sequence[int],
        suffix: Sequence[int],
        phone_ids: Sequence[int],
    ) -> None:
        self.prefix = tuple(prefix)
        self.suffix = tuple(suffix)
        self.phone_ids = frozenset(phone_ids)
        self.start: State = ("p", 0) if self.prefix else ("w", 0)

    def transitions(self, state: State, token: int) -> tuple[tuple[State, int], ...]:
        if token not in self.phone_ids:
            return ()
        kind, index = state
        if kind == "p":
            if token != self.prefix[index]:
                return ()
            destination = (
                ("w", 0) if index + 1 == len(self.prefix) else ("p", index + 1)
            )
            return ((destination, _NON_WILDCARD),)
        if kind == "s":
            if index >= len(self.suffix) or token != self.suffix[index]:
                return ()
            return ((("s", index + 1), _NON_WILDCARD),)
        if kind != "w":
            raise RuntimeError(f"invalid wildcard automaton state {state}")

        # Substitution branch: consume exactly one arbitrary phone.
        transitions: list[tuple[State, int]] = [
            (("s", 0), _WILDCARD)
        ]
        # Deletion branch: consume the first suffix phone directly.  This makes
        # the automaton unambiguous even when the substitute equals a neighbour.
        if self.suffix and token == self.suffix[0]:
            transitions.append((("s", 1), _NON_WILDCARD))
        return tuple(transitions)

    def accepts(self, state: State) -> bool:
        kind, index = state
        return (kind == "s" and index == len(self.suffix)) or (
            kind == "w" and not self.suffix
        )


@dataclass(frozen=True)
class _WildcardForwardResult:
    log_probability: float
    paper_forward_occurrence: float
    posterior_occurrence: float | None


def _wildcard_forward_many_reference(
    log_probs: np.ndarray,
    languages: Sequence[_SDLanguage],
    *,
    blank_id: int,
    phone_ids: Sequence[int],
    max_work: int | None,
    compute_posterior_occurrence: bool,
) -> list[_WildcardForwardResult]:
    """Vectorized exact wildcard DP over all requested canonical positions."""

    if not languages:
        return []
    state_lists: list[list[State]] = []
    state_maps: list[dict[State, int]] = []
    for language in languages:
        states = (
            [("p", index) for index in range(len(language.prefix))]
            + [("w", 0)]
            + [("s", index) for index in range(len(language.suffix) + 1)]
        )
        state_lists.append(states)
        state_maps.append({state: index for index, state in enumerate(states)})
    phone_array = np.asarray(tuple(phone_ids), dtype=np.int64)
    estimated_work = (
        log_probs.shape[0]
        * sum(len(states) for states in state_lists)
        * (len(phone_ids) + 1)
        * 8
    )
    if max_work is not None and estimated_work > max_work:
        raise RuntimeError(
            f"exact batched SD graph estimated work {estimated_work:,} exceeds "
            f"max_work={max_work:,}; increase the explicit budget rather than "
            "silently truncating the wildcard language"
        )

    edge_batches: list[int] = []
    edge_sources: list[int] = []
    edge_phone_positions: list[int] = []
    edge_tokens: list[int] = []
    edge_destinations: list[int] = []
    edge_roles: list[int] = []
    for batch_index, (language, states, state_map) in enumerate(
        zip(languages, state_lists, state_maps)
    ):
        for source, state in enumerate(states):
            for phone_position, phone in enumerate(phone_array.tolist()):
                for destination, role in language.transitions(state, phone):
                    edge_batches.append(batch_index)
                    edge_sources.append(source)
                    edge_phone_positions.append(phone_position)
                    edge_tokens.append(phone)
                    edge_destinations.append(state_map[destination])
                    edge_roles.append(role)
    edge_batch = np.asarray(edge_batches, dtype=np.int64)
    edge_source = np.asarray(edge_sources, dtype=np.int64)
    edge_phone_position = np.asarray(edge_phone_positions, dtype=np.int64)
    edge_token = np.asarray(edge_tokens, dtype=np.int64)
    edge_destination = np.asarray(edge_destinations, dtype=np.int64)
    edge_role = np.asarray(edge_roles, dtype=np.int64)

    batch_size = len(languages)
    max_states = max(map(len, state_lists))
    num_classes = log_probs.shape[1]
    log_mass = np.full(
        (batch_size, max_states, num_classes, 2), NEG_INF, dtype=np.float64
    )
    log_weighted = (
        np.full_like(log_mass, NEG_INF) if compute_posterior_occurrence else None
    )
    for batch_index, (language, state_map) in enumerate(zip(languages, state_maps)):
        log_mass[batch_index, state_map[language.start], blank_id, _NON_WILDCARD] = 0.0
    paper_occurrence = np.zeros(batch_size, dtype=np.float64)

    for frame in range(log_probs.shape[0]):
        state_mass = np.logaddexp.reduce(
            np.logaddexp.reduce(log_mass, axis=3), axis=2
        )
        next_mass = np.full_like(log_mass, NEG_INF)
        next_mass[:, :, blank_id, _NON_WILDCARD] = (
            state_mass + log_probs[frame, blank_id]
        )
        next_weighted = (
            np.full_like(log_mass, NEG_INF) if compute_posterior_occurrence else None
        )
        if compute_posterior_occurrence:
            assert log_weighted is not None and next_weighted is not None
            state_weighted = np.logaddexp.reduce(
                np.logaddexp.reduce(log_weighted, axis=3), axis=2
            )
            next_weighted[:, :, blank_id, _NON_WILDCARD] = (
                state_weighted + log_probs[frame, blank_id]
            )

        # A repeated nonblank token does not advance the collapsed sequence.
        repeat_mass = (
            log_mass[:, :, phone_array, :]
            + log_probs[frame, phone_array][None, None, :, None]
        )
        next_mass[:, :, phone_array, :] = np.logaddexp(
            next_mass[:, :, phone_array, :], repeat_mass
        )
        if compute_posterior_occurrence:
            assert log_weighted is not None and next_weighted is not None
            repeat_weighted = (
                log_weighted[:, :, phone_array, :]
                + log_probs[frame, phone_array][None, None, :, None]
            )
            repeat_weighted[:, :, :, _WILDCARD] = np.logaddexp(
                repeat_weighted[:, :, :, _WILDCARD],
                repeat_mass[:, :, :, _WILDCARD],
            )
            next_weighted[:, :, phone_array, :] = np.logaddexp(
                next_weighted[:, :, phone_array, :], repeat_weighted
            )

        # A new collapsed phone q can follow every previous raw token except q.
        same_mass = np.logaddexp.reduce(log_mass[:, :, phone_array, :], axis=3)
        different_mass = _logsubexp_array(state_mass[:, :, None], same_mass)
        emitted_mass = (
            different_mass[edge_batch, edge_source, edge_phone_position]
            + log_probs[frame, edge_token]
        )
        np.logaddexp.at(
            next_mass,
            (edge_batch, edge_destination, edge_token, edge_role),
            emitted_mass,
        )
        if compute_posterior_occurrence:
            assert log_weighted is not None and next_weighted is not None
            same_weighted = np.logaddexp.reduce(
                log_weighted[:, :, phone_array, :], axis=3
            )
            different_weighted = _logsubexp_array(
                state_weighted[:, :, None], same_weighted
            )
            emitted_weighted = (
                different_weighted[edge_batch, edge_source, edge_phone_position]
                + log_probs[frame, edge_token]
            )
            wildcard_edges = edge_role == _WILDCARD
            emitted_weighted[wildcard_edges] = np.logaddexp(
                emitted_weighted[wildcard_edges], emitted_mass[wildcard_edges]
            )
            np.logaddexp.at(
                next_weighted,
                (edge_batch, edge_destination, edge_token, edge_role),
                emitted_weighted,
            )
        log_mass = next_mass
        log_weighted = next_weighted

        active_mass = np.logaddexp.reduce(
            np.logaddexp.reduce(np.logaddexp.reduce(log_mass, axis=3), axis=2),
            axis=1,
        )
        central_mass = np.logaddexp.reduce(
            np.logaddexp.reduce(log_mass[:, :, phone_array, _WILDCARD], axis=2),
            axis=1,
        )
        active = np.isfinite(active_mass) & np.isfinite(central_mass)
        paper_occurrence[active] += np.exp(
            central_mass[active] - active_mass[active]
        )

    output: list[_WildcardForwardResult] = []
    for batch_index, (language, states) in enumerate(zip(languages, state_lists)):
        accepted = [
            index for index, state in enumerate(states) if language.accepts(state)
        ]
        final_mass = _logsumexp(log_mass[batch_index, accepted])
        posterior: float | None = None
        if compute_posterior_occurrence:
            assert log_weighted is not None
            weighted = _logsumexp(log_weighted[batch_index, accepted])
            posterior = (
                0.0
                if final_mass == NEG_INF or weighted == NEG_INF
                else float(exp(weighted - final_mass))
            )
        output.append(
            _WildcardForwardResult(
                final_mass, float(paper_occurrence[batch_index]), posterior
            )
        )
    return output


def _wildcard_forward_many(
    log_probs: np.ndarray,
    languages: Sequence[_SDLanguage],
    *,
    blank_id: int,
    phone_ids: Sequence[int],
    max_work: int | None,
    compute_posterior_occurrence: bool,
) -> list[_WildcardForwardResult]:
    """Exact compact SD-language forward pass over all positions.

    Only the wildcard label can be any phone.  Prefix and suffix states can
    therefore use ordinary scalar CTC states instead of carrying the historical
    ``[state, last-phone, role]`` tensor.  A small phone vector is retained for
    the wildcard itself until a blank or suffix label makes the branches safe
    to merge.  This recognizes the identical unambiguous language while
    reducing the dominant state tensor by roughly the phone inventory size.
    """

    if not languages:
        return []
    phone_array = np.asarray(tuple(phone_ids), dtype=np.int64)
    canonical_lengths = [
        len(language.prefix) + len(language.suffix) + 1 for language in languages
    ]
    max_scalar_states = 2 * max(canonical_lengths)
    estimated_work = (
        log_probs.shape[0]
        * sum(len(language.prefix) + len(language.suffix) + 2 for language in languages)
        * (len(phone_ids) + 1)
        * 8
    )
    if max_work is not None and estimated_work > max_work:
        raise RuntimeError(
            f"exact batched SD graph estimated work {estimated_work:,} exceeds "
            f"max_work={max_work:,}; increase the explicit budget rather than "
            "silently truncating the wildcard language"
        )

    batch_size = len(languages)
    state_tokens = np.full((batch_size, max_scalar_states), -1, dtype=np.int64)
    # Four scalar predecessors suffice: stay, ordinary predecessor, CTC skip,
    # and the additional deletion edge into the first suffix label.
    incoming_local = np.full(
        (batch_size, max_scalar_states, 4), -1, dtype=np.int64
    )
    prefix_blank = np.empty(batch_size, dtype=np.int64)
    prefix_label = np.full(batch_size, -1, dtype=np.int64)
    blank_after_wildcard = np.empty(batch_size, dtype=np.int64)
    first_suffix = np.full(batch_size, -1, dtype=np.int64)
    previous_phone = np.full(batch_size, -1, dtype=np.int64)
    next_phone = np.full(batch_size, -1, dtype=np.int64)
    scalar_counts = np.empty(batch_size, dtype=np.int64)

    def add_incoming(batch: int, destination: int, source: int) -> None:
        empty = np.flatnonzero(incoming_local[batch, destination] < 0)
        if not empty.size:
            raise RuntimeError("compact SD scalar state has too many predecessors")
        incoming_local[batch, destination, int(empty[0])] = source

    for batch, language in enumerate(languages):
        prefix = language.prefix
        suffix = language.suffix
        position = len(prefix)
        scalar_count = 2 * (len(prefix) + len(suffix) + 1)
        scalar_counts[batch] = scalar_count
        prefix_blank[batch] = 2 * position
        blank_after_wildcard[batch] = 2 * position + 1
        if prefix:
            prefix_label[batch] = 2 * position - 1
            previous_phone[batch] = prefix[-1]

        # Prefix: ordinary blank-interleaved CTC states through the blank just
        # before the wildcard.
        for state in range(2 * position + 1):
            token = blank_id if state % 2 == 0 else prefix[(state - 1) // 2]
            state_tokens[batch, state] = token
            add_incoming(batch, state, state)
            if state >= 1:
                add_incoming(batch, state, state - 1)
            if (
                state >= 2
                and token != blank_id
                and token != state_tokens[batch, state - 2]
            ):
                add_incoming(batch, state, state - 2)

        after = int(blank_after_wildcard[batch])
        state_tokens[batch, after] = blank_id
        add_incoming(batch, after, after)

        # Suffix states are shifted left by one relative to the canonical CTC
        # graph because the arbitrary wildcard is held in its own vector.
        for offset, token in enumerate(suffix):
            label_state = 2 * position + 2 + 2 * offset
            blank_state = label_state + 1
            state_tokens[batch, label_state] = token
            state_tokens[batch, blank_state] = blank_id
            add_incoming(batch, label_state, label_state)
            add_incoming(batch, blank_state, blank_state)
            add_incoming(batch, blank_state, label_state)
            if offset == 0:
                first_suffix[batch] = label_state
                next_phone[batch] = token
                add_incoming(batch, label_state, after)
                # Deletion consumes the first suffix label directly from the
                # pre-wildcard blank, or by a CTC skip from a different label.
                add_incoming(batch, label_state, int(prefix_blank[batch]))
                if prefix and prefix[-1] != token:
                    add_incoming(batch, label_state, int(prefix_label[batch]))
            else:
                add_incoming(batch, label_state, label_state - 1)
                if token != suffix[offset - 1]:
                    add_incoming(batch, label_state, label_state - 2)

    # Turn local scalar source indices into one gather over the flattened batch.
    sentinel = batch_size * max_scalar_states
    offsets = (
        np.arange(batch_size, dtype=np.int64)[:, None, None] * max_scalar_states
    )
    incoming = np.where(incoming_local >= 0, incoming_local + offsets, sentinel)
    scalar_mass = np.full(
        (batch_size, max_scalar_states), NEG_INF, dtype=np.float64
    )
    wildcard_mass = np.full(
        (batch_size, len(phone_array)), NEG_INF, dtype=np.float64
    )
    scalar_weighted = (
        np.full_like(scalar_mass, NEG_INF)
        if compute_posterior_occurrence
        else None
    )
    wildcard_weighted = (
        np.full_like(wildcard_mass, NEG_INF)
        if compute_posterior_occurrence
        else None
    )

    scalar_mass[:, 0] = log_probs[0, blank_id]
    for batch, language in enumerate(languages):
        if language.prefix:
            scalar_mass[batch, 1] = log_probs[0, language.prefix[0]]
        else:
            wildcard_mass[batch] = log_probs[0, phone_array]
            if compute_posterior_occurrence:
                assert wildcard_weighted is not None
                wildcard_weighted[batch] = wildcard_mass[batch]
            if language.suffix:
                destination = int(first_suffix[batch])
                scalar_mass[batch, destination] = log_probs[0, language.suffix[0]]

    paper_occurrence = np.zeros(batch_size, dtype=np.float64)

    def accumulate_occurrence() -> None:
        scalar_total = np.logaddexp.reduce(scalar_mass, axis=1)
        central = np.logaddexp.reduce(wildcard_mass, axis=1)
        active = np.logaddexp(scalar_total, central)
        usable = np.isfinite(active) & np.isfinite(central)
        paper_occurrence[usable] += np.exp(central[usable] - active[usable])

    accumulate_occurrence()
    valid_state = state_tokens >= 0
    safe_tokens = np.where(valid_state, state_tokens, blank_id)
    has_prefix = prefix_label >= 0
    has_suffix = first_suffix >= 0
    phone_emission = log_probs[:, phone_array]

    for frame in range(1, log_probs.shape[0]):
        flattened = np.concatenate((scalar_mass.ravel(), np.asarray([NEG_INF])))
        next_scalar = np.logaddexp.reduce(flattened[incoming], axis=2)
        next_scalar += log_probs[frame, safe_tokens]
        next_scalar[~valid_state] = NEG_INF

        # Enter or repeat the wildcard. A direct CTC skip from the previous
        # label is forbidden when it equals the candidate phone.
        wildcard_incoming = np.logaddexp(
            wildcard_mass, scalar_mass[np.arange(batch_size), prefix_blank, None]
        )
        if np.any(has_prefix):
            rows = np.flatnonzero(has_prefix)
            previous = scalar_mass[rows, prefix_label[rows], None]
            previous = np.where(
                phone_array[None, :] != previous_phone[rows, None],
                previous,
                NEG_INF,
            )
            wildcard_incoming[rows] = np.logaddexp(
                wildcard_incoming[rows], previous
            )
        next_wildcard = wildcard_incoming + phone_emission[frame]

        wildcard_total = np.logaddexp.reduce(wildcard_mass, axis=1)
        after = blank_after_wildcard
        next_scalar[np.arange(batch_size), after] = np.logaddexp(
            next_scalar[np.arange(batch_size), after],
            wildcard_total + log_probs[frame, blank_id],
        )
        if np.any(has_suffix):
            rows = np.flatnonzero(has_suffix)
            allowed = phone_array[None, :] != next_phone[rows, None]
            direct = np.logaddexp.reduce(
                np.where(allowed, wildcard_mass[rows], NEG_INF), axis=1
            )
            destinations = first_suffix[rows]
            next_scalar[rows, destinations] = np.logaddexp(
                next_scalar[rows, destinations],
                direct + log_probs[frame, next_phone[rows]],
            )

        next_scalar_weighted: np.ndarray | None = None
        next_wildcard_weighted: np.ndarray | None = None
        if compute_posterior_occurrence:
            assert scalar_weighted is not None and wildcard_weighted is not None
            flattened_weighted = np.concatenate(
                (scalar_weighted.ravel(), np.asarray([NEG_INF]))
            )
            next_scalar_weighted = np.logaddexp.reduce(
                flattened_weighted[incoming], axis=2
            )
            next_scalar_weighted += log_probs[frame, safe_tokens]
            next_scalar_weighted[~valid_state] = NEG_INF

            # Every emission which enters or repeats W contributes one new
            # wildcard-frame occurrence to the expectation semiring.
            next_wildcard_weighted = np.logaddexp(
                wildcard_weighted, wildcard_mass
            )
            entry = np.broadcast_to(
                scalar_mass[np.arange(batch_size), prefix_blank, None],
                wildcard_mass.shape,
            ).copy()
            if np.any(has_prefix):
                rows = np.flatnonzero(has_prefix)
                previous = scalar_mass[rows, prefix_label[rows], None]
                previous = np.where(
                    phone_array[None, :] != previous_phone[rows, None],
                    previous,
                    NEG_INF,
                )
                entry[rows] = np.logaddexp(entry[rows], previous)
            next_wildcard_weighted = np.logaddexp(
                next_wildcard_weighted, entry
            ) + phone_emission[frame]

            wildcard_weighted_total = np.logaddexp.reduce(
                wildcard_weighted, axis=1
            )
            next_scalar_weighted[np.arange(batch_size), after] = np.logaddexp(
                next_scalar_weighted[np.arange(batch_size), after],
                wildcard_weighted_total + log_probs[frame, blank_id],
            )
            if np.any(has_suffix):
                rows = np.flatnonzero(has_suffix)
                allowed = phone_array[None, :] != next_phone[rows, None]
                direct_weighted = np.logaddexp.reduce(
                    np.where(allowed, wildcard_weighted[rows], NEG_INF), axis=1
                )
                destinations = first_suffix[rows]
                next_scalar_weighted[rows, destinations] = np.logaddexp(
                    next_scalar_weighted[rows, destinations],
                    direct_weighted + log_probs[frame, next_phone[rows]],
                )

        scalar_mass = next_scalar
        wildcard_mass = next_wildcard
        scalar_weighted = next_scalar_weighted
        wildcard_weighted = next_wildcard_weighted
        accumulate_occurrence()

    output: list[_WildcardForwardResult] = []
    for batch, language in enumerate(languages):
        accepted_mass: list[float] = []
        accepted_weighted: list[float] = []
        count = int(scalar_counts[batch])
        if language.suffix:
            accepted_mass.extend((scalar_mass[batch, count - 2], scalar_mass[batch, count - 1]))
            if compute_posterior_occurrence:
                assert scalar_weighted is not None
                accepted_weighted.extend(
                    (scalar_weighted[batch, count - 2], scalar_weighted[batch, count - 1])
                )
        else:
            accepted_mass.extend(wildcard_mass[batch].tolist())
            accepted_mass.append(scalar_mass[batch, count - 1])
            accepted_mass.append(scalar_mass[batch, count - 2])
            if language.prefix:
                accepted_mass.append(scalar_mass[batch, count - 3])
            if compute_posterior_occurrence:
                assert scalar_weighted is not None and wildcard_weighted is not None
                accepted_weighted.extend(wildcard_weighted[batch].tolist())
                accepted_weighted.append(scalar_weighted[batch, count - 1])
                accepted_weighted.append(scalar_weighted[batch, count - 2])
                if language.prefix:
                    accepted_weighted.append(scalar_weighted[batch, count - 3])
        final_mass = _logsumexp(np.asarray(accepted_mass))
        posterior: float | None = None
        if compute_posterior_occurrence:
            weighted = _logsumexp(np.asarray(accepted_weighted))
            posterior = (
                0.0
                if final_mass == NEG_INF or weighted == NEG_INF
                else float(exp(weighted - final_mass))
            )
        output.append(
            _WildcardForwardResult(
                final_mass, float(paper_occurrence[batch]), posterior
            )
        )
    return output


@dataclass(frozen=True)
class SegmentationFreeScore:
    """One exact SD and SD-Norm phone score (higher is better)."""

    position: int
    phone_id: int
    numerator_log_probability: float
    denominator_log_probability: float
    score: float
    occurrence: float
    occurrence_floor: float
    normalized_score: float
    posterior_occurrence_diagnostic: float | None


def segmentation_free_scores(
    log_probs: np.ndarray | Sequence[Sequence[float]],
    canonical: Sequence[int],
    *,
    positions: Sequence[int] | None = None,
    blank_id: int = 0,
    phone_ids: Sequence[int] | None = None,
    occurrence_floor: float = 1.0,
    check_normalized: bool = True,
    max_work: int | None = None,
    numerator_log_probability: float | None = None,
    compute_posterior_diagnostic: bool = False,
) -> list[SegmentationFreeScore]:
    """Compute exact CTC-SF-SD scores for multiple target positions."""

    values = _validate_log_probs(log_probs, check_normalized=check_normalized)
    labels = _validate_target(canonical, values.shape[1], blank_id)
    resolved_positions = (
        tuple(range(len(labels)))
        if positions is None
        else tuple(int(position) for position in positions)
    )
    if not resolved_positions or len(resolved_positions) != len(set(resolved_positions)):
        raise ValueError("positions must be nonempty and unique")
    if any(position < 0 or position >= len(labels) for position in resolved_positions):
        raise IndexError("a requested position is outside the canonical sequence")
    phones = _validate_phone_ids(phone_ids, values.shape[1], blank_id, labels)
    if not isfinite(occurrence_floor) or occurrence_floor <= 0.0:
        raise ValueError("occurrence_floor must be finite and positive")
    numerator = (
        ctc_log_probability(values, labels, blank_id=blank_id, check_normalized=False)
        if numerator_log_probability is None
        else float(numerator_log_probability)
    )
    languages = [
        _SDLanguage(labels[:position], labels[position + 1 :], phones)
        for position in resolved_positions
    ]
    denominators = _wildcard_forward_many(
        values,
        languages,
        blank_id=blank_id,
        phone_ids=phones,
        max_work=max_work,
        compute_posterior_occurrence=compute_posterior_diagnostic,
    )
    output: list[SegmentationFreeScore] = []
    for position, denominator_result in zip(resolved_positions, denominators):
        denominator = denominator_result.log_probability
        if numerator == NEG_INF:
            score = NEG_INF
        elif denominator == NEG_INF:
            raise ArithmeticError(
                "SD denominator is empty while the canonical CTC path exists"
            )
        else:
            score = float(numerator - denominator)
            if score > 1.0e-8:
                raise ArithmeticError(
                    "CTC-SF log posterior is positive; SD failed to include canonical"
                )
            if 0.0 < score <= 1.0e-8:
                score = 0.0
        divisor = max(
            float(occurrence_floor), denominator_result.paper_forward_occurrence
        )
        output.append(
            SegmentationFreeScore(
                position=position,
                phone_id=labels[position],
                numerator_log_probability=numerator,
                denominator_log_probability=denominator,
                score=score,
                occurrence=denominator_result.paper_forward_occurrence,
                occurrence_floor=float(occurrence_floor),
                normalized_score=float(score / divisor),
                posterior_occurrence_diagnostic=denominator_result.posterior_occurrence,
            )
        )
    return output


def segmentation_free_score(
    log_probs: np.ndarray | Sequence[Sequence[float]],
    canonical: Sequence[int],
    position: int,
    **kwargs: object,
) -> SegmentationFreeScore:
    """Single-position convenience wrapper around the exact batched scorer."""

    kwargs = dict(kwargs)
    kwargs["positions"] = (int(position),)
    kwargs.setdefault("compute_posterior_diagnostic", True)
    return segmentation_free_scores(log_probs, canonical, **kwargs)[0]


__all__ = [
    "SegmentationFreeScore",
    "ctc_log_probability",
    "ctc_viterbi_state_path",
    "log_probs_from_logits",
    "segmentation_free_score",
    "segmentation_free_scores",
    "self_alignment_scores",
]
