"""Label-free phone selection before candidate-graph construction.

These policies remove graph branches; they do not modify acoustic probabilities
or apply the historical CF prior correction. Selection uses the whole recording,
so no reference-to-frame alignment or full-graph inference is required. The
retained acoustic mass is a frame-posterior statistic, not a guarantee about
retained CTC path probability or pronunciation-error recall.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from time import perf_counter

import numpy as np

from .parikh2025 import restricted_phone_graph
from .phonology import ARPABET_39, PHONE_TO_CTC_ID


@dataclass(frozen=True)
class CandidateSelection:
    substitution_phone_ids: tuple[tuple[int, ...], ...]
    insertion_phone_ids: tuple[tuple[int, ...], ...]
    active_phone_ids: tuple[int, ...]
    retained_acoustic_mass: float
    selection_sources: tuple[str, ...]
    selection_seconds: float

    @property
    def substitution_counts(self) -> tuple[int, ...]:
        return tuple(len(phones) for phones in self.substitution_phone_ids)

    @property
    def insertion_counts(self) -> tuple[int, ...]:
        return tuple(len(phones) for phones in self.insertion_phone_ids)


@lru_cache(maxsize=1)
def _phonological_neighbors() -> dict[int, frozenset[int]]:
    # This is the existing directed PP-AF relation, not a patient-derived map.
    return {
        PHONE_TO_CTC_ID[phone]: frozenset(PHONE_TO_CTC_ID[target] for target in targets)
        for phone, targets in restricted_phone_graph().items()
    }


def select_candidates(
    log_probs: np.ndarray,
    acceptable_phone_ids: Sequence[Sequence[int]],
    *,
    policy: str = "frame_topk",
    top_k: int = 2,
    posterior_threshold: float = 0.01,
    mass_coverage: float = 0.99,
    add_phonological_neighbors: bool = False,
    learned_neighbors: Sequence[Sequence[int]] | None = None,
) -> CandidateSelection:
    """Select SUB phones at each anchor and INS phones in its ``n+1`` gaps.

    Input IDs use the frozen model inventory: 0 is CTC blank, 1..39 are phones.
    ``frame_topk`` unions the top-k nonblank phones from each frame whose raw
    posterior is at least ``posterior_threshold``. ``mass_cover`` ranks phones
    by posterior mass summed over frames and retains the smallest prefix that
    covers ``mass_coverage`` of the nonblank mass. It has no hard size cap.

    Optional fixed phonological neighbors of every acceptable phone are added
    to each anchor's SUB set. This PP-AF relation is generic phonology, not
    empirical dysarthria confusion. ``learned_neighbors`` optionally supplies
    40 directed neighbor rows indexed by CTC ID; the caller must estimate them
    using training labels only (and cross-fit training-patient features).
    This function never reads labels or changes the supplied relation.
    Acceptable phones themselves are excluded from
    SUB, because the decoder retains them as OK arcs. Every gap uses the global
    acoustic set as INS candidates. Blank and the decoder's epsilon deletion
    arcs are distinct and never appear in either returned phone list.
    """
    started = perf_counter()
    values = np.asarray(log_probs, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(ARPABET_39) + 1 or not len(values):
        raise ValueError("log_probs must have nonempty shape [frames, 40]")
    if np.isnan(values).any() or np.isposinf(values).any():
        raise ValueError("log_probs may contain -inf but not NaN or +inf")
    if policy not in {"frame_topk", "mass_cover"}:
        raise ValueError("policy must be frame_topk or mass_cover")
    if not 1 <= top_k <= len(ARPABET_39) or not 0 <= posterior_threshold <= 1:
        raise ValueError("top_k must be in [1, 39] and posterior_threshold in [0, 1]")
    if not 0 < mass_coverage <= 1:
        raise ValueError("mass_coverage must be in (0, 1]")
    acceptable = tuple(frozenset(int(phone) for phone in row) for row in acceptable_phone_ids)
    if any(not row or min(row) < 1 or max(row) > len(ARPABET_39) for row in acceptable):
        raise ValueError("each acceptable set must contain phone IDs in [1, 39]")
    if learned_neighbors is not None:
        if len(learned_neighbors) != 40:
            raise ValueError("learned_neighbors must have 40 rows indexed by CTC ID")
        if any(phone < 1 or phone > 39 for row in learned_neighbors for phone in row):
            raise ValueError("learned neighbors must contain phone IDs in [1, 39]")

    posteriors = np.exp(values[:, 1:])
    phone_mass = posteriors.sum(axis=0)
    nonblank_mass = float(phone_mass.sum())
    if nonblank_mass == 0:
        selected = np.empty(0, dtype=np.int64)
    elif policy == "frame_topk":
        ranked = np.argsort(-posteriors, axis=1, kind="stable")[:, :top_k]
        selected = np.unique(ranked[np.take_along_axis(posteriors, ranked, axis=1) >= posterior_threshold])
    else:
        ranked = np.argsort(-phone_mass, kind="stable")
        prefix_length = min(
            int(np.searchsorted(np.cumsum(phone_mass[ranked]), mass_coverage * nonblank_mass)) + 1,
            len(ranked),
        )
        selected = np.sort(ranked[:prefix_length])

    active = frozenset(int(phone) + 1 for phone in selected)
    neighbors = _phonological_neighbors() if add_phonological_neighbors else {}
    substitutions = []
    for accepted in acceptable:
        candidates = set(active)
        for phone in accepted:
            candidates.update(neighbors.get(phone, ()))
            if learned_neighbors is not None:
                candidates.update(learned_neighbors[phone])
        substitutions.append(tuple(sorted(candidates - accepted)))
    active_ids = tuple(sorted(active))
    # With no nonblank mass there is no acoustic phone mass to discard.
    retained = float(phone_mass[selected].sum() / nonblank_mass) if nonblank_mass else 1.0
    sources = [f"acoustic_{policy}"]
    if add_phonological_neighbors:
        sources.append("ppaf_generic_phonology")
    if learned_neighbors is not None:
        sources.append("training_confusions")
    return CandidateSelection(
        substitution_phone_ids=tuple(substitutions),
        insertion_phone_ids=(active_ids,) * (len(acceptable) + 1),
        active_phone_ids=active_ids,
        retained_acoustic_mass=retained,
        selection_sources=tuple(sources),
        selection_seconds=perf_counter() - started,
    )
