"""Pre-declared same-backend baselines and the legacy hard-graph ablation.

All acoustic baselines consume the official blank-inclusive CTC timeline.  A
caller obtains the recalibrated variants by applying the fold recalibrator
before calling these functions; there is deliberately no second calibration
implementation here.

Score direction for every scalar returned by this module is
``higher_is_better`` (more compatible with the canonical pronunciation).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np

from .ctc_sf import (
    ctc_viterbi_state_path,
    log_probs_from_logits,
    segmentation_free_scores,
    self_alignment_scores,
)
from .priors import DELETION, DEFAULT_PHONE_INVENTORY


CONVENTIONAL_METHOD = "conventional_forced_alignment_gop"
CTC_SA_METHOD = "ctc_sf_sa"
CTC_SF_SD_METHOD = "ctc_sf_sd"
CTC_SF_SD_NORM_METHOD = "ctc_sf_sd_norm"
LEGACY_HARD_GRAPH_METHOD = "code9_legacy_hard_graph"
SCORE_DIRECTION = "higher_is_better"


@dataclass(frozen=True)
class ForcedAlignmentPhoneScore:
    """One conventional posterior GoP score on a canonical Viterbi segment.

    ``score`` is ``log(mean_t P(target | x_t))``, matching the conventional
    average-posterior baseline shipped with the CTC-SF reference code.  The
    log-ratio to the strongest nonblank competitor is retained as a diagnostic,
    not silently substituted for the registered score.
    """

    position: int
    phone_id: int
    start_frame: int
    end_frame: int
    frame_count: int
    score: float
    top_competitor_id: int
    top_competitor_score: float
    posterior_margin: float
    score_direction: str = SCORE_DIRECTION


def _logmeanexp(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        raise ValueError("cannot average an empty frame set")
    maximum = float(np.max(values))
    if maximum == -float("inf"):
        return maximum
    return float(maximum + np.log(np.exp(values - maximum).mean()))


def conventional_forced_alignment_gop(
    frame_outputs: np.ndarray | Sequence[Sequence[float]],
    canonical: Sequence[int],
    *,
    blank_id: int = 0,
    phone_ids: Sequence[int] | None = None,
    input_is_logits: bool = False,
    state_path: Sequence[int] | np.ndarray | None = None,
) -> list[ForcedAlignmentPhoneScore]:
    """Score phones after canonical CTC Viterbi forced alignment.

    The alignment is generated from the same CTC outputs being scored.  This is
    the intentionally alignment-dependent comparator; it never consumes PHN or
    TextGrid boundaries.  Blank frames are excluded from each phone segment and
    from the competing-phone set.
    """

    log_probs = (
        log_probs_from_logits(frame_outputs)
        if input_is_logits
        else np.asarray(frame_outputs, dtype=np.float64)
    )
    labels = tuple(int(value) for value in canonical)
    if not labels:
        raise ValueError("canonical must contain at least one phone")
    if state_path is None:
        path, _ = ctc_viterbi_state_path(log_probs, labels, blank_id=blank_id)
    else:
        path = np.asarray(state_path, dtype=np.int64)
        if (
            path.shape != (log_probs.shape[0],)
            or np.any(path < 0)
            or np.any(path >= 2 * len(labels) + 1)
        ):
            raise ValueError("state_path is incompatible with this utterance")
    inventory = (
        tuple(index for index in range(log_probs.shape[1]) if index != blank_id)
        if phone_ids is None
        else tuple(int(value) for value in phone_ids)
    )
    if not inventory or len(inventory) != len(set(inventory)):
        raise ValueError("phone_ids must be nonempty and unique")
    if any(
        phone == blank_id or phone < 0 or phone >= log_probs.shape[1]
        for phone in inventory
    ):
        raise ValueError("phone_ids must be valid nonblank class ids")
    if any(label not in inventory for label in labels):
        raise ValueError("all canonical labels must occur in phone_ids")

    output: list[ForcedAlignmentPhoneScore] = []
    for position, target in enumerate(labels):
        frames = np.flatnonzero(path == 2 * position + 1)
        if frames.size == 0:
            raise RuntimeError("valid Viterbi path skipped a canonical label state")
        target_score = _logmeanexp(log_probs[frames, target])
        competitors = tuple(phone for phone in inventory if phone != target)
        if not competitors:
            raise ValueError("conventional GoP requires at least two nonblank phones")
        competitor_scores = np.asarray(
            [_logmeanexp(log_probs[frames, phone]) for phone in competitors],
            dtype=np.float64,
        )
        winner = int(np.argmax(competitor_scores))
        top_score = float(competitor_scores[winner])
        output.append(
            ForcedAlignmentPhoneScore(
                position=position,
                phone_id=target,
                start_frame=int(frames[0]),
                end_frame=int(frames[-1]) + 1,
                frame_count=int(frames.size),
                score=target_score,
                top_competitor_id=competitors[winner],
                top_competitor_score=top_score,
                posterior_margin=float(target_score - top_score),
            )
        )
    return output


def ctc_sf_sd_norm(
    frame_outputs: np.ndarray | Sequence[Sequence[float]],
    canonical: Sequence[int],
    *,
    blank_id: int = 0,
    phone_ids: Sequence[int] | None = None,
    occurrence_floor: float = 1.0,
    max_work: int | None = None,
    input_is_logits: bool = False,
):
    """Return exact CTC-SF SD results, including registered SD-Norm scores."""

    log_probs = (
        log_probs_from_logits(frame_outputs)
        if input_is_logits
        else np.asarray(frame_outputs, dtype=np.float64)
    )
    return segmentation_free_scores(
        log_probs,
        canonical,
        blank_id=blank_id,
        phone_ids=phone_ids,
        occurrence_floor=occurrence_floor,
        max_work=max_work,
    )


def same_backend_scalar_baselines(
    frame_outputs: np.ndarray | Sequence[Sequence[float]],
    canonical: Sequence[int],
    *,
    blank_id: int = 0,
    phone_ids: Sequence[int] | None = None,
    occurrence_floor: float = 1.0,
    max_work: int | None = None,
    input_is_logits: bool = False,
) -> dict[str, list[float]]:
    """Compute all registered scalar alignment/CTC-SF comparator columns."""

    log_probs = (
        log_probs_from_logits(frame_outputs)
        if input_is_logits
        else np.asarray(frame_outputs, dtype=np.float64)
    )
    # Conventional GoP and CTC-SA intentionally use the same canonical
    # Viterbi alignment.  Compute it once; the previous implementation ran the
    # identical dynamic program twice.
    state_path, _ = ctc_viterbi_state_path(
        log_probs, canonical, blank_id=blank_id
    )
    forced = conventional_forced_alignment_gop(
        log_probs,
        canonical,
        blank_id=blank_id,
        phone_ids=phone_ids,
        input_is_logits=False,
        state_path=state_path,
    )
    sf = ctc_sf_sd_norm(
        log_probs,
        canonical,
        blank_id=blank_id,
        phone_ids=phone_ids,
        occurrence_floor=occurrence_floor,
        max_work=max_work,
        input_is_logits=False,
    )
    return {
        CONVENTIONAL_METHOD: [row.score for row in forced],
        CTC_SA_METHOD: self_alignment_scores(
            log_probs, canonical, blank_id=blank_id, state_path=state_path
        ),
        CTC_SF_SD_METHOD: [row.score for row in sf],
        CTC_SF_SD_NORM_METHOD: [row.normalized_score for row in sf],
    }


# Exact explicit candidate set and relative weights from the audited code9
# implementation.  It is retained only as the registered hard-pruning
# ablation; the DA-CF headline always gives all 39 error candidates positive
# mass.
LEGACY_HARD_CONFUSIONS: dict[str, dict[str, float]] = {
    "P": {"B": 1.0, "F": 0.7, "M": 0.45},
    "B": {"P": 1.0, "V": 0.7, "M": 0.45},
    "T": {"D": 1.0, "S": 0.7, "N": 0.45, "K": 0.35},
    "D": {"T": 1.0, "Z": 0.7, "N": 0.45, "G": 0.35},
    "K": {"G": 1.0, "T": 0.5},
    "G": {"K": 1.0, "D": 0.5},
    "CH": {"JH": 1.0, "SH": 0.8, "T": 0.5},
    "JH": {"CH": 1.0, "ZH": 0.8, "D": 0.5},
    "F": {"V": 1.0, "TH": 0.7, "P": 0.5},
    "V": {"F": 1.0, "DH": 0.7, "B": 0.5},
    "TH": {"DH": 1.0, "F": 0.7, "S": 0.5},
    "DH": {"TH": 1.0, "V": 0.7, "Z": 0.5},
    "S": {"Z": 1.0, "SH": 0.7, "TH": 0.55, "T": 0.45},
    "Z": {"S": 1.0, "ZH": 0.7, "D": 0.45},
    "SH": {"ZH": 1.0, "S": 0.7, "CH": 0.45},
    "ZH": {"SH": 1.0, "Z": 0.7, "JH": 0.45},
    "HH": {"F": 0.45, "TH": 0.45},
    "M": {"N": 0.8, "B": 0.5},
    "N": {"M": 0.7, "NG": 0.7, "D": 0.45},
    "NG": {"N": 0.8, "G": 0.45},
    "L": {"R": 1.0, "W": 0.55},
    "R": {"L": 1.0, "W": 0.55, "ER": 0.45},
    "W": {"R": 0.7, "UW": 0.55},
    "Y": {"IY": 0.55},
    "AA": {"AH": 1.0, "AO": 0.65},
    "AE": {"EH": 1.0, "AH": 0.85},
    "AH": {"AA": 0.75, "AE": 0.75, "IH": 0.75, "ER": 0.55},
    "AO": {"AA": 0.75, "OW": 0.65},
    "AW": {"AA": 0.55, "OW": 0.55},
    "AY": {"AA": 0.55, "EY": 0.55},
    "EH": {"AE": 1.0, "IH": 0.75, "AH": 0.55},
    "ER": {"R": 0.65, "AH": 0.65},
    "EY": {"EH": 0.75, "IH": 0.55},
    "IH": {"IY": 0.85, "EH": 0.75, "AH": 0.65},
    "IY": {"IH": 1.0, "Y": 0.55},
    "OW": {"AO": 0.75, "UH": 0.55},
    "OY": {"AO": 0.55, "IY": 0.45},
    "UH": {"UW": 0.85, "AH": 0.65},
    "UW": {"UH": 1.0, "W": 0.55},
}


def legacy_hard_candidate_graph(
    inventory: Sequence[str] = DEFAULT_PHONE_INVENTORY,
    *,
    include_deletion: bool = True,
    deletion_weight: float = 0.5,
) -> dict[str, dict[str, float]]:
    """Return the deterministic normalized code9 finite candidate graph."""

    phones = tuple(str(phone).upper() for phone in inventory)
    if not phones or len(phones) != len(set(phones)):
        raise ValueError("inventory must be nonempty and unique")
    missing = sorted(set(phones) - set(LEGACY_HARD_CONFUSIONS))
    if missing:
        raise ValueError(f"legacy graph has no explicit entry for {missing}")
    if not math.isfinite(deletion_weight) or deletion_weight <= 0.0:
        raise ValueError("deletion_weight must be finite and positive")
    allowed = set(phones)
    output: dict[str, dict[str, float]] = {}
    for target in phones:
        raw = {
            alternative: float(weight)
            for alternative, weight in LEGACY_HARD_CONFUSIONS[target].items()
            if alternative in allowed and alternative != target
        }
        if include_deletion:
            raw[DELETION] = float(deletion_weight)
        if not raw:
            raise ValueError(f"legacy hard graph leaves {target} without candidates")
        total = float(sum(raw.values()))
        output[target] = {
            alternative: weight / total for alternative, weight in raw.items()
        }
    return output


def project_priors_to_hard_graph(
    priors: Mapping[str, Mapping[str, float]],
    *,
    graph: Mapping[str, Mapping[str, float] | Sequence[str]] | None = None,
) -> dict[str, dict[str, float]]:
    """Hard-prune full priors and renormalize without changing their weights.

    This helper makes the hard-graph ablation comparable for fixed, uniform, or
    fold-adapted priors.  Candidates outside the explicit graph are truly
    absent from the denominator rather than assigned a tiny probability.
    """

    hard = legacy_hard_candidate_graph(tuple(priors)) if graph is None else graph
    if set(priors) != set(hard):
        raise ValueError("prior and hard graph must have identical target keys")
    output: dict[str, dict[str, float]] = {}
    for target in priors:
        candidates = tuple(hard[target])
        if not candidates or len(candidates) != len(set(candidates)):
            raise ValueError(f"hard graph candidates for {target} must be unique")
        missing = [candidate for candidate in candidates if candidate not in priors[target]]
        if missing:
            raise ValueError(f"full prior for {target} lacks hard candidates {missing}")
        raw = {candidate: float(priors[target][candidate]) for candidate in candidates}
        if any(not math.isfinite(value) or value <= 0.0 for value in raw.values()):
            raise ValueError("retained prior masses must be finite and positive")
        total = float(sum(raw.values()))
        output[target] = {
            candidate: value / total for candidate, value in raw.items()
        }
    return output


# Short aliases for orchestration and compatibility with experiment-table names.
forced_alignment_gop = conventional_forced_alignment_gop
build_hard_candidate_graph = legacy_hard_candidate_graph
hard_prune_priors = project_priors_to_hard_graph


__all__ = [
    "CONVENTIONAL_METHOD",
    "CTC_SA_METHOD",
    "CTC_SF_SD_METHOD",
    "CTC_SF_SD_NORM_METHOD",
    "LEGACY_HARD_GRAPH_METHOD",
    "SCORE_DIRECTION",
    "ForcedAlignmentPhoneScore",
    "LEGACY_HARD_CONFUSIONS",
    "build_hard_candidate_graph",
    "conventional_forced_alignment_gop",
    "ctc_sf_sd_norm",
    "forced_alignment_gop",
    "hard_prune_priors",
    "legacy_hard_candidate_graph",
    "project_priors_to_hard_graph",
    "same_backend_scalar_baselines",
]
