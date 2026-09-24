"""DA-CF-GoP likelihood-ratio and conditional alternative scoring."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Mapping, Sequence

import numpy as np

from .calibration import DeletionCalibrator, SpeakerWeightedPlattCalibrator, conditional_softmax
from .ctc import CounterfactualMatrix, DELETION_ID, logsumexp
from .priors import DELETION


@dataclass(frozen=True)
class PhoneScore:
    position: int
    target_phone: str
    gop: float
    error_evidence: float
    candidate_probabilities: dict[str, float]
    top_alternative: str
    top_alternative_probability_given_error: float
    top2_margin: float
    conditional_entropy: float
    error_probability: float | None = None


def _validate_inputs(
    canonical_log_probability: float,
    alternative_log_probabilities: Mapping[str, float],
    prior: Mapping[str, float],
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    if not np.isfinite(canonical_log_probability):
        raise ValueError("canonical sequence must have finite CTC probability")
    if set(alternative_log_probabilities) != set(prior):
        raise ValueError("alternative likelihoods and prior must have identical candidate keys")
    if not prior:
        raise ValueError("at least one error candidate is required")
    keys = tuple(sorted(prior))
    likelihoods = np.asarray([alternative_log_probabilities[key] for key in keys], dtype=np.float64)
    weights = np.asarray([prior[key] for key in keys], dtype=np.float64)
    if np.isnan(likelihoods).any() or np.isposinf(likelihoods).any() or not np.isfinite(likelihoods).any():
        raise ValueError("alternative likelihoods may contain -inf but need one finite candidate")
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("candidate prior must be finite and strictly positive")
    if not np.isclose(weights.sum(), 1.0, atol=1e-10, rtol=0.0):
        raise ValueError("candidate prior must sum to one")
    return keys, likelihoods, weights


def gop_log_likelihood_ratio(
    canonical_log_probability: float,
    alternative_log_probabilities: Mapping[str, float],
    prior: Mapping[str, float],
) -> float:
    """Compute target likelihood against the prior-weighted error mixture."""

    _keys, likelihoods, weights = _validate_inputs(
        canonical_log_probability, alternative_log_probabilities, prior
    )
    error_mixture = logsumexp(likelihoods + np.log(weights))
    if not np.isfinite(error_mixture):
        raise ValueError("the error mixture has no finite probability")
    return float(canonical_log_probability - error_mixture)


def conditional_alternative_probabilities(
    alternative_log_probabilities: Mapping[str, float],
    prior: Mapping[str, float],
    *,
    temperature: float = 1.0,
) -> dict[str, float]:
    """Return P(candidate | error, audio, position), excluding the target."""

    # A finite dummy canonical value lets the shared validation enforce the
    # exact candidate-key/simplex contract.
    keys, likelihoods, weights = _validate_inputs(0.0, alternative_log_probabilities, prior)
    probabilities = conditional_softmax(likelihoods + np.log(weights), temperature)
    result = {key: float(value) for key, value in zip(keys, probabilities)}
    # Correct the final ULP deterministically so downstream exact schema checks
    # see a simplex without changing candidate ordering.
    last = keys[-1]
    result[last] += 1.0 - sum(result.values())
    return result


def score_phone(
    *,
    position: int,
    target_phone: str,
    canonical_log_probability: float,
    alternative_log_probabilities: Mapping[str, float],
    prior: Mapping[str, float],
    deletion_calibrator: DeletionCalibrator | None = None,
    temperature: float = 1.0,
) -> PhoneScore:
    """Score one canonical token using all substitutions plus ``<DEL>``."""

    if isinstance(position, bool) or int(position) != position or position < 0:
        raise ValueError("position must be a non-negative integer")
    target = str(target_phone).upper()
    adjusted = {str(key): float(value) for key, value in alternative_log_probabilities.items()}
    if deletion_calibrator is not None:
        if DELETION not in adjusted:
            raise ValueError("deletion calibration requested but <DEL> is absent")
        adjusted[DELETION] = deletion_calibrator.calibrate(target, adjusted[DELETION])
    gop = gop_log_likelihood_ratio(canonical_log_probability, adjusted, prior)
    probabilities = conditional_alternative_probabilities(
        adjusted, prior, temperature=temperature
    )
    ranked = sorted(probabilities.items(), key=lambda item: (-item[1], item[0]))
    top_phone, top_probability = ranked[0]
    second_probability = ranked[1][1] if len(ranked) > 1 else 0.0
    values = np.asarray(list(probabilities.values()), dtype=np.float64)
    entropy = float(-np.sum(values * np.log(np.clip(values, 1e-300, 1.0))))
    return PhoneScore(
        int(position),
        target,
        gop,
        -gop,
        probabilities,
        top_phone,
        float(top_probability),
        float(top_probability - second_probability),
        entropy,
    )


def score_counterfactual_matrix(
    matrix: CounterfactualMatrix,
    canonical_phones: Sequence[str],
    id_to_phone: Mapping[int, str],
    priors: Mapping[str, Mapping[str, float]],
    *,
    deletion_calibrator: DeletionCalibrator | None = None,
    temperature: float = 1.0,
) -> list[PhoneScore]:
    """Convert a full CTC counterfactual matrix into DA-CF-GoP rows."""

    candidate_ids = np.asarray(matrix.candidate_ids)
    likelihoods = np.asarray(matrix.corrected_log_probabilities)
    if len(canonical_phones) != candidate_ids.shape[0]:
        raise ValueError("canonical phone count and counterfactual matrix differ")
    output: list[PhoneScore] = []
    for position, target_value in enumerate(canonical_phones):
        target = str(target_value).upper()
        alternatives: dict[str, float] = {}
        for candidate_id, log_probability in zip(candidate_ids[position], likelihoods[position]):
            if int(candidate_id) == DELETION_ID:
                key = DELETION
            else:
                if int(candidate_id) not in id_to_phone:
                    raise ValueError(f"candidate id {candidate_id} has no phone label")
                key = str(id_to_phone[int(candidate_id)]).upper()
            if key in alternatives:
                raise ValueError(f"duplicate candidate label after id mapping: {key}")
            alternatives[key] = float(log_probability)
        if target not in priors:
            raise ValueError(f"no prior for target phone {target}")
        output.append(
            score_phone(
                position=position,
                target_phone=target,
                canonical_log_probability=matrix.canonical_corrected_log_probability,
                alternative_log_probabilities=alternatives,
                prior=priors[target],
                deletion_calibrator=deletion_calibrator,
                temperature=temperature,
            )
        )
    return output


def apply_platt_calibration(
    scores: Sequence[PhoneScore],
    calibrator: SpeakerWeightedPlattCalibrator,
) -> list[PhoneScore]:
    """Attach calibrated p(error) without altering conditional alternatives."""

    if not scores:
        return []
    probabilities = calibrator.predict_error_probability(
        [score.error_evidence for score in scores]
    )
    return [
        replace(score, error_probability=float(probability))
        for score, probability in zip(scores, probabilities)
    ]


# Compact alias used in tables and downstream experiment code.
gop_llr = gop_log_likelihood_ratio

