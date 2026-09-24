"""All-phone fixed and speaker-disjoint dysarthria confusion priors."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


DELETION = "<DEL>"
LAMBDA_GRID = (0.0, 0.25, 0.5, 1.0)
DEFAULT_PHONE_INVENTORY = (
    "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH",
    "EH", "ER", "EY", "F", "G", "HH", "IH", "IY", "JH", "K",
    "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH",
    "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH",
)

# Standard ARPABET articulatory descriptors, migrated as data rather than as a
# runtime dependency on code6.
_VOWELS = {
    "IY": (1.00, 0.00, 0), "IH": (0.85, 0.10, 0), "EY": (0.70, 0.00, 0),
    "EH": (0.50, 0.00, 0), "AE": (0.15, 0.10, 0), "AA": (0.00, 1.00, 0),
    "AH": (0.40, 0.60, 0), "ER": (0.50, 0.50, 0), "AO": (0.30, 1.00, 1),
    "OW": (0.60, 1.00, 1), "UH": (0.85, 0.85, 1), "UW": (1.00, 1.00, 1),
    "AY": (0.10, 0.40, 0), "AW": (0.10, 0.60, 0), "OY": (0.40, 1.00, 1),
}
_CONSONANTS = {
    "B": ("bilabial", "stop", 1), "P": ("bilabial", "stop", 0),
    "M": ("bilabial", "nasal", 1), "W": ("bilabial", "glide", 1),
    "F": ("labiodental", "fricative", 0), "V": ("labiodental", "fricative", 1),
    "TH": ("dental", "fricative", 0), "DH": ("dental", "fricative", 1),
    "T": ("alveolar", "stop", 0), "D": ("alveolar", "stop", 1),
    "S": ("alveolar", "fricative", 0), "Z": ("alveolar", "fricative", 1),
    "N": ("alveolar", "nasal", 1), "L": ("alveolar", "liquid", 1),
    "SH": ("postalveolar", "fricative", 0), "ZH": ("postalveolar", "fricative", 1),
    "CH": ("postalveolar", "affricate", 0), "JH": ("postalveolar", "affricate", 1),
    "R": ("postalveolar", "liquid", 1), "Y": ("palatal", "glide", 1),
    "K": ("velar", "stop", 0), "G": ("velar", "stop", 1),
    "NG": ("velar", "nasal", 1), "HH": ("glottal", "fricative", 0),
}
_PLACE_ORDER = {
    "bilabial": 0, "labiodental": 1, "dental": 2, "alveolar": 3,
    "postalveolar": 4, "palatal": 5, "velar": 6, "glottal": 7,
}


def feature_distance(target: str, alternative: str) -> float:
    """Phonetically motivated distance migrated from the code6 graph."""

    target, alternative = str(target).upper(), str(alternative).upper()
    if target == alternative:
        return 0.0
    target_vowel, alternative_vowel = target in _VOWELS, alternative in _VOWELS
    if target_vowel and alternative_vowel:
        ht, bt, rt = _VOWELS[target]
        ha, ba, ra = _VOWELS[alternative]
        return float(abs(ht - ha) + abs(bt - ba) + 0.7 * abs(rt - ra))
    if not target_vowel and not alternative_vowel:
        if target not in _CONSONANTS or alternative not in _CONSONANTS:
            raise ValueError(f"phone lacks a fixed articulatory descriptor: {target}/{alternative}")
        pt, mt, vt = _CONSONANTS[target]
        pa, ma, va = _CONSONANTS[alternative]
        distance = 1.5 * abs(_PLACE_ORDER[pt] - _PLACE_ORDER[pa]) / 7.0
        distance += 0.0 if mt == ma else 0.6
        distance += 0.0 if vt == va else 0.4
        return float(distance)
    return 2.5


def literature_multiplier(target: str, alternative: str) -> float:
    """Frozen modest multipliers for common dysarthric error categories."""

    target, alternative = str(target).upper(), str(alternative).upper()
    if alternative == DELETION:
        # Exact code6 literature multiplier: consonant/cluster deletion was
        # boosted more strongly than vowel deletion.  The separate frozen
        # deletion base weight is applied by ``fixed_soft_priors``.
        return 1.1 if target in _VOWELS else 2.0
    target_vowel, alternative_vowel = target in _VOWELS, alternative in _VOWELS
    if target_vowel and alternative_vowel:
        return 2.0 if alternative == "AH" else 1.3
    if target_vowel != alternative_vowel:
        return 0.7
    if target not in _CONSONANTS or alternative not in _CONSONANTS:
        return 1.0
    place_t, manner_t, voice_t = _CONSONANTS[target]
    place_a, manner_a, voice_a = _CONSONANTS[alternative]
    multiplier = 1.0
    obstruents = {"stop", "fricative", "affricate"}
    if manner_t != manner_a and manner_t in obstruents and manner_a in obstruents:
        multiplier *= 1.8
    if voice_t != voice_a:
        multiplier *= 1.5
    if place_t != place_a and {place_t, place_a} <= {"alveolar", "velar"}:
        multiplier *= 1.6
    if manner_t == "nasal" or manner_a == "nasal":
        multiplier *= 1.3
    return float(multiplier)


def _normalise(weights: Mapping[str, float]) -> dict[str, float]:
    if not weights:
        raise ValueError("a prior must contain at least one alternative")
    values = {str(key): float(value) for key, value in weights.items()}
    if any(not np.isfinite(value) or value <= 0.0 for value in values.values()):
        raise ValueError("every alternative prior weight must be finite and positive")
    total = float(sum(values.values()))
    return {key: value / total for key, value in values.items()}


def alternatives_for(
    target: str,
    inventory: Sequence[str] = DEFAULT_PHONE_INVENTORY,
    *,
    include_deletion: bool = True,
) -> tuple[str, ...]:
    target = str(target).upper()
    phones = tuple(str(phone).upper() for phone in inventory)
    if len(phones) != len(set(phones)):
        raise ValueError("phone inventory contains duplicates")
    if target not in phones:
        raise ValueError(f"target phone is outside the inventory: {target}")
    return tuple(phone for phone in phones if phone != target) + (
        (DELETION,) if include_deletion else ()
    )


def fixed_soft_priors(
    inventory: Sequence[str] = DEFAULT_PHONE_INVENTORY,
    *,
    include_deletion: bool = True,
    deletion_weight: float = 0.5,
) -> dict[str, dict[str, float]]:
    """Return the fixed all-phone feature/literature prior.

    No edge is removed: feature distance and literature knowledge only change
    its positive mass.  Unlisted literature relations have multiplier one.
    """

    phones = tuple(str(phone).upper() for phone in inventory)
    result: dict[str, dict[str, float]] = {}
    for target in phones:
        raw: dict[str, float] = {}
        for alternative in alternatives_for(target, phones, include_deletion=include_deletion):
            if alternative == DELETION:
                raw[alternative] = float(deletion_weight) * literature_multiplier(
                    target, alternative
                )
            else:
                raw[alternative] = math.exp(-feature_distance(target, alternative)) * literature_multiplier(
                    target, alternative
                )
        result[target] = _normalise(raw)
    validate_priors(result, phones, include_deletion=include_deletion)
    return result


def _field(row: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(row, Mapping) and name in row:
            return row[name]
        if hasattr(row, name):
            return getattr(row, name)
    return default


def empirical_confusion_priors(
    records: Iterable[Any],
    inventory: Sequence[str] = DEFAULT_PHONE_INVENTORY,
    *,
    fixed: Mapping[str, Mapping[str, float]] | None = None,
    pseudocount: float = 20.0,
    enrichment_cap: float = 4.0,
    include_deletion: bool = True,
) -> dict[str, dict[str, float]]:
    """Estimate speaker-equal dysarthric-vs-healthy confusion enrichment.

    Only stable substitutions and deletions add empirical error counts.  Stable
    matches still establish that a speaker observed the target, so a speaker
    with no such error contributes the fixed Dirichlet prior rather than being
    silently omitted.
    """

    if pseudocount <= 0 or enrichment_cap < 1:
        raise ValueError("pseudocount must be positive and enrichment_cap at least one")
    phones = tuple(str(phone).upper() for phone in inventory)
    fixed_map = (
        {target: dict(weights) for target, weights in fixed.items()}
        if fixed is not None
        else fixed_soft_priors(phones, include_deletion=include_deletion)
    )
    validate_priors(fixed_map, phones, include_deletion=include_deletion)

    observed: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    counts: defaultdict[tuple[str, str, str], defaultdict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    for row in records:
        if not bool(_field(row, "stable", "is_stable", default=True)):
            continue
        event_type = str(_field(row, "event_type", "operation", default="")).lower()
        if event_type == "insertion":
            continue
        speaker = str(_field(row, "speaker_id", "speaker", default=""))
        group = str(_field(row, "speaker_group", "group", default="")).lower()
        group = "dysarthric" if group in {"dys", "patient", "dysarthric"} else (
            "healthy" if group in {"healthy", "control", "ctl"} else group
        )
        target = str(_field(row, "target_phone", "target", "canonical", default="")).upper()
        realized_raw = _field(row, "realized_phone", "realized", "alternative", default=None)
        if not speaker or group not in {"dysarthric", "healthy"} or target not in fixed_map:
            continue
        observed[(group, target)].add(speaker)
        if realized_raw is None:
            continue
        realized_text = str(realized_raw)
        realized = DELETION if realized_text.upper() in {DELETION, "<DEL>", "<DELETE>"} else realized_text.upper()
        if realized == target:
            continue
        if realized not in fixed_map[target]:
            continue
        counts[(group, target, speaker)][realized] += 1.0

    empirical: dict[str, dict[str, float]] = {}
    for target in phones:
        alternatives = tuple(fixed_map[target])
        group_means: dict[str, dict[str, float]] = {}
        for group in ("dysarthric", "healthy"):
            speakers = sorted(observed[(group, target)])
            posteriors: list[dict[str, float]] = []
            for speaker in speakers:
                local = counts[(group, target, speaker)]
                total_errors = float(sum(local.values()))
                denominator = total_errors + pseudocount
                posteriors.append(
                    {
                        alternative: (
                            local[alternative] + pseudocount * fixed_map[target][alternative]
                        ) / denominator
                        for alternative in alternatives
                    }
                )
            if posteriors:
                group_means[group] = {
                    alternative: float(np.mean([row[alternative] for row in posteriors]))
                    for alternative in alternatives
                }
            else:
                group_means[group] = dict(fixed_map[target])
        enriched = {}
        for alternative in alternatives:
            ratio = group_means["dysarthric"][alternative] / group_means["healthy"][alternative]
            ratio = float(np.clip(ratio, 1.0 / enrichment_cap, enrichment_cap))
            enriched[alternative] = fixed_map[target][alternative] * ratio
        empirical[target] = _normalise(enriched)
    validate_priors(empirical, phones, include_deletion=include_deletion)
    return empirical


def mix_soft_priors(
    fixed: Mapping[str, Mapping[str, float]],
    empirical: Mapping[str, Mapping[str, float]],
    lambda_value: float,
    *,
    uniform_epsilon: float = 0.05,
) -> dict[str, dict[str, float]]:
    """Apply the pre-registered epsilon-backoff/interpolation formula."""

    if lambda_value not in LAMBDA_GRID:
        raise ValueError(f"lambda must be one of {LAMBDA_GRID}")
    if not 0.0 <= uniform_epsilon < 1.0:
        raise ValueError("uniform_epsilon must lie in [0, 1)")
    if set(fixed) != set(empirical):
        raise ValueError("fixed and empirical priors have different targets")
    output: dict[str, dict[str, float]] = {}
    for target in fixed:
        if set(fixed[target]) != set(empirical[target]):
            raise ValueError(f"fixed and empirical candidates differ for {target}")
        candidates = tuple(fixed[target])
        uniform = 1.0 / len(candidates)
        output[target] = _normalise(
            {
                candidate: uniform_epsilon * uniform
                + (1.0 - uniform_epsilon)
                * (
                    (1.0 - lambda_value) * float(fixed[target][candidate])
                    + lambda_value * float(empirical[target][candidate])
                )
                for candidate in candidates
            }
        )
    return output


def build_soft_priors(
    records: Iterable[Any],
    inventory: Sequence[str] = DEFAULT_PHONE_INVENTORY,
    *,
    lambda_value: float,
    pseudocount: float = 20.0,
    enrichment_cap: float = 4.0,
    uniform_epsilon: float = 0.05,
    include_deletion: bool = True,
) -> dict[str, dict[str, float]]:
    fixed = fixed_soft_priors(inventory, include_deletion=include_deletion)
    empirical = empirical_confusion_priors(
        records,
        inventory,
        fixed=fixed,
        pseudocount=pseudocount,
        enrichment_cap=enrichment_cap,
        include_deletion=include_deletion,
    )
    return mix_soft_priors(
        fixed, empirical, lambda_value, uniform_epsilon=uniform_epsilon
    )


def validate_priors(
    priors: Mapping[str, Mapping[str, float]],
    inventory: Sequence[str] = DEFAULT_PHONE_INVENTORY,
    *,
    include_deletion: bool = True,
) -> None:
    phones = tuple(str(phone).upper() for phone in inventory)
    if set(priors) != set(phones):
        raise ValueError("prior targets do not exactly match the phone inventory")
    for target in phones:
        expected = set(alternatives_for(target, phones, include_deletion=include_deletion))
        if set(priors[target]) != expected:
            raise ValueError(f"prior candidates do not match the all-phone contract for {target}")
        values = np.asarray(list(priors[target].values()), dtype=np.float64)
        if not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError(f"prior for {target} is not strictly positive and finite")
        if not np.isclose(values.sum(), 1.0, atol=1e-12, rtol=0.0):
            raise ValueError(f"prior for {target} does not sum to one")


@dataclass(frozen=True)
class LambdaSelection:
    lambda_value: float
    macro_auprc: float
    per_lambda: dict[float, float]
    per_speaker: dict[float, dict[str, float]]


def select_lambda_by_macro_auprc(
    scores_by_lambda: Mapping[float, Sequence[float]],
    labels: Sequence[int],
    speakers: Sequence[str],
) -> LambdaSelection:
    """Select lambda on inner OOF scores; exact ties prefer smaller lambda."""

    from sklearn.metrics import average_precision_score

    truth = np.asarray(labels, dtype=np.int64)
    speaker_array = np.asarray(speakers, dtype=str)
    if truth.ndim != 1 or speaker_array.shape != truth.shape:
        raise ValueError("labels and speakers must be same-length vectors")
    if set(scores_by_lambda) != set(LAMBDA_GRID):
        raise ValueError(f"scores must be supplied for the frozen lambda grid {LAMBDA_GRID}")
    per_speaker: dict[float, dict[str, float]] = {}
    per_lambda: dict[float, float] = {}
    for lambda_value in sorted(scores_by_lambda):
        scores = np.asarray(scores_by_lambda[lambda_value], dtype=np.float64)
        if scores.shape != truth.shape or not np.isfinite(scores).all():
            raise ValueError("each lambda score vector must match labels and be finite")
        local: dict[str, float] = {}
        for speaker in sorted(set(speaker_array)):
            mask = speaker_array == speaker
            if len(np.unique(truth[mask])) < 2:
                continue
            local[speaker] = float(average_precision_score(truth[mask], scores[mask]))
        if not local:
            raise ValueError("macro AUPRC requires at least one speaker with both classes")
        per_speaker[float(lambda_value)] = local
        per_lambda[float(lambda_value)] = float(np.mean(list(local.values())))
    best = min(
        per_lambda,
        key=lambda value: (-round(per_lambda[value], 12), value),
    )
    return LambdaSelection(best, per_lambda[best], per_lambda, per_speaker)
