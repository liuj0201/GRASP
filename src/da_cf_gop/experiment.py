"""Frozen nested-LOSO phone experiment for DA-CF-GoP.

The orchestration in this module deliberately separates *fit* from *predict*.
Only training-patient and healthy-reference PHN labels are accepted by the fit
path.  :func:`predict_with_fitted_fold` consumes logits and the prompt-derived
canonical sequence only; gold labels are joined afterwards by the evaluation
writer.  This makes the outer-speaker leakage boundary executable rather than
merely documentary.

Historical ``phoneme_project/code*`` modules are never imported.  They are
represented only by independently migrated implementations in this package.
"""

from __future__ import annotations

from dataclasses import dataclass
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from .artifacts import ArtifactLayout, write_csv, write_jsonl_gz, write_stage_summary
from .backend import load_logits_cache
from .baselines import (
    CONVENTIONAL_METHOD,
    CTC_SF_SD_NORM_METHOD,
    LEGACY_HARD_GRAPH_METHOD,
    conventional_forced_alignment_gop,
    ctc_sf_sd_norm,
    legacy_hard_candidate_graph,
)
from .calibration import (
    AlternativeTemperatureCalibrator,
    CTCSequenceRecalibrator,
    CalibrationUtterance,
    DeletionCalibrator,
    SpeakerWeightedPlattCalibrator,
)
from .config import load_config
from .ctc import (
    CounterfactualMatrix,
    DELETION_ID,
    counterfactual_log_probability_matrix,
    ctc_log_probabilities_batch,
    ctc_minimum_frames,
    log_softmax,
    logsumexp,
    uniform_ctc_log_path_count,
)
from .data import ManifestRow, read_manifest
from .folds import OuterFold, nested_loso_folds, training_artifact_hash
from .llm_schema import SpecificOutputGate, fit_specific_output_gate
from .metrics import (
    evaluate_predictions,
    exact_sign_flip_test,
    paired_speaker_bootstrap,
)
from .phonology import ARPABET_39, CTC_ID_TO_PHONE, PHONE_TO_CTC_ID, StableTokenLabel
from .priors import (
    DELETION,
    LAMBDA_GRID,
    alternatives_for,
    empirical_confusion_priors,
    fixed_soft_priors,
    mix_soft_priors,
    select_lambda_by_macro_auprc,
)
from .provenance import (
    build_cache_descriptor,
    companion_descriptor_path,
    read_json,
    sha256_file,
    sha256_json,
    sha256_path,
    validate_cache_descriptor,
    write_deterministic_npz,
    write_json,
)
from .stages import layout_from_config, logit_cache_path, manifest_path


Progress = Callable[[str], None]
_PATH_HASH_CACHE: dict[str, str] = {}
_FILE_HASH_CACHE: dict[tuple[str, int, int], str] = {}
EXPERIMENT_IMPLEMENTATION_VERSION = "nested-loso.v2"

# Registered table labels.  Names are intentionally stable because they are
# part of the machine-readable evaluation protocol.
METHOD_CTC_SF_RECALIBRATED = "ctc_sf_sd_norm_sequence_recalibrated"
METHOD_UNIFORM_RAW = "all_phone_cf_uniform_raw"
METHOD_UNIFORM_CALIBRATED = "all_phone_cf_calibrated"
METHOD_FIXED = "da_cf_fixed_no_phn"
METHOD_ADAPTED = "da_cf_adapted"
METHOD_ABLATION_FIXED = "da_cf_adapted_fixed_prior"
METHOD_ABLATION_UNIFORM = "da_cf_adapted_uniform_prior"
METHOD_ABLATION_HARD = "da_cf_adapted_hard_pruning"
METHOD_ABLATION_NO_TOPOLOGY = "da_cf_adapted_no_topology_correction"
METHOD_ABLATION_NO_RECALIBRATION = "da_cf_adapted_no_sequence_recalibration"
METHOD_ABLATION_NO_DELETION_CALIBRATION = "da_cf_adapted_no_deletion_calibration"
METHOD_ABLATION_NO_DELETION = "da_cf_adapted_no_deletion"
METHOD_ABLATION_NO_ABSTENTION = "da_cf_adapted_no_abstention"

CORE_METHODS = (
    METHOD_UNIFORM_RAW,
    METHOD_UNIFORM_CALIBRATED,
    LEGACY_HARD_GRAPH_METHOD,
    METHOD_FIXED,
    METHOD_ADAPTED,
    METHOD_ABLATION_FIXED,
    METHOD_ABLATION_UNIFORM,
    METHOD_ABLATION_HARD,
    METHOD_ABLATION_NO_TOPOLOGY,
    METHOD_ABLATION_NO_RECALIBRATION,
    METHOD_ABLATION_NO_DELETION_CALIBRATION,
    METHOD_ABLATION_NO_DELETION,
    METHOD_ABLATION_NO_ABSTENTION,
)
BASELINE_METHODS = (
    CONVENTIONAL_METHOD,
    CTC_SF_SD_NORM_METHOD,
    METHOD_CTC_SF_RECALIBRATED,
)
ALL_METHODS = BASELINE_METHODS + CORE_METHODS
LAMBDA_DEPENDENT_METHODS = frozenset(
    {
        METHOD_ADAPTED,
        METHOD_ABLATION_HARD,
        METHOD_ABLATION_NO_TOPOLOGY,
        METHOD_ABLATION_NO_RECALIBRATION,
        METHOD_ABLATION_NO_DELETION_CALIBRATION,
        METHOD_ABLATION_NO_DELETION,
        METHOD_ABLATION_NO_ABSTENTION,
    }
)


def _config_value(config: Mapping[str, Any], *path: str, default: Any = None) -> Any:
    value: Any = config
    for component in path:
        if not isinstance(value, Mapping) or component not in value:
            return default
        value = value[component]
    return value


def _config_digest(config: Mapping[str, Any]) -> str:
    declared = config.get("_config_hash")
    if isinstance(declared, str) and len(declared) == 64:
        return declared
    # Test overrides need not have been loaded through config.load_config.
    clean = {key: value for key, value in config.items() if not str(key).startswith("_")}
    return sha256_json(clean)


def _array_sha256(array: np.ndarray) -> str:
    values = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode("ascii"))
    digest.update(json.dumps(values.shape).encode("ascii"))
    digest.update(values.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class ExperimentUtterance:
    """One cached acoustic input and its optional PHN-sidecar labels."""

    event_id: str
    speaker_id: str
    speaker_group: str
    canonical_phones: tuple[str, ...]
    canonical_ids: tuple[int, ...]
    logits: np.ndarray
    realized_ids: tuple[int, ...] = ()
    token_labels: tuple[StableTokenLabel, ...] = ()
    logits_source: str | None = None

    def __post_init__(self) -> None:
        values = np.asarray(self.logits)
        if not self.event_id or not self.speaker_id:
            raise ValueError("event and speaker identifiers are required")
        if self.speaker_group not in {"dysarthric", "healthy"}:
            raise ValueError("speaker_group must be dysarthric or healthy")
        if values.ndim != 2 or not values.shape[0] or not np.isfinite(values).all():
            raise ValueError("utterance logits must be a non-empty finite matrix")
        if len(self.canonical_phones) != len(self.canonical_ids) or not self.canonical_ids:
            raise ValueError("canonical phone/id sequences must be equal and non-empty")
        if any(phone not in ARPABET_39 for phone in self.canonical_phones):
            raise ValueError("canonical sequence is outside the frozen phone vocabulary")
        if any(value <= 0 or value >= values.shape[1] for value in self.canonical_ids):
            raise ValueError("canonical id is outside the nonblank vocabulary")
        expected = tuple(PHONE_TO_CTC_ID[phone] for phone in self.canonical_phones)
        if tuple(self.canonical_ids) != expected:
            raise ValueError("canonical phone/id mapping disagrees with frozen CTC vocabulary")
        if self.realized_ids and any(
            value <= 0 or value >= values.shape[1] for value in self.realized_ids
        ):
            raise ValueError("realized sequence contains an invalid CTC id")
        if self.token_labels:
            if len(self.token_labels) != len(self.canonical_phones):
                raise ValueError("token labels must cover every canonical position")
            if tuple(label.phone_index for label in self.token_labels) != tuple(
                range(len(self.canonical_phones))
            ):
                raise ValueError("token labels are not position ordered")

    def acoustic_only(self) -> "ExperimentUtterance":
        """Return the exact inference view, with every PHN-derived field gone."""

        return ExperimentUtterance(
            self.event_id,
            self.speaker_id,
            self.speaker_group,
            self.canonical_phones,
            self.canonical_ids,
            self.logits,
            (),
            (),
            self.logits_source,
        )

    @property
    def logits_sha256(self) -> str:
        source = Path(self.logits_source) if self.logits_source else None
        if source is not None and source.is_file():
            resolved = source.resolve()
            stat = resolved.stat()
            key = (resolved.as_posix(), int(stat.st_size), int(stat.st_mtime_ns))
            if key not in _FILE_HASH_CACHE:
                _FILE_HASH_CACHE[key] = sha256_file(resolved)
            return _FILE_HASH_CACHE[key]
        return _array_sha256(self.logits)


@dataclass(frozen=True)
class _AcousticManifestRow:
    """Gold-free projection used before cohort prediction finalization."""

    reading_event_id: str
    speaker_id: str
    speaker_group: str
    canonical_phones: tuple[str, ...]


def _manifest_json_records(path: str | Path) -> Iterable[dict[str, Any]]:
    source = Path(path)
    opener = gzip.open if source.suffix.lower() == ".gz" else open
    with opener(source, "rt", encoding="utf-8", newline="") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                raise ValueError(
                    f"invalid manifest row {line_number} in {source}"
                ) from error
            if not isinstance(value, dict):
                raise ValueError(f"manifest row {line_number} is not an object")
            yield value


def _read_acoustic_manifest(path: str | Path) -> list[_AcousticManifestRow]:
    """Project manifest records without constructing PHN/token objects."""

    rows: list[_AcousticManifestRow] = []
    for value in _manifest_json_records(path):
        rows.append(
            _AcousticManifestRow(
                reading_event_id=str(value["reading_event_id"]),
                speaker_id=str(value["speaker_id"]),
                speaker_group=str(value["speaker_group"]),
                canonical_phones=tuple(str(phone) for phone in value["canonical_phones"]),
            )
        )
    ids = [row.reading_event_id for row in rows]
    if not rows or len(ids) != len(set(ids)):
        raise ValueError("acoustic manifest is empty or contains duplicate events")
    return rows


def _read_training_manifest(
    path: str | Path, training_speakers: set[str]
) -> list[ManifestRow]:
    """Materialize PHN-bearing rows only under an explicit training fold."""

    rows = [
        ManifestRow.from_dict(value)
        for value in _manifest_json_records(path)
        if str(value.get("speaker_id", "")) in training_speakers
    ]
    ids = [row.reading_event_id for row in rows]
    if not rows or len(ids) != len(set(ids)):
        raise ValueError("training manifest is empty or contains duplicate events")
    return rows


@dataclass(frozen=True)
class _TokenEvidence:
    position: int
    target: str
    gop: float
    alternative_log_scores: dict[str, float]

    @property
    def error_evidence(self) -> float:
        return -float(self.gop)


@dataclass(frozen=True)
class _DeletionLikelihoods:
    """Canonical/deletion likelihoods needed by the q90 fit.

    The deletion calibrator never consumes substitution likelihoods.  Keeping
    this deliberately narrow representation prevents its training path from
    accidentally requesting the full 39-alternative counterfactual matrix.
    Both raw and topology-corrected values are retained because the frozen
    ablation table fits one calibrator in each score space.
    """

    canonical_log_probability: float
    canonical_corrected_log_probability: float
    deletion_log_probabilities: np.ndarray
    deletion_corrected_log_probabilities: np.ndarray

    def __post_init__(self) -> None:
        raw = np.asarray(self.deletion_log_probabilities)
        corrected = np.asarray(self.deletion_corrected_log_probabilities)
        if raw.ndim != 1 or corrected.shape != raw.shape or raw.size == 0:
            raise ValueError("deletion likelihood arrays must be equal non-empty vectors")


@dataclass(frozen=True)
class _OOFToken:
    method: str
    fold: str
    inner_validation_speaker: str
    speaker: str
    event: str
    phone_index: int
    target: str
    gold_event: str
    gold_realized: str | None
    evidence: _TokenEvidence


@dataclass(frozen=True)
class _DeferredOuterPredictions:
    """Gold-free outer predictions retained until every fold has predicted."""

    fold_id: str
    held_out_patient: str
    event_ids: tuple[str, ...]
    rows: tuple[dict[str, Any], ...]

    def __post_init__(self) -> None:
        forbidden = {"gold_event", "gold_realized", "stable"}
        if not self.event_ids or not self.rows:
            raise ValueError("deferred outer predictions cannot be empty")
        if any(forbidden.intersection(row) for row in self.rows):
            raise ValueError("gold was attached before outer prediction finalization")
        if {str(row["fold"]) for row in self.rows} != {self.fold_id}:
            raise ValueError("deferred prediction fold identity changed")
        if {str(row["speaker"]) for row in self.rows} != {self.held_out_patient}:
            raise ValueError("deferred prediction held-out identity changed")
        if {str(row["event"]) for row in self.rows} != set(self.event_ids):
            raise ValueError("deferred prediction event coverage changed")


@dataclass(frozen=True)
class FittedFold:
    """All training-only state required for PHN-free outer inference."""

    cohort: str
    fold_id: str
    held_out_patient: str
    training_speakers: tuple[str, ...]
    fit_hash: str
    selected_lambda: float
    lambda_metrics: dict[float, float]
    recalibrator: CTCSequenceRecalibrator
    priors: dict[str, dict[str, dict[str, float]]]
    deletion_calibrators: dict[str, DeletionCalibrator]
    platt_calibrators: dict[str, SpeakerWeightedPlattCalibrator]
    temperature_calibrators: dict[str, AlternativeTemperatureCalibrator]
    gate: SpecificOutputGate
    inner_oof_rows: tuple[dict[str, Any], ...]
    methods: tuple[str, ...]
    settings: dict[str, Any]
    recalibrator_path: str | None = None


def utterance_from_manifest(
    row: ManifestRow | _AcousticManifestRow,
    cache_path: str | Path,
    *,
    include_phn_sidecar: bool = True,
) -> ExperimentUtterance:
    """Load one acoustic cache, optionally deferring every PHN-derived field."""

    cached = load_logits_cache(cache_path)
    if cached.utterance_id != row.reading_event_id:
        raise ValueError("logit cache event does not match manifest")
    ids = tuple(int(value) for value in cached.canonical_ids)
    expected = tuple(PHONE_TO_CTC_ID[phone] for phone in row.canonical_phones)
    if ids != expected:
        raise ValueError("logit cache canonical sequence does not match manifest")
    realized_ids = (
        tuple(PHONE_TO_CTC_ID[phone] for phone in row.phn_model_phones)  # type: ignore[union-attr]
        if include_phn_sidecar
        else ()
    )
    return ExperimentUtterance(
        event_id=row.reading_event_id,
        speaker_id=row.speaker_id,
        speaker_group=row.speaker_group,
        canonical_phones=tuple(row.canonical_phones),
        canonical_ids=ids,
        logits=np.asarray(cached.logits, dtype=np.float64),
        realized_ids=realized_ids,
        token_labels=(
            tuple(row.stable_tokens)  # type: ignore[union-attr]
            if include_phn_sidecar
            else ()
        ),
        logits_source=str(Path(cache_path).resolve()),
    )


def _attach_phn_sidecar(
    acoustic: ExperimentUtterance,
    row: ManifestRow,
    *,
    role: str,
    held_out_patient: str,
) -> ExperimentUtterance:
    """Materialize PHN labels under an explicit training/evaluation role."""

    if role not in {"training", "evaluation"}:
        raise ValueError("PHN sidecar role must be training or evaluation")
    if acoustic.event_id != row.reading_event_id or acoustic.speaker_id != row.speaker_id:
        raise ValueError("PHN sidecar does not match its acoustic event")
    if acoustic.realized_ids or acoustic.token_labels:
        raise ValueError("PHN sidecar can only be attached to an acoustic-only event")
    if role == "training" and acoustic.speaker_id == held_out_patient:
        raise ValueError("outer held-out PHN cannot be attached for fold training")
    if role == "evaluation" and acoustic.speaker_id != held_out_patient:
        raise ValueError("evaluation PHN does not belong to the outer held-out speaker")
    return ExperimentUtterance(
        event_id=acoustic.event_id,
        speaker_id=acoustic.speaker_id,
        speaker_group=acoustic.speaker_group,
        canonical_phones=acoustic.canonical_phones,
        canonical_ids=acoustic.canonical_ids,
        logits=acoustic.logits,
        realized_ids=tuple(
            PHONE_TO_CTC_ID[phone] for phone in row.phn_model_phones
        ),
        token_labels=tuple(row.stable_tokens),
        logits_source=acoustic.logits_source,
    )


def load_experiment_utterances(
    rows: Sequence[ManifestRow | _AcousticManifestRow],
    config: Mapping[str, Any],
    *,
    include_phn_sidecar: bool = True,
) -> list[ExperimentUtterance]:
    """Join manifest events to content-bound, pickle-free raw-logit caches."""

    output: list[ExperimentUtterance] = []
    seen: set[str] = set()
    for row in sorted(rows, key=lambda value: value.reading_event_id):
        if row.reading_event_id in seen:
            raise ValueError(f"duplicate reading event: {row.reading_event_id}")
        seen.add(row.reading_event_id)
        source = logit_cache_path(dict(config), row.reading_event_id)
        if not source.is_file():
            raise FileNotFoundError(f"missing raw-logit cache: {source}")
        output.append(
            utterance_from_manifest(
                row, source, include_phn_sidecar=include_phn_sidecar
            )
        )
    return output


def _training_signature(utterance: ExperimentUtterance) -> dict[str, Any]:
    """Reduced record containing exactly the state consumed by fitting."""

    return {
        "event": utterance.event_id,
        "speaker": utterance.speaker_id,
        "group": utterance.speaker_group,
        "canonical_phones": list(utterance.canonical_phones),
        "canonical_ids": list(utterance.canonical_ids),
        "realized_ids": list(utterance.realized_ids),
        "stable_tokens": [label.to_dict() for label in utterance.token_labels],
        "logits_sha256": utterance.logits_sha256,
    }


def _checkpoint_digest(config: Mapping[str, Any]) -> str:
    path = _config_value(config, "paths", "checkpoint")
    if path and Path(str(path)).exists():
        source = Path(str(path)).resolve()
        cache_key = source.as_posix()
        if cache_key not in _PATH_HASH_CACHE:
            _PATH_HASH_CACHE[cache_key] = sha256_path(source)
        return _PATH_HASH_CACHE[cache_key]
    declared = _config_value(config, "backend", "checkpoint_sha256")
    return str(declared) if isinstance(declared, str) and len(declared) == 64 else "0" * 64


def _fold_fit_hash(
    fold: OuterFold,
    training: Sequence[ExperimentUtterance],
    config: Mapping[str, Any],
    *,
    methods: Sequence[str],
) -> str:
    signatures = [_training_signature(row) for row in training]
    return training_artifact_hash(
        stage=(
            f"phone-{EXPERIMENT_IMPLEMENTATION_VERSION}:" + ",".join(methods)
        ),
        config_sha256=_config_digest(config),
        checkpoint_sha256=_checkpoint_digest(config),
        training_rows=signatures,
    )


def _prior_records(utterances: Iterable[ExperimentUtterance]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for utterance in utterances:
        for label in utterance.token_labels:
            if not label.stable:
                continue
            realized: str | None = label.realized_phone
            if label.event_type == "deletion":
                realized = DELETION
            records.append(
                {
                    "speaker_id": utterance.speaker_id,
                    "speaker_group": utterance.speaker_group,
                    "target_phone": label.canonical_phone,
                    "realized_phone": realized,
                    "event_type": label.event_type,
                    "stable": True,
                }
            )
    return records


def _uniform_priors(*, include_deletion: bool = True) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for target in ARPABET_39:
        candidates = alternatives_for(target, ARPABET_39, include_deletion=include_deletion)
        result[target] = {candidate: 1.0 / len(candidates) for candidate in candidates}
    return result


def _fit_recalibrator(
    training: Sequence[ExperimentUtterance], config: Mapping[str, Any]
) -> CTCSequenceRecalibrator:
    rows = [
        CalibrationUtterance(
            row.event_id,
            row.speaker_id,
            np.asarray(row.logits, dtype=np.float32),
            row.realized_ids,
        )
        for row in training
    ]
    return CTCSequenceRecalibrator.fit(
        rows,
        blank_id=int(_config_value(config, "backend", "blank_id", default=0)),
        steps=int(_config_value(config, "recalibration", "steps", default=100)),
        learning_rate=float(
            _config_value(config, "recalibration", "learning_rate", default=0.02)
        ),
        l2_identity=float(
            _config_value(config, "recalibration", "identity_l2", default=1.0)
        ),
        max_utterances=int(
            _config_value(config, "recalibration", "max_utterances", default=500)
        ),
        batch_size=int(
            _config_value(config, "recalibration", "batch_size", default=32)
        ),
        seed=int(config.get("seed", 20260829)),
        device=str(_config_value(config, "backend", "device", default="auto")),
        enforce_entropy_guard=bool(
            _config_value(config, "recalibration", "enforce_entropy_guard", default=True)
        ),
    )


def _log_probs(
    utterance: ExperimentUtterance,
    recalibrator: CTCSequenceRecalibrator | None,
) -> np.ndarray:
    logits = (
        utterance.logits
        if recalibrator is None
        else recalibrator.apply(np.asarray(utterance.logits, dtype=np.float64))
    )
    return log_softmax(np.asarray(logits, dtype=np.float64), axis=1)


def _matrix(
    utterance: ExperimentUtterance,
    recalibrator: CTCSequenceRecalibrator | None,
    config: Mapping[str, Any],
    *,
    topology_correction: bool,
) -> CounterfactualMatrix:
    return counterfactual_log_probability_matrix(
        _log_probs(utterance, recalibrator),
        utterance.canonical_ids,
        blank_id=int(_config_value(config, "backend", "blank_id", default=0)),
        phone_ids=tuple(range(1, len(ARPABET_39) + 1)),
        include_deletion=True,
        topology_correction=topology_correction,
        batch_size=int(_config_value(config, "backend", "batch_size", default=256)),
        device=str(_config_value(config, "backend", "device", default="auto")),
    )


def _deletion_likelihoods(
    utterance: ExperimentUtterance,
    recalibrator: CTCSequenceRecalibrator | None,
    config: Mapping[str, Any],
) -> _DeletionLikelihoods:
    """Score only canonical plus one deletion hypothesis per position.

    This is numerically the same CTC operation used by :func:`_matrix`, with
    the unused 38 substitutions per position omitted.  In particular, no
    boundary or PHN alignment is introduced and impossible sequences retain
    their explicit ``-inf`` likelihood.
    """

    log_probs = _log_probs(utterance, recalibrator)
    canonical = tuple(int(value) for value in utterance.canonical_ids)
    hypotheses: list[list[int]] = [list(canonical)]
    hypotheses.extend(
        list(canonical[:position] + canonical[position + 1 :])
        for position in range(len(canonical))
    )
    likelihoods = ctc_log_probabilities_batch(
        log_probs,
        hypotheses,
        int(_config_value(config, "backend", "blank_id", default=0)),
        batch_size=int(_config_value(config, "backend", "batch_size", default=256)),
        device=str(_config_value(config, "backend", "device", default="auto")),
    )
    canonical_count = uniform_ctc_log_path_count(log_probs.shape[0], canonical)
    deletion_counts = np.asarray(
        [
            uniform_ctc_log_path_count(
                log_probs.shape[0], canonical[:position] + canonical[position + 1 :]
            )
            for position in range(len(canonical))
        ],
        dtype=np.float64,
    )
    canonical_lp = float(likelihoods[0])
    deletion_lp = np.asarray(likelihoods[1:], dtype=np.float64)
    canonical_corrected = (
        canonical_lp - canonical_count
        if np.isfinite(canonical_lp) and np.isfinite(canonical_count)
        else -float("inf")
    )
    deletion_corrected = np.full_like(deletion_lp, -float("inf"))
    finite = np.isfinite(deletion_lp) & np.isfinite(deletion_counts)
    deletion_corrected[finite] = deletion_lp[finite] - deletion_counts[finite]
    return _DeletionLikelihoods(
        canonical_lp,
        float(canonical_corrected),
        deletion_lp,
        deletion_corrected,
    )


def _deletions_from_matrix(matrix: CounterfactualMatrix) -> _DeletionLikelihoods:
    """Project an already-computed full matrix onto its deletion columns."""

    ids = np.asarray(matrix.candidate_ids)
    raw = np.asarray(matrix.log_probabilities)
    corrected = np.asarray(matrix.corrected_log_probabilities)
    columns = [np.flatnonzero(row == DELETION_ID) for row in ids]
    if any(len(column) != 1 for column in columns):
        raise ValueError("full counterfactual matrix lacks one deletion per position")
    positions = np.arange(ids.shape[0], dtype=np.int64)
    selected = np.asarray([int(column[0]) for column in columns], dtype=np.int64)
    return _DeletionLikelihoods(
        float(matrix.canonical_log_probability),
        float(matrix.canonical_corrected_log_probability),
        np.asarray(raw[positions, selected], dtype=np.float64),
        np.asarray(corrected[positions, selected], dtype=np.float64),
    )


def _candidate_likelihoods(
    matrix: CounterfactualMatrix, position: int
) -> dict[str, float]:
    result: dict[str, float] = {}
    for candidate_id, value in zip(
        np.asarray(matrix.candidate_ids)[position],
        np.asarray(matrix.corrected_log_probabilities)[position],
    ):
        key = DELETION if int(candidate_id) == DELETION_ID else CTC_ID_TO_PHONE[int(candidate_id)]
        result[key] = float(value)
    return result


def _score_matrix(
    matrix: CounterfactualMatrix,
    canonical_phones: Sequence[str],
    priors: Mapping[str, Mapping[str, float]],
    *,
    deletion_calibrator: DeletionCalibrator | None = None,
    allowed: Mapping[str, Sequence[str]] | None = None,
) -> list[_TokenEvidence]:
    """Score a full matrix, optionally hard-pruning only the denominator."""

    output: list[_TokenEvidence] = []
    for position, target in enumerate(canonical_phones):
        likelihoods = _candidate_likelihoods(matrix, position)
        if deletion_calibrator is not None:
            likelihoods[DELETION] = deletion_calibrator.calibrate(
                target, likelihoods[DELETION]
            )
        # Mapping insertion order is not scientific state.  In particular,
        # canonical JSON sorts object keys, so an otherwise exact fitted-fold
        # round trip changes an ARPABET-ordered prior to lexical order.  Always
        # reduce in the frozen vocabulary order; this both preserves the
        # original fresh-fit computation and makes cache reload/caller order
        # bitwise irrelevant.
        supplied = tuple(
            allowed[target] if allowed is not None else priors[target]
        )
        if len(supplied) != len(set(supplied)):
            raise ValueError(f"duplicate retained error candidate for {target}")
        frozen_order = alternatives_for(
            target, ARPABET_39, include_deletion=True
        )
        if set(supplied) - set(frozen_order):
            raise ValueError(f"unknown retained error candidate for {target}")
        retained = tuple(candidate for candidate in frozen_order if candidate in supplied)
        if not retained:
            raise ValueError(f"no retained error candidates for {target}")
        masses = np.asarray([float(priors[target][key]) for key in retained], dtype=np.float64)
        if np.any(masses <= 0) or not np.isfinite(masses).all():
            raise ValueError("retained prior mass must be finite and positive")
        masses /= masses.sum()
        log_scores = {
            key: float(likelihoods[key] + math.log(weight))
            for key, weight in zip(retained, masses)
        }
        denominator = logsumexp(list(log_scores.values()))
        canonical = float(matrix.canonical_corrected_log_probability)
        if not np.isfinite(canonical) or not np.isfinite(denominator):
            raise ValueError("non-finite canonical or error-mixture likelihood")
        full = {
            key: log_scores.get(key, -float("inf"))
            for key in alternatives_for(target, ARPABET_39, include_deletion=True)
        }
        output.append(
            _TokenEvidence(position, target, canonical - denominator, full)
        )
    return output


def _deletion_rows(
    training: Sequence[ExperimentUtterance],
    likelihoods: Mapping[str, _DeletionLikelihoods],
    *,
    topology_correction: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for utterance in training:
        scores = likelihoods[utterance.event_id]
        canonical = (
            scores.canonical_corrected_log_probability
            if topology_correction
            else scores.canonical_log_probability
        )
        deletions = (
            scores.deletion_corrected_log_probabilities
            if topology_correction
            else scores.deletion_log_probabilities
        )
        for label in utterance.token_labels:
            if not label.stable or label.event_type != "match":
                continue
            rows.append(
                {
                    "speaker_id": utterance.speaker_id,
                    "target_phone": label.canonical_phone,
                    "canonical_log_probability": canonical,
                    "deletion_log_probability": float(deletions[label.phone_index]),
                    "is_error": False,
                }
            )
    return rows


def _fit_deletion_calibrator(
    training: Sequence[ExperimentUtterance],
    likelihoods: Mapping[str, _DeletionLikelihoods],
    config: Mapping[str, Any],
    *,
    topology_correction: bool,
) -> DeletionCalibrator:
    return DeletionCalibrator.fit(
        _deletion_rows(
            training, likelihoods, topology_correction=topology_correction
        ),
        quantile=float(_config_value(config, "deletion", "quantile", default=0.9)),
        min_tokens=int(
            _config_value(config, "deletion", "min_tokens_per_phone", default=20)
        ),
        min_speakers=int(
            _config_value(config, "deletion", "min_speakers_per_phone", default=2)
        ),
    )


def _gold_event(label: StableTokenLabel) -> tuple[str, str | None]:
    if not label.stable:
        raise ValueError("unstable token cannot enter the headline evaluation")
    if label.event_type == "match":
        return "correct", label.canonical_phone
    if label.event_type == "substitution":
        if label.realized_phone is None:
            raise ValueError("stable substitution is missing its realized phone")
        return "substitution", label.realized_phone
    if label.event_type == "deletion":
        return "deletion", None
    raise ValueError(f"unsupported stable event type: {label.event_type}")


def _recalibrator_digest(recalibrator: CTCSequenceRecalibrator | None) -> str:
    if recalibrator is None:
        return "raw"
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(recalibrator.weight).tobytes())
    digest.update(np.ascontiguousarray(recalibrator.bias).tobytes())
    digest.update(str(recalibrator.blank_id).encode("ascii"))
    return digest.hexdigest()


def _without_topology(matrix: CounterfactualMatrix) -> CounterfactualMatrix:
    """Expose the same exact CTC likelihoods without subtracting path counts."""

    return CounterfactualMatrix(
        matrix.canonical_log_probability,
        matrix.canonical_log_path_count,
        matrix.canonical_log_probability,
        np.array(matrix.candidate_ids, copy=True),
        np.array(matrix.log_probabilities, copy=True),
        np.array(matrix.log_path_counts, copy=True),
        np.array(matrix.log_probabilities, copy=True),
    )


def _raw_content_key(
    utterance: ExperimentUtterance, config: Mapping[str, Any]
) -> str:
    """Content address for fold-independent official-logit calculations."""

    return sha256_json(
        {
            "algorithm": "all-phone-counterfactual.v1",
            "logits_sha256": utterance.logits_sha256,
            "canonical_ids": list(utterance.canonical_ids),
            "blank_id": int(_config_value(config, "backend", "blank_id", default=0)),
            "vocab_size": int(
                _config_value(config, "backend", "vocab_size", default=40)
            ),
            "batch_size": int(
                _config_value(config, "backend", "batch_size", default=256)
            ),
            "device": str(_config_value(config, "backend", "device", default="auto")),
        }
    )


class _SharedRawCache:
    """Fold-independent values shared across all outer/inner LOSO folds.

    Keys bind the raw-logit content and canonical sequence rather than an
    event name.  This both prevents stale reuse and lets the primary and inner
    paths share an exact raw matrix without any PHN-derived cache selector.
    """

    def __init__(self) -> None:
        self.matrices: dict[str, CounterfactualMatrix] = {}
        self.deletions: dict[str, _DeletionLikelihoods] = {}
        self.baselines: dict[tuple[str, str], Any] = {}


class _MatrixMemo:
    """Per-fold recalibrated memo plus content-addressed shared raw values."""

    def __init__(self, shared_raw: _SharedRawCache | None = None) -> None:
        self.values: dict[tuple[str, str], CounterfactualMatrix] = {}
        self.deletions: dict[tuple[str, str], _DeletionLikelihoods] = {}
        self.content_keys: dict[tuple[str, str | None, int, tuple[int, ...]], str] = {}
        self.recalibrator_digests: dict[int, str] = {}
        self.recalibrator_refs: dict[int, CTCSequenceRecalibrator] = {}
        self.shared_raw = shared_raw

    def content_key(
        self, utterance: ExperimentUtterance, config: Mapping[str, Any]
    ) -> str:
        identity = (
            utterance.event_id,
            utterance.logits_source,
            id(utterance.logits),
            utterance.canonical_ids,
        )
        if identity not in self.content_keys:
            self.content_keys[identity] = _raw_content_key(utterance, config)
        return self.content_keys[identity]

    def recalibrator_digest(
        self, recalibrator: CTCSequenceRecalibrator | None
    ) -> str:
        if recalibrator is None:
            return "raw"
        identity = id(recalibrator)
        if identity not in self.recalibrator_digests:
            self.recalibrator_digests[identity] = _recalibrator_digest(recalibrator)
            # Keep the tiny 40x40 object alive so CPython cannot recycle its
            # identity for a later inner-fold model while this memo is active.
            self.recalibrator_refs[identity] = recalibrator
        return self.recalibrator_digests[identity]

    def get(
        self,
        utterance: ExperimentUtterance,
        recalibrator: CTCSequenceRecalibrator | None,
        config: Mapping[str, Any],
    ) -> CounterfactualMatrix:
        content_key = self.content_key(utterance, config)
        if recalibrator is None and self.shared_raw is not None:
            if content_key not in self.shared_raw.matrices:
                self.shared_raw.matrices[content_key] = _matrix(
                    utterance, None, config, topology_correction=True
                )
            return self.shared_raw.matrices[content_key]
        key = (content_key, self.recalibrator_digest(recalibrator))
        if key not in self.values:
            self.values[key] = _matrix(
                utterance, recalibrator, config, topology_correction=True
            )
        return self.values[key]

    def get_deletions(
        self,
        utterance: ExperimentUtterance,
        recalibrator: CTCSequenceRecalibrator | None,
        config: Mapping[str, Any],
    ) -> _DeletionLikelihoods:
        content_key = self.content_key(utterance, config)
        if recalibrator is None and self.shared_raw is not None:
            if content_key not in self.shared_raw.deletions:
                self.shared_raw.deletions[content_key] = (
                    _deletions_from_matrix(self.shared_raw.matrices[content_key])
                    if content_key in self.shared_raw.matrices
                    else _deletion_likelihoods(utterance, None, config)
                )
            return self.shared_raw.deletions[content_key]
        key = (content_key, self.recalibrator_digest(recalibrator))
        if key not in self.deletions:
            self.deletions[key] = _deletion_likelihoods(
                utterance, recalibrator, config
            )
        return self.deletions[key]

    def raw_baseline(
        self,
        utterance: ExperimentUtterance,
        config: Mapping[str, Any],
        name: str,
    ) -> Any | None:
        if self.shared_raw is None:
            return None
        return self.shared_raw.baselines.get((self.content_key(utterance, config), name))

    def put_raw_baseline(
        self,
        utterance: ExperimentUtterance,
        config: Mapping[str, Any],
        name: str,
        value: Any,
    ) -> None:
        if self.shared_raw is not None:
            self.shared_raw.baselines[(self.content_key(utterance, config), name)] = value


def _build_priors(
    training: Sequence[ExperimentUtterance], config: Mapping[str, Any]
) -> tuple[
    dict[str, dict[str, float]],
    dict[str, dict[str, float]],
    dict[float, dict[str, dict[str, float]]],
]:
    fixed = fixed_soft_priors(
        ARPABET_39,
        include_deletion=True,
        deletion_weight=float(
            _config_value(config, "prior", "deletion_base_weight", default=0.5)
        ),
    )
    empirical = empirical_confusion_priors(
        _prior_records(training),
        ARPABET_39,
        fixed=fixed,
        pseudocount=float(
            _config_value(config, "prior", "dirichlet_pseudocount", default=20.0)
        ),
        enrichment_cap=float(
            _config_value(config, "prior", "enrichment_cap", default=4.0)
        ),
        include_deletion=True,
    )
    grid = tuple(float(value) for value in _config_value(
        config, "prior", "lambda_grid", default=LAMBDA_GRID
    ))
    if grid != LAMBDA_GRID:
        raise ValueError(f"frozen lambda grid changed: {grid}")
    mixed = {
        value: mix_soft_priors(
            fixed,
            empirical,
            value,
            uniform_epsilon=float(
                _config_value(config, "prior", "uniform_backoff", default=0.05)
            ),
        )
        for value in grid
    }
    return fixed, empirical, mixed


def _fit_deletion_contexts(
    training: Sequence[ExperimentUtterance],
    recalibrator: CTCSequenceRecalibrator,
    config: Mapping[str, Any],
    memo: _MatrixMemo,
) -> dict[str, DeletionCalibrator]:
    # q90 needs canonical and deletion likelihoods only.  Requesting the full
    # 39-column substitution matrix here would multiply CTC work by roughly
    # the phone inventory size for no change in the fitted penalty.
    rec = {
        row.event_id: memo.get_deletions(row, recalibrator, config)
        for row in training
    }
    raw = {
        row.event_id: memo.get_deletions(row, None, config) for row in training
    }
    return {
        "recalibrated_topology": _fit_deletion_calibrator(
            training, rec, config, topology_correction=True
        ),
        "raw_topology": _fit_deletion_calibrator(
            training, raw, config, topology_correction=True
        ),
        "recalibrated_raw": _fit_deletion_calibrator(
            training, rec, config, topology_correction=False
        ),
    }


def _method_evidence(
    utterance: ExperimentUtterance,
    *,
    recalibrator: CTCSequenceRecalibrator,
    fixed: Mapping[str, Mapping[str, float]],
    adapted: Mapping[str, Mapping[str, float]],
    deletion_calibrators: Mapping[str, DeletionCalibrator],
    config: Mapping[str, Any],
    memo: _MatrixMemo,
    methods: Sequence[str],
) -> dict[str, list[_TokenEvidence]]:
    """Compute every requested method on identical canonical token keys."""

    requested = set(methods)
    unknown = requested - set(ALL_METHODS)
    if unknown:
        raise ValueError(f"unknown registered methods: {sorted(unknown)}")
    values: dict[str, list[_TokenEvidence]] = {}

    def add(name: str, rows: list[_TokenEvidence]) -> None:
        if name in requested:
            values[name] = rows

    uniform_raw_consumers = {
        METHOD_UNIFORM_RAW,
        LEGACY_HARD_GRAPH_METHOD,
        CONVENTIONAL_METHOD,
        CTC_SF_SD_NORM_METHOD,
    }
    uniform_rec_consumers = {
        METHOD_UNIFORM_CALIBRATED,
        METHOD_ABLATION_UNIFORM,
        METHOD_CTC_SF_RECALIBRATED,
    }
    rec_top_consumers = uniform_rec_consumers | {
        METHOD_ADAPTED,
        METHOD_ABLATION_FIXED,
        METHOD_ABLATION_HARD,
        METHOD_ABLATION_NO_TOPOLOGY,
        METHOD_ABLATION_NO_DELETION_CALIBRATION,
        METHOD_ABLATION_NO_DELETION,
        METHOD_ABLATION_NO_ABSTENTION,
    }
    raw_top_consumers = uniform_raw_consumers | {
        METHOD_FIXED,
        METHOD_ABLATION_NO_RECALIBRATION,
    }
    rec_top = (
        memo.get(utterance, recalibrator, config)
        if requested.intersection(rec_top_consumers)
        else None
    )
    raw_top = (
        memo.get(utterance, None, config)
        if requested.intersection(raw_top_consumers)
        else None
    )
    uniform = (
        _uniform_priors(include_deletion=True)
        if requested.intersection(uniform_raw_consumers | uniform_rec_consumers)
        else None
    )

    uniform_raw: list[_TokenEvidence] | None = None
    if requested.intersection(uniform_raw_consumers):
        assert raw_top is not None and uniform is not None
        uniform_raw = _score_matrix(
            _without_topology(raw_top), utterance.canonical_phones, uniform
        )
        add(METHOD_UNIFORM_RAW, uniform_raw)

    uniform_calibrated: list[_TokenEvidence] | None = None
    if requested.intersection(uniform_rec_consumers):
        assert rec_top is not None and uniform is not None
        uniform_calibrated = _score_matrix(
            rec_top,
            utterance.canonical_phones,
            uniform,
            deletion_calibrator=deletion_calibrators["recalibrated_topology"],
        )
        add(METHOD_UNIFORM_CALIBRATED, uniform_calibrated)
        # Identical acoustic/prior score by design; only its registered label
        # differs in the ablation table.
        add(METHOD_ABLATION_UNIFORM, uniform_calibrated)

    if LEGACY_HARD_GRAPH_METHOD in requested:
        assert raw_top is not None
        hard = legacy_hard_candidate_graph(
            ARPABET_39,
            deletion_weight=float(
                _config_value(config, "prior", "deletion_base_weight", default=0.5)
            ),
        )
        add(
            LEGACY_HARD_GRAPH_METHOD,
            _score_matrix(
                _without_topology(raw_top), utterance.canonical_phones, hard
            ),
        )

    if METHOD_FIXED in requested:
        assert raw_top is not None
        # The registered no-PHN track has no learned sequence/deletion
        # transform: raw official logits, topology correction, frozen prior.
        add(METHOD_FIXED, _score_matrix(raw_top, utterance.canonical_phones, fixed))

    adapted_main: list[_TokenEvidence] | None = None
    if requested.intersection({METHOD_ADAPTED, METHOD_ABLATION_NO_ABSTENTION}):
        assert rec_top is not None
        adapted_main = _score_matrix(
            rec_top,
            utterance.canonical_phones,
            adapted,
            deletion_calibrator=deletion_calibrators["recalibrated_topology"],
        )
        add(METHOD_ADAPTED, adapted_main)
        add(METHOD_ABLATION_NO_ABSTENTION, adapted_main)

    if METHOD_ABLATION_FIXED in requested:
        assert rec_top is not None
        add(
            METHOD_ABLATION_FIXED,
            _score_matrix(
                rec_top,
                utterance.canonical_phones,
                fixed,
                deletion_calibrator=deletion_calibrators["recalibrated_topology"],
            ),
        )
    if METHOD_ABLATION_HARD in requested:
        assert rec_top is not None
        hard = legacy_hard_candidate_graph(
            ARPABET_39,
            deletion_weight=float(
                _config_value(config, "prior", "deletion_base_weight", default=0.5)
            ),
        )
        add(
            METHOD_ABLATION_HARD,
            _score_matrix(
                rec_top,
                utterance.canonical_phones,
                adapted,
                deletion_calibrator=deletion_calibrators["recalibrated_topology"],
                allowed={target: tuple(candidates) for target, candidates in hard.items()},
            ),
        )
    if METHOD_ABLATION_NO_TOPOLOGY in requested:
        assert rec_top is not None
        add(
            METHOD_ABLATION_NO_TOPOLOGY,
            _score_matrix(
                _without_topology(rec_top),
                utterance.canonical_phones,
                adapted,
                deletion_calibrator=deletion_calibrators["recalibrated_raw"],
            ),
        )
    if METHOD_ABLATION_NO_RECALIBRATION in requested:
        assert raw_top is not None
        add(
            METHOD_ABLATION_NO_RECALIBRATION,
            _score_matrix(
                raw_top,
                utterance.canonical_phones,
                adapted,
                deletion_calibrator=deletion_calibrators["raw_topology"],
            ),
        )
    if METHOD_ABLATION_NO_DELETION_CALIBRATION in requested:
        assert rec_top is not None
        add(
            METHOD_ABLATION_NO_DELETION_CALIBRATION,
            _score_matrix(rec_top, utterance.canonical_phones, adapted),
        )
    if METHOD_ABLATION_NO_DELETION in requested:
        assert rec_top is not None
        substitutions_only = {
            target: tuple(key for key in adapted[target] if key != DELETION)
            for target in adapted
        }
        add(
            METHOD_ABLATION_NO_DELETION,
            _score_matrix(
                rec_top,
                utterance.canonical_phones,
                adapted,
                allowed=substitutions_only,
            ),
        )

    # Scalar baselines still receive a complete conditional-alternative table
    # for the secondary identification metrics.  The table comes from the
    # corresponding uniform all-phone counterfactual model; its values never
    # enter the baseline's binary GoP score.
    if requested.intersection(BASELINE_METHODS):
        blank_id = int(_config_value(config, "backend", "blank_id", default=0))
        phone_ids = tuple(range(1, len(ARPABET_39) + 1))
        raw_log_probs: np.ndarray | None = None

        def get_raw_log_probs() -> np.ndarray:
            nonlocal raw_log_probs
            if raw_log_probs is None:
                raw_log_probs = _log_probs(utterance, None)
            return raw_log_probs

        if CONVENTIONAL_METHOD in requested:
            assert uniform_raw is not None
            forced = memo.raw_baseline(utterance, config, CONVENTIONAL_METHOD)
            if forced is None:
                forced = conventional_forced_alignment_gop(
                    get_raw_log_probs(),
                    utterance.canonical_ids,
                    blank_id=blank_id,
                    phone_ids=phone_ids,
                )
                memo.put_raw_baseline(
                    utterance, config, CONVENTIONAL_METHOD, forced
                )
            values[CONVENTIONAL_METHOD] = [
                _TokenEvidence(row.position, base.target, row.score, base.alternative_log_scores)
                for row, base in zip(forced, uniform_raw)
            ]
        if CTC_SF_SD_NORM_METHOD in requested:
            assert uniform_raw is not None
            sf = memo.raw_baseline(utterance, config, CTC_SF_SD_NORM_METHOD)
            if sf is None:
                sf = ctc_sf_sd_norm(
                    get_raw_log_probs(),
                    utterance.canonical_ids,
                    blank_id=blank_id,
                    phone_ids=phone_ids,
                    max_work=_config_value(config, "baselines", "ctc_sf_max_work"),
                )
                memo.put_raw_baseline(
                    utterance, config, CTC_SF_SD_NORM_METHOD, sf
                )
            values[CTC_SF_SD_NORM_METHOD] = [
                _TokenEvidence(row.position, base.target, row.normalized_score, base.alternative_log_scores)
                for row, base in zip(sf, uniform_raw)
            ]
        if METHOD_CTC_SF_RECALIBRATED in requested:
            assert uniform_calibrated is not None
            sf_recal = ctc_sf_sd_norm(
                _log_probs(utterance, recalibrator),
                utterance.canonical_ids,
                blank_id=blank_id,
                phone_ids=phone_ids,
                max_work=_config_value(config, "baselines", "ctc_sf_max_work"),
            )
            values[METHOD_CTC_SF_RECALIBRATED] = [
                _TokenEvidence(
                    row.position,
                    base.target,
                    row.normalized_score,
                    base.alternative_log_scores,
                )
                for row, base in zip(sf_recal, uniform_calibrated)
            ]

    if set(values) != requested:
        raise RuntimeError(f"method evidence inventory mismatch: {set(values) ^ requested}")
    lengths = {len(rows) for rows in values.values()}
    if lengths != {len(utterance.canonical_phones)}:
        raise RuntimeError("methods did not preserve canonical-token coverage")
    return values


def _inner_tokens(
    validation: Sequence[ExperimentUtterance],
    evidence: Mapping[str, Sequence[_TokenEvidence]],
    *,
    outer_fold: OuterFold,
    validation_speaker: str,
) -> list[_OOFToken]:
    """Join gold only after all validation predictions have been generated."""

    output: list[_OOFToken] = []
    by_method_event: Mapping[str, Mapping[str, Sequence[_TokenEvidence]]] = evidence  # type: ignore[assignment]
    for utterance in validation:
        if not utterance.token_labels:
            raise ValueError("inner OOF evaluation requires validation PHN labels")
        for method, per_event in by_method_event.items():
            rows = per_event[utterance.event_id]  # type: ignore[index]
            for label in utterance.token_labels:
                if not label.stable:
                    continue
                gold_event, gold_realized = _gold_event(label)
                row = rows[label.phone_index]
                output.append(
                    _OOFToken(
                        method,
                        outer_fold.fold_id,
                        validation_speaker,
                        utterance.speaker_id,
                        utterance.event_id,
                        label.phone_index,
                        label.canonical_phone,
                        gold_event,
                        gold_realized,
                        row,
                    )
                )
    return output


def _probabilities(
    log_scores: Mapping[str, float], temperature: float
) -> dict[str, float]:
    keys = tuple(sorted(log_scores))
    values = np.asarray([float(log_scores[key]) for key in keys], dtype=np.float64)
    if np.isnan(values).any() or np.isposinf(values).any() or not np.isfinite(values).any():
        raise ValueError("candidate log-score row is invalid")
    scaled = values / float(temperature)
    maximum = float(np.max(scaled))
    mass = np.exp(scaled - maximum)
    mass /= mass.sum()
    result = {key: float(value) for key, value in zip(keys, mass)}
    # Place the roundoff correction on the largest entry, preserving zeros for
    # hard-pruned candidates and deterministic ordering.
    winner = min(keys, key=lambda key: (-result[key], key))
    result[winner] += 1.0 - sum(result.values())
    return result


def _prediction_fields(
    evidence: _TokenEvidence,
    platt: SpeakerWeightedPlattCalibrator,
    temperature: AlternativeTemperatureCalibrator,
) -> dict[str, Any]:
    p_error = float(
        platt.predict_error_probability(np.asarray([evidence.error_evidence]))[0]
    )
    probabilities = _probabilities(
        evidence.alternative_log_scores, temperature.temperature
    )
    ranked = sorted(probabilities, key=lambda key: (-probabilities[key], key))
    top = ranked[0]
    margin = float(probabilities[top] - probabilities[ranked[1]])
    return {
        "phone_index": evidence.position,
        "target": evidence.target,
        "gop": float(evidence.gop),
        "p_error": p_error,
        "top_alt": top,
        "p_alt_given_error": float(probabilities[top]),
        "candidate_probabilities": probabilities,
        # Full prior-weighted alternative components before temperature.
        # Hard-pruned candidates are represented as JSON null rather than an
        # illegal non-finite JSON number.
        "candidate_log_scores": {
            key: float(value) if np.isfinite(value) else None
            for key, value in sorted(evidence.alternative_log_scores.items())
        },
        "top2_margin": margin,
    }


def _specific_output_fields(
    method: str,
    prediction: Mapping[str, Any],
    gate: SpecificOutputGate,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Attach eval-only selective semantics without changing raw GoP.

    The ``no_abstention`` ablation intentionally shares every acoustic score
    and calibrated probability with the headline.  Its only intervention is
    to always emit the top specific candidate for predicted errors, whereas
    the headline obeys the inner-OOF gate.  Other methods do not participate
    in this runtime-policy ablation and receive a deterministic false flag.
    """

    error_threshold = float(
        _config_value(config, "llm_export", "error_threshold", default=0.5)
    )
    predicted_error = float(prediction["p_error"]) >= error_threshold
    authorized = False
    specific_abstain = False
    if method == METHOD_ABLATION_NO_ABSTENTION and predicted_error:
        authorized = True
    elif method == METHOD_ADAPTED and predicted_error:
        authorized = bool(
            gate.enabled
            and gate.probability_threshold is not None
            and float(prediction["p_alt_given_error"]) >= gate.probability_threshold
            and float(prediction["top2_margin"]) >= gate.min_margin
        )
        specific_abstain = not authorized

    if not predicted_error:
        decision = "correct"
    elif authorized:
        decision = (
            "deletion" if prediction["top_alt"] == DELETION else "substitution"
        )
    else:
        decision = "atypical_unspecified"
    return {
        "predicted_error": predicted_error,
        "specific_output_authorized": authorized,
        # Transitional exact alias requested by the evaluation contract.
        "specific_prediction_authorized": authorized,
        "specific_abstain": specific_abstain,
        "decision": decision,
    }


def _fit_calibrators(
    rows: Sequence[_OOFToken],
    methods: Sequence[str],
    config: Mapping[str, Any],
) -> tuple[
    dict[str, SpeakerWeightedPlattCalibrator],
    dict[str, AlternativeTemperatureCalibrator],
]:
    platt: dict[str, SpeakerWeightedPlattCalibrator] = {}
    temperatures: dict[str, AlternativeTemperatureCalibrator] = {}
    for method in methods:
        local = [row for row in rows if row.method == method]
        if not local:
            raise ValueError(f"no inner OOF rows for {method}")
        if method == METHOD_FIXED:
            # A likelihood ratio with unit class odds has the closed-form
            # posterior sigmoid(-G).  Keeping coefficient=1/intercept=0 and
            # temperature=1 prevents the no-PHN track from learning from OOF
            # PHN labels at either probability-calibration stage.
            platt[method] = SpeakerWeightedPlattCalibrator(1.0, 0.0, 1.0)
            temperatures[method] = AlternativeTemperatureCalibrator(
                1.0, {1.0: 0.0}
            )
            continue
        platt[method] = SpeakerWeightedPlattCalibrator.fit(
            [row.evidence.error_evidence for row in local],
            [int(row.gold_event != "correct") for row in local],
            [row.speaker for row in local],
            c_value=float(_config_value(config, "calibration", "platt_c", default=1.0)),
        )
        error_rows = [
            row
            for row in local
            if row.gold_event in {"substitution", "deletion"}
        ]
        if not error_rows:
            raise ValueError(f"no stable substitution/deletion OOF rows for {method}")
        ordered_scores: list[list[float]] = []
        gold_indices: list[int] = []
        for row in error_rows:
            keys = tuple(sorted(row.evidence.alternative_log_scores))
            gold = DELETION if row.gold_event == "deletion" else str(row.gold_realized)
            if gold not in keys:
                raise ValueError(f"gold alternative is absent for {method}: {gold}")
            ordered_scores.append([row.evidence.alternative_log_scores[key] for key in keys])
            gold_indices.append(keys.index(gold))
        temperatures[method] = AlternativeTemperatureCalibrator.fit(
            np.asarray(ordered_scores, dtype=np.float64),
            gold_indices,
            [row.speaker for row in error_rows],
            grid=tuple(
                float(value)
                for value in _config_value(
                    config,
                    "calibration",
                    "alternative_temperature_grid",
                    default=(0.5, 0.75, 1.0, 1.5, 2.0),
                )
            ),
        )
    return platt, temperatures


def _materialize_inner_rows(
    rows: Sequence[_OOFToken],
    *,
    method: str,
    platt: SpeakerWeightedPlattCalibrator,
    temperature: AlternativeTemperatureCalibrator,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        if row.method != method:
            continue
        prediction = _prediction_fields(row.evidence, platt, temperature)
        output.append(
            {
                "fold": row.fold,
                "inner_validation_speaker": row.inner_validation_speaker,
                "speaker": row.speaker,
                "event": row.event,
                "phone_index": row.phone_index,
                "target": row.target,
                "gold_event": row.gold_event,
                "gold_realized": row.gold_realized,
                "stable": True,
                "p_error": prediction["p_error"],
                "top_alt": prediction["top_alt"],
                "p_alt_given_error": prediction["p_alt_given_error"],
                "top2_margin": prediction["top2_margin"],
            }
        )
    return output


def _fold_cache_paths(
    cache_dir: str | Path, cohort: str, fit_hash: str
) -> tuple[Path, Path, Path]:
    root = Path(cache_dir) / "folds" / cohort
    stem = root / fit_hash
    return (
        Path(f"{stem}.recalibrator.npz"),
        Path(f"{stem}.fit.json"),
        Path(f"{stem}.training-source.json"),
    )


def _fold_state_bundle_hash(
    cache_dir: str | Path, fitted: FittedFold
) -> str:
    """Bind every persisted learned component used by one outer fold."""

    recal_path, state_path, source_path = _fold_cache_paths(
        cache_dir, fitted.cohort, fitted.fit_hash
    )
    descriptor_path = companion_descriptor_path(recal_path)
    paths = {
        "fit_state": state_path,
        "recalibrator": recal_path,
        "recalibrator_descriptor": descriptor_path,
        "training_source": source_path,
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"persisted fold state is incomplete: {missing}")
    return sha256_json(
        {name: sha256_file(path) for name, path in sorted(paths.items())}
    )


def _serialize_fitted(fitted: FittedFold, config_digest: str) -> dict[str, Any]:
    return {
        "schema_version": "da-cf-gop.fitted-fold.v1",
        "cohort": fitted.cohort,
        "fold_id": fitted.fold_id,
        "held_out_patient": fitted.held_out_patient,
        "training_speakers": list(fitted.training_speakers),
        "fit_hash": fitted.fit_hash,
        "config_sha256": config_digest,
        "selected_lambda": fitted.selected_lambda,
        "lambda_metrics": {str(key): value for key, value in fitted.lambda_metrics.items()},
        "priors": fitted.priors,
        "deletion_calibrators": {
            key: {
                "per_phone": value.per_phone,
                "global_penalty": value.global_penalty,
                "quantile": value.quantile,
                "min_tokens": value.min_tokens,
                "min_speakers": value.min_speakers,
            }
            for key, value in fitted.deletion_calibrators.items()
        },
        "platt_calibrators": {
            key: {
                "coefficient": value.coefficient,
                "intercept": value.intercept,
                "c_value": value.c_value,
            }
            for key, value in fitted.platt_calibrators.items()
        },
        "temperature_calibrators": {
            key: {
                "temperature": value.temperature,
                "nll_by_temperature": {
                    str(candidate): score if np.isfinite(score) else None
                    for candidate, score in value.nll_by_temperature.items()
                },
            }
            for key, value in fitted.temperature_calibrators.items()
        },
        "gate": fitted.gate.to_dict(),
        "inner_oof_rows": list(fitted.inner_oof_rows),
        "methods": list(fitted.methods),
        "settings": fitted.settings,
    }


def _save_fitted_fold(
    fitted: FittedFold,
    training: Sequence[ExperimentUtterance],
    config: Mapping[str, Any],
    cache_dir: str | Path,
) -> None:
    recal_path, state_path, source_path = _fold_cache_paths(
        cache_dir, fitted.cohort, fitted.fit_hash
    )
    recal_path.parent.mkdir(parents=True, exist_ok=True)
    source_document = {
        "schema_version": "da-cf-gop.training-source.v1",
        "fold": fitted.fold_id,
        "fit_hash": fitted.fit_hash,
        "records": sorted(
            (_training_signature(row) for row in training),
            key=lambda value: (value["speaker"], value["event"]),
        ),
    }
    write_json(source_path, source_document)
    fitted.recalibrator.save(recal_path)
    # Serialize the complete learned state before constructing provenance so
    # the recalibrator descriptor can authenticate priors, all calibrators,
    # the gate and compact inner-OOF rows as one bound source artifact.
    write_json(state_path, _serialize_fitted(fitted, _config_digest(config)))
    fitted_state_sha256 = sha256_file(state_path)
    training_source_sha256 = sha256_file(source_path)
    descriptor = build_cache_descriptor(
        artifact_type="sequence_ctc_recalibrator",
        artifact=recal_path,
        config={"config_sha256": _config_digest(config)},
        sources={
            "fitted_state": state_path,
            "training_inputs": source_path,
        },
        fold=fitted.fold_id,
        parameters={
            "fit_hash": fitted.fit_hash,
            "blank_id": fitted.recalibrator.blank_id,
            "vocab_size": fitted.recalibrator.vocab_size,
            "training_speakers": list(fitted.training_speakers),
            "fitted_state_sha256": fitted_state_sha256,
            "training_source_sha256": training_source_sha256,
        },
        upstream_sha256={
            "fitted_state": fitted_state_sha256,
            "training_binding": fitted.fit_hash,
            "training_source": training_source_sha256,
        },
    )
    write_json(companion_descriptor_path(recal_path), descriptor)


def _load_fitted_fold(
    *,
    fold: OuterFold,
    fit_hash: str,
    config: Mapping[str, Any],
    cache_dir: str | Path,
    methods: Sequence[str],
) -> FittedFold | None:
    recal_path, state_path, _source_path = _fold_cache_paths(
        cache_dir, fold.cohort, fit_hash
    )
    descriptor_path = companion_descriptor_path(recal_path)
    if not (recal_path.is_file() and state_path.is_file() and descriptor_path.is_file()):
        return None
    try:
        descriptor = read_json(descriptor_path)
        if not isinstance(descriptor, Mapping):
            return None
        validate_cache_descriptor(
            descriptor,
            expected_config={"config_sha256": _config_digest(config)},
            expected_artifact_type="sequence_ctc_recalibrator",
            expected_fold=fold.fold_id,
            required_source_roles=("fitted_state", "training_inputs"),
        )
        sources = descriptor.get("sources")
        parameters = descriptor.get("parameters")
        upstream = descriptor.get("upstream_sha256")
        if not all(isinstance(value, Mapping) for value in (sources, parameters, upstream)):
            return None
        fitted_source = sources.get("fitted_state")
        training_source = sources.get("training_inputs")
        if not isinstance(fitted_source, Mapping) or not isinstance(
            training_source, Mapping
        ):
            return None
        fitted_state_sha256 = sha256_file(state_path)
        training_source_sha256 = sha256_file(_source_path)
        if (
            fitted_source.get("sha256") != fitted_state_sha256
            or training_source.get("sha256") != training_source_sha256
            or parameters.get("fitted_state_sha256") != fitted_state_sha256
            or parameters.get("training_source_sha256") != training_source_sha256
            or upstream.get("fitted_state") != fitted_state_sha256
            or upstream.get("training_source") != training_source_sha256
        ):
            return None
        state = read_json(state_path)
        if not isinstance(state, Mapping):
            return None
        if (
            state.get("schema_version") != "da-cf-gop.fitted-fold.v1"
            or state.get("fit_hash") != fit_hash
            or state.get("config_sha256") != _config_digest(config)
            or tuple(state.get("methods", ())) != tuple(methods)
        ):
            return None
        recalibrator = CTCSequenceRecalibrator.load(recal_path)
        deletion = {
            str(key): DeletionCalibrator(
                {str(phone): float(score) for phone, score in value["per_phone"].items()},
                float(value["global_penalty"]),
                float(value["quantile"]),
                int(value["min_tokens"]),
                int(value["min_speakers"]),
            )
            for key, value in state["deletion_calibrators"].items()
        }
        platt = {
            str(key): SpeakerWeightedPlattCalibrator(
                float(value["coefficient"]),
                float(value["intercept"]),
                float(value["c_value"]),
            )
            for key, value in state["platt_calibrators"].items()
        }
        temperatures = {
            str(key): AlternativeTemperatureCalibrator(
                float(value["temperature"]),
                {
                    float(candidate): (
                        float(score) if score is not None else float("inf")
                    )
                    for candidate, score in value["nll_by_temperature"].items()
                },
            )
            for key, value in state["temperature_calibrators"].items()
        }
        gate = SpecificOutputGate(**state["gate"])
        priors = {
            str(role): {
                str(target): {
                    str(candidate): float(weight)
                    for candidate, weight in alternatives.items()
                }
                for target, alternatives in values.items()
            }
            for role, values in state["priors"].items()
        }
        return FittedFold(
            cohort=str(state["cohort"]),
            fold_id=str(state["fold_id"]),
            held_out_patient=str(state["held_out_patient"]),
            training_speakers=tuple(str(value) for value in state["training_speakers"]),
            fit_hash=fit_hash,
            selected_lambda=float(state["selected_lambda"]),
            lambda_metrics={
                float(key): float(value) for key, value in state["lambda_metrics"].items()
            },
            recalibrator=recalibrator,
            priors=priors,
            deletion_calibrators=deletion,
            platt_calibrators=platt,
            temperature_calibrators=temperatures,
            gate=gate,
            inner_oof_rows=tuple(dict(value) for value in state["inner_oof_rows"]),
            methods=tuple(str(value) for value in state["methods"]),
            settings=dict(state["settings"]),
            recalibrator_path=str(recal_path.resolve()),
        )
    except (OSError, ValueError, KeyError, TypeError):
        # Invalid cache state is never used.  The caller performs a clean,
        # deterministic refit; there is no scientific fallback model.
        return None


def fit_outer_fold(
    fold: OuterFold,
    utterances: Sequence[ExperimentUtterance],
    config: Mapping[str, Any],
    *,
    methods: Sequence[str] = ALL_METHODS,
    cache_dir: str | Path | None = None,
    progress: Progress = lambda _message: None,
    shared_raw_cache: _SharedRawCache | None = None,
) -> FittedFold:
    """Fit one outer fold using training speakers and inner patient OOF only.

    Rows belonging to the outer held-out patient are ignored before any field
    other than ``speaker_id`` is inspected.  Consequently, changing their PHN
    labels (or even their logits) cannot change the returned fit hash or model.
    """

    method_tuple = tuple(methods)
    if not method_tuple or len(method_tuple) != len(set(method_tuple)):
        raise ValueError("methods must be a non-empty unique sequence")
    if set(method_tuple) - set(ALL_METHODS):
        raise ValueError("method sequence contains an unregistered method")
    if METHOD_ADAPTED not in method_tuple:
        raise ValueError("DA-CF adapted must be present for frozen lambda selection")
    training_speaker_set = set(fold.training_patients) | set(fold.healthy_references)
    training = sorted(
        (row for row in utterances if row.speaker_id in training_speaker_set),
        key=lambda row: (row.speaker_id, row.event_id),
    )
    observed = {row.speaker_id for row in training}
    missing = training_speaker_set - observed
    if missing:
        raise ValueError(f"fold is missing training speakers: {sorted(missing)}")
    if any(not row.realized_ids or not row.token_labels for row in training):
        raise ValueError("fold fitting requires PHN sequences and stable-token sidecars")
    fit_hash = _fold_fit_hash(fold, training, config, methods=method_tuple)
    if cache_dir is not None:
        cached = _load_fitted_fold(
            fold=fold,
            fit_hash=fit_hash,
            config=config,
            cache_dir=cache_dir,
            methods=method_tuple,
        )
        if cached is not None:
            progress(f"{fold.fold_id}: reused training-only fold fit {fit_hash[:12]}")
            return cached

    progress(f"{fold.fold_id}: fitting {len(fold.inner_folds)} inner LOSO models")
    memo = _MatrixMemo(shared_raw_cache)
    adapted_oof_by_lambda: dict[float, list[_OOFToken]] = {
        value: [] for value in LAMBDA_GRID
    }
    independent_oof: list[_OOFToken] = []
    inner_contexts: list[dict[str, Any]] = []
    independent_methods = tuple(
        method for method in method_tuple if method not in LAMBDA_DEPENDENT_METHODS
    )
    deferred_dependent_methods = tuple(
        method
        for method in method_tuple
        if method in LAMBDA_DEPENDENT_METHODS and method != METHOD_ADAPTED
    )
    for inner_index, inner in enumerate(fold.inner_folds, start=1):
        inner_train_speakers = set(inner.training_patients) | set(inner.healthy_references)
        inner_training = [row for row in training if row.speaker_id in inner_train_speakers]
        validation = [row for row in training if row.speaker_id == inner.held_out_patient]
        if not validation or {row.speaker_id for row in inner_training} != inner_train_speakers:
            raise ValueError(f"inner fold has incomplete speaker data: {inner.held_out_patient}")
        recalibrator = _fit_recalibrator(inner_training, config)
        fixed, _empirical, priors_by_lambda = _build_priors(inner_training, config)
        deletion = _fit_deletion_contexts(inner_training, recalibrator, config, memo)
        independent_events: dict[str, dict[str, list[_TokenEvidence]]] = {
            method: {} for method in independent_methods
        }
        adapted_events_by_lambda: dict[
            float, dict[str, dict[str, list[_TokenEvidence]]]
        ] = {
            value: {METHOD_ADAPTED: {}} for value in LAMBDA_GRID
        }
        # Gold is intentionally not read until every validation utterance has
        # produced all lambda-selection predictions.  Only DA-CF adapted is
        # needed to choose lambda; all other lambda-dependent ablations are
        # materialized once, after selection, from these same cached matrices.
        for row in validation:
            acoustic = row.acoustic_only()
            if independent_methods:
                independent = _method_evidence(
                    acoustic,
                    recalibrator=recalibrator,
                    fixed=fixed,
                    adapted=priors_by_lambda[LAMBDA_GRID[0]],
                    deletion_calibrators=deletion,
                    config=config,
                    memo=memo,
                    methods=independent_methods,
                )
                for method, tokens in independent.items():
                    independent_events[method][row.event_id] = tokens
            for lambda_value in LAMBDA_GRID:
                adapted_evidence = _method_evidence(
                    acoustic,
                    recalibrator=recalibrator,
                    fixed=fixed,
                    adapted=priors_by_lambda[lambda_value],
                    deletion_calibrators=deletion,
                    config=config,
                    memo=memo,
                    methods=(METHOD_ADAPTED,),
                )
                adapted_events_by_lambda[lambda_value][METHOD_ADAPTED][
                    row.event_id
                ] = adapted_evidence[METHOD_ADAPTED]
        if independent_methods:
            independent_oof.extend(
                _inner_tokens(
                    validation,
                    independent_events,
                    outer_fold=fold,
                    validation_speaker=inner.held_out_patient,
                )
            )
        for lambda_value in LAMBDA_GRID:
            adapted_oof_by_lambda[lambda_value].extend(
                _inner_tokens(
                    validation,
                    adapted_events_by_lambda[lambda_value],
                    outer_fold=fold,
                    validation_speaker=inner.held_out_patient,
                )
            )
        inner_contexts.append(
            {
                "validation": validation,
                "validation_speaker": inner.held_out_patient,
                "recalibrator": recalibrator,
                "fixed": fixed,
                "priors_by_lambda": priors_by_lambda,
                "deletion": deletion,
            }
        )
        progress(
            f"{fold.fold_id}: inner {inner_index}/{len(fold.inner_folds)} "
            f"held out {inner.held_out_patient}"
        )

    adapted_by_lambda = {
        lambda_value: [
            row.evidence.error_evidence
            for row in rows
        ]
        for lambda_value, rows in adapted_oof_by_lambda.items()
    }
    reference_rows = adapted_oof_by_lambda[LAMBDA_GRID[0]]
    selection = select_lambda_by_macro_auprc(
        adapted_by_lambda,
        [int(row.gold_event != "correct") for row in reference_rows],
        [row.speaker for row in reference_rows],
    )
    selected_oof = list(independent_oof)
    selected_oof.extend(adapted_oof_by_lambda[selection.lambda_value])
    if deferred_dependent_methods:
        for context in inner_contexts:
            deferred_events = {
                method: {} for method in deferred_dependent_methods
            }
            validation = context["validation"]
            for row in validation:
                deferred = _method_evidence(
                    row.acoustic_only(),
                    recalibrator=context["recalibrator"],
                    fixed=context["fixed"],
                    adapted=context["priors_by_lambda"][selection.lambda_value],
                    deletion_calibrators=context["deletion"],
                    config=config,
                    memo=memo,
                    methods=deferred_dependent_methods,
                )
                for method, tokens in deferred.items():
                    deferred_events[method][row.event_id] = tokens
            selected_oof.extend(
                _inner_tokens(
                    validation,
                    deferred_events,
                    outer_fold=fold,
                    validation_speaker=context["validation_speaker"],
                )
            )
    platt, temperatures = _fit_calibrators(selected_oof, method_tuple, config)
    compact_oof = _materialize_inner_rows(
        selected_oof,
        method=METHOD_ADAPTED,
        platt=platt[METHOD_ADAPTED],
        temperature=temperatures[METHOD_ADAPTED],
    )
    gate = fit_specific_output_gate(
        compact_oof,
        min_precision=float(
            _config_value(config, "llm_export", "precision_floor", default=0.8)
        ),
        min_coverage=float(
            _config_value(config, "llm_export", "coverage_floor", default=0.1)
        ),
        min_margin=float(
            _config_value(config, "llm_export", "alternative_margin_floor", default=0.1)
        ),
        threshold_grid=tuple(
            float(value)
            for value in _config_value(
                config,
                "llm_export",
                "threshold_grid",
                default=(0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95),
            )
        ),
    )

    progress(f"{fold.fold_id}: fitting final outer-training recalibrator")
    final_recalibrator = _fit_recalibrator(training, config)
    final_fixed, final_empirical, final_by_lambda = _build_priors(training, config)
    final_deletion = _fit_deletion_contexts(training, final_recalibrator, config, memo)
    recal_path = (
        str(_fold_cache_paths(cache_dir, fold.cohort, fit_hash)[0].resolve())
        if cache_dir is not None
        else None
    )
    fitted = FittedFold(
        cohort=fold.cohort,
        fold_id=fold.fold_id,
        held_out_patient=fold.held_out_patient,
        training_speakers=tuple(sorted(training_speaker_set)),
        fit_hash=fit_hash,
        selected_lambda=selection.lambda_value,
        lambda_metrics=dict(selection.per_lambda),
        recalibrator=final_recalibrator,
        priors={
            "fixed": final_fixed,
            "empirical": final_empirical,
            "adapted": final_by_lambda[selection.lambda_value],
            "uniform": _uniform_priors(include_deletion=True),
        },
        deletion_calibrators=final_deletion,
        platt_calibrators=platt,
        temperature_calibrators=temperatures,
        gate=gate,
        inner_oof_rows=tuple(compact_oof),
        methods=method_tuple,
        settings={
            "experiment_implementation_version": EXPERIMENT_IMPLEMENTATION_VERSION,
            "blank_id": int(_config_value(config, "backend", "blank_id", default=0)),
            "vocab_size": int(_config_value(config, "backend", "vocab_size", default=40)),
            "selected_lambda": selection.lambda_value,
            "lambda_selection_population": "inner_patient_loso_stable_tokens",
            "platt_population": "inner_patient_oof_only",
            "external_alignment_at_inference": False,
            "phn_at_inference": False,
        },
        recalibrator_path=recal_path,
    )
    if cache_dir is not None:
        _save_fitted_fold(fitted, training, config, cache_dir)
    return fitted


def _counterfactual_cache_path(
    cache_dir: str | Path,
    cohort: str,
    fit_hash: str,
    event_id: str,
) -> Path:
    event_hash = hashlib.sha256(event_id.encode("utf-8")).hexdigest()[:24]
    return (
        Path(cache_dir)
        / "counterfactual"
        / cohort
        / fit_hash
        / f"{event_hash}.npz"
    )


def _artifact_with_descriptor_hash(path: str | Path) -> str:
    artifact = Path(path)
    descriptor = companion_descriptor_path(artifact)
    if not artifact.is_file() or not descriptor.is_file():
        raise FileNotFoundError(f"artifact/descriptor pair is incomplete: {artifact}")
    return sha256_json(
        {
            "artifact": sha256_file(artifact),
            "descriptor": sha256_file(descriptor),
        }
    )


def _persist_counterfactual_matrix(
    utterance: ExperimentUtterance,
    fitted: FittedFold,
    matrix: CounterfactualMatrix,
    config: Mapping[str, Any],
    cache_dir: str | Path,
) -> Path | None:
    """Persist the headline [position, 39-candidate] outer matrix and binding."""

    if not utterance.logits_source or not fitted.recalibrator_path:
        return None
    source_logits = Path(utterance.logits_source)
    source_recal = Path(fitted.recalibrator_path)
    if not source_logits.is_file() or not source_recal.is_file():
        return None
    destination = _counterfactual_cache_path(
        cache_dir,
        fitted.cohort,
        fitted.fit_hash,
        utterance.event_id,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_deterministic_npz(
        destination,
        {
            "canonical_log_probability": np.asarray(
                matrix.canonical_log_probability, dtype=np.float64
            ),
            "canonical_log_path_count": np.asarray(
                matrix.canonical_log_path_count, dtype=np.float64
            ),
            "canonical_corrected_log_probability": np.asarray(
                matrix.canonical_corrected_log_probability, dtype=np.float64
            ),
            "candidate_ids": np.asarray(matrix.candidate_ids, dtype=np.int64),
            "log_probabilities": np.asarray(matrix.log_probabilities, dtype=np.float64),
            "log_path_counts": np.asarray(matrix.log_path_counts, dtype=np.float64),
            "corrected_log_probabilities": np.asarray(
                matrix.corrected_log_probabilities, dtype=np.float64
            ),
        },
    )
    descriptor = build_cache_descriptor(
        artifact_type="counterfactual_sequence_matrix",
        artifact=destination,
        config={"config_sha256": _config_digest(config)},
        # The raw cache path can legitimately be regenerated in place.  Its
        # immutable content digest is therefore an upstream binding, while the
        # content-addressed fold recalibrator is the descriptor's live source.
        sources={"fold_recalibrator": source_recal},
        fold=fitted.fold_id,
        parameters={
            "fit_hash": fitted.fit_hash,
            "reading_event_sha256": hashlib.sha256(
                utterance.event_id.encode("utf-8")
            ).hexdigest(),
            "shape": list(np.asarray(matrix.candidate_ids).shape),
            "candidate_count_per_position": 39,
            "topology_correction": True,
            "external_alignment_used": False,
        },
        upstream_sha256={
            "fold_fit": fitted.fit_hash,
            "raw_logits": utterance.logits_sha256,
        },
    )
    write_json(companion_descriptor_path(destination), descriptor)
    return destination


def predict_with_fitted_fold(
    fitted: FittedFold,
    utterances: Sequence[ExperimentUtterance],
    config: Mapping[str, Any],
    *,
    cache_dir: str | Path | None = None,
    shared_raw_cache: _SharedRawCache | None = None,
) -> list[dict[str, Any]]:
    """Predict outer utterances from audio logits + prompt canonical only.

    Any attached ``realized_ids`` or ``token_labels`` are discarded before
    scoring.  The returned rows contain no gold label and cover every canonical
    position for every registered method.
    """

    memo = _MatrixMemo(shared_raw_cache)
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for supplied in sorted(utterances, key=lambda row: row.event_id):
        if supplied.speaker_id != fitted.held_out_patient:
            raise ValueError(
                f"outer predictor expected {fitted.held_out_patient}, got {supplied.speaker_id}"
            )
        if supplied.event_id in seen:
            raise ValueError(f"duplicate outer event: {supplied.event_id}")
        seen.add(supplied.event_id)
        utterance = supplied.acoustic_only()
        evidence = _method_evidence(
            utterance,
            recalibrator=fitted.recalibrator,
            fixed=fitted.priors["fixed"],
            adapted=fitted.priors["adapted"],
            deletion_calibrators=fitted.deletion_calibrators,
            config=config,
            memo=memo,
            methods=fitted.methods,
        )
        for method in fitted.methods:
            for token in evidence[method]:
                prediction_fields = _prediction_fields(
                    token,
                    fitted.platt_calibrators[method],
                    fitted.temperature_calibrators[method],
                )
                output.append(
                    {
                        "method": method,
                        "cohort": fitted.cohort,
                        "fold": fitted.fold_id,
                        "speaker": utterance.speaker_id,
                        "event": utterance.event_id,
                        **prediction_fields,
                        **_specific_output_fields(
                            method, prediction_fields, fitted.gate, config
                        ),
                    }
                )
        if cache_dir is not None:
            headline_matrix = memo.get(utterance, fitted.recalibrator, config)
            _persist_counterfactual_matrix(
                utterance, fitted, headline_matrix, config, cache_dir
            )
    return output


def _join_outer_gold(
    predictions: Sequence[Mapping[str, Any]],
    held_out: Sequence[ExperimentUtterance],
    event_metadata: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Create stable-only evaluation rows after outer prediction is complete."""

    labels: dict[tuple[str, int], StableTokenLabel] = {}
    for utterance in held_out:
        if not utterance.token_labels:
            raise ValueError("outer evaluation requires the held-out PHN sidecar")
        for label in utterance.token_labels:
            labels[(utterance.event_id, label.phone_index)] = label
    output: list[dict[str, Any]] = []
    for prediction in predictions:
        label = labels[(str(prediction["event"]), int(prediction["phone_index"]))]
        if not label.stable:
            continue
        event, realized = _gold_event(label)
        row = dict(prediction)
        row.update(
            {
                "gold_event": event,
                "gold_realized": realized,
                "stable": True,
            }
        )
        if event_metadata is not None:
            metadata = event_metadata.get(str(prediction["event"]))
            if metadata is None:
                raise ValueError("outer event lacks frozen supplemental metadata")
            # Eval-only stratification fields.  The runtime source is built
            # from a strict whitelist before/independently of this gold join.
            row.update(
                {
                    "audio_microphone": metadata["audio_microphone"],
                    "phn_microphone": metadata["phn_microphone"],
                    "severity_rank": metadata["severity_rank"],
                    "severity_label": metadata["severity_label"],
                }
            )
        output.append(row)
    return output


def _finalize_deferred_outer_predictions(
    deferred: Sequence[_DeferredOuterPredictions],
    acoustic_by_event: Mapping[str, ExperimentUtterance],
    manifest_by_event: Mapping[str, ManifestRow],
    event_metadata: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Attach held-out evaluation PHN only after cohort prediction completes."""

    if not deferred:
        raise ValueError("no outer predictions are available for finalization")
    output: list[dict[str, Any]] = []
    seen_folds: set[str] = set()
    for pending in deferred:
        if pending.fold_id in seen_folds:
            raise ValueError(f"duplicate deferred outer fold: {pending.fold_id}")
        seen_folds.add(pending.fold_id)
        held_out: list[ExperimentUtterance] = []
        for event_id in pending.event_ids:
            acoustic = acoustic_by_event[event_id]
            if acoustic.realized_ids or acoustic.token_labels:
                raise ValueError("outer acoustic pool acquired PHN before finalization")
            row = manifest_by_event[event_id]
            if row.speaker_id != pending.held_out_patient:
                raise ValueError("deferred held-out event changed speaker")
            held_out.append(
                _attach_phn_sidecar(
                    acoustic,
                    row,
                    role="evaluation",
                    held_out_patient=pending.held_out_patient,
                )
            )
        output.extend(_join_outer_gold(pending.rows, held_out, event_metadata))
    return output


def _runtime_source_rows(
    predictions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Whitelist the private fields needed to build local-only LLM evidence."""

    allowed = (
        "cohort",
        "fold",
        "speaker",
        "event",
        "phone_index",
        "target",
        "p_error",
        "top_alt",
        "p_alt_given_error",
        "top2_margin",
    )
    return [
        {key: row[key] for key in allowed}
        for row in predictions
        if row["method"] == METHOD_ADAPTED
    ]


def _flatten_metrics(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in report["results"]:
        row: dict[str, Any] = {
            "cohort": result["cohort"],
            "method": result["method"],
            "n_speakers": result["n_speakers"],
            "n_stable_tokens": result["n_stable_tokens"],
        }
        for name, entry in result["macro"].items():
            row[f"macro_{name}"] = entry["value"]
            row[f"macro_{name}_n_speakers"] = entry["n_valid_speakers"]
        rows.append(row)
    return rows


def _flatten_speaker_metrics(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return one CSV-safe row for every method/speaker metric record."""

    rows: list[dict[str, Any]] = []
    for result in report["results"]:
        for speaker_metrics in result["per_speaker"]:
            row: dict[str, Any] = {
                "cohort": result["cohort"],
                "method": result["method"],
            }
            for key, value in speaker_metrics.items():
                row[key] = (
                    json.dumps(value, ensure_ascii=False, sort_keys=True)
                    if isinstance(value, (Mapping, list, tuple))
                    else value
                )
            rows.append(row)
    return rows


def _metric_group(report: Mapping[str, Any], method: str) -> Mapping[str, Any]:
    matches = [result for result in report["results"] if result["method"] == method]
    if len(matches) != 1:
        raise ValueError(f"metrics report lacks exactly one group for {method}")
    return matches[0]


def _difference_or_none(left: Any, right: Any) -> float | None:
    return None if left is None or right is None else float(left) - float(right)


def _all_paired_comparisons(
    report: Mapping[str, Any], config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Compare headline AUPRC with every other predeclared method by speaker."""

    adapted = _metric_group(report, METHOD_ADAPTED)
    adapted_values = {
        str(row["speaker"]): float(row["auprc"])
        for row in adapted["per_speaker"]
        if row["auprc"] is not None
    }
    output: list[dict[str, Any]] = []
    for result in sorted(report["results"], key=lambda row: str(row["method"])):
        comparator = str(result["method"])
        if comparator == METHOD_ADAPTED:
            continue
        other_values = {
            str(row["speaker"]): float(row["auprc"])
            for row in result["per_speaker"]
            if row["auprc"] is not None
        }
        common = sorted(set(adapted_values) & set(other_values))
        if not common:
            raise ValueError(f"no paired speaker AUPRC values for {comparator}")
        left = {speaker: adapted_values[speaker] for speaker in common}
        right = {speaker: other_values[speaker] for speaker in common}
        differences = {
            speaker: left[speaker] - right[speaker] for speaker in common
        }
        output.append(
            {
                "adapted_method": METHOD_ADAPTED,
                "comparator": comparator,
                "metric": "speaker_auprc",
                "per_speaker_difference": differences,
                "n_improved": sum(value > 0 for value in differences.values()),
                "bootstrap": paired_speaker_bootstrap(
                    left,
                    right,
                    n_bootstrap=int(
                        _config_value(
                            config,
                            "evaluation",
                            "bootstrap_replicates",
                            default=10_000,
                        )
                    ),
                    seed=int(config.get("seed", 20260829)),
                ),
                "exact_sign_flip": exact_sign_flip_test(differences),
            }
        )
    return output


def _paired_and_success(
    report: Mapping[str, Any], config: Mapping[str, Any], cohort: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    comparator_name = str(
        _config_value(
            config, "evaluation", "primary_comparator", default=METHOD_UNIFORM_CALIBRATED
        )
    )
    adapted = _metric_group(report, METHOD_ADAPTED)
    comparator = _metric_group(report, comparator_name)
    adapted_speakers = {
        row["speaker"]: row["auprc"]
        for row in adapted["per_speaker"]
        if row["auprc"] is not None
    }
    comparator_speakers = {
        row["speaker"]: row["auprc"]
        for row in comparator["per_speaker"]
        if row["auprc"] is not None
    }
    bootstrap = paired_speaker_bootstrap(
        adapted_speakers,
        comparator_speakers,
        n_bootstrap=int(
            _config_value(config, "evaluation", "bootstrap_replicates", default=10_000)
        ),
        seed=int(config.get("seed", 20260829)),
    )
    common = sorted(set(adapted_speakers) & set(comparator_speakers))
    improvements = {
        speaker: adapted_speakers[speaker] - comparator_speakers[speaker]
        for speaker in common
    }
    sign_flip = exact_sign_flip_test(improvements)
    paired = {
        "schema_version": "da-cf-gop.paired-comparison.v1",
        "adapted_method": METHOD_ADAPTED,
        "comparator": comparator_name,
        "metric": "speaker_auprc",
        "per_speaker_difference": improvements,
        "n_improved": sum(value > 0 for value in improvements.values()),
        "bootstrap": bootstrap,
        "exact_sign_flip": sign_flip,
    }

    adapted_macro = adapted["macro"]
    comparator_macro = comparator["macro"]
    auprc_difference = _difference_or_none(
        adapted_macro["auprc"]["value"], comparator_macro["auprc"]["value"]
    )
    auroc_difference = _difference_or_none(
        adapted_macro["auroc"]["value"], comparator_macro["auroc"]["value"]
    )
    brier_difference = _difference_or_none(
        adapted_macro["brier"]["value"], comparator_macro["brier"]["value"]
    )
    top1_difference = _difference_or_none(
        adapted_macro["substitution_top1"]["value"],
        comparator_macro["substitution_top1"]["value"],
    )
    deletion_difference = _difference_or_none(
        adapted_macro["deletion_f1"]["value"],
        comparator_macro["deletion_f1"]["value"],
    )
    all_auprc = [
        result["macro"]["auprc"]["value"]
        for result in report["results"]
        if result["macro"]["auprc"]["value"] is not None
    ]
    adapted_auprc = adapted_macro["auprc"]["value"]
    best_controlled = (
        adapted_auprc is not None
        and bool(all_auprc)
        and float(adapted_auprc) >= max(float(value) for value in all_auprc) - 1e-12
    )
    required_improved = int(
        _config_value(
            config,
            "evaluation",
            "required_primary_speakers_improved"
            if cohort == "primary5"
            else "required_sensitivity_speakers_improved",
            default=4 if cohort == "primary5" else 5,
        )
    )
    checks = {
        "adapted_is_highest_controlled_macro_auprc": bool(best_controlled),
        "auprc_absolute_improvement": (
            auprc_difference is not None
            and auprc_difference
            >= float(
                _config_value(
                    config,
                    "evaluation",
                    "required_absolute_improvement",
                    default=0.03,
                )
                if cohort == "primary5"
                else 0.0
            )
        ),
        "required_speakers_improved": paired["n_improved"] >= required_improved,
        "auroc_non_degradation": (
            auroc_difference is not None
            and auroc_difference
            >= -float(
                _config_value(
                    config, "evaluation", "maximum_auroc_drop", default=0.01
                )
            )
        ),
        "brier_non_degradation": (
            brier_difference is not None
            and brier_difference
            <= float(
                _config_value(
                    config, "evaluation", "maximum_brier_increase", default=0.01
                )
            )
        ),
        "substitution_top1_non_degradation": (
            top1_difference is not None
            and top1_difference
            >= -float(
                _config_value(
                    config,
                    "evaluation",
                    "maximum_identification_drop",
                    default=0.01,
                )
            )
        ),
        "deletion_f1_non_degradation": (
            deletion_difference is not None
            and deletion_difference
            >= -float(
                _config_value(
                    config,
                    "evaluation",
                    "maximum_identification_drop",
                    default=0.01,
                )
            )
        ),
    }
    # Sensitivity is directional and cannot rescue a failed primary result.
    required_checks = (
        checks
        if cohort == "primary5"
        else {
            "auprc_direction_positive": auprc_difference is not None
            and auprc_difference > 0,
            "required_speakers_improved": checks["required_speakers_improved"],
        }
    )
    success = {
        "schema_version": "da-cf-gop.success-assessment.v1",
        "cohort": cohort,
        "adapted_macro_auprc": adapted_auprc,
        "comparator_macro_auprc": comparator_macro["auprc"]["value"],
        "auprc_difference": auprc_difference,
        "auroc_difference": auroc_difference,
        "brier_difference": brier_difference,
        "substitution_top1_difference": top1_difference,
        "deletion_f1_difference": deletion_difference,
        "checks": checks,
        "cohort_success": all(bool(value) for value in required_checks.values()),
        "negative_result_frozen_if_false": True,
        "test_set_driven_retuning_permitted": False,
    }
    return paired, success


def _write_final_assessment_if_ready(layout: ArtifactLayout) -> Path | None:
    """Freeze the permitted claim once both predeclared cohorts exist."""

    primary_path = layout.evaluation / "primary5" / "metrics.json"
    sensitivity_path = layout.evaluation / "sensitivity7" / "metrics.json"
    if not (primary_path.is_file() and sensitivity_path.is_file()):
        return None
    primary = read_json(primary_path)
    sensitivity = read_json(sensitivity_path)
    if not isinstance(primary, Mapping) or not isinstance(sensitivity, Mapping):
        raise ValueError("cohort metrics artifacts must be JSON objects")
    if (
        not isinstance(primary.get("config_sha256"), str)
        or primary.get("config_sha256") != sensitivity.get("config_sha256")
    ):
        raise ValueError("cross-cohort metrics do not share one frozen configuration")
    primary_success = bool(
        primary["success_assessment"]["cohort_success"]  # type: ignore[index]
    )
    sensitivity_success = sensitivity["success_assessment"]  # type: ignore[index]
    sensitivity_paired = sensitivity["paired_comparison"]  # type: ignore[index]
    sensitivity_positive = bool(
        sensitivity_success["auprc_difference"] is not None
        and float(sensitivity_success["auprc_difference"]) > 0.0
    )
    sensitivity_five_of_seven = int(sensitivity_paired["n_improved"]) >= 5
    ci_above_zero = float(sensitivity_paired["bootstrap"]["ci_low"]) > 0.0
    exact_below_point_zero_five = (
        float(sensitivity_paired["exact_sign_flip"]["p_two_sided"]) < 0.05
    )
    strictly_supported = bool(
        primary_success
        and sensitivity_positive
        and sensitivity_five_of_seven
        and ci_above_zero
        and exact_below_point_zero_five
    )
    destination = layout.evaluation / "final_assessment.json"
    write_json(
        destination,
        {
            "schema_version": "da-cf-gop.final-assessment.v1",
            "primary5_success": primary_success,
            "sensitivity7_direction_positive": sensitivity_positive,
            "sensitivity7_at_least_5_of_7_improved": sensitivity_five_of_seven,
            "sensitivity7_n_improved": int(sensitivity_paired["n_improved"]),
            "sensitivity7_paired_bootstrap_ci_low": float(
                sensitivity_paired["bootstrap"]["ci_low"]
            ),
            "sensitivity7_exact_sign_flip_p_two_sided": float(
                sensitivity_paired["exact_sign_flip"]["p_two_sided"]
            ),
            "strictly_supported_superiority": strictly_supported,
            "permitted_wording": (
                "statistically_supported_superiority"
                if strictly_supported
                else "point_estimate_or_negative_result"
            ),
            "sensitivity7_cannot_rescue_primary5": True,
            "test_set_driven_retuning_permitted": False,
            "source_metrics_sha256": {
                "primary5": sha256_file(primary_path),
                "sensitivity7": sha256_file(sensitivity_path),
            },
        },
    )
    return destination


def run_phone_loso_stage(
    config: dict[str, Any] | None = None,
    *,
    cohort: str,
    progress: Progress = print,
) -> dict[str, Any]:
    """Run, persist and evaluate the frozen primary5 or sensitivity7 LOSO."""

    cfg = load_config() if config is None else config
    if cohort not in {"primary5", "sensitivity7"}:
        raise ValueError("cohort must be primary5 or sensitivity7")
    declared_methods = tuple(str(value) for value in cfg.get("methods", ())) + tuple(
        str(value) for value in cfg.get("ablations", ())
    )
    methods = declared_methods or ALL_METHODS
    if methods != ALL_METHODS:
        raise ValueError(
            "frozen method/ablation inventory or order differs from the implementation: "
            f"configured={methods}, expected={ALL_METHODS}"
        )
    layout: ArtifactLayout = layout_from_config(cfg)
    source_manifest = manifest_path(cfg, cohort)
    audit_path = layout.manifests / f"{cohort}.audit.json"
    if not source_manifest.is_file():
        raise FileNotFoundError(
            f"manifest is absent: {source_manifest}; run build-manifests first"
        )
    if not audit_path.is_file():
        raise FileNotFoundError(
            f"manifest audit is absent: {audit_path}; run build-manifests first"
        )
    acoustic_manifest_rows = _read_acoustic_manifest(source_manifest)
    # Build the cohort inference pool without PHN-derived fields.  Training
    # sidecars are attached fold-locally below; held-out evaluation sidecars
    # are not attached until every outer fold has produced predictions.
    loaded_utterances = load_experiment_utterances(
        acoustic_manifest_rows, cfg, include_phn_sidecar=False
    )
    impossible_events = {
        row.event_id
        for row in loaded_utterances
        if np.asarray(row.logits).shape[0] < ctc_minimum_frames(row.canonical_ids)
    }
    # Impossible complete sequences are explicitly excluded once and therefore
    # cannot turn into zero losses or method-specific coverage differences.
    utterances = [
        row for row in loaded_utterances if row.event_id not in impossible_events
    ]
    acoustic_by_event = {row.event_id: row for row in utterances}
    folds = nested_loso_folds(
        cohort, cfg["cohorts"][cohort], cfg["cohorts"]["healthy_phn"]
    )

    deferred_predictions: list[_DeferredOuterPredictions] = []
    runtime_source: list[dict[str, Any]] = []
    inner_oof: list[dict[str, Any]] = []
    gates: dict[str, Any] = {}
    fold_details: dict[str, Any] = {}
    fold_state_hashes: dict[str, str] = {}
    counterfactual_matrix_hashes: dict[str, str] = {}
    if any(row.logits_source is None for row in utterances):
        raise ValueError("run-phone LOSO requires provenance-bound logit sources")
    raw_logits_set_hash = sha256_json(
        {
            row.event_id: _artifact_with_descriptor_hash(row.logits_source)
            for row in sorted(utterances, key=lambda value: value.event_id)
            if row.logits_source is not None
        }
    )
    # Raw official-logit calculations are fold independent.  Sharing them by
    # content address avoids recomputing each inner-validation utterance in
    # every outer fold while leaving recalibrated fold state isolated.
    shared_raw_cache = _SharedRawCache()
    for index, fold in enumerate(folds, start=1):
        progress(f"phone LOSO {cohort}: outer {index}/{len(folds)} {fold.held_out_patient}")
        training_speakers = set(fold.training_patients) | set(
            fold.healthy_references
        )
        training_manifest_rows = _read_training_manifest(
            source_manifest, training_speakers
        )
        training_utterances = [
            _attach_phn_sidecar(
                acoustic_by_event[row.reading_event_id],
                row,
                role="training",
                held_out_patient=fold.held_out_patient,
            )
            for row in training_manifest_rows
            if row.reading_event_id in acoustic_by_event
        ]
        fitted = fit_outer_fold(
            fold,
            training_utterances,
            cfg,
            methods=methods,
            cache_dir=layout.cache,
            progress=progress,
            shared_raw_cache=shared_raw_cache,
        )
        fold_state_hashes[fitted.fold_id] = _fold_state_bundle_hash(
            layout.cache, fitted
        )
        held_out = [
            row for row in utterances if row.speaker_id == fold.held_out_patient
        ]
        if not held_out:
            raise ValueError(f"outer test speaker has no utterances: {fold.held_out_patient}")
        # Produce all predictions without exposing the PHN sidecar to inference.
        private_predictions = predict_with_fitted_fold(
            fitted,
            held_out,
            cfg,
            cache_dir=layout.cache,
            shared_raw_cache=shared_raw_cache,
        )
        fold_matrix_hashes: dict[str, str] = {}
        for row in held_out:
            matrix_path = _counterfactual_cache_path(
                layout.cache,
                fitted.cohort,
                fitted.fit_hash,
                row.event_id,
            )
            matrix_hash = _artifact_with_descriptor_hash(matrix_path)
            fold_matrix_hashes[row.event_id] = matrix_hash
            counterfactual_matrix_hashes[
                f"{fitted.fold_id}:{row.event_id}"
            ] = matrix_hash
        deferred_predictions.append(
            _DeferredOuterPredictions(
                fitted.fold_id,
                fitted.held_out_patient,
                tuple(row.event_id for row in held_out),
                tuple(private_predictions),
            )
        )
        runtime_source.extend(_runtime_source_rows(private_predictions))
        inner_oof.extend(dict(row) for row in fitted.inner_oof_rows)
        gates[fitted.fold_id] = fitted.gate.to_dict()
        fold_details[fitted.fold_id] = {
            "held_out_patient": fitted.held_out_patient,
            "training_speakers": list(fitted.training_speakers),
            "fit_hash": fitted.fit_hash,
            "fold_state_bundle_sha256": fold_state_hashes[fitted.fold_id],
            "counterfactual_matrix_set_sha256": sha256_json(fold_matrix_hashes),
            "selected_lambda": fitted.selected_lambda,
            "lambda_macro_auprc": {
                str(key): value for key, value in fitted.lambda_metrics.items()
            },
            "specific_output_gate": fitted.gate.to_dict(),
        }

    # This is the only held-out evaluation-sidecar attachment point.  It is
    # deliberately outside the outer-fold loop, after all cohort predictions.
    manifest_rows = read_manifest(source_manifest)
    manifest_by_event = {
        row.reading_event_id: row
        for row in manifest_rows
        if row.reading_event_id in acoustic_by_event
    }
    if set(acoustic_by_event) != set(manifest_by_event):
        raise ValueError("acoustic and evaluation-manifest inventories differ")
    event_metadata = {
        row.reading_event_id: {
            "audio_microphone": row.audio_microphone,
            "phn_microphone": row.phn_microphone,
            "severity_rank": row.severity_rank,
            "severity_label": row.severity_label,
        }
        for row in manifest_rows
    }
    predictions = _finalize_deferred_outer_predictions(
        deferred_predictions,
        acoustic_by_event,
        manifest_by_event,
        event_metadata,
    )
    # The audit contains aggregate PHN-derived stability statistics.  It is
    # therefore also deferred until predictions are complete and can only
    # govern reporting eligibility, never a model, threshold or cache choice.
    manifest_audit = read_json(audit_path)
    if not isinstance(manifest_audit, Mapping) or type(
        manifest_audit.get("exact_substitution_deletion_headline_allowed")
    ) is not bool:
        raise ValueError("manifest audit lacks the frozen headline stability decision")
    exact_headline_allowed = bool(
        manifest_audit["exact_substitution_deletion_headline_allowed"]
    )

    output_dir = layout.evaluation / cohort
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "phone_predictions.jsonl.gz"
    runtime_source_path = output_dir / "runtime_source_predictions.jsonl.gz"
    inner_oof_path = output_dir / "inner_oof_predictions.jsonl.gz"
    gates_path = output_dir / "specific_output_gates.json"
    metrics_path = output_dir / "metrics.json"
    csv_path = output_dir / "method_metrics.csv"
    speaker_csv_path = output_dir / "speaker_metrics.csv"
    write_jsonl_gz(predictions_path, predictions)
    write_jsonl_gz(runtime_source_path, runtime_source)
    write_jsonl_gz(inner_oof_path, inner_oof)
    write_json(
        gates_path,
        {
            "schema_version": "da-cf-gop.specific-output-gates.v1",
            "cohort": cohort,
            "folds": gates,
        },
    )
    report = evaluate_predictions(predictions)
    report["config_sha256"] = _config_digest(cfg)
    paired, success = _paired_and_success(report, cfg, cohort)
    report["headline_policy"] = {
        "exact_substitution_deletion_headline_allowed": exact_headline_allowed,
        "error_detection_status": (
            "headline" if exact_headline_allowed else "exploratory_only"
        ),
        "substitution_deletion_status": (
            "headline" if exact_headline_allowed else "exploratory_only"
        ),
        "minimum_stable_fraction": manifest_audit.get("minimum_stable_fraction"),
        "observed_patient_stable_fraction": manifest_audit.get(
            "patient_stable_fraction"
        ),
    }
    if not exact_headline_allowed:
        # The predeclared success rule includes exact substitution/deletion
        # non-degradation; below 80% stable labels it is not evaluable rather
        # than a pass or a test-set-tuned failure.
        success["cohort_success"] = None
        success["assessment_status"] = (
            "not_evaluable_alignment_stability_below_threshold"
        )
        success["negative_result_frozen_if_false"] = False
    else:
        success["assessment_status"] = "evaluated"
    report["paired_comparison"] = paired
    report["paired_comparisons"] = _all_paired_comparisons(report, cfg)
    report["success_assessment"] = success
    write_json(metrics_path, report)
    write_csv(csv_path, _flatten_metrics(report))
    write_csv(speaker_csv_path, _flatten_speaker_metrics(report))
    final_assessment_path = _write_final_assessment_if_ready(layout)

    outputs = {
        "phone_predictions": sha256_file(predictions_path),
        "runtime_source_predictions": sha256_file(runtime_source_path),
        "inner_oof_predictions": sha256_file(inner_oof_path),
        "specific_output_gates": sha256_file(gates_path),
        "metrics": sha256_file(metrics_path),
        "method_metrics": sha256_file(csv_path),
        "speaker_metrics": sha256_file(speaker_csv_path),
    }
    if final_assessment_path is not None:
        outputs["final_assessment"] = sha256_file(final_assessment_path)
    summary = write_stage_summary(
        layout.summaries / f"run-phone-loso.{cohort}.json",
        stage="run-phone-loso",
        config_sha256=_config_digest(cfg),
        training_speakers=tuple(cfg["cohorts"]["healthy_phn"])
        + tuple(cfg["cohorts"][cohort]),
        source_files={
            "frozen_manifest": sha256_file(source_manifest),
            "manifest_audit": sha256_file(audit_path),
        },
        models={"official_ctc_sf_checkpoint": _checkpoint_digest(cfg)},
        manifests={cohort: sha256_file(source_manifest)},
        upstream_artifacts={
            "counterfactual_matrix_set": sha256_json(
                counterfactual_matrix_hashes
            ),
            "fold_state_set": sha256_json(fold_state_hashes),
            "raw_logits_set": raw_logits_set_hash,
        },
        exclusions={
            "alignment_uncertain": sum(row.uncertain_count for row in manifest_rows),
            "ctc_impossible_canonical_sequence": len(impossible_events),
        },
        outputs=outputs,
        details={
            "cohort": cohort,
            "folds": fold_details,
            "n_evaluation_rows": len(predictions),
            "n_runtime_source_rows": len(runtime_source),
            "all_evaluation_methods_share_stable_token_keys": True,
            "outer_evaluation_phn_sidecars_attached_after_all_outer_predictions": True,
            "gold_join_performed_after_all_cohort_predictions": True,
            "training_and_outer_evaluation_phn_roles_are_explicitly_separated": True,
            "held_out_phn_never_consumed_by_its_own_fold_fit_or_prediction": True,
            "phn_derived_manifest_audit_read_after_all_predictions": True,
            "runtime_inference_uses_audio_and_prompt_only": True,
            "success_assessment": success,
        },
    )
    return summary


__all__ = [
    "ALL_METHODS",
    "BASELINE_METHODS",
    "CORE_METHODS",
    "ExperimentUtterance",
    "FittedFold",
    "METHOD_ADAPTED",
    "METHOD_FIXED",
    "METHOD_UNIFORM_CALIBRATED",
    "fit_outer_fold",
    "load_experiment_utterances",
    "predict_with_fitted_fold",
    "run_phone_loso_stage",
    "utterance_from_manifest",
]
