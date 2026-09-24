"""DA-CF-GoP scoring for the frozen eight-speaker severity audit.

The historical code9 cache is treated as an immutable source of *raw acoustic
logits only*.  Canonical phones are regenerated from the prompt with the
frozen local eSpeak G2P.  In particular, the cached ``canonical_ids`` and
``metadata_json`` fields are never inputs to scoring.

M03 has no usable PHN sidecar in the extracted TORGO tree.  The severity LOSO
policy therefore uses every *PHN-labelled* non-test patient: seven patients
when M03 is held out, and six patients when one of the labelled patients is
held out.  Four healthy PHN speakers remain training references in every
fold.  This is explicit in the emitted fold document and never imputed.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import ArtifactLayout, write_jsonl_gz, write_stage_summary
from .backend import (
    OfficialCTCSFBackend,
    build_backend_hashes,
    build_cache_binding,
    load_audio,
    load_logits_cache,
    migrate_legacy_logits_cache,
    save_logits_cache,
)
from .config import load_config
from .ctc import ctc_minimum_frames
from .data import ManifestRow, build_torgo_manifest, read_manifest
from .experiment import (
    ALL_METHODS,
    METHOD_ADAPTED,
    ExperimentUtterance,
    FittedFold,
    fit_outer_fold,
    load_experiment_utterances,
    predict_with_fitted_fold,
    _fold_fit_hash,
    _load_fitted_fold,
)
from .folds import InnerFold, OuterFold, nested_loso_folds
from .phonology import ARPABET_39, PHONE_TO_CTC_ID, texts_to_phones
from .provenance import (
    companion_descriptor_path,
    read_json,
    sha256_file,
    sha256_json,
    write_json,
)
from .severity import (
    EXPECTED_DYSARTHRIC_SPEAKERS,
    NEW_METHOD,
    SeverityError,
    build_audited_severity_events,
    write_severity_inference_manifest,
)
from .stages import _logit_descriptor, _raw_frozen_config, layout_from_config, logit_cache_path


TRAINING_COHORT = "severity8"
TRAINING_MANIFEST_NAME = "severity_training8.jsonl.gz"
FOLD_SCHEMA_VERSION = "da-cf-gop.severity-folds.v1"
PHONE_SCORE_SCHEMA_VERSION = "da-cf-gop.severity-phone-score.v1"
PROVENANCE_SCHEMA_VERSION = "da-cf-gop.severity-scoring-provenance.v1"

Progress = Callable[[str], None]
ManifestBuilder = Callable[..., tuple[list[ManifestRow], list[Any], dict[str, object]]]
G2PFunction = Callable[[Sequence[str]], tuple[list[list[str]], list[list[str]]]]


@dataclass(frozen=True)
class AuditedAcousticInput:
    """One label-free audited recording and its prompt-derived inference view."""

    recording_id: str
    event_id: str
    speaker_id: str
    session: str
    stem: str
    source_logits: str
    source_sha256: str
    legacy_canonical_ids_match: bool
    utterance: ExperimentUtterance


def validate_audited_backend_descriptor(
    config: Mapping[str, Any], logits_cache: str | Path
) -> dict[str, Any]:
    """Validate code9's global backend descriptor against current model bytes.

    The historical cache predates per-NPZ companion descriptors.  We therefore
    require its one global backend descriptor, verify the vocabulary and model
    weight hashes it declares, and bind every selected NPZ by SHA-256 in the
    new score provenance.
    """

    cache_root = Path(logits_cache).resolve()
    descriptor_path = cache_root / "backend_descriptor.json"
    if not descriptor_path.is_file():
        raise SeverityError(
            f"audited logits backend descriptor is missing: {descriptor_path}"
        )
    descriptor = read_json(descriptor_path)
    if not isinstance(descriptor, Mapping):
        raise SeverityError("audited backend descriptor is not a JSON object")
    backend = _require_mapping(config.get("backend"), "backend")
    paths = _require_mapping(config.get("paths"), "paths")
    if int(descriptor.get("blank_id", -1)) != int(backend["blank_id"]):
        raise SeverityError("audited backend descriptor has the wrong blank id")
    if int(descriptor.get("vocab_size", -1)) != int(backend["vocab_size"]):
        raise SeverityError("audited backend descriptor has the wrong vocabulary size")
    checkpoint = Path(str(paths["checkpoint"])).resolve()
    processor = Path(str(paths["processor"])).resolve()
    model_path = checkpoint / "pytorch_model.bin"
    vocab_path = processor / "vocab.json"
    if not model_path.is_file() or not vocab_path.is_file():
        raise SeverityError("current frozen checkpoint or processor vocabulary is missing")
    observed_model = sha256_file(model_path)
    observed_vocab = sha256_file(vocab_path)
    if descriptor.get("pytorch_model_bin_sha256") != observed_model:
        raise SeverityError("audited logits checkpoint hash no longer matches frozen weights")
    if descriptor.get("vocab_sha256") != observed_vocab:
        raise SeverityError("audited logits vocabulary hash no longer matches frozen processor")
    declared_config = str(descriptor.get("config_sha256", ""))
    if len(declared_config) != 64 or any(
        value not in "0123456789abcdef" for value in declared_config.casefold()
    ):
        raise SeverityError("audited backend descriptor has an invalid extraction config hash")
    if int(descriptor.get("n_requested", 0)) < int(
        _require_mapping(config.get("severity"), "severity")["audited_recordings"]
    ):
        raise SeverityError("audited backend descriptor covers too few recordings")
    return {
        "descriptor_path": str(descriptor_path),
        "descriptor_sha256": sha256_file(descriptor_path),
        "checkpoint_weight_sha256": observed_model,
        "vocab_sha256": observed_vocab,
        "blank_id": int(descriptor["blank_id"]),
        "vocab_size": int(descriptor["vocab_size"]),
        "historical_extraction_config_sha256": declared_config,
        "per_npz_companion_descriptors": False,
        "per_npz_content_binding": "SHA-256 inventory in DA-CF severity provenance",
    }


def _require_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SeverityError(f"{name} must be a mapping")
    return value


def severity_training_manifest_path(config: Mapping[str, Any]) -> Path:
    artifacts = Path(str(_require_mapping(config.get("paths"), "paths")["artifacts"]))
    return artifacts.resolve() / "manifests" / TRAINING_MANIFEST_NAME


def severity_loso_folds(
    all_patients: Sequence[str],
    phn_labelled_patients: Sequence[str],
    healthy_references: Sequence[str],
) -> tuple[OuterFold, ...]:
    """Create outer folds over eight patients without inventing missing PHN.

    Hyperparameters and calibration are selected only on inner OOF folds of
    the PHN-labelled training patients.  An unlabelled patient (M03 in TORGO)
    remains a valid outer target but cannot enter a PHN-adapted training fold.
    """

    patients = tuple(sorted(str(value).upper() for value in all_patients))
    labelled = tuple(sorted(str(value).upper() for value in phn_labelled_patients))
    healthy = tuple(sorted(str(value).upper() for value in healthy_references))
    if len(patients) < 3 or len(labelled) < 3 or not healthy:
        raise SeverityError("severity LOSO needs >=3 patients, >=3 PHN patients, and healthy references")
    if len(patients) != len(set(patients)) or len(labelled) != len(set(labelled)):
        raise SeverityError("severity speaker lists contain duplicates")
    if not set(labelled).issubset(patients):
        raise SeverityError("PHN-labelled severity patients must be outer patients")
    if set(patients) & set(healthy):
        raise SeverityError("patient and healthy speaker sets overlap")

    folds: list[OuterFold] = []
    for held_out in patients:
        training = tuple(value for value in labelled if value != held_out)
        inner = tuple(
            InnerFold(
                held_out_patient=validation,
                training_patients=tuple(
                    value for value in training if value != validation
                ),
                healthy_references=healthy,
            )
            for validation in training
        )
        folds.append(
            OuterFold(
                cohort=TRAINING_COHORT,
                held_out_patient=held_out,
                training_patients=training,
                healthy_references=healthy,
                inner_folds=inner,
            )
        )
    return tuple(folds)


def _fold_document(
    folds: Sequence[OuterFold],
    *,
    all_patients: Sequence[str],
    labelled_patients: Sequence[str],
    healthy_references: Sequence[str],
) -> dict[str, Any]:
    all_set = set(str(value).upper() for value in all_patients)
    labelled_set = set(str(value).upper() for value in labelled_patients)
    return {
        "schema_version": FOLD_SCHEMA_VERSION,
        "outer_test_speakers": sorted(all_set),
        "phn_labelled_patient_pool": sorted(labelled_set),
        "phn_unavailable_patient_pool": sorted(all_set - labelled_set),
        "healthy_references": sorted(str(value).upper() for value in healthy_references),
        "missing_phn_policy": "exclude from PHN-adapted training; retain as outer target",
        "severity_labels_available_to_fitting": False,
        "folds": [fold.to_dict() for fold in folds],
    }


def build_severity_training_manifest_stage(
    config: Mapping[str, Any] | None = None,
    *,
    progress: Progress = print,
    manifest_builder: ManifestBuilder = build_torgo_manifest,
) -> dict[str, Any]:
    """Build the deterministic PHN training sidecar for severity LOSO."""

    cfg = load_config() if config is None else dict(config)
    severity = _require_mapping(cfg.get("severity"), "severity")
    cohorts = _require_mapping(cfg.get("cohorts"), "cohorts")
    paths = _require_mapping(cfg.get("paths"), "paths")
    labels = _require_mapping(cfg.get("labels"), "labels")
    audio_policy = _require_mapping(cfg.get("audio_policy"), "audio_policy")
    all_patients = tuple(str(value).upper() for value in severity["dysarthric_speakers"])
    healthy = tuple(str(value).upper() for value in cohorts["healthy_phn"])
    if tuple(sorted(all_patients)) != EXPECTED_DYSARTHRIC_SPEAKERS:
        raise SeverityError("severity config does not contain the frozen eight patients")

    rows, exclusions, original_audit = manifest_builder(
        paths["data_root"],
        dys_speakers=all_patients,
        healthy_speakers=healthy,
        cohort=TRAINING_COHORT,
        minimum_stable_fraction=float(labels["stable_alignment_min_fraction"]),
        allow_cross_mic_phn_for=audio_policy["allow_cross_mic_phn_sequence_for"],
    )
    # Severity is not a fitting feature.  Remove it before serializing the
    # training manifest, even though ManifestRow retains nullable schema slots.
    scrubbed = [replace(row, severity_rank=None, severity_label="") for row in rows]
    labelled_patients = tuple(sorted({
        row.speaker_id for row in scrubbed if row.speaker_group == "dysarthric"
    }))
    observed_healthy = {row.speaker_id for row in scrubbed if row.speaker_group == "healthy"}
    if observed_healthy != set(healthy):
        raise SeverityError(
            f"severity training manifest is missing healthy speakers: {sorted(set(healthy) - observed_healthy)}"
        )
    folds = severity_loso_folds(all_patients, labelled_patients, healthy)

    layout = ArtifactLayout.from_path(paths["artifacts"])
    layout.create()
    manifest_path = layout.manifests / TRAINING_MANIFEST_NAME
    exclusions_path = layout.manifests / "severity_training8.exclusions.jsonl.gz"
    audit_path = layout.manifests / "severity_training8.audit.json"
    fold_path = layout.manifests / "severity_training8.folds.json"
    write_jsonl_gz(manifest_path, (row.to_dict() for row in scrubbed))
    write_jsonl_gz(exclusions_path, (row.to_dict() for row in exclusions))
    semantic_manifest_hash = hashlib.sha256(
        json.dumps(
            [row.to_dict() for row in scrubbed],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    audit = {
        **original_audit,
        "schema_version": "da-cf-gop.severity-training-manifest.v1",
        "source_scan_manifest_sha256": original_audit.get("manifest_sha256"),
        "manifest_sha256": semantic_manifest_hash,
        "requested_outer_patients": sorted(all_patients),
        "phn_labelled_patients": list(labelled_patients),
        "phn_unavailable_patients": sorted(set(all_patients) - set(labelled_patients)),
        "severity_fields_scrubbed": True,
        "severity_used_for_training": False,
    }
    write_json(audit_path, audit)
    write_json(
        fold_path,
        _fold_document(
            folds,
            all_patients=all_patients,
            labelled_patients=labelled_patients,
            healthy_references=healthy,
        ),
    )
    outputs = {
        "manifest": sha256_file(manifest_path),
        "exclusions": sha256_file(exclusions_path),
        "audit": sha256_file(audit_path),
        "folds": sha256_file(fold_path),
    }
    input_hashes = original_audit.get("input_source_hashes")
    g2p = original_audit.get("g2p_provenance")
    if not isinstance(input_hashes, Mapping) or not isinstance(g2p, Mapping):
        raise SeverityError("severity training manifest lacks source/G2P provenance")
    source_files = {"frozen_config": str(cfg["_config_hash"])}
    for role in ("wav", "prompt", "phn"):
        record = input_hashes.get(role)
        if not isinstance(record, Mapping) or not isinstance(
            record.get("inventory_sha256"), str
        ):
            raise SeverityError(f"severity manifest lacks {role} source hashes")
        source_files[f"{role}_input_inventory"] = str(record["inventory_sha256"])
    summary = write_stage_summary(
        layout.summaries / "build-severity-training-manifest.json",
        stage="build-severity-training-manifest",
        config_sha256=str(cfg["_config_hash"]),
        training_speakers=(*labelled_patients, *healthy),
        source_files=source_files,
        # Stage-summary model inventories are digest-only.  Human-readable
        # runtime versions remain available in the manifest audit itself.
        models={
            "g2p_descriptor": str(g2p["descriptor_sha256"]),
            **{
                model_name: str(g2p[provenance_key])
                for model_name, provenance_key in (
                    ("espeak_library", "espeak_library_sha256"),
                    ("espeak_data", "espeak_data_sha256"),
                )
                if isinstance(g2p.get(provenance_key), str)
            },
        },
        manifests={TRAINING_COHORT: outputs["manifest"]},
        upstream_artifacts={},
        exclusions=Counter(str(item.reason) for item in exclusions),
        outputs=outputs,
        details={
            "n_rows": len(scrubbed),
            "n_outer_targets": len(all_patients),
            "phn_labelled_patients": list(labelled_patients),
            "phn_unavailable_patients": sorted(set(all_patients) - set(labelled_patients)),
            "severity_used_for_training": False,
        },
    )
    progress(
        f"severity training manifest: {len(scrubbed)} rows; PHN patients="
        f"{','.join(labelled_patients)}; unavailable="
        f"{','.join(sorted(set(all_patients) - set(labelled_patients)))}"
    )
    return summary


def extract_severity_training_logits_stage(
    config: Mapping[str, Any] | None = None,
    *,
    progress: Progress = print,
    limit: int | None = None,
) -> dict[str, Any]:
    """Ensure official, content-bound logits for every severity training row.

    The cache namespace is shared with the primary phone experiment because
    the event id, audio, backend and frozen config binding are identical.  A
    separate index is written so this stage cannot overwrite the headline
    ``logits.index.json`` inventory.
    """

    cfg = load_config() if config is None else dict(config)
    layout = layout_from_config(cfg)
    source_manifest = severity_training_manifest_path(cfg)
    if not source_manifest.is_file():
        raise FileNotFoundError(
            f"severity training manifest is absent: {source_manifest}"
        )
    rows = read_manifest(source_manifest)
    if limit is not None:
        if int(limit) <= 0:
            raise ValueError("limit must be positive")
        rows = rows[: int(limit)]
    checkpoint = Path(str(_require_mapping(cfg.get("paths"), "paths")["checkpoint"]))
    processor = Path(str(_require_mapping(cfg.get("paths"), "paths")["processor"]))
    backend_hashes = build_backend_hashes(checkpoint=checkpoint, processor=processor)
    try:
        raw_config = _raw_frozen_config(cfg)
    except (KeyError, OSError, ValueError):
        raw_config = {
            key: value for key, value in cfg.items() if not str(key).startswith("_")
        }
    backend: OfficialCTCSFBackend | None = None
    created = 0
    reused = 0
    migrated = 0
    entries: dict[str, str] = {}
    backend_cfg = _require_mapping(cfg.get("backend"), "backend")
    for index, row in enumerate(rows, start=1):
        destination = logit_cache_path(cfg, row.reading_event_id)
        canonical_ids = tuple(PHONE_TO_CTC_ID[phone] for phone in row.canonical_phones)
        binding = build_cache_binding(
            audio_path=row.wav_path,
            checkpoint=checkpoint,
            processor=processor,
            experiment_config=str(cfg["_config_hash"]),
            canonical_ids=canonical_ids,
            config_path=cfg.get("_config_path"),
            backend_hashes=backend_hashes,
        )
        valid = False
        if destination.is_file():
            try:
                cached = load_logits_cache(
                    destination, expected_fingerprint=str(binding["fingerprint"])
                )
                valid = (
                    cached.utterance_id == row.reading_event_id
                    and tuple(int(value) for value in cached.canonical_ids)
                    == canonical_ids
                )
            except (OSError, ValueError):
                if migrate_legacy_logits_cache(
                    destination,
                    expected_utterance_id=row.reading_event_id,
                    expected_canonical_ids=canonical_ids,
                    new_binding=binding,
                ):
                    cached = load_logits_cache(
                        destination, expected_fingerprint=str(binding["fingerprint"])
                    )
                    valid = (
                        cached.utterance_id == row.reading_event_id
                        and tuple(int(value) for value in cached.canonical_ids)
                        == canonical_ids
                    )
                    migrated += int(valid)
                else:
                    valid = False
        if valid:
            reused += 1
        else:
            if backend is None:
                backend = OfficialCTCSFBackend(
                    checkpoint,
                    processor,
                    device=str(backend_cfg["device"]),
                )
                if backend.blank_id != int(backend_cfg["blank_id"]):
                    raise ValueError("frozen blank id disagrees with processor")
                if backend.vocab_size != int(backend_cfg["vocab_size"]):
                    raise ValueError("frozen vocabulary size disagrees with processor")
            waveform = load_audio(row.wav_path, int(backend_cfg["sample_rate"]))
            logits = backend.infer(waveform, int(backend_cfg["sample_rate"]))
            save_logits_cache(
                destination,
                utterance_id=row.reading_event_id,
                logits=logits,
                canonical_ids=canonical_ids,
                duration_s=row.duration_s,
                binding=binding,
                metadata={
                    "schema_version": "da-cf-gop.logits.v1",
                    "prompt_sha256": hashlib.sha256(row.prompt.encode("utf-8")).hexdigest(),
                    "audio_microphone": "headMic",
                    "external_alignment_used": False,
                },
            )
            created += 1
        descriptor = _logit_descriptor(
            artifact=destination,
            audio_path=Path(row.wav_path),
            binding=binding,
            raw_config=raw_config,
            event_id=row.reading_event_id,
            canonical_ids=canonical_ids,
        )
        write_json(companion_descriptor_path(destination), descriptor)
        entries[row.reading_event_id] = sha256_file(destination)
        if index == 1 or index % 100 == 0 or index == len(rows):
            progress(
                f"severity training logits {index}/{len(rows)} "
                f"(new={created}, reused={reused})"
            )

    index_path = layout.cache / "severity_training8.logits.index.json"
    document = {
        "schema_version": "da-cf-gop.severity-training-logit-index.v1",
        "manifest_sha256": sha256_file(source_manifest),
        "backend_sha256": backend_hashes,
        "config_sha256": str(cfg["_config_hash"]),
        "entries": dict(sorted(entries.items())),
        "partial_limit": limit,
    }
    write_json(index_path, document)
    return write_stage_summary(
        layout.summaries / "extract-severity-training-logits.json",
        stage="extract-severity-training-logits",
        config_sha256=str(cfg["_config_hash"]),
        training_speakers=sorted({row.speaker_id for row in rows}),
        source_files={"frozen_config": str(cfg["_config_hash"])},
        models=backend_hashes,
        manifests={TRAINING_COHORT: sha256_file(source_manifest)},
        upstream_artifacts={},
        exclusions={},
        outputs={"severity_training_logit_index": sha256_file(index_path)},
        details={
            "n_rows": len(rows),
            "created": created,
            "reused": reused,
            "migrated_legacy_bindings": migrated,
            "partial_limit": limit,
            "severity_used_for_training": False,
        },
    )


def load_audited_acoustic_inputs(
    recordings: Sequence[Mapping[str, object]],
    *,
    vocab_size: int = 40,
    g2p: G2PFunction = texts_to_phones,
) -> list[AuditedAcousticInput]:
    """Load code9 raw logits while deriving every canonical phone from prompt."""

    if not recordings:
        raise SeverityError("severity inference recording list is empty")
    prompts = list(dict.fromkeys(str(row.get("prompt", "")).strip() for row in recordings))
    if any(not prompt for prompt in prompts):
        raise SeverityError("severity inference prompt is empty")
    mapped, unknown = g2p(prompts)
    if len(mapped) != len(prompts) or len(unknown) != len(prompts):
        raise SeverityError("severity G2P returned a different number of rows")
    phones_by_prompt: dict[str, tuple[str, ...]] = {}
    for prompt, phones, unmapped in zip(prompts, mapped, unknown):
        canonical = tuple(str(phone).upper() for phone in phones)
        invalid = sorted(set(canonical) - set(ARPABET_39))
        if unmapped or not canonical or invalid:
            raise SeverityError(
                f"severity prompt cannot be mapped to frozen phones: {prompt!r}; "
                f"unknown={list(unmapped)}, invalid={invalid}"
            )
        phones_by_prompt[prompt] = canonical

    output: list[AuditedAcousticInput] = []
    seen: set[str] = set()
    for row in sorted(recordings, key=lambda value: str(value.get("recording_id", ""))):
        recording_id = str(row.get("recording_id", "")).strip()
        if not recording_id or recording_id in seen:
            raise SeverityError(f"invalid or duplicate severity recording_id: {recording_id!r}")
        seen.add(recording_id)
        speaker = str(row.get("speaker_id", "")).strip().upper()
        event_id = str(row.get("event_id", "")).strip()
        session = str(row.get("session", "")).strip()
        stem = str(row.get("stem", "")).strip()
        prompt = str(row.get("prompt", "")).strip()
        source = Path(str(row.get("official_logits_npz", ""))).resolve()
        if not speaker or not event_id or not session or not stem or not source.is_file():
            raise SeverityError(f"incomplete severity inference row: {recording_id}")
        canonical = phones_by_prompt[prompt]
        canonical_ids = tuple(PHONE_TO_CTC_ID[phone] for phone in canonical)
        try:
            with np.load(source, allow_pickle=False) as archive:
                required = {
                    "utterance_id",
                    "logits",
                    "canonical_ids",
                    "duration_s",
                    "metadata_json",
                }
                if not required.issubset(archive.files):
                    raise SeverityError(f"audited logits lack required arrays: {source}")
                # Load every member with allow_pickle=False to reject object
                # arrays, but never parse metadata_json or use its contents.
                cached_id_array = np.asarray(archive["utterance_id"])
                raw_logits = np.asarray(archive["logits"])
                raw_legacy_ids = np.asarray(archive["canonical_ids"])
                duration_array = np.asarray(archive["duration_s"])
                metadata_array = np.asarray(archive["metadata_json"])
        except (OSError, ValueError) as error:
            raise SeverityError(f"invalid audited logits: {source}") from error
        if cached_id_array.shape != () or cached_id_array.dtype.kind not in {"U", "S"}:
            raise SeverityError(f"audited logits utterance_id is not a scalar string: {source}")
        if metadata_array.shape != () or metadata_array.dtype.kind not in {"U", "S"}:
            raise SeverityError(f"audited logits metadata is not a scalar string: {source}")
        if duration_array.shape != ():
            raise SeverityError(f"audited logits duration is not scalar: {source}")
        duration = float(duration_array.item())
        if not math.isfinite(duration) or duration <= 0:
            raise SeverityError(f"audited logits duration is invalid: {source}")
        if raw_logits.dtype != np.dtype("float32"):
            raise SeverityError(f"audited logits dtype must be float32: {source}")
        if raw_legacy_ids.ndim != 1 or raw_legacy_ids.dtype.kind not in {"i", "u"}:
            raise SeverityError(f"audited cached canonical ids are malformed: {source}")
        if np.any(raw_legacy_ids <= 0) or np.any(raw_legacy_ids >= int(vocab_size)):
            raise SeverityError(f"audited cached canonical ids are outside vocabulary: {source}")
        cached_id = str(cached_id_array.item())
        logits = raw_logits.astype(np.float64, copy=True)
        legacy_ids = tuple(int(value) for value in raw_legacy_ids.tolist())
        if cached_id != recording_id:
            raise SeverityError(
                f"audited logits id mismatch: expected {recording_id}, got {cached_id}"
            )
        if (
            logits.ndim != 2
            or logits.shape[0] == 0
            or logits.shape[1] != int(vocab_size)
            or not np.isfinite(logits).all()
        ):
            raise SeverityError(f"audited logits are not finite [frames,{vocab_size}]: {source}")
        if ctc_minimum_frames(canonical_ids) > logits.shape[0]:
            raise SeverityError(
                f"prompt-derived canonical sequence is CTC-impossible: {recording_id}"
            )
        # The synthetic event id is recording-specific because the 415 audit
        # intentionally contains repeated physical reading events.  It is
        # mapped back to the frozen 329-event identity in the output rows.
        inference_id = f"severity_recording::{recording_id}"
        utterance = ExperimentUtterance(
            event_id=inference_id,
            speaker_id=speaker,
            speaker_group="dysarthric",
            canonical_phones=canonical,
            canonical_ids=canonical_ids,
            logits=logits,
            logits_source=str(source),
        )
        output.append(
            AuditedAcousticInput(
                recording_id=recording_id,
                event_id=event_id,
                speaker_id=speaker,
                session=session,
                stem=stem,
                source_logits=str(source),
                source_sha256=sha256_file(source),
                legacy_canonical_ids_match=(legacy_ids == canonical_ids),
                utterance=utterance,
            )
        )
    return output


def _usable_training_utterances(
    rows: Sequence[ManifestRow], config: Mapping[str, Any]
) -> list[ExperimentUtterance]:
    loaded = load_experiment_utterances(rows, config)
    usable = [
        row
        for row in loaded
        if ctc_minimum_frames(row.canonical_ids) <= np.asarray(row.logits).shape[0]
    ]
    missing_speakers = {row.speaker_id for row in rows} - {
        row.speaker_id for row in usable
    }
    if missing_speakers:
        raise SeverityError(
            f"CTC filtering removed every training row for: {sorted(missing_speakers)}"
        )
    return usable


def generate_severity_phone_scores(
    config: Mapping[str, Any],
    training_rows: Sequence[ManifestRow],
    audited_inputs: Sequence[AuditedAcousticInput],
    folds: Sequence[OuterFold],
    *,
    fold_cache_dir: str | Path,
    progress: Progress = print,
    training_loader: Callable[
        [Sequence[ManifestRow], Mapping[str, Any]], Sequence[ExperimentUtterance]
    ] = _usable_training_utterances,
    fit_fold: Callable[..., FittedFold] = fit_outer_fold,
    predict_fold: Callable[..., list[dict[str, Any]]] = predict_with_fitted_fold,
    fit_fold_overrides: Mapping[str, OuterFold] | None = None,
    fit_methods_by_speaker: Mapping[str, Sequence[str]] | None = None,
    fit_cache_by_speaker: Mapping[str, str | Path] | None = None,
    require_cached_fit_targets: Sequence[str] = (),
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Fit eight leakage-safe folds and score all 415 audited recordings."""

    by_speaker: dict[str, list[AuditedAcousticInput]] = defaultdict(list)
    for item in audited_inputs:
        by_speaker[item.speaker_id].append(item)
    expected_targets = {fold.held_out_patient for fold in folds}
    if set(by_speaker) != expected_targets:
        raise SeverityError(
            "severity scoring target speakers disagree with folds; "
            f"missing={sorted(expected_targets - set(by_speaker))}, "
            f"extra={sorted(set(by_speaker) - expected_targets)}"
        )

    output: list[dict[str, Any]] = []
    fit_hashes: dict[str, str] = {}
    metadata_by_inference_id = {
        item.utterance.event_id: item for item in audited_inputs
    }
    require_cached = {str(value).upper() for value in require_cached_fit_targets}
    if not require_cached.issubset(expected_targets):
        raise SeverityError("required cached fits contain a non-target speaker")
    for index, fold in enumerate(folds, start=1):
        fitting_fold = (fit_fold_overrides or {}).get(fold.held_out_patient, fold)
        if (
            fitting_fold.held_out_patient != fold.held_out_patient
            or set(fitting_fold.training_patients) != set(fold.training_patients)
            or set(fitting_fold.healthy_references) != set(fold.healthy_references)
        ):
            raise SeverityError(
                f"fit reuse plan changes the severity LOSO population: {fold.held_out_patient}"
            )
        training_speakers = set(fold.training_patients) | set(fold.healthy_references)
        # This is the leakage boundary: rows for the outer speaker are not
        # supplied to the loader and hence their PHN fields are never read.
        selected_rows = [
            row for row in training_rows if row.speaker_id in training_speakers
        ]
        if {row.speaker_id for row in selected_rows} != training_speakers:
            raise SeverityError(f"incomplete severity training fold: {fold.fold_id}")
        loaded = list(training_loader(selected_rows, config))
        fit_methods = tuple(
            (fit_methods_by_speaker or {}).get(
                fold.held_out_patient, (METHOD_ADAPTED,)
            )
        )
        if METHOD_ADAPTED not in fit_methods:
            raise SeverityError("severity fit plan must include DA-CF adapted")
        selected_cache = (fit_cache_by_speaker or {}).get(
            fold.held_out_patient, fold_cache_dir
        )
        if fold.held_out_patient in require_cached:
            expected_fit_hash = _fold_fit_hash(
                fitting_fold, loaded, config, methods=fit_methods
            )
            fitted = _load_fitted_fold(
                fold=fitting_fold,
                fit_hash=expected_fit_hash,
                config=config,
                cache_dir=selected_cache,
                methods=fit_methods,
            )
            if fitted is None:
                raise SeverityError(
                    "completed sensitivity7 run has no valid matching full fold cache for "
                    f"{fold.held_out_patient}: cohort={fitting_fold.cohort}, "
                    f"fit_hash={expected_fit_hash}, methods={list(fit_methods)}"
                )
            progress(
                f"{fitting_fold.fold_id}: reused validated full fold fit "
                f"{expected_fit_hash[:12]}"
            )
        else:
            fitted = fit_fold(
                fitting_fold,
                loaded,
                config,
                methods=fit_methods,
                cache_dir=selected_cache,
                progress=progress,
            )
        fit_hashes[fold.held_out_patient] = str(fitted.fit_hash)
        # A full sensitivity7 fit can be reused for the seven shared outer
        # speakers.  Prediction is narrowed back to the one registered
        # severity method so no baseline matrices are computed for the 415
        # audited recordings.
        if tuple(fitted.methods) != (METHOD_ADAPTED,):
            fitted = replace(fitted, methods=(METHOD_ADAPTED,))
        targets = sorted(
            by_speaker[fold.held_out_patient], key=lambda value: value.recording_id
        )
        predictions = predict_fold(
            fitted,
            [item.utterance for item in targets],
            config,
            cache_dir=None,
        )
        expected_tokens = sum(len(item.utterance.canonical_phones) for item in targets)
        if len(predictions) != expected_tokens:
            raise SeverityError(
                f"severity fold {fold.fold_id} produced {len(predictions)} rows; "
                f"expected {expected_tokens}"
            )
        for prediction in predictions:
            if str(prediction.get("method")) != METHOD_ADAPTED:
                raise SeverityError("severity scorer emitted an unregistered method")
            inference_id = str(prediction.get("event", ""))
            if inference_id not in metadata_by_inference_id:
                raise SeverityError("severity prediction has unknown recording identity")
            source = metadata_by_inference_id[inference_id]
            phone_index = prediction.get("phone_index")
            target = str(prediction.get("target", ""))
            gop = float(prediction.get("gop", math.nan))
            if type(phone_index) is not int or not 0 <= phone_index < len(
                source.utterance.canonical_phones
            ):
                raise SeverityError("severity prediction phone index is invalid")
            if target != source.utterance.canonical_phones[phone_index]:
                raise SeverityError("severity prediction target disagrees with prompt G2P")
            if not math.isfinite(gop):
                raise SeverityError("severity prediction GoP is non-finite")
            output.append(
                {
                    "schema_version": PHONE_SCORE_SCHEMA_VERSION,
                    "method": NEW_METHOD,
                    "cohort": TRAINING_COHORT,
                    "fold": fold.fold_id,
                    "speaker": source.speaker_id,
                    "event": source.event_id,
                    "recording_id": source.recording_id,
                    "phone_index": phone_index,
                    "canonical_phone": target,
                    "gop": gop,
                }
            )
        progress(
            f"severity fold {index}/{len(folds)} {fold.held_out_patient}: "
            f"{len(targets)} recordings, {expected_tokens} phones"
        )

    output.sort(
        key=lambda row: (
            str(row["speaker"]),
            str(row["event"]),
            str(row["recording_id"]),
            int(row["phone_index"]),
        )
    )
    seen_keys = {
        (str(row["recording_id"]), int(row["phone_index"])) for row in output
    }
    if len(seen_keys) != len(output):
        raise SeverityError("severity scoring produced duplicate phone rows")
    observed_recordings = {str(row["recording_id"]) for row in output}
    expected_recordings = {item.recording_id for item in audited_inputs}
    if observed_recordings != expected_recordings:
        raise SeverityError("severity scoring lost an audited recording")
    return output, fit_hashes


def _implementation_sha256() -> str:
    package = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for source in sorted(package.glob("*.py"), key=lambda value: value.name):
        name = source.name.encode("utf-8")
        digest.update(len(name).to_bytes(4, "big"))
        digest.update(name)
        digest.update(bytes.fromhex(sha256_file(source)))
    return digest.hexdigest()


def sensitivity_fit_reuse_plan(
    artifacts: str | Path,
    severity_folds: Sequence[OuterFold],
    labelled_patients: Sequence[str],
    healthy_references: Sequence[str],
) -> tuple[
    dict[str, OuterFold],
    dict[str, Sequence[str]],
    dict[str, str | Path],
]:
    """Plan exact sensitivity7 fold-cache reuse when its run is complete."""

    layout = ArtifactLayout.from_path(artifacts)
    completed = layout.evaluation / "sensitivity7" / "phone_predictions.jsonl.gz"
    if not completed.is_file():
        return {}, {}, {}
    severity_by_target = {fold.held_out_patient: fold for fold in severity_folds}
    shared_folds = nested_loso_folds(
        "sensitivity7", labelled_patients, healthy_references
    )
    overrides: dict[str, OuterFold] = {}
    methods: dict[str, Sequence[str]] = {}
    caches: dict[str, str | Path] = {}
    for fitting_fold in shared_folds:
        target = fitting_fold.held_out_patient
        declared = severity_by_target.get(target)
        if declared is None:
            raise SeverityError(f"sensitivity7 target is absent from severity LOSO: {target}")
        if (
            set(fitting_fold.training_patients) != set(declared.training_patients)
            or set(fitting_fold.healthy_references) != set(declared.healthy_references)
            or tuple(fitting_fold.inner_folds) != tuple(declared.inner_folds)
        ):
            raise SeverityError(
                f"sensitivity7 fold is not identical to severity fold for {target}"
            )
        overrides[target] = fitting_fold
        methods[target] = ALL_METHODS
        caches[target] = layout.cache
    return overrides, methods, caches


def _score_cache_key(
    config: Mapping[str, Any],
    *,
    training_manifest: Path,
    training_logit_index: Path,
    fold_document: Path,
    inference_manifest: Path,
    audited_inputs: Sequence[AuditedAcousticInput],
    implementation_sha256: str,
    audited_backend: Mapping[str, Any],
) -> str:
    acoustic_inventory = [
        {
            "recording_id": item.recording_id,
            "source_sha256": item.source_sha256,
            "canonical_phones": list(item.utterance.canonical_phones),
        }
        for item in audited_inputs
    ]
    return sha256_json(
        {
            "schema_version": PROVENANCE_SCHEMA_VERSION,
            "config_sha256": str(config["_config_hash"]),
            "implementation_sha256": implementation_sha256,
            "training_manifest_sha256": sha256_file(training_manifest),
            "training_logit_index_sha256": sha256_file(training_logit_index),
            "fold_document_sha256": sha256_file(fold_document),
            "inference_manifest_sha256": sha256_file(inference_manifest),
            "audited_acoustic_inventory": acoustic_inventory,
            "audited_backend_descriptor_sha256": audited_backend["descriptor_sha256"],
            "audited_checkpoint_weight_sha256": audited_backend[
                "checkpoint_weight_sha256"
            ],
            "audited_vocab_sha256": audited_backend["vocab_sha256"],
            "canonical_source": "frozen_local_espeak_prompt_g2p",
            "legacy_cached_canonical_ids_used": False,
            "severity_used_for_training": False,
        }
    )


def run_severity_scoring_stage(
    config: Mapping[str, Any] | None = None,
    *,
    progress: Progress = print,
    force: bool = False,
) -> dict[str, Any]:
    """Produce ``severity_phone_scores.jsonl.gz`` with eight outer LOSO fits."""

    cfg = load_config() if config is None else dict(config)
    build_severity_training_manifest_stage(cfg, progress=progress)
    extract_severity_training_logits_stage(cfg, progress=progress)

    severity = _require_mapping(cfg.get("severity"), "severity")
    paths = _require_mapping(cfg.get("paths"), "paths")
    layout = ArtifactLayout.from_path(paths["artifacts"])
    layout.create()
    audited = build_audited_severity_events(
        severity["audited_manifest"],
        logits_cache=severity["audited_logits_cache"],
        expected_manifest_rows=int(severity["audited_recordings"]),
        expected_events=int(severity["unique_reading_events"]),
        expected_speakers=len(tuple(severity["dysarthric_speakers"])),
        require_files=True,
    )
    audited_backend = validate_audited_backend_descriptor(
        cfg, severity["audited_logits_cache"]
    )
    inference_manifest = layout.manifests / "severity_audited_recordings.jsonl"
    write_severity_inference_manifest(audited["recordings"], inference_manifest)
    audited_inputs = load_audited_acoustic_inputs(
        audited["recordings"],
        vocab_size=int(_require_mapping(cfg.get("backend"), "backend")["vocab_size"]),
    )
    training_manifest = severity_training_manifest_path(cfg)
    training_rows = read_manifest(training_manifest)
    labelled_patients = tuple(sorted({
        row.speaker_id for row in training_rows if row.speaker_group == "dysarthric"
    }))
    healthy = tuple(str(value).upper() for value in _require_mapping(
        cfg.get("cohorts"), "cohorts"
    )["healthy_phn"])
    all_patients = tuple(str(value).upper() for value in severity["dysarthric_speakers"])
    folds = severity_loso_folds(all_patients, labelled_patients, healthy)

    training_index = layout.cache / "severity_training8.logits.index.json"
    fold_document = layout.manifests / "severity_training8.folds.json"
    implementation_hash = _implementation_sha256()
    cache_key = _score_cache_key(
        cfg,
        training_manifest=training_manifest,
        training_logit_index=training_index,
        fold_document=fold_document,
        inference_manifest=inference_manifest,
        audited_inputs=audited_inputs,
        implementation_sha256=implementation_hash,
        audited_backend=audited_backend,
    )
    score_path = layout.evaluation / "severity_phone_scores.jsonl.gz"
    provenance_path = layout.evaluation / "severity_phone_scores.provenance.json"
    reused = False
    provenance: Mapping[str, Any] | None = None
    if not force and score_path.is_file() and provenance_path.is_file():
        candidate = read_json(provenance_path)
        if isinstance(candidate, Mapping):
            provenance = candidate
            reused = (
                candidate.get("schema_version") == PROVENANCE_SCHEMA_VERSION
                and candidate.get("cache_key") == cache_key
                and candidate.get("output_sha256") == sha256_file(score_path)
            )
    fit_hashes: dict[str, str] = {}
    if reused:
        fit_hashes = {
            str(key): str(value)
            for key, value in _require_mapping(
                provenance.get("fold_fit_hashes") if provenance else None,
                "cached fold_fit_hashes",
            ).items()
        }
        progress("severity phone scores: reused exact content-bound artifact")
    else:
        fold_cache = layout.cache / "severity" / implementation_hash[:16]
        # Seven folds have exactly the same training speakers and inner-LOSO
        # population as the completed sensitivity7 headline experiment.  When
        # that artifact exists, reuse its validated full-method fold fits and
        # score only the adapted method.  M03 still receives a new fold fitted
        # on all seven PHN-labelled patients.  If sensitivity7 has not run,
        # severity remains independently executable with adapted-only fits.
        fit_overrides, fit_methods, fit_caches = sensitivity_fit_reuse_plan(
            layout.root, folds, labelled_patients, healthy
        )
        if fit_overrides:
            progress(
                "severity scoring: reusing seven validated sensitivity7 outer fits; "
                "fitting one additional M03 fold"
            )
        rows, fit_hashes = generate_severity_phone_scores(
            cfg,
            training_rows,
            audited_inputs,
            folds,
            fold_cache_dir=fold_cache,
            progress=progress,
            fit_fold_overrides=fit_overrides,
            fit_methods_by_speaker=fit_methods,
            fit_cache_by_speaker=fit_caches,
            require_cached_fit_targets=tuple(fit_overrides),
        )
        write_jsonl_gz(score_path, rows)
        provenance = {
            "schema_version": PROVENANCE_SCHEMA_VERSION,
            "cache_key": cache_key,
            "output_sha256": sha256_file(score_path),
            "config_sha256": str(cfg["_config_hash"]),
            "implementation_sha256": implementation_hash,
            "fold_fit_hashes": dict(sorted(fit_hashes.items())),
            "n_recordings": len(audited_inputs),
            "n_events": int(audited["n_events"]),
            "n_phone_rows": len(rows),
            "canonical_source": "frozen_local_espeak_prompt_g2p",
            "legacy_cached_canonical_ids_used": False,
            "legacy_cached_canonical_ids_mismatched": sum(
                not item.legacy_canonical_ids_match for item in audited_inputs
            ),
            "phn_or_textgrid_used_at_inference": False,
            "severity_used_for_training": False,
            "audited_backend": audited_backend,
        }
        write_json(provenance_path, provenance)

    output_hash = sha256_file(score_path)
    summary = write_stage_summary(
        layout.summaries / "score-severity.json",
        stage="score-severity",
        config_sha256=str(cfg["_config_hash"]),
        training_speakers=(*labelled_patients, *healthy),
        source_files={
            "audited_manifest": sha256_file(severity["audited_manifest"]),
            "frozen_config": str(cfg["_config_hash"]),
        },
        models={"implementation": implementation_hash},
        manifests={
            TRAINING_COHORT: sha256_file(training_manifest),
            "severity_audited_recordings": sha256_file(inference_manifest),
        },
        upstream_artifacts={
            "severity_training_logit_index": sha256_file(training_index),
        },
        exclusions={"phn_unavailable_training_speaker": len(set(all_patients) - set(labelled_patients))},
        outputs={
            "severity_phone_scores": output_hash,
            "severity_phone_scores_provenance": sha256_file(provenance_path),
        },
        details={
            "reused": reused,
            "n_recordings": len(audited_inputs),
            "n_events": int(audited["n_events"]),
            "n_outer_folds": len(folds),
            "fold_fit_hashes": dict(sorted(fit_hashes.items())),
            "phn_labelled_patients": list(labelled_patients),
            "phn_unavailable_patients": sorted(set(all_patients) - set(labelled_patients)),
            "legacy_cached_canonical_ids_mismatched": sum(
                not item.legacy_canonical_ids_match for item in audited_inputs
            ),
            "canonical_source": "frozen_local_espeak_prompt_g2p",
            "legacy_cached_canonical_ids_used": False,
            "phn_or_textgrid_used_at_inference": False,
            "severity_used_for_training": False,
            "audited_backend": audited_backend,
        },
    )
    return summary


__all__ = [
    "AuditedAcousticInput",
    "build_severity_training_manifest_stage",
    "extract_severity_training_logits_stage",
    "generate_severity_phone_scores",
    "load_audited_acoustic_inputs",
    "run_severity_scoring_stage",
    "sensitivity_fit_reuse_plan",
    "severity_loso_folds",
    "severity_training_manifest_path",
    "validate_audited_backend_descriptor",
]
