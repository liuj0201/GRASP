"""Manifest and raw-logit stages shared by the command-line interface."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Sequence

from .artifacts import ArtifactLayout, write_jsonl_gz, write_stage_summary
from .backend import (
    OfficialCTCSFBackend,
    build_backend_hashes,
    build_cache_binding,
    build_logits_cache_descriptor,
    load_audio,
    load_logits_cache,
    load_vocab,
    migrate_legacy_logits_cache,
    save_logits_cache,
)
from .config import config_hash, load_config
from .data import build_manifest, read_manifest, write_audit
from .folds import folds_document, nested_loso_folds
from .provenance import companion_descriptor_path, sha256_file, write_json


Progress = Callable[[str], None]


def _raw_frozen_config(config: dict[str, Any]) -> dict[str, Any]:
    source = Path(config["_config_path"])
    return json.loads(source.read_text(encoding="utf-8"))


def layout_from_config(config: dict[str, Any]) -> ArtifactLayout:
    layout = ArtifactLayout.from_path(config["paths"]["artifacts"])
    layout.create()
    return layout


def manifest_path(config: dict[str, Any], cohort: str) -> Path:
    return layout_from_config(config).manifests / f"{cohort}.jsonl.gz"


def logit_cache_path(config: dict[str, Any], reading_event_id: str) -> Path:
    digest = hashlib.sha256(reading_event_id.encode("utf-8")).hexdigest()[:24]
    return layout_from_config(config).cache / "logits" / f"{digest}.npz"


def build_manifests_stage(
    config: dict[str, Any] | None = None,
    *,
    cohorts: Sequence[str] = ("primary5", "sensitivity7"),
    progress: Progress = print,
) -> dict[str, dict[str, Any]]:
    cfg = load_config() if config is None else config
    layout = layout_from_config(cfg)
    summaries: dict[str, dict[str, Any]] = {}
    for cohort in cohorts:
        if cohort not in cfg["cohorts"] or cohort == "healthy_phn":
            raise ValueError(f"unknown patient cohort: {cohort}")
        progress(f"building manifest: {cohort}")
        rows, exclusions, audit = build_manifest(
            cfg["paths"]["data_root"],
            cohort=cohort,
            healthy_speakers=cfg["cohorts"]["healthy_phn"],
            minimum_stable_fraction=cfg["labels"]["stable_alignment_min_fraction"],
            allow_cross_mic_phn_for=cfg["audio_policy"]["allow_cross_mic_phn_sequence_for"],
        )
        output = layout.manifests / f"{cohort}.jsonl.gz"
        exclusion_path = layout.manifests / f"{cohort}.exclusions.jsonl.gz"
        audit_path = layout.manifests / f"{cohort}.audit.json"
        fold_path = layout.manifests / f"{cohort}.folds.json"
        write_jsonl_gz(output, (row.to_dict() for row in rows))
        write_jsonl_gz(exclusion_path, (row.to_dict() for row in exclusions))
        write_audit(audit, audit_path)
        folds = nested_loso_folds(
            cohort,
            cfg["cohorts"][cohort],
            cfg["cohorts"]["healthy_phn"],
        )
        write_json(fold_path, folds_document(folds))
        output_hashes = {
            "manifest": sha256_file(output),
            "exclusions": sha256_file(exclusion_path),
            "audit": sha256_file(audit_path),
            "folds": sha256_file(fold_path),
        }
        input_hashes = audit.get("input_source_hashes")
        g2p = audit.get("g2p_provenance")
        if not isinstance(input_hashes, dict) or not isinstance(g2p, dict):
            raise ValueError("manifest audit lacks source/G2P provenance")
        source_files = {"frozen_config": cfg["_config_hash"]}
        for role in ("wav", "prompt", "phn"):
            record = input_hashes.get(role)
            if not isinstance(record, dict) or not isinstance(
                record.get("inventory_sha256"), str
            ):
                raise ValueError(f"manifest audit lacks {role} source hash inventory")
            source_files[f"{role}_input_inventory"] = record["inventory_sha256"]
        # Stage-summary model inventories are digest-only.  Human-readable
        # runtime versions remain available in details.g2p_provenance.
        models = {"g2p_descriptor": str(g2p["descriptor_sha256"])}
        for model_name, provenance_key in (
            ("espeak_library", "espeak_library_sha256"),
            ("espeak_data", "espeak_data_sha256"),
        ):
            digest = g2p.get(provenance_key)
            if isinstance(digest, str):
                models[model_name] = digest
        summary_path = layout.summaries / f"build-manifests.{cohort}.json"
        summary = write_stage_summary(
            summary_path,
            stage="build-manifests",
            config_sha256=cfg["_config_hash"],
            training_speakers=[],
            source_files=source_files,
            models=models,
            manifests={cohort: output_hashes["manifest"]},
            upstream_artifacts={},
            exclusions=Counter(item.reason for item in exclusions),
            outputs=output_hashes,
            details={"cohort": cohort, **audit},
        )
        summaries[cohort] = summary
        progress(
            f"{cohort}: {audit['n_rows']} utterances, "
            f"stable={100.0 * float(audit['stable_fraction']):.2f}%"
        )
    return summaries


def _logit_descriptor(
    *,
    artifact: Path,
    audio_path: Path,
    binding: dict[str, Any],
    raw_config: dict[str, Any],
    event_id: str,
    canonical_ids: Sequence[int],
) -> dict[str, Any]:
    if config_hash(raw_config) != binding["config_sha256"]:
        raise ValueError("raw-logit descriptor config disagrees with binding")
    paths = binding.get("source_paths")
    if not isinstance(paths, dict) or Path(str(paths.get("audio", ""))).resolve() != audio_path.resolve():
        raise ValueError("raw-logit descriptor audio path disagrees with binding")
    return build_logits_cache_descriptor(
        artifact,
        utterance_id=event_id,
        canonical_ids=canonical_ids,
        binding=binding,
    )


def extract_logits_stage(
    config: dict[str, Any] | None = None,
    *,
    cohort: str = "sensitivity7",
    limit: int | None = None,
    progress: Progress = print,
) -> dict[str, Any]:
    """Extract content-bound official CTC-SF logits for a frozen manifest."""

    cfg = load_config() if config is None else config
    layout = layout_from_config(cfg)
    source_manifest = manifest_path(cfg, cohort)
    if not source_manifest.is_file():
        raise FileNotFoundError(
            f"manifest is absent: {source_manifest}; run build-manifests first"
        )
    all_rows = read_manifest(source_manifest)
    rows = all_rows if limit is None else all_rows[: int(limit)]
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    checkpoint = Path(cfg["paths"]["checkpoint"])
    processor = Path(cfg["paths"]["processor"])
    backend_hashes = build_backend_hashes(checkpoint=checkpoint, processor=processor)
    phone_to_id, _id_to_phone, processor_blank_id = load_vocab(processor)
    if processor_blank_id != int(cfg["backend"]["blank_id"]):
        raise ValueError("frozen blank id disagrees with processor")
    if len(phone_to_id) != int(cfg["backend"]["vocab_size"]):
        raise ValueError("frozen vocabulary size disagrees with processor")
    raw_config = _raw_frozen_config(cfg)
    backend: OfficialCTCSFBackend | None = None
    created = 0
    reused = 0
    migrated = 0
    cache_hashes: dict[str, str] = {}
    for index, row in enumerate(rows, start=1):
        destination = logit_cache_path(cfg, row.reading_event_id)
        try:
            canonical_ids = tuple(phone_to_id[phone] for phone in row.canonical_phones)
        except KeyError as error:
            raise ValueError(
                f"canonical phone is outside processor vocabulary: {error.args[0]}"
            ) from error
        binding = build_cache_binding(
            audio_path=row.wav_path,
            checkpoint=checkpoint,
            processor=processor,
            experiment_config=cfg["_config_hash"],
            canonical_ids=canonical_ids,
            config_path=cfg["_config_path"],
            backend_hashes=backend_hashes,
        )
        valid = False
        if destination.is_file():
            try:
                cached = load_logits_cache(
                    destination, expected_fingerprint=binding["fingerprint"]
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
                        destination, expected_fingerprint=binding["fingerprint"]
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
                    device=cfg["backend"]["device"],
                )
                if backend.blank_id != cfg["backend"]["blank_id"]:
                    raise ValueError("frozen blank id disagrees with processor")
                if backend.vocab_size != cfg["backend"]["vocab_size"]:
                    raise ValueError("frozen vocabulary size disagrees with processor")
            waveform = load_audio(row.wav_path, cfg["backend"]["sample_rate"])
            logits = backend.infer(waveform, cfg["backend"]["sample_rate"])
            backend_canonical_ids = backend.canonical_ids(row.canonical_phones)
            if tuple(int(value) for value in backend_canonical_ids) != canonical_ids:
                raise ValueError("backend and processor canonical ids disagree")
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
        cache_hashes[row.reading_event_id] = sha256_file(destination)
        if index == 1 or index % 25 == 0 or index == len(rows):
            progress(f"logits {index}/{len(rows)} (new={created}, reused={reused})")

    index_path = layout.cache / "logits.index.json"
    index_document = {
        "schema_version": "da-cf-gop.logit-index.v1",
        "cohort": cohort,
        "manifest_sha256": sha256_file(source_manifest),
        "backend_sha256": backend_hashes,
        "config_sha256": cfg["_config_hash"],
        "entries": dict(sorted(cache_hashes.items())),
        "partial_limit": limit,
    }
    write_json(index_path, index_document)
    summary = write_stage_summary(
        layout.summaries / "extract-logits.json",
        stage="extract-logits",
        config_sha256=cfg["_config_hash"],
        training_speakers=[],
        source_files={"frozen_config": cfg["_config_hash"]},
        models=backend_hashes,
        manifests={cohort: sha256_file(source_manifest)},
        upstream_artifacts={},
        exclusions={},
        outputs={"logit_index": sha256_file(index_path)},
        details={
            "cohort": cohort,
            "n_rows": len(rows),
            "created": created,
            "reused": reused,
            "migrated_legacy_bindings": migrated,
            "partial_limit": limit,
            "all_audio_inputs_headmic": True,
            "external_alignment_used": False,
        },
    )
    return summary
