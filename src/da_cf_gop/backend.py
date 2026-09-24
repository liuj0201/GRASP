"""Frozen CTC-SF acoustic backend and content-addressed logit caches.

This module is intentionally self contained.  In particular, it never imports
from the historical ``phoneme_project/code*`` trees.  Heavy ML dependencies are
loaded lazily so that manifest construction and cache verification stay cheap.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .config import config_hash as frozen_config_hash
from .provenance import (
    CACHE_SCHEMA,
    companion_descriptor_path,
    path_record,
    read_json,
    sha256_json as provenance_sha256_json,
    validate_cache_descriptor,
    write_deterministic_npz,
    write_json,
)


_HASH_CHUNK_SIZE = 8 * 1024 * 1024


def sha256_file(path: str | Path, chunk_size: int = _HASH_CHUNK_SIZE) -> str:
    """Return the SHA-256 digest of a regular file."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"not a regular file: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    """Hash a JSON value using a deterministic, UTF-8 representation."""

    payload = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_path(path: str | Path) -> str:
    """Hash a file or a directory tree, including relative file names.

    Directory metadata and mtimes are deliberately excluded.  Renaming a model
    file or changing any byte changes the digest, while copying an identical
    checkpoint to another directory does not.
    """

    source = Path(path)
    if source.is_file():
        return sha256_file(source)
    if not source.is_dir():
        raise FileNotFoundError(f"path does not exist: {source}")
    digest = hashlib.sha256()
    files = sorted(item for item in source.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"cannot hash an empty directory: {source}")
    for item in files:
        relative = item.relative_to(source).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        file_digest = bytes.fromhex(sha256_file(item))
        digest.update(file_digest)
    return digest.hexdigest()


def _config_file_hash(path: str | Path) -> str:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot hash frozen JSON configuration: {source}") from error
    # Frozen configurations have their own canonical digest contract.  The
    # general artifact JSON serializer intentionally appends a newline, while
    # ``config_hash`` does not; mixing the two makes a live config reject the
    # digest declared by ``load_config``.
    return frozen_config_hash(value)


def _canonical_ids_hash(values: Iterable[int]) -> str:
    return sha256_json([int(value) for value in values])


@lru_cache(maxsize=16)
def _validate_large_source(path_text: str, expected_sha256: str) -> None:
    """Hash immutable model trees once per process and compare to the binding."""

    if sha256_path(path_text) != expected_sha256:
        raise ValueError(f"cache source hash changed: {path_text}")


def _vocab_path(processor: str | Path) -> Path:
    candidate = Path(processor)
    return candidate if candidate.is_file() else candidate / "vocab.json"


def load_vocab(
    processor: str | Path,
    *,
    blank_token: str = "<pad>",
) -> tuple[dict[str, int], dict[int, str], int]:
    """Load a contiguous vocabulary and return both maps and the blank id.

    The released checkpoint uses ``<pad>`` as CTC blank, but the implementation
    is dynamic and therefore also supports test vocabularies with another id.
    """

    path = _vocab_path(processor)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not raw:
        raise ValueError("vocab.json must contain a non-empty JSON object")
    phone_to_id = {str(phone): int(index) for phone, index in raw.items()}
    if len(set(phone_to_id.values())) != len(phone_to_id):
        raise ValueError("vocabulary ids must be unique")
    expected = list(range(len(phone_to_id)))
    if sorted(phone_to_id.values()) != expected:
        raise ValueError("vocabulary ids must be contiguous from zero")
    if blank_token not in phone_to_id:
        raise ValueError(f"vocabulary has no declared blank token {blank_token!r}")
    id_to_phone = {index: phone for phone, index in phone_to_id.items()}
    return phone_to_id, id_to_phone, phone_to_id[blank_token]


def load_audio(path: str | Path, target_sample_rate: int = 16_000) -> np.ndarray:
    """Read, downmix and (when necessary) resample a finite waveform."""

    import soundfile as sf
    from scipy.signal import resample_poly

    audio, sample_rate = sf.read(str(path), always_2d=True, dtype="float32")
    waveform = audio.mean(axis=1)
    if sample_rate != target_sample_rate:
        divisor = int(np.gcd(sample_rate, target_sample_rate))
        waveform = resample_poly(
            waveform,
            target_sample_rate // divisor,
            sample_rate // divisor,
        ).astype(np.float32)
    if waveform.size == 0 or not np.isfinite(waveform).all():
        raise ValueError(f"audio must be non-empty and finite: {path}")
    return np.ascontiguousarray(waveform, dtype=np.float32)


def build_cache_binding(
    *,
    audio_path: str | Path,
    checkpoint: str | Path,
    processor: str | Path,
    experiment_config: Mapping[str, Any] | str,
    canonical_ids: Iterable[int],
    config_path: str | Path | None = None,
    backend_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build the immutable provenance binding for one raw-logit cache.

    ``experiment_config`` may be a parsed mapping or the already-computed
    SHA-256 of the frozen configuration.  The returned fingerprint covers the
    audio, all checkpoint files, all processor files (including vocabulary),
    the experiment configuration, and the exact canonical id sequence.  Live
    source paths are retained so a scoring-only consumer can validate the
    binding without relying on its caller to reconstruct a fingerprint.
    """

    config_digest = (
        str(experiment_config)
        if isinstance(experiment_config, str)
        else frozen_config_hash(dict(experiment_config))
    )
    if len(config_digest) != 64 or any(
        char not in "0123456789abcdef" for char in config_digest.lower()
    ):
        raise ValueError("experiment_config string must be a hexadecimal SHA-256 digest")
    labels = tuple(int(value) for value in canonical_ids)
    if not labels or any(value < 0 for value in labels):
        raise ValueError("canonical_ids must be a non-empty non-negative sequence")
    source_hashes = (
        build_backend_hashes(checkpoint=checkpoint, processor=processor)
        if backend_hashes is None
        else dict(backend_hashes)
    )
    required_hashes = {"checkpoint_sha256", "processor_sha256", "vocab_sha256"}
    if set(source_hashes) != required_hashes:
        raise ValueError(f"backend_hashes must contain exactly {sorted(required_hashes)}")
    if any(
        len(value) != 64 or any(char not in "0123456789abcdef" for char in value.lower())
        for value in source_hashes.values()
    ):
        raise ValueError("backend hashes must be hexadecimal SHA-256 digests")
    audio = Path(audio_path).resolve()
    checkpoint_path = Path(checkpoint).resolve()
    processor_path = Path(processor).resolve()
    vocab_path = _vocab_path(processor_path).resolve()
    frozen_config_path = Path(config_path).resolve() if config_path is not None else None
    if frozen_config_path is not None:
        observed_config_hash = _config_file_hash(frozen_config_path)
        if observed_config_hash != config_digest.lower():
            raise ValueError("frozen config file does not match experiment_config")
    binding: dict[str, Any] = {
        "schema_version": "da-cf-gop.logit-binding.v2",
        "audio_sha256": sha256_file(audio_path),
        **{key: value.lower() for key, value in source_hashes.items()},
        "config_sha256": config_digest.lower(),
        "canonical_ids_sha256": _canonical_ids_hash(labels),
        "source_paths": {
            "audio": audio.as_posix(),
            "checkpoint": checkpoint_path.as_posix(),
            "processor": processor_path.as_posix(),
            "vocab": vocab_path.as_posix(),
            "config": frozen_config_path.as_posix() if frozen_config_path is not None else None,
        },
    }
    binding["fingerprint"] = sha256_json(binding)
    return binding


def build_backend_hashes(
    *, checkpoint: str | Path, processor: str | Path
) -> dict[str, str]:
    """Hash frozen model sources once for reuse across all utterance bindings."""

    return {
        "checkpoint_sha256": sha256_path(checkpoint),
        "processor_sha256": sha256_path(processor),
        "vocab_sha256": sha256_file(_vocab_path(processor)),
    }


@dataclass(frozen=True)
class CachedLogits:
    utterance_id: str
    logits: np.ndarray
    canonical_ids: np.ndarray
    duration_s: float
    binding: dict[str, Any]
    metadata: dict[str, Any]


_LOGIT_BINDING_KEYS = frozenset(
    {
        "schema_version",
        "audio_sha256",
        "checkpoint_sha256",
        "processor_sha256",
        "vocab_sha256",
        "config_sha256",
        "canonical_ids_sha256",
        "source_paths",
        "fingerprint",
    }
)


def _validate_binding_shape(binding: Mapping[str, Any]) -> None:
    if set(binding) != _LOGIT_BINDING_KEYS:
        raise ValueError(
            "legacy or incomplete raw-logit binding; rerun extract-logits"
        )
    if binding.get("schema_version") != "da-cf-gop.logit-binding.v2":
        raise ValueError("raw-logit binding schema changed")
    for name in (
        "audio_sha256",
        "checkpoint_sha256",
        "processor_sha256",
        "vocab_sha256",
        "config_sha256",
        "canonical_ids_sha256",
        "fingerprint",
    ):
        value = binding.get(name)
        if not isinstance(value, str) or len(value) != 64 or any(
            char not in "0123456789abcdef" for char in value.lower()
        ):
            raise ValueError(f"invalid raw-logit binding hash: {name}")
    paths = binding.get("source_paths")
    if not isinstance(paths, Mapping) or set(paths) != {
        "audio", "checkpoint", "processor", "vocab", "config"
    }:
        raise ValueError("raw-logit binding source paths are incomplete")
    if any(not isinstance(paths[name], str) or not str(paths[name]) for name in (
        "audio", "checkpoint", "processor", "vocab"
    )):
        raise ValueError("raw-logit binding has an invalid source path")
    if paths["config"] is not None and (
        not isinstance(paths["config"], str) or not str(paths["config"])
    ):
        raise ValueError("raw-logit binding has an invalid config path")


def _validate_current_binding_sources(
    binding: Mapping[str, Any],
    canonical_ids: Iterable[int],
    *,
    require_config_path: bool,
) -> None:
    _validate_binding_shape(binding)
    paths = binding["source_paths"]
    assert isinstance(paths, Mapping)
    if sha256_file(str(paths["audio"])) != binding["audio_sha256"]:
        raise ValueError("raw-logit cache audio hash changed")
    _validate_large_source(str(paths["checkpoint"]), str(binding["checkpoint_sha256"]))
    _validate_large_source(str(paths["processor"]), str(binding["processor_sha256"]))
    if sha256_file(str(paths["vocab"])) != binding["vocab_sha256"]:
        raise ValueError("raw-logit cache vocabulary hash changed")
    config_source = paths["config"]
    if config_source is None:
        if require_config_path:
            raise ValueError(
                "raw-logit cache has no live frozen-config path; rerun extract-logits"
            )
    elif _config_file_hash(str(config_source)) != binding["config_sha256"]:
        raise ValueError("raw-logit cache frozen-config hash changed")
    if _canonical_ids_hash(canonical_ids) != binding["canonical_ids_sha256"]:
        raise ValueError("raw-logit cache canonical sequence changed")


def build_logits_cache_descriptor(
    artifact: str | Path,
    *,
    utterance_id: str,
    canonical_ids: Iterable[int],
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the companion descriptor checked by every cache consumer."""

    labels = tuple(int(value) for value in canonical_ids)
    _validate_binding_shape(binding)
    if _canonical_ids_hash(labels) != binding["canonical_ids_sha256"]:
        raise ValueError("descriptor canonical sequence disagrees with cache binding")
    paths = binding["source_paths"]
    assert isinstance(paths, Mapping)
    body: dict[str, Any] = {
        "schema_version": CACHE_SCHEMA,
        "artifact_type": "raw_ctc_logits",
        "artifact": path_record(artifact),
        "config_sha256": str(binding["config_sha256"]),
        "fold": None,
        "sources": {"audio": path_record(str(paths["audio"]))},
        "parameters": {
            "reading_event_id_sha256": hashlib.sha256(
                str(utterance_id).encode("utf-8")
            ).hexdigest(),
            "canonical_phone_count": len(labels),
            "canonical_ids_sha256": str(binding["canonical_ids_sha256"]),
            "binding_fingerprint": str(binding["fingerprint"]),
            "stored_dtype": "float32",
        },
        "upstream_sha256": {
            key: str(binding[key])
            for key in ("checkpoint_sha256", "processor_sha256", "vocab_sha256")
        },
    }
    body["fingerprint"] = provenance_sha256_json(body)
    return body


def save_logits_cache(
    path: str | Path,
    *,
    utterance_id: str,
    logits: np.ndarray,
    canonical_ids: Iterable[int],
    duration_s: float,
    binding: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Write a pickle-free compressed NPZ cache."""

    values = np.asarray(logits, dtype=np.float32)
    labels = np.asarray(list(canonical_ids), dtype=np.int64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("logits must be a non-empty [frames, vocabulary] matrix")
    if not np.isfinite(values).all():
        raise ValueError("logits must be finite")
    if labels.ndim != 1 or np.any(labels < 0) or np.any(labels >= values.shape[1]):
        raise ValueError("canonical ids must lie inside the logit vocabulary")
    if not np.isfinite(duration_s) or float(duration_s) <= 0:
        raise ValueError("duration_s must be finite and positive")
    binding_dict = dict(binding)
    _validate_binding_shape(binding_dict)
    expected = sha256_json({key: binding_dict[key] for key in binding_dict if key != "fingerprint"})
    if binding_dict.get("fingerprint") != expected:
        raise ValueError("cache binding fingerprint is absent or inconsistent")
    if _canonical_ids_hash(labels.tolist()) != binding_dict["canonical_ids_sha256"]:
        raise ValueError("cache binding does not cover the stored canonical ids")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_deterministic_npz(
        destination,
        {
        "utterance_id": np.asarray(str(utterance_id)),
        "logits": values,
        "canonical_ids": labels,
        "duration_s": np.asarray(float(duration_s), dtype=np.float64),
        "binding_json": np.asarray(json.dumps(binding_dict, sort_keys=True, separators=(",", ":"))),
        "metadata_json": np.asarray(
            json.dumps(dict(metadata or {}), ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
        ),
        },
    )
    write_json(
        companion_descriptor_path(destination),
        build_logits_cache_descriptor(
            destination,
            utterance_id=str(utterance_id),
            canonical_ids=labels.tolist(),
            binding=binding_dict,
        ),
    )
    return destination


def load_logits_cache(
    path: str | Path,
    *,
    expected_fingerprint: str | None = None,
) -> CachedLogits:
    """Load and fully validate a raw-logit cache."""

    with np.load(path, allow_pickle=False) as archive:
        required = {
            "utterance_id", "logits", "canonical_ids", "duration_s", "binding_json", "metadata_json"
        }
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"logit cache is missing fields: {sorted(missing)}")
        utterance_id = str(archive["utterance_id"].item())
        logits = archive["logits"].astype(np.float32, copy=True)
        canonical_ids = archive["canonical_ids"].astype(np.int64, copy=True)
        duration_s = float(archive["duration_s"].item())
        binding = json.loads(str(archive["binding_json"].item()))
        metadata = json.loads(str(archive["metadata_json"].item()))
    if logits.ndim != 2 or logits.shape[0] == 0 or logits.shape[1] == 0 or not np.isfinite(logits).all():
        raise ValueError("cached logits are invalid")
    if canonical_ids.ndim != 1 or np.any(canonical_ids < 0) or np.any(canonical_ids >= logits.shape[1]):
        raise ValueError("cached canonical ids are invalid")
    _validate_binding_shape(binding)
    actual_fingerprint = sha256_json({key: binding[key] for key in binding if key != "fingerprint"})
    if binding.get("fingerprint") != actual_fingerprint:
        raise ValueError("cached provenance fingerprint is inconsistent")
    if expected_fingerprint is not None and actual_fingerprint != expected_fingerprint:
        raise ValueError("cache fingerprint does not match the current inputs")
    _validate_current_binding_sources(
        binding,
        canonical_ids.tolist(),
        # Scoring consumers intentionally omit an expected fingerprint; in
        # that path the cache itself must point to the live frozen config.
        require_config_path=expected_fingerprint is None,
    )
    descriptor_path = companion_descriptor_path(path)
    if not descriptor_path.is_file():
        raise ValueError(f"raw-logit cache descriptor is missing: {descriptor_path}")
    descriptor = read_json(descriptor_path)
    if not isinstance(descriptor, Mapping):
        raise ValueError("raw-logit cache descriptor is not an object")
    artifact = validate_cache_descriptor(
        descriptor,
        expected_artifact_type="raw_ctc_logits",
        required_source_roles=("audio",),
    )
    if artifact.resolve() != Path(path).resolve():
        raise ValueError("raw-logit descriptor points to a different artifact")
    if descriptor.get("config_sha256") != binding["config_sha256"]:
        raise ValueError("raw-logit descriptor config binding changed")
    if descriptor.get("upstream_sha256") != {
        key: binding[key]
        for key in ("checkpoint_sha256", "processor_sha256", "vocab_sha256")
    }:
        raise ValueError("raw-logit descriptor backend binding changed")
    parameters = descriptor.get("parameters")
    if not isinstance(parameters, Mapping) or (
        parameters.get("canonical_ids_sha256") != binding["canonical_ids_sha256"]
        or parameters.get("binding_fingerprint") != binding["fingerprint"]
        or parameters.get("canonical_phone_count") != len(canonical_ids)
        or parameters.get("reading_event_id_sha256")
        != hashlib.sha256(utterance_id.encode("utf-8")).hexdigest()
    ):
        raise ValueError("raw-logit descriptor canonical/event binding changed")
    return CachedLogits(utterance_id, logits, canonical_ids, duration_s, binding, metadata)


def migrate_legacy_logits_cache(
    path: str | Path,
    *,
    expected_utterance_id: str,
    expected_canonical_ids: Iterable[int],
    new_binding: Mapping[str, Any],
) -> bool:
    """Upgrade a fully matching v1 cache without rerunning the acoustic model.

    Migration is deliberately available only to the explicit extraction stage.
    Scoring remains read-only and fails closed on legacy provenance.
    """

    source = Path(path)
    descriptor_path = companion_descriptor_path(source)
    if not source.is_file() or not descriptor_path.is_file():
        return False
    expected_ids = tuple(int(value) for value in expected_canonical_ids)
    try:
        descriptor = read_json(descriptor_path)
        if not isinstance(descriptor, Mapping):
            return False
        artifact = validate_cache_descriptor(
            descriptor,
            expected_artifact_type="raw_ctc_logits",
            required_source_roles=("audio",),
        )
        if artifact.resolve() != source.resolve():
            return False
        with np.load(source, allow_pickle=False) as archive:
            required = {
                "utterance_id",
                "logits",
                "canonical_ids",
                "duration_s",
                "binding_json",
                "metadata_json",
            }
            if set(archive.files) != required:
                return False
            utterance_id = str(archive["utterance_id"].item())
            logits = archive["logits"].astype(np.float32, copy=True)
            canonical_ids = tuple(
                int(value) for value in archive["canonical_ids"].astype(np.int64).tolist()
            )
            duration_s = float(archive["duration_s"].item())
            old_binding = json.loads(str(archive["binding_json"].item()))
            metadata = json.loads(str(archive["metadata_json"].item()))
        legacy_keys = {
            "audio_sha256",
            "checkpoint_sha256",
            "processor_sha256",
            "vocab_sha256",
            "config_sha256",
            "fingerprint",
        }
        if not isinstance(old_binding, dict) or set(old_binding) != legacy_keys:
            return False
        old_fingerprint = sha256_json(
            {key: old_binding[key] for key in old_binding if key != "fingerprint"}
        )
        if old_binding.get("fingerprint") != old_fingerprint:
            return False
        if utterance_id != str(expected_utterance_id) or canonical_ids != expected_ids:
            return False
        if logits.ndim != 2 or logits.shape[0] == 0 or not np.isfinite(logits).all():
            return False
        if not np.isfinite(duration_s) or duration_s <= 0 or not isinstance(metadata, dict):
            return False
        _validate_current_binding_sources(
            new_binding, expected_ids, require_config_path=True
        )
        if any(
            old_binding[key] != new_binding[key]
            for key in (
                "audio_sha256",
                "checkpoint_sha256",
                "processor_sha256",
                "vocab_sha256",
                "config_sha256",
            )
        ):
            return False
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False
    save_logits_cache(
        source,
        utterance_id=utterance_id,
        logits=logits,
        canonical_ids=canonical_ids,
        duration_s=duration_s,
        binding=new_binding,
        metadata=metadata,
    )
    return True


class OfficialCTCSFBackend:
    """Local-only wrapper around the authors' Wav2Vec2 CTC checkpoint."""

    def __init__(
        self,
        checkpoint: str | Path,
        processor: str | Path,
        *,
        blank_token: str = "<pad>",
        device: str = "auto",
    ) -> None:
        import torch
        from transformers import AutoFeatureExtractor, Wav2Vec2ForCTC

        self.checkpoint = Path(checkpoint)
        self.processor = Path(processor)
        self.phone_to_id, self.id_to_phone, self.blank_id = load_vocab(
            self.processor, blank_token=blank_token
        )
        self.vocab_size = len(self.phone_to_id)
        selected = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
        self.device = torch.device(selected)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            self.processor, local_files_only=True
        )
        self.model = Wav2Vec2ForCTC.from_pretrained(
            self.checkpoint, local_files_only=True
        )
        if int(self.model.config.vocab_size) != self.vocab_size:
            raise ValueError("checkpoint and processor vocabulary sizes differ")
        configured_blank = getattr(self.model.config, "pad_token_id", None)
        if configured_blank is not None and int(configured_blank) != self.blank_id:
            raise ValueError("checkpoint and processor blank ids differ")
        self.model.eval().to(self.device)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def canonical_ids(self, phones: Iterable[str]) -> np.ndarray:
        result: list[int] = []
        for phone in phones:
            if phone not in self.phone_to_id or self.phone_to_id[phone] == self.blank_id:
                raise ValueError(f"canonical phone is outside the scoring inventory: {phone}")
            result.append(self.phone_to_id[phone])
        return np.asarray(result, dtype=np.int64)

    def descriptor(self) -> dict[str, Any]:
        import torch

        descriptor = {
            "backend": "official_ctc_sf_local_checkpoint",
            "checkpoint_sha256": sha256_path(self.checkpoint),
            "processor_sha256": sha256_path(self.processor),
            "vocab_sha256": sha256_file(_vocab_path(self.processor)),
            "blank_id": self.blank_id,
            "vocab_size": self.vocab_size,
            "torch_version": torch.__version__,
            "device": str(self.device),
            "inference_dtype": "float32",
        }
        descriptor["descriptor_sha256"] = sha256_json(descriptor)
        return descriptor

    def infer(self, waveform: np.ndarray, sampling_rate: int = 16_000) -> np.ndarray:
        import torch

        values = np.asarray(waveform, dtype=np.float32)
        if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
            raise ValueError("waveform must be a non-empty finite vector")
        encoded = self.feature_extractor(
            values, sampling_rate=sampling_rate, return_tensors="pt"
        )
        inputs = encoded.input_values.to(self.device)
        attention_mask = getattr(encoded, "attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        with torch.inference_mode():
            logits = self.model(input_values=inputs, attention_mask=attention_mask).logits[0]
        result = logits.float().cpu().numpy()
        if result.ndim != 2 or result.shape[1] != self.vocab_size or not np.isfinite(result).all():
            raise ValueError(f"backend emitted invalid logits with shape {result.shape}")
        return result.astype(np.float32, copy=False)

    def infer_file(self, audio_path: str | Path, sampling_rate: int = 16_000) -> np.ndarray:
        """Extract raw logits from one audio file without reading any PHN data."""

        return self.infer(load_audio(audio_path, sampling_rate), sampling_rate=sampling_rate)

    def cache_audio(
        self,
        *,
        audio_path: str | Path,
        canonical_phones: Iterable[str],
        output_path: str | Path,
        utterance_id: str,
        experiment_config: Mapping[str, Any] | str,
        experiment_config_path: str | Path | None = None,
        backend_hashes: Mapping[str, str] | None = None,
        metadata: Mapping[str, Any] | None = None,
        sampling_rate: int = 16_000,
        overwrite: bool = False,
    ) -> Path:
        """Extract and cache one utterance with fail-closed provenance checks."""

        phones = tuple(str(phone) for phone in canonical_phones)
        labels = self.canonical_ids(phones)
        binding = build_cache_binding(
            audio_path=audio_path,
            checkpoint=self.checkpoint,
            processor=self.processor,
            experiment_config=experiment_config,
            canonical_ids=labels.tolist(),
            config_path=experiment_config_path,
            backend_hashes=backend_hashes,
        )
        destination = Path(output_path)
        if destination.exists() and not overwrite:
            cached = load_logits_cache(
                destination, expected_fingerprint=str(binding["fingerprint"])
            )
            if cached.utterance_id != str(utterance_id):
                raise ValueError("cache utterance id does not match the requested record")
            return destination
        waveform = load_audio(audio_path, sampling_rate)
        logits = self.infer(waveform, sampling_rate=sampling_rate)
        cache_metadata = dict(metadata or {})
        cache_metadata.update(
            {
                "audio_path": str(Path(audio_path).resolve()),
                "sampling_rate": sampling_rate,
                "input_samples": int(waveform.size),
                "output_frames": int(logits.shape[0]),
                "blank_id": self.blank_id,
                "vocab_size": self.vocab_size,
                "scoring_policy": "complete_sequence_ctc_without_forced_alignment",
            }
        )
        return save_logits_cache(
            destination,
            utterance_id=utterance_id,
            logits=logits,
            canonical_ids=labels,
            duration_s=float(waveform.size / sampling_rate),
            binding=binding,
            metadata=cache_metadata,
        )
