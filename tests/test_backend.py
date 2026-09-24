from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pytest

from da_cf_gop.backend import (
    build_cache_binding,
    load_logits_cache,
    load_vocab,
    migrate_legacy_logits_cache,
    save_logits_cache,
    sha256_file,
    sha256_json,
    sha256_path,
)
from da_cf_gop.config import config_hash
from da_cf_gop.data import build_torgo_manifest, write_manifest
import da_cf_gop.stages as stages
from da_cf_gop.provenance import (
    build_cache_descriptor,
    companion_descriptor_path,
    write_deterministic_npz,
    write_json,
)


def test_hashes_are_content_addressed(tmp_path) -> None:
    first = tmp_path / "model" / "a.bin"
    first.parent.mkdir()
    first.write_bytes(b"weights")
    before = sha256_path(first.parent)
    assert sha256_file(first) == sha256_file(first)
    assert sha256_json({"b": 2, "a": 1}) == sha256_json({"a": 1, "b": 2})
    first.write_bytes(b"changed")
    assert sha256_path(first.parent) != before


def test_dynamic_vocab_and_blank(tmp_path) -> None:
    processor = tmp_path / "processor"
    processor.mkdir()
    (processor / "vocab.json").write_text(
        json.dumps({"AA": 0, "<blank>": 1, "B": 2}), encoding="utf-8"
    )
    forward, reverse, blank = load_vocab(processor, blank_token="<blank>")
    assert forward["B"] == 2
    assert reverse[0] == "AA"
    assert blank == 1
    (processor / "vocab.json").write_text(
        json.dumps({"<blank>": 0, "AA": 2}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="contiguous"):
        load_vocab(processor, blank_token="<blank>")


def test_pickle_free_cache_checks_full_binding(tmp_path) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"audio bytes")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "model.bin").write_bytes(b"model")
    processor = tmp_path / "processor"
    processor.mkdir()
    (processor / "vocab.json").write_text(
        json.dumps({"<pad>": 0, "AA": 1, "B": 2}), encoding="utf-8"
    )
    config = tmp_path / "frozen.json"
    config.write_text(json.dumps({"seed": 7}), encoding="utf-8")
    binding = build_cache_binding(
        audio_path=audio,
        checkpoint=checkpoint,
        processor=processor,
        experiment_config={"seed": 7},
        canonical_ids=[1, 2],
        config_path=config,
    )
    assert binding["config_sha256"] == config_hash({"seed": 7})
    destination = save_logits_cache(
        tmp_path / "cache.npz",
        utterance_id="u1",
        logits=np.zeros((4, 3)),
        canonical_ids=[1, 2],
        duration_s=0.25,
        binding=binding,
        metadata={"split": "train"},
    )
    loaded = load_logits_cache(destination, expected_fingerprint=binding["fingerprint"])
    assert loaded.utterance_id == "u1"
    assert loaded.logits.shape == (4, 3)
    assert loaded.metadata == {"split": "train"}
    # A scoring consumer supplies no expected fingerprint; the embedded paths,
    # exact canonical sequence, and companion descriptor must still validate.
    assert load_logits_cache(destination).canonical_ids.tolist() == [1, 2]
    assert not loaded.logits.dtype.hasobject
    first_hash = sha256_file(destination)
    save_logits_cache(
        destination,
        utterance_id="u1",
        logits=np.zeros((4, 3)),
        canonical_ids=[1, 2],
        duration_s=0.25,
        binding=binding,
        metadata={"split": "train"},
    )
    assert sha256_file(destination) == first_hash

    audio.write_bytes(b"different audio")
    changed = build_cache_binding(
        audio_path=audio,
        checkpoint=checkpoint,
        processor=processor,
        experiment_config={"seed": 7},
        canonical_ids=[1, 2],
        config_path=config,
    )
    assert changed["fingerprint"] != binding["fingerprint"]
    with pytest.raises(ValueError, match="does not match|audio hash changed|hash changed"):
        load_logits_cache(destination, expected_fingerprint=changed["fingerprint"])


def test_same_length_canonical_and_live_config_invalidate_raw_logits(tmp_path) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"audio")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "weights.bin").write_bytes(b"weights")
    processor = tmp_path / "processor"
    processor.mkdir()
    (processor / "vocab.json").write_text(
        json.dumps({"<pad>": 0, "AA": 1, "B": 2}), encoding="utf-8"
    )
    config = tmp_path / "frozen.json"
    config.write_text(json.dumps({"seed": 7}), encoding="utf-8")
    first = build_cache_binding(
        audio_path=audio,
        checkpoint=checkpoint,
        processor=processor,
        experiment_config={"seed": 7},
        canonical_ids=[1, 2],
        config_path=config,
    )
    path = save_logits_cache(
        tmp_path / "cache.npz",
        utterance_id="u1",
        logits=np.zeros((4, 3)),
        canonical_ids=[1, 2],
        duration_s=0.25,
        binding=first,
    )
    reordered = build_cache_binding(
        audio_path=audio,
        checkpoint=checkpoint,
        processor=processor,
        experiment_config={"seed": 7},
        canonical_ids=[2, 1],
        config_path=config,
    )
    assert reordered["fingerprint"] != first["fingerprint"]
    with pytest.raises(ValueError, match="does not match"):
        load_logits_cache(path, expected_fingerprint=reordered["fingerprint"])

    config.write_text(json.dumps({"seed": 8}), encoding="utf-8")
    with pytest.raises(ValueError, match="config"):
        load_logits_cache(path)


def test_extract_stage_rebuilds_same_length_canonical_change(tmp_path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    session = data_root / "F" / "F03" / "Session1"
    for directory in ("wav_headMic", "prompts", "phn_headMic"):
        (session / directory).mkdir(parents=True, exist_ok=True)
    (session / "wav_headMic" / "0001.wav").write_bytes(b"audio")
    (session / "prompts" / "0001.txt").write_text("Bat", encoding="utf-8")
    (session / "phn_headMic" / "0001.PHN").write_text(
        "0 1 b\n1 2 ae\n2 3 t\n", encoding="utf-8"
    )
    rows, _exclusions, _audit = build_torgo_manifest(
        data_root,
        dys_speakers=("F03",),
        healthy_speakers=(),
        g2p=lambda texts: ([['B', 'AE', 'T'] for _ in texts], [[] for _ in texts]),
        audio_probe=lambda _path: (1.0, 16_000),
    )

    artifacts = tmp_path / "artifacts"
    manifest = artifacts / "manifests" / "sensitivity7.jsonl.gz"
    write_manifest(rows, manifest)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "weights.bin").write_bytes(b"weights")
    processor = tmp_path / "processor"
    processor.mkdir()
    vocabulary = {"<pad>": 0, "B": 1, "AE": 2, "T": 3, "K": 4}
    (processor / "vocab.json").write_text(json.dumps(vocabulary), encoding="utf-8")
    raw_config = {
        "paths": {
            "artifacts": str(artifacts),
            "checkpoint": str(checkpoint),
            "processor": str(processor),
        },
        "backend": {
            "blank_id": 0,
            "vocab_size": len(vocabulary),
            "sample_rate": 16_000,
            "device": "cpu",
        },
    }
    config_path = tmp_path / "frozen.json"
    config_path.write_text(json.dumps(raw_config), encoding="utf-8")
    cfg = {
        **raw_config,
        "_config_path": str(config_path),
        "_config_hash": config_hash(raw_config),
    }
    inference_calls: list[int] = []

    class FakeBackend:
        blank_id = 0
        vocab_size = len(vocabulary)

        def __init__(self, *_args, **_kwargs):
            pass

        def infer(self, _waveform, _sampling_rate):
            inference_calls.append(1)
            return np.zeros((4, len(vocabulary)), dtype=np.float32)

        def canonical_ids(self, phones):
            return np.asarray([vocabulary[phone] for phone in phones], dtype=np.int64)

    monkeypatch.setattr(stages, "OfficialCTCSFBackend", FakeBackend)
    monkeypatch.setattr(stages, "load_audio", lambda *_args: np.zeros(16, dtype=np.float32))
    first_summary = stages.extract_logits_stage(
        cfg, cohort="sensitivity7", progress=lambda _message: None
    )
    assert first_summary["details"]["created"] == 1

    # Same token count, different token identity: this must miss the old cache.
    changed = replace(rows[0], canonical_phones=("K", "AE", "T"))
    write_manifest([changed], manifest)
    second_summary = stages.extract_logits_stage(
        cfg, cohort="sensitivity7", progress=lambda _message: None
    )
    assert second_summary["details"]["created"] == 1
    assert second_summary["details"]["reused"] == 0
    assert len(inference_calls) == 2
    cached_path = stages.logit_cache_path(cfg, changed.reading_event_id)
    assert load_logits_cache(cached_path).canonical_ids.tolist() == [4, 2, 3]


def test_matching_legacy_binding_is_upgraded_without_model_inference(tmp_path) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"audio")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "weights.bin").write_bytes(b"weights")
    processor = tmp_path / "processor"
    processor.mkdir()
    (processor / "vocab.json").write_text(
        json.dumps({"<pad>": 0, "AA": 1, "B": 2}), encoding="utf-8"
    )
    config = tmp_path / "frozen.json"
    config.write_text(json.dumps({"seed": 7}), encoding="utf-8")
    new_binding = build_cache_binding(
        audio_path=audio,
        checkpoint=checkpoint,
        processor=processor,
        experiment_config={"seed": 7},
        canonical_ids=[1, 2],
        config_path=config,
    )
    legacy = {
        key: new_binding[key]
        for key in (
            "audio_sha256",
            "checkpoint_sha256",
            "processor_sha256",
            "vocab_sha256",
            "config_sha256",
        )
    }
    legacy["fingerprint"] = sha256_json(legacy)
    path = tmp_path / "legacy.npz"
    write_deterministic_npz(
        path,
        {
            "utterance_id": np.asarray("u1"),
            "logits": np.zeros((4, 3), dtype=np.float32),
            "canonical_ids": np.asarray([1, 2], dtype=np.int64),
            "duration_s": np.asarray(0.25, dtype=np.float64),
            "binding_json": np.asarray(json.dumps(legacy, sort_keys=True, separators=(",", ":"))),
            "metadata_json": np.asarray("{}"),
        },
    )
    write_json(
        companion_descriptor_path(path),
        build_cache_descriptor(
            artifact_type="raw_ctc_logits",
            artifact=path,
            config={"legacy": True},
            sources={"audio": audio},
            upstream_sha256={
                key: legacy[key]
                for key in ("checkpoint_sha256", "processor_sha256", "vocab_sha256")
            },
        ),
    )
    assert migrate_legacy_logits_cache(
        path,
        expected_utterance_id="u1",
        expected_canonical_ids=[1, 2],
        new_binding=new_binding,
    )
    assert load_logits_cache(path).binding["schema_version"] == "da-cf-gop.logit-binding.v2"
