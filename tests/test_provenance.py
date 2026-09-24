from __future__ import annotations

import math

import numpy as np
import pytest

from da_cf_gop.provenance import (
    ProvenanceError,
    build_cache_descriptor,
    canonical_json_bytes,
    load_npz,
    read_json,
    sha256_file,
    validate_cache_descriptor,
    write_deterministic_npz,
)


def test_canonical_json_is_order_independent_and_rejects_nan() -> None:
    assert canonical_json_bytes({"b": 2, "a": 1}) == b'{"a":1,"b":2}\n'
    with pytest.raises(ProvenanceError, match="non-finite"):
        canonical_json_bytes({"bad": math.nan})


def test_npz_is_deterministic_and_pickle_free(tmp_path) -> None:
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    arrays = {"logits": np.arange(12, dtype=np.float32).reshape(3, 4),
              "length": np.asarray([3], dtype=np.int64)}
    write_deterministic_npz(first, arrays)
    write_deterministic_npz(second, dict(reversed(list(arrays.items()))))
    assert sha256_file(first) == sha256_file(second)
    loaded = load_npz(first, expected_sha256=sha256_file(first))
    np.testing.assert_array_equal(loaded["logits"], arrays["logits"])
    with pytest.raises(ProvenanceError, match="object arrays"):
        write_deterministic_npz(tmp_path / "unsafe.npz", {"x": np.asarray([{}], dtype=object)})


def test_cache_descriptor_binds_config_source_fold_and_artifact(tmp_path) -> None:
    source = tmp_path / "audio.wav"
    source.write_bytes(b"audio")
    artifact = tmp_path / "logits.npz"
    write_deterministic_npz(artifact, {"logits": np.zeros((2, 3), dtype=np.float32)})
    config = {"checkpoint": "frozen", "blank_id": 0}
    descriptor = build_cache_descriptor(
        artifact_type="ctc_logits",
        artifact=artifact,
        config=config,
        sources={"audio": source},
        fold="outer_F03",
        parameters={"sample_rate": 16000},
        root=tmp_path,
    )
    assert validate_cache_descriptor(
        descriptor,
        root=tmp_path,
        expected_config=config,
        expected_artifact_type="ctc_logits",
        expected_fold="outer_F03",
        required_source_roles=("audio",),
    ) == artifact.resolve()

    source.write_bytes(b"changed")
    with pytest.raises(ProvenanceError, match="size changed|hash changed"):
        validate_cache_descriptor(descriptor, root=tmp_path)


def test_descriptor_fingerprint_detects_metadata_edit(tmp_path) -> None:
    source = tmp_path / "source"
    source.write_text("x", encoding="utf-8")
    artifact = tmp_path / "artifact.npz"
    write_deterministic_npz(artifact, {"x": np.asarray([1])})
    descriptor = build_cache_descriptor(
        artifact_type="test", artifact=artifact, config={}, sources={"audio": source},
        root=tmp_path,
    )
    descriptor["fold"] = "tampered"
    with pytest.raises(ProvenanceError, match="fingerprint"):
        validate_cache_descriptor(descriptor, root=tmp_path)


def test_strict_json_reader_rejects_duplicate_keys(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"a": 1, "a": 2}', encoding="utf-8")
    with pytest.raises(ProvenanceError, match="duplicate JSON key"):
        read_json(path)

