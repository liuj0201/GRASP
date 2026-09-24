from __future__ import annotations

from pathlib import Path

import pytest
import da_cf_gop.stages as stages

from da_cf_gop.data import (
    CROSS_MIC_PHN_SPEAKERS,
    PRIMARY5,
    SENSITIVITY7,
    build_torgo_manifest,
    read_manifest,
    reading_event_id,
    write_manifest,
)


def _speaker_set(speaker: str) -> str:
    return "".join(character for character in speaker if character.isalpha())


def _add_event(
    root: Path,
    speaker: str,
    session: str,
    stem: str,
    *,
    prompt: str = "Bad cat",
    phn_microphone: str = "headMic",
    phn: str = "0 100 b\n100 200 ae\n200 300 t\n",
) -> None:
    session_dir = root / _speaker_set(speaker) / speaker / session
    wav_dir = session_dir / "wav_headMic"
    prompt_dir = session_dir / "prompts"
    phn_dir = session_dir / f"phn_{phn_microphone}"
    wav_dir.mkdir(parents=True, exist_ok=True)
    prompt_dir.mkdir(parents=True, exist_ok=True)
    phn_dir.mkdir(parents=True, exist_ok=True)
    (wav_dir / f"{stem}.wav").write_bytes(b"not-read-by-test-probe")
    (prompt_dir / f"{stem}.txt").write_text(prompt, encoding="utf-8")
    (phn_dir / f"{stem}.PHN").write_text(phn, encoding="utf-8")


def _fake_probe(_: Path) -> tuple[float, int]:
    return 1.25, 16_000


def _fake_g2p(texts):
    mapping = {
        "Bad cat": ["B", "AE", "D", "K", "AE", "T"],
        "Bat": ["B", "AE", "T"],
        "Cat": ["K", "AE", "T"],
    }
    return [mapping[text] for text in texts], [[] for _ in texts]


def test_frozen_cohort_constants() -> None:
    assert PRIMARY5 == ("F03", "F04", "M01", "M04", "M05")
    assert SENSITIVITY7 == ("F01", "F03", "F04", "M01", "M02", "M04", "M05")
    assert CROSS_MIC_PHN_SPEAKERS == {"F01", "M02"}


def test_f01_m02_cross_mic_fallback_and_headmic_phn_priority(tmp_path) -> None:
    _add_event(tmp_path, "F01", "Session1", "0001", prompt="Bat", phn_microphone="arrayMic")
    _add_event(tmp_path, "M02", "Session1", "0001", prompt="Bat", phn_microphone="arrayMic")
    _add_event(tmp_path, "M02", "Session1", "0002", prompt="Cat", phn_microphone="headMic")
    m02_array = tmp_path / "M" / "M02" / "Session1" / "phn_arrayMic"
    m02_array.mkdir(parents=True, exist_ok=True)
    (m02_array / "0002.PHN").write_text("0 100 aa\n", encoding="utf-8")
    _add_event(tmp_path, "F03", "Session1", "0001", prompt="Bat", phn_microphone="arrayMic")

    rows, exclusions, audit = build_torgo_manifest(
        tmp_path,
        dys_speakers=("F01", "M02", "F03"),
        healthy_speakers=(),
        g2p=_fake_g2p,
        audio_probe=_fake_probe,
    )

    assert len(rows) == 3
    by_event = {row.reading_event_id: row for row in rows}
    assert by_event["F01_Session1_0001"].phn_microphone == "arrayMic"
    assert by_event["M02_Session1_0001"].phn_microphone == "arrayMic"
    assert by_event["M02_Session1_0002"].phn_microphone == "headMic"
    for row in rows:
        assert row.audio_microphone == "headMic"
        assert "wav_headMic" in row.wav_path
        assert len(row.wav_sha256) == len(row.prompt_sha256) == len(row.phn_sha256) == 64
        assert len(row.g2p_descriptor_sha256) == 64
        assert not any("start_sample" in key or "end_sample" in key for key in row.to_dict())
    assert {item.reason for item in exclusions} == {"missing_phn_sequence"}
    assert audit["phn_microphone_counts"] == {"arrayMic": 2, "headMic": 1}
    assert set(audit["input_source_hashes"]) == {"wav", "prompt", "phn"}


def test_prompt_validation_event_identity_and_session_deduplication(tmp_path) -> None:
    _add_event(tmp_path, "F03", "Session1", "0001", prompt="Bat")
    _add_event(tmp_path, "F03", "Session2", "0001", prompt="Cat", phn="0 100 k\n100 200 ae\n200 300 t\n")
    _add_event(tmp_path, "F03", "Session2", "0002", prompt="[sustain ah]")

    rows, exclusions, _ = build_torgo_manifest(
        tmp_path,
        dys_speakers=("F03",),
        healthy_speakers=(),
        g2p=_fake_g2p,
        audio_probe=_fake_probe,
    )

    assert [row.reading_event_id for row in rows] == [
        "F03_Session1_0001", "F03_Session2_0001"
    ]
    assert len({row.reading_event_id for row in rows}) == 2
    assert rows[0].prompt_path.endswith("prompts\\0001.txt") or rows[0].prompt_path.endswith("prompts/0001.txt")
    assert any(item.reason == "elicitation_instruction" for item in exclusions)
    assert reading_event_id("F03", "Session1", "0001") != reading_event_id("F03", "Session2", "0001")


def test_manifest_layers_stable_summary_and_json_roundtrip(tmp_path) -> None:
    _add_event(
        tmp_path, "F03", "Session1", "0001", prompt="Bat",
        phn="0 10 h#\n10 20 b\n20 30 ae\n30 40 dx\n",
    )
    rows, exclusions, audit = build_torgo_manifest(
        tmp_path,
        dys_speakers=("F03",),
        healthy_speakers=(),
        g2p=_fake_g2p,
        audio_probe=_fake_probe,
    )
    assert not exclusions
    row = rows[0]
    assert row.phn_raw_labels == ("h#", "b", "ae", "dx")
    assert row.phn_timit_phones == ("B", "AE", "DX")
    assert row.phn_model_phones == ("B", "AE", "T")
    assert [token.event_type for token in row.stable_tokens] == ["match", "match", "match"]
    assert row.stable_fraction == 1.0
    assert audit["exact_substitution_deletion_headline_allowed"] is True
    assert audit["stable_fraction"] == 1.0
    assert audit["patient_stable_fraction"] == 1.0
    assert audit["headline_stability_population"] == "dysarthric canonical tokens only"
    assert audit["n_unique_prompts"] == 1

    path = tmp_path / "manifest.jsonl.gz"
    write_manifest(rows, path)
    restored = read_manifest(path)
    assert restored == rows
    assert restored[0].phones == ["B", "AE", "T"]
    assert restored[0].utterance_id == "F03_Session1_0001"

    duplicate_path = tmp_path / "duplicate.jsonl.gz"
    write_manifest([row, row], duplicate_path)
    with pytest.raises(ValueError, match="duplicate reading_event_id"):
        read_manifest(duplicate_path)


def test_global_headline_gate_uses_all_canonical_tokens(tmp_path) -> None:
    _add_event(tmp_path, "F03", "Session1", "0001", prompt="Bat", phn="0 10 aa\n")
    rows, _, audit = build_torgo_manifest(
        tmp_path,
        dys_speakers=("F03",),
        healthy_speakers=(),
        g2p=_fake_g2p,
        audio_probe=_fake_probe,
    )
    assert len(rows) == 1
    assert audit["stable_fraction"] < 0.8
    assert audit["exact_substitution_deletion_headline_allowed"] is False


def test_unknown_g2p_and_missing_prompt_are_explicit_exclusions(tmp_path) -> None:
    _add_event(tmp_path, "F03", "Session1", "0001", prompt="Bat")
    session = tmp_path / "F" / "F03" / "Session1"
    (session / "wav_headMic" / "0002.wav").write_bytes(b"x")
    (session / "phn_headMic" / "0002.PHN").write_text("0 10 b\n", encoding="utf-8")

    def unknown_g2p(texts):
        return [[] for _ in texts], [["?"] for _ in texts]

    rows, exclusions, audit = build_torgo_manifest(
        tmp_path,
        dys_speakers=("F03",),
        healthy_speakers=(),
        g2p=unknown_g2p,
        audio_probe=_fake_probe,
    )
    assert rows == []
    assert {item.reason for item in exclusions} == {"missing_prompt", "unmapped_g2p_phone"}
    assert audit["n_rows"] == 0
    assert audit["exact_substitution_deletion_headline_allowed"] is False


def test_manifest_stage_summary_binds_each_source_inventory(tmp_path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    _add_event(data_root, "F03", "Session1", "0001", prompt="Bat")
    rows, exclusions, audit = build_torgo_manifest(
        data_root,
        dys_speakers=("F03",),
        healthy_speakers=(),
        g2p=_fake_g2p,
        audio_probe=_fake_probe,
    )
    monkeypatch.setattr(
        stages,
        "build_manifest",
        lambda *_args, **_kwargs: (rows, exclusions, audit),
    )
    monkeypatch.setattr(stages, "nested_loso_folds", lambda *_args, **_kwargs: [])
    cfg = {
        "_config_hash": "a" * 64,
        "paths": {"artifacts": str(tmp_path / "artifacts"), "data_root": str(data_root)},
        "cohorts": {"primary5": ["F03"], "healthy_phn": []},
        "labels": {"stable_alignment_min_fraction": 0.8},
        "audio_policy": {"allow_cross_mic_phn_sequence_for": ["F01", "M02"]},
    }
    summary = stages.build_manifests_stage(
        cfg, cohorts=("primary5",), progress=lambda _message: None
    )["primary5"]
    expected = audit["input_source_hashes"]
    assert summary["source_files"] == {
        "frozen_config": "a" * 64,
        "wav_input_inventory": expected["wav"]["inventory_sha256"],
        "prompt_input_inventory": expected["prompt"]["inventory_sha256"],
        "phn_input_inventory": expected["phn"]["inventory_sha256"],
    }
    assert summary["models"] == {
        "g2p_descriptor": audit["g2p_provenance"]["descriptor_sha256"]
    }
