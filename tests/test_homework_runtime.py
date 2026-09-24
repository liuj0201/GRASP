"""Audio boundary and reference-anchor checks for the live inference adapter."""
import io

import numpy as np
import pytest
import soundfile as sf

from da_cf_gop.backend import load_audio
from da_cf_gop.dual_lexicon import build_acceptable_graph
from da_cf_gop.homework_runtime import HomeworkEngine, decode_audio_bytes, present_predictions


@pytest.mark.parametrize("sample_rate,channels", [(16000, 1), (44100, 2), (48000, 2)])
def test_uploaded_audio_matches_offline_loading(tmp_path, sample_rate, channels):
    t = np.arange(sample_rate // 2) / sample_rate
    wave = .1 * np.sin(2 * np.pi * 440 * t)
    audio = wave if channels == 1 else np.column_stack((wave, wave * .5))
    path = tmp_path / "recording.wav"
    sf.write(path, audio, sample_rate, subtype="PCM_16")
    uploaded = decode_audio_bytes(path.read_bytes(), path.name)
    np.testing.assert_array_equal(uploaded, load_audio(path))
    assert uploaded.dtype == np.float32
    assert len(uploaded) == 8000


def test_invalid_and_overlong_uploads_are_rejected():
    with pytest.raises(ValueError, match="decode"):
        decode_audio_bytes(b"not audio")
    buffer = io.BytesIO()
    sf.write(buffer, np.zeros(21 * 16000), 16000, format="WAV")
    with pytest.raises(ValueError, match="20 seconds"):
        decode_audio_bytes(buffer.getvalue())


def test_repeated_words_and_unsupported_variants_keep_reference_indices():
    text = "cat, read cat"
    graph = build_acceptable_graph(text, dictionary={
        "cat": [("K", "AE1", "T")], "read": [("R", "IY1", "D"), ("R", "EH1", "D")]})
    predictions = [{"p_error": .9, "predicted_error": True} for _ in graph["canonical_phones"]]
    result = present_predictions(text, graph, predictions)
    assert [w["phones"] for w in result["words"]] == [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    assert [p["word_index"] for p in result["phones"]] == [0]*3 + [1]*3 + [2]*3
    assert result["summary"] == {"assessed_phones": 6, "review_phones": 6, "unassessed_phones": 3}
    for phone in result["phones"][3:6]:
        assert phone["status"] == "unassessed"
        assert phone["error_probability"] is None
        assert phone["error_flag"] is None
    assert all(p["diagnostic_candidate"] is None for p in result["phones"])


@pytest.mark.parametrize("audio,text,match", [
    (np.zeros(16000), "cat", "silent"),
    (np.ones(100), "cat", "0.1"),
    (np.ones(21*16000), "cat", "20"),
    (np.full(16000, np.nan), "cat", "finite"),
    (np.ones(16000), "   ", "assigned"),
    (np.ones(16000), "café", "English"),
    (np.ones(16000), "cat\x1b[D", "control"),
])
def test_invalid_inputs_never_load_models(audio, text, match, monkeypatch):
    engine = HomeworkEngine()
    def fail_load():
        pytest.fail("model loaded for invalid input")
    monkeypatch.setattr(engine, "_load", fail_load)
    with pytest.raises(ValueError, match=match):
        engine.assess(audio, text)


def test_policy_is_explicit_and_status_does_not_load():
    engine = HomeworkEngine()
    assert engine.status()["policy_id"] == "sensitivity7_test_F01"
    assert not engine.status()["engine_loaded"]
    with pytest.raises(ValueError, match="policy"):
        HomeworkEngine(policy_speaker="../somewhere")
