"""HTTP contract, lazy startup and worker serialization of the migrated app."""
import asyncio
import threading
import time

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from da_cf_gop import homework_web


class SmallEngine:
    def __init__(self):
        self.calls = []

    def status(self):
        return {"engine_loaded": True, "policy_id": "test-policy"}

    def assess(self, audio, reference_text):
        self.calls.append((audio.copy(), reference_text))
        return {
            "reference_text": reference_text,
            "words": [{"word_index": 0, "text": reference_text, "start_phone": 0,
                       "end_phone": 1, "phones": [0], "status": "review"}],
            "phones": [{"phone_index": 0, "word_index": 0, "canonical": "T", "ipa": "t",
                        "acceptable": ["T"], "assessable": True, "status": "review",
                        "error_probability": .7, "error_flag": True,
                        "diagnostic_candidate": None, "diagnostic_status": "withheld"}],
            "summary": {"assessed_phones": 1, "review_phones": 1, "unassessed_phones": 0},
            "provenance": {"model_version": "test", "policy_id": "test-policy", "replayed": False},
            "timings": {"inference_seconds": .001},
        }


@pytest.fixture
def decoder(monkeypatch):
    calls = []

    def decode(body, filename):
        calls.append((body, filename))
        if body == b"bad":
            raise ValueError("Use a readable WAV, FLAC or OGG recording.")
        return np.array([.1, -.1], dtype=np.float32)

    monkeypatch.setattr(homework_web, "decode_audio_bytes", decode)
    return calls


def test_page_and_status_do_not_construct_engine():
    app = homework_web.create_app()
    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        assert client.get("/static/homework.js").status_code == 200
        assert client.get("/api/status").json()["engine_loaded"] is False
        assert app.state.engine is None
        assert client.get("/demo").status_code == 404


def test_assessment_passes_assigned_text_and_new_audio_to_engine(decoder):
    engine = SmallEngine()
    with TestClient(homework_web.create_app(engine)) as client:
        response = client.post("/api/assess", files={"audio": ("take.wav", b"new audio", "audio/wav")},
                               data={"reference_text": "  test  ", "demo_id": "old-fixture"})
    assert response.status_code == 200
    result = response.json()
    assert decoder == [(b"new audio", "take.wav")]
    assert engine.calls[0][1] == "test"
    np.testing.assert_array_equal(engine.calls[0][0], np.array([.1, -.1], dtype=np.float32))
    assert result["provenance"]["replayed"] is False
    assert result["phones"][0]["diagnostic_candidate"] is None
    assert result["feedback"]["mode_used"] == "template"
    assert result["feedback"]["text"]
    assert result["timings"]["web_total_seconds"] >= result["timings"]["feedback_seconds"]


@pytest.mark.parametrize("text,body,mode,status", [
    (" ", b"audio", "template", 400),
    ("test", b"", "template", 400),
    ("test", b"bad", "template", 400),
    ("test", b"audio", "unknown", 422),
    ("a" * 501, b"audio", "template", 422),
])
def test_invalid_requests_do_not_call_scorer(decoder, text, body, mode, status):
    engine = SmallEngine()
    with TestClient(homework_web.create_app(engine)) as client:
        response = client.post("/api/assess", files={"audio": ("take.wav", body)},
                               data={"reference_text": text, "feedback_mode": mode})
    assert response.status_code == status
    assert not engine.calls


def test_upload_limit_and_missing_text(decoder, monkeypatch):
    monkeypatch.setattr(homework_web, "MAX_UPLOAD_BYTES", 8)
    engine = SmallEngine()
    with TestClient(homework_web.create_app(engine)) as client:
        assert client.post("/api/assess", files={"audio": ("take.wav", b"123456789")},
                           data={"reference_text": "test"}).status_code == 413
        assert client.post("/api/assess", files={"audio": ("take.wav", b"audio")}).status_code == 422
    assert not engine.calls
    assert not decoder


def test_optional_feedback_mode_is_passed_without_changing_scores(decoder, monkeypatch):
    seen = []

    def feedback(result, mode):
        seen.append((result["phones"][0]["error_probability"], mode))
        return {"mode_requested": mode, "mode_used": "template", "text": "Review test.",
                "fallback_reason": "model_not_available"}

    monkeypatch.setattr(homework_web, "make_feedback", feedback)
    with TestClient(homework_web.create_app(SmallEngine())) as client:
        response = client.post("/api/assess", files={"audio": ("take.wav", b"audio")},
                               data={"reference_text": "test", "feedback_mode": "qwen"})
    assert response.status_code == 200
    assert seen == [(.7, "qwen")]
    assert response.json()["phones"][0]["error_probability"] == .7


def test_scoring_is_serialized_while_status_remains_responsive(decoder):
    entered = threading.Event()
    release = threading.Event()

    class WaitingEngine(SmallEngine):
        active = 0
        max_active = 0

        def assess(self, audio, text):
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            entered.set()
            assert release.wait(3)
            time.sleep(.01)
            result = super().assess(audio, text)
            self.active -= 1
            return result

    engine = WaitingEngine()
    app = homework_web.create_app(engine)

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            args = {"files": {"audio": ("take.wav", b"audio")}, "data": {"reference_text": "test"}}
            first = asyncio.create_task(client.post("/api/assess", **args))
            assert await asyncio.to_thread(entered.wait, 2)
            second = asyncio.create_task(client.post("/api/assess", **args))
            try:
                status = await asyncio.wait_for(client.get("/api/status"), timeout=1)
                assert status.status_code == 200
                assert not first.done()
            finally:
                release.set()
            responses = await asyncio.gather(first, second)
            assert all(response.status_code == 200 for response in responses)

    asyncio.run(exercise())
    assert engine.max_active == 1
    assert len(engine.calls) == 2
