"""Local homework UI, migrated from code1/webapp onto the paper's scorer.

Run ``python -m da_cf_gop.homework_web`` and open http://127.0.0.1:8765.
Only uploaded audio and the assigned text enter the inference path.
"""
from __future__ import annotations

import asyncio
import os
import threading
import time
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse


STATIC_DIR = Path(__file__).with_name("static")
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


def decode_audio_bytes(body: bytes, filename: str):
    from .homework_runtime import decode_audio_bytes as decode

    return decode(body, filename)


def make_feedback(result: dict, mode: str) -> dict:
    from .homework_feedback import make_feedback as generate

    return generate(result, mode=mode)


def create_app(engine=None) -> FastAPI:
    """Create a lazy local app; an injected engine supports interface checks."""
    application = FastAPI(title="DA-CF-GOP Homework Practice")
    application.state.engine = engine
    application.state.inference_lock = threading.Lock()

    @application.get("/", include_in_schema=False)
    async def index():
        return FileResponse(STATIC_DIR / "homework.html")

    @application.get("/static/homework.js", include_in_schema=False)
    async def script():
        return FileResponse(STATIC_DIR / "homework.js", media_type="text/javascript")

    @application.get("/api/status")
    async def status():
        if application.state.engine is None:
            return {"engine_loaded": False, "state": "ready", "model_loading": "first assessment"}
        return await asyncio.to_thread(application.state.engine.status)

    def assess_sync(body: bytes, filename: str, reference: str, mode: str) -> dict:
        started = time.perf_counter()
        audio = decode_audio_bytes(body, filename)
        # The lock lives in the worker: cancellation of an HTTP task cannot let
        # another request load or run models while this worker is still active.
        with application.state.inference_lock:
            if application.state.engine is None:
                from .homework_runtime import HomeworkEngine

                application.state.engine = HomeworkEngine()
            result = dict(application.state.engine.assess(audio, reference))
            feedback_started = time.perf_counter()
            result["feedback"] = make_feedback(result, mode)
        result["timings"] = dict(result.get("timings", {}))
        result["timings"]["feedback_seconds"] = time.perf_counter() - feedback_started
        result["timings"]["web_total_seconds"] = time.perf_counter() - started
        return result

    @application.post("/api/assess")
    async def assess(
        audio: UploadFile = File(...),
        reference_text: str = Form(..., max_length=500),
        feedback_mode: Literal["template", "qwen"] = Form("template"),
    ):
        reference = reference_text.strip()
        if not reference:
            raise HTTPException(400, "Enter the assigned practice text before assessment.")
        try:
            body = await audio.read(MAX_UPLOAD_BYTES + 1)
        finally:
            await audio.close()
        if not body:
            raise HTTPException(400, "The audio file is empty.")
        if len(body) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "Choose an audio file smaller than 20 MB.")
        try:
            return await asyncio.to_thread(assess_sync, body, audio.filename or "recording.wav", reference, feedback_mode)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except (FileNotFoundError, RuntimeError) as exc:
            raise HTTPException(503, str(exc)) from exc

    return application


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run(app, host=os.environ.get("DA_CF_GOP_HOST", "127.0.0.1"),
                port=int(os.environ.get("DA_CF_GOP_PORT", "8765")))


if __name__ == "__main__":
    main()
