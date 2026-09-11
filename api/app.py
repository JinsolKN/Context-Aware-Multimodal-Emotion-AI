"""Minimal CPU API. Run from this directory: python -m uvicorn app:app."""
from contextlib import asynccontextmanager
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
import threading
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import soundfile as sf
import torch

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from inference import DEFAULT_CHECKPOINT, ROOT, EmotionPredictor

logger = logging.getLogger(__name__)
MAX_AUDIO_BYTES = 10 * 1024 * 1024


def error(code, message):
    return {"code": code, "message": message}


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.predictor = None
    app.state.model_error = None
    app.state.predict_lock = threading.Lock()
    checkpoint = Path(os.environ.get("EMOTION_CHECKPOINT", str(DEFAULT_CHECKPOINT)))
    try:
        if not checkpoint.is_file():
            raise FileNotFoundError("Configured checkpoint is absent")
        torch.set_num_threads(4)
        app.state.predictor = EmotionPredictor(
            checkpoint_path=checkpoint,
            assets_path=os.environ.get("EMOTION_ASSETS", str(ROOT / "handoff_assets")),
            device="cpu",
        )
    except Exception:
        logger.exception("Emotion model initialization failed")
        app.state.model_error = (
            error("checkpoint_missing", "Checkpoint not found. Check EMOTION_CHECKPOINT and restart the server.")
            if not checkpoint.is_file() else
            error("model_initialization_failed", "Model initialization failed. Check server logs, checkpoint and assets, then restart.")
        )
    try:
        yield
    finally:
        app.state.predictor = None


app = FastAPI(title="MELD Emotion Inference", version="1.0.0", lifespan=lifespan)


class Prediction(BaseModel):
    emotion: str
    confidence: float = Field(ge=0, le=1)
    probabilities: dict[str, float]


@app.get("/")
def index():
    return {"service": "MELD Audio + Text + Context inference", "docs": "/docs", "health": "/health", "predict": "/predict"}


@app.get("/health", responses={503: {"description": "Model unavailable"}})
def health(request: Request):
    if request.app.state.predictor is None:
        return JSONResponse(status_code=503, content={
            "status": "error", "model_loaded": False, "detail": request.app.state.model_error,
        })
    return {"status": "ok", "model_loaded": True}


@app.post("/predict", response_model=Prediction, responses={
    413: {"description": "Audio exceeds 10 MiB"},
    422: {"description": "Missing or invalid text, context or audio"},
    503: {"description": "Model unavailable or busy"},
    500: {"description": "Inference or temporary storage failed"},
})
def predict(
    request: Request,
    text: Annotated[str, Form(description="Current utterance, nonempty. Passed unchanged to inference.")],
    audio: Annotated[UploadFile, File(description="Audio file, preferably WAV: 16 kHz, at least 400 samples, max 10 MiB. No resampling.")],
    context: Annotated[str, Form(description='JSON array of past turns, oldest first: [{"speaker":"Ross","text":"What happened?"}]. Speaker-prefixed strings also supported. Inference selects the last 3 turns.')] = "[]",
):
    # A sync endpoint keeps CPU work off the async event loop. One active prediction
    # per process bounds concurrent model memory use on a CPU laptop.
    acquired = False
    try:
        if not text.strip():
            raise HTTPException(422, error("empty_text", "text must contain a non-whitespace utterance."))
        try:
            history = json.loads(context)
        except (ValueError, RecursionError):
            raise HTTPException(422, error("invalid_context", "context must be a JSON array of past utterances.")) from None
        if not isinstance(history, list):
            raise HTTPException(422, error("invalid_context", "context must be a JSON array; use [] for no history."))
        predictor = request.app.state.predictor
        if predictor is None:
            raise HTTPException(503, request.app.state.model_error)
        acquired = request.app.state.predict_lock.acquire(blocking=False)
        if not acquired:
            raise HTTPException(503, error("model_busy", "Another prediction is running. Retry shortly."), headers={"Retry-After": "1"})
        # Validate through the existing module; do not duplicate context formatting.
        try:
            predictor.preprocessor.build_context(history)
        except (TypeError, ValueError):
            raise HTTPException(422, error("invalid_context", 'Each turn needs string speaker/text with a nonempty speaker, or a "Speaker: utterance" string.')) from None
        with tempfile.TemporaryDirectory(prefix="emotion-api-") as directory:
            # Never use a client-supplied filename as a local path. Close before
            # inference opens it, which is required on Windows.
            audio_path = Path(directory) / "upload.wav"
            size = 0
            with audio_path.open("wb") as target:
                while chunk := audio.file.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_AUDIO_BYTES:
                        raise HTTPException(413, error("audio_too_large", "audio must be at most 10 MiB."))
                    target.write(chunk)
            if not size:
                raise HTTPException(422, error("invalid_audio", "audio file is empty."))
            try:
                return predictor.predict(text=text, audio_path=audio_path, context=history)
            except sf.LibsndfileError:
                raise HTTPException(422, error("invalid_audio", "Cannot decode audio. Upload a valid 16 kHz WAV file.")) from None
            except ValueError as exc:
                if str(exc).startswith("Invalid waveform:"):
                    raise HTTPException(422, error("invalid_audio", "Audio must be 16 kHz, at least 400 samples, and contain only finite values. No resampling is performed.")) from None
                raise
    except HTTPException:
        raise
    except Exception:
        logger.exception("Emotion inference or temporary audio storage failed")
        raise HTTPException(500, error("inference_failed", "Inference failed. Check server logs and available memory.")) from None
    finally:
        if acquired:
            request.app.state.predict_lock.release()
        audio.file.close()
