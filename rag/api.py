"""Multipart /recommend adapter; existing /predict owns all emotion/audio validation."""
import json
import logging
import threading
from typing import Annotated, Literal

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

logger = logging.getLogger(__name__)
Category = Literal["music", "comedy", "motivation", "relaxation", "drama"]


class Options(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    query: str = Field(min_length=1, max_length=1000)
    preferences: list[Category] = Field(default_factory=list, max_length=5)
    retrieval_k: int = Field(default=10, ge=1, le=50, strict=True)
    top_k: int = Field(default=3, ge=1, le=10, strict=True)
    use_emotion: bool = Field(default=True, strict=True)
    explanation_mode: Literal["auto", "template", "llm"] = "auto"

    @model_validator(mode="after")
    def check_k(self):
        """Input: parsed options. Output: validated options; avoid silently truncating a requested ranking."""
        if self.top_k > self.retrieval_k:
            raise ValueError("top_k cannot exceed retrieval_k")
        return self


def get_service(request):
    """Input: request. Output: shared lazy service; avoid affecting /predict startup or duplicate model loads."""
    state = request.app.state
    with state.recommend_init_lock:
        if state.recommend_service is None:
            from rag.service import RecommendationService
            state.recommend_service = RecommendationService()
        return state.recommend_service


def create_router(app, predict_fn):
    """Input: FastAPI app and existing predict function. Output: router; reuse inference without HTTP roundtrip."""
    app.state.recommend_service = None
    app.state.recommend_init_lock = threading.Lock()
    app.state.recommend_lock = threading.Lock()
    router = APIRouter()

    @router.post("/recommend", responses={
        422: {"description": "Invalid options or model input"},
        413: {"description": "Audio exceeds 10 MiB"},
        503: {"description": "Emotion model/retrieval unavailable or busy"},
        500: {"description": "Recommendation failed"},
    })
    def recommend(
        request: Request,
        options: Annotated[str, Form(description='JSON: {"query":"calm music", "preferences":["music"]}')],
        text: Annotated[str | None, Form()] = None,
        audio: Annotated[UploadFile | None, File()] = None,
        context: Annotated[str, Form()] = "[]",
    ):
        """Input: options JSON, plus /predict fields when emotion is enabled.

        Output: prediction (null when disabled), ranked items and cited explanations.

        A nonblocking lock bounds retrieval/LLM work per process. Existing inference
        retains its own lock, audio limits, context rules and temporary-file cleanup.
        """
        acquired = False
        try:
            try:
                config = Options.model_validate(json.loads(options))
            except (ValueError, ValidationError, RecursionError):
                raise HTTPException(422, {"code": "invalid_options", "message": "Check query, categories, K and explanation_mode in options JSON."}) from None
            acquired = request.app.state.recommend_lock.acquire(blocking=False)
            if not acquired:
                raise HTTPException(503, {"code": "recommend_busy", "message": "Retry shortly."}, headers={"Retry-After": "1"})
            prediction = None
            if config.use_emotion:
                if text is None or audio is None:
                    raise HTTPException(422, {"code": "emotion_input_missing", "message": "Supply text and audio for emotion analysis."})
                prediction = predict_fn(request=request, text=text, audio=audio, context=context)
            # Same internal validator placeholder used by ToolSession for emotion-free ranking.
            # It has zero weight and is never presented as a classifier prediction.
            from demo.ranking import LABELS
            probabilities = prediction["probabilities"] if prediction else {label: 1 / 7 for label in LABELS}
            try:
                service = get_service(request)
            except Exception:
                logger.exception("Retrieval initialization failed")
                raise HTTPException(503, {"code": "retrieval_unavailable", "message": "Check recommendation dependencies and embedding model access."}) from None
            result = service.recommend(probabilities=probabilities, **config.model_dump())
            return {"prediction": prediction, **result}
        except HTTPException:
            raise
        except Exception:
            logger.exception("Recommendation failed")
            raise HTTPException(500, {"code": "recommendation_failed", "message": "Check server logs."}) from None
        finally:
            if acquired:
                request.app.state.recommend_lock.release()
            if audio is not None:
                audio.file.close()

    return router
