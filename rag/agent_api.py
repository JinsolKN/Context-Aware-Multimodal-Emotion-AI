"""Additive /agent multipart route; keep original emotion/recommendation handlers unchanged."""
import json
import os
import threading
from typing import Annotated

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from rag.agent import AgentFailure, ToolSession, run_agent
from rag.api import Category, get_service


class AgentInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)
    message: str = Field(min_length=1, max_length=2000)
    preferences: list[Category] = Field(default_factory=list, max_length=5)


def create_agent_router(app, predict_fn):
    """Input: existing FastAPI app/predict handler. Output: independent router with lazy tools and its own lock."""
    app.state.agent_lock = threading.Lock()
    router = APIRouter()

    @router.post("/agent", responses={
        422: {"description": "Invalid message/preferences/input pair"},
        502: {"description": "LLM authentication, protocol or provider failure"},
        503: {"description": "Missing LLM configuration, busy, quota or budget limit"},
        504: {"description": "LLM timeout"},
    })
    def agent(
        request: Request,
        message: Annotated[str, Form()],
        preferences: Annotated[str, Form(description='JSON category array, e.g. ["music"]')] = "[]",
        text: Annotated[str | None, Form()] = None,
        audio: Annotated[UploadFile | None, File()] = None,
        context: Annotated[str, Form()] = "[]",
    ):
        """Input: message + preferences; optional existing /predict fields. Output: answer/trace or explicit failure.

        Only analyze_emotion reads audio/text/context, using the unchanged /predict adapter.
        Search-only requests do not need checkpoint/audio; input files are closed even when no tool uses them.
        """
        acquired = False
        session = None
        model = os.getenv("OPENAI_MODEL", "").strip() or None
        try:
            try:
                parsed = AgentInput(message=message, preferences=json.loads(preferences))
                if (audio is None) != (text is None) or (text is not None and not text.strip()):
                    raise ValueError("Supply audio and nonempty text together")
            except (ValueError, ValidationError, RecursionError):
                raise AgentFailure("invalid_agent_input", "Check message, preferences JSON, and paired text/audio input.", 422) from None
            acquired = request.app.state.agent_lock.acquire(blocking=False)
            if not acquired:
                raise AgentFailure("agent_busy", "Another agent request is running. Retry shortly.", 503)
            def infer():
                """Input: captured upload fields. Output: original prediction; preserve existing validation and locks."""
                return predict_fn(request=request, text=text, audio=audio, context=context)
            session = ToolSession(lambda: get_service(request), parsed.preferences,
                                  predict_fn=infer if audio is not None else None)
            return run_agent(parsed.message, session)
        except AgentFailure as exc:
            return JSONResponse(status_code=exc.http_status, content=exc.payload(model),
                                headers={"Retry-After": "1"} if exc.code == "agent_busy" else None)
        except Exception:
            failure = AgentFailure("agent_internal_error", "Unexpected agent failure. Check local dependencies.", 500)
            failure.trace = session.trace if session is not None else []
            return JSONResponse(status_code=500, content=failure.payload(model))
        finally:
            if acquired:
                request.app.state.agent_lock.release()
            if audio is not None:
                audio.file.close()

    return router
