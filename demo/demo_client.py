"""Lightweight HTTP boundary shared by Streamlit and the live smoke test."""
import json
import math
import os
from urllib.parse import urlsplit

import httpx

if __package__:
    from .ranking import validate_probabilities
else:
    from ranking import validate_probabilities

API_URL = os.getenv("EMOTION_API_URL", "http://localhost:8000")
MAX_AUDIO_BYTES = 10 * 1024 * 1024


class DemoError(ValueError):
    """Safe, user-facing failure message."""


def parse_context(value):
    turns = [line.strip() for line in value.splitlines() if line.strip()]
    for turn in turns:
        speaker, separator, text = turn.partition(":")
        if not separator or not speaker.strip() or not text.strip():
            raise DemoError("Context는 한 줄에 Speaker: utterance 형식으로 입력하세요.")
    return turns  # The existing inference module selects the last three turns.


def predict_emotion(text, audio_bytes, context, api_url=API_URL):
    if not text.strip():
        raise DemoError("Current Utterance에 현재 발화를 입력하세요.")
    if not audio_bytes:
        raise DemoError("16 kHz WAV 오디오를 업로드하세요.")
    if len(audio_bytes) > MAX_AUDIO_BYTES:
        raise DemoError("오디오는 최대 10 MiB까지 업로드할 수 있습니다.")
    try:
        url = urlsplit(api_url.strip())
        _ = url.port
    except ValueError:
        raise DemoError("API 주소의 hostname과 port 형식을 확인하세요.") from None
    if url.scheme not in ("http", "https") or not url.netloc or url.query or url.fragment:
        raise DemoError("API 주소는 http:// 또는 https://로 시작하는 서버 주소여야 합니다.")
    try:
        response = httpx.post(
            api_url.strip().rstrip("/") + "/predict",
            data={"text": text, "context": json.dumps(context)},
            files={"audio": ("upload.wav", audio_bytes, "audio/wav")},
            timeout=httpx.Timeout(180, connect=5), trust_env=False,
        )
    except httpx.TimeoutException:
        raise DemoError("API 응답 시간이 초과되었습니다. CPU 처리 상태를 확인한 뒤 다시 시도하세요.") from None
    except httpx.RequestError:
        raise DemoError("FastAPI 서버에 연결할 수 없습니다. 서버 실행 여부와 API 주소를 확인하세요.") from None
    if response.is_error:
        messages = {
            "checkpoint_missing": "Checkpoint가 없습니다. EMOTION_CHECKPOINT를 확인하고 API를 재시작하세요.",
            "model_initialization_failed": "모델 로딩 실패: checkpoint/assets와 API 로그를 확인하세요.",
            "model_busy": "다른 추론이 진행 중입니다. 잠시 후 다시 시도하세요.",
            "invalid_audio": "유효한 16 kHz WAV가 필요합니다(최소 400 samples). 자동 resampling은 하지 않습니다.",
            "invalid_context": "Context의 Speaker: utterance 형식을 확인하세요.",
            "audio_too_large": "오디오가 10 MiB 제한을 초과했습니다.",
            "empty_text": "현재 발화를 입력하세요.",
            "inference_failed": "API 추론에 실패했습니다. 서버 로그와 사용 가능한 메모리를 확인하세요.",
        }
        try:
            detail = response.json().get("detail", {})
            code = detail.get("code") if isinstance(detail, dict) else None
        except (ValueError, AttributeError):
            code = None
        raise DemoError(messages.get(code, f"API 요청 실패 (HTTP {response.status_code}). 입력과 서버 로그를 확인하세요."))
    try:
        result = response.json()
        validate_probabilities(result["probabilities"])
        probabilities = result["probabilities"]
        confidence = result["confidence"]
        if (isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or
                not math.isfinite(confidence) or not 0 <= confidence <= 1 or
                result["emotion"] not in probabilities or
                probabilities[result["emotion"]] != max(probabilities.values()) or
                abs(result["confidence"] - max(probabilities.values())) > 1e-5):
            raise ValueError("Inconsistent prediction")
        return result
    except (ValueError, TypeError, KeyError):
        raise DemoError("API 응답의 감정 label 또는 확률 형식이 올바르지 않습니다.") from None
