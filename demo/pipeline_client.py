"""HTTP-only adapters: no model imports or credentials in the demo process."""
import json
from urllib.parse import urlsplit

import httpx

if __package__:
    from .demo_client import DemoError, MAX_AUDIO_BYTES
else:
    from demo_client import DemoError, MAX_AUDIO_BYTES


class PipelineError(DemoError):
    def __init__(self, message, payload=None):
        super().__init__(message)
        self.payload = payload


ERROR_MESSAGES = {
    "llm_not_configured": "서버에 OPENAI_API_KEY와 OPENAI_MODEL 설정이 필요합니다.",
    "llm_authentication_failed": "LLM 인증에 실패했습니다. 서버의 키 설정을 확인하세요.",
    "llm_rate_limited": "LLM 사용량 또는 요청 한도를 초과했습니다.",
    "llm_timeout": "LLM 응답 시간이 초과되었습니다.",
    "retrieval_unavailable": "검색 모델을 준비하지 못했습니다. 서버의 RAG 의존성과 모델 경로를 확인하세요.",
    "checkpoint_missing": "감정 checkpoint가 없습니다. 서버의 EMOTION_CHECKPOINT를 확인하세요.",
    "model_initialization_failed": "감정 모델을 로드하지 못했습니다. 서버의 모델 자산과 로그를 확인하세요.",
    "invalid_audio": "16 kHz WAV 음성을 확인하세요. 최소 400 samples가 필요합니다.",
    "emotion_input_missing": "감정 활용에는 현재 발화와 WAV 음성을 함께 입력하세요.",
    "invalid_options": "검색어, 카테고리와 K 설정을 확인하세요. Retrieval K ≥ Top K여야 합니다.",
    "agent_busy": "다른 Agent 요청이 실행 중입니다. 잠시 후 다시 시도하세요.",
    "recommend_busy": "다른 추천 요청이 실행 중입니다. 잠시 후 다시 시도하세요.",
    "model_busy": "다른 감정 추론이 실행 중입니다. 잠시 후 다시 시도하세요.",
}


def request_pipeline(endpoint, data, api_url, audio_bytes=None):
    try:
        url = urlsplit(api_url.strip())
        _ = url.port
        if url.scheme not in ("http", "https") or not url.hostname or url.query or url.fragment or url.username or url.password:
            raise ValueError
    except ValueError:
        raise PipelineError("FastAPI 서버 주소 형식을 확인하세요.") from None
    if audio_bytes is not None and (not audio_bytes or len(audio_bytes) > MAX_AUDIO_BYTES):
        raise PipelineError("비어 있지 않은 10 MiB 이하 WAV 음성을 업로드하세요.")
    files = {"audio": ("upload.wav", audio_bytes, "audio/wav")} if audio_bytes is not None else None
    try:
        response = httpx.post(api_url.strip().rstrip("/") + endpoint, data=data, files=files,
                              timeout=httpx.Timeout(240, connect=5), trust_env=False)
    except httpx.TimeoutException:
        raise PipelineError("API 응답 시간이 초과되었습니다. 서버 처리 상태를 확인하세요.") from None
    except httpx.RequestError:
        raise PipelineError("FastAPI에 연결할 수 없습니다. 서버 실행 여부와 주소를 확인하세요.") from None
    try:
        body = response.json()
        if not isinstance(body, dict):
            raise ValueError
    except ValueError:
        raise PipelineError(f"API 응답이 JSON 객체가 아닙니다 (HTTP {response.status_code}).") from None
    if response.is_error or body.get("status") == "failed":
        detail = body.get("error") or body.get("detail") or {}
        code = detail.get("code") if isinstance(detail, dict) else None
        # Show only known safe messages; never render raw upstream errors/validation input.
        message = ERROR_MESSAGES.get(code, f"API 요청 실패 (HTTP {response.status_code}). 입력과 서버 로그를 확인하세요.")
        raise PipelineError(message, body if endpoint == "/agent" and isinstance(body.get("tool_calls"), list) else None)
    required = ("items", "explanation", "retrieved_ids") if endpoint == "/recommend" else ("answer", "tool_calls")
    if any(key not in body for key in required):
        raise PipelineError("API 응답에 필요한 결과 필드가 없습니다.")
    return body


def recommend_contents(query, preferences, retrieval_k, top_k, use_emotion, explanation_mode,
                       api_url, text="", audio_bytes=None, context=()):
    if not query.strip() or len(query.strip()) > 1000:
        raise PipelineError("검색어는 1~1000자로 입력하세요.")
    if not 1 <= top_k <= 10 or not top_k <= retrieval_k <= 50:
        raise PipelineError("1 ≤ Top K ≤ 10, Top K ≤ Retrieval K ≤ 50으로 설정하세요.")
    if use_emotion and (not text.strip() or not audio_bytes):
        raise PipelineError(ERROR_MESSAGES["emotion_input_missing"])
    data = {"options": json.dumps(dict(query=query, preferences=preferences, retrieval_k=retrieval_k,
                                        top_k=top_k, use_emotion=use_emotion, explanation_mode=explanation_mode))}
    if use_emotion:
        data.update(text=text, context=json.dumps(context))
    return request_pipeline("/recommend", data, api_url, audio_bytes if use_emotion else None)


def chat_agent(message, preferences, api_url, text="", audio_bytes=None, context=()):
    if not message.strip() or len(message.strip()) > 2000:
        raise PipelineError("메시지는 1~2000자로 입력하세요.")
    if bool(text.strip()) != (audio_bytes is not None):
        raise PipelineError(ERROR_MESSAGES["emotion_input_missing"])
    data = {"message": message, "preferences": json.dumps(preferences)}
    if audio_bytes is not None:
        data.update(text=text, context=json.dumps(context))
    return request_pipeline("/agent", data, api_url, audio_bytes)
