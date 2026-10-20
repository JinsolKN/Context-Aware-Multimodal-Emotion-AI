"""OFFLINE: mocked Responses function calls; real dispatcher and existing rank_candidates. No network."""
from contextlib import nullcontext
import json
from types import SimpleNamespace
from unittest.mock import Mock

from fastapi.testclient import TestClient
import httpx2
from openai import APITimeoutError, AuthenticationError
import pytest

from api.app import app
from demo.ranking import LABELS, load_catalog
from rag.agent import AgentFailure, MAX_TOOL_CALLS, TOOLS, ToolSession, run_agent

PROBABILITIES = {label: float(label == "sadness") for label in LABELS}


def call(name, arguments, call_id="call_1"):
    """Input: synthetic tool intent. Output: SDK-shaped function_call fixture, not a real LLM decision."""
    return SimpleNamespace(type="function_call", name=name, call_id=call_id,
                           arguments=arguments if isinstance(arguments, str) else json.dumps(arguments))


def response(*calls, text="", status="completed"):
    return SimpleNamespace(status=status, output=list(calls), output_text=text)


def fake_client(*responses):
    return SimpleNamespace(responses=SimpleNamespace(create=Mock(side_effect=list(responses))))


@pytest.fixture
def service():
    rows = [{**row, "similarity": 0.5} for row in load_catalog()[:3]]
    return SimpleNamespace(retriever=SimpleNamespace(search=Mock(return_value=rows)))


@pytest.fixture
def session(service):
    return ToolSession(lambda: service, ["music"])


def auth_error():
    request = httpx2.Request("POST", "https://api.openai.com/v1/responses")
    return AuthenticationError("DO_NOT_LEAK_SECRET", response=httpx2.Response(401, request=request), body=None)


def test_tool_schema_strict_and_required():
    assert {t["name"] for t in TOOLS} == {"search_contents", "rank_contents", "analyze_emotion"}
    for tool in TOOLS:
        assert tool["type"] == "function" and tool["strict"] is True
        assert tool["parameters"]["additionalProperties"] is False
        assert set(tool["parameters"]["required"]) == set(tool["parameters"]["properties"])


def test_search_result_goes_back_to_llm_and_no_ranking(session, service):
    tool_call = call("search_contents", {"query": "calm music", "top_k": 3})
    client = fake_client(response(tool_call), response(text="가상 콘텐츠 [01]을 찾았습니다."))
    result = run_agent("검색만 해줘", session, "gpt-4.1-mini", client)
    assert result["agent_success"] is True and result["llm_turns"] == 2
    assert result["tool_calls"][0]["result_sent_to_llm"] is True
    assert result["tool_calls"][0]["result_sent_turn"] == 2
    assert [t["name"] for t in result["tool_calls"]] == ["search_contents"]
    service.retriever.search.assert_called_once_with(query="calm music", top_k=3)
    history = client.responses.create.call_args_list[1].kwargs["input"]
    assert history[1] is tool_call
    assert history[2]["type"] == "function_call_output" and history[2]["call_id"] == tool_call.call_id
    assert json.loads(history[2]["output"])["search_id"] == "search_1"
    assert json.loads(history[2]["output"])["items"][0]["content_id"] == "01"
    assert client.responses.create.call_args.kwargs["tool_choice"] == "auto"
    assert client.responses.create.call_args.kwargs["parallel_tool_calls"] is False
    assert client.responses.create.call_args.kwargs["store"] is False


def test_greeting_uses_no_tools(session, service):
    result = run_agent("안녕", session, "gpt-4.1-mini", fake_client(response(text="안녕하세요")))
    assert result["agent_success"] and result["tool_calls"] == []
    service.retriever.search.assert_not_called()


def test_rank_reuses_original_function_and_server_candidates(session, monkeypatch):
    from rag.ranking import rank_candidates
    spy = Mock(wraps=rank_candidates)
    monkeypatch.setattr("rag.agent.rank_candidates", spy)
    client = fake_client(
        response(call("search_contents", {"query": "music", "top_k": 3})),
        response(call("rank_contents", {"search_id": "search_1", "top_k": 2, "use_emotion": False}, "call_2")),
        response(text="[01], [02]를 추천합니다."),
    )
    result = run_agent("감정 없이 추천", session, "gpt-4.1-mini", client)
    assert result["agent_success"] and spy.call_count == 1
    assert spy.call_args.args[0] is session.searches["search_1"]
    assert spy.call_args.args[2] == ["music"]
    assert spy.call_args.args[4] is False and session.prediction is None
    assert all(item["weights"]["emotion"] == 0 for item in result["tool_calls"][-1]["output"]["items"])


def test_emotion_callback_and_cached_result(service):
    predictor = Mock(return_value={"emotion": "sadness", "confidence": 1.0, "probabilities": PROBABILITIES})
    session = ToolSession(lambda: service, ["music"], predictor)
    client = fake_client(
        response(call("analyze_emotion", {})),
        response(call("search_contents", {"query": "music", "top_k": 3}, "call_2")),
        response(call("rank_contents", {"search_id": "search_1", "top_k": 2, "use_emotion": True}, "call_3")),
        response(text="모델 예측을 반영한 결과 [01]입니다."),
    )
    result = run_agent("감정 분석과 추천", session, "gpt-4.1-mini", client)
    assert result["agent_success"] and predictor.call_count == 1
    assert result["tool_calls"][-1]["output"]["use_emotion"] is True
    cached = session.execute(call("analyze_emotion", {}, "call_4"))
    assert cached["cache_hit"] is True
    assert session.trace[-1]["executed"] is False and predictor.call_count == 1


@pytest.mark.parametrize("tool, code", [
    (call("unregistered", {}), "unknown_tool"),
    (call("search_contents", "broken JSON"), "invalid_tool_arguments"),
    (call("search_contents", {"query": "x", "top_k": True}), "invalid_tool_arguments"),
    (call("search_contents", {"query": "x", "top_k": 999}), "invalid_tool_arguments"),
    (call("search_contents", {"query": "x", "top_k": 3, "unexpected": 1}), "invalid_tool_arguments"),
    (call("rank_contents", {"search_id": "forged", "top_k": 1, "use_emotion": False}), "search_required"),
    (call("analyze_emotion", {}), "emotion_input_missing"),
])
def test_rejected_tool_calls_are_not_executed(session, tool, code):
    output = session.execute(tool)
    assert output["error"]["code"] == code
    assert session.trace[0]["executed"] is False


def test_missing_emotion_and_insufficient_candidates(session):
    session.execute(call("search_contents", {"query": "music", "top_k": 3}))
    for args, code in [
        ({"search_id": "search_1", "top_k": 2, "use_emotion": True}, "emotion_required"),
        ({"search_id": "search_1", "top_k": 4, "use_emotion": False}, "insufficient_candidates"),
    ]:
        assert session.execute(call("rank_contents", args))["error"]["code"] == code
        assert session.trace[-1]["executed"] is False


def test_tool_error_is_sent_to_llm_and_not_marked_success(session):
    client = fake_client(response(call("analyze_emotion", {})), response(text="음성과 텍스트가 필요합니다."))
    result = run_agent("분석", session, "gpt-4.1-mini", client)
    assert result["status"] == "completed_with_tool_errors" and result["agent_success"] is False
    output = client.responses.create.call_args_list[1].kwargs["input"][-1]
    assert json.loads(output["output"])["error"]["code"] == "emotion_input_missing"


def test_backend_error_is_safe_and_traced(session, service):
    service.retriever.search.side_effect = RuntimeError("DO_NOT_LEAK_SECRET")
    output = session.execute(call("search_contents", {"query": "x", "top_k": 3}))
    assert output["error"]["code"] == "tool_execution_failed"
    assert "DO_NOT_LEAK" not in json.dumps(session.trace) and session.trace[0]["executed"]


@pytest.mark.parametrize("error, code", [
    (auth_error(), "llm_authentication_failed"),
    (APITimeoutError(request=httpx2.Request("POST", "https://api.openai.com/v1/responses")), "llm_timeout"),
])
def test_provider_error_after_tool_keeps_trace_and_never_falls_back(session, error, code):
    client = fake_client(response(call("search_contents", {"query": "x", "top_k": 3})), error)
    with pytest.raises(AgentFailure) as caught:
        run_agent("검색", session, "gpt-4.1-mini", client)
    payload = caught.value.payload("gpt-4.1-mini")
    assert payload["error"]["code"] == code and payload["agent_success"] is False
    assert payload["answer"] is None and len(payload["tool_calls"]) == 1
    assert "DO_NOT_LEAK" not in json.dumps(payload)


def test_tool_budget_stops_loop(session):
    client = fake_client(*[response(call("search_contents", {"query": "x", "top_k": 3}, f"call_{i}"))
                           for i in range(MAX_TOOL_CALLS + 1)])
    with pytest.raises(AgentFailure, match="tool-call budget") as caught:
        run_agent("계속 검색", session, "gpt-4.1-mini", client)
    assert caught.value.code == "tool_call_limit" and len(session.trace) == MAX_TOOL_CALLS


def test_duplicate_call_id_is_protocol_error(session):
    repeated = call("search_contents", {"query": "x", "top_k": 3})
    with pytest.raises(AgentFailure) as caught:
        run_agent("검색", session, "gpt-4.1-mini", fake_client(response(repeated), response(repeated)))
    assert caught.value.code == "invalid_tool_protocol" and len(session.trace) == 1


@pytest.mark.parametrize("reply, code", [
    (response(status="incomplete"), "llm_incomplete"),
    (response(text=" "), "llm_no_answer"),
])
def test_incomplete_or_empty_answer_is_failure(session, reply, code):
    with pytest.raises(AgentFailure) as caught:
        run_agent("안녕", session, "gpt-4.1-mini", fake_client(reply))
    assert caught.value.code == code


@pytest.fixture
def api_client(monkeypatch, service):
    monkeypatch.setenv("EMOTION_CHECKPOINT", "nonexistent-agent-test-checkpoint")
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4.1-mini")
    monkeypatch.setattr("rag.agent_api.get_service", lambda request: service)
    with TestClient(app) as client:
        yield client


def test_agent_http_auth_failure(api_client, monkeypatch, service):
    fake = fake_client(auth_error())
    monkeypatch.setattr("rag.agent.make_client", lambda: nullcontext(fake))
    reply = api_client.post("/agent", data={"message": "음악 검색"})
    assert reply.status_code == 502
    assert reply.json()["error"]["code"] == "llm_authentication_failed"
    assert reply.json()["agent_success"] is False and reply.json()["tool_calls"] == []
    service.retriever.search.assert_not_called()
    assert not app.state.agent_lock.locked()


def test_missing_config_does_not_start_tools(api_client, monkeypatch, service):
    monkeypatch.delenv("OPENAI_API_KEY")
    reply = api_client.post("/agent", data={"message": "검색"})
    assert reply.status_code == 503 and reply.json()["error"]["code"] == "llm_not_configured"
    service.retriever.search.assert_not_called()


def test_agent_http_emotion_reuses_predict_and_closes_tempfile(api_client, monkeypatch):
    predictor = Mock(return_value={"emotion": "sadness", "confidence": 1.0, "probabilities": PROBABILITIES})
    app.state.predictor = SimpleNamespace(preprocessor=SimpleNamespace(build_context=lambda value: value), predict=predictor)
    fake = fake_client(response(call("analyze_emotion", {})), response(text="모델 예측은 sadness입니다."))
    monkeypatch.setattr("rag.agent.make_client", lambda: nullcontext(fake))
    reply = api_client.post("/agent", data={"message": "감정 분석", "text": "private transcript sentinel"},
                            files={"audio": ("../../unsafe.wav", b"audio-sentinel", "audio/wav")})
    assert reply.status_code == 200 and reply.json()["agent_success"]
    assert reply.json()["tool_calls"][0]["executed"] is True
    assert not predictor.call_args.kwargs["audio_path"].exists()
    assert "private transcript sentinel" not in str(fake.responses.create.call_args_list)
    app.state.predictor = None


@pytest.mark.parametrize("fields", [
    {"message": " "}, {"message": "x", "preferences": "broken"},
    {"message": "x", "preferences": '["invalid"]'}, {"message": "x", "text": "no audio"},
])
def test_agent_http_invalid_input(api_client, fields):
    reply = api_client.post("/agent", data=fields)
    assert reply.status_code == 422 and reply.json()["agent_success"] is False


def test_agent_busy_releases_only_its_own_lock(api_client):
    app.state.agent_lock.acquire()
    try:
        reply = api_client.post("/agent", data={"message": "hello"})
        assert reply.status_code == 503 and reply.headers["retry-after"] == "1"
        assert app.state.agent_lock.locked()
    finally:
        app.state.agent_lock.release()
