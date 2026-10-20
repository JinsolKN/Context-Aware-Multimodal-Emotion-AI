"""Offline HTTP adapter and Streamlit interaction tests; no model or paid LLM calls."""
import json
from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest
from streamlit.testing.v1 import AppTest

from demo.pipeline_client import PipelineError, chat_agent, recommend_contents

ROOT = Path(__file__).resolve().parents[1]


def recommendation():
    return {"prediction": None, "use_emotion": False, "retrieved_ids": ["01"],
            "embedding_model": "test-encoder", "items": [{"content_id": "01", "title": "Test content",
            "description": "Test description", "similarity": 0.8, "score": 0.7,
            "components": {"similarity": 0.9, "preference": 0, "emotion": 0},
            "weights": {"similarity": 5/7, "preference": 2/7, "emotion": 0}}],
            "explanation": {"mode": "template", "status": "llm_not_configured", "reasons": [
                {"content_id": "01", "reason": "Test reason", "evidence_quote": "Test description"}]}}


def response(body, status=200):
    return httpx.Response(status, json=body)


def test_recommend_http_fields_and_no_emotion_input(monkeypatch):
    post = Mock(return_value=response(recommendation()))
    monkeypatch.setattr(httpx, "post", post)
    assert recommend_contents("calm", ["music"], 10, 3, False, "template", "http://localhost:8000")["items"]
    args = post.call_args
    assert args.args[0].endswith("/recommend")
    assert args.kwargs["files"] is None
    assert "text" not in args.kwargs["data"]
    assert json.loads(args.kwargs["data"]["options"])["use_emotion"] is False


@pytest.mark.parametrize("failure", [httpx.ConnectError("secret"), httpx.ReadTimeout("secret")])
def test_network_errors_safe(monkeypatch, failure):
    monkeypatch.setattr(httpx, "post", Mock(side_effect=failure))
    with pytest.raises(PipelineError) as error:
        chat_agent("hello", [], "http://localhost:8000")
    assert "secret" not in str(error.value)


def test_agent_failure_keeps_actual_trace(monkeypatch):
    body = {"status": "failed", "tool_calls": [{"name": "search_contents", "executed": True,
             "status": "ok", "output": {"items": []}}], "error": {"code": "llm_timeout", "message": "secret"}}
    monkeypatch.setattr(httpx, "post", Mock(return_value=response(body, 504)))
    with pytest.raises(PipelineError) as error:
        chat_agent("hello", [], "http://localhost:8000")
    assert error.value.payload["tool_calls"] == body["tool_calls"]
    assert "secret" not in str(error.value)


def test_invalid_k_and_missing_audio_prevent_http(monkeypatch):
    post = Mock()
    monkeypatch.setattr(httpx, "post", post)
    for retrieval_k, top_k, use_emotion in [(1, 3, False), (10, 3, True)]:
        with pytest.raises(PipelineError):
            recommend_contents("calm", [], retrieval_k, top_k, use_emotion, "template", "http://localhost:8000")
    post.assert_not_called()


@pytest.fixture
def ui(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "demo"))
    return AppTest.from_file(str(ROOT / "demo/streamlit_app.py"), default_timeout=20).run()


def test_tabs_and_existing_missing_audio_behavior(ui):
    assert not ui.exception
    assert [tab.label for tab in ui.tabs] == ["Emotion Analysis", "RAG Recommendation", "AI Agent"]
    ui.text_area[0].set_value("hello")
    ui.button[0].click().run()
    assert not ui.exception
    assert any("오디오를 업로드" in error.value for error in ui.error)


def test_recommend_ui_persists_real_fields(ui, monkeypatch):
    post = Mock(return_value=response(recommendation()))
    monkeypatch.setattr(httpx, "post", post)
    ui.text_area(key="rag_query").set_value("calm")
    next(b for b in ui.button if b.label == "추천 실행").click().run()
    assert not ui.exception
    assert ui.dataframe
    assert any("Test reason" in text.value for text in ui.markdown)
    assert any("llm_not_configured" in text.value for text in ui.caption)
    ui.run()
    assert post.call_count == 1


def test_agent_ui_greeting_does_not_invent_tools(ui, monkeypatch):
    post = Mock(return_value=response({"answer": "안녕하세요!", "status": "completed", "llm_turns": 1,
                                      "tool_calls": [], "agent_success": True}))
    monkeypatch.setattr(httpx, "post", post)
    ui.chat_input[0].set_value("안녕하세요.").run()
    assert not ui.exception
    assert len(ui.chat_message) == 2
    assert any("Tool 호출 기록이 없습니다" in text.value for text in ui.caption)
    assert "history" not in post.call_args.kwargs["data"]


def test_agent_ui_actual_ranking_trace(ui, monkeypatch):
    body = {"answer": "[01] Test content", "status": "completed", "llm_turns": 3, "agent_success": True,
            "tool_calls": [{"name": "rank_contents", "executed": True, "status": "ok",
                            "arguments": {"use_emotion": False}, "output": {"items": recommendation()["items"]}}]}
    monkeypatch.setattr(httpx, "post", Mock(return_value=response(body)))
    ui.chat_input[0].set_value("감정을 제외하고 추천해줘").run()
    assert not ui.exception
    assert ui.dataframe
    assert any("rank_contents" in text.value and "executed=True" in text.value for text in ui.markdown)


def test_existing_emotion_result_rendering(ui, monkeypatch):
    import demo_client
    from demo.ranking import LABELS
    probabilities = {label: float(label == "sadness") for label in LABELS}
    monkeypatch.setattr(demo_client, "predict_emotion", Mock(return_value={
        "emotion": "sadness", "confidence": 1.0, "probabilities": probabilities}))
    ui.text_area[0].set_value("hello")
    ui.button[0].click().run()
    assert not ui.exception
    assert any("sadness" in metric.value for metric in ui.metric)
    assert ui.session_state["prediction"]["result"]["probabilities"] == probabilities


def test_llm_explanation_and_failed_agent_trace_rendering(ui, monkeypatch):
    result = recommendation()
    result["explanation"].update(mode="llm", status="generated")
    monkeypatch.setattr(httpx, "post", Mock(return_value=response(result)))
    ui.text_area(key="rag_query").set_value("calm")
    next(b for b in ui.button if b.label == "추천 실행").click().run()
    assert not ui.exception
    assert any("llm" in text.value and "generated" in text.value for text in ui.caption)
    failure = {"status": "failed", "llm_turns": 2, "error": {"code": "llm_timeout"},
               "tool_calls": [{"name": "search_contents", "executed": True, "status": "ok",
                               "output": {"items": []}}]}
    monkeypatch.setattr(httpx, "post", Mock(return_value=response(failure, 504)))
    ui.chat_input[0].set_value("검색해줘").run()
    assert not ui.exception
    assert ui.error
    assert any("search_contents" in text.value for text in ui.markdown)
