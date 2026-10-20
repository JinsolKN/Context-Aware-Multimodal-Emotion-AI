"""LIVE ONLY: real LLM + actual tool selection + real embedding/FAISS. No mocks or fallback success.

Enable RUN_AGENT_LIVE=1 with key/model in this process. Optional full emotion case
also needs RUN_AGENT_EMOTION=1, EMOTION_CHECKPOINT/ASSETS and EMOTION_TEST_BUNDLE.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from api.app import app
from tests.live_checks import agent_checks, recommendation_checks, write_live_report

pytestmark = [pytest.mark.live, pytest.mark.skipif(
    os.getenv("RUN_AGENT_LIVE") != "1", reason="Opt-in paid/network test: set RUN_AGENT_LIVE=1")]


@pytest.fixture(scope="module")
def live_client():
    """Input: explicitly enabled live env. Output: real HTTP client and report; record failures as failures."""
    report = {"executed_at_utc": datetime.now(timezone.utc).isoformat(),
              "run_id": os.getenv("LIVE_VERIFICATION_RUN_ID"),
              "model": os.getenv("OPENAI_MODEL"), "uses_mocks": False,
              "purpose": "Function-calling smoke, not a quality benchmark", "cases": []}
    try:
        with TestClient(app) as client:
            app.state.recommend_service = None
            yield client, report
            app.state.recommend_service = None
    finally:
        write_live_report(report, Path(__file__).resolve().parents[1])


def check_result(client, report, case_id, data, expected, files=None):
    """Input: scenario + expected tools. Output: verified response; persist actual trace before asserting."""
    response = client.post("/agent", data=data, files=files)
    body = response.json()
    checks = agent_checks(response.status_code, body, expected)
    passed = all(checks.values())
    report["cases"].append({"case_id": case_id, "expected_tools": sorted(expected),
                            "http_status": response.status_code, "passed": passed, "checks": checks, "response": body})
    assert passed, f"Live scenario failed: {body.get('error')}; failed checks={[k for k, v in checks.items() if not v]}"


def test_live_search_only(live_client):
    client, report = live_client
    check_result(client, report, "search_only", {
        "message": "차분한 음악 콘텐츠 3개를 검색만 해 주세요. Ranking과 감정 분석은 하지 마세요.",
        "preferences": '["music"]',
    }, {"search_contents"})


def test_live_recommend_without_emotion(live_client):
    client, report = live_client
    check_result(client, report, "rank_without_emotion", {
        "message": "조용한 음악을 검색한 뒤 선호와 검색 유사도로 2개를 Ranking해서 추천해 주세요. 감정은 제외하세요.",
        "preferences": '["music"]',
    }, {"search_contents", "rank_contents"})


def test_live_greeting_without_tools(live_client):
    client, report = live_client
    check_result(client, report, "greeting", {"message": "안녕하세요. 한 문장으로 인사만 해 주세요."}, set())


@pytest.mark.skipif(os.getenv("RUN_AGENT_EMOTION") != "1", reason="Needs real audio/checkpoint; set RUN_AGENT_EMOTION=1")
def test_live_emotion_recommendation(live_client):
    client, report = live_client
    bundle = Path(os.environ["EMOTION_TEST_BUNDLE"])
    sample = json.loads(bundle.read_text(encoding="utf-8"))[0]
    audio = (bundle.parent / sample["audio_filename"]).read_bytes()
    check_result(client, report, "emotion_recommendation", {
        "message": "첨부한 음성과 발화로 모델 감정을 분석하고, 차분한 음악을 검색해 감정과 선호를 반영하여 2개 추천해 주세요.",
        "preferences": '["music"]', "text": sample["current_utterance"]["text"],
        "context": json.dumps(sample["context_3"]),
    }, {"analyze_emotion", "search_contents", "rank_contents"}, files={"audio": ("sample.wav", audio, "audio/wav")})


@pytest.mark.skipif(os.getenv("RUN_RECOMMEND_LIVE") != "1", reason="Needs real checkpoint/audio/key: set RUN_RECOMMEND_LIVE=1")
def test_live_recommend_llm_explanation(live_client):
    """Input: real bundle/env. Output: real /recommend result; HTTP 200 template fallback is a test failure."""
    client, report = live_client
    bundle = Path(os.environ["EMOTION_TEST_BUNDLE"])
    sample = json.loads(bundle.read_text(encoding="utf-8"))[0]
    audio = (bundle.parent / sample["audio_filename"]).read_bytes()
    response = client.post("/recommend", data={
        "text": sample["current_utterance"]["text"], "context": json.dumps(sample["context_3"]),
        "options": json.dumps({"query": "조용한 음악이나 빗소리", "preferences": ["music", "relaxation"],
                               "top_k": 3, "retrieval_k": 10, "use_emotion": True, "explanation_mode": "llm"}),
    }, files={"audio": ("sample.wav", audio, "audio/wav")})
    body = response.json()
    checks = recommendation_checks(response.status_code, body, os.getenv("OPENAI_MODEL", "").strip())
    passed = all(checks.values())
    report["cases"].append({"case_id": "recommend_llm", "http_status": response.status_code,
                            "passed": passed, "checks": checks, "response": body})
    assert passed, f"Real LLM explanation failed: {[k for k, v in checks.items() if not v]}"
