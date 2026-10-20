"""OFFLINE: validate live-result classification/reporting using labeled synthetic response fixtures."""
from copy import deepcopy
import json

from tests.live_checks import agent_checks, recommendation_checks, write_live_report


def agent_body():
    """Input: none. Output: synthetic protocol evidence; never a claim about a real provider run."""
    return {"status": "completed", "agent_success": True, "answer": "fixture answer",
            "final_response_id": "mock_final", "llm_turns": 2, "tool_calls": [
                {"name": "search_contents", "call_id": "mock_call", "executed": True,
                 "status": "ok", "result_sent_to_llm": True, "result_response_id": "mock_response", "result_sent_turn": 2}]}


def recommendation_body():
    """Input: none. Output: synthetic citation fixture for pure checks; no LLM or API invocation."""
    return {"items": [{"content_id": str(i), "description": "fixture description"} for i in range(3)],
            "explanation": {"mode": "llm", "status": "generated", "model": "test-model",
                            "reasons": [{"content_id": str(i), "reason": "fixture reason", "evidence_quote": "description"} for i in range(3)]}}


def test_live_agent_requires_result_delivery_and_actual_execution():
    body = agent_body()
    assert all(agent_checks(200, body, {"search_contents"}).values())
    body["tool_calls"][0]["result_sent_to_llm"] = False
    assert not all(agent_checks(200, body, {"search_contents"}).values())
    body["tool_calls"][0]["executed"] = False
    assert not agent_checks(200, body, {"search_contents"})["python_functions_executed"]


def test_live_agent_requires_final_answer_and_success():
    body = agent_body()
    body["answer"] = " "
    assert not agent_checks(200, body, {"search_contents"})["final_answer_generated"]
    body["agent_success"] = False
    assert not agent_checks(200, body, {"search_contents"})["agent_success"]


def test_live_greeting_no_tools_is_valid_but_search_without_tools_fails():
    body = agent_body()
    body.update(tool_calls=[], llm_turns=1)
    assert all(agent_checks(200, body, set()).values())
    assert not agent_checks(200, body, {"search_contents"})["expected_tools_selected"]


def test_live_recommendation_never_accepts_template_http_200():
    body = recommendation_body()
    assert all(recommendation_checks(200, body, "test-model").values())
    body["explanation"].update(mode="template", status="llm_failed_or_invalid")
    checks = recommendation_checks(200, body, "test-model")
    assert checks["http_200"] and not checks["llm_mode"] and not checks["generated_status"]


def test_live_recommendation_requires_correct_ids_and_quotes():
    body = recommendation_body()
    body["explanation"]["reasons"][0]["content_id"] = "2"
    assert not recommendation_checks(200, body, "test-model")["ids_match_once"]
    body = recommendation_body()
    body["explanation"]["reasons"][0]["evidence_quote"] = "invented quote"
    assert not recommendation_checks(200, body, "test-model")["quotes_in_evidence"]


def test_report_records_observations_and_not_run_separately(tmp_path):
    body = agent_body()
    report = {"executed_at_utc": "fixture-time", "model": "test-model", "uses_mocks": False,
              "cases": [{"case_id": "search_only", "http_status": 200, "passed": True,
                         "checks": agent_checks(200, body, {"search_contents"}), "response": body}]}
    # The fixture's explicit metadata tests the reporter only; all files stay in pytest's temporary folder.
    write_live_report(deepcopy(report), tmp_path)
    saved = json.loads((tmp_path / "evaluation/agent_live_results.json").read_text(encoding="utf-8"))
    assert "recommend_llm" in saved["not_run"] and "search_only" not in saved["not_run"]
    text = (tmp_path / "evaluation/LIVE_LLM_RESULTS.md").read_text(encoding="utf-8")
    assert "| recommend_llm | 미실행 |" in text
    assert len(list((tmp_path / "evaluation/live_runs").glob("*.json"))) == 1
