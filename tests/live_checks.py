"""Pure success checks + real-run reporting. No provider calls and no invented success metrics."""
from datetime import datetime, timezone
import json
from pathlib import Path

CASE_IDS = ("search_only", "rank_without_emotion", "greeting", "emotion_recommendation", "recommend_llm")


def agent_checks(http_status, body, expected_tools):
    """Input: actual HTTP/body and required tool set. Output: separate protocol/execution/final-response checks."""
    trace = body.get("tool_calls", [])
    return {
        "http_200": http_status == 200,
        "agent_success": body.get("agent_success") is True,
        "completed": body.get("status") == "completed",
        "expected_tools_selected": {entry.get("name") for entry in trace} == set(expected_tools),
        "python_functions_executed": (all(entry.get("status") == "ok" for entry in trace)
                                      and {entry.get("name") for entry in trace if entry.get("executed") is True} == set(expected_tools)),
        "tool_outputs_returned_to_llm": all(entry.get("result_sent_to_llm") is True
                                            and bool(entry.get("result_response_id")) for entry in trace),
        "final_answer_generated": bool((body.get("answer") or "").strip()) and bool(body.get("final_response_id")),
        "final_after_tool_outputs": all(1 < entry.get("result_sent_turn", 0) <= body.get("llm_turns", 0) for entry in trace),
    }


def recommendation_checks(http_status, body, model):
    """Input: actual /recommend response and requested model. Output: strict LLM success/citation checks; fallback fails."""
    explanation = body.get("explanation", {})
    items = body.get("items", [])
    reasons = explanation.get("reasons", [])
    by_id = {row["content_id"]: row for row in items}
    ids = [reason.get("content_id") for reason in reasons]
    return {
        "http_200": http_status == 200,
        "three_ranked_items": len(items) == 3,
        "llm_mode": explanation.get("mode") == "llm",
        "generated_status": explanation.get("status") == "generated",
        "requested_model": explanation.get("model") == model,
        "ids_match_once": bool(items) and len(ids) == len(items) and set(ids) == set(by_id),
        "nonempty_reasons": bool(reasons) and all((r.get("reason") or "").strip() for r in reasons),
        "quotes_in_evidence": bool(reasons) and all(
            bool((r.get("evidence_quote") or "").strip()) and r.get("content_id") in by_id
            and r["evidence_quote"] in by_id[r["content_id"]]["description"] for r in reasons),
    }


def write_live_report(report, root):
    """Input: actual live observations and repo root. Output: timestamped JSON/Markdown + latest copies; never mix mocks."""
    root = Path(root)
    report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    observed = {case["case_id"]: case for case in report["cases"]}
    report["not_run"] = [case_id for case_id in CASE_IDS if case_id not in observed]
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    archive = root / "evaluation/live_runs"
    archive.mkdir(parents=True, exist_ok=True)
    json_text = json.dumps(report, indent=2, ensure_ascii=False)
    lines = ["# 실제 LLM 통합 검증 결과", "", f"- 실행 시작(UTC): {report['executed_at_utc']}",
             f"- 모델: {report['model']}", "- Mock 사용: false. 실제 API/함수 실행 관측만 기록.",
             "- 이 문서가 최신 실호출 결과입니다. 과거 Mock 통과/인증 실패 기록과 구분합니다.", "",
             "| 시나리오 | 실제 상태 | HTTP | agent_success | 실제 함수 호출 |",
             "|---|---|---:|---|---|"]
    for case_id in CASE_IDS:
        case = observed.get(case_id)
        if case is None:
            lines.append(f"| {case_id} | 미실행 | — | — | — |")
            continue
        body = case["response"]
        trace = body.get("tool_calls", [])
        names = ", ".join(entry["name"] for entry in trace if entry.get("executed")) or "없음"
        success = str(body["agent_success"]).lower() if "agent_success" in body else "해당 없음"
        lines.append(f"| {case_id} | {'성공' if case['passed'] else '실패'} | {case['http_status']} | {success} | {names} |")
    for case in report["cases"]:
        lines += ["", f"## {case['case_id']}", "", "검사 결과:", ""]
        lines += [f"- {key}: {'PASS' if passed else 'FAIL'}" for key, passed in case["checks"].items()]
        body = case["response"]
        if body.get("error"):
            lines += ["", f"오류 코드: `{body['error']['code']}`"]
        if body.get("tool_calls"):
            lines += ["", "| Tool | call_id | executed | status | 결과 전달 확인 | 응답 turn |",
                      "|---|---|---|---|---|---:|"]
            for entry in body["tool_calls"]:
                lines.append(f"| {entry['name']} | {entry['call_id']} | {entry['executed']} | {entry['status']} | "
                             f"{entry.get('result_sent_to_llm', False)} | {entry.get('result_sent_turn', '—')} |")
        if body.get("answer"):
            lines += ["", "실제 최종 답변:", "", body["answer"]]
        if case["case_id"] == "recommend_llm":
            explanation = body.get("explanation", {})
            lines += ["", f"실제 설명: mode={explanation.get('mode')}, status={explanation.get('status')}", ""]
            for reason in explanation.get("reasons", []):
                lines += [f"- [{reason['content_id']}] {reason['reason']}", f"  - 원문 인용: {reason['evidence_quote']}"]
    lines += ["", "## 해석 한계", "", "Tool 결과 전달은 해당 call_id의 결과를 포함한 요청 뒤 실제 LLM 응답을 받았다는 뜻입니다.",
              "LLM 내부의 이해나 설명 전체의 사실성을 증명하지 않습니다. train split 음성을 사용하는 smoke이며 정확도 평가가 아닙니다.",
              "Mock 회귀 결과는 agent-offline-tests.xml 및 과거 AGENT_TEST_RESULTS.md와 구분합니다.",
              "API 키·인증 헤더·SDK 원문 예외는 기록하지 않습니다.", ""]
    markdown = "\n".join(lines)
    (archive / f"{run_id}.json").write_text(json_text, encoding="utf-8")
    (archive / f"{run_id}.md").write_text(markdown, encoding="utf-8")
    (root / "evaluation/agent_live_results.json").write_text(json_text, encoding="utf-8")
    (root / "evaluation/LIVE_LLM_RESULTS.md").write_text(markdown, encoding="utf-8")
