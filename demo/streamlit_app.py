"""Run from project root: python -m streamlit run demo/streamlit_app.py. No model imports."""
import streamlit as st

from demo_client import API_URL, DemoError, parse_context, predict_emotion
from ranking import CATEGORIES, LABELS, load_catalog, rank_contents
from pipeline_client import PipelineError, chat_agent, recommend_contents

EMOTION_LABELS_KO = {
    "anger": "분노", "disgust": "혐오", "fear": "두려움", "joy": "기쁨",
    "neutral": "중립", "sadness": "슬픔", "surprise": "놀람",
}
CATEGORY_LABELS_KO = {
    "music": "음악", "comedy": "가벼운 코미디", "motivation": "동기부여",
    "relaxation": "휴식", "drama": "드라마",
}
ENERGY_LABELS_KO = {None: "선택 안 함", "low": "낮음", "medium": "보통", "high": "높음"}
CONTENT_TEXT_KO = {
    "Relaxing Piano": ("편안한 피아노", "차분한 피아노 연주를 상상한 데모 콘텐츠"),
    "Sunny Acoustic": ("햇살 같은 어쿠스틱", "따뜻한 어쿠스틱 연주를 상상한 데모 콘텐츠"),
    "Power Beats": ("파워 비트", "활기찬 리듬을 상상한 데모 콘텐츠"),
    "Quiet Rain": ("고요한 빗소리", "부드러운 빗소리를 상상한 데모 콘텐츠"),
    "Evening Breathing": ("저녁 호흡", "느린 호흡 안내를 상상한 데모 콘텐츠"),
    "Soft Forest": ("잔잔한 숲", "숲 소리를 상상한 데모 콘텐츠"),
    "Everyday Mixups": ("일상의 작은 실수", "가벼운 상황극을 상상한 데모 콘텐츠"),
    "Tiny Happy Moments": ("작은 행복의 순간", "부드러운 유머 장면을 상상한 데모 콘텐츠"),
    "Unexpected Guests": ("뜻밖의 손님", "활기찬 코미디 장면을 상상한 데모 콘텐츠"),
    "One Small Step": ("작은 한 걸음", "용기를 북돋는 짧은 이야기를 상상한 데모 콘텐츠"),
    "Fresh Start": ("새로운 시작", "새 출발을 다짐하는 이야기를 상상한 데모 콘텐츠"),
    "Keep Going": ("계속 나아가기", "힘을 북돋는 성장 이야기를 상상한 데모 콘텐츠"),
    "Letters Home": ("집으로 보내는 편지", "가족을 돌아보는 이야기를 상상한 데모 콘텐츠"),
    "The Missing Key": ("사라진 열쇠", "가벼운 미스터리 장면을 상상한 데모 콘텐츠"),
    "A Difficult Choice": ("어려운 선택", "의견 충돌에 관한 이야기를 상상한 데모 콘텐츠"),
}

def emotion_name(label):
    return f"{EMOTION_LABELS_KO[label]} ({label})"


def ranking_reason(item, preferences, desired_energy, probabilities):
    tags = sorted(item["emotion_tags"], key=lambda tag: (-probabilities[tag], tag))
    emotion = ", ".join(f"{EMOTION_LABELS_KO[tag]} {probabilities[tag]:.1%}" for tag in tags)
    if preferences:
        preference = "선택한 선호 장르와 일치" if item["category"] in preferences else "선택한 선호 장르와 다름"
    else:
        preference = "선호 장르를 선택하지 않음"
    if desired_energy is None:
        energy = "콘텐츠 에너지를 선택하지 않음"
    else:
        energy = "원하는 에너지와 일치" if item["energy"] == desired_energy else "원하는 에너지와 다름"
    return f"감정 일치: {emotion} · 장르: {preference} · 에너지: {energy}"


def emotion_inputs(prefix):
    text = st.text_area("감정 모델에 전달할 현재 발화", key=f"{prefix}_text")
    audio = st.file_uploader("감정 모델용 음성 · 16 kHz WAV, 최대 10 MiB", type=["wav"], key=f"{prefix}_audio")
    context = st.text_area("최근 대화 · 한 줄에 Speaker: utterance", key=f"{prefix}_context")
    st.caption("최근 3개 발화를 사용합니다. 텍스트만으로는 기존 멀티모달 모델의 감정을 추론하지 않습니다.")
    return text, audio, context


def render_items(items, explanation=None):
    if not items:
        st.info("반환된 콘텐츠가 없습니다.")
        return
    scores = []
    for item in items:
        components = item.get("components", {})
        scores.append({"ID": item["content_id"], "제목": item["title"],
                       **({"Ranking Score": item["score"]} if "score" in item else {}),
                       **({"Cosine Similarity": item["similarity"]} if "similarity" in item else {}),
                       **{f"{key.title()} Score": value for key, value in components.items()}})
    st.dataframe(scores, hide_index=True, width="stretch")
    if all("score" in item for item in items):
        st.bar_chart({"콘텐츠": [item["content_id"] for item in items],
                      "Ranking Score": [item["score"] for item in items]}, x="콘텐츠", y="Ranking Score")
    reasons = {r["content_id"]: r for r in (explanation or {}).get("reasons", [])}
    for item in items:
        st.markdown(f"**[{item['content_id']}] {item['title']}**")
        st.write(item["description"])
        reason = reasons.get(item["content_id"])
        if reason:
            st.write("추천 이유: " + reason["reason"])
            if reason.get("evidence_quote"):
                st.text("근거 인용: " + reason["evidence_quote"])
    st.caption("가상 데모 콘텐츠입니다. 점수는 추천 확률이 아닙니다. Similarity Score는 Ranking용 정규화 값이며 Cosine Similarity와 구분합니다.")


def render_recommendation(api_url):
    st.subheader("RAG Recommendation")
    st.caption("Query → Embedding → FAISS Retrieval → Ranking → Top K → LLM Explanation (또는 template)")
    query = st.text_area("사용자 텍스트 / 검색어", max_chars=1000, key="rag_query")
    preferences = st.multiselect("선호 콘텐츠 카테고리 (선택)", CATEGORIES,
                                format_func=lambda x: CATEGORY_LABELS_KO[x], key="rag_preferences")
    use_emotion = st.checkbox("Emotion 활용", value=False)
    text, audio, context = emotion_inputs("rag") if use_emotion else ("", None, "")
    left, right = st.columns(2)
    retrieval_k = left.number_input("Retrieval K", min_value=1, max_value=50, value=10)
    top_k = right.number_input("Recommendation Top K", min_value=1, max_value=10, value=3)
    mode = st.selectbox("설명 모드", ["auto", "template", "llm"],
                        format_func=lambda x: {"auto": "Auto · 설정 시 LLM, 실패 시 template", "template": "Template · LLM 호출 안 함", "llm": "LLM · 실패 시 기존 template fallback"}[x])
    st.caption("기존 가중치: Similarity 0.5 · Preference 0.2 · Emotion 0.3. 감정을 끄면 0.5/0.7, 0.2/0.7, 0으로 재정규화됩니다.")
    snapshot = (api_url, query, preferences, use_emotion, text, audio.file_id if audio else None, context, retrieval_k, top_k, mode)
    if st.button("추천 실행", type="primary"):
        st.session_state.pop("rag_result", None)
        try:
            with st.spinner("검색·Ranking·설명을 실행하고 있습니다…"):
                result = recommend_contents(query, preferences, retrieval_k, top_k, use_emotion, mode, api_url,
                                            text, audio.getvalue() if audio else None, parse_context(context))
            st.session_state.rag_result = (snapshot, result)
        except DemoError as exc:
            st.error(str(exc))
    saved = st.session_state.get("rag_result")
    if saved:
        if saved[0] != snapshot:
            st.info("입력이 변경되었습니다. 추천 실행을 다시 눌러주세요.")
            return
        result = saved[1]
        explanation = result["explanation"]
        st.caption(f"실제 설명 방식: {explanation['mode']} · 상태: {explanation['status']}")
        if explanation["status"] in ("llm_not_configured", "llm_failed_or_invalid"):
            st.info("LLM 설명을 생성하지 못해 기존 template 설명을 표시합니다. 추천 결과는 유지됩니다.")
        if result.get("prediction"):
            st.write("모델 예측 감정: " + emotion_name(result["prediction"]["emotion"]))
        render_items(result["items"], explanation)
        with st.expander("검색 및 Ranking 정보"):
            st.json({key: result[key] for key in ("retrieved_ids", "embedding_model", "embedding_revision", "use_emotion") if key in result})
            if result["items"]:
                st.write("실제 적용 가중치", result["items"][0].get("weights", {}))


def render_agent_result(result):
    if result.get("status") == "completed_with_tool_errors":
        st.warning("Agent가 답변했지만 일부 Tool에 오류가 있었습니다.")
    calls = result.get("tool_calls", [])
    with st.expander("Tool 실행 기록 및 최종 추천 콘텐츠"):
        if not calls:
            st.caption("반환된 Tool 호출 기록이 없습니다.")
        for index, call in enumerate(calls, 1):
            st.write(f"{index}. {call['name']} · executed={call['executed']} · status={call['status']}")
            st.json({key: call[key] for key in ("arguments", "output", "result_sent_to_llm") if key in call})
        ranked = [call for call in calls if call.get("name") == "rank_contents" and call.get("status") == "ok"]
        if ranked:
            st.write("마지막으로 성공한 Ranking 결과")
            render_items(ranked[-1]["output"]["items"])
        st.caption(f"Agent 상태: {result.get('status')} · LLM turns: {result.get('llm_turns')}")


def render_agent(api_url):
    st.subheader("AI Agent")
    st.caption("인사, 콘텐츠 검색, 감정 제외 추천 또는 감정 기반 추천을 요청하세요.")
    st.info("기존 Agent는 요청별 독립 실행입니다. 아래 채팅 기록은 화면에만 저장되며 이전 메시지는 서버에 전송되지 않습니다.")
    preferences = st.multiselect("Agent 선호 카테고리", CATEGORIES, key="agent_preferences",
                                format_func=lambda x: CATEGORY_LABELS_KO[x])
    attach = st.checkbox("감정 분석용 텍스트·음성 첨부", value=False)
    text, audio, context = emotion_inputs("agent") if attach else ("", None, "")
    st.caption("예: 안녕하세요. / 기분 전환할 콘텐츠를 찾아줘. / 감정을 제외하고 콘텐츠를 추천해줘. / 첨부한 감정을 분석해 콘텐츠를 추천해줘.")
    if st.button("화면 대화 기록 지우기"):
        st.session_state.agent_messages = []
    messages = st.session_state.setdefault("agent_messages", [])
    for entry in messages:
        with st.chat_message(entry["role"]):
            if entry.get("error"):
                st.error(entry["content"])
            else:
                st.write(entry["content"])
            if entry.get("result") is not None:
                render_agent_result(entry["result"])
    message = st.chat_input("Agent에게 요청하세요", max_chars=2000)
    if message:
        messages.append({"role": "user", "content": message})
        with st.chat_message("user"):
            st.write(message)
        with st.chat_message("assistant"):
            try:
                if attach and (not text.strip() or audio is None):
                    raise PipelineError("감정 분석용 현재 발화와 WAV 음성을 함께 입력하세요.")
                with st.spinner("Agent가 요청을 처리하고 있습니다…"):
                    result = chat_agent(message, preferences, api_url, text,
                                        audio.getvalue() if audio else None, parse_context(context))
                st.write(result["answer"])
                render_agent_result(result)
                messages.append({"role": "assistant", "content": result["answer"], "result": result})
            except DemoError as exc:
                st.error(str(exc))
                result = getattr(exc, "payload", None)
                if result is not None:
                    render_agent_result(result)
                messages.append({"role": "assistant", "content": str(exc), "error": True, "result": result})


def main():
    st.set_page_config(page_title="Multimodal Emotion AI", page_icon="🎧")
    st.title("Multimodal Emotion AI")
    st.caption("감정 분석 · RAG 추천 · Tool Calling Agent")
    api_url = st.sidebar.text_input("FastAPI server URL", value=API_URL)
    st.sidebar.caption("모델은 FastAPI에서만 로드됩니다. CPU 추론과 첫 검색은 시간이 걸릴 수 있습니다.")
    emotion, recommendation, agent = st.tabs(["Emotion Analysis", "RAG Recommendation", "AI Agent"])
    with emotion:
        render_emotion(api_url)
    with recommendation:
        render_recommendation(api_url)
    with agent:
        render_agent(api_url)


def render_emotion(api_url):
    text = st.text_area("현재 발화", placeholder="I can't believe you did that.")
    audio = st.file_uploader("음성 파일 · 16 kHz WAV, 최대 10 MiB", type=["wav"])
    context = st.text_area("최근 대화", placeholder="Ross: What happened?\nRachel: I need to tell you something.")
    st.caption("한 줄에 Speaker: utterance. 오래된 순으로 입력하세요. 최종 checkpoint의 context 설정은 최근 3개로 고정되어 있습니다. 빈 context도 가능합니다.")
    preferences = st.multiselect("선호하는 콘텐츠 장르 (선택)", CATEGORIES,
                                 format_func=lambda value: CATEGORY_LABELS_KO[value])
    st.caption("선택한 장르에는 추천 점수가 더해집니다. 선택하지 않으면 감정 확률만으로 순위를 계산합니다.")
    energy = st.selectbox("지금 원하는 콘텐츠 분위기 (에너지)", [None, "low", "medium", "high"],
                          format_func=lambda value: ENERGY_LABELS_KO[value])
    st.caption("에너지는 콘텐츠의 자극과 활동성입니다. 감정의 강도가 아닙니다. 낮음: 조용하고 편안함 · 보통: 부담 없이 즐김 · 높음: 활기차고 자극적임")
    top_k = st.radio("보여줄 추천 수", [3, 5], format_func=lambda value: f"Top {value}", horizontal=True)
    if st.button("감정 분석하기", type="primary"):
        st.session_state.pop("prediction", None)
        try:
            with st.spinner("FastAPI에서 감정을 분석하고 있습니다…"):
                prediction = predict_emotion(text, audio.getvalue() if audio else b"", parse_context(context), api_url)
            st.session_state["prediction"] = {"result": prediction, "text": text, "context": context,
                                               "audio_id": audio.file_id if audio else None, "api_url": api_url}
        except DemoError as exc:
            st.error(str(exc))
    saved = st.session_state.get("prediction")
    if saved:
        if (saved["text"], saved["context"], saved["audio_id"], saved["api_url"]) != (text, context, audio.file_id if audio else None, api_url):
            st.info("입력이 변경되었습니다. 감정 분석하기를 다시 눌러주세요.")
            return
        prediction = saved["result"]
        st.subheader("예측 감정")
        st.metric("감정", emotion_name(prediction["emotion"]), f"신뢰도 {prediction['confidence']:.1%}")
        st.caption("신뢰도는 보정(calibration)하지 않은 모델 softmax 값입니다.")
        st.subheader("감정 분포")
        st.bar_chart({"감정": [emotion_name(label) for label in LABELS],
                      "확률": [prediction["probabilities"][label] for label in LABELS]}, x="감정", y="확률")
        st.subheader("추천 콘텐츠 순위")
        st.caption("모든 항목은 데모를 위해 만든 가상 콘텐츠입니다. 실제 재생 콘텐츠가 아니며, 점수는 추천 확률이 아닙니다.")
        try:
            contents = load_catalog()
            ranked = rank_contents(prediction["probabilities"], preferences, contents, top_k, energy)
        except (OSError, ValueError):
            st.error("Ranking catalog가 없거나 올바르지 않습니다. data/demo_contents.csv를 확인하세요.")
            return
        for index, item in enumerate(ranked, 1):
            title, description = CONTENT_TEXT_KO[item["title"]]
            st.markdown(f"**{index}. {title}** · {CATEGORY_LABELS_KO[item['category']]} · {ENERGY_LABELS_KO[item['energy']]} 에너지")
            st.write(f"추천 점수: {item['score']:.3f}")
            st.write(description)
            st.write("이유: " + ranking_reason(item, preferences, energy, prediction["probabilities"]))
            st.caption(" · ".join((
                f"감정 태그 일치: {item['components']['emotion_match']:.3f}",
                f"선호 장르 일치: {item['components']['preference_match']:.3f}",
                f"에너지 일치: {item['components']['context_match']:.3f}",
            )))
        st.caption("추천 점수 계산: 0.5 × 감정 태그 확률합 + 0.3 × 선호 장르 일치 + 0.2 × 원하는 에너지 일치. 학습으로 최적화하지 않은 데모용 가중치입니다.")
if __name__ == "__main__":
    main()
