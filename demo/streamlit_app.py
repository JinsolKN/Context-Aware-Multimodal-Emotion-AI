"""Run: python -m streamlit run streamlit_app.py. No model imports here."""
import streamlit as st

from demo_client import API_URL, DemoError, parse_context, predict_emotion
from ranking import CATEGORIES, LABELS, load_catalog, rank_contents

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


def main():
    st.set_page_config(page_title="Multimodal Emotion AI", page_icon="🎧")
    st.title("Multimodal Emotion AI")
    st.caption("텍스트 + 음성 + 최근 대화 → 감정 확률 → 간단한 콘텐츠 순위")
    api_url = st.sidebar.text_input("FastAPI server URL", value=API_URL)
    st.sidebar.caption("CPU 추론은 시간이 걸릴 수 있습니다. 모델은 FastAPI에서만 로드됩니다.")
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
                                               "audio_id": audio.file_id, "api_url": api_url}
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
