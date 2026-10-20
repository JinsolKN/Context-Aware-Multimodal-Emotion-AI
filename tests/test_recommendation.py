"""Offline tests: real FAISS, deterministic test encoder, stub emotion model/LLM.

These are contract/error tests, never claimed as real model quality evidence.
"""
import json
from types import SimpleNamespace
from unittest.mock import Mock

from fastapi.testclient import TestClient
import numpy as np
import pytest
import soundfile as sf

from api.app import app
from demo.ranking import LABELS, load_catalog
from tests.ranking_metrics import metrics
from rag.explanation import Reasons, Reason, explain
from rag.ranking import rank_candidates
from rag.retrieval import ContentRetriever
from rag.service import RecommendationService

PROBS = {label: float(label == "sadness") for label in LABELS}


class TestEncoder:
    def encode(self, texts, **kwargs):
        """Input: descriptions/query. Output: tiny test vectors with a known cosine ordering."""
        return np.array([[1, 0] if "calm" in text else [0, 1] for text in texts], dtype="float32")


@pytest.fixture
def retriever():
    return ContentRetriever(load_catalog()[:3], TestEncoder())


@pytest.fixture
def client(monkeypatch, retriever):
    monkeypatch.setenv("EMOTION_CHECKPOINT", "nonexistent-test-checkpoint")
    with TestClient(app) as test_client:
        preprocessor = SimpleNamespace(build_context=lambda context: context)
        app.state.predictor = SimpleNamespace(
            preprocessor=preprocessor,
            predict=Mock(return_value={"emotion": "sadness", "confidence": 1.0, "probabilities": PROBS}),
        )
        app.state.recommend_service = RecommendationService(retriever)
        yield test_client
        app.state.recommend_service = None


def post(client, options=None, audio=b"test-audio", **fields):
    """Input: test fields. Output: multipart HTTP response; exercise actual route composition."""
    data = {"text": "A test utterance", "context": "[]", "options": json.dumps(
        options if options is not None else {"query": "calm", "explanation_mode": "template"})}
    data.update(fields)
    return client.post("/recommend", data=data, files={"audio": ("../../unsafe.wav", audio, "audio/wav")})


def test_faiss_search(retriever):
    rows = retriever.search("calm", 50)
    assert len(rows) == 3
    assert rows[0]["content_id"] == "01"
    assert rows[0]["similarity"] == pytest.approx(1)
    assert rows[1]["similarity"] == pytest.approx(0)


def test_ablation_can_change_order_and_uses_whole_distribution():
    a, b = load_catalog()[:2]
    candidates = [{**a, "similarity": 0.4}, {**b, "similarity": 0.5}]
    mixed = {label: 0.0 for label in LABELS}
    mixed.update(sadness=0.45, joy=0.35, neutral=0.2)
    assert rank_candidates(candidates, mixed, use_emotion=False)[0]["content_id"] == "02"
    aware = rank_candidates(candidates, mixed)
    assert aware[0]["content_id"] == "01"
    assert aware[0]["components"]["emotion"] == pytest.approx(0.65)
    assert sum(aware[0]["weights"].values()) == pytest.approx(1)


@pytest.mark.parametrize("bad", [{}, {**PROBS, "sadness": float("nan")}, {**PROBS, "sadness": 0.5}])
def test_invalid_probabilities(bad):
    with pytest.raises(ValueError):
        rank_candidates([], bad)


def test_recommend_composes_existing_predict(client):
    response = post(client)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"]["probabilities"] == PROBS
    assert body["items"][0]["content_id"] == "01"
    assert body["explanation"]["mode"] == "template"
    assert body["explanation"]["reasons"][0]["content_id"] == "01"
    call = app.state.predictor.predict.call_args.kwargs
    assert not call["audio_path"].exists()  # Existing temporary-file cleanup still executes.
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}
    original = client.post("/predict", data={"text": "hello"}, files={"audio": ("a.wav", b"test")})
    assert original.json() == body["prediction"]


def test_without_emotion_needs_no_model_audio_or_key(client, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    app.state.predictor = None
    response = client.post("/recommend", data={"options": json.dumps({
        "query": "calm", "use_emotion": False, "explanation_mode": "auto"})})
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] is None
    assert body["explanation"]["status"] == "llm_not_configured"
    assert body["items"]
    for row in body["items"]:
        assert row["components"]["emotion"] == 0
        assert row["weights"] == pytest.approx({"similarity": 5 / 7, "preference": 2 / 7, "emotion": 0})


def test_disabled_emotion_does_not_call_predict(client):
    response = post(client, options={"query": "calm", "use_emotion": False, "explanation_mode": "template"})
    assert response.status_code == 200
    app.state.predictor.predict.assert_not_called()


def test_enabled_emotion_requires_existing_inputs(client):
    response = client.post("/recommend", data={"options": '{"query":"calm"}'})
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "emotion_input_missing"


@pytest.mark.parametrize("options", [
    {"query": " "}, {"query": "x", "top_k": 5, "retrieval_k": 2},
    {"query": "x", "preferences": ["unknown"]}, {"query": "x", "top_k": 0},
    {"query": "x", "top_k": 2.5}, {"query": "x", "explanation_mode": "unknown"},
    {"query": "x", "unexpected": "ignored?"}, [],
])
def test_invalid_options(client, options):
    response = post(client, options=options)
    assert response.status_code == 422
    app.state.predictor.predict.assert_not_called()


@pytest.mark.parametrize("raw", ["null", "broken-json"])
def test_malformed_options(client, raw):
    response = client.post("/recommend", data={"text": "hello", "options": raw},
                           files={"audio": ("a.wav", b"test")})
    assert response.status_code == 422


@pytest.mark.parametrize("fields", [{"text": " "}, {"context": "broken"}, {"context": "{}"}])
def test_predict_validation_preserved(client, fields):
    assert post(client, **fields).status_code == 422


def test_missing_checkpoint(client):
    app.state.predictor = None
    assert post(client).status_code == 503


def test_busy_locks_and_recovery(client):
    for lock in (app.state.recommend_lock, app.state.predict_lock):
        lock.acquire()
        try:
            response = post(client)
            assert response.status_code == 503
            assert response.headers["retry-after"] == "1"
        finally:
            lock.release()
    assert post(client).status_code == 200


def test_audio_limits(client):
    assert post(client, audio=b"").status_code == 422
    assert post(client, audio=b"x" * (10 * 1024 * 1024 + 1)).status_code == 413
    app.state.predictor.predict.side_effect = sf.LibsndfileError(1)
    assert post(client).status_code == 422
    assert not app.state.predict_lock.locked()
    assert not app.state.recommend_lock.locked()


def test_retrieval_failure_is_503(client, monkeypatch):
    monkeypatch.setattr("rag.api.get_service", Mock(side_effect=RuntimeError("unavailable")))
    assert post(client).status_code == 503


def test_recommend_failure_is_500(client):
    app.state.recommend_service.recommend = Mock(side_effect=RuntimeError("unavailable"))
    assert post(client).status_code == 500


def test_llm_unconfigured(retriever, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    rows = rank_candidates(retriever.search("calm"), PROBS)
    assert explain(rows, "query")["status"] == "llm_not_configured"


def test_llm_trims_key_before_building_authorization_header(retriever, monkeypatch):
    rows = rank_candidates(retriever.search("calm"), PROBS)
    monkeypatch.setenv("OPENAI_API_KEY", "  sk-test-key\n")
    monkeypatch.setenv("OPENAI_MODEL", " test-model ")
    reasons = Reasons(reasons=[
        Reason(content_id=r["content_id"], reason="설명", evidence_quote=r["description"])
        for r in rows
    ])
    factory = Mock()
    factory.return_value.__enter__ = Mock(return_value=SimpleNamespace(
        responses=SimpleNamespace(parse=Mock(return_value=SimpleNamespace(output_parsed=reasons)))))
    factory.return_value.__exit__ = Mock(return_value=False)
    monkeypatch.setattr("openai.OpenAI", factory)
    assert explain(rows, "query")["mode"] == "llm"
    factory.assert_called_once_with(api_key="sk-test-key", timeout=20.0, max_retries=0)


def test_llm_whitespace_key_is_unconfigured(retriever, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", " \n\t")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    rows = rank_candidates(retriever.search("calm"), PROBS)
    assert explain(rows, "query")["status"] == "llm_not_configured"


@pytest.mark.parametrize("count", [3, 10])
def test_llm_schema_requires_every_selected_id_and_exact_quote(count):
    from pydantic import ValidationError
    from rag.explanation import selected_reasons_schema

    rows = rank_candidates([{**r, "similarity": 0.5} for r in load_catalog()], PROBS, top_k=count)
    schema = selected_reasons_schema(rows)
    payload = {"reasons": {r["content_id"]: {"reason": "설명", "evidence_quote": r["description"]}
                           for r in rows}}
    parsed = schema.model_validate(payload)
    parse = Mock(return_value=SimpleNamespace(output_parsed=parsed))
    result = explain(rows, "query", "llm", client=SimpleNamespace(responses=SimpleNamespace(parse=parse)))
    assert result["mode"] == "llm"
    assert [r["content_id"] for r in result["reasons"]] == [r["content_id"] for r in rows]
    key = rows[0]["content_id"]
    payload["reasons"][key]["evidence_quote"] = "translated or fabricated quote"
    with pytest.raises(ValidationError):
        schema.model_validate(payload)
    del payload["reasons"][key]
    with pytest.raises(ValidationError):
        schema.model_validate(payload)


@pytest.mark.parametrize("variant", ["valid", "unknown_id", "bad_quote", "refusal", "timeout", "duplicate"])
def test_llm_citations_and_fallback(retriever, variant):
    rows = rank_candidates(retriever.search("calm"), PROBS)
    reasons = [Reason(content_id=r["content_id"], reason="설명", evidence_quote=r["description"]) for r in rows]
    if variant == "unknown_id":
        reasons[0].content_id = "999"
    if variant == "bad_quote":
        reasons[0].evidence_quote = "invented fact"
    if variant == "duplicate":
        reasons[0].content_id = reasons[1].content_id
    parse = Mock(return_value=SimpleNamespace(output_parsed=None if variant == "refusal" else Reasons(reasons=reasons)))
    if variant == "timeout":
        parse.side_effect = TimeoutError
    result = explain(rows, "query", "llm", client=SimpleNamespace(responses=SimpleNamespace(parse=parse)))
    assert result["mode"] == ("llm" if variant == "valid" else "template")
    assert parse.call_args.kwargs["store"] is False


def test_metric_known_values():
    assert metrics(["a", "b"], {"a": 3, "b": 1}, 2) == {"ndcg": 1.0, "recall": 1.0, "mrr": 1.0}
    assert metrics(["x", "a"], {"a": 3}, 2)["mrr"] == 0.5
    assert metrics(["x"], {"a": 3}, 1)["ndcg"] == 0
