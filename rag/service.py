"""Composition layer shared by HTTP endpoint and experiments."""
from rag.explanation import explain
from rag.ranking import rank_candidates


class RecommendationService:
    def __init__(self, retriever=None):
        """Input: optional retriever. Output: reusable service; defer heavy imports until first use."""
        if retriever is None:
            from rag.retrieval import ContentRetriever
            retriever = ContentRetriever()
        self.retriever = retriever

    def recommend(self, query, probabilities, preferences=(), retrieval_k=10, top_k=3,
                  use_emotion=True, explanation_mode="auto"):
        """Input: query, model probabilities, user options. Output: ranked evidence + explanation provenance."""
        if top_k > retrieval_k:
            raise ValueError("top_k cannot exceed retrieval_k")
        candidates = self.retriever.search(query, retrieval_k)
        rows = rank_candidates(candidates, probabilities, preferences, top_k, use_emotion)
        return {"items": rows, "retrieved_ids": [row["content_id"] for row in candidates],
                "use_emotion": use_emotion,
                "embedding_model": self.retriever.model_name,
                "embedding_revision": self.retriever.revision,
                "explanation": explain(rows, query, explanation_mode)}
