"""Sentence Transformer descriptions + exact FAISS cosine retrieval (no training)."""
import os
import threading

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from demo.ranking import load_catalog

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class ContentRetriever:
    def __init__(self, contents=None, encoder=None):
        """Input: optional catalog/encoder. Output: in-memory index; embed once to reuse per request."""
        self.contents = load_catalog() if contents is None else contents
        if not self.contents:
            raise ValueError("Catalog must not be empty")
        self.model_name = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        self.revision = os.getenv("EMBEDDING_REVISION") or None
        self.encoder = encoder if encoder is not None else SentenceTransformer(
            self.model_name, revision=self.revision, device="cpu", trust_remote_code=False,
        )
        if encoder is None:
            self.revision = getattr(self.encoder[0].auto_model.config, "_commit_hash", self.revision)
        self.lock = threading.Lock()
        vectors = self._encode([c["description"] for c in self.contents])
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

    def _encode(self, texts):
        """Input: texts. Output: normalized float32 matrix; inner products then equal cosine similarity."""
        with self.lock:
            vectors = np.asarray(self.encoder.encode(
                texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False,
            ), dtype="float32")
        if vectors.ndim != 2 or len(vectors) != len(texts) or not np.isfinite(vectors).all():
            raise ValueError("Encoder must return a finite matrix")
        if np.any(np.linalg.norm(vectors, axis=1) == 0):
            raise ValueError("Encoder returned a zero vector")
        vectors = np.ascontiguousarray(vectors)
        faiss.normalize_L2(vectors)
        return vectors

    def search(self, query, top_k=10):
        """Input: nonempty query, K. Output: at most K catalog rows + cosine; bound work to catalog size."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be nonempty")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be positive")
        scores, positions = self.index.search(self._encode([query]), min(top_k, len(self.contents)))
        rows = [{**self.contents[int(i)], "similarity": float(np.clip(s, -1, 1))}
                for s, i in zip(scores[0], positions[0]) if i >= 0]
        return sorted(rows, key=lambda row: (-row["similarity"], row["content_id"]))
