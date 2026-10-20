"""Ranking metrics used by offline contract tests."""
import math


def metrics(ids, relevance, k):
    """Input: ranked IDs, manual graded judgments, K. Output: NDCG/Recall/MRR; measure ranking quality."""
    grades = [relevance.get(item, 0) for item in ids[:k]]
    dcg = sum((2 ** grade - 1) / math.log2(i + 2) for i, grade in enumerate(grades))
    ideal = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum((2 ** grade - 1) / math.log2(i + 2) for i, grade in enumerate(ideal))
    relevant = {key for key, value in relevance.items() if value > 0}
    return {"ndcg": dcg / idcg if idcg else 0.0,
            "recall": len(set(ids[:k]) & relevant) / len(relevant) if relevant else 0.0,
            "mrr": next((1 / (i + 1) for i, grade in enumerate(grades) if grade > 0), 0.0)}
