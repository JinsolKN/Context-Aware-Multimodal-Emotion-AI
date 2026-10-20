"""Inspectable heuristic ranking; scores are not probabilities or learned preferences."""
import math

from demo.ranking import CATEGORIES, validate_probabilities

WEIGHTS = {"similarity": 0.5, "preference": 0.2, "emotion": 0.3}


def rank_candidates(candidates, probabilities, preferences=(), top_k=3, use_emotion=True):
    """Input: retrieved rows, seven probabilities, categories, K, ablation flag.

    Output: sorted rows with component scores. Keep candidates fixed for ablation;
    remove emotion and renormalize the remaining weights to retain a [0,1] scale.
    """
    validate_probabilities(probabilities)
    if not set(preferences) <= set(CATEGORIES):
        raise ValueError("Unknown category")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be positive")
    weights = dict(WEIGHTS)
    if not use_emotion:
        weights["emotion"] = 0.0
        total = sum(weights.values())
        weights = {key: value / total for key, value in weights.items()}
    results = []
    for row in candidates:
        cosine = row["similarity"]
        if not math.isfinite(cosine) or not -1 <= cosine <= 1:
            raise ValueError("Invalid cosine similarity")
        components = {
            "similarity": (cosine + 1) / 2,
            "preference": float(row["category"] in preferences),
            "emotion": sum(probabilities[tag] for tag in set(row["emotion_tags"])) if use_emotion else 0.0,
        }
        results.append({**row, "components": components, "weights": weights,
                        "score": sum(weights[key] * value for key, value in components.items())})
    return sorted(results, key=lambda row: (-row["score"], row["content_id"]))[:top_k]
