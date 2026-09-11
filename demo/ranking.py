"""Explainable synthetic-content ranking; no trained recommendation model."""
import csv
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LABELS = ("anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise")
DEFAULT_CATALOG = ROOT / "data/demo_contents.csv"
CATEGORIES = ("music", "comedy", "motivation", "relaxation", "drama")
# Manually selected demo heuristics, NOT learned or optimized weights.
WEIGHTS = {"emotion_match": 0.5, "preference_match": 0.3, "context_match": 0.2}


def validate_probabilities(probabilities):
    if not isinstance(probabilities, dict) or set(probabilities) != set(LABELS):
        raise ValueError("Expected the project's seven emotion labels.")
    if any(isinstance(p, bool) or not isinstance(p, (int, float)) or
           not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities.values()):
        raise ValueError("Probabilities must be finite numbers between zero and one.")
    if not math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-5):
        raise ValueError("Emotion probabilities must sum to one.")


def load_catalog(path=DEFAULT_CATALOG):
    with Path(path).open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError("Content catalog is empty.")
    seen = set()
    for row in rows:
        required = ("content_id", "title", "category", "emotion_tags", "energy", "description", "synthetic")
        if any(not row.get(key) for key in required):
            raise ValueError("Content metadata has missing fields.")
        row["emotion_tags"] = row["emotion_tags"].split("|")
        if (row["content_id"] in seen or row["category"] not in CATEGORIES or
                row["energy"] not in ("low", "medium", "high") or
                not set(row["emotion_tags"]) <= set(LABELS) or row["synthetic"] != "true"):
            raise ValueError("Invalid synthetic content metadata.")
        seen.add(row["content_id"])
    return rows


def rank_contents(emotion_probabilities, user_preferences, contents, top_k=3, desired_energy=None):
    """Use probability mass on tags, category membership, and desired energy.

    No preferences/energy selected contributes zero, without reweighting.
    Context here is an explicit listening/viewing energy choice; dialogue context
    is already consumed by the emotion model. Scores are not probabilities.
    """
    validate_probabilities(emotion_probabilities)
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer.")
    preferences = set(user_preferences)
    if not preferences <= set(CATEGORIES) or desired_energy not in (None, "low", "medium", "high"):
        raise ValueError("Unknown preference or energy.")
    results = []
    for content in contents:
        tags = sorted(set(content["emotion_tags"]), key=lambda tag: (-emotion_probabilities[tag], tag))
        components = {
            "emotion_match": sum(emotion_probabilities[tag] for tag in tags),
            "preference_match": float(content["category"] in preferences),
            "context_match": float(desired_energy is not None and content["energy"] == desired_energy),
        }
        reason = "Emotion tag mass: " + ", ".join(f"{tag} {emotion_probabilities[tag]:.1%}" for tag in tags)
        reason += f"; {content['category']} " + ("matches preference" if components["preference_match"] else "has no preference bonus")
        reason += f"; {content['energy']} energy " + ("matches requested energy" if components["context_match"] else "has no energy bonus")
        results.append({**content, "score": sum(WEIGHTS[key] * value for key, value in components.items()),
                        "components": components, "reason": reason})
    return sorted(results, key=lambda item: (-item["score"], item["content_id"]))[:top_k]
