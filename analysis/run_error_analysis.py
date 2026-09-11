"""Prediction-only MELD error analysis; does not load or modify model weights."""
import ast
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score


ROOT = Path("/workspace")
OUT = ROOT / "experiments" / "error_analysis"
KEYS = ["split", "Dialogue_ID", "Utterance_ID"]
EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
PREDICTION_FILES = {
    "text_only": ROOT / "experiments/robustness/clean/text_only/predictions.csv",
    "audio_only": ROOT / "experiments/robustness_audio_only_rerun/clean/predictions.csv",
    "audio_text": ROOT / "experiments/robustness/clean/audio_text/predictions.csv",
    "context_3": ROOT / "experiments/robustness/clean/context_best/predictions.csv",
}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def normalize(frame):
    result = frame.copy()
    if "split" not in result:
        result["split"] = "test"
    result["split"] = result["split"].fillna("test").astype(str)
    result["Dialogue_ID"] = result["Dialogue_ID"].astype(int)
    result["Utterance_ID"] = result["Utterance_ID"].astype(int)
    return result


def parse_context(value):
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = ast.literal_eval(value)
        return [str(item) for item in parsed] if isinstance(parsed, list) else []
    except (ValueError, SyntaxError):
        return []


def stratified_representatives(frame, limit=28):
    """Take candidates in true-label round-robin order, then fill to limit."""
    pieces = []
    for emotion in EMOTIONS:
        candidates = frame[frame["true_emotion"] == emotion]
        if not candidates.empty:
            pieces.append(candidates.head(4))
    selected = pd.concat(pieces, ignore_index=False).drop_duplicates(subset=KEYS) if pieces else frame.head(0)
    if len(selected) < min(limit, len(frame)):
        selected = pd.concat([selected, frame.drop(index=selected.index).head(limit - len(selected))])
    return selected.head(limit).copy()


def class_metrics(y_true, y_pred, model):
    report = classification_report(y_true, y_pred, labels=EMOTIONS, target_names=EMOTIONS, output_dict=True, zero_division=0)
    return pd.DataFrame(
        [{"emotion": emotion, "model": model, "precision": report[emotion]["precision"], "recall": report[emotion]["recall"], "f1": report[emotion]["f1-score"], "support": int(report[emotion]["support"])} for emotion in EMOTIONS]
    )


def transition_table(frame, old_col, new_col, old_name, new_name):
    result = frame[["true_emotion", old_col, new_col, f"{old_name}_correct", f"{new_name}_correct"]].copy()
    result = result[result[old_col] != result[new_col]].copy()
    result["transition_type"] = np.select(
        [~result[f"{old_name}_correct"] & result[f"{new_name}_correct"], result[f"{old_name}_correct"] & ~result[f"{new_name}_correct"], result[f"{old_name}_correct"] & result[f"{new_name}_correct"]],
        ["wrong_to_correct", "correct_to_wrong", "both_correct_prediction_changed"],
        default="both_wrong_prediction_changed",
    )
    return result.groupby(["true_emotion", old_col, new_col, "transition_type"], as_index=False).size().rename(columns={old_col: "old_prediction", new_col: "new_prediction", "size": "count"}).sort_values(["transition_type", "count"], ascending=[True, False])


def save_representatives(frame, name):
    columns = ["split", "Dialogue_ID", "Utterance_ID", "Speaker", "Utterance", "previous_context", "previous_turn_count", "previous_true_emotion", "emotion_transition", "true_emotion", "text_only_prediction", "audio_only_prediction", "audio_text_prediction", "context_3_prediction", "text_only_correct", "audio_only_correct", "audio_text_correct", "context_3_correct"]
    chosen = stratified_representatives(frame)
    chosen.insert(0, "candidate_group_size", len(frame))
    chosen.to_csv(OUT / name, index=False, columns=[column for column in columns if column in chosen] + ["candidate_group_size"])
    return len(frame), len(chosen)


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    frames = {name: normalize(pd.read_csv(path)) for name, path in PREDICTION_FILES.items()}
    source = normalize(pd.read_csv(ROOT / "emotion-ai/data/meld_metadata/test_sent_emo.csv"))
    source = source[~source.apply(lambda row: (row.Dialogue_ID, row.Utterance_ID) in {(38, 4), (220, 0)}, axis=1)].reset_index(drop=True)

    alignment = {"source_test_count": len(source), "models": {}, "status": "passed"}
    source_keys = set(map(tuple, source[KEYS].to_numpy()))
    source_truth = source.set_index(KEYS)["Emotion"]
    for name, frame in frames.items():
        keys = set(map(tuple, frame[KEYS].to_numpy()))
        duplicate_count = int(frame.duplicated(KEYS).sum())
        truth_ok = keys == source_keys and all(source_truth.loc[tuple(row[KEYS]) ] == row["Emotion"] for _, row in frame.iterrows())
        alignment["models"][name] = {"sample_count": len(frame), "duplicate_key_count": duplicate_count, "key_set_matches_source": keys == source_keys, "ground_truth_matches_source": bool(truth_ok)}
        if len(frame) != len(source) or duplicate_count or keys != source_keys or not truth_ok:
            alignment["status"] = "failed"
    write_json(OUT / "prediction_alignment.json", alignment)
    if alignment["status"] != "passed":
        raise RuntimeError(f"Prediction alignment failed: {alignment}")

    # The Context prediction export supplies the past-only context text; raw metadata
    # supplies canonical current utterance/speaker fields.
    metadata = source.merge(frames["context_3"][KEYS + ["context_turn_texts"]], on=KEYS, how="inner", validate="one_to_one")
    for name, frame in frames.items():
        metadata = metadata.merge(frame[KEYS + ["prediction"]].rename(columns={"prediction": f"{name}_prediction"}), on=KEYS, how="inner", validate="one_to_one")
    metadata = metadata.rename(columns={"Emotion": "true_emotion"})
    metadata["context_turns"] = metadata["context_turn_texts"].map(parse_context)
    metadata["previous_context"] = metadata["context_turns"].map(lambda turns: " | ".join(turns))
    metadata["previous_turn_count"] = metadata["context_turns"].map(len)
    metadata["current_word_count"] = metadata["Utterance"].astype(str).str.split().map(len)
    metadata["context_word_count"] = metadata["context_turns"].map(lambda turns: sum(len(turn.split()) for turn in turns))
    metadata["short_current_proxy"] = metadata["current_word_count"] <= 3

    # Previous true emotion derives only from strictly previous turns in the official test split.
    raw_full = normalize(pd.read_csv(ROOT / "emotion-ai/data/meld_metadata/test_sent_emo.csv"))
    previous = {}
    for dialogue, group in raw_full.groupby("Dialogue_ID", sort=False):
        past = []
        for row in group.sort_values("Utterance_ID", kind="stable").itertuples(index=False):
            previous[("test", dialogue, row.Utterance_ID)] = past[-1] if past else None
            past.append(row.Emotion)
    metadata["previous_true_emotion"] = [previous.get(tuple(row[KEYS])) for _, row in metadata.iterrows()]
    metadata["emotion_transition"] = np.where(metadata["previous_true_emotion"].isna(), "no_previous", np.where(metadata["previous_true_emotion"] == metadata["true_emotion"], "same", "different"))
    for name in frames:
        metadata[f"{name}_correct"] = metadata[f"{name}_prediction"] == metadata["true_emotion"]
    metadata.to_csv(OUT / "merged_predictions.csv", index=False)

    # Model-level and class-level summaries.
    overview = []
    classwise = []
    for name in frames:
        y_pred = metadata[f"{name}_prediction"]
        overview.append({"row_type": "model", "model": name, "count": len(metadata), "accuracy": accuracy_score(metadata.true_emotion, y_pred), "macro_f1": f1_score(metadata.true_emotion, y_pred, labels=EMOTIONS, average="macro", zero_division=0)})
        classwise.append(class_metrics(metadata.true_emotion, y_pred, name))
    comparisons = {"text_to_audio_text": ("text_only", "audio_text"), "audio_text_to_context_3": ("audio_text", "context_3")}
    for label, (old, new) in comparisons.items():
        for group, mask in {"both_correct": metadata[f"{old}_correct"] & metadata[f"{new}_correct"], "old_correct_new_wrong": metadata[f"{old}_correct"] & ~metadata[f"{new}_correct"], "old_wrong_new_correct": ~metadata[f"{old}_correct"] & metadata[f"{new}_correct"], "both_wrong": ~metadata[f"{old}_correct"] & ~metadata[f"{new}_correct"]}.items():
            overview.append({"row_type": "comparison_group", "model": label, "group": group, "count": int(mask.sum()), "test_fraction": float(mask.mean())})
    pd.DataFrame(overview).to_csv(OUT / "model_comparison_summary.csv", index=False)
    classwise = pd.concat(classwise, ignore_index=True)
    pivot = classwise.pivot(index="emotion", columns="model", values=["precision", "recall", "f1"]).reset_index()
    pivot.columns = ["_".join(column).rstrip("_") if isinstance(column, tuple) else column for column in pivot.columns]
    pivot["f1_text_to_audio_text"] = pivot["f1_audio_text"] - pivot["f1_text_only"]
    pivot["recall_text_to_audio_text"] = pivot["recall_audio_text"] - pivot["recall_text_only"]
    pivot["precision_text_to_audio_text"] = pivot["precision_audio_text"] - pivot["precision_text_only"]
    pivot["f1_audio_text_to_context_3"] = pivot["f1_context_3"] - pivot["f1_audio_text"]
    pivot["recall_audio_text_to_context_3"] = pivot["recall_context_3"] - pivot["recall_audio_text"]
    pivot["precision_audio_text_to_context_3"] = pivot["precision_context_3"] - pivot["precision_audio_text"]
    pivot.to_csv(OUT / "classwise_model_comparison.csv", index=False)

    audio_helped = metadata[~metadata.text_only_correct & metadata.audio_text_correct].copy()
    audio_hurt = metadata[metadata.text_only_correct & ~metadata.audio_text_correct].copy()
    context_helped = metadata[~metadata.audio_text_correct & metadata.context_3_correct].copy()
    context_hurt = metadata[metadata.audio_text_correct & ~metadata.context_3_correct].copy()
    all_wrong = metadata[~metadata.text_only_correct & ~metadata.audio_only_correct & ~metadata.audio_text_correct & ~metadata.context_3_correct].copy()
    candidate_counts = {}
    for frame, filename, label in [(audio_helped, "audio_helped.csv", "audio_helped"), (audio_hurt, "audio_hurt.csv", "audio_hurt"), (context_helped, "context_helped.csv", "context_helped"), (context_hurt, "context_hurt.csv", "context_hurt"), (all_wrong, "all_models_wrong.csv", "all_models_wrong")]:
        candidate_count, saved_count = save_representatives(frame, filename)
        candidate_counts[label] = {"candidate_count": candidate_count, "saved_representatives": saved_count}
    audio_likely_helped = audio_helped[audio_helped.audio_only_correct].copy()
    audio_likely_hurt = audio_hurt[~audio_hurt.audio_only_correct].copy()
    save_representatives(audio_likely_helped, "audio_likely_helped.csv")
    save_representatives(audio_likely_hurt, "audio_likely_hurt.csv")

    text_fusion_transitions = transition_table(metadata, "text_only_prediction", "audio_text_prediction", "text_only", "audio_text")
    fusion_context_transitions = transition_table(metadata, "audio_text_prediction", "context_3_prediction", "audio_text", "context_3")
    text_fusion_transitions.to_csv(OUT / "text_to_fusion_transitions.csv", index=False)
    fusion_context_transitions.to_csv(OUT / "fusion_to_context_transitions.csv", index=False)

    # Context characteristics: group-level comparisons based entirely on prior test turns.
    context_rows = []
    for name, group in [("context_helped", context_helped), ("context_hurt", context_hurt), ("both_correct", metadata[metadata.audio_text_correct & metadata.context_3_correct]), ("both_wrong", metadata[~metadata.audio_text_correct & ~metadata.context_3_correct])]:
        for feature in ["current_word_count", "context_word_count", "previous_turn_count", "Utterance_ID"]:
            context_rows.append({"group": name, "metric": feature, "count": len(group), "mean": float(group[feature].mean()) if len(group) else None, "median": float(group[feature].median()) if len(group) else None})
        for value, count in group["emotion_transition"].value_counts(dropna=False).items():
            context_rows.append({"group": name, "metric": f"emotion_transition:{value}", "count": int(count), "mean": None, "median": None})
        context_rows.append({"group": name, "metric": "short_current_proxy_rate", "count": len(group), "mean": float(group.short_current_proxy.mean()) if len(group) else None, "median": None})
    pd.DataFrame(context_rows).to_csv(OUT / "context_characteristics.csv", index=False)

    # Basic diagnostic plots.
    def grouped_plot(frame, columns, title, path):
        plot = frame.set_index("emotion")[columns]
        ax = plot.plot(kind="bar", figsize=(10, 5), rot=0)
        ax.set_ylim(0, 1); ax.set_ylabel("F1"); ax.set_title(title); ax.legend(title="Model")
        plt.tight_layout(); plt.savefig(path, dpi=160); plt.close()
    grouped_plot(pivot, ["f1_text_only", "f1_audio_text"], "Class-wise F1: Text-only vs Audio+Text", OUT / "text_vs_fusion_class_f1.png")
    grouped_plot(pivot, ["f1_audio_text", "f1_context_3"], "Class-wise F1: Audio+Text vs Context-3", OUT / "fusion_vs_context_class_f1.png")
    def distribution_plot(helped, hurt, title, path):
        values = pd.DataFrame({"helped": helped.true_emotion.value_counts(), "hurt": hurt.true_emotion.value_counts()}).reindex(EMOTIONS).fillna(0)
        ax = values.plot(kind="bar", figsize=(10, 5), rot=0); ax.set_ylabel("Candidate count"); ax.set_title(title); ax.legend(title="Group")
        plt.tight_layout(); plt.savefig(path, dpi=160); plt.close()
    distribution_plot(audio_helped, audio_hurt, "Emotion Distribution: Audio Helped vs Hurt", OUT / "audio_helped_vs_hurt_emotions.png")
    distribution_plot(context_helped, context_hurt, "Emotion Distribution: Context Helped vs Hurt", OUT / "context_helped_vs_hurt_emotions.png")

    robust = pd.read_csv(ROOT / "experiments/robustness/robustness_summary_corrected_audio_only.csv")
    robust_rows = robust[((robust.corruption_type == "text_word_dropout") & (robust.severity.astype(str) == "0.3")) | ((robust.corruption_type == "audio_noise") & robust.severity.astype(str).isin(["20dB", "10dB"]))].to_dict(orient="records")
    f1_improvement = pivot.sort_values("f1_text_to_audio_text", ascending=False).iloc[0]
    f1_decline = pivot.sort_values("f1_text_to_audio_text").iloc[0]
    context_improvement = pivot.sort_values("f1_audio_text_to_context_3", ascending=False).iloc[0]
    context_decline = pivot.sort_values("f1_audio_text_to_context_3").iloc[0]
    summary = {"alignment": alignment, "candidate_counts": candidate_counts, "audio_signal_candidate_counts": {"text_wrong_audio_correct_fusion_correct": int(len(audio_likely_helped)), "text_correct_audio_wrong_fusion_wrong": int(len(audio_likely_hurt))}, "largest_classwise_changes": {"text_to_fusion_f1_improvement": {"emotion": f1_improvement.emotion, "difference": float(f1_improvement.f1_text_to_audio_text)}, "text_to_fusion_f1_decline": {"emotion": f1_decline.emotion, "difference": float(f1_decline.f1_text_to_audio_text)}, "fusion_to_context_f1_improvement": {"emotion": context_improvement.emotion, "difference": float(context_improvement.f1_audio_text_to_context_3)}, "fusion_to_context_f1_decline": {"emotion": context_decline.emotion, "difference": float(context_decline.f1_audio_text_to_context_3)}}, "top_transitions": {"text_to_fusion": text_fusion_transitions.head(10).to_dict(orient="records"), "fusion_to_context": fusion_context_transitions.head(10).to_dict(orient="records")}, "robustness_rows": robust_rows, "confirmed_observations": ["All model comparisons use an identical, key-aligned 2,608-sample test set.", "Text dropout results and audio-noise results are reported from the corrected robustness summary; they do not establish causal modality use."], "hypotheses_not_conclusions": ["Wrong-to-correct cases are candidates for complementary information, not proof that a particular modality caused the correction.", "Near-zero F1 change under audio noise is compatible with relatively stronger text/context reliance, but predictions alone cannot establish attention or causal reliance."]}
    write_json(OUT / "error_analysis_summary.json", summary)


if __name__ == "__main__":
    main()
