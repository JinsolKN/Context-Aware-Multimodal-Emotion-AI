"""Reproduce MELD baselines and evaluate input robustness without changing source data.

This script deliberately lives outside the historical experiment runners.  It uses
their recorded full-run configurations and writes only under experiments/robustness.
"""
import copy
import csv
import json
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from transformers import AutoFeatureExtractor, AutoModel, AutoTokenizer, WavLMModel


WORKSPACE = Path("/workspace")
PROJECT = WORKSPACE / "emotion-ai"
OUTPUT = WORKSPACE / "experiments" / "robustness"
CHECKPOINTS = OUTPUT / "checkpoints"
EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
LABEL2ID = {label: index for index, label in enumerate(EMOTIONS)}
ID2LABEL = {index: label for label, index in LABEL2ID.items()}
BAD_TEST_KEYS = {(38, 4), (220, 0)}
SEED = 42
TEXT_MODEL = "roberta-base"
AUDIO_MODEL = "microsoft/wavlm-base-plus"
SAMPLE_RATE = 16000


def configure_logging():
    # A controlled resume reuses only this task's interrupted directory; it never
    # touches a historical experiment directory.
    OUTPUT.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(OUTPUT / "execution.log", encoding="utf-8"), logging.StreamHandler()],
    )


def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def add_noise_at_snr(waveform, snr_db, generator):
    """Return a noisy copy; the caller's waveform and on-disk audio remain unchanged."""
    signal = torch.as_tensor(waveform, dtype=torch.float32).clone()
    rms = torch.sqrt(torch.mean(signal.square()))
    if not torch.isfinite(rms) or rms.item() == 0.0:
        return signal.numpy()
    target_noise_rms = rms / (10.0 ** (float(snr_db) / 20.0))
    noise = torch.randn(signal.shape, generator=generator, dtype=signal.dtype)
    noise = noise * (target_noise_rms / torch.sqrt(torch.mean(noise.square())).clamp_min(1e-12))
    return (signal + noise).numpy()


def word_dropout(text, probability, seed):
    """Deterministic raw-word deletion before tokenization; preserves at least one word."""
    words = str(text).split()
    if not words or probability <= 0:
        return str(text)
    rng = random.Random(seed)
    retained = [word for word in words if rng.random() >= probability]
    if not retained:
        retained = [words[rng.randrange(len(words))]]
    return " ".join(retained)


def audio_path(split, dialogue_id, utterance_id):
    return PROJECT / "data" / "meld_audio" / "extracted" / split / f"dia{dialogue_id}_utt{utterance_id}.flac"


def prepare_text_splits():
    directory = PROJECT / "data" / "meld_metadata"
    train = pd.read_csv(directory / "train_sent_emo.csv")
    dev = pd.read_csv(directory / "dev_sent_emo.csv")
    test = pd.read_csv(directory / "test_sent_emo.csv")
    test = test[~test.apply(lambda row: (row.Dialogue_ID, row.Utterance_ID) in BAD_TEST_KEYS, axis=1)].reset_index(drop=True)
    return {"train": train, "dev": dev, "test": test}


def prepare_audio_splits(context_size=None):
    text_dir = PROJECT / "data" / "meld_metadata"
    audio_dir = PROJECT / "data" / "meld_audio"
    frames = {}
    for split in ["train", "dev", "test"]:
        text_frame = pd.read_csv(text_dir / f"{split}_sent_emo.csv")
        audio_frame = pd.read_csv(audio_dir / f"{split}.csv")
        keys = ["Dialogue_ID", "Utterance_ID"]
        paired = text_frame.merge(audio_frame[keys + ["Emotion"]], on=keys, how="outer", suffixes=("", "_audio"), indicator=True, validate="one_to_one")
        if not paired["_merge"].eq("both").all() or not paired["Emotion"].eq(paired["Emotion_audio"]).all():
            raise ValueError(f"{split}: text/audio metadata pairing differs from historical run")
        frame = text_frame.copy()
        frame["split"] = split
        frame["audio_path"] = [str(audio_path(split, d, u)) for d, u in frame[keys].itertuples(index=False, name=None)]
        exists = frame["audio_path"].map(lambda p: Path(p).is_file())
        bad = pd.Series([split == "test" and (d, u) in BAD_TEST_KEYS for d, u in frame[keys].itertuples(index=False, name=None)], index=frame.index)
        if split == "test" and (~exists & ~bad).any():
            raise FileNotFoundError("Unexpected missing audio in common test split")
        if context_size:
            # Historical context construction precedes audio filtering: a target may
            # use an earlier official text turn whose own audio was unavailable.
            frame = add_dialogue_context(frame, context_size)
        frame = frame[exists & ~bad].reset_index(drop=True)
        frames[split] = frame
    if len(frames["test"]) != 2608:
        raise ValueError(f"Expected common multimodal test count 2608, got {len(frames['test'])}")
    return frames


def add_dialogue_context(frame, context_size):
    """Past-only context from the same official split; target's past stays clean."""
    result = frame.copy()
    lookup = {}
    for dialogue_id, group in frame.groupby("Dialogue_ID", sort=False):
        past = []
        for row in group.sort_values("Utterance_ID", kind="stable").itertuples(index=False):
            previous = past[-context_size:]
            if not all(item[0] < row.Utterance_ID for item in previous):
                raise AssertionError("Future/current context leak")
            lookup[(dialogue_id, row.Utterance_ID)] = [item[1] for item in previous]
            past.append((row.Utterance_ID, f"{row.Speaker}: {row.Utterance}"))
    result["context_turn_texts"] = [lookup[(d, u)] for d, u in result[["Dialogue_ID", "Utterance_ID"]].itertuples(index=False, name=None)]
    return result


class TextDataset(Dataset):
    def __init__(self, frame, tokenizer, max_length, text_dropout=0.0):
        self.frame, self.tokenizer, self.max_length, self.text_dropout = frame.reset_index(drop=True), tokenizer, max_length, text_dropout

    def __len__(self): return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        text = word_dropout(row.Utterance, self.text_dropout, SEED + int(row.Dialogue_ID) * 10000 + int(row.Utterance_ID))
        encoded = self.tokenizer(text, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        return {"input_ids": encoded.input_ids.squeeze(0), "attention_mask": encoded.attention_mask.squeeze(0), "labels": torch.tensor(LABEL2ID[row.Emotion]), "index": index}


class TextClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(TEXT_MODEL)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, 7)

    def forward(self, input_ids, attention_mask):
        return self.classifier(self.dropout(self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]))


class MultimodalDataset(Dataset):
    def __init__(self, frame, tokenizer, max_length=128, context_size=0, text_dropout=0.0, snr_db=None):
        self.frame, self.tokenizer = frame.reset_index(drop=True), tokenizer
        self.max_length, self.context_size, self.text_dropout, self.snr_db = max_length, context_size, text_dropout, snr_db

    def __len__(self): return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        waveform, rate = sf.read(row.audio_path, dtype="float32")
        if waveform.ndim == 2: waveform = waveform.mean(axis=1)
        if rate != SAMPLE_RATE or len(waveform) < 400 or not np.isfinite(waveform).all(): raise ValueError(f"Invalid waveform: {row.audio_path}")
        if self.snr_db is not None:
            noise_seed = SEED + 1_000_000 + int(row.Dialogue_ID) * 10000 + int(row.Utterance_ID) + int(self.snr_db * 100)
            waveform = add_noise_at_snr(waveform, self.snr_db, torch.Generator().manual_seed(noise_seed))
        current = word_dropout(row.Utterance, self.text_dropout, SEED + int(row.Dialogue_ID) * 10000 + int(row.Utterance_ID))
        text = self.tokenizer(current, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        item = {"input_ids": text.input_ids.squeeze(0), "text_attention_mask": text.attention_mask.squeeze(0), "waveform": waveform, "labels": torch.tensor(LABEL2ID[row.Emotion]), "index": index}
        if self.context_size:
            # Context turns are intentionally not corrupted for current-utterance robustness.
            context = f" {self.tokenizer.sep_token} ".join(row.context_turn_texts)
            encoded_context = self.tokenizer(context, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
            item["context_input_ids"] = encoded_context.input_ids.squeeze(0)
            item["context_attention_mask"] = encoded_context.attention_mask.squeeze(0)
        return item


class MultimodalCollator:
    def __init__(self, processor): self.processor = processor
    def __call__(self, batch):
        audio = self.processor([item["waveform"] for item in batch], sampling_rate=SAMPLE_RATE, padding=True, return_attention_mask=True, return_tensors="pt")
        result = {"input_ids": torch.stack([x["input_ids"] for x in batch]), "text_attention_mask": torch.stack([x["text_attention_mask"] for x in batch]), "input_values": audio.input_values, "audio_attention_mask": audio.attention_mask, "labels": torch.stack([x["labels"] for x in batch]), "index": torch.tensor([x["index"] for x in batch])}
        if "context_input_ids" in batch[0]:
            result["context_input_ids"] = torch.stack([x["context_input_ids"] for x in batch])
            result["context_attention_mask"] = torch.stack([x["context_attention_mask"] for x in batch])
        return result


class AudioClassifier(nn.Module):
    def __init__(self):
        super().__init__(); self.encoder = WavLMModel.from_pretrained(AUDIO_MODEL); self.dropout = nn.Dropout(0.1); self.classifier = nn.Linear(self.encoder.config.hidden_size, 7)
    def forward(self, input_values, audio_attention_mask, **_unused_inputs):
        # The shared multimodal collator also carries raw text tensors, but the
        # historical audio-only architecture consumes only WavLM inputs.
        hidden = self.encoder(input_values=input_values, attention_mask=audio_attention_mask).last_hidden_state
        mask = self.encoder._get_feature_vector_attention_mask(hidden.shape[1], audio_attention_mask).unsqueeze(-1).to(hidden.dtype)
        return self.classifier(self.dropout((hidden * mask).sum(1) / mask.sum(1).clamp(min=1)))


class FusionClassifier(nn.Module):
    def __init__(self, with_context=False):
        super().__init__(); self.with_context = with_context
        self.text_encoder = AutoModel.from_pretrained(TEXT_MODEL)
        self.audio_encoder = WavLMModel.from_pretrained(AUDIO_MODEL)
        width = self.text_encoder.config.hidden_size + self.audio_encoder.config.hidden_size + (self.text_encoder.config.hidden_size if with_context else 0)
        self.dropout = nn.Dropout(0.1); self.classifier = nn.Linear(width, 7)
    def forward(self, input_ids, text_attention_mask, input_values, audio_attention_mask, context_input_ids=None, context_attention_mask=None):
        text = self.text_encoder(input_ids=input_ids, attention_mask=text_attention_mask).last_hidden_state[:, 0, :]
        hidden = self.audio_encoder(input_values=input_values, attention_mask=audio_attention_mask).last_hidden_state
        mask = self.audio_encoder._get_feature_vector_attention_mask(hidden.shape[1], audio_attention_mask).unsqueeze(-1).to(hidden.dtype)
        audio = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        features = [text, audio]
        if self.with_context:
            features.append(self.text_encoder(input_ids=context_input_ids, attention_mask=context_attention_mask).last_hidden_state[:, 0, :])
        return self.classifier(self.dropout(torch.cat(features, dim=-1)))


def configure_model(model, name):
    if name != "text_only":
        for parameter in model.encoder.feature_extractor.parameters() if name == "audio_only" else model.audio_encoder.feature_extractor.parameters(): parameter.requires_grad = False
        audio_encoder = model.encoder if name == "audio_only" else model.audio_encoder
        audio_encoder.config.apply_spec_augment = False
        audio_encoder.gradient_checkpointing_enable()
        if name != "audio_only": model.text_encoder.gradient_checkpointing_enable()
    return model


def move_inputs(batch, device):
    return {key: value.to(device) for key, value in batch.items() if key not in {"labels", "index"}}


def smoke_test(factory, loader, criterion, optimizer_factory, device, name):
    set_seed(); model = configure_model(factory(), name).to(device); optimizer = optimizer_factory(model)
    model.train(); records = []
    for step, batch in enumerate(loader):
        if step == 2: break
        optimizer.zero_grad(set_to_none=True); labels = batch["labels"].to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=name != "text_only" and device.type == "cuda"):
            loss = criterion(model(**move_inputs(batch, device)), labels)
        loss.backward()
        if name == "text_only": torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step(); records.append(float(loss.item()))
    del model, optimizer
    torch.cuda.empty_cache()
    return {"status": "passed", "batches": len(records), "losses": records, "forward_loss_backward_optimizer_step": True}


def evaluate(model, loader, criterion, device):
    model.eval(); weighted_loss, weight_sum, truths, preds = 0.0, 0.0, [], []
    with torch.no_grad():
        for batch in loader:
            labels = batch["labels"].to(device); logits = model(**move_inputs(batch, device)); loss = criterion(logits, labels)
            weights = criterion.weight[labels].sum().item() if criterion.weight is not None else float(len(labels))
            weighted_loss += loss.item() * weights; weight_sum += weights
            truths.extend(labels.cpu().tolist()); preds.extend(logits.argmax(1).cpu().tolist())
    return weighted_loss / weight_sum, float(np.mean(np.array(truths) == np.array(preds))), float(f1_score(truths, preds, labels=range(7), average="macro", zero_division=0)), truths, preds


def write_evaluation(folder, frame, model, loader, criterion, device, prediction_column, checkpoint, metadata):
    folder.mkdir(parents=True, exist_ok=True)
    loss, accuracy, macro_f1, truth, pred = evaluate(model, loader, criterion, device)
    report = classification_report(truth, pred, labels=range(7), target_names=EMOTIONS, output_dict=True, zero_division=0)
    matrix = confusion_matrix(truth, pred, labels=range(7))
    metrics = {"loss": loss, "accuracy": accuracy, "macro_f1": macro_f1, "sample_count": len(truth), "checkpoint": str(checkpoint), "classification_report": report, **metadata}
    save_json(folder / "metrics.json", metrics)
    pd.DataFrame(report).transpose().to_csv(folder / "classification_report.csv")
    pd.DataFrame(matrix, index=EMOTIONS, columns=EMOTIONS).to_csv(folder / "confusion_matrix.csv")
    predictions = frame.copy()
    predictions["true_id"] = truth; predictions["prediction_id"] = pred; predictions["prediction"] = [ID2LABEL[x] for x in pred]; predictions["model_prediction_column"] = prediction_column
    predictions.to_csv(folder / "predictions.csv", index=False)
    return metrics


def model_config(name):
    common = {"text_model": TEXT_MODEL, "audio_model": AUDIO_MODEL, "max_text_length": 128, "sample_rate": 16000, "dropout": 0.1, "batch_size": 2, "accumulation_steps": 4, "learning_rate": 2e-5, "weight_decay": 0.01, "betas": [0.9, 0.999], "eps": 1e-8, "epochs": 3, "seed": SEED, "checkpoint_metric": "dev_macro_f1", "freeze_audio_feature_extractor": True, "apply_spec_augment": False, "gradient_checkpointing": True, "label2id": LABEL2ID, "id2label": ID2LABEL}
    if name == "text_only": return {"model": TEXT_MODEL, "max_length": 128, "batch_size": 16, "epochs": 3, "learning_rate": 2e-5, "seed": SEED, "loss": "CrossEntropyLoss (unweighted)", "optimizer": "AdamW defaults (weight_decay=0.01, betas=[0.9,0.999], eps=1e-8)", "label2id": LABEL2ID, "id2label": ID2LABEL}
    common["loss"] = "train_balanced_cross_entropy"; common["optimizer"] = "AdamW"; common["context_size"] = 3 if name == "context_best" else 0
    return common


def main():
    configure_logging(); set_seed(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda": raise RuntimeError("CUDA is required to reproduce these full WavLM runs in this execution")
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL); processor = AutoFeatureExtractor.from_pretrained(AUDIO_MODEL)
    text_frames, audio_frames, context_frames = prepare_text_splits(), prepare_audio_splits(), prepare_audio_splits(context_size=3)
    context_note = {"selected_context_size": 3, "selection_reason": "highest historical full-run dev Macro F1: context-3=0.46544165994155584; context-5=0.46510982397357375; context-1=0.4539381259805884. Test scores were not used."}
    historical_test_f1 = {"text_only": 0.4697484400963926, "audio_only": 0.2927321531627027, "audio_text": 0.4570728209481464, "context_best": 0.4477547906495752}
    run_config = {"source_configs": {"text_only": "experiments/text_only/20260908T024432_853390Z/config.json", "audio_only": "experiments/audio_only/20260908T025037_408302Z/config.json", "audio_text": "experiments/audio_text_fusion/20260908T032151_424758Z_full/config.json", "context": "emotion-ai/experiments/audio_text_dialogue_context/fusion_context_3/20260909T005523_889264Z_full/config.json"}, "context_selection": context_note, "corruptions": {"audio_noise": {"snr_db": [20, 10], "seed": SEED, "raw_audio_modified": False}, "text_word_dropout": {"probability": [0.1, 0.3], "seed": SEED, "raw_csv_modified": False, "context_policy": "current utterance only; past context clean"}}, "models": {}}
    summaries = []
    specs = [
        ("text_only", text_frames, lambda: TextClassifier(), 16, 0, False),
        ("audio_only", audio_frames, lambda: AudioClassifier(), 2, 0, True),
        ("audio_text", audio_frames, lambda: FusionClassifier(False), 2, 0, True),
        ("context_best", context_frames, lambda: FusionClassifier(True), 2, 3, True),
    ]
    for name, frames, factory, batch_size, context_size, weighted in specs:
        logging.info("Starting %s smoke test and full retraining", name)
        config = model_config(name)
        train_dir = OUTPUT / "training" / name; train_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = CHECKPOINTS / f"{name}_best.pt"; CHECKPOINTS.mkdir(parents=True, exist_ok=True)
        if name == "text_only":
            train_dataset = TextDataset(frames["train"], tokenizer, 128)
            dev_dataset = TextDataset(frames["dev"], tokenizer, 128)
            test_builder = lambda text_dropout=0.0, snr=None: TextDataset(frames["test"], tokenizer, 128, text_dropout)
            collator = None; criterion_weight = None
        else:
            train_dataset = MultimodalDataset(frames["train"], tokenizer, context_size=context_size)
            dev_dataset = MultimodalDataset(frames["dev"], tokenizer, context_size=context_size)
            test_builder = lambda text_dropout=0.0, snr=None: MultimodalDataset(frames["test"], tokenizer, context_size=context_size, text_dropout=text_dropout, snr_db=snr)
            collator = MultimodalCollator(processor); criterion_weight = compute_class_weight("balanced", classes=np.arange(7), y=frames["train"].Emotion.map(LABEL2ID).to_numpy()) if weighted else None
        criterion = nn.CrossEntropyLoss(weight=None if criterion_weight is None else torch.tensor(criterion_weight, dtype=torch.float32, device=device))
        if criterion_weight is not None: save_json(train_dir / "class_weights.json", dict(zip(EMOTIONS, criterion_weight.tolist())))
        optimizer_factory = lambda model: torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config["learning_rate"], **({"weight_decay": config["weight_decay"], "betas": tuple(config["betas"]), "eps": config["eps"]} if name != "text_only" else {}))
        resume_checkpoint = checkpoint_path.is_file() and (train_dir / "history.json").is_file()
        if resume_checkpoint:
            logging.info("Resuming %s from its already completed dev-best checkpoint", name)
            model = configure_model(factory(), name).to(device)
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
            model.load_state_dict(checkpoint["model_state_dict"])
            best_epoch, best_f1 = checkpoint["epoch"], checkpoint["best_dev_macro_f1"]
        else:
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)
            smoke = smoke_test(factory, train_loader, criterion, optimizer_factory, device, name); save_json(train_dir / "smoke_test.json", smoke)
            set_seed(); model = configure_model(factory(), name).to(device); optimizer = optimizer_factory(model)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)
            dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
            history, best_f1, best_epoch = [], -1.0, None
            for epoch in range(1, 4):
                model.train(); optimizer.zero_grad(set_to_none=True); total, count = 0.0, 0; batches = len(train_loader)
                for step, batch in enumerate(train_loader):
                    labels = batch["labels"].to(device)
                    group_start = (step // config.get("accumulation_steps", 1)) * config.get("accumulation_steps", 1); group_size = min(config.get("accumulation_steps", 1), batches - group_start)
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=name != "text_only"): loss = criterion(model(**move_inputs(batch, device)), labels)
                    denominator = 4 if name == "audio_only" else group_size
                    (loss / denominator).backward()
                    if name == "text_only": torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    if (step + 1) % config.get("accumulation_steps", 1) == 0 or step + 1 == batches: optimizer.step(); optimizer.zero_grad(set_to_none=True)
                    weight = criterion.weight[labels].sum().item() if criterion.weight is not None else len(labels); total += loss.item() * weight; count += weight
                dev_loss, dev_accuracy, dev_f1, _, _ = evaluate(model, dev_loader, criterion, device)
                record = {"epoch": epoch, "train_loss": total / count, "dev_loss": dev_loss, "dev_accuracy": dev_accuracy, "dev_macro_f1": dev_f1, "learning_rate": optimizer.param_groups[0]["lr"]}; history.append(record); save_json(train_dir / "history.json", history); pd.DataFrame(history).to_csv(train_dir / "history.csv", index=False); logging.info("%s epoch=%s train_loss=%.6f dev_loss=%.6f dev_acc=%.6f dev_f1=%.6f", name, epoch, record["train_loss"], dev_loss, dev_accuracy, dev_f1)
                if dev_f1 > best_f1:
                    best_f1, best_epoch = dev_f1, epoch
                    torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()), "epoch": epoch, "best_dev_macro_f1": dev_f1, "config": config, "label2id": LABEL2ID, "id2label": ID2LABEL}, checkpoint_path)
            model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True)["model_state_dict"])
        run_config["models"][name] = {"config": config, "best_epoch": best_epoch, "best_dev_macro_f1": best_f1, "checkpoint": str(checkpoint_path), "split_sizes": {key: len(value) for key, value in frames.items()}}
        conditions = [("clean", "clean", 0.0, None, OUTPUT / "clean" / name)]
        if name in {"text_only", "audio_text", "context_best"}: conditions += [("text_word_dropout", "0.1", 0.1, None, OUTPUT / "text_corruption" / name / "0.1"), ("text_word_dropout", "0.3", 0.3, None, OUTPUT / "text_corruption" / name / "0.3")]
        if name in {"audio_only", "audio_text", "context_best"}: conditions += [("audio_noise", "20dB", 0.0, 20, OUTPUT / "audio_noise" / name / "20dB"), ("audio_noise", "10dB", 0.0, 10, OUTPUT / "audio_noise" / name / "10dB")]
        clean_f1 = None
        for corruption, severity, drop, snr, folder in conditions:
            loader = DataLoader(test_builder(drop, snr), batch_size=batch_size, shuffle=False, collate_fn=collator)
            metadata = {"model": name, "corruption_type": corruption, "severity": severity, "text_corruption_scope": "current utterance only; context remains clean" if context_size else "current utterance only", "audio_source_modified": False, "csv_source_modified": False}
            metrics = write_evaluation(folder, frames["test"], model, loader, criterion, device, "prediction", checkpoint_path, metadata)
            if corruption == "clean":
                clean_f1 = metrics["macro_f1"]
                difference = clean_f1 - historical_test_f1[name]
                reproduction_check = {"historical_test_macro_f1": historical_test_f1[name], "retrained_clean_test_macro_f1": clean_f1, "absolute_difference": difference, "large_difference_threshold": 0.10, "requires_attention": abs(difference) > 0.10, "checked_settings": {"seed": SEED, "label_mapping": LABEL2ID, "test_exclusions": [[38, 4], [220, 0]], "preprocessing": "historical tokenizer/audio processor, 128 text tokens, 16kHz mono mean", "checkpoint_selection": "dev Macro F1 only", "corruption_not_used_for_training": True}}
                save_json(train_dir / "clean_reproduction_check.json", reproduction_check)
                if reproduction_check["requires_attention"]: logging.warning("%s clean F1 differs from historical result by %.6f; recorded reproducibility audit before corruption evaluation", name, difference)
            summaries.append({"model": name, "corruption_type": corruption, "severity": severity, "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"], "clean_macro_f1": clean_f1 if clean_f1 is not None else metrics["macro_f1"], "absolute_drop": 0.0 if corruption == "clean" else clean_f1 - metrics["macro_f1"], "relative_drop": 0.0 if corruption == "clean" else (clean_f1 - metrics["macro_f1"]) / clean_f1})
        del model; torch.cuda.empty_cache()
    pd.DataFrame(summaries).to_csv(OUTPUT / "robustness_summary.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    save_json(OUTPUT / "run_config.json", run_config)
    (OUTPUT / "README.md").write_text(
        "# MELD robustness evaluation\n\n"
        "All results use one dev-Macro-F1-selected checkpoint per model. No source CSV or audio file is changed. "
        "Audio noise is an in-memory Gaussian perturbation at 20 dB or 10 dB SNR. "
        "Word dropout is applied to raw current-utterance text before tokenization at p=0.1 or p=0.3. "
        "For the selected Context-3 model, past context turns remain clean and only the current utterance is corrupted.\n\n"
        "Context-3 was selected exclusively by historical dev Macro F1 (0.465442), the highest of Context-1/3/5; historical test values were not used for model selection.\n",
        encoding="utf-8",
    )
    logging.info("Completed all robustness experiments: %s", OUTPUT)


if __name__ == "__main__": main()
