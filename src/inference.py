"""Single-utterance inference using the final MELD Context-3 implementation.

No runner entry point is executed. Paths are relative to this file, not /workspace.
Context must contain only earlier turns from the same conversation, oldest first.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from collections.abc import Mapping, Sequence

import pandas as pd
import torch
from transformers import AutoConfig, AutoFeatureExtractor, AutoModel, AutoTokenizer, WavLMModel


ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = ROOT / "experiments/robustness/checkpoints/context_best_best.pt"


def _load_source():
    # A private module namespace permits config-only encoder construction without
    # modifying the runner file or monkey-patching global transformers classes.
    path = ROOT.parent / "analysis" / "run_robustness_evaluation.py"
    spec = importlib.util.spec_from_file_location("_meld_inference_source", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EmotionPreprocessor:
    """Reuse the final training Dataset/Collator, including audio validation.

    Kept independent of model loading so preprocessing can be tested cheaply.
    """

    def __init__(self, assets_path=None):
        self.assets_path = Path(assets_path) if assets_path is not None else ROOT / "handoff_assets"
        self.source = _load_source()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.assets_path / "tokenizer", local_files_only=True
        )
        self.processor = AutoFeatureExtractor.from_pretrained(
            self.assets_path / "processor", local_files_only=True
        )
        if self.processor.sampling_rate != 16000 or self.processor.do_normalize:
            raise ValueError("Expected the saved 16 kHz, do_normalize=False audio processor.")
        self.collator = self.source.MultimodalCollator(self.processor)

    def build_context(self, context=None):
        """Select the last 3 past turns; preserve speaker prefixes and whitespace.

        Accept {'speaker': 'Ross', 'text': '...'} or 'Ross: ...'.
        Caller supplies past-only history; text alone cannot establish chronology.
        """
        if context is None:
            return []
        if isinstance(context, (str, bytes)) or not isinstance(context, Sequence):
            raise TypeError("context must be a sequence of past turns, oldest to newest.")
        turns = []
        for turn in context:
            if isinstance(turn, Mapping):
                speaker, utterance = turn.get("speaker"), turn.get("text")
                if not isinstance(speaker, str) or not speaker.strip() or not isinstance(utterance, str):
                    raise ValueError("Each context object needs nonempty speaker and string text.")
                turns.append(f"{speaker}: {utterance}")
            elif isinstance(turn, str) and ": " in turn and turn.split(": ", 1)[0].strip():
                turns.append(turn)
            else:
                raise ValueError("Training requires speaker-prefixed context: use {'speaker': 'Ross', 'text': '...'} or 'Ross: ...'.")
        return turns[-3:]

    def prepare_inputs(self, text, audio_path, context=None):
        if not isinstance(text, str):
            raise TypeError("text must be a string.")
        path = Path(audio_path)
        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {path.resolve()}")
        # Labels/index are required only by the existing Dataset interface. The
        # placeholder is discarded before forward and never used as a prediction.
        frame = pd.DataFrame([{
            "Utterance": text, "audio_path": str(path), "Emotion": "neutral",
            "Dialogue_ID": 0, "Utterance_ID": 0,
            "context_turn_texts": self.build_context(context),
        }])
        dataset = self.source.MultimodalDataset(
            frame, self.tokenizer, max_length=128, context_size=3,
            text_dropout=0.0, snr_db=None,
        )
        batch = self.collator([dataset[0]])
        return {key: value for key, value in batch.items() if key not in {"labels", "index"}}


class EmotionPredictor:
    def __init__(self, checkpoint_path=DEFAULT_CHECKPOINT, *, assets_path=None, device=None):
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Context checkpoint not found: {path.resolve()}. "
                "Pass checkpoint_path pointing to the trained context_best_best.pt. No training is performed."
            )
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        if self.device.type == "cuda" and not torch.cuda.is_available():
            self.device = torch.device("cpu")
        # The real checkpoint contains optimizer state as well. mmap avoids
        # eagerly allocating its entire 2.59 GB payload on a CPU laptop.
        checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if not isinstance(checkpoint, dict) or not {"model_state_dict", "config", "label2id", "id2label"} <= checkpoint.keys():
            raise ValueError("Expected final runner checkpoint with model_state_dict, config, label2id and id2label; a bare state_dict lacks verified metadata.")
        self.config = dict(checkpoint["config"])
        expected = {"text_model": "roberta-base", "audio_model": "microsoft/wavlm-base-plus",
                    "context_size": 3, "max_text_length": 128, "sample_rate": 16000, "dropout": 0.1}
        for key, value in expected.items():
            if self.config.get(key) != value:
                raise ValueError(f"Unsupported checkpoint config {key}={self.config.get(key)!r}; expected {value!r}.")
        assets = Path(assets_path) if assets_path is not None else ROOT / "handoff_assets"
        mapping = json.loads((assets / "label_mapping.json").read_text(encoding="utf-8"))
        self.id2label = {int(key): value for key, value in checkpoint["id2label"].items()}
        if (checkpoint["label2id"] != mapping["label2id"] or
                self.id2label != {int(key): value for key, value in mapping["id2label"].items()} or
                self.config.get("label2id") != mapping["label2id"]):
            raise ValueError("Checkpoint and saved asset label mappings disagree.")
        self.preprocessor = EmotionPreprocessor(assets)
        source = self.preprocessor.source
        if checkpoint["label2id"] != source.LABEL2ID:
            raise ValueError("Checkpoint labels disagree with the reused training implementation.")
        self.tokenizer, self.processor = self.preprocessor.tokenizer, self.preprocessor.processor
        text_config = AutoConfig.from_pretrained(assets / "text", local_files_only=True)
        audio_config = AutoConfig.from_pretrained(assets / "audio", local_files_only=True)

        def text_factory(name):
            if name != self.config["text_model"]:
                raise ValueError(f"Unexpected text encoder: {name}")
            return AutoModel.from_config(text_config)

        def audio_factory(name):
            if name != self.config["audio_model"]:
                raise ValueError(f"Unexpected audio encoder: {name}")
            return WavLMModel(audio_config)

        source.AutoModel = SimpleNamespace(from_pretrained=text_factory)
        source.WavLMModel = SimpleNamespace(from_pretrained=audio_factory)
        # Use the original constructor and forward verbatim, but construct its
        # encoders from exported configs: trained weights already exist locally.
        self.model = source.FusionClassifier(with_context=True)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True, assign=True)
        self.model.audio_encoder.config.apply_spec_augment = False
        self.model.to(self.device).eval()
        self.model.requires_grad_(False)

    def prepare_inputs(self, text, audio_path, context=None):
        return self.preprocessor.prepare_inputs(text, audio_path, context)

    @torch.inference_mode()
    def predict(self, text, audio_path, context=None):
        inputs = self.prepare_inputs(text, audio_path, context)
        logits = self.model(**{key: value.to(self.device) for key, value in inputs.items()})
        if logits.shape != (1, 7) or not torch.isfinite(logits).all():
            raise RuntimeError(f"Expected finite logits [1, 7], got {tuple(logits.shape)}.")
        probabilities = torch.softmax(logits.float(), dim=-1)[0].cpu()
        prediction = int(probabilities.argmax())
        return {"emotion": self.id2label[prediction], "confidence": float(probabilities[prediction]),
                "probabilities": {self.id2label[i]: float(probabilities[i]) for i in range(7)}}
