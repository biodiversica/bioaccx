"""Foundation model loading and embedding extraction.

Supports ONNX, TFLite, and TF SavedModel (protobuf) formats.
Models can be loaded from a local path or downloaded from Hugging Face Hub.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np

from bioaccx.audio import load_mono, to_fixed_length
from bioaccx.config import FoundationModelConfig


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseEmbedder(ABC):
    def __init__(self, cfg: FoundationModelConfig) -> None:
        self.cfg = cfg
        self._window = cfg.get_window_samples()

    @abstractmethod
    def embed(self, audio: np.ndarray) -> np.ndarray:
        """Return embedding vector of shape (embedding_size,)."""

    def embed_file(
        self,
        path: Path,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> np.ndarray:
        """Load an audio file (or segment) and return its embedding."""
        offset = start_time or 0.0
        duration = (end_time - offset) if end_time is not None else None
        audio = load_mono(path, self.cfg.sample_rate, offset=offset, duration=duration)
        audio = to_fixed_length(audio, self._window)
        return self.embed(audio)


# ---------------------------------------------------------------------------
# ONNX embedder
# ---------------------------------------------------------------------------

class ONNXEmbedder(BaseEmbedder):
    def __init__(self, cfg: FoundationModelConfig, model_path: Path) -> None:
        super().__init__(cfg)
        import onnxruntime as ort
        self._session = ort.InferenceSession(str(model_path))
        print(f"  [embedder] loaded ONNX model: {model_path}")

    def embed(self, audio: np.ndarray) -> np.ndarray:
        out = self._session.run(
            [self.cfg.output_name],
            {self.cfg.input_name: audio[np.newaxis, :]},
        )
        return out[0].squeeze(0).astype(np.float32)


# ---------------------------------------------------------------------------
# TFLite embedder
# ---------------------------------------------------------------------------

class TFLiteEmbedder(BaseEmbedder):
    def __init__(self, cfg: FoundationModelConfig, model_path: Path) -> None:
        super().__init__(cfg)
        import tensorflow as tf
        self._interp = tf.lite.Interpreter(model_path=str(model_path))
        self._interp.allocate_tensors()
        self._input_idx = self._interp.get_input_details()[0]["index"]
        out_details = self._interp.get_output_details()
        # Try to match by name first, then fall back to index 0
        name_match = [
            d for d in out_details if self.cfg.output_name in d["name"]
        ]
        self._output_idx = name_match[0]["index"] if name_match else out_details[0]["index"]
        print(f"  [embedder] loaded TFLite model: {model_path}")

    def embed(self, audio: np.ndarray) -> np.ndarray:
        self._interp.resize_input_tensor(self._input_idx, [1, len(audio)])
        self._interp.allocate_tensors()
        self._interp.set_tensor(self._input_idx, audio[np.newaxis, :])
        self._interp.invoke()
        out = self._interp.get_tensor(self._output_idx)
        return out.squeeze(0).astype(np.float32)


# ---------------------------------------------------------------------------
# Protobuf / TF SavedModel embedder
# ---------------------------------------------------------------------------

class ProtobufEmbedder(BaseEmbedder):
    def __init__(self, cfg: FoundationModelConfig, model_path: Path) -> None:
        super().__init__(cfg)
        import os
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        import tensorflow as tf
        self._model = tf.saved_model.load(str(model_path))
        # Prefer a named signature matching output_name; fall back to __call__
        sigs = list(self._model.signatures.keys())
        self._sig_key = self.cfg.output_name if self.cfg.output_name in sigs else (
            "embeddings" if "embeddings" in sigs else sigs[0] if sigs else None
        )
        self._tf = tf
        print(f"  [embedder] loaded SavedModel: {model_path}  sig={self._sig_key}")

    def embed(self, audio: np.ndarray) -> np.ndarray:
        import tensorflow as tf
        t = tf.constant(audio[np.newaxis, :], dtype=tf.float32)
        if self._sig_key is not None:
            out = self._model.signatures[self._sig_key](t)
            emb = list(out.values())[0].numpy()
        else:
            emb = self._model(t).numpy()
        return emb.squeeze(0).astype(np.float32)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def _resolve_model_path(cfg: FoundationModelConfig) -> Path:
    if cfg.source == "local":
        if cfg.path is None:
            raise ValueError("foundation_model.path must be set when source='local'")
        return Path(cfg.path)

    # Hugging Face Hub
    if cfg.hf_repo is None:
        raise ValueError("foundation_model.hf_repo must be set when source='huggingface'")
    from huggingface_hub import hf_hub_download
    filename = cfg.hf_filename or _default_hf_filename(cfg)
    local = hf_hub_download(
        repo_id=cfg.hf_repo,
        filename=filename,
        revision=cfg.hf_revision,
    )
    return Path(local)


def _default_hf_filename(cfg: FoundationModelConfig) -> str:
    ext = {"onnx": ".onnx", "tflite": ".tflite", "protobuf": ""}.get(cfg.format, ".onnx")
    return f"model{ext}"


def load_embedder(cfg: FoundationModelConfig) -> BaseEmbedder:
    path = _resolve_model_path(cfg)
    if cfg.format == "onnx":
        return ONNXEmbedder(cfg, path)
    if cfg.format == "tflite":
        return TFLiteEmbedder(cfg, path)
    if cfg.format == "protobuf":
        return ProtobufEmbedder(cfg, path)
    raise ValueError(f"Unsupported foundation_model.format: {cfg.format!r}")
