"""Foundation model loading and embedding extraction.

Supports ONNX, TFLite, and TF SavedModel (protobuf) formats.
Models can be loaded from a local path or downloaded from Hugging Face Hub
or Kaggle (via kagglehub).
"""
from __future__ import annotations

import threading
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
    """Abstract base for all foundation-model embedders.

    Subclasses implement ``embed(audio)`` for a specific inference backend
    (ONNX, TFLite, TF SavedModel).  The common ``embed_file`` method handles
    file loading, segmentation, and fixed-length padding before delegating to
    the backend.
    """

    def __init__(self, cfg: FoundationModelConfig) -> None:
        self.cfg = cfg
        self._window = cfg.get_window_samples()

    @abstractmethod
    def embed(self, audio: np.ndarray) -> np.ndarray:
        """Return embedding vector of shape (embedding_size,)."""

    def _pool(self, raw: np.ndarray) -> np.ndarray:
        """Squeeze batch dim, mean-pool sequence dim if present, and validate."""
        emb = raw.squeeze(0)
        if emb.ndim > 1:
            emb = emb.mean(axis=0)
        if emb.shape[0] != self.cfg.embedding_size:
            raise ValueError(
                f"Embedding size mismatch: model produced {emb.shape[0]}, "
                f"but config specifies embedding_size={self.cfg.embedding_size}"
            )
        return emb.astype(np.float32)

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
    """Embedder backed by an ONNX Runtime inference session.

    ONNX Runtime releases the Python GIL during ``session.run()``, so multiple
    ONNXEmbedder instances running in threads achieve genuine parallelism.
    """

    def __init__(self, cfg: FoundationModelConfig, model_path: Path) -> None:
        super().__init__(cfg)
        import onnxruntime as ort
        providers = cfg.onnx_providers or ort.get_available_providers()
        self._session = ort.InferenceSession(str(model_path), providers=providers)
        print(f"  [embedder] loaded ONNX model: {model_path}  providers={self._session.get_providers()}")

    def embed(self, audio: np.ndarray) -> np.ndarray:
        """Run inference and return a 1-D embedding of shape (embedding_size,).

        The audio array is wrapped in a batch dimension (shape [1, window]) to
        satisfy models that require a batch axis.
        """
        out = self._session.run(
            [self.cfg.output_name],
            {self.cfg.input_name: audio[np.newaxis, :].astype(np.float32)},
        )
        return self._pool(out[0])

    def embed_batch(self, audio_batch: np.ndarray) -> np.ndarray:
        """Run inference on N audio windows; return shape (N, embedding_size).

        audio_batch must have shape (N, window_samples).
        """
        out = self._session.run(
            [self.cfg.output_name],
            {self.cfg.input_name: audio_batch.astype(np.float32)},
        )
        raw = out[0]  # (N, embedding_size) or (N, seq, embedding_size)
        if raw.ndim == 3:
            raw = raw.mean(axis=1)
        if raw.shape[1] != self.cfg.embedding_size:
            raise ValueError(
                f"Embedding size mismatch: model produced {raw.shape[1]}, "
                f"but config specifies embedding_size={self.cfg.embedding_size}"
            )
        return raw.astype(np.float32)


# ---------------------------------------------------------------------------
# TFLite embedder
# ---------------------------------------------------------------------------

# Trimmed TFLite graphs, keyed by (path, mtime, size, embedding tensor index).
# extract_embeddings gives every worker thread its own embedder, so without a
# cache each thread would repeat the same (slow, memory-hungry) trim. The lock
# is held for the whole trim so concurrent threads wait rather than duplicating
# the work — and its peak memory.
_TRIM_CACHE: dict[tuple, "bytes | None"] = {}
_TRIM_LOCK = threading.Lock()


def _trimmed_tflite_bytes(model_path: Path, output_idx: int) -> "bytes | None":
    """Return *model_path* trimmed to compute only tensor *output_idx*.

    Returns ``None`` (and warns) when the trim fails, so the caller can fall
    back to the model as shipped — a failed optimisation must not fail a run.
    """
    stat = model_path.stat()
    key = (str(model_path), stat.st_mtime_ns, stat.st_size, output_idx)
    with _TRIM_LOCK:
        if key in _TRIM_CACHE:
            return _TRIM_CACHE[key]
        try:
            from bioaccx.exporters.tflite_exporter import _trim_tflite_to_output
            trimmed = _trim_tflite_to_output(model_path.read_bytes(), output_idx)
        except Exception as exc:  # noqa: BLE001 — any failure falls back safely
            print(f"  [embedder] Warning: could not trim TFLite graph ({exc}); "
                  f"using the model as shipped.")
            trimmed = None
        _TRIM_CACHE[key] = trimmed
        return trimmed


class TFLiteEmbedder(BaseEmbedder):
    """Embedder backed by a TFLite Interpreter.

    TFLite does not release the GIL during inference, so parallel threads do
    not provide real speedup; each worker thread must own its own instance
    (enforced in extract_embeddings via thread-local storage).

    When the model computes more than the embedding — a bundled classifier head
    (BirdNET, Perch) or auxiliary outputs — the graph is trimmed to the
    embedding tensor once at load time, so that work is not repeated on every
    window.  Disable with ``tflite_trim_to_embedding: false``.
    """

    def __init__(self, cfg: FoundationModelConfig, model_path: Path) -> None:
        super().__init__(cfg)
        import tensorflow as tf

        # This first interpreter only resolves tensor indices, so it never needs
        # preserve_all_tensors; the fallback path below rebuilds it if required.
        interp = tf.lite.Interpreter(model_path=str(model_path))
        interp.allocate_tensors()
        out_details = interp.get_output_details()
        # Try to match by name first, then fall back to index 0
        name_match = [d for d in out_details if self.cfg.output_name in d["name"]]
        base_idx = name_match[0]["index"] if name_match else out_details[0]["index"]
        output_idx = base_idx + cfg.tflite_output_tensor_offset

        # Trimming pays off when the graph produces more than the embedding, and
        # when the embedding is an intermediate tensor — turning it into a real
        # output also removes the need for the slow preserve_all_tensors path.
        trimmed = None
        if cfg.tflite_trim_to_embedding and (
            len(out_details) > 1 or cfg.tflite_output_tensor_offset != 0
        ):
            trimmed = _trimmed_tflite_bytes(model_path, output_idx)

        if trimmed is not None:
            candidate = tf.lite.Interpreter(model_content=trimmed)
            candidate.allocate_tensors()
            cand_out = candidate.get_output_details()
            shape = list(cand_out[0]["shape"]) if len(cand_out) == 1 else []
            if shape and int(shape[-1]) == cfg.embedding_size:
                interp, output_idx = candidate, cand_out[0]["index"]
                print(f"  [embedder] trimmed TFLite graph to the embedding tensor "
                      f"({len(out_details)} output(s) → 1)")
            else:
                trimmed = None
                print(f"  [embedder] Warning: trimmed graph does not expose a single "
                      f"{cfg.embedding_size}-dim output (got {shape or len(cand_out)}); "
                      f"using the model as shipped.")

        if trimmed is None and cfg.tflite_output_tensor_offset != 0:
            # preserve_all_tensors is required when the embedding stays an
            # intermediate tensor, otherwise its memory is freed after invoke()
            # and get_tensor() returns null data.
            interp = tf.lite.Interpreter(
                model_path=str(model_path), experimental_preserve_all_tensors=True
            )
            interp.allocate_tensors()

        self._interp = interp
        self._input_idx = interp.get_input_details()[0]["index"]
        self._output_idx = output_idx
        print(f"  [embedder] loaded TFLite model: {model_path}")

    def embed(self, audio: np.ndarray) -> np.ndarray:
        """Run inference on a single audio window and return its embedding.

        TFLite requires resize_tensor_input + allocate_tensors every time the
        input shape changes, which also happens for dynamic-shape models even
        when the shape is the same.  We do this unconditionally to be safe.
        """
        self._interp.resize_tensor_input(self._input_idx, [1, len(audio)])
        self._interp.allocate_tensors()
        self._interp.set_tensor(self._input_idx, audio[np.newaxis, :])
        self._interp.invoke()
        out = self._interp.get_tensor(self._output_idx)
        return self._pool(out)


# ---------------------------------------------------------------------------
# Protobuf / TF SavedModel embedder
# ---------------------------------------------------------------------------

class ProtobufEmbedder(BaseEmbedder):
    """Embedder backed by a TensorFlow SavedModel (protobuf directory).

    Signature selection priority: cfg.output_name → "embeddings" → first key
    → None (direct __call__).  The selected key is stored at construction time
    so each call doesn't re-scan the signature dict.
    """

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
        """Run a SavedModel signature or direct __call__ and return the embedding.

        SavedModel signatures return a dict of output tensors; we take the first
        value (which is conventionally the embedding tensor).
        """
        import tensorflow as tf
        t = tf.constant(audio[np.newaxis, :], dtype=tf.float32)
        if self._sig_key is not None:
            out = self._model.signatures[self._sig_key](t)
            emb = list(out.values())[0].numpy()
        else:
            emb = self._model(t).numpy()
        return self._pool(emb)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def _resolve_model_path(cfg: FoundationModelConfig) -> Path:
    """Resolve the foundation model to a local filesystem path.

    For HuggingFace and Kaggle sources the model is downloaded (or retrieved
    from the hub's local cache) on the first call.  Subsequent calls are fast
    because both hf_hub_download and kagglehub cache downloads on disk.

    For Kaggle with format != 'protobuf' and no explicit kaggle_filename, the
    function scans the downloaded directory for a single matching file; if
    multiple or zero candidates are found, a descriptive ValueError is raised
    so the user knows to set kaggle_filename.
    """
    if cfg.source == "local":
        if cfg.path is None:
            raise ValueError("foundation_model.path must be set when source='local'")
        return Path(cfg.path)

    if cfg.source == "huggingface":
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

    # Kaggle: download to kagglehub's local model cache.
    if cfg.kaggle_handle is None:
        raise ValueError("foundation_model.kaggle_handle must be set when source='kaggle'")
    import kagglehub
    local_dir = Path(kagglehub.model_download(cfg.kaggle_handle))
    if cfg.kaggle_filename:
        return local_dir / cfg.kaggle_filename
    # Protobuf SavedModels are directories; return the directory itself.
    if cfg.format == "protobuf":
        return local_dir
    # For single-file formats, find the unique matching file or raise a helpful error.
    ext = {"onnx": ".onnx", "tflite": ".tflite"}.get(cfg.format, ".onnx")
    candidates = sorted(local_dir.rglob(f"*{ext}"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise ValueError(f"No {ext} file found in Kaggle download: {local_dir}")
    raise ValueError(
        f"Multiple {ext} files found in {local_dir}. "
        f"Set kaggle_filename to specify one: {[str(c.relative_to(local_dir)) for c in candidates]}"
    )


def _default_hf_filename(cfg: FoundationModelConfig) -> str:
    """Return the conventional HuggingFace filename for the configured format.

    Assumes the repo follows the ``model.<ext>`` convention.  Protobuf
    (SavedModel) repos may not have a single file at this path; users should
    set hf_filename explicitly in that case.
    """
    ext = {"onnx": ".onnx", "tflite": ".tflite", "protobuf": ""}.get(cfg.format, ".onnx")
    return f"model{ext}"


def load_embedder(cfg: FoundationModelConfig) -> BaseEmbedder:
    """Instantiate the appropriate embedder for the configured model format.

    Resolves the model path (downloading from HF/Kaggle if needed) and
    constructs the matching backend class.  Raises ValueError for unknown formats.
    """
    path = _resolve_model_path(cfg)
    if cfg.format == "onnx":
        return ONNXEmbedder(cfg, path)
    if cfg.format == "tflite":
        return TFLiteEmbedder(cfg, path)
    if cfg.format == "protobuf":
        return ProtobufEmbedder(cfg, path)
    raise ValueError(f"Unsupported foundation_model.format: {cfg.format!r}")
