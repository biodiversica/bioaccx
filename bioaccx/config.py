"""Configuration dataclasses and config file loader (JSON / YAML)."""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional


def _from_dict(cls, data: dict):
    """Instantiate a dataclass from a dict, silently ignoring unknown keys."""
    known = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class FoundationModelConfig:
    name: str
    version: str = "unknown"
    # Model format on disk
    format: Literal["onnx", "tflite", "protobuf"] = "onnx"
    # Source: local file path or huggingface hub
    source: Literal["local", "huggingface"] = "local"
    path: Optional[str] = None          # path to local model file/dir
    hf_repo: Optional[str] = None       # e.g. "biodiversica/birdnet"
    hf_filename: Optional[str] = None   # specific file within HF repo
    hf_revision: Optional[str] = None   # branch / tag / commit
    # Audio preprocessing
    sample_rate: int = 48000
    window_samples: Optional[int] = None    # takes priority over window_seconds
    window_seconds: Optional[float] = None  # e.g. 3.0
    # ONNX / TFLite node names
    input_name: str = "input"
    output_name: str = "embedding"
    embedding_size: int = 1024

    def get_window_samples(self) -> int:
        if self.window_samples is not None:
            return self.window_samples
        if self.window_seconds is not None:
            return int(self.window_seconds * self.sample_rate)
        raise ValueError(
            "foundation_model must specify either 'window_samples' or 'window_seconds'"
        )


@dataclass
class DatasetConfig:
    # Local audio source(s); may be empty when using ext_table_file exclusively
    data_dir: str | list[str] = field(default_factory=list)
    # How labels are organized in data_dir
    label_mode: Literal["subfolders", "table", "file_per_label"] = "subfolders"
    table_file: Optional[str] = None   # CSV/TSV; used when label_mode="table"
    audio_extensions: list[str] = field(
        default_factory=lambda: ["wav", "flac", "mp3", "ogg"]
    )
    # Overlap between consecutive windows when chunking label segments (0.0–1.0)
    overlap: float = 0.0
    # Parallel workers for embedding extraction (capped to available CPU cores)
    embedding_workers: int = 4
    # Path to look for pre-computed .npy embeddings before running the model
    embeddings_cache_path: Optional[str] = None
    # Path to an existing exported dataset (subfolders layout with train/test).
    # New samples from data_dir that are not already present are appended to it.
    append_dataset_path: Optional[str] = None
    # Train / test split; ignored when the dataset already encodes the split
    test_ratio: float = 0.2
    random_seed: int = 42
    # Column names used in table / ext_table_file modes
    filename_col: str = "filename"
    label_col: str = "label"
    start_col: str = "start_time"
    end_col: str = "end_time"
    split_col: str = "split"  # optional; values "train" / "test"
    # iNaturalist table — CSV/TSV with observation_id rows (or mixed with filename rows)
    ext_table_file: Optional[str] = None
    obs_id_col: str = "observation_id"
    sound_index_col: str = "sound_index"
    xc_id_col: str = "xc_id"
    # Local cache for downloaded remote audio; defaults to ~/.cache/bioaccx/ext
    ext_cache_dir: Optional[str] = None
    # Xeno-canto API v3 key — required for metadata (scientific name lookup).
    # Audio can be downloaded without a key via the direct download URL.
    # Register at https://xeno-canto.org/explore/api
    xc_api_key: Optional[str] = None
    # Audio preprocessing applied before chunking (filter → speed → chunks)
    # filter: 'hpf' | 'lpf' | 'bpf' | null
    # filter_freq: Hz value for hpf/lpf; [low_hz, high_hz] list for bpf
    filter: Optional[str] = None
    filter_freq: Optional[float | list[float]] = None
    filter_order: int = 5
    # Playback speed multiplier (>1 faster / shorter, <1 slower / longer).
    # Label times are scaled accordingly: new_time = old_time / speed.
    speed: float = 1.0


@dataclass
class KerasConfig:
    hidden_units: int = 256
    dropout: float = 0.25
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-4
    # Output activation: None (logits, default) | "sigmoid" | "softmax"
    output_activation: Optional[str] = None


@dataclass
class SklearnConfig:
    C: float = 1.0
    max_iter: int = 2000
    solver: str = "lbfgs"


@dataclass
class TrainingConfig:
    classifier: Literal["keras", "sklearn", "both"] = "keras"
    keras: KerasConfig = field(default_factory=KerasConfig)
    sklearn: SklearnConfig = field(default_factory=SklearnConfig)


@dataclass
class OutputConfig:
    output_path: str = "./outputs"
    model_name: str = "custom_classifier"
    model_version: str = "1.0"
    # head = classifier only; full = foundation + classifier; both = save both
    output_type: Literal["head", "full", "both"] = "head"
    output_format: Literal["onnx", "tflite", "both"] = "onnx"
    # Labels to exclude from the exported classifier output (still used during training)
    exclude_labels: list[str] = field(default_factory=list)
    # Export chunked audio samples as WAV files in label subfolders
    export_dataset: bool = False
    # Save computed embeddings as .npy files for reuse
    export_embeddings: bool = False
    # Directory for exported embeddings; defaults to <output_dir>/embeddings
    embeddings_path: Optional[str] = None


@dataclass
class BioaccxConfig:
    foundation_model: FoundationModelConfig
    dataset: DatasetConfig
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @property
    def output_dir(self) -> Path:
        name = f"{self.output.model_name}_v{self.output.model_version}"
        return Path(self.output.output_path) / name


def load_config(path: str | Path) -> BioaccxConfig:
    p = Path(path)
    text = p.read_text()
    if p.suffix in (".yaml", ".yml"):
        import yaml  # deferred so JSON-only users avoid the dep
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    return _parse_config(data)


def _parse_config(data: dict) -> BioaccxConfig:
    fm = _from_dict(FoundationModelConfig, data["foundation_model"])

    ds = _from_dict(DatasetConfig, data["dataset"])

    tr_raw = dict(data.get("training", {}))
    keras_raw = tr_raw.pop("keras", {})
    sklearn_raw = tr_raw.pop("sklearn", {})
    tr = TrainingConfig(
        **{k: v for k, v in tr_raw.items() if k in {"classifier"}},
        keras=_from_dict(KerasConfig, keras_raw),
        sklearn=_from_dict(SklearnConfig, sklearn_raw),
    )

    out = _from_dict(OutputConfig, data.get("output", {}))

    return BioaccxConfig(foundation_model=fm, dataset=ds, training=tr, output=out)
