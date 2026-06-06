"""Configuration dataclasses and config file loader (JSON / YAML)."""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

from bioaccx.registry import get_registry_defaults, lookup_foundation_model_id


def _from_dict(cls, data: dict):
    """Instantiate a dataclass from a dict, silently ignoring unknown keys.

    Unknown keys are dropped rather than raising TypeError, which lets users
    add comments or future-compat fields to their config files without errors.
    """
    known = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class FoundationModelConfig:
    name: str
    version: str = "unknown"
    data_type: str = "FP32"
    # Model format on disk
    format: Literal["onnx", "tflite", "protobuf"] = "onnx"
    # Source: local file path, huggingface hub, or kaggle
    source: Literal["local", "huggingface", "kaggle"] = "local"
    path: Optional[str] = None          # path to local model file/dir
    hf_repo: Optional[str] = None       # e.g. "biodiversica/birdnet"
    hf_filename: Optional[str] = None   # specific file within HF repo
    hf_revision: Optional[str] = None   # branch / tag / commit
    kaggle_handle: Optional[str] = None  # e.g. "google/bird-vocalization-classifier/tensorFlow2/bird-vocalization-classifier"
    kaggle_filename: Optional[str] = None  # specific file within the downloaded dir (onnx/tflite)
    # Audio preprocessing
    sample_rate: int = 48000
    window_samples: Optional[int] = None    # takes priority over window_seconds
    window_seconds: Optional[float] = None  # e.g. 3.0
    # ONNX / TFLite node names
    input_name: str = "input"
    output_name: str = "embedding"
    embedding_size: int = 1024
    # ONNX Runtime execution providers (e.g. ["CUDAExecutionProvider", "CPUExecutionProvider"]).
    # Defaults to ORT's own provider priority when None.
    onnx_providers: Optional[list[str]] = None
    # Number of audio windows to process in a single ONNX inference call.
    # Values > 1 enable GPU batch mode (used only for format="onnx").
    onnx_batch_size: int = 1
    # Offset applied to the resolved TFLite output tensor index to reach the
    # embedding tensor.  Use -1 for models like BirdNET tflite where the
    # classifier head is the first output and the embedding sits one slot before
    # it in the graph's tensor list.
    tflite_output_tensor_offset: int = 0

    def get_window_samples(self) -> int:
        """Return the embedding window length in samples.

        ``window_samples`` takes priority over ``window_seconds`` when both are
        set.  Raises ValueError if neither is provided.
        """
        if self.window_samples is not None:
            return self.window_samples
        if self.window_seconds is not None:
            return int(self.window_seconds * self.sample_rate)
        raise ValueError(
            "foundation_model must specify either 'window_samples' or 'window_seconds'"
        )


@dataclass
class AugmentationConfig:
    augmentation_dir: str
    snr_levels: list[float]
    keep_original: bool = True
    augment_test: bool = False
    skip_labels: Optional[list[str]] = None
    # When True, all files in augmentation_dir are concatenated into a single
    # in-memory array used as the sole noise source. If a sibling Audacity .txt
    # label file exists for an audio file, only the labeled segments are used;
    # otherwise the entire file is included.
    concatenate_augmentation_dir: bool = False


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
    # Arbimon / rfcx — path to the persisted credentials file produced by
    # rfcx.Client().authenticate(persisted_credentials_path=...).
    # Column names used to identify Arbimon rows in ext_table_file.
    arbimon_credentials_path: Optional[str] = None
    arbimon_stream_id_col: str = "stream_id"
    arbimon_date_col: str = "date"
    arbimon_time_col: str = "time"
    arbimon_utc_offset_col: str = "utc_offset"
    # Audio preprocessing applied before chunking (filter → speed → chunks)
    # filter: 'hpf' | 'lpf' | 'bpf' | null
    # filter_freq: Hz value for hpf/lpf; [low_hz, high_hz] list for bpf
    filter: Optional[str] = None
    filter_freq: Optional[float | list[float]] = None
    filter_order: int = 5
    # Playback speed multiplier (>1 faster / shorter, <1 slower / longer).
    # Label times are scaled accordingly: new_time = old_time / speed.
    speed: float = 1.0
    # SSH / SFTP access — when ssh_host is set, data_dir paths are treated as
    # remote paths on the SSH server and mirrored locally via paramiko before
    # the pipeline runs.  Requires: pip install paramiko
    ssh_host: Optional[str] = None
    ssh_user: Optional[str] = None
    ssh_port: int = 22
    ssh_key_path: Optional[str] = None   # path to private key file
    augmentation: Optional[AugmentationConfig] = None
    # When True, samples shorter than the foundation model window are placed at a
    # random offset within the window rather than always starting at position 0.
    # Each augmented copy of the same sample gets a distinct offset.
    random_sample_shift: bool = False
    # Minimum fraction of the window that must be new (uncovered) audio for the
    # anchor chunk at the end of a file to be emitted. Prevents near-duplicate
    # chunks when files are only slightly longer than the window (e.g. 3.013 s
    # with a 3 s window). Set to 0.0 to always emit the anchor chunk.
    min_anchor_fraction: float = 0.1


@dataclass
class KerasConfig:
    hidden_units: int = 256
    dropout: float = 0.25
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-4
    # Output activation: None (logits, default) | "sigmoid" | "softmax"
    output_activation: Optional[str] = None
    # Z-score normalization of input embeddings (adapted on X_train)
    normalize_embeddings: bool = True
    # Focal loss (replaces cross-entropy when enabled)
    focal_loss: bool = False
    focal_loss_gamma: float = 2.0
    focal_loss_alpha: float = 0.25
    # Label smoothing applied to one-hot targets before training
    label_smoothing: bool = False
    label_smoothing_alpha: float = 0.1
    # Mixup data augmentation on training embeddings
    mixup: bool = False
    mixup_ratio: float = 0.25
    mixup_alpha: float = 0.2
    # Upsampling of minority classes before training
    upsampling_ratio: float = 0.0
    upsampling_mode: Literal["repeat", "mean", "linear", "smote"] = "repeat"
    # Set TF + numpy random seeds before training for reproducibility.
    # Uses dataset.random_seed. Set to false to disable (training will vary run-to-run).
    seed: bool = True


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
    # Save computed embeddings for reuse
    export_embeddings: bool = False
    # Storage format for exported/cached embeddings: npy (one file per sample)
    # or sqlite (single .db file per run, more portable)
    embeddings_format: Literal["npy", "sqlite"] = "npy"
    # Directory (npy) or file path (sqlite) for exported embeddings;
    # defaults to <output_dir>/embeddings or <output_dir>/embeddings.db
    embeddings_path: Optional[str] = None
    # Path to an existing ONNX classifier head for --merge (no training required)
    head_path: Optional[str] = None


@dataclass
class BioaccxConfig:
    foundation_model: FoundationModelConfig
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @property
    def model_stem(self) -> str:
        """Output name stem: ``<model_name>_<foundation_model_id>_v<model_version>``."""
        fm = self.foundation_model
        fm_id = lookup_foundation_model_id(fm.name, fm.version, fm.data_type, fm.format)
        return f"{self.output.model_name}_{fm_id}_v{self.output.model_version}"

    @property
    def output_dir(self) -> Path:
        """Fully qualified output directory: ``<output_path>/<model_stem>``."""
        return Path(self.output.output_path) / self.model_stem


def load_config(path: str | Path) -> BioaccxConfig:
    """Parse a JSON or YAML config file and return a validated BioaccxConfig.

    YAML support requires the ``pyyaml`` package; it is imported lazily so
    that JSON-only installations are unaffected.
    """
    p = Path(path)
    text = p.read_text()
    if p.suffix in (".yaml", ".yml"):
        import yaml  # deferred so JSON-only users avoid the dep
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    return _parse_config(data)


def _resolve_foundation_model(raw: dict) -> dict:
    """Merge registry defaults with user-supplied foundation model fields.

    If ``registry_id`` is present the registry entry for that ID is used as the
    base.  Otherwise a lookup by ``(name, version, data_type, format)`` is
    attempted automatically.  User-supplied fields always override registry
    defaults.  ``registry_id`` itself is consumed here and not forwarded to the
    dataclass.
    """
    raw = dict(raw)
    registry_id = raw.pop("registry_id", None)

    if registry_id is not None:
        defaults = get_registry_defaults(registry_id)
        if defaults is None:
            raise ValueError(
                f"Unknown registry_id {registry_id!r}. "
                f"Check bioaccx.registry.list_registry_ids() for valid IDs."
            )
    else:
        # Auto-lookup by model identity fields using the same defaults as the
        # dataclass so that partially-specified configs resolve correctly.
        name = raw.get("name", "")
        version = raw.get("version", "unknown")
        data_type = raw.get("data_type", "FP32")
        fmt = raw.get("format", "onnx")
        defaults = get_registry_defaults(
            # reuse the same key the registry indexes on
            _registry_id_from_fields(name, version, data_type, fmt)
        )

    if defaults is not None:
        raw = {**defaults, **raw}

    return raw


def _registry_id_from_fields(name: str, version: str, data_type: str, fmt: str) -> int:
    """Return the int registry key for (name, version, data_type, format), or
    a sentinel that is guaranteed not to be in the registry."""
    from bioaccx.registry import _KEY_INDEX  # noqa: PLC0415
    return _KEY_INDEX.get(
        (name.lower(), version, data_type.upper(), fmt.lower()), -1
    )


def _parse_config(data: dict) -> BioaccxConfig:
    """Build a BioaccxConfig from a raw parsed dict.

    Training is handled separately from its nested ``keras``/``sklearn`` blocks
    because _from_dict cannot recursively construct nested dataclasses — the
    inner dicts would be passed as plain dicts rather than typed objects.
    """
    fm = _from_dict(FoundationModelConfig, _resolve_foundation_model(data["foundation_model"]))

    ds_raw = dict(data.get("dataset", {}))
    aug_raw = ds_raw.pop("augmentation", None)
    ds = _from_dict(DatasetConfig, ds_raw)
    if aug_raw is not None:
        ds.augmentation = _from_dict(AugmentationConfig, aug_raw)

    # Pop nested trainer configs before passing the remainder to TrainingConfig.
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
