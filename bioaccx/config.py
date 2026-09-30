"""Configuration dataclasses and config file loader (JSON / YAML)."""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

from bioaccx.registry import get_registry_defaults, lookup_foundation_model_id


# Output precisions supported for the exported model/classifier.
VALID_OUTPUT_DATA_TYPES = ("FP32", "FP16", "INT8")


def _from_dict(cls, data: dict):
    """Instantiate a dataclass from a dict, silently ignoring unknown keys.

    Unknown keys are dropped rather than raising TypeError, which lets users
    add comments or future-compat fields to their config files without errors.
    """
    known = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class FoundationModelConfig:
    name: str                           # used in reports and output filenames
    version: str = "unknown"            # model version string
    data_type: str = "FP32"             # weight precision, e.g. FP32 / INT8
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
    # Sample rate the backbone expects, in Hz
    sample_rate: int = 48000
    window_samples: Optional[int] = None    # takes priority over window_seconds
    window_seconds: Optional[float] = None  # e.g. 3.0
    input_name: str = "input"           # name of the audio input tensor
    output_name: str = "embedding"      # name of the output (embedding) tensor
    embedding_size: int = 1024          # embedding vector dimensionality
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
    # Trim a TFLite graph down to the embedding tensor when the model computes
    # more than that (a bundled classifier head, auxiliary outputs). Done once
    # at load time; every later window then skips the discarded branches.
    # Set false to run the model exactly as shipped.
    tflite_trim_to_embedding: bool = True

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
    # SNR values in dB; one augmented copy per noise file per level
    snr_levels: list[float]
    # Directory of WAV files used as noise sources. Optional when
    # augmentation_labels is set; if both are given the noise pool is the union.
    augmentation_dir: Optional[str] = None
    # Labels of the dataset being created whose audio is used as additional
    # noise sources (in addition to any augmentation_dir). The samples of these
    # labels are mixed into the other samples' augmentation but are themselves
    # never augmented (treated like skip_labels). They remain trainable classes.
    augmentation_labels: Optional[list[str]] = None
    # Keep the clean sample alongside its augmented copies
    keep_original: bool = True
    # Augment the test set as well, not just the training set
    augment_test: bool = False
    # Labels left unaugmented; they stay trainable classes
    skip_labels: Optional[list[str]] = None
    # When True, all files in augmentation_dir are concatenated into a single
    # in-memory array used as the sole noise source. If a sibling Audacity .txt
    # label file exists for an audio file, only the labeled segments are used;
    # otherwise the entire file is included.
    concatenate_augmentation_dir: bool = False
    # When True, files in augmentation_dir are shuffled once (using random_seed)
    # and assigned round-robin to augmented samples — one file per (sample, SNR)
    # pair, cycling without repetition within each pass. This produces one noise
    # condition (like concatenate_augmentation_dir) but draws from individual
    # files rather than a merged track.
    random_augmentation_dir: bool = False


@dataclass
class DatasetConfig:
    # Local audio source(s); may be empty when using ext_table_file exclusively
    data_dir: str | list[str] = field(default_factory=list)
    # How labels are organized in data_dir
    label_mode: Literal["subfolders", "table", "file_per_label"] = "subfolders"
    table_file: Optional[str] = None   # CSV/TSV; used when label_mode="table"
    # Accepted audio file extensions (case-insensitive)
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
    random_seed: int = 42               # seed for splits, shuffling and noise offsets
    # Column names used in table / ext_table_file modes
    filename_col: str = "filename"
    label_col: str = "label"            # class label of the row
    start_col: str = "start_time"       # segment start, in seconds
    end_col: str = "end_time"           # segment end, in seconds
    split_col: str = "split"  # optional; values "train" / "test"
    # iNaturalist table — CSV/TSV with observation_id rows (or mixed with filename rows)
    ext_table_file: Optional[str] = None
    obs_id_col: str = "observation_id"  # iNaturalist observation id
    sound_index_col: str = "sound_index"  # iNaturalist sound index (0-based)
    xc_id_col: str = "xc_id"            # Xeno-canto recording id
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
    arbimon_stream_id_col: str = "stream_id"   # Arbimon stream / site id
    arbimon_date_col: str = "date"             # local recording date
    arbimon_time_col: str = "time"             # local recording start time
    arbimon_utc_offset_col: str = "utc_offset" # UTC offset of those times
    # Audio preprocessing applied before chunking (filter → speed → chunks).
    # Filter type: 'hpf' | 'lpf' | 'bpf' | null
    filter: Optional[str] = None
    # Cut-off in Hz for hpf/lpf; [low_hz, high_hz] for bpf
    filter_freq: Optional[float | list[float]] = None
    filter_order: int = 5               # Butterworth filter order
    # Playback speed multiplier (>1 faster / shorter, <1 slower / longer).
    # Label times are scaled accordingly: new_time = old_time / speed.
    speed: float = 1.0
    # SSH / SFTP access — when ssh_host is set, data_dir paths are treated as
    # remote paths on the SSH server and mirrored locally via paramiko before
    # the pipeline runs.  Requires: pip install paramiko
    ssh_host: Optional[str] = None
    ssh_user: Optional[str] = None      # SSH user; defaults to the local one
    ssh_port: int = 22                  # SSH port
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
    # Units in the hidden Dense layer; 0 = no hidden layer (linear classifier)
    hidden_units: int = 256
    dropout: float = 0.25               # dropout rate before each Dense layer
    epochs: int = 50                    # maximum epochs; early stopping may halt sooner
    batch_size: int = 32                # mini-batch size
    # Adam peak learning rate (cosine decay with linear warmup)
    learning_rate: float = 1e-4
    # Output activation: None (logits, default) | "sigmoid" | "softmax"
    #                    | "grouped_softmax" (requires label_groups)
    output_activation: Optional[str] = None
    # Grouped softmax: softmax within each group, groups independent of each
    # other.  Maps group name -> member labels; each group gains a synthetic
    # "<group>_none" output column.  Members of one group are mutually
    # exclusive, members of different groups can fire together.  Training
    # labels not listed in any group are background: they get no output column
    # and supply the "none" target for every group.
    label_groups: dict[str, list[str]] = field(default_factory=dict)
    # Strip the output activation layer before exporting, so the exported head
    # emits raw logits while training still used output_activation (this is what
    # BirdNET-Analyzer does with classifier.pop()). No-op when
    # output_activation is None. Callers must apply the activation themselves.
    # Rejected with exclude_labels on a softmax or grouped head, where the
    # excluded logits are needed to rebuild the softmax.
    export_logits: bool = False
    # Z-score normalization of input embeddings (adapted on X_train)
    normalize_embeddings: bool = True
    # Focal loss (replaces cross-entropy when enabled)
    focal_loss: bool = False
    focal_loss_gamma: float = 2.0       # focal loss focusing parameter γ
    focal_loss_alpha: float = 0.25      # focal loss class balance parameter α
    # Label smoothing applied to one-hot targets before training
    label_smoothing: bool = False
    # Subtracted from positive labels and redistributed to the negatives
    label_smoothing_alpha: float = 0.1
    # Mixup data augmentation on training embeddings
    mixup: bool = False
    mixup_ratio: float = 0.25           # fraction of positive samples to mix
    mixup_alpha: float = 0.2            # Beta parameter of the mixing coefficient
    # Upsampling of minority classes before training
    upsampling_ratio: float = 0.0
    # repeat = random duplication, mean = pairwise mean, linear = random
    # interpolation, smote = k-NN interpolation
    upsampling_mode: Literal["repeat", "mean", "linear", "smote"] = "repeat"
    # Set TF + numpy random seeds before training for reproducibility.
    # Uses dataset.random_seed. Set to false to disable (training will vary run-to-run).
    seed: bool = True


@dataclass
class SklearnConfig:
    C: float = 1.0                      # inverse regularisation strength
    max_iter: int = 2000                # maximum solver iterations
    solver: str = "lbfgs"               # LogisticRegression solver algorithm


@dataclass
class AudioMixupConfig:
    # Mix the audio of train windows from different classes, embed the mixture
    # and train on the union of their labels. Requires a keras sigmoid head.
    enabled: bool = True
    # Labels whose windows may be mixed; windows of any other label are never
    # used as a source (they still train as real samples). Empty = every label
    labels: list[str] = field(default_factory=list)
    # Number of mixes; overrides ratio when set
    n_mixes: Optional[int] = None
    # Number of mixes as a fraction of the real train windows
    ratio: float = 0.5
    # Sources per mix: 2, or 3 (with p_three_sources)
    max_sources: int = 2
    # Share of 3-source mixes when max_sources is 3
    p_three_sources: float = 0.0
    # [min, max] level in dB of each added source relative to the first
    snr_db: list[float] = field(default_factory=lambda: [-6.0, 6.0])
    # balanced = pick classes uniformly, then a clip; uniform = pick clips uniformly
    pairing: Literal["balanced", "uniform"] = "balanced"
    # Groups of mutually exclusive labels (e.g. call types or intensity levels
    # of one species) that are never mixed with each other
    exclusive_groups: dict[str, list[str]] = field(default_factory=dict)
    # Labels that are background, not species: a mix holds at most one of
    # them, and always at least one non-background source
    background_labels: list[str] = field(default_factory=list)
    # drop = a background source adds no label to the mix target;
    # include = it adds its own label
    background_target: Literal["drop", "include"] = "drop"
    # Seed for drawing the mixes; defaults to dataset.random_seed
    seed: Optional[int] = None
    # Also build mixes from test windows only, as a fraction of the test windows
    # that may be mixed (0 = none); reported as a separate section and never
    # pooled with the real test set
    test_mix_ratio: float = 0.0


@dataclass
class TrainingConfig:
    # Which head(s) to train: keras, sklearn, or both
    classifier: Literal["keras", "sklearn", "both"] = "keras"
    keras: KerasConfig = field(default_factory=KerasConfig)
    sklearn: SklearnConfig = field(default_factory=SklearnConfig)
    # Audio-level mixup for multi-label sigmoid heads
    audio_mixup: Optional[AudioMixupConfig] = None


@dataclass
class OutputConfig:
    output_path: str = "./outputs"      # parent directory for all output
    model_name: str = "custom_classifier"  # used in filenames and the subdirectory
    model_version: str = "1.0"          # version string used in filenames
    # head = classifier only; full = foundation + classifier; both = save both
    output_type: Literal["head", "full", "both"] = "head"
    # File format(s) of the exported model
    output_format: Literal["onnx", "tflite", "both"] = "onnx"
    # Labels to exclude from the exported classifier output (still used during training)
    exclude_labels: list[str] = field(default_factory=list)
    # Export chunked audio samples as WAV files in label subfolders
    export_dataset: bool = False
    # Export only the audio mixes (training.audio_mixup) as WAV files in
    # dataset/mixes/, to check them when the real windows exist already
    export_mixes: bool = False
    # Save computed embeddings for reuse
    export_embeddings: bool = False
    # Storage format for exported/cached embeddings: npy (one file per sample)
    # or sqlite (single .db file per run, more portable)
    embeddings_format: Literal["npy", "sqlite"] = "npy"
    # Directory (npy) or file path (sqlite) for exported embeddings;
    # defaults to <output_dir>/embeddings or <output_dir>/embeddings.db
    embeddings_path: Optional[str] = None
    # --embeddings mode only: when an embedding store already exists at the
    # resolved path, recompute and overwrite it. When False (default), the
    # existing store is reused as-is — no embeddings are recomputed and the run
    # only (re)builds the UMAP outputs (when umap.enabled).
    embeddings_overwrite: bool = False
    # Path to an existing ONNX classifier head for --merge (no training required)
    head_path: Optional[str] = None
    # Path to a full BirdNET-Analyzer model (.tflite) for --extract_head: the
    # trailing classifier head is read from the flatbuffer and re-exported as a
    # head-only model (no backbone conversion, no training).
    extract_from: Optional[str] = None
    # Optional class-label file for --extract_head (one label per line). When
    # omitted, a sibling ``<model>_Labels.txt`` is used if present.
    labels_file: Optional[str] = None
    # Output precisions to export: subset of {"FP32", "FP16", "INT8"}.
    # None → defaults to [foundation_model.data_type]. Each precision yields a
    # separate exported file tagged with the precision in its filename. INT8 uses
    # dynamic/weight-only quantization (no calibration dataset).
    data_types: Optional[list[str]] = None


@dataclass
class UmapConfig:
    """UMAP projection parameters used by the ``--embeddings`` CLI mode.

    Disabled by default: UMAP and plotting require the optional ``[umap]``
    extra (``pip install bioaccx[umap]``). Set ``enabled: true`` to compute
    the projection and write the UMAP data CSV and scatter-plot PNG.
    """
    # Fit the projection and write the CSV and plots (needs the [umap] extra)
    enabled: bool = False
    # Neighbourhood size: low = local structure, high = global structure
    n_neighbors: int = 15
    min_dist: float = 0.1               # how tightly points may be packed
    n_components: int = 2               # dimensions of the projection
    metric: str = "euclidean"           # distance metric between embeddings
    # Random seed for UMAP; falls back to dataset.random_seed when None.
    random_seed: Optional[int] = None
    # Path to a previously written UMAP data CSV. When set and the file exists,
    # the embeddings run skips all computation (dataset load, embedding
    # extraction, KMeans, UMAP fit) and only redraws the plots from the cached
    # coordinates — useful for tweaking plot styling without recomputing.
    cache_csv: Optional[str] = None


# Dataset fields that describe the whole run rather than a single source.
# When ``dataset.sources`` is used, these are taken from the top-level
# ``dataset`` block only; any per-source override of them is ignored so that
# the run has a single, unambiguous split / seed / append target / credentials.
RUN_LEVEL_DATASET_FIELDS = frozenset({
    "test_ratio",
    "random_seed",
    "append_dataset_path",
    "embedding_workers",
    "embeddings_cache_path",
    "ext_cache_dir",
    "xc_api_key",
    "arbimon_credentials_path",
    "audio_extensions",
})


@dataclass
class BioaccxConfig:
    foundation_model: FoundationModelConfig
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    umap: UmapConfig = field(default_factory=UmapConfig)
    # Resolved per-source dataset blocks (built from ``dataset.sources``).
    # None for the common single-source case; use the ``dataset_blocks``
    # property instead of reading this directly.
    _dataset_blocks: Optional[list[DatasetConfig]] = field(default=None, repr=False)

    @property
    def dataset_blocks(self) -> list[DatasetConfig]:
        """Per-source dataset blocks to load and merge for this run.

        Returns the resolved ``dataset.sources`` blocks when present, otherwise
        a single-element list holding the top-level ``dataset`` block. Callers
        can always iterate this without special-casing the single-source path.
        Run-level settings (split, seed, append, credentials) always live on
        ``self.dataset``; see :data:`RUN_LEVEL_DATASET_FIELDS`.
        """
        return self._dataset_blocks if self._dataset_blocks else [self.dataset]

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

    @property
    def output_data_types(self) -> list[str]:
        """Resolved output precisions, upper-cased and de-duplicated (first-seen order).

        Defaults to ``[foundation_model.data_type]`` when ``output.data_types`` is
        unset. Raises ``ValueError`` on any precision outside
        :data:`VALID_OUTPUT_DATA_TYPES`.
        """
        raw = self.output.data_types or [self.foundation_model.data_type]
        resolved: list[str] = []
        for dt in raw:
            up = dt.upper()
            if up not in VALID_OUTPUT_DATA_TYPES:
                raise ValueError(
                    f"Unknown output data_type {dt!r}. "
                    f"Valid options: {', '.join(VALID_OUTPUT_DATA_TYPES)}."
                )
            if up not in resolved:
                resolved.append(up)
        return resolved


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

    ds, ds_blocks = _parse_dataset_section(dict(data.get("dataset", {})))

    # Pop nested trainer configs before passing the remainder to TrainingConfig.
    tr_raw = dict(data.get("training", {}))
    keras_raw = tr_raw.pop("keras", {})
    sklearn_raw = tr_raw.pop("sklearn", {})
    mixup_raw = tr_raw.pop("audio_mixup", None)
    tr = TrainingConfig(
        **{k: v for k, v in tr_raw.items() if k in {"classifier"}},
        keras=_from_dict(KerasConfig, keras_raw),
        sklearn=_from_dict(SklearnConfig, sklearn_raw),
        audio_mixup=_parse_audio_mixup(mixup_raw) if mixup_raw is not None else None,
    )

    out = _from_dict(OutputConfig, data.get("output", {}))

    umap_cfg = _from_dict(UmapConfig, data.get("umap", {}))

    return BioaccxConfig(
        foundation_model=fm, dataset=ds, training=tr, output=out, umap=umap_cfg,
        _dataset_blocks=ds_blocks,
    )


def _parse_audio_mixup(raw: dict) -> AudioMixupConfig:
    """Build and sanity-check a ``training.audio_mixup`` block."""
    mix = _from_dict(AudioMixupConfig, raw or {})
    if mix.max_sources not in (2, 3):
        raise ValueError(f"audio_mixup.max_sources must be 2 or 3, got {mix.max_sources}")
    if not 0.0 <= mix.p_three_sources <= 1.0:
        raise ValueError("audio_mixup.p_three_sources must be in [0, 1]")
    if len(mix.snr_db) != 2 or mix.snr_db[0] > mix.snr_db[1]:
        raise ValueError(f"audio_mixup.snr_db must be [min, max], got {mix.snr_db}")
    if mix.n_mixes is None and mix.ratio <= 0:
        raise ValueError("audio_mixup needs n_mixes or a positive ratio")
    if mix.test_mix_ratio < 0:
        raise ValueError("audio_mixup.test_mix_ratio must be >= 0")
    seen: dict[str, str] = {}
    for group, members in mix.exclusive_groups.items():
        for label in members:
            if label in seen:
                raise ValueError(f"audio_mixup.exclusive_groups: {label!r} is in both "
                                 f"{seen[label]!r} and {group!r}")
            seen[label] = group
    return mix


def _parse_dataset_block(ds_raw: dict) -> DatasetConfig:
    """Build a single DatasetConfig from a raw dict, handling the nested
    ``augmentation`` block (which _from_dict cannot construct recursively).

    An explicit ``augmentation: null`` yields a block with no augmentation,
    which is how a per-source block opts out of an inherited augmentation.
    """
    ds_raw = dict(ds_raw)
    aug_raw = ds_raw.pop("augmentation", None)
    ds = _from_dict(DatasetConfig, ds_raw)
    if aug_raw is not None:
        ds.augmentation = _from_dict(AugmentationConfig, aug_raw)
        if not ds.augmentation.augmentation_dir and not ds.augmentation.augmentation_labels:
            raise ValueError(
                "dataset.augmentation requires 'augmentation_dir' and/or "
                "'augmentation_labels'"
            )
    return ds


def _parse_dataset_section(ds_raw: dict) -> tuple[DatasetConfig, Optional[list[DatasetConfig]]]:
    """Parse the ``dataset`` config section into (run_level, blocks).

    Without a ``sources`` key this returns the single parsed block and ``None``
    (the legacy single-source layout, fully unchanged).

    With ``sources`` (a list of partial dataset dicts), each source inherits the
    top-level ``dataset`` fields and overrides them with its own. Run-level
    fields (:data:`RUN_LEVEL_DATASET_FIELDS`) are never overridden per source —
    they always come from the top-level block — so splitting, seeding, append
    and credentials stay consistent across the run. The returned run-level
    DatasetConfig (built from the top-level fields without ``sources``) carries
    those settings; the blocks carry the per-source loading parameters.
    """
    ds_raw = dict(ds_raw)
    sources = ds_raw.pop("sources", None)

    run_level = _parse_dataset_block(ds_raw)
    if not sources:
        return run_level, None

    if not isinstance(sources, list):
        raise ValueError("dataset.sources must be a list of source blocks")

    blocks: list[DatasetConfig] = []
    for i, src in enumerate(sources):
        if not isinstance(src, dict):
            raise ValueError(f"dataset.sources[{i}] must be a mapping")
        merged = dict(ds_raw)  # inherit top-level (incl. any shared augmentation)
        for k, v in src.items():
            if k in RUN_LEVEL_DATASET_FIELDS:
                continue  # run-level only; ignore per-source override
            merged[k] = v
        # append_dataset_path is inherited (run-level) so each block can
        # de-duplicate its new samples against the existing exported dataset,
        # but the existing samples themselves are loaded and prepended only once
        # at the run level (see train.load_and_prepare_blocks).
        blocks.append(_parse_dataset_block(merged))

    return run_level, blocks
