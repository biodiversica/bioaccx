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
    # Model name, used in reports and output filenames.
    name: str
    version: str = "unknown"            # Model version string.
    # Weight precision, for example FP32 or INT8.
    data_type: str = "FP32"
    # Model file format on disk.
    format: Literal["onnx", "tflite", "protobuf"] = "onnx"
    # Where the model is loaded from: a local path, the Hugging Face Hub, or Kaggle.
    source: Literal["local", "huggingface", "kaggle"] = "local"
    path: Optional[str] = None          # Path to the local model file or directory.
    # Hugging Face repository, for example "biodiversica/birdnet".
    hf_repo: Optional[str] = None
    # File to download from the Hugging Face repository.
    hf_filename: Optional[str] = None
    # Branch, tag or commit of the Hugging Face repository.
    hf_revision: Optional[str] = None
    # Kaggle model handle, for example
    # "google/bird-vocalization-classifier/tensorFlow2/bird-vocalization-classifier".
    kaggle_handle: Optional[str] = None
    # File to use within the downloaded Kaggle directory (ONNX or TFLite).
    kaggle_filename: Optional[str] = None
    # Sample rate the backbone expects, in Hz.
    sample_rate: int = 48000
    # Window length in samples. Takes priority over window_seconds.
    window_samples: Optional[int] = None
    # Window length in seconds, for example 3.0.
    window_seconds: Optional[float] = None
    input_name: str = "input"           # Name of the audio input tensor.
    output_name: str = "embedding"      # Name of the embedding output tensor.
    embedding_size: int = 1024          # Dimensionality of the embedding vector.
    # ONNX Runtime execution providers, for example
    # ["CUDAExecutionProvider", "CPUExecutionProvider"]. When unset, ONNX Runtime
    # uses its own provider priority.
    onnx_providers: Optional[list[str]] = None
    # Number of audio windows processed in a single ONNX inference call. Values
    # above 1 enable GPU batch mode. Applies only when format is "onnx".
    onnx_batch_size: int = 1
    # Offset applied to the resolved TFLite output tensor index to reach the
    # embedding tensor. Use -1 for models such as the BirdNET TFLite model, where
    # the classifier head is the first output and the embedding sits one position
    # before it in the graph's tensor list.
    tflite_output_tensor_offset: int = 0
    # Trim a TFLite graph down to the embedding tensor when the model computes
    # more than that, such as a bundled classifier head or auxiliary outputs.
    # This is done once at load time, and every later window then skips the
    # discarded branches. Set to false to run the model exactly as shipped.
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
    # SNR values in dB. One augmented copy is made per noise file per level.
    snr_levels: list[float]
    # Directory of WAV files used as noise sources. Optional when
    # augmentation_labels is set; if both are given, the noise pool is their union.
    augmentation_dir: Optional[str] = None
    # Labels of the dataset being created whose audio is used as additional
    # noise sources, alongside any augmentation_dir. Samples of these labels are
    # mixed into the augmentation of other samples but are never augmented
    # themselves, as with skip_labels. They remain trainable classes.
    augmentation_labels: Optional[list[str]] = None
    # Keep the clean sample alongside its augmented copies.
    keep_original: bool = True
    # Augment the test set as well as the training set.
    augment_test: bool = False
    # Labels that are not augmented. They remain trainable classes.
    skip_labels: Optional[list[str]] = None
    # When true, all files in augmentation_dir are concatenated into a single
    # in-memory array used as the only noise source. If an Audacity .txt label
    # file sits next to an audio file, only its labelled segments are used;
    # otherwise the entire file is included.
    concatenate_augmentation_dir: bool = False
    # When true, the files in augmentation_dir are shuffled once (using
    # random_seed) and assigned in rotation to augmented samples: one file per
    # (sample, SNR) pair, without repetition within each pass. Like
    # concatenate_augmentation_dir, this produces a single noise condition, but
    # it draws from individual files rather than a merged track.
    random_augmentation_dir: bool = False


@dataclass
class DatasetConfig:
    # Local audio source or sources. May be empty when only ext_table_file is used.
    data_dir: str | list[str] = field(default_factory=list)
    # How the labels are organised in data_dir.
    label_mode: Literal["subfolders", "table", "file_per_label"] = "subfolders"
    # CSV or TSV label table, used when label_mode is "table".
    table_file: Optional[str] = None
    # Accepted audio file extensions (not case-sensitive).
    audio_extensions: list[str] = field(
        default_factory=lambda: ["wav", "flac", "mp3", "ogg"]
    )
    # Overlap between consecutive windows when label segments are chunked,
    # from 0.0 to 1.0.
    overlap: float = 0.0
    # Parallel workers for embedding extraction, capped at the available CPU cores.
    embedding_workers: int = 4
    # Path searched for pre-computed .npy embeddings before the model is run.
    embeddings_cache_path: Optional[str] = None
    # Path to an existing exported dataset (subfolders layout with train and
    # test). New samples from data_dir that are not already present are appended
    # to it.
    append_dataset_path: Optional[str] = None
    # Fraction of the data held out for testing. Ignored when the dataset
    # already defines the split.
    test_ratio: float = 0.2
    # Seed for the split, shuffling and noise offsets.
    random_seed: int = 42

    # Column names used in the table and ext_table_file modes.

    filename_col: str = "filename"      # Column holding the audio filename.
    label_col: str = "label"            # Column holding the class label.
    start_col: str = "start_time"       # Column holding the segment start, in seconds.
    end_col: str = "end_time"           # Column holding the segment end, in seconds.
    # Optional column holding the split, with the values "train" or "test".
    split_col: str = "split"
    # iNaturalist table: a CSV or TSV file of observation_id rows, optionally
    # mixed with filename rows.
    ext_table_file: Optional[str] = None
    obs_id_col: str = "observation_id"  # Column holding the iNaturalist observation ID.
    # Column holding the iNaturalist sound index (zero-based).
    sound_index_col: str = "sound_index"
    xc_id_col: str = "xc_id"            # Column holding the Xeno-canto recording ID.
    # Local cache for downloaded remote audio. Defaults to ~/.cache/bioaccx/ext.
    ext_cache_dir: Optional[str] = None
    # Xeno-canto API v3 key, required for metadata (scientific name lookup).
    # Audio can be downloaded without a key through the direct download URL.
    # Keys are issued at https://xeno-canto.org/explore/api.
    xc_api_key: Optional[str] = None
    # Arbimon (RFCx) credentials file, as written by
    # rfcx.Client().authenticate(persisted_credentials_path=...).
    arbimon_credentials_path: Optional[str] = None

    # Column names used to identify Arbimon rows in ext_table_file.

    # Column holding the Arbimon stream (site) ID.
    arbimon_stream_id_col: str = "stream_id"
    arbimon_date_col: str = "date"      # Column holding the local recording date.
    # Column holding the local recording start time.
    arbimon_time_col: str = "time"
    # Column holding the UTC offset of the recording date and time.
    arbimon_utc_offset_col: str = "utc_offset"

    # Audio preprocessing applied before chunking (filter, then speed, then chunks).

    # Filter applied to the audio before chunking: "hpf" (high-pass), "lpf"
    # (low-pass), "bpf" (band-pass) or null.
    filter: Optional[str] = None
    # Cut-off frequency in Hz for hpf and lpf; [low_hz, high_hz] for bpf.
    filter_freq: Optional[float | list[float]] = None
    filter_order: int = 5               # Butterworth filter order.
    # Playback speed multiplier: above 1 is faster and shorter, below 1 slower
    # and longer. Label times are scaled to match: new_time = old_time / speed.
    speed: float = 1.0

    # SSH/SFTP access.

    # SSH server holding the audio. When set, data_dir paths are treated as
    # remote paths on this server and mirrored locally with paramiko before the
    # pipeline runs. Requires paramiko (pip install paramiko).
    ssh_host: Optional[str] = None
    # SSH user name. Defaults to the local user.
    ssh_user: Optional[str] = None
    ssh_port: int = 22                  # SSH port.
    ssh_key_path: Optional[str] = None  # Path to the private key file.
    augmentation: Optional[AugmentationConfig] = None
    # When true, samples shorter than the foundation model window are placed at
    # a random offset within the window instead of always starting at position 0.
    # Each augmented copy of the same sample gets a different offset.
    random_sample_shift: bool = False
    # Minimum fraction of the window that must be new (uncovered) audio for the
    # anchor chunk at the end of a file to be emitted. This prevents
    # near-duplicate chunks when files are only slightly longer than the window
    # (for example, 3.013 s with a 3 s window). Set to 0.0 to always emit the
    # anchor chunk.
    min_anchor_fraction: float = 0.1


@dataclass
class KerasConfig:
    # Units in the hidden Dense layer. 0 means no hidden layer (a linear classifier).
    hidden_units: int = 256
    dropout: float = 0.25               # Dropout rate before each Dense layer.
    # Maximum number of epochs. Early stopping may end training sooner.
    epochs: int = 50
    batch_size: int = 32                # Mini-batch size.
    # Peak Adam learning rate (cosine decay with a linear warm-up).
    learning_rate: float = 1e-4
    # Output activation: null (trained with softmax, exports logits; the
    # default), "sigmoid", "softmax" or "grouped_softmax" (requires label_groups).
    output_activation: Optional[str] = None
    # Grouped softmax: a softmax within each group, with the groups independent
    # of each other. Maps each group name to its member labels, and each group
    # gains a synthetic "<group>_none" output column. Members of one group are
    # mutually exclusive; members of different groups can be predicted together.
    # Training labels not listed in any group are background: they get no
    # output column and supply the "none" target for every group.
    label_groups: dict[str, list[str]] = field(default_factory=dict)
    # Remove the output activation layer before export, so that the exported
    # head emits raw logits while training still used output_activation (as
    # BirdNET-Analyzer does with classifier.pop()). Has no effect when
    # output_activation is null. Callers must apply the activation themselves.
    # Rejected together with exclude_labels on a softmax or grouped head, where
    # the excluded logits are needed to rebuild the softmax (output_activation
    # null with exclude_labels is rejected for the same reason).
    export_logits: bool = False
    # Z-score normalisation of the input embeddings, fitted on the training set.
    normalize_embeddings: bool = True
    # Use focal loss instead of cross-entropy.
    focal_loss: bool = False
    focal_loss_gamma: float = 2.0       # Focal loss focusing parameter γ.
    focal_loss_alpha: float = 0.25      # Focal loss class-balance parameter α.
    # Apply label smoothing to the one-hot targets before training.
    label_smoothing: bool = False
    # Amount subtracted from positive labels and redistributed to the negatives.
    label_smoothing_alpha: float = 0.1
    # Apply mixup data augmentation to the training embeddings.
    mixup: bool = False
    mixup_ratio: float = 0.25           # Fraction of positive samples to mix.
    # Beta distribution parameter of the mixing coefficient.
    mixup_alpha: float = 0.2
    # Upsampling ratio for minority classes before training.
    upsampling_ratio: float = 0.0
    # Upsampling method: repeat (random duplication), mean (pairwise mean),
    # linear (random interpolation) or smote (k-NN interpolation).
    upsampling_mode: Literal["repeat", "mean", "linear", "smote"] = "repeat"
    # Set the TensorFlow and NumPy random seeds from dataset.random_seed before
    # training, for reproducibility. Set to false to disable; training results
    # will then vary from run to run.
    seed: bool = True


@dataclass
class SklearnConfig:
    C: float = 1.0                      # Inverse regularisation strength.
    max_iter: int = 2000                # Maximum number of solver iterations.
    solver: str = "lbfgs"               # LogisticRegression solver algorithm.


@dataclass
class AudioMixupConfig:
    # Mix the audio of training windows from different classes, embed the
    # mixture and train on the union of their labels. Requires a Keras sigmoid
    # head.
    enabled: bool = True
    # Labels whose windows may be mixed. Windows of any other label are never
    # used as a source, although they still train as real samples. Empty means
    # every label.
    labels: list[str] = field(default_factory=list)
    # Number of mixes. Overrides ratio when set.
    n_mixes: Optional[int] = None
    # Number of mixes as a fraction of the real training windows.
    ratio: float = 0.5
    # Maximum number of sources per mix: 2, or 3 (see p_three_sources).
    max_sources: int = 2
    # Proportion of three-source mixes when max_sources is 3.
    p_three_sources: float = 0.0
    # Level range [min, max] in dB of each added source, relative to the first.
    snr_db: list[float] = field(default_factory=lambda: [-6.0, 6.0])
    # balanced: select a class uniformly, then a clip from it; uniform: select
    # clips uniformly.
    pairing: Literal["balanced", "uniform"] = "balanced"
    # Groups of mutually exclusive labels (for example, call types or intensity
    # levels of one species) that are never mixed with each other.
    exclusive_groups: dict[str, list[str]] = field(default_factory=dict)
    # Labels that represent background rather than species. A mix contains at
    # most one of them, and always at least one non-background source.
    background_labels: list[str] = field(default_factory=list)
    # drop: a background source adds no label to the mix target; include: it
    # adds its own label.
    background_target: Literal["drop", "include"] = "drop"
    # Seed for drawing the mixes. Defaults to dataset.random_seed.
    seed: Optional[int] = None
    # Also build mixes from test windows only, as a fraction of the test windows
    # that may be mixed (0 means none). They are reported in a separate section
    # and never pooled with the real test set.
    test_mix_ratio: float = 0.0


@dataclass
class TrainingConfig:
    # Which classifier head to train: keras, sklearn, or both.
    classifier: Literal["keras", "sklearn", "both"] = "keras"
    keras: KerasConfig = field(default_factory=KerasConfig)
    sklearn: SklearnConfig = field(default_factory=SklearnConfig)
    # Audio-level mixup for multi-label sigmoid heads.
    audio_mixup: Optional[AudioMixupConfig] = None


@dataclass
class OutputConfig:
    output_path: str = "./outputs"      # Parent directory for all output.
    # Model name, used in filenames and for the output subdirectory.
    model_name: str = "custom_classifier"
    model_version: str = "1.0"          # Version string used in filenames.
    # What to export: head (the classifier only), full (backbone and classifier
    # combined) or both.
    output_type: Literal["head", "full", "both"] = "head"
    # File format of the exported model.
    output_format: Literal["onnx", "tflite", "both"] = "onnx"
    # Labels to exclude from the exported classifier output. They are still
    # used during training. Requires a Keras head that exports probabilities or
    # per-class logits, so it is rejected with output_activation null, or with
    # softmax and export_logits.
    exclude_labels: list[str] = field(default_factory=list)
    # Export the chunked audio samples as WAV files in label subfolders.
    export_dataset: bool = False
    # Export only the audio mixes (training.audio_mixup) as WAV files in
    # dataset/mixes/, for checking them when the real windows already exist.
    export_mixes: bool = False
    # Save the computed embeddings for reuse.
    export_embeddings: bool = False
    # Storage format for exported and cached embeddings: npy (one file per
    # sample) or sqlite (a single, more portable .db file per run).
    embeddings_format: Literal["npy", "sqlite"] = "npy"
    # Directory (npy) or file path (sqlite) for exported embeddings. Defaults
    # to <output_dir>/embeddings or <output_dir>/embeddings.db.
    embeddings_path: Optional[str] = None
    # Applies to the --embeddings mode only. When an embedding store already
    # exists at the resolved path, recompute and overwrite it. When false (the
    # default), the existing store is reused: stored embeddings are never
    # recomputed, and only the samples missing from it are computed and added.
    embeddings_overwrite: bool = False
    # Path to an existing ONNX classifier head for --merge (no training required).
    head_path: Optional[str] = None
    # Path to a full BirdNET-Analyzer model (.tflite) for --extract_head. The
    # trailing classifier head is read from the flatbuffer and re-exported as a
    # head-only model, without backbone conversion or training.
    extract_from: Optional[str] = None
    # Optional class-label file for --extract_head, with one label per line.
    # When omitted, a ``<model>_Labels.txt`` file next to the model is used if
    # present.
    labels_file: Optional[str] = None
    # Output precisions to export: a subset of {"FP32", "FP16", "INT8"}. When
    # unset, defaults to [foundation_model.data_type]. Each precision produces a
    # separate exported file, tagged with the precision in its filename. INT8
    # uses dynamic (weight-only) quantisation, with no calibration dataset.
    data_types: Optional[list[str]] = None


@dataclass
class UmapConfig:
    """UMAP projection parameters used by the ``--embeddings`` CLI mode.

    Disabled by default: UMAP and plotting require the optional ``[umap]``
    extra (``pip install bioaccx[umap]``). Set ``enabled: true`` to compute
    the projection and write the UMAP data CSV and scatter-plot PNG.
    """
    # Fit the projection and write the CSV and plots. Requires the [umap] extra.
    enabled: bool = False
    # Neighbourhood size: low values preserve local structure, high values
    # global structure.
    n_neighbors: int = 15
    min_dist: float = 0.1               # How tightly points may be packed.
    n_components: int = 2               # Number of dimensions of the projection.
    metric: str = "euclidean"           # Distance metric between embeddings.
    # Random seed for UMAP. Falls back to dataset.random_seed when unset.
    random_seed: Optional[int] = None
    # Path to a previously written UMAP data CSV. When set and the file exists,
    # the embeddings run skips all computation (dataset loading, embedding
    # extraction, KMeans and the UMAP fit) and only redraws the plots from the
    # cached coordinates. This is useful for adjusting plot styling without
    # recomputing.
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
    text = p.read_text(encoding="utf-8-sig")
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
