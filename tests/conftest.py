"""Shared pytest fixtures for bioaccx tests.

Hierarchy
---------
dummy_onnx_model  — simple random MatMul ONNX embedder (fast inference, for
                    unit / integration tests where accuracy doesn't matter)
dft_onnx_model    — phase-invariant spectral-energy ONNX embedder (for the
                    realistic test where classifiers must actually learn)

Audio datasets
--------------
subfolders_dataset        — 3 classes × 6 WAV files in label subfolders
predefined_split_dataset  — same but with train/ and test/ top-level dirs
file_per_label_dataset    — WAV + paired .txt annotation file per audio file
table_dataset             — WAV files + a single CSV annotation table
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

# ---- shared constants -------------------------------------------------------

SAMPLE_RATE = 48_000
WINDOW_SAMPLES = 4_800   # 0.1 s — fast inference
WINDOW_SECONDS = WINDOW_SAMPLES / SAMPLE_RATE
EMBED_DIM = 16

# Class frequencies (Hz) — very different so the DFT embedder separates them
CLASS_FREQS = {"bird": 440, "frog": 2_000, "background": 0}


# ---- low-level helpers ------------------------------------------------------

def make_sine_wav(path: Path, frequency: int, duration: float,
                  sample_rate: int = SAMPLE_RATE,
                  noise_std: float = 0.02, seed: int | None = None) -> Path:
    """Write a sine wave (+ optional noise) WAV file; frequency=0 → pure noise."""
    rng = np.random.default_rng(seed)
    n = int(duration * sample_rate)
    t = np.arange(n) / sample_rate
    if frequency == 0:
        audio = rng.standard_normal(n).astype(np.float32) * 0.3
    else:
        audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)
        audio += rng.standard_normal(n).astype(np.float32) * noise_std
    sf.write(str(path), audio, sample_rate)
    return path


def _make_random_onnx_embedder(path: Path, window_samples: int = WINDOW_SAMPLES,
                                embed_dim: int = EMBED_DIM,
                                input_name: str = "INPUT",
                                output_name: str = "embedding") -> Path:
    """Build a minimal MatMul ONNX model: output = input @ W."""
    import onnx
    from onnx import TensorProto, helper
    from onnx import numpy_helper as nph

    rng = np.random.default_rng(0)
    W = (rng.standard_normal((window_samples, embed_dim)) * 0.01).astype(np.float32)

    W_init = nph.from_array(W, name="W")
    inp = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [None, window_samples])
    out = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [None, embed_dim])
    node = helper.make_node("MatMul", inputs=[input_name, "W"], outputs=[output_name])
    graph = helper.make_graph([node], "random_embedder", [inp], [out], initializer=[W_init])

    model = helper.make_model(graph)
    model.ir_version = 8
    op = model.opset_import[0]
    op.domain = ""
    op.version = 13

    onnx.save(model, str(path))
    return path


def _make_dft_onnx_embedder(path: Path, window_samples: int = WINDOW_SAMPLES,
                              embed_dim: int = EMBED_DIM,
                              input_name: str = "INPUT",
                              output_name: str = "embedding") -> Path:
    """Build a phase-invariant spectral-energy ONNX model.

    Computes sqrt(|DFT_real[k]|^2 + |DFT_imag[k]|^2) for *embed_dim* frequency
    bins. Bins are chosen to include the test-class frequencies exactly:
      k=44  → 440 Hz  (bird class, sr=48000, window=4800)
      k=200 → 2000 Hz (frog class)
    This makes features phase-invariant and linearly separable across classes.
    """
    import onnx
    from onnx import TensorProto, helper
    from onnx import numpy_helper as nph

    N = window_samples
    n = np.arange(N, dtype=np.float32)

    # Handcrafted bins that explicitly include 440 Hz (k=44) and 2000 Hz (k=200)
    # at sr=48000, window=4800 (bin_freq = k * sr / N = k * 10 Hz)
    assert embed_dim == 16, "DFT embedder bins are designed for embed_dim=16"
    bins = np.array([1, 2, 5, 10, 22, 44, 88, 100, 150, 200, 300, 400, 500, 600, 800, 1000])

    cos_mat = np.stack([np.cos(2 * np.pi * k * n / N) / N for k in bins], axis=1).astype(np.float32)
    sin_mat = np.stack([np.sin(2 * np.pi * k * n / N) / N for k in bins], axis=1).astype(np.float32)

    cos_init = nph.from_array(cos_mat, name="cos_matrix")
    sin_init = nph.from_array(sin_mat, name="sin_matrix")
    inp = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [None, N])
    out = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [None, embed_dim])

    nodes = [
        helper.make_node("MatMul", [input_name, "cos_matrix"], ["cos_part"]),
        helper.make_node("MatMul", [input_name, "sin_matrix"], ["sin_part"]),
        helper.make_node("Mul", ["cos_part", "cos_part"], ["cos_sq"]),
        helper.make_node("Mul", ["sin_part", "sin_part"], ["sin_sq"]),
        helper.make_node("Add", ["cos_sq", "sin_sq"], ["sum_sq"]),
        helper.make_node("Sqrt", ["sum_sq"], [output_name]),
    ]
    graph = helper.make_graph(nodes, "dft_embedder", [inp], [out],
                              initializer=[cos_init, sin_init])

    model = helper.make_model(graph)
    model.ir_version = 8
    op = model.opset_import[0]
    op.domain = ""
    op.version = 13

    onnx.save(model, str(path))
    return path


# ---- session-scoped model fixtures ------------------------------------------

@pytest.fixture(scope="session")
def dummy_onnx_model(tmp_path_factory):
    path = tmp_path_factory.mktemp("models") / "dummy_embedder.onnx"
    return _make_random_onnx_embedder(path)


@pytest.fixture(scope="session")
def dft_onnx_model(tmp_path_factory):
    path = tmp_path_factory.mktemp("models") / "dft_embedder.onnx"
    return _make_dft_onnx_embedder(path)


@pytest.fixture(scope="session")
def foundation_cfg(dummy_onnx_model):
    from bioaccx.config import FoundationModelConfig
    return FoundationModelConfig(
        name="dummy", version="0.1",
        format="onnx", source="local", path=str(dummy_onnx_model),
        sample_rate=SAMPLE_RATE, window_samples=WINDOW_SAMPLES,
        input_name="INPUT", output_name="embedding", embedding_size=EMBED_DIM,
    )


@pytest.fixture(scope="session")
def dft_foundation_cfg(dft_onnx_model):
    from bioaccx.config import FoundationModelConfig
    return FoundationModelConfig(
        name="dummy_dft", version="0.1",
        format="onnx", source="local", path=str(dft_onnx_model),
        sample_rate=SAMPLE_RATE, window_samples=WINDOW_SAMPLES,
        input_name="INPUT", output_name="embedding", embedding_size=EMBED_DIM,
    )


# ---- session-scoped dataset fixtures ----------------------------------------

@pytest.fixture(scope="session")
def subfolders_dataset(tmp_path_factory):
    """3 classes × 6 WAV files in <root>/<class>/ subfolders (0.3 s each)."""
    root = tmp_path_factory.mktemp("subfolders_ds")
    for i, (cls, freq) in enumerate(CLASS_FREQS.items()):
        cls_dir = root / cls
        cls_dir.mkdir()
        for j in range(6):
            make_sine_wav(cls_dir / f"{cls}_{j:02d}.wav", freq, 0.3, seed=i * 100 + j)
    return root


@pytest.fixture(scope="session")
def predefined_split_dataset(tmp_path_factory):
    """2 classes with predefined train/test top-level split."""
    root = tmp_path_factory.mktemp("split_ds")
    classes = {"bird": 440, "frog": 2_000}
    for si, (split, n) in enumerate([("train", 6), ("test", 2)]):
        for ci, (cls, freq) in enumerate(classes.items()):
            d = root / split / cls
            d.mkdir(parents=True)
            for j in range(n):
                make_sine_wav(d / f"{cls}_{j:02d}.wav", freq, 0.3, seed=si * 200 + ci * 10 + j)
    return root


@pytest.fixture(scope="session")
def file_per_label_dataset(tmp_path_factory):
    """3 classes × 5 audio files; each paired with a .txt annotation (one row)."""
    root = tmp_path_factory.mktemp("fpl_ds")
    for ci, (cls, freq) in enumerate(CLASS_FREQS.items()):
        for j in range(5):
            wav = root / f"{cls}_{j:02d}.wav"
            txt = root / f"{cls}_{j:02d}.txt"
            make_sine_wav(wav, freq, 0.5, seed=ci * 50 + j)
            txt.write_text(f"0.0\t0.3\t{cls}\n")
    return root


@pytest.fixture(scope="session")
def file_per_label_dataset_multi_row(tmp_path_factory):
    """2 classes × 3 audio files; each .txt has 2 annotation rows."""
    root = tmp_path_factory.mktemp("fpl_multi_ds")
    classes = {"bird": 440, "frog": 2_000}
    for ci, (cls, freq) in enumerate(classes.items()):
        for j in range(3):
            wav = root / f"{cls}_{j:02d}.wav"
            txt = root / f"{cls}_{j:02d}.txt"
            make_sine_wav(wav, freq, 1.0, seed=ci * 30 + j)
            txt.write_text(f"0.0\t0.4\t{cls}\n0.5\t0.9\t{cls}\n")
    return root


@pytest.fixture(scope="session")
def table_dataset(tmp_path_factory):
    """3 classes × 5 audio files annotated in a CSV table."""
    root = tmp_path_factory.mktemp("table_ds")
    rows = ["filename,label,start_time,end_time"]
    for ci, (cls, freq) in enumerate(CLASS_FREQS.items()):
        for j in range(5):
            fname = f"{cls}_{j:02d}.wav"
            make_sine_wav(root / fname, freq, 0.5, seed=ci * 50 + j)
            rows.append(f"{fname},{cls},0.0,0.3")
    table_path = root / "annotations.csv"
    table_path.write_text("\n".join(rows) + "\n")
    return root, table_path


@pytest.fixture(scope="session")
def table_dataset_with_split(tmp_path_factory):
    """Table dataset where the CSV already encodes train/test split."""
    root = tmp_path_factory.mktemp("table_split_ds")
    classes = {"bird": 440, "frog": 2_000}
    rows = ["filename,label,start_time,end_time,split"]
    for ci, (cls, freq) in enumerate(classes.items()):
        for j in range(8):
            fname = f"{cls}_{j:02d}.wav"
            make_sine_wav(root / fname, freq, 0.5, seed=ci * 80 + j)
            split = "train" if j < 6 else "test"
            rows.append(f"{fname},{cls},0.0,0.3,{split}")
    table_path = root / "annotations.csv"
    table_path.write_text("\n".join(rows) + "\n")
    return root, table_path
