# Installation

bioaccx runs models through ONNX Runtime, and **there is no default one** — pick
`cpu` or `gpu` when you install. A plain `bioaccx` resolves and installs
happily, then fails the moment it tries to run a model, so the extra is not
optional in practice.

## As a command-line tool

The usual way to use bioaccx if you are not developing it. `uv tool install`
gives it an isolated environment and puts `bioaccx` on your PATH:

```bash
uv tool install "bioaccx[cpu]"              # CPU inference
uv tool install "bioaccx[gpu]"              # CUDA 12 (see GPU section below)
uv tool install "bioaccx[cpu,gui,umap]"     # with the browser GUI and UMAP plots
```

`uv tool install` has **no `--extra` flag** — extras go inside the brackets, and
the whole argument needs quoting so the shell does not try to glob them.

```bash
uv tool install "/path/to/bioaccx[cpu]"     # from a local checkout
uv tool upgrade bioaccx                     # later
uv tool uninstall bioaccx
```

## In a project

```bash
uv add "bioaccx[cpu]"                       # as a dependency
pip install "bioaccx[cpu]"                  # or with pip
```

## Working on bioaccx itself

```bash
git clone git@github.com:biodiversica/bioaccx.git
cd bioaccx
uv sync --extra cpu                         # or --extra gpu
uv sync --extra cpu --extra gui --extra umap
uv run bioaccx --help
uv run pytest
```

`uv sync --extra X` **replaces** the active extra set rather than adding to it,
so name every extra you want on each run — syncing with `--extra gui` alone will
remove `onnxruntime` and `umap-learn` again.

## Extras

| Extra | Adds | When needed |
|---|---|---|
| `cpu` | `onnxruntime` | Running any model on CPU. Pick this or `gpu`. |
| `gpu` | `onnxruntime-gpu` | Running on CUDA 12. Pick this or `cpu`. |
| `gui` | `fastapi`, `uvicorn`, `ruamel.yaml` | `bioaccx gui` — the browser config editor and results explorer |
| `umap` | `umap-learn`, `matplotlib` | The UMAP projection and plots from `bioaccx embeddings` (`umap.enabled: true`) |
| `ssh` | `paramiko` | Remote `data_dir` over SSH/SFTP (already a core dependency; the extra is kept for older configs) |
| `arbimon` | — | Arbimon sources. The `rfcx` SDK is not on PyPI; install the release wheel by hand: `pip install https://github.com/rfcx/rfcx-sdk-python/releases/download/0.3.1/rfcx-0.3.1-py3-none-any.whl` |

TensorFlow, scikit-learn, ONNX, HuggingFace Hub and Kaggle Hub are **core**
dependencies — they are always installed, and need no extra.

`cpu` and `gpu` are declared mutually exclusive, but that is only enforced when
syncing this project; installing `bioaccx[cpu,gpu]` as a tool or a dependency
will happily give you both runtimes. Ask for one.

## Disk space

The environment is large: TensorFlow alone is about 1.3 GB, and a checkout
synced with `cpu`, `gui`, `umap` and the dev group comes to roughly 2.2 GB. A
`uv tool install` gets its own copy of all of it.

## GPU acceleration (ONNX Runtime + CUDA)

To run the ONNX backbone on a GPU, set `onnx_providers` in the config:

```yaml
foundation_model:
  onnx_providers: [CUDAExecutionProvider, CPUExecutionProvider]
```

ONNX Runtime's CUDA provider requires several NVIDIA libraries that are **not** bundled with the `onnxruntime-gpu` package and must be installed separately.  The `apt` packages are the simplest route if you have sudo access:

```bash
sudo apt install libcurand-12 libcufft-12 libcudart-12
```

If you are working in a virtualenv without system-level access, install the pip equivalents into the same environment as bioaccx:

```bash
uv pip install nvidia-curand-cu12 nvidia-cufft-cu12 nvidia-cuda-runtime-cu12
# or: pip install nvidia-curand-cu12 nvidia-cufft-cu12 nvidia-cuda-runtime-cu12
```

These packages install the `.so` files under `site-packages/nvidia/*/lib/`, but ONNX Runtime loads them via `dlopen` before Python's import machinery runs, so they are invisible to the dynamic linker by default.  The fix is a `sitecustomize.py` file that preloads them at interpreter startup.  Create it at:

```
<venv>/lib/python3.x/site-packages/sitecustomize.py
```

with the following content:

```python
import ctypes, pathlib

_nvidia_base = pathlib.Path(__file__).parent / "nvidia"
for lib in _nvidia_base.glob("*/lib/lib*.so.*"):
    try:
        ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
    except OSError:
        pass
```

This loads every nvidia `.so` into the process with `RTLD_GLOBAL` so that subsequent `dlopen` calls from ONNX Runtime can resolve them.  No changes to `LD_LIBRARY_PATH` or the shell environment are needed.

To verify the CUDA provider is active after setup:

```python
import onnxruntime as ort
print(ort.get_available_providers())
# ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
```
