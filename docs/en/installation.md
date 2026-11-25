# Installation

## The installer (no Python required)

On Windows, open PowerShell and run:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://github.com/biodiversica/bioaccx/releases/latest/download/install.ps1 | Out-String | iex"
```

On macOS (Apple Silicon) and Linux, open a terminal and run:

```sh
curl -LsSf https://github.com/biodiversica/bioaccx/releases/latest/download/install.sh | sh
```

The installer performs the following steps:

1. Installs [uv](https://docs.astral.sh/uv/) into `~/.local/bin` if it is not
   already present. uv downloads its own Python, so no prior installation is
   required, and any existing Python installation is left unchanged.
2. Runs `uv tool install --python 3.12 "bioaccx[cpu,gui,umap]"`, which installs
   the CPU runtime, the browser GUI and the UMAP plots, and adds `bioaccx` to the
   PATH.
3. Creates a working folder, `~/bioaccx`, in which the GUI starts. Saved
   configurations and trained models are stored there unless another location
   is chosen.
4. Adds a **bioaccx** launcher, with the bioaccx logo, that runs `bioaccx gui` in
   that folder: a Desktop and Start menu shortcut on Windows, `bioaccx.command`
   on the macOS Desktop, and an applications menu entry on Linux. The launcher opens a terminal window
   next to the browser. This window runs the server, and closing it (or pressing
   Ctrl+C) stops the server.

Before any of these steps, the installer lists what it will do and asks for
confirmation. Running it again upgrades bioaccx. Environment variables change
its defaults: `BIOACCX_PACKAGE` (the package to install, for example
`bioaccx[gpu,gui]`), `BIOACCX_WORKDIR` (the working folder) and `BIOACCX_YES=1`
(proceed without asking, for scripts and CI; without it, the installer refuses
to run where there is no terminal to ask in).

### Uninstalling

The same installer uninstalls bioaccx when run with `BIOACCX_UNINSTALL=1`. It
also lists what it will remove and asks first:

```powershell
powershell -ExecutionPolicy ByPass -c '$env:BIOACCX_UNINSTALL=1; irm https://github.com/biodiversica/bioaccx/releases/latest/download/install.ps1 | Out-String | iex'
```

```sh
curl -LsSf https://github.com/biodiversica/bioaccx/releases/latest/download/install.sh | BIOACCX_UNINSTALL=1 sh
```

`BIOACCX_UNINSTALL=1` removes bioaccx and its launcher. `BIOACCX_UNINSTALL=all`
also removes the BirdNET and Perch backbones that bioaccx downloaded into the
Hugging Face cache, and no other model stored there. On Windows, close the GUI
first: Windows cannot remove a program that is running.

**What uninstalling leaves behind.** bioaccx's own environment, with every
Python package installed for it, is removed, together with the `bioaccx`
command. What it was installed with stays, even with `=all`, since the
uninstaller cannot tell whether anything else uses it:

| Left in place | Where | Reason |
|---|---|---|
| The working folder | `~/bioaccx` | It holds the saved configurations and trained models. |
| uv | `~/.local/bin/uv` and `uvx` (`.exe` on Windows) | Other programs may use it. |
| Python 3.12 | `uv python dir` | uv's own Python, which other uv tools may use. If a Python 3.12 was already installed, uv used it and downloaded none. |
| uv's download cache | `uv cache dir` | Shared by everything uv installs. It holds the downloaded packages, TensorFlow by far the largest, and can take more than 2 GB. |
| Models from Kaggle or another source | their own caches | Only the registry's Hugging Face backbones are removed, by `=all`. |

If nothing else on the machine uses uv, the rest can be removed by hand after
uninstalling:

```bash
uv cache clean                # every downloaded package
uv python uninstall 3.12      # uv's Python, if it downloaded one
```

and then uv itself, by deleting `uv` and `uvx` from `~/.local/bin`.

None of the cache commands can break an installed program: an environment does
not read the cache once it is installed, and each command waits while another
uv is using the cache. They differ in what they delete:

- `uv cache prune` deletes only what nothing refers to any more, such as entries
  left by older uv versions. It is the most cautious, but the downloaded
  packages are still referred to, so after uninstalling it frees little.
- `uv cache clean PACKAGE...` deletes only the packages named. `uv cache clean
  tensorflow tensorflow-cpu` frees most of the space bioaccx took, unless other
  tools use TensorFlow too.
- `uv cache clean` deletes the whole cache, for every program. It frees the
  most, and anything installed with uv later downloads its packages again.

## Supported systems

| System | |
|---|---|
| Linux x86-64 and ARM64 | ✓ |
| Windows x86-64 | ✓ |
| macOS on Apple Silicon (M1 and later) | ✓ |
| macOS on Intel | ✗ (TensorFlow does not publish builds for it) |

Python 3.12 or 3.13 is required, as these are the versions TensorFlow currently
supports. The installer and the commands below pin version 3.12, so that a newer
Python on the machine is never selected by accident.

## As a command-line tool

This is the usual installation for users familiar with the terminal. bioaccx
runs models through ONNX Runtime, and **no runtime is installed by default**:
choose `cpu` or `gpu` at installation. Installing `bioaccx` without an extra
succeeds, but the first attempt to run a model fails, so in practice the extra
is required.

`uv tool install` installs bioaccx in an isolated environment and adds `bioaccx`
to the PATH:

```bash
uv tool install --python 3.12 "bioaccx[cpu]"            # CPU inference
uv tool install --python 3.12 "bioaccx[gpu]"            # CUDA 12 (see the GPU section below)
uv tool install --python 3.12 "bioaccx[cpu,gui,umap]"   # with the browser GUI and UMAP plots
```

`uv tool install` has **no `--extra` option**: extras are given inside the
brackets, and the whole argument must be quoted so that the shell does not
expand them.

```bash
uv tool install --python 3.12 "/path/to/bioaccx[cpu]"   # from a local checkout
uv tool upgrade bioaccx                                 # upgrade
uv tool uninstall bioaccx                               # uninstall
```

## In a project

```bash
uv add "bioaccx[cpu]"                       # as a dependency
pip install "bioaccx[cpu]"                  # or with pip
```

## Development setup

```bash
git clone git@github.com:biodiversica/bioaccx.git
cd bioaccx
uv sync --extra cpu                         # or --extra gpu
uv sync --extra cpu --extra gui --extra umap
uv run bioaccx --help
uv run pytest
```

`uv sync --extra X` **replaces** the active set of extras rather than adding to
it, so every required extra must be named on each run. Syncing with
`--extra gui` alone removes `onnxruntime` and `umap-learn`.

### Making a release

Set the new version in both `pyproject.toml` and `bioaccx/__init__.py`, commit
the change, and then create and push a tag:

```bash
git tag v0.2.0 && git push origin v0.2.0
```

The tag triggers `.github/workflows/release.yml`. The tests and the installers
run on Linux, macOS and Windows, the wheel is uploaded to PyPI, and a **draft**
GitHub release is created with the installers attached and the notes from
`.github/release-notes.md`. After the highlights have been written, the release
must be published, because the installation commands download from the latest
*published* release.

## Extras

| Extra | Adds | Required for |
|---|---|---|
| `cpu` | `onnxruntime` | Running any model on the CPU. Choose this or `gpu`. |
| `gpu` | `onnxruntime-gpu` | Running on CUDA 12. Choose this or `cpu`. |
| `gui` | `fastapi`, `uvicorn`, `ruamel.yaml` | `bioaccx gui`, the browser configuration editor and results explorer |
| `umap` | `umap-learn`, `matplotlib` | The UMAP projection and plots from `bioaccx embeddings` (`umap.enabled: true`) |
| `ssh` | `paramiko` | Remote `data_dir` over SSH/SFTP (already a core dependency; the extra is kept for compatibility with older installations) |
| `arbimon` | — | Arbimon sources. The `rfcx` SDK is not published on PyPI, and its release wheel must be installed manually: `pip install https://github.com/rfcx/rfcx-sdk-python/releases/download/0.3.1/rfcx-0.3.1-py3-none-any.whl` |

TensorFlow, scikit-learn, ONNX, Hugging Face Hub and Kaggle Hub are **core**
dependencies: they are always installed and require no extra.

`cpu` and `gpu` are declared mutually exclusive, but this is enforced only when
syncing this project; installing `bioaccx[cpu,gpu]` as a tool or a dependency
installs both runtimes. Only one should be requested.

## Disk space

The environment is large: TensorFlow alone occupies about 1.3 GB, and a checkout
synced with `cpu`, `gui`, `umap` and the development group occupies about
2.2 GB. An installation with `uv tool install` has its own separate copy.

## GPU acceleration (ONNX Runtime + CUDA)

To run the ONNX backbone on a GPU, set `onnx_providers` in the configuration:

```yaml
foundation_model:
  onnx_providers: [CUDAExecutionProvider, CPUExecutionProvider]
```

The CUDA provider of ONNX Runtime requires several NVIDIA libraries that are **not** bundled with the `onnxruntime-gpu` package and must be installed separately. With administrator (sudo) access, the `apt` packages are the simplest option:

```bash
sudo apt install libcurand-12 libcufft-12 libcudart-12
```

In a virtual environment without system-level access, install the pip equivalents into the same environment as bioaccx:

```bash
uv pip install nvidia-curand-cu12 nvidia-cufft-cu12 nvidia-cuda-runtime-cu12
# or: pip install nvidia-curand-cu12 nvidia-cufft-cu12 nvidia-cuda-runtime-cu12
```

These packages install the `.so` files under `site-packages/nvidia/*/lib/`, but ONNX Runtime loads them with `dlopen` before the Python import machinery runs, so by default they are not visible to the dynamic linker. The solution is a `sitecustomize.py` file that preloads them when the interpreter starts. Create it at:

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

This loads every NVIDIA `.so` file into the process with `RTLD_GLOBAL`, so that subsequent `dlopen` calls from ONNX Runtime can resolve them. No changes to `LD_LIBRARY_PATH` or to the shell environment are required.

To verify that the CUDA provider is active after setup:

```python
import onnxruntime as ort
print(ort.get_available_providers())
# ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
```
