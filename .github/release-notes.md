## Install

No Python needed: the installer sets everything up and adds a **bioaccx** shortcut that opens the GUI in your browser. Run it again to upgrade.

**Windows** — open PowerShell and paste:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://github.com/biodiversica/bioaccx/releases/latest/download/install.ps1 | iex"
```

**macOS (Apple Silicon) and Linux** — open a terminal and paste:

```sh
curl -LsSf https://github.com/biodiversica/bioaccx/releases/latest/download/install.sh | sh
```

Already using Python? `uv tool install --python 3.12 "bioaccx[cpu,gui,umap]"` or `pip install "bioaccx[cpu]"`. See the [installation guide](https://github.com/biodiversica/bioaccx/blob/master/docs/en/installation.md) ([português](https://github.com/biodiversica/bioaccx/blob/master/docs/pt-BR/installation.md)).

The download is about 2 GB, mostly TensorFlow.

## What's in this release

<!-- Write the highlights here before publishing. -->
