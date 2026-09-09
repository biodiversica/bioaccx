"""CLI entry point for bioaccx.

Usage
-----
    bioaccx train config.yaml
    bioaccx validate config.yaml
    bioaccx dataset config.yaml --no-split
    bioaccx merge head.onnx --backbone 0xbb00
    python -m bioaccx train config.yaml

The pre-subcommand form (``bioaccx config.yaml --dataset``) still works; see
:mod:`bioaccx._legacy_cli`, which rewrites it into the equivalent subcommand.

Heavy imports (``bioaccx.train``, ``bioaccx.config``) are deferred into the
command bodies so that ``validate``, ``registry`` and ``--help`` don't pay the
TensorFlow/ONNX import cost.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import click
import typer

#: Suffixes that mark a path as a config file rather than a model file. ``merge``
#: and ``extract-head`` accept either, and decide which they were given by suffix.
CONFIG_SUFFIXES = frozenset({".yaml", ".yml", ".json"})
MODEL_SUFFIXES = frozenset({".onnx", ".tflite"})

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    # Help text is plain prose, not rich markup: without this, bracketed names
    # like [gui] and [umap] are parsed as style tags and vanish from --help.
    rich_markup_mode=None,
    context_settings={"help_option_names": ["-h", "--help"]},
    help=(
        "Bioacoustic custom classifier tool.\n\n"
        "Most commands read a JSON or YAML config file describing the foundation "
        "model, the dataset, the training run and what to export. See "
        "example_config.yaml for a fully commented reference, and the README for "
        "the complete key list.\n\n"
        "Config sections: foundation_model (backbone and its audio window), "
        "dataset (sources, labelling, preprocessing, augmentation, splitting), "
        "training (keras and sklearn heads), output (paths, formats, precisions, "
        "exports) and umap (projection settings for the embeddings command)."
    ),
)


def _fail(message: str) -> "click.UsageError":
    """A usage error, reported on stderr with exit code 2."""
    return click.UsageError(message)


def _looks_like_config(path: Path) -> bool:
    return path.suffix.lower() in CONFIG_SUFFIXES


def _target_kind(target: Path, option_hint: str) -> str:
    """Classify ``merge``/``extract-head``'s positional as 'config' or 'model'.

    The suffix decides, so the error for an unrecognised one can name both
    forms rather than failing later inside a YAML parse.
    """
    suffix = target.suffix.lower()
    if suffix in CONFIG_SUFFIXES:
        return "config"
    if suffix in MODEL_SUFFIXES:
        return "model"
    raise _fail(
        f"cannot tell what {target.name!r} is: pass a config file "
        f"({', '.join(sorted(CONFIG_SUFFIXES))}) or a model file "
        f"({', '.join(sorted(MODEL_SUFFIXES))}) together with {option_hint}"
    )


def _load(config: Path):
    """Read and validate a config file, reporting parse errors as usage errors."""
    from bioaccx.config import load_config
    try:
        return load_config(str(config))
    except (FileNotFoundError, ValueError) as exc:
        raise _fail(str(exc)) from exc


def _backbone_foundation(backbone: Optional[str], embed_dim: Optional[int]):
    """Build a FoundationModelConfig from --backbone / --embed-dim.

    *backbone* is either a registry ID (``0xbb00`` — full defaults, including the
    HuggingFace source and embedding size) or a path to a local backbone file.
    A local file tells us nothing about the model, so its input tensor name is
    read from the file itself (keeping the merged graph's input name unchanged)
    and the embedding size must come from --embed-dim.
    """
    from bioaccx.config import FoundationModelConfig, _from_dict
    from bioaccx.registry import get_registry_defaults, list_registry_ids

    if backbone and not Path(backbone).exists():
        defaults = get_registry_defaults(backbone) if backbone.lower().startswith("0x") else None
        if defaults is None:
            raise _fail(
                f"--backbone {backbone!r} is neither an existing file nor a known registry "
                f"ID. Known IDs: {', '.join(list_registry_ids())} (see `bioaccx registry`)."
            )
        fm = _from_dict(FoundationModelConfig, defaults)
    elif backbone:
        path = Path(backbone)
        fm = FoundationModelConfig(
            name="custom", version="unknown", source="local", path=str(path),
            format="tflite" if path.suffix.lower() == ".tflite" else "onnx",
        )
        if fm.format == "onnx":
            import onnx
            graph_inputs = onnx.load(str(path), load_external_data=False).graph.input
            if graph_inputs:
                fm.input_name = graph_inputs[0].name
    else:
        fm = FoundationModelConfig(name="custom", version="unknown")

    if embed_dim:
        fm.embedding_size = embed_dim
    return fm


# ── pipeline commands ────────────────────────────────────────────────────────

@app.command()
def train(
    config: Path = typer.Argument(..., metavar="CONFIG",
                                  help="JSON or YAML config file."),
) -> None:
    """Run the full pipeline: dataset, embeddings, training, export, reports."""
    from bioaccx.train import run
    run(_load(config))


@app.command()
def validate(
    config: Path = typer.Argument(..., metavar="CONFIG",
                                  help="JSON or YAML config file."),
) -> None:
    """Parse and validate a config without running anything."""
    cfg = _load(config)
    print("Config parsed successfully.")
    print(f"  Foundation model : {cfg.foundation_model.name} v{cfg.foundation_model.version}")
    data_dir = cfg.dataset.data_dir
    data_dir_str = data_dir if isinstance(data_dir, str) else ", ".join(data_dir)
    print(f"  Dataset          : {data_dir_str}  mode={cfg.dataset.label_mode}")
    print(f"  Classifier       : {cfg.training.classifier}")
    print(f"  Output           : {cfg.output_dir}")


@app.command()
def dataset(
    config: Path = typer.Argument(..., metavar="CONFIG",
                                  help="JSON or YAML config file."),
    no_split: bool = typer.Option(
        False, "--no-split",
        help="Export every sample into dataset/<label>/ without a train/test split, "
             "ignoring test_ratio and any predefined split. The sample list CSV is "
             "written with an empty split column, ready to be filled in by hand and "
             "fed back as a label_mode: table source.",
    ),
) -> None:
    """Load, split and export the dataset as chunked WAV files, without training."""
    from bioaccx.train import run_dataset_export
    run_dataset_export(_load(config), no_split=no_split)


@app.command()
def embeddings(
    config: Path = typer.Argument(..., metavar="CONFIG",
                                  help="JSON or YAML config file."),
) -> None:
    """Compute the embedding database (and UMAP, when enabled) without training.

    If the dataset has not been prepared it is loaded and split first. Embeddings
    are always exported — SQLite by default, or .npy per output.embeddings_format.
    When umap.enabled is set in the config (requires the [umap] extra), a UMAP
    projection is fitted and written as a data CSV plus a scatter-plot PNG.
    """
    from bioaccx.train import run_embeddings
    run_embeddings(_load(config))


# ── model surgery ────────────────────────────────────────────────────────────

@app.command()
def merge(
    target: Path = typer.Argument(
        ..., metavar="CONFIG|HEAD",
        help="A config file carrying foundation_model + output.head_path, or a "
             "classifier head file to merge with --backbone.",
    ),
    backbone: Optional[str] = typer.Option(
        None, "--backbone", metavar="ID|PATH",
        help="Foundation model for a head given directly: a registry ID such as "
             "0xbb00 (downloaded if needed; see `bioaccx registry`) or a local "
             "backbone file.",
    ),
    embed_dim: Optional[int] = typer.Option(
        None, "--embed-dim", metavar="N",
        help="Embedding size, when it is not implied by --backbone.",
    ),
    out: Optional[Path] = typer.Option(
        None, "-o", "--out", metavar="PATH",
        help="Where the merged model is written. Defaults to a sibling of the head "
             "file. Ignored when the destination comes from a config.",
    ),
) -> None:
    """Merge a backbone and a classifier head into one full ONNX model.

    An ONNX backbone is combined with an ONNX or TFLite classifier head; a TFLite
    head is converted to ONNX automatically first. The backbone may be a local
    file or downloaded from HuggingFace. No dataset or training is involved.
    """
    from bioaccx.config import BioaccxConfig, OutputConfig
    from bioaccx.train import run_merge

    kind = _target_kind(target, "--backbone")
    try:
        if kind == "config":
            if backbone or embed_dim:
                raise _fail(
                    "give either a config file or --backbone/--embed-dim, not both — "
                    "the config already carries the backbone and the head path"
                )
            run_merge(_load(target))
            return

        if not backbone:
            raise _fail(
                "merging a head file requires --backbone (a registry ID or a backbone "
                "file); pass a config file instead to take it from there"
            )
        cfg = BioaccxConfig(
            foundation_model=_backbone_foundation(backbone, embed_dim),
            output=OutputConfig(head_path=str(target)),
        )
        run_merge(cfg, out_path=out or target.with_name(f"{target.stem}_full.onnx"))
    except (FileNotFoundError, ValueError) as exc:
        raise _fail(str(exc)) from exc


@app.command("extract-head")
def extract_head(
    target: Path = typer.Argument(
        ..., metavar="CONFIG|MODEL",
        help="A config file carrying output.extract_from, or a full model to "
             "extract from — a BirdNET-Analyzer .tflite or an .onnx "
             "backbone+head model.",
    ),
    backbone: Optional[str] = typer.Option(
        None, "--backbone", metavar="ID|PATH",
        help="Foundation model for a model given directly: a registry ID such as "
             "0xbb00 (see `bioaccx registry`) or a local backbone file.",
    ),
    embed_dim: Optional[int] = typer.Option(
        None, "--embed-dim", metavar="N",
        help="Embedding size, when it is not implied by --backbone. Marks the "
             "boundary between backbone and head (e.g. 1024 for BirdNET).",
    ),
    labels: Optional[Path] = typer.Option(
        None, "--labels", metavar="PATH",
        help="Class label file (one per line); defaults to a sibling *_Labels.txt.",
    ),
    fmt: str = typer.Option(
        "both", "--format", metavar="FORMAT",
        click_type=click.Choice(["onnx", "tflite", "both"]),
        help="Head formats to write.",
    ),
    out: Optional[Path] = typer.Option(
        None, "-o", "--out", metavar="PATH",
        help="Directory to write the extracted head into. Defaults to a sibling of "
             "the source file. Ignored when the destination comes from a config.",
    ),
) -> None:
    """Extract the classifier head from a full model and re-export it head-only.

    The head weights are read directly from the source graph — the backbone is
    never converted or run. No dataset or training is involved.
    """
    from bioaccx.config import BioaccxConfig, OutputConfig
    from bioaccx.train import run_extract_head

    kind = _target_kind(target, "--backbone or --embed-dim")
    try:
        if kind == "config":
            if backbone or embed_dim or labels:
                raise _fail(
                    "give either a config file or --backbone/--embed-dim/--labels, not "
                    "both — the config already carries the model path and the output "
                    "location"
                )
            run_extract_head(_load(target))
            return

        if not backbone and not embed_dim:
            raise _fail(
                "extracting from a model file requires --backbone (a registry ID) or "
                "--embed-dim — the embedding size marks where the head starts"
            )
        cfg = BioaccxConfig(
            foundation_model=_backbone_foundation(backbone, embed_dim),
            output=OutputConfig(
                extract_from=str(target),
                labels_file=str(labels) if labels else None,
                output_format=fmt,
            ),
        )
        run_extract_head(
            cfg,
            out_dir=out or target.with_name(f"{target.stem}_head"),
            stem=target.stem,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise _fail(str(exc)) from exc


@app.command("convert-head")
def convert_head_cmd(
    head: Path = typer.Argument(..., metavar="HEAD",
                               help="The .onnx or .tflite classifier head to convert."),
    out: Optional[Path] = typer.Option(
        None, "-o", "--out", metavar="PATH",
        help="Where the converted head is written. Defaults to the source path with "
             "the other suffix.",
    ),
) -> None:
    """Convert a classifier head between ONNX and TFLite.

    The direction is taken from the file suffix and the result is verified
    against the source. No config file, backbone or training involved.
    """
    from bioaccx.convert_head import convert_head
    try:
        convert_head(head, out)
    except (FileNotFoundError, ValueError) as exc:
        raise _fail(str(exc)) from exc


@app.command()
def registry() -> None:
    """List the registered foundation models and their default parameters."""
    from bioaccx.registry import print_registry
    print_registry()


# ── browser GUI ──────────────────────────────────────────────────────────────

LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def _is_loopback(host: str) -> bool:
    import ipaddress
    if host in LOOPBACK_HOSTS:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _lan_address() -> str:
    """This machine's address on the network it routes through, for the hint line."""
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("10.255.255.255", 1))
        return sock.getsockname()[0]
    except OSError:
        return socket.gethostbyname(socket.gethostname())
    finally:
        sock.close()


def _tailscale_address() -> Optional[str]:
    """This node's Tailscale IPv4, when Tailscale is installed and up.

    A machine on a tailnet routes over its LAN interface, so ``_lan_address``
    reports the LAN IP — useless to anyone reaching this host over Tailscale.
    Asking the CLI is cheap, needs no dependency, and stays quiet when Tailscale
    is absent.
    """
    import shutil
    import subprocess

    exe = shutil.which("tailscale")
    if not exe:
        return None
    try:
        result = subprocess.run([exe, "ip", "-4"], capture_output=True,
                                text=True, timeout=2, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return lines[0] if lines else None


@app.command()
def gui(
    config: Optional[Path] = typer.Argument(
        None, metavar="[CONFIG]",
        help="Config file to open on start. Omitted, the editor starts from a template.",
    ),
    host: str = typer.Option(
        "127.0.0.1", "--host",
        help="Address to bind. Use 0.0.0.0 to reach the editor from another machine.",
    ),
    models_dir: Path = typer.Option(
        Path("custom_models"), "--models-dir",
        help="Directory of trained models the results explorer reads.",
    ),
    port: int = typer.Option(8765, "--port", "-p", min=1, max=65535,
                             help="Port to listen on."),
    lang: str = typer.Option(
        "en", "--lang", "-l",
        help="Interface language the editor opens in. The picker in its toolbar "
             "lists the languages installed, and remembers a change per browser.",
    ),
    token: Optional[str] = typer.Option(
        None, "--token",
        help="Access token required by the editor. One is generated when binding a "
             "non-loopback address; use --no-auth to serve without it.",
    ),
    no_auth: bool = typer.Option(False, "--no-auth",
                                help="Serve without an access token."),
    open_browser: bool = typer.Option(True, "--open/--no-open",
                                      help="Open the editor in a browser on start."),
) -> None:
    """Edit a config file in the browser.

    The editor writes the same config files the other commands read — the form
    is generated from the config schema, and the file it produces is shown as
    you edit. Training is still run from the terminal.

    Requires the [gui] extra: uv tool install "bioaccx[cpu,gui]"
    """
    import os
    import secrets
    import threading
    import webbrowser

    try:
        import uvicorn
        from bioaccx.gui import i18n
        from bioaccx.gui.server import create_app
    except ImportError as exc:
        raise _fail(
            f"the GUI needs the [gui] extra ({exc.name} is missing). Install with: "
            'uv tool install "bioaccx[cpu,gui]"  —  or:  uv sync --extra gui'
        ) from exc

    if config is not None and not config.exists():
        raise _fail(f"no such config file: {config}")

    access_token: Optional[str] = None
    if not no_auth:
        access_token = token or (None if _is_loopback(host) else secrets.token_urlsafe(16))

    language = i18n.normalize(lang)
    if language != lang:
        typer.secho(f"unknown language {lang!r}; using {language!r} "
                    f"(installed: {', '.join(i18n.available())})",
                    fg=typer.colors.YELLOW)

    server = create_app(config_path=config, token=access_token,
                        models_dir=models_dir, lang=language)

    shown_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    suffix = f"?token={access_token}" if access_token else ""
    url = f"http://{shown_host}:{port}/{suffix}"

    typer.echo(f"bioaccx config editor  —  {config or 'new config'}")
    typer.echo(f"  models: {models_dir}")
    typer.secho(f"  {url}", fg=typer.colors.GREEN, bold=True)
    if host in ("0.0.0.0", "::"):
        typer.echo(f"  http://{_lan_address()}:{port}/{suffix}   (this LAN)")
        tailnet = _tailscale_address()
        if tailnet:
            typer.echo(f"  http://{tailnet}:{port}/{suffix}   (tailnet)")
        typer.echo("  Bound to every interface. To expose it on one network only, "
                   "pass that interface's address to --host.")
    if access_token:
        typer.echo("  The token in the link is required; it is set as a cookie on first load.")
    typer.echo("  Press Ctrl+C to stop.")

    if open_browser and not os.environ.get("BIOACCX_NO_BROWSER"):
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    uvicorn.run(server, host=host, port=port, log_level="warning", access_log=False)


# ── entry point ──────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    """Dispatch *argv* (default ``sys.argv[1:]``) to a subcommand.

    Legacy flag invocations are rewritten to their subcommand equivalent first,
    so there is only ever one dispatch path.

    Click is driven with ``standalone_mode=False`` so that a successful command
    returns normally instead of raising ``SystemExit(0)`` — callers, including
    the tests, invoke ``main()`` directly. Usage errors are reported here
    instead, keeping argparse's exit code 2 and stderr output.
    """
    from bioaccx import _legacy_cli

    args = list(sys.argv[1:] if argv is None else argv)
    if _legacy_cli.is_legacy(args):
        args = _legacy_cli.translate(args)

    command = typer.main.get_command(app)
    try:
        command(args, standalone_mode=False)
    except click.UsageError as exc:
        exc.show()
        raise SystemExit(exc.exit_code) from exc
    except click.exceptions.Exit as exc:
        if exc.exit_code:
            raise SystemExit(exc.exit_code) from exc
    except click.Abort as exc:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(1) from exc
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
