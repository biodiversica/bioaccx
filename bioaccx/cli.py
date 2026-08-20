"""CLI entry point for bioaccx.

Usage
-----
    bioaccx config.yaml
    bioaccx config.yaml --validate
    python -m bioaccx config.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _backbone_foundation(parser, backbone: str | None, embed_dim: int | None):
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
            parser.error(
                f"--backbone {backbone!r} is neither an existing file nor a known registry "
                f"ID. Known IDs: {', '.join(list_registry_ids())} (see --registry)."
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


def _run_merge_cli(parser, args) -> None:
    """--merge with the head path (and backbone) given on the command line."""
    from bioaccx.config import BioaccxConfig, OutputConfig
    from bioaccx.train import run_merge

    head = Path(args.merge)
    if not args.backbone:
        parser.error("--merge <HEAD> requires --backbone (a registry ID or a backbone file)")
    cfg = BioaccxConfig(
        foundation_model=_backbone_foundation(parser, args.backbone, args.embed_dim),
        output=OutputConfig(head_path=str(head)),
    )
    out_path = Path(args.out) if args.out else head.with_name(f"{head.stem}_full.onnx")
    run_merge(cfg, out_path=out_path)


def _run_extract_head_cli(parser, args) -> None:
    """--extract-head with the source model given on the command line."""
    from bioaccx.config import BioaccxConfig, OutputConfig
    from bioaccx.train import run_extract_head

    src = Path(args.extract_head)
    if not args.backbone and not args.embed_dim:
        parser.error(
            "--extract-head <MODEL> requires --backbone (a registry ID) or --embed-dim "
            "— the embedding size marks where the head starts"
        )
    fm = _backbone_foundation(parser, args.backbone, args.embed_dim)
    cfg = BioaccxConfig(
        foundation_model=fm,
        output=OutputConfig(
            extract_from=str(src),
            labels_file=args.labels,
            output_format=args.format,
        ),
    )
    out_dir = Path(args.out) if args.out else src.with_name(f"{src.stem}_head")
    run_extract_head(cfg, out_dir=out_dir, stem=src.stem)


def main(argv: list[str] | None = None) -> None:
    """Parse CLI arguments and dispatch to the appropriate pipeline entry point.

    Modes (mutually exclusive flags):
      (none)     — full training pipeline: load dataset, extract embeddings,
                   train classifier(s), export models, write reports.
      --validate — parse and validate the config only; no IO beyond reading the file.
      --dataset  — load, split, and export chunked WAV files; no training.
                   With --no-split the train/test split is skipped entirely.
      --embeddings — compute the embedding database + UMAP outputs; no training.
      --merge    — merge an existing backbone + head into a single full ONNX model.
      --extract-head — extract the head from a full custom BirdNET-Analyzer TFLite model
                   and re-export it head-only (no backbone conversion).

    --merge and --extract-head take either a config file or their few inputs
    directly on the command line (a head/model path plus --backbone); the model
    surgery modes need no dataset or training settings, so a config is optional.
      --convert-head — convert an existing head file between ONNX and TFLite.
                   Takes the head path directly; no config file needed.

    Heavy imports (bioaccx.train, bioaccx.config) are deferred so that
    --validate and simple invocations don't pay the TF/ONNX import cost
    unnecessarily.
    """
    parser = argparse.ArgumentParser(
        prog="bioaccx",
        description="Bioacoustic custom classifier tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Config file (JSON or YAML) keys
--------------------------------
foundation_model:
  name, version, format (onnx|tflite|protobuf),
  source (local|huggingface), path|hf_repo+hf_filename,
  sample_rate, window_seconds|window_samples,
  input_name, output_name, embedding_size

dataset:
  data_dir, label_mode (subfolders|table|file_per_label),
  table_file, audio_extensions, test_ratio, random_seed,
  filename_col, label_col, start_col, end_col, split_col
  sources: list of per-source blocks combined in one run; each inherits the
           top-level dataset fields and overrides them (mix label modes /
           preprocessing / augmentation / SSH per source). Run-level fields
           (test_ratio, random_seed, append_dataset_path, embedding_workers,
           credentials, audio_extensions) come from the top level only.

training:
  classifier (keras|sklearn|both)
  keras:  hidden_units, dropout, epochs, batch_size, learning_rate
  sklearn: C, max_iter, solver

output:
  output_path, model_name, model_version,
  output_type (head|full|both), output_format (onnx|tflite|both),
  data_types (subset of [FP32, FP16, INT8]; default = foundation model data_type)
""",
    )
    parser.add_argument(
        "config",
        nargs="?",
        help="Path to a JSON or YAML config file",
    )
    parser.add_argument(
        "--registry",
        action="store_true",
        help="List all registered foundation models and their default parameters, then exit",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Parse and validate the config without running training",
    )
    parser.add_argument(
        "--dataset",
        action="store_true",
        help="Load, split, and export the dataset as chunked WAV files without training",
    )
    parser.add_argument(
        "--no-split",
        action="store_true",
        help=(
            "Only with --dataset: export every sample into dataset/<label>/ without a "
            "train/test split, ignoring test_ratio and any predefined split. The sample "
            "list CSV is written with an empty split column, ready to be filled in by hand "
            "and fed back as a label_mode: table source."
        ),
    )
    parser.add_argument(
        "--embeddings",
        action="store_true",
        help=(
            "Compute the embedding database without training a classifier. "
            "If the dataset has not been prepared it is loaded and split first. "
            "Embeddings are always exported (SQLite by default, or .npy per "
            "output.embeddings_format). When umap.enabled is set in the config "
            "(requires the [umap] extra), also fits a UMAP projection and writes "
            "a UMAP data CSV and a scatter-plot PNG."
        ),
    )
    parser.add_argument(
        "--merge",
        nargs="?",
        const=True,
        metavar="HEAD",
        help=(
            "Merge a backbone and a classifier head into one full ONNX model. Either pass "
            "the head path here together with --backbone (no config file needed), or use a "
            "config carrying foundation_model + output.head_path. "
            "Merge an existing ONNX backbone and ONNX or TFLite classifier head into a single full ONNX model. "
            "Backbone can be a local file (foundation_model.path) or downloaded from HuggingFace "
            "(foundation_model.source=huggingface + hf_repo). "
            "Requires output.head_path (head) in the config. "
            "A TFLite head is converted to ONNX automatically before merging. "
            "No dataset or training is performed."
        ),
    )
    parser.add_argument(
        "--extract-head", "--extract_head",
        dest="extract_head",
        nargs="?",
        const=True,
        metavar="MODEL",
        help=(
            "Extract the classifier head from a full model — a BirdNET-Analyzer .tflite or "
            "an .onnx backbone+head model. Either pass the model path here together with "
            "--backbone or --embed-dim (no config file needed), or use a config carrying "
            "output.extract_from. "
            "Extract the classifier head from a full BirdNET-Analyzer TFLite model and "
            "re-export it as a head-only model (ONNX and/or TFLite per output.output_format). "
            "The head weights are read directly from the flatbuffer — the backbone is never "
            "converted or run. Requires output.extract_from (path to the .tflite model). "
            "No dataset or training is performed."
        ),
    )

    parser.add_argument(
        "--convert-head",
        metavar="HEAD",
        help=(
            "Convert an existing classifier head between formats: a .tflite head is "
            "converted to ONNX and an .onnx head to TFLite. No config file, backbone or "
            "training involved — the direction is taken from the file suffix and the "
            "result is verified against the source. Writes next to the source file with "
            "the other suffix unless -o is given."
        ),
    )
    parser.add_argument(
        "-o", "--out",
        metavar="PATH",
        help=(
            "Output path, for the modes that take their inputs on the command line: the "
            "converted head (--convert-head), the merged model (--merge), or the directory "
            "to write the extracted head into (--extract-head). Defaults to a sibling of "
            "the source file. Ignored when the destination comes from a config."
        ),
    )
    parser.add_argument(
        "--backbone",
        metavar="ID|PATH",
        help=(
            "Foundation model for --merge / --extract-head given on the command line: a "
            "registry ID such as 0xbb00 (downloaded if needed; see --registry) or a path to "
            "a local backbone file."
        ),
    )
    parser.add_argument(
        "--embed-dim",
        type=int,
        metavar="N",
        help=(
            "Embedding size for --extract-head, when it is not implied by --backbone. "
            "Marks the boundary between backbone and head (e.g. 1024 for BirdNET)."
        ),
    )
    parser.add_argument(
        "--labels",
        metavar="PATH",
        help="Class label file (one per line) for --extract-head; defaults to a sibling *_Labels.txt",
    )
    parser.add_argument(
        "--format",
        choices=("onnx", "tflite", "both"),
        default="both",
        help="Head formats written by --extract-head on the command line (default: both)",
    )

    args = parser.parse_args(argv)

    if args.no_split and not args.dataset:
        parser.error("--no-split is only valid together with --dataset")

    cli_merge = isinstance(args.merge, str)
    cli_extract = isinstance(args.extract_head, str)

    if args.out and not (args.convert_head or cli_merge or cli_extract):
        parser.error(
            "-o/--out is only valid with --convert-head, or with --merge / --extract-head "
            "when their inputs are given on the command line"
        )
    if (args.backbone or args.embed_dim or args.labels) and not (cli_merge or cli_extract):
        parser.error(
            "--backbone / --embed-dim / --labels are only valid with --merge <HEAD> or "
            "--extract-head <MODEL>; with a config file these come from the config"
        )
    if args.config and (cli_merge or cli_extract):
        parser.error(
            "give either a config file or the command-line inputs, not both — "
            "the config already carries the head/model path and the output location"
        )

    if args.convert_head or cli_merge or cli_extract:
        from bioaccx.convert_head import convert_head
        try:
            if args.convert_head:
                convert_head(Path(args.convert_head), Path(args.out) if args.out else None)
            elif cli_merge:
                _run_merge_cli(parser, args)
            else:
                _run_extract_head_cli(parser, args)
        except (FileNotFoundError, ValueError) as exc:
            parser.error(str(exc))
        except KeyboardInterrupt:
            print("\nInterrupted.", file=sys.stderr)
            sys.exit(1)
        return

    if args.registry:
        from bioaccx.registry import print_registry
        print_registry()
        return

    if args.config is None:
        parser.error(
            "a config file is required unless --registry / --convert-head is used, or "
            "--merge / --extract-head are given their inputs directly"
        )

    from bioaccx.config import load_config
    cfg = load_config(args.config)

    if args.validate:
        print("Config parsed successfully.")
        print(f"  Foundation model : {cfg.foundation_model.name} v{cfg.foundation_model.version}")
        data_dir = cfg.dataset.data_dir
        data_dir_str = data_dir if isinstance(data_dir, str) else ", ".join(data_dir)
        print(f"  Dataset          : {data_dir_str}  mode={cfg.dataset.label_mode}")
        print(f"  Classifier       : {cfg.training.classifier}")
        print(f"  Output           : {cfg.output_dir}")
        return

    from bioaccx.train import run, run_dataset_export, run_embeddings, run_extract_head, run_merge
    try:
        if args.merge:
            run_merge(cfg)
        elif args.extract_head:
            run_extract_head(cfg)
        elif args.dataset:
            run_dataset_export(cfg, no_split=args.no_split)
        elif args.embeddings:
            run_embeddings(cfg)
        else:
            run(cfg)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
