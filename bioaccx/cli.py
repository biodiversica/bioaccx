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


def main(argv: list[str] | None = None) -> None:
    """Parse CLI arguments and dispatch to the appropriate pipeline entry point.

    Modes (mutually exclusive flags):
      (none)     — full training pipeline: load dataset, extract embeddings,
                   train classifier(s), export models, write reports.
      --validate — parse and validate the config only; no IO beyond reading the file.
      --dataset  — load, split, and export chunked WAV files; no training.
      --embeddings — compute the embedding database + UMAP outputs; no training.
      --merge    — merge an existing backbone + head into a single full ONNX model.
      --extract_head — extract the head from a full custom BirdNET-Analyzer TFLite model
                   and re-export it head-only (no backbone conversion).

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
        action="store_true",
        help=(
            "Merge an existing ONNX backbone and ONNX or TFLite classifier head into a single full ONNX model. "
            "Backbone can be a local file (foundation_model.path) or downloaded from HuggingFace "
            "(foundation_model.source=huggingface + hf_repo). "
            "Requires output.head_path (head) in the config. "
            "A TFLite head is converted to ONNX automatically before merging. "
            "No dataset or training is performed."
        ),
    )
    parser.add_argument(
        "--extract_head",
        action="store_true",
        help=(
            "Extract the classifier head from a full BirdNET-Analyzer TFLite model and "
            "re-export it as a head-only model (ONNX and/or TFLite per output.output_format). "
            "The head weights are read directly from the flatbuffer — the backbone is never "
            "converted or run. Requires output.extract_from (path to the .tflite model). "
            "No dataset or training is performed."
        ),
    )

    args = parser.parse_args(argv)

    if args.registry:
        from bioaccx.registry import print_registry
        print_registry()
        return

    if args.config is None:
        parser.error("a config file is required unless --registry is used")

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
            run_dataset_export(cfg)
        elif args.embeddings:
            run_embeddings(cfg)
        else:
            run(cfg)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
