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
    parser = argparse.ArgumentParser(
        prog="bioaccx",
        description="Bioacoustic custom classifier generator",
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

training:
  classifier (keras|sklearn|both)
  keras:  hidden_units, dropout, epochs, batch_size, learning_rate
  sklearn: C, max_iter, solver

output:
  output_path, model_name, model_version,
  output_type (head|full|both), output_format (onnx|tflite|both)
""",
    )
    parser.add_argument(
        "config",
        help="Path to a JSON or YAML config file",
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

    args = parser.parse_args(argv)

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

    from bioaccx.train import run, run_dataset_export
    try:
        if args.dataset:
            run_dataset_export(cfg)
        else:
            run(cfg)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
