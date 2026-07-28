#!/usr/bin/env python3
"""Imports and packages a reusable-executable Hugging Face decoder stack."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--first-layer", type=int, default=0)
    parser.add_argument(
        "--layer-count",
        type=int,
        help="number of decoder layers; defaults to the remaining HF layers",
    )
    args = parser.parse_args()

    config = json.loads(
        (args.model_dir / "config.json").read_text(encoding="utf-8")
    )
    total_layers = int(config["num_hidden_layers"])
    layer_count = (
        args.layer_count
        if args.layer_count is not None
        else total_layers - args.first_layer
    )
    if args.first_layer < 0 or layer_count <= 0:
        raise ValueError("decoder layer range must be non-empty")
    if args.first_layer + layer_count > total_layers:
        raise ValueError("decoder layer range exceeds the HF model")

    tools_dir = Path(__file__).resolve().parent
    importer = tools_dir / "import_hf_decoder_layer.py"
    packager = tools_dir / "build_decoder_layer_package.py"
    golden_dirs: list[Path] = []
    preceding_output: Path | None = None

    for layer in range(args.first_layer, args.first_layer + layer_count):
        layer_dir = args.output_dir / f"layer{layer}_seq{args.seq_len}"
        command = [
            sys.executable,
            str(importer),
            "--model-dir",
            str(args.model_dir),
            "--output-dir",
            str(layer_dir),
            "--layer",
            str(layer),
            "--seq-len",
            str(args.seq_len),
        ]
        if preceding_output is not None:
            command.extend(["--input-f16", str(preceding_output)])
        subprocess.run(command, check=True)
        golden_dirs.append(layer_dir)
        preceding_output = layer_dir / "golden.f16.bin"

    package_command = [sys.executable, str(packager)]
    for golden_dir in golden_dirs:
        package_command.extend(["--golden-dir", str(golden_dir)])
    package_command.extend([
        "--executable",
        str(args.executable),
        "--output",
        str(args.output),
    ])
    subprocess.run(package_command, check=True)


if __name__ == "__main__":
    main()
