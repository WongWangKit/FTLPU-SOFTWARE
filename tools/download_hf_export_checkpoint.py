#!/usr/bin/env python3
"""Populate a shared official HF cache for offline Linux model export."""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True, help="use a fixed HF commit for reproducibility")
    parser.add_argument("--cache-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / ".cache" / "hf-hub")
    args = parser.parse_args()
    path = snapshot_download(
        args.model, revision=args.revision, cache_dir=str(args.cache_dir), token=False,
        allow_patterns=["config.json", "generation_config.json", "model*.safetensors", "model*.safetensors.index.json"],
    )
    print(f"Checkpoint snapshot: {path}")


if __name__ == "__main__":
    main()
