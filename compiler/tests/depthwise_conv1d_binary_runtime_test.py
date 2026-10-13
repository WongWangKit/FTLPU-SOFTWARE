#!/usr/bin/env python3
"""Numerically validates the spatial Conv1D tree, feature waves and causal tail."""

import argparse
import os
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile", type=Path, required=True)
    parser.add_argument("--runtime-test", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    template = args.input.read_text(encoding="utf-8")
    for tokens, channels, activation in (
        (1, 32, "silu"), (32, 32, "silu"), (32, 64, "silu"),
        (7, 64, "silu"), (32, 32, "none"),
    ):
        name = f"conv_t{tokens}_c{channels}_{activation}"
        source = args.output_dir / f"{name}.kernel.mlir"
        binary = args.output_dir / f"{name}.ftlpu"
        source.write_text(template.replace("32x32xbf16", f"{tokens}x{channels}xbf16")
                          .replace("32x4xbf16", f"{channels}x4xbf16")
                          .replace('activation = "silu"', f'activation = "{activation}"'),
                          encoding="utf-8")
        subprocess.run([str(args.compile), "--input", str(source),
                        "--output", str(binary)], check=True)
        environment = os.environ.copy()
        if args.trace:
            environment["FTLPU_DEPTHWISE_TRACE_PREFIX"] = str(args.output_dir / name)
        subprocess.run([str(args.runtime_test), str(binary), str(tokens),
                        str(channels), activation], env=environment, check=True)
    print("spatial Conv1D runtime regression passed: five shapes/activation cases")


if __name__ == "__main__":
    main()
