#!/usr/bin/env python3
"""Checks M-tile weight reuse for a 128-token W8A16 FFN."""

import argparse
import subprocess
from pathlib import Path


def lower(tool: Path, source: Path, output: Path) -> str:
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [str(tool), "--input", str(source), "--output", str(output),
         "--pipeline", "ftlpu-stablehlo-to-schedule"],
        check=True,
    )
    return output.read_text(encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    schedule = lower(args.tool, args.input, args.output)
    if "tensor<128x64xf16>" not in schedule:
        raise AssertionError("M=128 FFN did not reach Schedule IR")

    load_count = schedule.count("ftlpu.schedule.mxm_load")
    # Projection: 2 N-pairs * 2 K-tiles * 2 hemispheres * (gate + up) = 16.
    # Down projection adds 2 units * 4 K-tiles = 8. The four M=32 tiles reuse
    # the same projection weights, so the total is 24 rather than 72.
    if load_count != 24:
        raise AssertionError(
            f"Expected 24 MXM loads with four-way M-tile reuse, got {load_count}")

    compute_count = schedule.count("ftlpu.schedule.mxm_compute")
    if compute_count < 64:
        raise AssertionError(
            f"Expected four M=32 projection computes per loaded weight tile, got {compute_count}")


if __name__ == "__main__":
    main()
