#!/usr/bin/env python3
"""Checks Schedule MEM blocked-outer address lowering and validation."""

import argparse
import re
import subprocess
from pathlib import Path


def run(opt: Path, source: Path, output: Path, target: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            str(opt),
            "--input", str(source),
            "--output", str(output),
            "--pipeline", "ftlpu-schedule-to-commands",
            "--target-config", str(target),
        ],
        text=True,
        capture_output=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opt", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--target-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    command = args.output_dir / "command.mlir"
    result = run(args.opt, args.input, command, args.target_config)
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout)
    text = command.read_text(encoding="utf-8")
    expected = [
        67, 131075, 33619972, 8388679, 163840,
        8491008, 32843, 33554496, 65536,
    ]
    packets = [
        [int(value.strip()) for value in match.split(",")]
        for match in re.findall(
            r"ftlpu\.command\.mem_3d.*?words = \[([^]]+)\]", text)
    ]
    if packets != [expected, expected]:
        raise RuntimeError(f"unexpected blocked-outer MEM packets: {packets}")

    invalid = args.output_dir / "partial.schedule.mlir"
    invalid.write_text(
        args.input.read_text(encoding="utf-8").replace(
            "      outer_inner_stride = 4 : i64,\n", ""),
        encoding="utf-8",
    )
    rejected = run(
        args.opt, invalid, args.output_dir / "partial.command.mlir",
        args.target_config)
    if rejected.returncode == 0 or "must be specified together" not in rejected.stderr:
        raise RuntimeError("partial blocked-outer MEM metadata was not rejected")


if __name__ == "__main__":
    main()
