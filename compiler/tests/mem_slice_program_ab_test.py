#!/usr/bin/env python3
"""Checks MEM slice profitability and exact logical issue equivalence."""

import argparse
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--translate", type=Path, required=True)
    parser.add_argument("--inspect", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    enabled = args.output_dir / "mem-slice-on.ftlpu"
    disabled = args.output_dir / "mem-slice-off.ftlpu"
    defaulted = args.output_dir / "mem-slice-default.ftlpu"

    common = [
        str(args.translate), "--input", str(args.input),
        "--icu-compression", "macro",
    ]
    subprocess.run(
        common + ["--output", str(enabled),
                  "--mem-slice-program", "on", "--verify-icu-issues"],
        check=True,
    )
    subprocess.run(
        common + ["--output", str(disabled),
                  "--mem-slice-program", "off"],
        check=True,
    )
    subprocess.run(
        common + ["--output", str(defaulted)],
        check=True,
    )
    if enabled.stat().st_size >= disabled.stat().st_size:
        raise RuntimeError(
            "profitable two-body MEM slice fixture did not shrink the binary: "
            f"on={enabled.stat().st_size}, off={disabled.stat().st_size}"
        )

    result = subprocess.run(
        [str(args.inspect), str(enabled), "--compare", str(disabled)],
        check=True,
        text=True,
        capture_output=True,
    )
    if "binary compare result=equivalent" not in result.stdout:
        raise RuntimeError("runtime inspector did not prove A/B equivalence")
    if "mem_slice_program=1 mem_slice_body=2" not in result.stdout:
        raise RuntimeError("compiler did not form the expected two-body program")

    default_result = subprocess.run(
        [str(args.inspect), str(defaulted), "--compare", str(disabled)],
        check=True,
        text=True,
        capture_output=True,
    )
    if "binary compare result=equivalent" not in default_result.stdout:
        raise RuntimeError("default MEM encoding differs from explicit hardware-compatible mode")
    if "mem_slice_program=0" not in default_result.stdout:
        raise RuntimeError("compiler enabled legacy MEM_SLICE_PROGRAM by default")


if __name__ == "__main__":
    main()
