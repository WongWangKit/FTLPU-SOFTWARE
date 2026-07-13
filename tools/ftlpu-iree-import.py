#!/usr/bin/env python3
"""Stages frontend MLIR through an IREE-compatible import boundary.

This tool is intentionally thin. It lets FTLPU accept already-imported MLIR
today, while leaving room to invoke a local IREE toolchain when available.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


SUPPORTED_INPUTS = {
    "stablehlo",
    "tosa",
    "linalg",
    "mlir",
}

IREE_STAGES = {
    "flow",
    "stream",
    "hal",
    "vm",
}


def build_iree_command(args):
    command = [
        args.iree_compile,
        str(args.input),
        f"--compile-to={args.iree_stage}",
        "-o",
        str(args.output),
    ]
    if args.input_format != "mlir":
        command.append(f"--iree-input-type={args.input_format}")
    command.extend(args.extra_iree_arg)
    return command


def stage_common_ir(args):
    text = args.input.read_text(encoding="utf-8")
    header = (
        "// FTLPU common IR staged from IREE-compatible frontend input.\n"
        f"// input_format: {args.input_format}\n"
        "// next: lower this MLIR to ftlpu.stream, then ftlpu.icu.\n\n"
    )
    args.output.write_text(header + text, encoding="utf-8")


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--input-format",
        choices=sorted(SUPPORTED_INPUTS),
        default="stablehlo",
        help="Frontend MLIR dialect boundary.",
    )
    parser.add_argument(
        "--mode",
        choices=["stage-common-ir", "iree-compile"],
        default="stage-common-ir",
    )
    parser.add_argument(
        "--iree-stage",
        choices=sorted(IREE_STAGES),
        default="flow",
        help="IREE compiler stage used with --mode=iree-compile.",
    )
    parser.add_argument("--iree-compile", default="iree-compile")
    parser.add_argument("--extra-iree-arg", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if not args.input.exists():
        parser.error(f"input file does not exist: {args.input}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "stage-common-ir":
        stage_common_ir(args)
        return 0

    command = build_iree_command(args)
    if args.dry_run:
        print(" ".join(command))
        return 0

    if shutil.which(args.iree_compile) is None:
        parser.error(
            f"{args.iree_compile!r} was not found. Install IREE or use --dry-run."
        )
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
