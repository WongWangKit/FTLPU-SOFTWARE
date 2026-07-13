#!/usr/bin/env python3
"""Stages frontend models through an IREE-compatible import boundary.

This tool is intentionally thin. It lets FTLPU accept already-imported MLIR
today, while leaving room to invoke a local IREE toolchain when available.
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


SUPPORTED_INPUTS = {
    "stablehlo",
    "tosa",
    "linalg",
    "mlir",
    "onnx",
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


def build_onnx_import_command(args, output):
    command = [
        args.iree_import_onnx,
        str(args.input),
        "-o",
        str(output),
    ]
    if args.onnx_opset_version is not None:
        command.extend(["--opset-version", str(args.onnx_opset_version)])
    command.extend(args.extra_onnx_arg)
    return command


def stage_common_ir(args):
    if args.input_format == "onnx":
        raise ValueError(
            "ONNX is a binary/protobuf graph. Use --mode=import-onnx first, "
            "or --mode=iree-compile with a local IREE ONNX importer."
        )
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
        choices=["stage-common-ir", "import-onnx", "iree-compile"],
        default="stage-common-ir",
    )
    parser.add_argument(
        "--iree-stage",
        choices=sorted(IREE_STAGES),
        default="flow",
        help="IREE compiler stage used with --mode=iree-compile.",
    )
    parser.add_argument("--iree-compile", default="iree-compile")
    parser.add_argument("--iree-import-onnx", default="iree-import-onnx")
    parser.add_argument("--onnx-opset-version", type=int)
    parser.add_argument("--extra-iree-arg", action="append", default=[])
    parser.add_argument("--extra-onnx-arg", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if not args.input.exists():
        parser.error(f"input file does not exist: {args.input}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "stage-common-ir":
        try:
            stage_common_ir(args)
        except ValueError as exc:
            parser.error(str(exc))
        return 0

    if args.mode == "import-onnx":
        if args.input_format != "onnx":
            parser.error("--mode=import-onnx requires --input-format=onnx")
        command = build_onnx_import_command(args, args.output)
        if args.dry_run:
            print(" ".join(command))
            return 0
        if shutil.which(args.iree_import_onnx) is None:
            parser.error(
                f"{args.iree_import_onnx!r} was not found. Install "
                "iree-base-compiler[onnx] or use --dry-run."
            )
        return subprocess.call(command)

    if args.mode == "iree-compile" and args.input_format == "onnx":
        if args.dry_run:
            imported = args.output.with_suffix(".imported.mlir")
            print(" ".join(build_onnx_import_command(args, imported)))
            compile_args = argparse.Namespace(**vars(args))
            compile_args.input = imported
            compile_args.input_format = "mlir"
            print(" ".join(build_iree_command(compile_args)))
            return 0
        if shutil.which(args.iree_import_onnx) is None:
            parser.error(
                f"{args.iree_import_onnx!r} was not found. Install "
                "iree-base-compiler[onnx] or use --dry-run."
            )
        if shutil.which(args.iree_compile) is None:
            parser.error(
                f"{args.iree_compile!r} was not found. Install IREE or use --dry-run."
            )
        with tempfile.TemporaryDirectory(prefix="ftlpu-onnx-import-") as temp_dir:
            imported = Path(temp_dir) / (args.input.stem + ".mlir")
            import_result = subprocess.call(build_onnx_import_command(args, imported))
            if import_result != 0:
                return import_result
            compile_args = argparse.Namespace(**vars(args))
            compile_args.input = imported
            compile_args.input_format = "mlir"
            return subprocess.call(build_iree_command(compile_args))

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
