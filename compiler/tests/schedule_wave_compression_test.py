#!/usr/bin/env python3
"""Checks nested MEM repeat/wave compression in Schedule-to-Command lowering."""

import argparse
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opt", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--target-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            str(args.opt),
            "--input",
            str(args.input),
            "--output",
            str(args.output),
            "--pipeline",
            "ftlpu-schedule-to-commands",
            "--target-config",
            str(args.target_config),
        ],
        check=True,
    )
    text = args.output.read_text(encoding="utf-8")
    if text.count("ftlpu.command.mem") != 1:
        raise RuntimeError("expected one compressed MEM command")
    for attribute in (
        "repeat_count = 4",
        "repeat_interval = 1",
        "wave_count = 3",
        "wave_interval = 128",
        "wave_address_stride = 4",
    ):
        if attribute not in text:
            raise RuntimeError(f"missing compressed attribute: {attribute}")
    sxm_commands = [
        line for line in text.splitlines()
        if "ftlpu.command.sxm" in line
    ]
    if len(sxm_commands) != 1:
        raise RuntimeError("expected one compressed SXM command")
    if (
        "repeat_count = 3" not in sxm_commands[0]
        or "repeat_interval = 2" not in sxm_commands[0]
    ):
        raise RuntimeError("SXM command is missing repeat metadata")
    mxm_commands = [
        line for line in text.splitlines()
        if "ftlpu.command.mxm" in line
    ]
    if len(mxm_commands) != 2:
        raise RuntimeError(
            f"expected two compressed MXM commands, got {len(mxm_commands)}")
    iw = next(line for line in mxm_commands if 'opcode = "iw"' in line)
    if (
        "wave_count = 4" not in iw
        or "wave_interval = 1" not in iw
        or "wave_weight_column_stride = 1" not in iw
    ):
        raise RuntimeError("MXM IW command is missing column-wave metadata")
    compute = next(
        line for line in mxm_commands if 'opcode = "compute"' in line)
    if (
        "group_count = 3" not in compute
        or "group_interval = 16" not in compute
    ):
        raise RuntimeError("MXM compute command is missing group metadata")


if __name__ == "__main__":
    main()
