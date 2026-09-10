#!/usr/bin/env python3
"""Checks the supported ICU modes and preserves the experimental slice mode."""

import argparse
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--translate", type=Path, required=True)
    parser.add_argument("--inspect", type=Path, required=True)
    parser.add_argument("--runtime-test", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    none = args.output_dir / "none.ftlpu"
    repeat = args.output_dir / "repeat.ftlpu"
    macro = args.output_dir / "macro.ftlpu"
    macro_slice = args.output_dir / "macro-slice.ftlpu"
    legacy_slice = args.output_dir / "legacy-mem-slice-on.ftlpu"

    common = [str(args.translate), "--input", str(args.input)]
    for mode, output in [
        ("none", none),
        ("repeat", repeat),
        ("macro", macro),
        ("macro-slice", macro_slice),
    ]:
        command = common + [
            "--output", str(output), "--icu-compression", mode,
        ]
        if mode == "macro":
            command.append("--verify-icu-issues")
        subprocess.run(command, check=True)

    # Preserve the old independent switch as a compatibility spelling of the
    # fourth mode.
    subprocess.run(
        common + ["--output", str(legacy_slice),
                  "--icu-compression", "macro",
                  "--mem-slice-program", "on"],
        check=True,
    )
    if legacy_slice.read_bytes() != macro_slice.read_bytes():
        raise RuntimeError(
            "legacy --mem-slice-program on did not select macro-slice"
        )

    # macro-slice remains parseable and keeps its compatibility spelling, but
    # its prototype physical encoding and compression result are deliberately
    # outside the active validation matrix.
    for candidate in (repeat, macro):
        result = subprocess.run(
            [str(args.inspect), str(candidate), "--compare", str(none)],
            check=True,
            text=True,
            capture_output=True,
        )
        if "binary compare result=equivalent" not in result.stdout:
            raise RuntimeError(
                f"{candidate.stem} differs from the none baseline"
            )
        if candidate == macro and (
            "macro=2 mem_stream_nd=0" not in result.stdout
        ):
            raise RuntimeError(
                "macro mode did not emit the expected 2-D Macro v1 records"
            )

    subprocess.run(
        [str(args.runtime_test), str(none), str(macro)], check=True
    )


if __name__ == "__main__":
    main()
