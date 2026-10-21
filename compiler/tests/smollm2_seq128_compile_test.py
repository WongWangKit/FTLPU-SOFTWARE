#!/usr/bin/env python3
"""Compile the complete seq128 decoder against finite LPU resources."""

import argparse
from pathlib import Path
import re
import subprocess


def run(command):
    result = subprocess.run([str(x) for x in command], capture_output=True,
                            text=True, encoding="utf-8", errors="replace")
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {command}\n"
                           f"{result.stdout}{result.stderr}")
    return result.stdout + result.stderr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("compile", "inspect", "input", "target-config", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for mode in ("none", "raw-macro"):
        binary = args.output_dir / f"seq128-{mode}.ftlpu"
        run([args.compile, "--input", args.input, "--input-stage", "stablehlo",
             "--target-config", args.target_config, "--weight-bank", "1",
             "--kv-cache-capacity", "256", "--mxm-execution", "vector",
             "--projection-rope-overlap", "off", "--ffn-schedule", "tail",
             "--icu-compression", mode, "--output", binary])
        report = run([args.inspect, binary, "--all-queues"])
        binary.with_suffix(".inspect.txt").write_text(report, encoding="utf-8")
        if "deployable=yes" not in report or "overflow_queues=0" not in report:
            raise AssertionError(f"{mode} exceeded the target i-MEM capacity")
        if mode == "raw-macro":
            resources = set()
            for line in report.splitlines():
                if not line.startswith("binary queue_occupancy "):
                    continue
                fields = dict(re.findall(r"(\w+)=([^ ]+)", line))
                resource = fields["resource"]
                if resource in {"mem", "mxm_load", "mxm_compute", "mxm_dequant"}:
                    if int(fields["commands"]) != int(fields["macro"]):
                        raise AssertionError(f"raw Macro silently fell back: {line}")
                    resources.add(resource)
            if len(resources) != 4:
                raise AssertionError("seq128 did not exercise every MEM/MXM queue kind")
        print(f"seq128 {mode}: full compilation and binary validation passed; "
              f"{binary.stat().st_size} bytes", flush=True)


if __name__ == "__main__":
    main()
