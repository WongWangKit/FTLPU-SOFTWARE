"""Numerically verify chain VXM SwiGLU and its Gate/Up/Down handoffs.

The historical test name is retained for existing CTest invocations.
"""

import argparse
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("compile", "runtime-test", "input", "output-dir", "target-config"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    for name in ("FTLPU_STOP_CYCLE", "FTLPU_CMODEL_LOG", "FTLPU_CMODEL_LOG_START",
                 "FTLPU_CMODEL_LOG_CYCLES", "FTLPU_SKIP_QWEN_NUMERIC_CHECK"):
        env.pop(name, None)
    for strategy in ("fused", "tail"):
        binary = args.output_dir / f"ffn_{strategy}.ftlpu"
        subprocess.run([
            str(args.compile), "--input", str(args.input), "--output", str(binary),
            "--input-stage", "stablehlo", "--target-config", str(args.target_config),
            "--weight-bank", "1", "--mxm-execution", "vector",
            "--ffn-schedule", strategy,
        ], check=True, env=env)
        subprocess.run([str(args.runtime_test), str(binary)], check=True, env=env)


if __name__ == "__main__":
    main()
