#!/usr/bin/env python3
"""Export a configured Hugging Face tensor entry to a validated StableHLO bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "compiler" / "tools"))
from ftlpu_frontend import export_model  # noqa: E402
from ftlpu_frontend.bundle import validate_bundle  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    mode.add_argument("--validate-bundle", type=Path, help="validate an existing bundle without loading HF")
    parser.add_argument("--verify-numerics", action="store_true")
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--backend-tool", type=Path)
    parser.add_argument("--backend-pipeline", default="ftlpu-stablehlo-to-kernel")
    args = parser.parse_args()
    if args.validate_bundle is not None:
        if args.output is not None:
            parser.error("--output is only valid with --config")
        validate_bundle(args.validate_bundle)
        print(f"Bundle validation passed: {args.validate_bundle.resolve()}")
        return
    if args.output is None:
        parser.error("--config and --output are required for export")
    try:
        config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, ValueError) as error:
        parser.error(f"cannot read JSON configuration {args.config}: {error}")
    export_model(config, args.output,
                 verify_numerics=args.verify_numerics, rtol=args.rtol, atol=args.atol,
                 backend_tool=args.backend_tool, backend_pipeline=args.backend_pipeline)


if __name__ == "__main__":
    main()
