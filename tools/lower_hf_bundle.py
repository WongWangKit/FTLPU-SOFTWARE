#!/usr/bin/env python3
"""Lower a converted captured decoder bundle and run existing ModelSession/CModel."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "compiler" / "tools"))
from ftlpu_frontend.package_bridge import lower_decoder_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--profile", type=Path, help="defaults to bundle/backend_profile.json")
    parser.add_argument("--opt", type=Path, required=True)
    parser.add_argument("--translate", type=Path, required=True)
    parser.add_argument("--runtime-test", type=Path, required=True)
    parser.add_argument("--target-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--constant-bindings", type=Path)
    args = parser.parse_args()
    result = lower_decoder_bundle(bundle=args.bundle, profile_path=args.profile or args.bundle / "backend_profile.json",
                                  opt=args.opt, translate=args.translate, runtime_test=args.runtime_test,
                                  target_config=args.target_config, output=args.output_dir,
                                  constant_bindings=args.constant_bindings)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
