#!/usr/bin/env python3
"""Select an exact Qwen2.5 decode program from a bucket manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--past-len", type=int, required=True)
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    matches = [
        bucket
        for bucket in manifest.get("buckets", [])
        if int(bucket["past_len"]) == args.past_len
    ]
    if len(matches) != 1:
        available = ", ".join(
            str(bucket["past_len"]) for bucket in manifest.get("buckets", [])
        )
        raise SystemExit(
            f"no exact decode bucket for past_len={args.past_len}; "
            f"available=[{available}]. A larger bucket cannot be substituted because "
            "RoPE position and the valid attention length are compiled into the program."
        )

    selected = dict(matches[0])
    root = manifest_path.parent
    selected["program"] = str((root / selected["program"]).resolve())
    selected["fixture"] = str((root / selected["fixture"]).resolve())
    print(json.dumps(selected, ensure_ascii=False))


if __name__ == "__main__":
    main()
