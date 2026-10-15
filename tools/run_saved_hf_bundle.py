#!/usr/bin/env python3
"""Execute a verified saved XLA bundle; no HF loader or torch.export is needed."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "compiler" / "tools"))
from ftlpu_frontend.bundle import validate_bundle, write_json  # noqa: E402


def run_saved_bundle(root: Path) -> dict:
    import numpy as np
    import torch
    from torch.utils._pytree import tree_flatten
    from ftlpu_frontend.xla_compat import load_saved_program
    from ftlpu_frontend.tensor_io import torch_tensor

    start = time.perf_counter()
    root = Path(root)
    manifest = validate_bundle(root)
    seal = json.loads((root / "bundle_checksums.json").read_text(encoding="utf-8"))
    verification = seal["verification"]
    if verification["status"] != "passed":
        raise ValueError("saved bundle does not have successful numerical verification")
    if "validation_inputs" in manifest:
        inputs = tuple(torch_tensor(np.load(root / spec["file"], allow_pickle=False), spec["dtype"])
                       for spec in manifest["validation_inputs"])
    else:
        inputs = tuple(torch.full(spec["shape"], spec["fill"], dtype=getattr(torch, spec["dtype"]))
                       for spec in manifest["configuration"]["inputs"])
    program = load_saved_program(root)
    with torch.inference_mode():
        values, _ = tree_flatten(program(*inputs))
    references = verification["reference_outputs"]
    if len(values) != len(references):
        raise ValueError("saved output count differs from verified reference")
    errors = []
    tolerance = verification["stablehlo"]
    for value, reference in zip(values, references):
        expected = torch_tensor(np.load(root / reference["file"], allow_pickle=False), reference["dtype"])
        actual = torch_tensor(value).cpu()
        if tolerance.get("criterion") == "legacy_bf16_checkpoint":
            from ftlpu_frontend.profile_validation import check_legacy_bf16
            check_legacy_bf16(actual.float().numpy(), expected.float().numpy())
        else:
            torch.testing.assert_close(actual, expected, rtol=tolerance["rtol"], atol=tolerance["atol"])
        errors.append(float((actual.to(torch.float64) - expected.to(torch.float64)).abs().max())
                      if actual.numel() else 0.0)
    return {"status": "passed", "bundle": str(root.resolve()),
            "mode": "saved_xla_bundle", "hf_loading": False, "torch_capture": False,
            "stablehlo_conversion": False, "artifact_checksums": "passed",
            "max_absolute_errors": errors, "criterion": tolerance.get("criterion", "strict_torch_close"),
            "rtol": tolerance.get("rtol"), "atol": tolerance.get("atol"),
            "elapsed_seconds": round(time.perf_counter() - start, 6)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = run_saved_bundle(args.bundle)
    if args.report is not None:
        if args.report.exists():
            raise ValueError(f"Refusing to overwrite existing reuse report: {args.report}")
        args.report.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.report, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
