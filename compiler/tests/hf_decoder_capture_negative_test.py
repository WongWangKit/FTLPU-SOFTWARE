#!/usr/bin/env python3
"""Reject semantic mutations of a real saved HF decoder at strict Kernel coverage.

Mutations are diagnostic fixtures outside the sealed bundle. They must never be
saved as verified model artifacts. Parameter positions come from native metadata.
"""
import argparse
import copy
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import ml_dtypes

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from ftlpu_frontend.bundle import validate_bundle, write_json
from ftlpu_frontend.package_bridge import compiler_configuration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--tool", type=Path, required=True)
    parser.add_argument("--target-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=False)
    manifest = validate_bundle(args.bundle)
    profile = json.loads((args.bundle / "backend_profile.json").read_text())
    constants_path, profile_path = compiler_configuration(args.bundle, manifest, profile, root)
    constants = json.loads(constants_path.read_text())
    function = manifest["functions"][0]
    source = (args.bundle / function["mlir"]).read_text()
    common = [str(args.tool), "--pipeline", "ftlpu-stablehlo-to-kernel",
              "--kernel-mode", "complete", "--fold-static-mask", "on",
              "--target-config", str(args.target_config), "--prune-unused-inputs"]
    # First obtain a normalized static mask using the actual upstream pass.
    normalized = root / "normalized.mlir"
    subprocess.run([str(args.tool), "--input", str(args.bundle / function["mlir"]),
        "--output", str(normalized), "--pipeline", "ftlpu-normalize-stablehlo",
        "--fold-static-mask", "on", "--constant-bindings", str(constants_path)], check=True)
    normalized_source = normalized.read_text()
    sequence = profile["sequence_length"]
    cases = []
    modified = copy.deepcopy(constants)
    entry = next(item for item in modified["functions"][function["ir_entry"]]
                 if item.get("logical_target") == profile["rope"]["target"])
    entry["values"][0] *= 1.125
    cases.append(("modified_inv_freq", source, modified, json.loads(profile_path.read_text())))
    changed_theta = json.loads(profile_path.read_text())
    changed_theta["rope_theta"] *= 2
    cases.append(("theta_disagrees_with_captured_frequency", source, constants, changed_theta))
    position_pattern = rf'stablehlo.constant dense<"0x[0-9A-Fa-f]+"> : tensor<1x1x{sequence}xf32>'
    offset_positions = np.arange(1, sequence + 1, dtype="<f4").tobytes().hex()
    offset_source, count = re.subn(position_pattern,
        f'stablehlo.constant dense<"0x{offset_positions}"> : tensor<1x1x{sequence}xf32>', source)
    if count != 1:
        raise ValueError("fixture must identify one actual captured position tensor")
    cases.append(("nonzero_position_offset", offset_source, constants, json.loads(profile_path.read_text())))
    # Keep all positions and outputs; block key 0 in every row, an actual padding
    # mask which cannot be encoded by the old Softmax causal flag.
    mask = np.where(np.arange(sequence)[:, None] >= np.arange(sequence), 0,
                    -float(ml_dtypes.finfo(ml_dtypes.bfloat16).max)).astype(ml_dtypes.bfloat16)
    mask[:, 0] = -float(ml_dtypes.finfo(ml_dtypes.bfloat16).max)
    mask_pattern = rf'stablehlo.constant dense<[^\n]+> : tensor<1x1x{sequence}x{sequence}xbf16>'
    padded, count = re.subn(mask_pattern,
        f'stablehlo.constant dense<"0x{mask.tobytes().hex()}"> : tensor<1x1x{sequence}x{sequence}xbf16>',
        normalized_source)
    if count != 1:
        raise ValueError("fixture must identify one normalized captured mask tensor")
    cases.append(("padding_mask", padded, constants, json.loads(profile_path.read_text())))
    cases.append(("nondefault_dot_precision", source.replace("[DEFAULT, DEFAULT]", "[HIGH, HIGH]"),
                  constants, json.loads(profile_path.read_text())))
    results = []
    for name, text, bindings, compiler_profile in cases:
        folder = root / name
        folder.mkdir()
        ir = folder / "input.mlir"
        ir.write_text(text, encoding="utf-8")
        write_json(folder / "bindings.json", bindings)
        write_json(folder / "profile.json", compiler_profile)
        command = common + ["--input", str(ir), "--output", str(folder / "output.mlir"),
            "--diagnostic-dir", str(folder / "diagnostics"),
            "--constant-bindings", str(folder / "bindings.json"),
            "--backend-profile", str(folder / "profile.json")]
        completed = subprocess.run(command, capture_output=True, text=True)
        (folder / "stderr.txt").write_text(completed.stderr, encoding="utf-8")
        write_json(folder / "command.json", command)
        diagnostic = json.loads((folder / "diagnostics/report.json").read_text())
        expected_stages = {"ftlpu-verify-kernel-coverage"}
        if name == "nondefault_dot_precision":
            expected_stages.add("ftlpu-stablehlo-to-kernel")
        if completed.returncode == 0 or diagnostic["failed_stage"] not in expected_stages:
            raise AssertionError(f"semantic mutation was not refused by Kernel coverage: {name}\n{completed.stderr}")
        results.append({"case": name, "status": "rejected_as_expected",
                        "failure_stage": diagnostic["failed_stage"]})
    write_json(root / "report.json", {"status": "passed", "source_bundle": str(args.bundle),
        "bundle_modified": False, "cases": results})
    print(f"{len(results)} actual captured decoder semantic negative cases passed")


if __name__ == "__main__":
    main()
