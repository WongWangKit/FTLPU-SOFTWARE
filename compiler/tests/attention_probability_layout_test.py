#!/usr/bin/env python3
"""Reject a probability window that overwrites Value before PV consumes it."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from qwen2_5_1_5b_decoder_layer_ir_test import find_dictionary, parse_placement

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from generate_dense_decoder_layer import generate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool", type=Path, required=True)
    parser.add_argument("--target-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = (Path(__file__).resolve().parents[1] / "examples"
              / "qwen2_5_1_5b_decoder_layer" / "decoder_layer_seq32.stablehlo.mlir")
    text = generate(source, name="probability_overlap", hidden=2176,
                    intermediate=1536, query_heads=9, kv_heads=3, head_dim=128,
                    rope_theta=10000.0, rms_norm_eps=1e-5)
    # This head size uses 64 rotary pairs, so 32 denotes only the token
    # extent in tensor shapes and slice limits of this generated fixture.
    text = text.replace("32x", "128x").replace("array<i64: 32,", "array<i64: 128,")
    input_path = args.output_dir / "decoder.stablehlo.mlir"
    input_path.write_text(text, encoding="utf-8")
    target = json.loads(args.target_config.read_text(encoding="utf-8-sig"))
    target["mem"]["rows_per_bank"] = 32768
    target_path = args.output_dir / "target.json"
    target_path.write_text(json.dumps(target), encoding="utf-8")
    output_path = args.output_dir / "decoder.tensor.mlir"
    result = subprocess.run([
        str(args.tool), "--input", str(input_path), "--output", str(output_path),
        "--pipeline", "ftlpu-stablehlo-to-tensor", "--mxm-execution", "vector",
        "--weight-bank", "0", "--target-config", str(target_path),
    ], capture_output=True, text=True)
    if result.returncode:
        if "cannot place the attention probability pack outside" not in result.stderr:
            raise AssertionError(f"unexpected lowering failure: {result.stderr}")
        return
    # A future allocator may find a safe placement instead of rejecting the
    # fixture. Either outcome must preserve the live Value tensor.
    tensor = output_path.read_text(encoding="utf-8")
    value = parse_placement(find_dictionary(tensor, "value"))
    probability = parse_placement(find_dictionary(tensor, "probability_pack"))
    overlap = (
        value["bank"] == probability["bank"]
        and set(value["slices"]).intersection(probability["slices"])
        and value["base"] < probability["base"] + probability["count"]
        and probability["base"] < value["base"] + value["count"]
    )
    if overlap:
        raise AssertionError(f"probability pack overwrites live Value: {probability} vs {value}")


if __name__ == "__main__":
    main()
