#!/usr/bin/env python3
"""Complete dense DeltaNet decoder test: actual compiled ICU/FU arithmetic."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from qwen3_5_compiled_layer import CompiledStages, bf16
from qwen3_5_reference import Qwen35DeltaConfig, linear_decoder_layer
from qwen3_5_gated_deltanet_reference_test import random_weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    c = Qwen35DeltaConfig(64, 128, 1, 2, 32, 32)
    rng = np.random.default_rng(3506)
    weights = random_weights(c, rng)
    packed, scales = {}, {}
    for name in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj", "gate_proj", "up_proj", "down_proj"):
        w = getattr(weights, name)
        scale = float(np.max(np.abs(w)) / 127)
        packed[name] = np.clip(np.rint(w / scale), -127, 127).astype(np.int8)
        scales[name] = scale
        setattr(weights, name, packed[name].astype(np.float32) * np.float32(scale))
    for name in ("conv1d", "delta_norm", "input_norm", "post_attention_norm"):
        setattr(weights, name, bf16(getattr(weights, name)))
    x = bf16(rng.normal(0, 0.2, (32, c.hidden_size)).astype(np.float32))
    stages = CompiledStages(args.compile.resolve(), args.runner.resolve(), args.output_dir.resolve())
    # Exercise nonlinear controls across both branches, not only narrow random
    # projection values near zero. Golden math remains isolated from execution.
    probe = np.array([[-12., -4., -1., 0., 1., 4., 12.]], np.float32)
    actual_softplus = stages.softplus("softplus_probe", probe)
    expected_softplus = np.logaddexp(np.float32(0), probe)
    np.testing.assert_allclose(actual_softplus, expected_softplus, rtol=0.002, atol=0.0003)
    probe_stage_count = len(stages.stages)
    actual, actual_stages = stages.layer(x, weights, packed, scales, c)
    golden_stages = {}
    golden, state = linear_decoder_layer(x, weights, c, stages=golden_stages)
    errors = {}
    for name, value in actual_stages.items():
        expected = golden_stages[name]
        if value.shape != expected.shape:
            raise AssertionError(f"{name}: {value.shape} != {expected.shape}")
        errors[name] = float(np.max(np.abs(value - expected)))
        # BF16 stage boundaries and LUT interpolation differ from the all-FP32
        # NumPy reference. Validate the internal stages as well as the residual.
        tolerance = 0.035 if "norm" in name else 0.015
        np.testing.assert_allclose(value, expected, rtol=0.12, atol=tolerance, err_msg=name)
    np.testing.assert_allclose(actual, golden, rtol=0.03, atol=0.008)
    actual.astype("<f4").tofile(args.output_dir / "output.f32.bin")
    golden.astype("<f4").tofile(args.output_dir / "golden.f32.bin")
    state.recurrent.astype("<f4").tofile(args.output_dir / "golden.state.f32.bin")
    summary = dict(status="passed", seq_len=32, hidden_size=c.hidden_size,
                   intermediate_size=c.intermediate_size, key_heads=1, value_heads=2,
                   head_dim=32, execution="serial_compiled_stages_with_host_layout_transfers",
                   arithmetic_backend="CModel ICU/FU", synthetic_weights=True,
                   max_abs_errors=errors, stage_count=len(stages.stages),
                   probe_stage_count=probe_stage_count,
                   decoder_stage_count=len(stages.stages) - probe_stage_count,
                   decoder_binary_bytes=sum(s["binary_bytes"] for s in stages.stages[probe_stage_count:]),
                   total_binary_bytes=sum(s["binary_bytes"] for s in stages.stages))
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
