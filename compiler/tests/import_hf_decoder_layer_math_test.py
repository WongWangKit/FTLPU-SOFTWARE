#!/usr/bin/env python3
"""Checks that the HF reference models LPU arithmetic."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from import_hf_decoder_layer import (  # noqa: E402
    bf16,
    biased_linear,
    biased_rope_linear,
    linear,
    lpu_softmax,
    rms_norm,
)


def test_bf16_dequant_scale() -> None:
    value = bf16(np.asarray([[1.0, 0.0]], dtype=np.float32))
    weight = np.asarray([[121, -121], [0, 0]], dtype=np.int8)
    scale = np.float32(0.0037524607)
    hardware_scale = bf16(np.asarray([scale], dtype=np.float32))[0]

    expected = bf16(value @ bf16(weight.astype(np.float32) * hardware_scale))
    observed = linear(value, weight, float(scale), bf16_scale=True)
    if not np.array_equal(observed, expected):
        raise AssertionError("HF linear reference did not BF16-encode scale")

    fp32_scale = bf16(value @ bf16(weight.astype(np.float32) * scale))
    if np.array_equal(observed, fp32_scale):
        raise AssertionError("test scale does not distinguish BF16 hardware arithmetic")


def test_projection_bias_rounding() -> None:
    value = np.asarray(
        [[1.578125, 0.31640625, 0.51171875, -1.4921875]],
        dtype=np.float32,
    )
    weight = np.asarray(
        [
            [93, -66, -117, 110],
            [-5, -39, 38, -34],
            [105, -47, 16, 59],
            [-119, 124, -117, 120],
        ],
        dtype=np.int8,
    )
    scale = np.float32(0.018651502)
    bias = np.asarray(
        [-0.03299041, -0.08806465, -0.0656285, -0.06720147],
        dtype=np.float32,
    )
    projected = linear(value, weight, float(scale), bf16_scale=True)
    expected = bf16(projected + bf16(bias))
    observed = biased_linear(value, weight, float(scale), bias)
    if not np.array_equal(observed, expected):
        raise AssertionError("projection bias was not added after MXM BF16 output")

    hardware_scale = bf16(np.asarray([scale], dtype=np.float32))[0]
    single_rounding = bf16(
        value @ bf16(weight.astype(np.float32) * hardware_scale) + bf16(bias)
    )
    if np.array_equal(observed, single_rounding):
        raise AssertionError("test does not distinguish the MXM-to-VXM boundary")


def test_rmsnorm_lut_golden() -> None:
    value = bf16(np.asarray([[1.0, 0.5, -0.25, 2.0]], dtype=np.float32))
    weight = bf16(np.asarray([1.0, 0.75, 1.25, 0.5], dtype=np.float32))
    expected = np.asarray(
        [[0.8671875, 0.326171875, -0.271484375, 0.8671875]],
        dtype=np.float32,
    )
    observed = rms_norm(value, weight, 1.0e-6)
    if not np.array_equal(observed, expected):
        raise AssertionError(f"RMSNorm CModel golden differs: {observed}")


def test_causal_softmax_lut_golden() -> None:
    scores = np.asarray(
        [[
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 3.0, 2.0, 1.0],
            [0.5, -0.5, 2.0, 0.0],
            [-1.0, 0.0, 1.0, 2.0],
        ]],
        dtype=np.float32,
    )
    expected = np.asarray(
        [[
            [1.0, 0.0, 0.0, 0.0],
            [0.62109375, 0.376953125, 0.0, 0.0],
            [0.267578125, 0.1630859375, 0.5703125, 0.0],
            [0.1015625, 0.1669921875, 0.275390625, 0.455078125],
        ]],
        dtype=np.float32,
    )
    observed = lpu_softmax(scores, head_dim=4)
    if not np.array_equal(observed, expected):
        raise AssertionError(f"causal softmax CModel golden differs: {observed}")
    if not np.all(np.isfinite(observed)):
        raise AssertionError("causal softmax produced non-finite probabilities")


def test_token_zero_bias_rope() -> None:
    value = bf16(np.asarray([[1.0, 0.5, -0.25, 2.0]], dtype=np.float32))
    weight = np.eye(4, dtype=np.int8)
    bias = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    expected = biased_linear(value, weight, 1.0, bias).reshape(1, 1, 4)
    observed = biased_rope_linear(
        value, weight, 1.0, bias, heads=1, theta=10000.0
    )
    if not np.array_equal(observed, expected):
        raise AssertionError("token-zero Q/K bias was not fused into RoPE")


def main() -> None:
    test_bf16_dequant_scale()
    test_projection_bias_rounding()
    test_rmsnorm_lut_golden()
    test_causal_softmax_lut_golden()
    test_token_zero_bias_rope()


if __name__ == "__main__":
    main()
