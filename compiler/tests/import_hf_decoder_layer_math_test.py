#!/usr/bin/env python3
"""Checks that the HF reference models LPU dequant arithmetic."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from import_hf_decoder_layer import bf16, linear  # noqa: E402


def main() -> None:
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


if __name__ == "__main__":
    main()
