#!/usr/bin/env python3
"""Builds a standalone, real-checkpoint Hugging Face FFN golden package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from import_hf_decoder_layer import (
    SafeTensorStore,
    bf16,
    quantize_linear,
    write_bf16,
)


def fp16_ftz(value: np.ndarray) -> np.ndarray:
    """Matches VxmDataFormat::round_fp16_ftz for finite FFN values."""
    source = np.asarray(value, dtype=np.float32)
    rounded = source.astype(np.float16).astype(np.float32)
    return np.where(np.abs(source) < np.float32(2.0**-14),
                    np.copysign(np.float32(0.0), source), rounded)


def target_dequantized_weight(weight: np.ndarray, scale: float) -> np.ndarray:
    # The MXM instruction carries a BF16 scale and its dequantizer emits BF16.
    encoded_scale = float(bf16(np.asarray([scale], dtype=np.float32))[0])
    return bf16(weight.astype(np.float32) * np.float32(encoded_scale))


def target_linear(value: np.ndarray, weight: np.ndarray, scale: float) -> np.ndarray:
    """Models vector MXM's 32-wide partials and persistent FP32 accumulator."""
    dequantized = target_dequantized_weight(weight, scale)
    result = np.zeros((value.shape[0], weight.shape[1]), dtype=np.float32)
    for reduction in range(0, value.shape[1], 32):
        result += value[:, reduction:reduction + 32] @ \
            dequantized[reduction:reduction + 32, :]
    return bf16(result)


def make_vxm_lut(input_min: float, width: float, fn) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(256, dtype=np.float32)
    x0 = np.float32(input_min) + indices * np.float32(width)
    y0 = fn(x0).astype(np.float32)
    slopes = (fn(x0 + np.float32(width)).astype(np.float32) - y0) \
        / np.float32(width)
    return slopes.astype(np.float16).astype(np.float32), \
        y0.astype(np.float16).astype(np.float32)


def vxm_lut_lookup(value: np.ndarray, opcode: str) -> np.ndarray:
    value = fp16_ftz(value)
    if opcode == "exp":
        ln2 = np.float32(0.6931471805599453)
        width = np.float32(ln2 / np.float32(256.0))
        input_min = np.float32(-ln2 / np.float32(2.0))
        exponent = np.rint(value * np.float32(1.4426950408889634)).astype(np.int32)
        local = value - exponent.astype(np.float32) * ln2
        multiplier = np.ones_like(value, dtype=np.float32)
        slopes, offsets = make_vxm_lut(input_min, width, np.exp)
        result_exponent = exponent
    elif opcode == "reciprocal":
        width = np.float32(1.0 / 256.0)
        input_min = np.float32(1.0)
        mantissa, exponent = np.frexp(np.abs(value))
        local = mantissa.astype(np.float32) * np.float32(2.0)
        result_exponent = -(exponent.astype(np.int32) - 1)
        multiplier = np.copysign(np.ones_like(value, dtype=np.float32), value)
        slopes, offsets = make_vxm_lut(
            input_min, width, lambda x: np.float32(1.0) / x)
    else:
        raise ValueError(f"unsupported VXM LUT opcode {opcode}")

    position = (local - input_min) / width
    index = np.clip(np.floor(position).astype(np.int64), 0, 255)
    x0 = input_min + index.astype(np.float32) * width
    dx = local - x0
    interpolated = slopes[index] * dx + offsets[index]
    restored = np.ldexp(interpolated * multiplier, result_exponent)
    return fp16_ftz(restored)


def target_swiglu(gate: np.ndarray, up: np.ndarray) -> np.ndarray:
    exponent = vxm_lut_lookup(-gate, "exp")
    reciprocal = vxm_lut_lookup(np.float32(1.0) + exponent, "reciprocal")
    return bf16((reciprocal * gate) * up)


def stablehlo_text(seq_len: int, hidden: int, intermediate: int,
                   scales: dict[str, float]) -> str:
    def scaled_weight(role: str, rows: int, columns: int) -> str:
        return f"""    %{role}_f = stablehlo.convert %{role}_w :
        (tensor<{rows}x{columns}xi8>) -> tensor<{rows}x{columns}xf32>
    %{role}_scale = stablehlo.constant dense<{scales[role]:.9e}> : tensor<f32>
    %{role}_scale_b = stablehlo.broadcast_in_dim %{role}_scale, dims = [] :
        (tensor<f32>) -> tensor<{rows}x{columns}xf32>
    %{role}_scaled = stablehlo.multiply %{role}_f, %{role}_scale_b :
        tensor<{rows}x{columns}xf32>"""

    return f"""module {{
  func.func @hf_ffn_layer0_seq{seq_len}(
      %x: tensor<{seq_len}x{hidden}xbf16>,
      %gate_w: tensor<{hidden}x{intermediate}xi8>,
      %up_w: tensor<{hidden}x{intermediate}xi8>,
      %down_w: tensor<{intermediate}x{hidden}xi8>)
      -> tensor<{seq_len}x{hidden}xbf16> {{
    %x_f = stablehlo.convert %x :
        (tensor<{seq_len}x{hidden}xbf16>) -> tensor<{seq_len}x{hidden}xf32>
{scaled_weight('gate', hidden, intermediate)}
{scaled_weight('up', hidden, intermediate)}
{scaled_weight('down', intermediate, hidden)}
    %gate = stablehlo.dot_general %x_f, %gate_scaled,
      contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] :
      (tensor<{seq_len}x{hidden}xf32>, tensor<{hidden}x{intermediate}xf32>)
        -> tensor<{seq_len}x{intermediate}xf32>
    %up = stablehlo.dot_general %x_f, %up_scaled,
      contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] :
      (tensor<{seq_len}x{hidden}xf32>, tensor<{hidden}x{intermediate}xf32>)
        -> tensor<{seq_len}x{intermediate}xf32>
    %sigmoid = stablehlo.logistic %gate : tensor<{seq_len}x{intermediate}xf32>
    %silu = stablehlo.multiply %gate, %sigmoid : tensor<{seq_len}x{intermediate}xf32>
    %hidden = stablehlo.multiply %silu, %up : tensor<{seq_len}x{intermediate}xf32>
    %down = stablehlo.dot_general %hidden, %down_scaled,
      contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] :
      (tensor<{seq_len}x{intermediate}xf32>, tensor<{intermediate}x{hidden}xf32>)
        -> tensor<{seq_len}x{hidden}xf32>
    %result = stablehlo.convert %down :
        (tensor<{seq_len}x{hidden}xf32>) -> tensor<{seq_len}x{hidden}xbf16>
    return %result : tensor<{seq_len}x{hidden}xbf16>
  }}
}}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=32)
    args = parser.parse_args()

    config = json.loads((args.model_dir / "config.json").read_text(
        encoding="utf-8"))
    hidden = int(config["hidden_size"])
    intermediate = int(config["intermediate_size"])
    if args.seq_len % 32:
        raise ValueError("current LPU FFN requires seq_len divisible by 32")

    store = SafeTensorStore(args.model_dir)
    prefix = f"model.layers.{args.layer}.mlp"
    source_names = {
        role: f"{prefix}.{role}_proj.weight"
        for role in ("gate", "up", "down")
    }
    weights: dict[str, np.ndarray] = {}
    scales: dict[str, float] = {}
    for role, name in source_names.items():
        weights[role], scales[role] = quantize_linear(store.read(name))

    embedding = store.read("model.embed_tokens.weight")
    activation = bf16(embedding[np.arange(args.seq_len) % embedding.shape[0]])
    gate = target_linear(activation, weights["gate"], scales["gate"])
    up = target_linear(activation, weights["up"], scales["up"])
    swiglu = target_swiglu(gate, up)
    golden = target_linear(swiglu, weights["down"], scales["down"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_bf16(args.output_dir / "input.bf16.bin", activation)
    write_bf16(args.output_dir / "golden.bf16.bin", golden)
    for role, value in weights.items():
        value.tofile(args.output_dir / f"{role}.i8.bin")
    (args.output_dir / "ffn.stablehlo.mlir").write_text(
        stablehlo_text(args.seq_len, hidden, intermediate, scales),
        encoding="utf-8")
    metadata = {
        "model": args.model_dir.name,
        "layer": args.layer,
        "seq_len": args.seq_len,
        "hidden_size": hidden,
        "intermediate_size": intermediate,
        "quantization": "symmetric_per_tensor_int8",
        "golden_semantics": "vector_mxm_bf16_dequant_vxm_fp16_lut",
        "scales": scales,
        "source_tensors": source_names,
        "activation_source": "embedding_rows_0_to_seq_len_minus_1",
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
