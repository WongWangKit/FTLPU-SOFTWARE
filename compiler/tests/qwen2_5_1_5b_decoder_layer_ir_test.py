#!/usr/bin/env python3
"""Lowers a Qwen2.5-1.5B decoder layer through the public FTLPU IRs."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def lower(tool: Path, target: Path, source: Path, output: Path,
          pipeline: str, weight_bank: int | None) -> str:
    command = [
        str(tool), "--input", str(source), "--output", str(output),
        "--pipeline", pipeline, "--mxm-execution", "block8",
        "--ffn-schedule", "tail", "--target-config", str(target),
        "--rmsnorm-strategy", "vxm-feedback",
    ]
    if weight_bank is not None:
        command += ["--weight-bank", str(weight_bank)]
    subprocess.run(command, check=True)
    return output.read_text(encoding="utf-8")


def require(text: str, markers: tuple[str, ...], layer: str) -> None:
    missing = [marker for marker in markers if marker not in text]
    if missing:
        raise AssertionError(f"{layer} IR is missing {missing}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--target-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weight-bank", type=int)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.input, args.output_dir / "decoder_layer.stablehlo.mlir")

    stablehlo = args.input.read_text(encoding="utf-8")
    require(stablehlo, (
        "tensor<128x1536xbf16>", "tensor<1536x8960xi8>",
        "tensor<8960x1536xi8>", "tensor<128x12x128xbf16>",
        "tensor<128x2x128xbf16>", "dense<1.000000e+06>",
        "dense<1.000000e-06>", "stablehlo.compare GE",
    ), "StableHLO")

    kernel = lower(args.tool, args.target_config, args.input,
                   args.output_dir / "decoder_layer.kernel.mlir",
                   "ftlpu-stablehlo-to-kernel", args.weight_bank)
    require(kernel, (
        "ftlpu.kernel.rms_norm", "ftlpu.kernel.rope",
        "ftlpu.kernel.softmax", "ftlpu.kernel.batch_matmul",
        "ftlpu.kernel.swish", "head_dim = 128 : i64",
        "query_heads = 12 : i64", "kv_heads = 2 : i64",
        "theta = 1.000000e+06 : f32", "n = 8960 : i64",
    ), "Kernel")

    tensor = lower(args.tool, args.target_config, args.input,
                   args.output_dir / "decoder_layer.tensor.mlir",
                   "ftlpu-stablehlo-to-tensor", args.weight_bank)
    require(tensor, (
        "ftlpu.tensor.rms_norm_task", "ftlpu.tensor.projection_task",
        "ftlpu.tensor.rope_task", "ftlpu.tensor.softmax_task",
        "ftlpu.tensor.swish_task", 'kind = "w8a16_mxm_weight_striped"',
        'kind = "fp16_mxm_distributed_16"',
    ), "Tensor")
    if args.weight_bank is not None:
        require(tensor, (
            f"bank = {args.weight_bank} : i64, base_row = 0 : i64",
            "instruction_count = 4608 : i64, kind = \"w8a16_attention_weight_striped\"",
            "base_row = 4608 : i64, hemisphere = \"both\", instruction_count = 768 : i64",
            "base_row = 5376 : i64, hemisphere = \"both\", instruction_count = 768 : i64",
            "base_row = 6144 : i64, hemisphere = \"both\", instruction_count = 4608 : i64",
            "instruction_count = 26880 : i64, kind = \"w8a16_mxm_weight_striped\"",
            "instruction_count = 13440 : i64, kind = \"w8a16_block8_weight_wave_striped\"",
            "word = 32384 : i64", "word = 32576 : i64",
        ), "Paged Tensor")

    stream = lower(args.tool, args.target_config, args.input,
                   args.output_dir / "decoder_layer.stream.mlir",
                   "ftlpu-stablehlo-to-stream", args.weight_bank)
    require(stream, (
        "ftlpu.stream.rms_norm_task", "ftlpu.stream.projection_task",
        "ftlpu.stream.rope_task", "ftlpu.stream.softmax_task",
        "ftlpu.stream.swish_task", "ftlpu.stream.batch_matmul_task",
        "head_dim = 128 : i64", "query_heads = 12 : i64",
        "kv_heads = 2 : i64", "rope_theta = 1.000000e+06 : f32",
        "stream_count = 16 : i64",
    ), "Stream")
    for legacy in ("ftlpu.kernel.attention", "ftlpu.tensor.attention",
                   "ftlpu.stream.attention", "ftlpu.stream.ffn"):
        if legacy in stream:
            raise AssertionError(f"legacy compound op remained: {legacy}")


if __name__ == "__main__":
    main()
