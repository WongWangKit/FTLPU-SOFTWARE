#!/usr/bin/env python3
"""Compiles and runs the real quantized SmolLM2-135M tied LM head."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
from import_hf_decoder_layer import (  # noqa: E402
    SafeTensorStore,
    bf16,
    bf16_bits,
    quantize_linear,
)

ROWS = 32
HIDDEN = 576
SHARD = 4096
VOCAB = 49152


def stablehlo(scale: float) -> str:
    return f"""module {{
  func.func @smollm2_135m_lm_head_shard(
      %hidden: tensor<{ROWS}x{HIDDEN}xbf16>,
      %weight: tensor<{HIDDEN}x{SHARD}xi8>)
      -> tensor<{ROWS}x{SHARD}xbf16> {{
    %weight_bf16 = stablehlo.convert %weight :
        (tensor<{HIDDEN}x{SHARD}xi8>)
        -> tensor<{HIDDEN}x{SHARD}xbf16>
    %scale = stablehlo.constant dense<{scale:.9e}> : tensor<bf16>
    %scale_matrix = stablehlo.broadcast_in_dim %scale, dims = [] :
        (tensor<bf16>) -> tensor<{HIDDEN}x{SHARD}xbf16>
    %scaled_weight = stablehlo.multiply %weight_bf16, %scale_matrix :
        tensor<{HIDDEN}x{SHARD}xbf16>
    %logits = stablehlo.dot_general %hidden, %scaled_weight,
        contracting_dims = [1] x [0], precision = [] :
        (tensor<{ROWS}x{HIDDEN}xbf16>,
         tensor<{HIDDEN}x{SHARD}xbf16>)
        -> tensor<{ROWS}x{SHARD}xbf16>
    return %logits : tensor<{ROWS}x{SHARD}xbf16>
  }}
}}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opt", type=Path, required=True)
    parser.add_argument("--translate", type=Path, required=True)
    parser.add_argument("--runtime-test", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-config", type=Path, required=True)
    parser.add_argument("--shard-count", type=int, default=12)
    args = parser.parse_args()

    config = json.loads(
        (args.model_dir / "config.json").read_text(encoding="utf-8")
    )
    if (
        int(config["hidden_size"]) != HIDDEN
        or int(config["vocab_size"]) != VOCAB
        or not bool(config["tie_word_embeddings"])
    ):
        raise RuntimeError("checkpoint is not tied-weight SmolLM2-135M")

    data_dir = args.output_dir / "data"
    ir_dir = args.output_dir / "ir"
    data_dir.mkdir(parents=True, exist_ok=True)
    for layer in ("stablehlo", "kernel", "tensor", "stream", "schedule", "command"):
        (ir_dir / layer).mkdir(parents=True, exist_ok=True)

    store = SafeTensorStore(args.model_dir)
    tied_weight = store.read("model.embed_tokens.weight")
    if tied_weight.shape != (VOCAB, HIDDEN):
        raise RuntimeError(
            f"unexpected tied embedding shape {tied_weight.shape}"
        )
    quantized, original_scale = quantize_linear(tied_weight)
    hardware_scale = float(
        bf16(np.asarray([original_scale], dtype=np.float32))[0]
    )
    token_ids = np.arange(1, ROWS + 1, dtype=np.int64)
    activation = bf16(tied_weight[token_ids])
    golden = np.empty((ROWS, VOCAB), dtype=np.float32)
    for begin in range(0, VOCAB, SHARD):
        dequantized = bf16(
            quantized[:, begin : begin + SHARD].astype(np.float32)
            * hardware_scale
        )
        golden[:, begin : begin + SHARD] = bf16(
            activation @ dequantized
        )

    activation_path = data_dir / "activation.bf16"
    weight_path = data_dir / "lm_head_weight.i8"
    golden_path = data_dir / "golden_logits.bf16"
    bf16_bits(activation).tofile(activation_path)
    quantized.tofile(weight_path)
    bf16_bits(golden).tofile(golden_path)
    metadata = {
        "model": "HuggingFaceTB/SmolLM2-135M",
        "source_tensor": "model.embed_tokens.weight",
        "tied_lm_head": True,
        "activation_token_ids": token_ids.tolist(),
        "shape": {
            "activation": [ROWS, HIDDEN],
            "weight": [HIDDEN, VOCAB],
            "logits": [ROWS, VOCAB],
        },
        "quantization": {
            "scheme": "symmetric_per_tensor_int8",
            "original_scale": original_scale,
            "hardware_bf16_scale": hardware_scale,
            "minimum": int(quantized.min()),
            "maximum": int(quantized.max()),
        },
    }
    (data_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    stablehlo_path = ir_dir / "stablehlo" / "lm_head.stablehlo.mlir"
    stablehlo_path.write_text(stablehlo(hardware_scale), encoding="utf-8")
    pipelines = {
        "kernel": "ftlpu-stablehlo-to-kernel",
        "tensor": "ftlpu-stablehlo-to-tensor",
        "stream": "ftlpu-stablehlo-to-stream",
        "schedule": "ftlpu-stablehlo-to-schedule",
        "command": "ftlpu-stablehlo-to-commands",
    }
    outputs: dict[str, Path] = {}
    for layer, pipeline in pipelines.items():
        output = ir_dir / layer / f"lm_head.{layer}.mlir"
        subprocess.run(
            [
                str(args.opt),
                "--input",
                str(stablehlo_path),
                "--output",
                str(output),
                "--pipeline",
                pipeline,
                "--target-config",
                str(args.target_config),
            ],
            check=True,
        )
        outputs[layer] = output

    binary = args.output_dir / "smollm2_135m_lm_head_shard.ftlpu"
    subprocess.run(
        [
            str(args.translate),
            "--input",
            str(outputs["command"]),
            "--output",
            str(binary),
        ],
        check=True,
    )
    runtime_result = subprocess.run(
        [
            str(args.runtime_test),
            str(binary),
            str(activation_path),
            str(weight_path),
            str(golden_path),
            str(args.shard_count),
        ],
        text=True,
        capture_output=True,
    )
    result_text = runtime_result.stdout + runtime_result.stderr
    print(result_text, end="")
    (args.output_dir / "cmodel_result.txt").write_text(
        result_text, encoding="utf-8"
    )
    runtime_result.check_returncode()


if __name__ == "__main__":
    main()
