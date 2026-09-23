#!/usr/bin/env python3
"""Generate a deterministic Qwen2.5 decode fixture for ICU/CModel."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from qwen2_5_1_5b_decode_reference_test import (
    HIDDEN,
    KV_HEADS,
    HEAD_DIM,
    PREFILL,
    make_fixture,
)

import sys

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from import_hf_decoder_layer import (  # noqa: E402
    decoder_layer_decode_reference,
    decoder_layer_prefill_kv_reference,
    write_bf16,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--capacity", type=int, default=256)
    args = parser.parse_args()
    if args.capacity < PREFILL + 1:
        raise ValueError("KV capacity must hold the prefill and decode token")

    activation, norm0, norm1, weights, scales, config, biases = make_fixture()
    prefill_key, prefill_value = decoder_layer_prefill_kv_reference(
        activation[:PREFILL], norm0, weights, scales, config, biases
    )
    stages: dict[str, np.ndarray] = {}
    output, present_key, present_value = decoder_layer_decode_reference(
        activation[PREFILL:],
        prefill_key,
        prefill_value,
        PREFILL,
        norm0,
        norm1,
        weights,
        scales,
        config,
        biases,
        stage_outputs=stages,
    )

    state_shape = (args.capacity, KV_HEADS, HEAD_DIM)
    past_key = np.zeros(state_shape, dtype=np.float32)
    past_value = np.zeros(state_shape, dtype=np.float32)
    past_key[:PREFILL] = prefill_key
    past_value[:PREFILL] = prefill_value

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_bf16(args.output_dir / "input.bf16.bin", activation[PREFILL:PREFILL + 1])
    write_bf16(args.output_dir / "golden.bf16.bin", output)
    write_bf16(args.output_dir / "input_layernorm.bf16.bin", norm0)
    write_bf16(args.output_dir / "post_attention_layernorm.bf16.bin", norm1)
    write_bf16(args.output_dir / "query_bias.bf16.bin", biases["query"])
    write_bf16(args.output_dir / "key_bias.bf16.bin", biases["key"])
    write_bf16(args.output_dir / "value_bias.bf16.bin", biases["value"])
    write_bf16(args.output_dir / "past_key.bf16.bin", past_key)
    write_bf16(args.output_dir / "past_value.bf16.bin", past_value)
    write_bf16(args.output_dir / "golden.key.bf16.bin", present_key)
    write_bf16(args.output_dir / "golden.value.bf16.bin", present_value)
    # Keep stage references beside the end-to-end golden so a failed decode
    # can be localized without changing the deterministic fixture inputs.
    for name, value in stages.items():
        write_bf16(args.output_dir / f"golden.{name}.bf16.bin", value)
    write_bf16(args.output_dir / "golden.norm0.bf16.bin", stages["normalized"])

    roles = ("query", "key", "value", "output", "gate", "up", "down")
    for role in roles:
        weights[role].astype(np.int8, copy=False).tofile(
            args.output_dir / f"{role}.i8.bin"
        )
    np.asarray([scales[role] for role in roles], dtype="<f4").tofile(
        args.output_dir / "quant_scales.f32.bin"
    )

    print(
        "Qwen2.5 decode CModel fixture generated: "
        f"past={PREFILL}, position={PREFILL}, capacity={args.capacity}, "
        f"output={output.shape}, cache={present_key.shape}"
    )


if __name__ == "__main__":
    main()
