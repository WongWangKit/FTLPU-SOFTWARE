#!/usr/bin/env python3
"""Checks Qwen3.5 kernel-4 Conv1D and recurrent DeltaNet state semantics."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from qwen3_5_reference import (  # noqa: E402
    Qwen35DeltaConfig,
    Qwen35DeltaState,
    Qwen35DeltaWeights,
    causal_depthwise_conv1d,
    linear_decoder_layer,
)


def random_weights(
    config: Qwen35DeltaConfig, rng: np.random.Generator
) -> Qwen35DeltaWeights:
    def matrix(rows: int, columns: int, scale: float = 0.04) -> np.ndarray:
        return rng.normal(0.0, scale, (rows, columns)).astype(np.float32)

    return Qwen35DeltaWeights(
        in_proj_qkv=matrix(config.conv_width, config.hidden_size),
        in_proj_z=matrix(config.value_width, config.hidden_size),
        in_proj_b=matrix(config.num_value_heads, config.hidden_size),
        in_proj_a=matrix(config.num_value_heads, config.hidden_size),
        conv1d=matrix(config.conv_width, config.conv_kernel_size, 0.08),
        dt_bias=rng.normal(0.1, 0.05, config.num_value_heads).astype(np.float32),
        a_log=rng.normal(-1.0, 0.05, config.num_value_heads).astype(np.float32),
        delta_norm=rng.normal(1.0, 0.03, config.value_head_dim).astype(np.float32),
        out_proj=matrix(config.hidden_size, config.value_width),
        input_norm=rng.normal(0.0, 0.02, config.hidden_size).astype(np.float32),
        post_attention_norm=rng.normal(0.0, 0.02, config.hidden_size).astype(np.float32),
        gate_proj=matrix(config.intermediate_size, config.hidden_size),
        up_proj=matrix(config.intermediate_size, config.hidden_size),
        down_proj=matrix(config.hidden_size, config.intermediate_size),
    )


def check_depthwise_conv() -> None:
    value = np.arange(1, 13, dtype=np.float32).reshape(4, 3) / 8.0
    weight = np.asarray(
        [[1.0, 2.0, 3.0, 4.0], [0.5, -0.5, 1.0, 2.0], [2.0, 0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    output, state = causal_depthwise_conv1d(
        value, weight, np.zeros_like(weight)
    )
    padded = np.pad(value, ((3, 0), (0, 0)))
    expected_linear = np.stack(
        [
            np.sum(padded[token : token + 4].T * weight, axis=-1)
            for token in range(value.shape[0])
        ]
    )
    expected = expected_linear / (1.0 + np.exp(-expected_linear))
    np.testing.assert_allclose(output, expected, rtol=2.0e-6, atol=2.0e-6)
    np.testing.assert_array_equal(state, padded[-4:].T)


def main() -> None:
    check_depthwise_conv()
    config = Qwen35DeltaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_key_heads=2,
        num_value_heads=4,
        key_head_dim=8,
        value_head_dim=8,
        conv_kernel_size=4,
    )
    rng = np.random.default_rng(35)
    weights = random_weights(config, rng)
    activation = rng.normal(0.0, 0.2, (32, config.hidden_size)).astype(np.float32)

    prefill_stages: dict[str, np.ndarray] = {}
    prefill_output, prefill_state = linear_decoder_layer(
        activation, weights, config, stages=prefill_stages
    )

    decode_state = Qwen35DeltaState.zeros(config)
    decode_outputs = []
    for token in activation:
        output, decode_state = linear_decoder_layer(
            token[None, :], weights, config, decode_state
        )
        decode_outputs.append(output[0])
    decode_output = np.stack(decode_outputs)

    np.testing.assert_allclose(decode_output, prefill_output, rtol=2.0e-5, atol=2.0e-6)
    np.testing.assert_allclose(
        decode_state.conv, prefill_state.conv, rtol=3.0e-5, atol=1.0e-7
    )
    np.testing.assert_allclose(
        decode_state.recurrent,
        prefill_state.recurrent,
        rtol=2.0e-5,
        atol=2.0e-6,
    )
    if prefill_stages["conv_qkv"].shape != (32, config.conv_width):
        raise AssertionError("Qwen3.5 Conv1D stage has the wrong shape")
    if prefill_stages["recurrent_state"].shape != (
        config.num_value_heads,
        config.key_head_dim,
        config.value_head_dim,
    ):
        raise AssertionError("Qwen3.5 recurrent state has the wrong shape")

    official_state_bytes = 32 * 128 * 128 * 4
    official_conv_bytes = (2 * 16 * 128 + 32 * 128) * 4 * 2
    if official_state_bytes != 2 * 1024 * 1024:
        raise AssertionError("Qwen3.5-9B recurrent-state size changed")
    if official_conv_bytes != 64 * 1024:
        raise AssertionError("Qwen3.5-9B BF16 Conv1D-state size changed")
    print(
        "Qwen3.5 DeltaNet reference passed: seq32 prefill == 32 decode steps, "
        "kernel=4, conv_state=64 KiB/layer, recurrent_state=2 MiB/layer"
    )


if __name__ == "__main__":
    main()
