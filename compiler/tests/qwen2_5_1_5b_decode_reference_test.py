#!/usr/bin/env python3
"""Checks one Qwen2.5-1.5B prefill-to-decode layer step."""

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
    decoder_layer_decode_reference,
    decoder_layer_prefill_kv_reference,
    linear,
    lpu_softmax,
    rms_norm,
)


PREFILL = 32
HIDDEN = 1536
INTERMEDIATE = 8960
QUERY_HEADS = 12
KV_HEADS = 2
HEAD_DIM = 128
ROPE_THETA = 1_000_000.0
EPSILON = 1.0e-6


def sparse_weight(rows: int, columns: int, seed: int) -> np.ndarray:
    """Build a deterministic real-shape INT8 matrix with two values per column."""
    result = np.zeros((rows, columns), dtype=np.int8)
    column = np.arange(columns, dtype=np.int64)
    row0 = (column * (17 + 2 * seed) + 11 * seed) % rows
    row1 = (column * (29 + 2 * seed) + 7 * seed + 1) % rows
    value0 = ((column * 5 + seed * 3) % 15 + 1).astype(np.int8)
    value1 = -((column * 7 + seed * 2) % 13 + 1).astype(np.int8)
    result[row0, column] = value0
    result[row1, column] = value1
    return result


def make_fixture() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
    dict[str, float],
    dict[str, object],
    dict[str, np.ndarray],
]:
    token = np.arange(PREFILL + 1, dtype=np.int32)[:, None]
    feature = np.arange(HIDDEN, dtype=np.int32)[None, :]
    activation = bf16(
        (((token * 13 + feature * 7) % 37) - 18).astype(np.float32)
        / np.float32(32.0)
    )
    index = np.arange(HIDDEN, dtype=np.int32)
    norm0 = bf16(
        np.float32(0.75) + (index % 11).astype(np.float32) / np.float32(64.0)
    )
    norm1 = bf16(
        np.float32(0.875) + (index % 7).astype(np.float32) / np.float32(64.0)
    )
    weights = {
        "query": sparse_weight(HIDDEN, HIDDEN, 1),
        "key": sparse_weight(HIDDEN, KV_HEADS * HEAD_DIM, 2),
        "value": sparse_weight(HIDDEN, KV_HEADS * HEAD_DIM, 3),
        "output": sparse_weight(HIDDEN, HIDDEN, 4),
        "gate": sparse_weight(HIDDEN, INTERMEDIATE, 5),
        "up": sparse_weight(HIDDEN, INTERMEDIATE, 6),
        "down": sparse_weight(INTERMEDIATE, HIDDEN, 7),
    }
    scales = {
        "query": 0.0078125,
        "key": 0.009765625,
        "value": 0.01171875,
        "output": 0.0068359375,
        "gate": 0.005859375,
        "up": 0.0048828125,
        "down": 0.00390625,
    }
    biases = {
        "query": bf16(
            ((np.arange(HIDDEN, dtype=np.int32) % 17) - 8).astype(np.float32)
            / np.float32(512.0)
        ),
        "key": bf16(
            ((np.arange(KV_HEADS * HEAD_DIM, dtype=np.int32) % 13) - 6)
            .astype(np.float32)
            / np.float32(512.0)
        ),
        "value": bf16(
            ((np.arange(KV_HEADS * HEAD_DIM, dtype=np.int32) % 19) - 9)
            .astype(np.float32)
            / np.float32(512.0)
        ),
    }
    config: dict[str, object] = {
        "hidden_size": HIDDEN,
        "intermediate_size": INTERMEDIATE,
        "num_attention_heads": QUERY_HEADS,
        "num_key_value_heads": KV_HEADS,
        "rope_theta": ROPE_THETA,
        "rms_norm_eps": EPSILON,
    }
    return activation, norm0, norm1, weights, scales, config, biases


def last_token_reference(
    activation: np.ndarray,
    norm0: np.ndarray,
    norm1: np.ndarray,
    weights: dict[str, np.ndarray],
    scales: dict[str, float],
    config: dict[str, object],
    biases: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normalized = rms_norm(activation, norm0, EPSILON)
    query = biased_rope_linear(
        normalized,
        weights["query"],
        scales["query"],
        biases["query"],
        QUERY_HEADS,
        ROPE_THETA,
    )
    key = biased_rope_linear(
        normalized,
        weights["key"],
        scales["key"],
        biases["key"],
        KV_HEADS,
        ROPE_THETA,
    )
    value = biased_linear(
        normalized,
        weights["value"],
        scales["value"],
        biases["value"],
    ).reshape(PREFILL + 1, KV_HEADS, HEAD_DIM)

    repeats = QUERY_HEADS // KV_HEADS
    query_last = np.transpose(query[-1:], (1, 0, 2))
    key_all = np.transpose(np.repeat(key, repeats, axis=1), (1, 0, 2))
    value_all = np.transpose(np.repeat(value, repeats, axis=1), (1, 0, 2))
    scores = bf16(query_last @ np.transpose(key_all, (0, 2, 1)))
    probability = lpu_softmax(scores, HEAD_DIM, causal=False)
    context = bf16(
        np.transpose(probability @ value_all, (1, 0, 2)).reshape(1, HIDDEN)
    )
    attention = linear(
        context, weights["output"], scales["output"], bf16_scale=True
    )
    residual = bf16(activation[-1:] + attention)
    normalized_ffn = rms_norm(residual, norm1, EPSILON)
    gate = linear(normalized_ffn, weights["gate"], scales["gate"], True)
    up = linear(normalized_ffn, weights["up"], scales["up"], True)
    swiglu = bf16((gate / (1.0 + np.exp(-gate))) * up)
    down = linear(swiglu, weights["down"], scales["down"], True)
    return bf16(residual + down), key, value


def main() -> None:
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
        stages,
    )
    expected_output, expected_key, expected_value = last_token_reference(
        activation, norm0, norm1, weights, scales, config, biases
    )

    if present_key.shape != (PREFILL + 1, KV_HEADS, HEAD_DIM):
        raise AssertionError(f"unexpected key cache shape {present_key.shape}")
    if present_value.shape != present_key.shape:
        raise AssertionError(f"unexpected value cache shape {present_value.shape}")
    if not np.array_equal(present_key[:PREFILL], prefill_key):
        raise AssertionError("decode modified the prefill key-cache prefix")
    if not np.array_equal(present_value[:PREFILL], prefill_value):
        raise AssertionError("decode modified the prefill value-cache prefix")
    if not np.array_equal(present_key, expected_key):
        raise AssertionError("decode key-cache append differs from full-sequence golden")
    if not np.array_equal(present_value, expected_value):
        raise AssertionError("decode value-cache append differs from full-sequence golden")
    if not np.array_equal(output, expected_output):
        error = float(np.max(np.abs(output - expected_output)))
        raise AssertionError(f"decode layer output mismatch max_error={error}")

    decode_normalized = rms_norm(activation[PREFILL:], norm0, EPSILON)
    position_zero_key = biased_rope_linear(
        decode_normalized,
        weights["key"],
        scales["key"],
        biases["key"],
        KV_HEADS,
        ROPE_THETA,
    )
    if np.array_equal(stages["new_key"], position_zero_key):
        raise AssertionError("decode RoPE ignored the absolute cache position")
    if not np.all(np.isfinite(output)):
        raise AssertionError("decode produced a non-finite layer output")
    if not np.any(output != 0.0):
        raise AssertionError("decode produced an all-zero layer output")

    logical_cache_bytes = (present_key.size + present_value.size) * 2
    if logical_cache_bytes != (PREFILL + 1) * 1024:
        raise AssertionError("unexpected Qwen per-layer BF16 KV-cache size")
    print(
        "Qwen2.5-1.5B single-layer decode reference passed: "
        f"prefill={PREFILL}, position={PREFILL}, output={output.shape}, "
        f"key/value={present_key.shape}, cache_bytes={logical_cache_bytes}, "
        f"output_checksum={float(np.sum(output, dtype=np.float64)):.9f}"
    )


if __name__ == "__main__":
    main()
