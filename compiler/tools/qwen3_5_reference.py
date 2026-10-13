#!/usr/bin/env python3
"""Numerical reference for a dense Qwen3.5 Gated DeltaNet decoder layer.

The implementation follows the public Transformers Qwen3.5 fallback kernels.
It intentionally uses the recurrent rule for both prefill and decode so the
cache ABI can be verified independently of a future chunked lowering.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Qwen35DeltaConfig:
    hidden_size: int
    intermediate_size: int
    num_key_heads: int
    num_value_heads: int
    key_head_dim: int
    value_head_dim: int
    conv_kernel_size: int = 4
    rms_norm_eps: float = 1.0e-6

    @classmethod
    def from_hf(cls, config: dict[str, object]) -> "Qwen35DeltaConfig":
        return cls(
            hidden_size=int(config["hidden_size"]),
            intermediate_size=int(config["intermediate_size"]),
            num_key_heads=int(config["linear_num_key_heads"]),
            num_value_heads=int(config["linear_num_value_heads"]),
            key_head_dim=int(config["linear_key_head_dim"]),
            value_head_dim=int(config["linear_value_head_dim"]),
            conv_kernel_size=int(config["linear_conv_kernel_dim"]),
            rms_norm_eps=float(config["rms_norm_eps"]),
        )

    @property
    def key_width(self) -> int:
        return self.num_key_heads * self.key_head_dim

    @property
    def value_width(self) -> int:
        return self.num_value_heads * self.value_head_dim

    @property
    def conv_width(self) -> int:
        return 2 * self.key_width + self.value_width

    def validate(self) -> None:
        if self.conv_kernel_size <= 0:
            raise ValueError("Qwen3.5 convolution kernel must be positive")
        if self.num_value_heads % self.num_key_heads:
            raise ValueError("Qwen3.5 value heads must be a multiple of key heads")
        if self.key_head_dim != self.value_head_dim:
            raise ValueError("current LPU DeltaNet path requires equal K/V head dimensions")


@dataclass
class Qwen35DeltaWeights:
    in_proj_qkv: np.ndarray
    in_proj_z: np.ndarray
    in_proj_b: np.ndarray
    in_proj_a: np.ndarray
    conv1d: np.ndarray
    dt_bias: np.ndarray
    a_log: np.ndarray
    delta_norm: np.ndarray
    out_proj: np.ndarray
    input_norm: np.ndarray
    post_attention_norm: np.ndarray
    gate_proj: np.ndarray
    up_proj: np.ndarray
    down_proj: np.ndarray

    def validate(self, config: Qwen35DeltaConfig) -> None:
        config.validate()
        h = config.hidden_size
        i = config.intermediate_size
        c = config.conv_width
        v = config.value_width
        vh = config.num_value_heads
        d = config.value_head_dim
        expected = {
            "in_proj_qkv": (c, h),
            "in_proj_z": (v, h),
            "in_proj_b": (vh, h),
            "in_proj_a": (vh, h),
            "conv1d": (c, config.conv_kernel_size),
            "dt_bias": (vh,),
            "a_log": (vh,),
            "delta_norm": (d,),
            "out_proj": (h, v),
            "input_norm": (h,),
            "post_attention_norm": (h,),
            "gate_proj": (i, h),
            "up_proj": (i, h),
            "down_proj": (h, i),
        }
        for name, shape in expected.items():
            actual = np.asarray(getattr(self, name)).shape
            if actual != shape:
                raise ValueError(f"{name} must have shape {shape}, got {actual}")


@dataclass
class Qwen35DeltaState:
    conv: np.ndarray
    recurrent: np.ndarray

    @classmethod
    def zeros(cls, config: Qwen35DeltaConfig) -> "Qwen35DeltaState":
        # Transformers keeps a kernel-sized rolling buffer.  At a step, the
        # oldest slot is discarded and the current projected QKV is appended.
        conv = np.zeros(
            (config.conv_width, config.conv_kernel_size), dtype=np.float32
        )
        recurrent = np.zeros(
            (
                config.num_value_heads,
                config.key_head_dim,
                config.value_head_dim,
            ),
            dtype=np.float32,
        )
        return cls(conv=conv, recurrent=recurrent)

    def copy(self) -> "Qwen35DeltaState":
        return Qwen35DeltaState(self.conv.copy(), self.recurrent.copy())


def silu(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    return value / (np.float32(1.0) + np.exp(-value, dtype=np.float32))


def softplus(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    # Stable equivalent of log(1 + exp(x)).
    return (
        np.maximum(value, np.float32(0.0))
        + np.log1p(np.exp(-np.abs(value), dtype=np.float32), dtype=np.float32)
    ).astype(np.float32)


def zero_centered_rms_norm(
    value: np.ndarray, weight: np.ndarray, epsilon: float
) -> np.ndarray:
    value32 = np.asarray(value, dtype=np.float32)
    variance = np.mean(value32 * value32, axis=-1, keepdims=True, dtype=np.float32)
    normalized = value32 / np.sqrt(variance + np.float32(epsilon), dtype=np.float32)
    return normalized * (np.float32(1.0) + np.asarray(weight, dtype=np.float32))


def causal_depthwise_conv1d(
    value: np.ndarray,
    weight: np.ndarray,
    state: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Runs the exact kernel-sized rolling-buffer form used by decode."""
    value = np.asarray(value, dtype=np.float32)
    weight = np.asarray(weight, dtype=np.float32)
    state = np.asarray(state, dtype=np.float32).copy()
    if value.ndim != 2:
        raise ValueError("depthwise Conv1D input must have shape [tokens, channels]")
    channels = value.shape[1]
    if weight.ndim != 2 or weight.shape[0] != channels:
        raise ValueError("depthwise Conv1D weight must have shape [channels, kernel]")
    if state.shape != weight.shape:
        raise ValueError("depthwise Conv1D state must match its weight shape")
    output = np.empty_like(value, dtype=np.float32)
    for token, current in enumerate(value):
        state[:, :-1] = state[:, 1:]
        state[:, -1] = current
        output[token] = np.sum(state * weight, axis=-1, dtype=np.float32)
    return silu(output), state


def recurrent_gated_delta_rule(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    decay_log: np.ndarray,
    beta: np.ndarray,
    initial_state: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    query = np.asarray(query, dtype=np.float32)
    key = np.asarray(key, dtype=np.float32)
    value = np.asarray(value, dtype=np.float32)
    decay_log = np.asarray(decay_log, dtype=np.float32)
    beta = np.asarray(beta, dtype=np.float32)
    state = np.asarray(initial_state, dtype=np.float32).copy()
    if query.shape != key.shape:
        raise ValueError("DeltaNet Q and K shapes must match after head replication")
    if query.shape[:2] != value.shape[:2]:
        raise ValueError("DeltaNet Q/K/V token and head dimensions must match")
    if decay_log.shape != query.shape[:2] or beta.shape != query.shape[:2]:
        raise ValueError("DeltaNet decay and beta must have shape [tokens, heads]")
    q_norm = np.sqrt(
        np.sum(query * query, axis=-1, keepdims=True, dtype=np.float32)
        + np.float32(1.0e-6),
        dtype=np.float32,
    )
    k_norm = np.sqrt(
        np.sum(key * key, axis=-1, keepdims=True, dtype=np.float32)
        + np.float32(1.0e-6),
        dtype=np.float32,
    )
    query = query / q_norm / np.sqrt(np.float32(query.shape[-1]), dtype=np.float32)
    key = key / k_norm
    output = np.empty_like(value, dtype=np.float32)
    for token in range(query.shape[0]):
        state *= np.exp(decay_log[token], dtype=np.float32)[:, None, None]
        remembered = np.einsum("hkv,hk->hv", state, key[token], optimize=True)
        delta = (value[token] - remembered) * beta[token, :, None]
        state += np.einsum("hk,hv->hkv", key[token], delta, optimize=True)
        output[token] = np.einsum("hkv,hk->hv", state, query[token], optimize=True)
    return output, state


def gated_rms_norm(
    value: np.ndarray, gate: np.ndarray, weight: np.ndarray, epsilon: float
) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    variance = np.mean(value * value, axis=-1, keepdims=True, dtype=np.float32)
    normalized = value / np.sqrt(variance + np.float32(epsilon), dtype=np.float32)
    return normalized * np.asarray(weight, dtype=np.float32) * silu(gate)


def gated_deltanet(
    hidden_states: np.ndarray,
    weights: Qwen35DeltaWeights,
    config: Qwen35DeltaConfig,
    state: Qwen35DeltaState | None = None,
    stages: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, Qwen35DeltaState]:
    """Runs a Qwen3.5 linear-attention block for prefill or decode."""
    weights.validate(config)
    hidden_states = np.asarray(hidden_states, dtype=np.float32)
    if hidden_states.ndim != 2 or hidden_states.shape[1] != config.hidden_size:
        raise ValueError("hidden_states must have shape [tokens, hidden_size]")
    current = Qwen35DeltaState.zeros(config) if state is None else state.copy()
    qkv_projected = hidden_states @ np.asarray(weights.in_proj_qkv, dtype=np.float32).T
    mixed_qkv, current.conv = causal_depthwise_conv1d(
        qkv_projected, weights.conv1d, current.conv
    )
    key_width = config.key_width
    query = mixed_qkv[:, :key_width].reshape(
        -1, config.num_key_heads, config.key_head_dim
    )
    key = mixed_qkv[:, key_width : 2 * key_width].reshape(
        -1, config.num_key_heads, config.key_head_dim
    )
    value = mixed_qkv[:, 2 * key_width :].reshape(
        -1, config.num_value_heads, config.value_head_dim
    )
    repeats = config.num_value_heads // config.num_key_heads
    query = np.repeat(query, repeats, axis=1)
    key = np.repeat(key, repeats, axis=1)
    z = (hidden_states @ np.asarray(weights.in_proj_z, dtype=np.float32).T).reshape(
        -1, config.num_value_heads, config.value_head_dim
    )
    b = hidden_states @ np.asarray(weights.in_proj_b, dtype=np.float32).T
    a = hidden_states @ np.asarray(weights.in_proj_a, dtype=np.float32).T
    beta = np.float32(1.0) / (np.float32(1.0) + np.exp(-b, dtype=np.float32))
    decay_log = -np.exp(weights.a_log, dtype=np.float32) * softplus(
        a + np.asarray(weights.dt_bias, dtype=np.float32)
    )
    core, current.recurrent = recurrent_gated_delta_rule(
        query, key, value, decay_log, beta, current.recurrent
    )
    gated = gated_rms_norm(
        core, z, weights.delta_norm, config.rms_norm_eps
    ).reshape(-1, config.value_width)
    output = gated @ np.asarray(weights.out_proj, dtype=np.float32).T
    if stages is not None:
        stages.update(
            qkv_projected=qkv_projected,
            conv_qkv=mixed_qkv,
            beta=beta,
            decay_log=decay_log,
            delta_core=core,
            delta_gated=gated,
            delta_output=output,
            conv_state=current.conv,
            recurrent_state=current.recurrent,
        )
    return output.astype(np.float32), current


def linear_decoder_layer(
    activation: np.ndarray,
    weights: Qwen35DeltaWeights,
    config: Qwen35DeltaConfig,
    state: Qwen35DeltaState | None = None,
    stages: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, Qwen35DeltaState]:
    """Runs one complete Qwen3.5 DeltaNet + dense FFN decoder layer."""
    normalized = zero_centered_rms_norm(
        activation, weights.input_norm, config.rms_norm_eps
    )
    attention, final_state = gated_deltanet(
        normalized, weights, config, state, stages
    )
    residual = np.asarray(activation, dtype=np.float32) + attention
    ffn_input = zero_centered_rms_norm(
        residual, weights.post_attention_norm, config.rms_norm_eps
    )
    gate = ffn_input @ np.asarray(weights.gate_proj, dtype=np.float32).T
    up = ffn_input @ np.asarray(weights.up_proj, dtype=np.float32).T
    down = (silu(gate) * up) @ np.asarray(weights.down_proj, dtype=np.float32).T
    output = residual + down
    if stages is not None:
        stages.update(
            input_norm=normalized,
            residual0=residual,
            post_attention_norm=ffn_input,
            swiglu=silu(gate) * up,
            down=down,
            output=output,
        )
    return output.astype(np.float32), final_state
