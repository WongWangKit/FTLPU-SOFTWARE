#!/usr/bin/env python3
"""Imports one supported Hugging Face decoder layer without PyTorch."""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path

import numpy as np


SUPPORTED_MODEL_TYPES = {
    "llama",
    "qwen2",
    "qwen3",
    "qwen3_moe",
    "ministral3",
    "phi3",
    "smollm3",
    "minicpm",
    "glm",
}


def decoder_config(root_config: dict[str, object]) -> dict[str, object]:
    """Returns the text decoder config from a text or multimodal wrapper."""
    if root_config.get("model_type") is None:
        architectures = root_config.get("architectures", [])
        if any(str(name).startswith("MiniCPM") for name in architectures):
            config = dict(root_config)
            config["model_type"] = "minicpm"
            return config
    if root_config.get("model_type") != "mistral3":
        return root_config
    nested = root_config.get("text_config")
    if not isinstance(nested, dict):
        raise ValueError("Mistral3 config is missing text_config")
    config = dict(nested)
    config.setdefault("model_type", "ministral3")
    return config


def checkpoint_prefixes(
    root_config: dict[str, object], layer: int
) -> tuple[str, str]:
    """Returns decoder-layer and embedding prefixes used by HF checkpoints."""
    if root_config.get("model_type") == "mistral3":
        return (
            f"language_model.model.layers.{layer}",
            "language_model.model.embed_tokens.weight",
        )
    return f"model.layers.{layer}", "model.embed_tokens.weight"


class SafeTensorFile:
    def __init__(self, path: Path) -> None:
        self.path = path
        with path.open("rb") as stream:
            header_size = struct.unpack("<Q", stream.read(8))[0]
            self.data_base = 8 + header_size
            self.header = json.loads(stream.read(header_size))

    def read(self, name: str) -> np.ndarray:
        metadata = self.header[name]
        begin, end = metadata["data_offsets"]
        with self.path.open("rb") as stream:
            stream.seek(self.data_base + begin)
            data = stream.read(end - begin)
        dtype = metadata["dtype"]
        shape = tuple(metadata["shape"])
        if dtype == "BF16":
            words = np.frombuffer(data, dtype="<u2").astype(np.uint32)
            return (words << 16).view(np.float32).reshape(shape)
        numpy_dtype = {
            "F16": "<f2",
            "F32": "<f4",
            "I8": "i1",
        }.get(dtype)
        if numpy_dtype is None:
            raise ValueError(f"unsupported safetensors dtype {dtype}")
        return np.frombuffer(data, dtype=numpy_dtype).reshape(shape).astype(np.float32)


class SafeTensorStore:
    def __init__(self, model_dir: Path) -> None:
        index_path = model_dir / "model.safetensors.index.json"
        if index_path.exists():
            index = json.loads(index_path.read_text(encoding="utf-8"))
            self.weight_map = index["weight_map"]
            paths = {model_dir / name for name in self.weight_map.values()}
        else:
            path = model_dir / "model.safetensors"
            if not path.exists():
                raise FileNotFoundError(
                    f"{model_dir} has no model.safetensors checkpoint"
                )
            reader = SafeTensorFile(path)
            self.weight_map = {
                name: path.name
                for name in reader.header
                if name != "__metadata__"
            }
            paths = {path}
        self.readers = {path.name: SafeTensorFile(path) for path in paths}

    def read(self, name: str) -> np.ndarray:
        filename = self.weight_map.get(name)
        if filename is None:
            raise KeyError(f"checkpoint tensor is missing: {name}")
        return self.readers[filename].read(name)

    def contains(self, name: str) -> bool:
        return name in self.weight_map


def quantize_linear(weight: np.ndarray) -> tuple[np.ndarray, float]:
    # Hugging Face stores Linear weights as [out, in]; StableHLO consumes [in, out].
    maximum = float(np.max(np.abs(weight)))
    scale = maximum / 127.0 if maximum else 1.0
    quantized = np.clip(np.rint(weight / scale), -127, 127).astype(np.int8)
    return np.ascontiguousarray(quantized.T), scale


def quantize_expert_linear(weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quantizes stacked HF expert weights with one scale per expert."""
    if weight.ndim != 3:
        raise ValueError("stacked expert weight must have shape [experts, out, in]")
    quantized = []
    scales = []
    for expert in weight:
        packed, scale = quantize_linear(expert)
        quantized.append(packed)
        scales.append(scale)
    return np.ascontiguousarray(np.stack(quantized)), np.asarray(scales, dtype=np.float32)


def bf16_bits(value: np.ndarray) -> np.ndarray:
    words = np.asarray(value, dtype=np.float32).view(np.uint32)
    rounding_bias = np.uint32(0x7FFF) + ((words >> 16) & 1)
    return ((words + rounding_bias) >> 16).astype("<u2")


def bf16(value: np.ndarray) -> np.ndarray:
    words = bf16_bits(value).astype(np.uint32) << 16
    return words.view(np.float32)


def fp16_ftz(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        rounded = value.astype(np.float16).astype(np.float32)
    minimum_normal = np.float32(2.0 ** -14)
    rounded = np.where(
        np.isfinite(value) & (np.abs(value) < minimum_normal),
        np.copysign(np.float32(0.0), value),
        rounded,
    )
    return np.where(
        np.isfinite(rounded) & (np.abs(rounded) < minimum_normal),
        np.copysign(np.float32(0.0), rounded),
        rounded,
    ).astype(np.float32)


def fma32(lhs: np.ndarray, rhs: np.ndarray, addend: np.ndarray) -> np.ndarray:
    # Float64 is wide enough to evaluate a float32 multiply-add before the
    # single rounding performed by the CModel's std::fma implementation.
    return (
        np.asarray(lhs, dtype=np.float64)
        * np.asarray(rhs, dtype=np.float64)
        + np.asarray(addend, dtype=np.float64)
    ).astype(np.float32)


def write_bf16(path: Path, value: np.ndarray) -> None:
    bf16_bits(value).tofile(path)


def read_bf16(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    words = np.fromfile(path, dtype="<u2").astype(np.uint32) << 16
    return words.view(np.float32).reshape(shape)


def rms_norm(value: np.ndarray, weight: np.ndarray, epsilon: float) -> np.ndarray:
    squares = value * value
    square_sum = np.add.accumulate(
        squares, axis=-1, dtype=np.float32
    )[..., -1:]
    stored_sum = bf16(square_sum)
    mean_square = (
        stored_sum * np.float32(1.0 / value.shape[-1])
        + np.float32(epsilon)
    )
    factor = vxm_lut(mean_square, "rsqrt")
    return bf16((value * bf16(weight)) * factor)


def attention_head_dim(config: dict[str, object]) -> int:
    """Returns the explicit attention head width used by the checkpoint."""
    query_heads = int(config["num_attention_heads"])
    hidden = int(config["hidden_size"])
    if query_heads <= 0:
        raise ValueError("attention head count must be positive")
    configured = config.get("head_dim")
    if configured is None and hidden % query_heads:
        raise ValueError("hidden_size must divide evenly across attention heads")
    head_dim = int(configured if configured is not None else hidden // query_heads)
    if head_dim <= 0:
        raise ValueError("attention head count and head_dim must be positive")
    return head_dim


def layer_uses_rope(config: dict[str, object], layer: int) -> bool:
    if config.get("model_type") != "smollm3":
        return True
    flags = config.get("no_rope_layers")
    if not isinstance(flags, list) or layer >= len(flags):
        raise ValueError("SmolLM3 config is missing the per-layer RoPE map")
    # This field has a historical inverted name. Transformers assigns it
    # directly to Attention.use_rope: 1 means apply RoPE, 0 means NoPE.
    return bool(flags[layer])


def residual_scale(config: dict[str, object]) -> float:
    if config.get("model_type") != "minicpm":
        return 1.0
    layers = int(config["num_hidden_layers"])
    return float(config["scale_depth"]) / math.sqrt(layers)


def rope_theta_value(config: dict[str, object]) -> float:
    params = config.get("rope_parameters") or config.get("rope_scaling") or {}
    return float(config.get("rope_theta", params.get("rope_theta", 10_000.0)))


def rope_parameters(
    config: dict[str, object], seq_len: int
) -> tuple[np.ndarray, float, int]:
    """Returns inverse frequencies, cos/sin scale, and rotary width."""
    head_dim = attention_head_dim(config)
    partial = float(config.get("partial_rotary_factor", 1.0))
    rotary_dim = int(head_dim * partial)
    if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2:
        raise ValueError("rotary dimension must be positive, even, and <= head_dim")

    params_value = config.get("rope_parameters", config.get("rope_scaling"))
    params = dict(params_value) if isinstance(params_value, dict) else {}
    rope_type = str(params.get("rope_type", params.get("type", "default")))
    base = float(params.get("rope_theta", config.get("rope_theta", 10_000.0)))
    exponent = np.arange(0, rotary_dim, 2, dtype=np.float32) / np.float32(rotary_dim)
    base_frequency = np.power(np.float32(base), exponent)
    attention_factor = 1.0

    if rope_type == "longrope":
        original = int(params.get(
            "original_max_position_embeddings",
            config.get("original_max_position_embeddings", config["max_position_embeddings"]),
        ))
        factors = params["long_factor"] if seq_len > original else params["short_factor"]
        ext = np.asarray(factors, dtype=np.float32)
        if ext.shape != (rotary_dim // 2,):
            raise ValueError(
                f"LongRoPE factor must have shape {(rotary_dim // 2,)}, got {ext.shape}"
            )
        factor = float(params.get(
            "factor", float(config["max_position_embeddings"]) / original
        ))
        attention_factor = float(params.get(
            "attention_factor",
            1.0 if factor <= 1.0 else math.sqrt(
                1.0 + math.log(factor) / math.log(original)
            ),
        ))
        inverse = 1.0 / (ext * base_frequency)
    elif rope_type == "yarn":
        factor = float(params["factor"])
        original = int(params["original_max_position_embeddings"])
        interpolation = 1.0 / (factor * base_frequency)
        extrapolation = 1.0 / base_frequency
        beta_fast = float(params.get("beta_fast", 32.0))
        beta_slow = float(params.get("beta_slow", 1.0))

        def correction(rotations: float) -> float:
            return (
                rotary_dim
                * math.log(original / (rotations * 2.0 * math.pi))
                / (2.0 * math.log(base))
            )

        low = max(math.floor(correction(beta_fast)), 0)
        high = min(math.ceil(correction(beta_slow)), rotary_dim - 1)
        if low == high:
            high += 0.001
        ramp = np.clip(
            (np.arange(rotary_dim // 2, dtype=np.float32) - low)
            / (high - low),
            0.0,
            1.0,
        )
        extrapolation_weight = 1.0 - ramp
        inverse = (
            interpolation * (1.0 - extrapolation_weight)
            + extrapolation * extrapolation_weight
        )
        if "attention_factor" in params:
            attention_factor = float(params["attention_factor"])
        elif params.get("mscale") and params.get("mscale_all_dim"):
            def mscale(scale: float, multiplier: float) -> float:
                return 1.0 if scale <= 1.0 else (
                    0.1 * multiplier * math.log(scale) + 1.0
                )
            attention_factor = (
                mscale(factor, float(params["mscale"]))
                / mscale(factor, float(params["mscale_all_dim"]))
            )
        else:
            attention_factor = (
                1.0 if factor <= 1.0 else 0.1 * math.log(factor) + 1.0
            )
    elif rope_type == "llama3":
        inverse = 1.0 / base_frequency
        factor = float(params["factor"])
        low_factor = float(params["low_freq_factor"])
        high_factor = float(params["high_freq_factor"])
        original = float(params["original_max_position_embeddings"])
        wavelength = 2.0 * math.pi / inverse
        low_wavelength = original / low_factor
        high_wavelength = original / high_factor
        scaled = np.where(wavelength > low_wavelength, inverse / factor, inverse)
        smooth = (
            (original / wavelength - low_factor)
            / (high_factor - low_factor)
        )
        smoothed = (1.0 - smooth) * scaled / factor + smooth * scaled
        medium = (wavelength >= high_wavelength) & (wavelength <= low_wavelength)
        inverse = np.where(medium, smoothed, scaled)
    elif rope_type in ("default", "none"):
        # Preserve the original reference's float32 evaluation order. Cache
        # append tests compare this path bit-for-bit with prefill.
        inverse = base ** (
            -np.arange(rotary_dim // 2, dtype=np.float32)
            * 2.0 / rotary_dim
        )
    else:
        raise ValueError(f"unsupported RoPE type {rope_type}")
    return np.asarray(inverse, dtype=np.float32), attention_factor, rotary_dim


def head_rms_norm(
    value: np.ndarray, weight: np.ndarray, epsilon: float
) -> np.ndarray:
    """Applies the Qwen3 Q/K RMSNorm independently to every attention head."""
    if value.ndim != 3:
        raise ValueError("head RMSNorm input must have shape [tokens, heads, dim]")
    head_dim = value.shape[-1]
    if weight.shape != (head_dim,):
        raise ValueError(
            f"head RMSNorm weight must have shape {(head_dim,)}, got {weight.shape}"
        )
    return rms_norm(value.reshape(-1, head_dim), weight, epsilon).reshape(
        value.shape
    )


def linear(
    value: np.ndarray,
    weight: np.ndarray,
    scale: float,
    bf16_scale: bool = False,
) -> np.ndarray:
    if bf16_scale:
        scale = float(bf16(np.asarray([scale], dtype=np.float32))[0])
    dequantized = bf16(weight.astype(np.float32) * scale)
    return bf16(value @ dequantized)


def biased_linear(
    value: np.ndarray,
    weight: np.ndarray,
    scale: float,
    bias: np.ndarray | None,
) -> np.ndarray:
    result = linear(value, weight, scale, bf16_scale=True)
    return result if bias is None else bf16(result + bf16(bias))


def rope_at_positions(
    value: np.ndarray,
    positions: np.ndarray,
    theta: float,
    *,
    inverse_frequencies: np.ndarray | None = None,
    attention_factor: float = 1.0,
    rotary_dim: int | None = None,
    use_rope: bool = True,
) -> np.ndarray:
    seq_len, _, head_dim = value.shape
    positions = np.asarray(positions, dtype=np.float32)
    if positions.shape != (seq_len,):
        raise ValueError("RoPE positions must have one entry per token")
    if not use_rope:
        return value
    rotary_dim = head_dim if rotary_dim is None else rotary_dim
    if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2:
        raise ValueError("invalid partial RoPE width")
    half = rotary_dim // 2
    inverse = (
        theta ** (-np.arange(half, dtype=np.float32) * 2.0 / rotary_dim)
        if inverse_frequencies is None
        else np.asarray(inverse_frequencies, dtype=np.float32)
    )
    if inverse.shape != (half,):
        raise ValueError(
            f"RoPE inverse frequencies must have shape {(half,)}, got {inverse.shape}"
        )
    angle = positions[:, None] * inverse[None, :]
    cosine = bf16(np.cos(angle) * attention_factor)[:, None, :]
    sine = bf16(np.sin(angle) * attention_factor)[:, None, :]
    rotated = value[:, :, :rotary_dim]
    low = rotated[:, :, :half]
    high = rotated[:, :, half:]
    result = bf16(np.concatenate(
        (low * cosine - high * sine, high * cosine + low * sine), axis=-1
    ))
    if rotary_dim != head_dim:
        result = np.concatenate((result, value[:, :, rotary_dim:]), axis=-1)
    return result


def rope(value: np.ndarray, theta: float) -> np.ndarray:
    return rope_at_positions(
        value, np.arange(value.shape[0], dtype=np.float32), theta
    )


def biased_rope_linear_at_positions(
    value: np.ndarray,
    weight: np.ndarray,
    scale: float,
    bias: np.ndarray | None,
    heads: int,
    theta: float,
    positions: np.ndarray,
) -> np.ndarray:
    projection = linear(value, weight, scale, bf16_scale=True).reshape(
        value.shape[0], heads, -1
    )
    positions = np.asarray(positions, dtype=np.float32)
    if positions.shape != (value.shape[0],):
        raise ValueError("RoPE positions must have one entry per token")
    if bias is None:
        return rope_at_positions(projection, positions, theta)

    _, _, head_dim = projection.shape
    half = head_dim // 2
    inverse = theta ** (
        -np.arange(half, dtype=np.float32) * np.float32(2.0 / head_dim)
    )
    angle = positions[:, None] * inverse[None, :]
    cosine = bf16(np.cos(angle))[:, None, :]
    sine = bf16(np.sin(angle))[:, None, :]
    bias = bf16(bias).reshape(1, heads, head_dim)
    low = projection[:, :, :half]
    high = projection[:, :, half:]
    bias_low = bias[:, :, :half]
    bias_high = bias[:, :, half:]

    low_cos = bf16(fma32(bias_low, cosine, low * cosine))
    high_sin = bf16(fma32(bias_high, sine, high * sine))
    high_cos = bf16(fma32(bias_high, cosine, high * cosine))
    low_sin = bf16(fma32(bias_low, sine, low * sine))
    return bf16(np.concatenate(
        (low_cos - high_sin, high_cos + low_sin), axis=-1
    ))


def normalized_rope_linear_at_positions(
    value: np.ndarray,
    weight: np.ndarray,
    scale: float,
    bias: np.ndarray | None,
    heads: int,
    theta: float,
    positions: np.ndarray,
    norm_weight: np.ndarray | None = None,
    epsilon: float = 1.0e-6,
    inverse_frequencies: np.ndarray | None = None,
    attention_factor: float = 1.0,
    rotary_dim: int | None = None,
    use_rope: bool = True,
) -> np.ndarray:
    """Projects Q/K, applies optional per-head RMSNorm, and then RoPE."""
    projected_head_dim = weight.shape[1] // heads
    full_rotary_dim = (
        projected_head_dim if rotary_dim is None else rotary_dim
    )
    default_inverse = theta ** (
        -np.arange(projected_head_dim // 2, dtype=np.float32)
        * 2.0 / projected_head_dim
    )
    uses_default_rope = (
        use_rope
        and full_rotary_dim == projected_head_dim
        and attention_factor == 1.0
        and (
            inverse_frequencies is None
            or np.array_equal(inverse_frequencies, default_inverse)
        )
    )
    if (
        norm_weight is None
        and uses_default_rope
    ):
        return biased_rope_linear_at_positions(
            value, weight, scale, bias, heads, theta, positions
        )
    projection = biased_linear(value, weight, scale, bias).reshape(
        value.shape[0], heads, -1
    )
    if norm_weight is not None:
        projection = head_rms_norm(projection, norm_weight, epsilon)
    return rope_at_positions(
        projection,
        positions,
        theta,
        inverse_frequencies=inverse_frequencies,
        attention_factor=attention_factor,
        rotary_dim=rotary_dim,
        use_rope=use_rope,
    )


def biased_rope_linear(
    value: np.ndarray,
    weight: np.ndarray,
    scale: float,
    bias: np.ndarray | None,
    heads: int,
    theta: float,
) -> np.ndarray:
    return biased_rope_linear_at_positions(
        value,
        weight,
        scale,
        bias,
        heads,
        theta,
        np.arange(value.shape[0], dtype=np.float32),
    )


def vxm_lut(value: np.ndarray, operation: str) -> np.ndarray:
    value = fp16_ftz(value)
    finite = np.isfinite(value)
    safe_value = np.where(finite, value, np.float32(0.0))
    entries = np.float32(256.0)

    if operation == "exp":
        ln2 = np.float32(0.6931471805599453)
        exponent = np.rint(
            safe_value * np.float32(1.4426950408889634)
        ).astype(np.int32)
        local = safe_value - exponent.astype(np.float32) * ln2
        input_min = -ln2 / np.float32(2.0)
        width = ln2 / entries
        multiplier = np.ones_like(value)
        result_exponent = exponent
        function = np.exp
    elif operation == "reciprocal":
        mantissa, exponent = np.frexp(np.abs(safe_value))
        local = mantissa.astype(np.float32) * np.float32(2.0)
        input_min = np.float32(1.0)
        width = np.float32(1.0) / entries
        multiplier = np.copysign(np.ones_like(value), safe_value)
        result_exponent = -(exponent - 1)
        function = lambda x: np.float32(1.0) / x
    elif operation == "rsqrt":
        mantissa, exponent = np.frexp(safe_value)
        local = mantissa.astype(np.float32) * np.float32(2.0)
        exponent = exponent - 1
        odd_exponent = (exponent & 1) != 0
        local = np.where(odd_exponent, local * np.float32(2.0), local)
        exponent = np.where(odd_exponent, exponent - 1, exponent)
        input_min = np.float32(1.0)
        width = np.float32(3.0) / entries
        multiplier = np.ones_like(value)
        result_exponent = -exponent // 2
        function = lambda x: np.float32(1.0) / np.sqrt(x)
    else:
        raise ValueError(f"unsupported VXM LUT operation: {operation}")

    index = np.clip(
        np.floor((local - input_min) / width).astype(np.int32), 0, 255
    )
    x0 = input_min + index.astype(np.float32) * width
    y0 = function(x0).astype(np.float32)
    y1 = function(x0 + width).astype(np.float32)
    slope = fp16_ftz((y1 - y0) / width)
    intercept = fp16_ftz(y0)
    interpolated = fma32(slope, local - x0, intercept)
    result = fp16_ftz(np.ldexp(
        interpolated * multiplier, result_exponent
    ))
    if operation == "exp":
        result = np.where(np.isneginf(value), np.float32(0.0), result)
        result = np.where(np.isposinf(value), np.float32(np.inf), result)
    elif operation == "rsqrt":
        result = np.where(value == 0.0, np.float32(np.inf), result)
        result = np.where(np.isposinf(value), np.float32(0.0), result)
        result = np.where(value < 0.0, np.float32(np.nan), result)
    return result


def lpu_softmax(
    scores: np.ndarray, head_dim: int, causal: bool = True
) -> np.ndarray:
    sequence = scores.shape[-1]
    scores = bf16(scores)
    if causal:
        mask_value = bf16(np.asarray([-1.0e9], dtype=np.float32))[0]
        scores = scores + np.triu(
            np.full((sequence, sequence), mask_value, dtype=np.float32), 1
        )
    scores = bf16(
        scores * np.float32(1.0 / np.sqrt(np.float32(head_dim)))
    )
    maximum = np.max(scores, axis=-1, keepdims=True)
    exponentials = vxm_lut(scores - maximum, "exp")
    # VXM accumulates one streamed element per cycle in key order.
    denominator = np.add.accumulate(
        exponentials, axis=-1, dtype=np.float32
    )[..., -1:]
    reciprocal = vxm_lut(denominator, "reciprocal")
    return bf16(exponentials * reciprocal)


def decoder_layer_prefill_kv_reference(
    activation: np.ndarray,
    norm0: np.ndarray,
    weights: dict[str, np.ndarray],
    scales: dict[str, float],
    config: dict[str, object],
    biases: dict[str, np.ndarray] | None = None,
    qk_norms: dict[str, np.ndarray] | None = None,
    layer: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    if activation.ndim != 2:
        raise ValueError("prefill activation must have shape [tokens, hidden]")
    seq_len, hidden = activation.shape
    query_heads = int(config["num_attention_heads"])
    kv_heads = int(config["num_key_value_heads"])
    head_dim = attention_head_dim(config)
    query_width = query_heads * head_dim
    kv_width = kv_heads * head_dim
    if weights["query"].shape != (hidden, query_width):
        raise ValueError("query projection shape does not match Qwen configuration")
    if weights["key"].shape != (hidden, kv_width):
        raise ValueError("key projection shape does not match Qwen configuration")
    epsilon = float(config["rms_norm_eps"])
    biases = biases or {}
    qk_norms = qk_norms or {}
    inverse, rope_scale, rotary_dim = rope_parameters(config, seq_len)
    use_rope = layer_uses_rope(config, layer)

    normalized = rms_norm(activation, norm0, epsilon)
    positions = np.arange(seq_len, dtype=np.float32)
    key = normalized_rope_linear_at_positions(
        normalized,
        weights["key"],
        scales["key"],
        biases.get("key"),
        kv_heads,
        rope_theta_value(config),
        positions,
        qk_norms.get("key"),
        epsilon,
        inverse,
        rope_scale,
        rotary_dim,
        use_rope,
    )
    value = biased_linear(
        normalized,
        weights["value"],
        scales["value"],
        biases.get("value"),
    ).reshape(seq_len, kv_heads, head_dim)
    return key, value


def decoder_layer_decode_reference(
    activation: np.ndarray,
    key_cache: np.ndarray,
    value_cache: np.ndarray,
    position: int,
    norm0: np.ndarray,
    norm1: np.ndarray,
    weights: dict[str, np.ndarray],
    scales: dict[str, float],
    config: dict[str, object],
    biases: dict[str, np.ndarray] | None = None,
    stage_outputs: dict[str, np.ndarray] | None = None,
    qk_norms: dict[str, np.ndarray] | None = None,
    layer: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if activation.ndim != 2 or activation.shape[0] != 1:
        raise ValueError("decode activation must have shape [1, hidden]")
    _, hidden = activation.shape
    query_heads = int(config["num_attention_heads"])
    kv_heads = int(config["num_key_value_heads"])
    if query_heads % kv_heads != 0:
        raise ValueError("invalid grouped-query attention dimensions")
    head_dim = attention_head_dim(config)
    query_width = query_heads * head_dim
    expected_cache_shape = (position, kv_heads, head_dim)
    if key_cache.shape != expected_cache_shape:
        raise ValueError(
            f"key cache shape must be {expected_cache_shape}, got {key_cache.shape}"
        )
    if value_cache.shape != expected_cache_shape:
        raise ValueError(
            f"value cache shape must be {expected_cache_shape}, got {value_cache.shape}"
        )
    biases = biases or {}
    qk_norms = qk_norms or {}
    epsilon = float(config["rms_norm_eps"])
    positions = np.asarray([position], dtype=np.float32)
    inverse, rope_scale, rotary_dim = rope_parameters(config, position + 1)
    use_rope = layer_uses_rope(config, layer)

    normalized = rms_norm(activation, norm0, epsilon)
    query = normalized_rope_linear_at_positions(
        normalized,
        weights["query"],
        scales["query"],
        biases.get("query"),
        query_heads,
        rope_theta_value(config),
        positions,
        qk_norms.get("query"),
        epsilon,
        inverse,
        rope_scale,
        rotary_dim,
        use_rope,
    )
    new_key = normalized_rope_linear_at_positions(
        normalized,
        weights["key"],
        scales["key"],
        biases.get("key"),
        kv_heads,
        rope_theta_value(config),
        positions,
        qk_norms.get("key"),
        epsilon,
        inverse,
        rope_scale,
        rotary_dim,
        use_rope,
    )
    new_value = biased_linear(
        normalized,
        weights["value"],
        scales["value"],
        biases.get("value"),
    ).reshape(1, kv_heads, head_dim)
    present_key = np.concatenate((key_cache, new_key), axis=0)
    present_value = np.concatenate((value_cache, new_value), axis=0)

    repeats = query_heads // kv_heads
    attention_key = np.transpose(
        np.repeat(present_key, repeats, axis=1), (1, 0, 2)
    )
    attention_value = np.transpose(
        np.repeat(present_value, repeats, axis=1), (1, 0, 2)
    )
    query = np.transpose(query, (1, 0, 2))
    scores = bf16(query @ np.transpose(attention_key, (0, 2, 1)))
    probability = lpu_softmax(scores, head_dim, causal=False)
    context = bf16(
        np.transpose(probability @ attention_value, (1, 0, 2)).reshape(
            1, query_width
        )
    )
    attention = linear(
        context, weights["output"], scales["output"], bf16_scale=True
    )
    depth_scale = residual_scale(config)
    residual = bf16(activation + bf16(attention * depth_scale))
    normalized_ffn = rms_norm(residual, norm1, epsilon)
    gate = linear(normalized_ffn, weights["gate"], scales["gate"], True)
    up = linear(normalized_ffn, weights["up"], scales["up"], True)
    swiglu = bf16((gate / (1.0 + np.exp(-gate))) * up)
    down = linear(swiglu, weights["down"], scales["down"], True)
    output = bf16(residual + bf16(down * depth_scale))

    if stage_outputs is not None:
        stage_outputs["normalized"] = normalized
        stage_outputs["query"] = np.transpose(query, (1, 0, 2))
        stage_outputs["new_key"] = new_key
        stage_outputs["new_value"] = new_value
        stage_outputs["scores"] = scores
        stage_outputs["probability"] = probability
        stage_outputs["context"] = context
        stage_outputs["attention"] = attention
        stage_outputs["residual0"] = residual
        stage_outputs["norm1"] = normalized_ffn
        stage_outputs["swiglu"] = swiglu
        stage_outputs["down"] = down
    return output, present_key, present_value


def qwen3_moe_reference(
    value: np.ndarray,
    weights: dict[str, np.ndarray],
    scales: dict[str, object],
    config: dict[str, object],
    routing_outputs: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Runs the quantized Qwen3 MoE router, experts, and weighted combine."""
    expert_count = int(config["num_experts"])
    top_k = int(config["num_experts_per_tok"])
    if top_k <= 0 or top_k > expert_count:
        raise ValueError("num_experts_per_tok must be in [1, num_experts]")
    router_logits = linear(
        value, weights["router"], float(scales["router"]), True
    )
    shifted = router_logits.astype(np.float32) - np.max(
        router_logits, axis=-1, keepdims=True
    )
    probability = np.exp(shifted, dtype=np.float32)
    probability /= np.sum(probability, axis=-1, keepdims=True, dtype=np.float32)
    # A stable order gives deterministic behavior for synthetic equal-logit
    # tests. Real checkpoints almost never contain exact routing ties.
    selected = np.argsort(-probability, axis=-1, kind="stable")[:, :top_k]
    routing_weights = np.take_along_axis(probability, selected, axis=-1)
    if bool(config.get("norm_topk_prob", False)):
        routing_weights /= np.sum(
            routing_weights, axis=-1, keepdims=True, dtype=np.float32
        )
    routing_weights = bf16(routing_weights)

    expert_scales = {
        role: np.asarray(scales[role], dtype=np.float32)
        for role in ("expert_gate", "expert_up", "expert_down")
    }
    result = np.zeros_like(value, dtype=np.float32)
    for expert in range(expert_count):
        token_index, slot_index = np.where(selected == expert)
        if token_index.size == 0:
            continue
        expert_input = value[token_index]
        gate = linear(
            expert_input, weights["expert_gate"][expert],
            float(expert_scales["expert_gate"][expert]), True,
        )
        up = linear(
            expert_input, weights["expert_up"][expert],
            float(expert_scales["expert_up"][expert]), True,
        )
        hidden = bf16((gate / (1.0 + np.exp(-gate))) * up)
        output = linear(
            hidden, weights["expert_down"][expert],
            float(expert_scales["expert_down"][expert]), True,
        )
        contribution = bf16(
            output * routing_weights[token_index, slot_index, None]
        )
        np.add.at(result, token_index, contribution)
    result = bf16(result)
    if routing_outputs is not None:
        routing_outputs["logits"] = router_logits
        routing_outputs["indices"] = selected.astype(np.int32)
        routing_outputs["weights"] = routing_weights
        counts = np.bincount(selected.reshape(-1), minlength=expert_count)
        routing_outputs["expert_counts"] = counts.astype(np.int32)
    return result


def decoder_layer_reference(
    activation: np.ndarray,
    norm0: np.ndarray,
    norm1: np.ndarray,
    weights: dict[str, np.ndarray],
    scales: dict[str, float],
    config: dict[str, object],
    biases: dict[str, np.ndarray] | None = None,
    stage_outputs: dict[str, np.ndarray] | None = None,
    qk_norms: dict[str, np.ndarray] | None = None,
    layer: int = 0,
    routing_outputs: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    seq_len, hidden = activation.shape
    query_heads = int(config["num_attention_heads"])
    kv_heads = int(config["num_key_value_heads"])
    head_dim = attention_head_dim(config)
    query_width = query_heads * head_dim
    epsilon = float(config["rms_norm_eps"])

    normalized = rms_norm(activation, norm0, epsilon)
    if stage_outputs is not None:
        stage_outputs["norm0"] = normalized
    biases = biases or {}
    qk_norms = qk_norms or {}
    positions = np.arange(seq_len, dtype=np.float32)
    inverse, rope_scale, rotary_dim = rope_parameters(config, seq_len)
    use_rope = layer_uses_rope(config, layer)
    query = normalized_rope_linear_at_positions(
        normalized, weights["query"], scales["query"],
        biases.get("query"), query_heads,
        rope_theta_value(config), positions,
        qk_norms.get("query"), epsilon,
        inverse, rope_scale, rotary_dim, use_rope,
    )
    key = normalized_rope_linear_at_positions(
        normalized, weights["key"], scales["key"],
        biases.get("key"), kv_heads,
        rope_theta_value(config), positions,
        qk_norms.get("key"), epsilon,
        inverse, rope_scale, rotary_dim, use_rope,
    )
    value = biased_linear(normalized, weights["value"], scales["value"],
                          biases.get("value")).reshape(
        seq_len, kv_heads, head_dim
    )
    if stage_outputs is not None:
        stage_outputs["query"] = query.reshape(seq_len, query_width)
        stage_outputs["key"] = key.reshape(seq_len, kv_heads * head_dim)
        stage_outputs["value"] = value.reshape(seq_len, kv_heads * head_dim)
    repeats = query_heads // kv_heads
    key = np.repeat(key, repeats, axis=1)
    value = np.repeat(value, repeats, axis=1)
    query = np.transpose(query, (1, 0, 2))
    key = np.transpose(key, (1, 0, 2))
    value = np.transpose(value, (1, 0, 2))
    scores = bf16(query @ np.transpose(key, (0, 2, 1)))
    probability = lpu_softmax(scores, head_dim)
    if stage_outputs is not None:
        stage_outputs["scores"] = scores
        stage_outputs["probability"] = probability
    context = bf16(
        np.transpose(probability @ value, (1, 0, 2)).reshape(
            seq_len, query_width
        )
    )
    attention = linear(
        context, weights["output"], scales["output"], bf16_scale=True
    )
    depth_scale = residual_scale(config)
    residual = bf16(activation + bf16(attention * depth_scale))
    if stage_outputs is not None:
        stage_outputs["context"] = bf16(context)
        stage_outputs["attention"] = attention
        stage_outputs["residual0"] = residual

    normalized = rms_norm(residual, norm1, epsilon)
    if stage_outputs is not None:
        stage_outputs["norm1"] = normalized
    # Vector FFN uses the MXM local INT8 dequant instruction, whose immediate
    # scale is encoded as BF16. Attention retains its existing projection
    # reference semantics until that lowering uses the same local path.
    if config.get("model_type") == "qwen3_moe":
        down = qwen3_moe_reference(
            normalized, weights, scales, config, routing_outputs
        )
        if stage_outputs is not None:
            stage_outputs["moe"] = down
            stage_outputs["down"] = down
    else:
        gate = linear(normalized, weights["gate"], scales["gate"], True)
        up = linear(normalized, weights["up"], scales["up"], True)
        swiglu = bf16((gate / (1.0 + np.exp(-gate))) * up)
        down = linear(swiglu, weights["down"], scales["down"], True)
        if stage_outputs is not None:
            stage_outputs["swiglu"] = swiglu
            stage_outputs["down"] = down
    return bf16(residual + bf16(down * depth_scale))


def load_decoder_weights(
    store: SafeTensorStore,
    prefix: str,
    config: dict[str, object],
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Reads HF linear tensors and expands supported fused layouts."""
    model_type = str(config["model_type"])
    hidden = int(config["hidden_size"])
    query_width = int(config["num_attention_heads"]) * attention_head_dim(config)
    kv_width = int(config["num_key_value_heads"]) * attention_head_dim(config)
    intermediate = int(config["intermediate_size"])
    source_names: dict[str, str] = {}
    weights: dict[str, np.ndarray] = {}

    if model_type == "phi3":
        qkv_name = f"{prefix}.self_attn.qkv_proj.weight"
        qkv = store.read(qkv_name)
        expected = (query_width + 2 * kv_width, hidden)
        if qkv.shape != expected:
            raise ValueError(f"fused QKV weight must have shape {expected}, got {qkv.shape}")
        query_end = query_width
        key_end = query_end + kv_width
        weights["query"] = qkv[:query_end]
        weights["key"] = qkv[query_end:key_end]
        weights["value"] = qkv[key_end:]
        source_names.update({role: qkv_name for role in ("query", "key", "value")})
    else:
        for role, stem in (("query", "q_proj"), ("key", "k_proj"),
                           ("value", "v_proj")):
            name = f"{prefix}.self_attn.{stem}.weight"
            weights[role] = store.read(name)
            source_names[role] = name

    output_name = f"{prefix}.self_attn.o_proj.weight"
    weights["output"] = store.read(output_name)
    source_names["output"] = output_name

    if model_type == "qwen3_moe":
        expert_count = int(config["num_experts"])
        moe_intermediate = int(config["moe_intermediate_size"])
        router_name = f"{prefix}.mlp.gate.weight"
        weights["router"] = store.read(router_name)
        source_names["router"] = router_name

        def first_tensor(*names: str) -> tuple[np.ndarray, str]:
            for name in names:
                if store.contains(name):
                    return store.read(name), name
            raise KeyError("checkpoint tensor is missing: " + " or ".join(names))

        gate_up, gate_up_name = first_tensor(
            f"{prefix}.mlp.experts.gate_up_proj",
            f"{prefix}.mlp.experts.gate_up_proj.weight",
        )
        down, expert_down_name = first_tensor(
            f"{prefix}.mlp.experts.down_proj",
            f"{prefix}.mlp.experts.down_proj.weight",
        )
        expected_gate_up = (expert_count, 2 * moe_intermediate, hidden)
        expected_down = (expert_count, hidden, moe_intermediate)
        if gate_up.shape != expected_gate_up:
            raise ValueError(
                f"stacked expert Gate/Up weight must have shape {expected_gate_up}, "
                f"got {gate_up.shape}"
            )
        if down.shape != expected_down:
            raise ValueError(
                f"stacked expert Down weight must have shape {expected_down}, "
                f"got {down.shape}"
            )
        weights["expert_gate"] = gate_up[:, :moe_intermediate, :]
        weights["expert_up"] = gate_up[:, moe_intermediate:, :]
        weights["expert_down"] = down
        source_names.update({
            "expert_gate": gate_up_name,
            "expert_up": gate_up_name,
            "expert_down": expert_down_name,
        })
    elif model_type in ("phi3", "glm"):
        gate_up_name = f"{prefix}.mlp.gate_up_proj.weight"
        gate_up = store.read(gate_up_name)
        expected = (2 * intermediate, hidden)
        if gate_up.shape != expected:
            raise ValueError(
                f"fused Gate/Up weight must have shape {expected}, got {gate_up.shape}"
            )
        weights["gate"] = gate_up[:intermediate]
        weights["up"] = gate_up[intermediate:]
        source_names.update({"gate": gate_up_name, "up": gate_up_name})
    else:
        for role in ("gate", "up"):
            name = f"{prefix}.mlp.{role}_proj.weight"
            weights[role] = store.read(name)
            source_names[role] = name
    if model_type != "qwen3_moe":
        down_name = f"{prefix}.mlp.down_proj.weight"
        weights["down"] = store.read(down_name)
        source_names["down"] = down_name
    return weights, source_names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument(
        "--input-bf16",
        type=Path,
        help="optional BF16 activation produced by the preceding layer",
    )
    parser.add_argument(
        "--ignore-attention-bias",
        action="store_true",
        help=(
            "replace checkpoint Q/K/V biases with zero-valued operands"
        ),
    )
    args = parser.parse_args()

    root_config = json.loads(
        (args.model_dir / "config.json").read_text(encoding="utf-8")
    )
    config = decoder_config(root_config)
    if config.get("model_type") not in SUPPORTED_MODEL_TYPES:
        raise ValueError(
            "unsupported dense decoder model_type "
            f"{config.get('model_type')!r}; supported: "
            f"{', '.join(sorted(SUPPORTED_MODEL_TYPES))}"
        )
    if config.get("hidden_act", "silu") not in ("silu", "swish"):
        raise ValueError("current LPU dense decoder requires a SiLU-gated MLP")
    if args.seq_len % 32:
        raise ValueError("current LPU decoder executable requires seq_len divisible by 32")

    store = SafeTensorStore(args.model_dir)
    prefix, embedding_name = checkpoint_prefixes(root_config, args.layer)
    source_weights, names = load_decoder_weights(store, prefix, config)
    quantized: dict[str, np.ndarray] = {}
    scales: dict[str, object] = {}
    for role, value in source_weights.items():
        if role.startswith("expert_"):
            quantized[role], scales[role] = quantize_expert_linear(value)
        else:
            quantized[role], scales[role] = quantize_linear(value)

    source_has_attention_bias = config.get(
        "attention_bias", config.get("model_type") == "qwen2"
    )
    biases: dict[str, np.ndarray] = {}
    if source_has_attention_bias and not args.ignore_attention_bias:
        for role, stem in (("query", "q_proj"), ("key", "k_proj"),
                           ("value", "v_proj")):
            biases[role] = store.read(f"{prefix}.self_attn.{stem}.bias")

    hidden = int(config["hidden_size"])
    query_heads = int(config["num_attention_heads"])
    kv_heads = int(config["num_key_value_heads"])
    head_dim = attention_head_dim(config)
    query_width = query_heads * head_dim
    kv_width = kv_heads * head_dim
    expected_weight_shapes = {
        "query": (hidden, query_width),
        "key": (hidden, kv_width),
        "value": (hidden, kv_width),
        "output": (query_width, hidden),
    }
    if config.get("model_type") == "qwen3_moe":
        experts = int(config["num_experts"])
        top_k = int(config["num_experts_per_tok"])
        moe_intermediate = int(config["moe_intermediate_size"])
        if top_k <= 0 or top_k > experts:
            raise ValueError("Qwen3 MoE top-k must be in [1, num_experts]")
        expected_weight_shapes.update({
            "router": (hidden, experts),
            "expert_gate": (experts, hidden, moe_intermediate),
            "expert_up": (experts, hidden, moe_intermediate),
            "expert_down": (experts, moe_intermediate, hidden),
        })
    else:
        expected_weight_shapes.update({
            "gate": (hidden, int(config["intermediate_size"])),
            "up": (hidden, int(config["intermediate_size"])),
            "down": (int(config["intermediate_size"]), hidden),
        })
    for role, expected in expected_weight_shapes.items():
        if quantized[role].shape != expected:
            raise ValueError(
                f"{role} weight must have shape {expected}, "
                f"got {quantized[role].shape}"
            )
    serialized_biases = {
        "query": biases.get("query", np.zeros(query_width, dtype=np.float32)),
        "key": biases.get("key", np.zeros(kv_width, dtype=np.float32)),
        "value": biases.get("value", np.zeros(kv_width, dtype=np.float32)),
    }

    qk_norms: dict[str, np.ndarray] = {}
    if config.get("model_type") in ("qwen3", "qwen3_moe"):
        qk_norms = {
            "query": store.read(f"{prefix}.self_attn.q_norm.weight"),
            "key": store.read(f"{prefix}.self_attn.k_norm.weight"),
        }
        for role, value in qk_norms.items():
            if value.shape != (head_dim,):
                raise ValueError(
                    f"{role} norm weight must have shape {(head_dim,)}, "
                    f"got {value.shape}"
                )

    norm0 = store.read(f"{prefix}.input_layernorm.weight")
    norm1 = store.read(f"{prefix}.post_attention_layernorm.weight")
    embedding = store.read(embedding_name)
    token_ids = np.arange(args.seq_len, dtype=np.int64) % embedding.shape[0]
    if args.input_bf16:
        activation = read_bf16(
            args.input_bf16,
            (args.seq_len, int(config["hidden_size"])),
        )
    else:
        activation = bf16(
            embedding[token_ids] * float(config.get("scale_emb", 1.0))
        )
    stage_outputs: dict[str, np.ndarray] = {}
    routing_outputs: dict[str, np.ndarray] = {}
    golden = decoder_layer_reference(
        activation, norm0, norm1, quantized, scales, config, biases,
        stage_outputs, qk_norms, args.layer, routing_outputs,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_bf16(args.output_dir / "input.bf16.bin", activation)
    write_bf16(
        args.output_dir / "input_layernorm.bf16.bin", norm0
    )
    write_bf16(
        args.output_dir / "post_attention_layernorm.bf16.bin", norm1
    )
    write_bf16(args.output_dir / "golden.bf16.bin", golden)
    for stage, value in stage_outputs.items():
        write_bf16(args.output_dir / f"golden.{stage}.bf16.bin", value)
    token_ids.astype("<i8").tofile(args.output_dir / "token_ids.i64.bin")
    for role, value in quantized.items():
        value.tofile(args.output_dir / f"{role}.i8.bin")
    dense_scale_roles = ["query", "key", "value", "output"]
    if config.get("model_type") != "qwen3_moe":
        dense_scale_roles.extend(("gate", "up", "down"))
    else:
        dense_scale_roles.append("router")
    np.asarray(
        [float(scales[role]) for role in dense_scale_roles], dtype="<f4"
    ).tofile(args.output_dir / "quant_scales.f32.bin")
    if routing_outputs:
        routing_outputs["indices"].astype("<i4").tofile(
            args.output_dir / "router_indices.i32.bin"
        )
        routing_outputs["expert_counts"].astype("<i4").tofile(
            args.output_dir / "expert_counts.i32.bin"
        )
        write_bf16(
            args.output_dir / "router_logits.bf16.bin",
            routing_outputs["logits"],
        )
        write_bf16(
            args.output_dir / "router_weights.bf16.bin",
            routing_outputs["weights"],
        )
        np.stack([
            np.asarray(scales[role], dtype=np.float32)
            for role in ("expert_gate", "expert_up", "expert_down")
        ]).astype("<f4").tofile(args.output_dir / "expert_quant_scales.f32.bin")
    for role, value in serialized_biases.items():
        write_bf16(args.output_dir / f"{role}_bias.bf16.bin", value)
    for role, value in qk_norms.items():
        write_bf16(args.output_dir / f"{role}_norm.bf16.bin", value)

    rope_config = config.get("rope_parameters") or config.get("rope_scaling") or {}
    metadata = {
        "model": args.model_dir.name,
        "architecture": root_config.get(
            "architectures", config.get("architectures", ["LlamaForCausalLM"])
        )[0],
        "model_type": config["model_type"],
        "layer": args.layer,
        "seq_len": args.seq_len,
        "hidden_size": int(config["hidden_size"]),
        "intermediate_size": int(config["intermediate_size"]),
        "query_heads": query_heads,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "query_width": query_width,
        "kv_width": kv_width,
        "rope_theta": rope_theta_value(config),
        "rope_type": str(rope_config.get(
            "rope_type", rope_config.get("type", "default")
        )),
        "rotary_dim": rope_parameters(config, args.seq_len)[2],
        "uses_rope": layer_uses_rope(config, args.layer),
        "rms_norm_eps": float(config["rms_norm_eps"]),
        "residual_scale": residual_scale(config),
        "embedding_scale": float(config.get("scale_emb", 1.0)),
        "scales": {
            role: (value.tolist() if isinstance(value, np.ndarray) else value)
            for role, value in scales.items()
        },
        "source_tensors": names,
        "embedding_tensor": embedding_name,
        "attention_bias": bool(biases),
        # Preserve the executable ABI even when checkpoint biases are
        # intentionally replaced with zeroes.  Architectures that do not
        # define attention bias must not shift the following norm/FFN
        # bindings by three slots.
        "attention_bias_operands": bool(source_has_attention_bias),
        "source_attention_bias": bool(source_has_attention_bias),
        "ignored_attention_bias": bool(
            source_has_attention_bias and args.ignore_attention_bias
        ),
        "bias_roles": sorted(biases),
        "bias_operand_roles": sorted(serialized_biases),
        "qk_norm": bool(qk_norms),
        "qk_norm_roles": sorted(qk_norms),
        "qk_norm_tensors": {
            "query": f"{prefix}.self_attn.q_norm.weight",
            "key": f"{prefix}.self_attn.k_norm.weight",
        } if qk_norms else {},
        "bias_tensors": {
            role: f"{prefix}.self_attn.{stem}.bias"
            for role, stem in (("query", "q_proj"), ("key", "k_proj"),
                               ("value", "v_proj"))
            if role in biases
        },
        "activation_element_type": "bf16",
        "input_source": (
            str(args.input_bf16) if args.input_bf16 else "embedding"
        ),
    }
    if config.get("model_type") == "qwen3_moe":
        metadata["moe"] = {
            "num_experts": int(config["num_experts"]),
            "num_experts_per_tok": int(config["num_experts_per_tok"]),
            "moe_intermediate_size": int(config["moe_intermediate_size"]),
            "norm_topk_prob": bool(config.get("norm_topk_prob", False)),
            "expert_counts": routing_outputs["expert_counts"].tolist(),
        }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
