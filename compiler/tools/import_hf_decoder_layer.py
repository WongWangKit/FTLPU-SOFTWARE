#!/usr/bin/env python3
"""Imports one Llama/Qwen2 Hugging Face decoder layer without PyTorch."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np


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


def quantize_linear(weight: np.ndarray) -> tuple[np.ndarray, float]:
    # Hugging Face stores Linear weights as [out, in]; StableHLO consumes [in, out].
    maximum = float(np.max(np.abs(weight)))
    scale = maximum / 127.0 if maximum else 1.0
    quantized = np.clip(np.rint(weight / scale), -127, 127).astype(np.int8)
    return np.ascontiguousarray(quantized.T), scale


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


def rope(value: np.ndarray, theta: float) -> np.ndarray:
    seq_len, _, head_dim = value.shape
    half = head_dim // 2
    inverse = theta ** (-np.arange(half, dtype=np.float32) * 2.0 / head_dim)
    angle = np.arange(seq_len, dtype=np.float32)[:, None] * inverse[None, :]
    cosine = bf16(np.cos(angle))[:, None, :]
    sine = bf16(np.sin(angle))[:, None, :]
    low = value[:, :, :half]
    high = value[:, :, half:]
    return bf16(np.concatenate(
        (low * cosine - high * sine, high * cosine + low * sine), axis=-1
    ))


def biased_rope_linear(
    value: np.ndarray,
    weight: np.ndarray,
    scale: float,
    bias: np.ndarray | None,
    heads: int,
    theta: float,
) -> np.ndarray:
    projection = linear(value, weight, scale, bf16_scale=True).reshape(
        value.shape[0], heads, -1
    )
    if bias is None:
        return rope(projection, theta)

    seq_len, _, head_dim = projection.shape
    half = head_dim // 2
    inverse = theta ** (
        -np.arange(half, dtype=np.float32) * np.float32(2.0 / head_dim)
    )
    angle = np.arange(seq_len, dtype=np.float32)[:, None] * inverse[None, :]
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


def decoder_layer_reference(
    activation: np.ndarray,
    norm0: np.ndarray,
    norm1: np.ndarray,
    weights: dict[str, np.ndarray],
    scales: dict[str, float],
    config: dict[str, object],
    biases: dict[str, np.ndarray] | None = None,
    stage_outputs: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    seq_len, hidden = activation.shape
    query_heads = int(config["num_attention_heads"])
    kv_heads = int(config["num_key_value_heads"])
    head_dim = hidden // query_heads
    epsilon = float(config["rms_norm_eps"])

    normalized = rms_norm(activation, norm0, epsilon)
    if stage_outputs is not None:
        stage_outputs["norm0"] = normalized
    biases = biases or {}
    query = biased_rope_linear(
        normalized, weights["query"], scales["query"],
        biases.get("query"), query_heads,
        float(config["rope_theta"]),
    )
    key = biased_rope_linear(
        normalized, weights["key"], scales["key"],
        biases.get("key"), kv_heads,
        float(config["rope_theta"]),
    )
    value = biased_linear(normalized, weights["value"], scales["value"],
                          biases.get("value")).reshape(
        seq_len, kv_heads, head_dim
    )
    if stage_outputs is not None:
        stage_outputs["query"] = query.reshape(seq_len, hidden)
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
    context = bf16(
        np.transpose(probability @ value, (1, 0, 2)).reshape(seq_len, hidden)
    )
    attention = linear(
        context, weights["output"], scales["output"], bf16_scale=True
    )
    residual = bf16(activation + attention)
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
    gate = linear(normalized, weights["gate"], scales["gate"], True)
    up = linear(normalized, weights["up"], scales["up"], True)
    swiglu = bf16((gate / (1.0 + np.exp(-gate))) * up)
    down = linear(swiglu, weights["down"], scales["down"], True)
    if stage_outputs is not None:
        stage_outputs["swiglu"] = swiglu
        stage_outputs["down"] = down
    return bf16(residual + down)


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

    config = json.loads(
        (args.model_dir / "config.json").read_text(encoding="utf-8")
    )
    if config.get("model_type") not in ("llama", "qwen2"):
        raise ValueError("only standard Llama and Qwen2 checkpoints are supported")
    if args.seq_len % 32:
        raise ValueError("current LPU decoder executable requires seq_len divisible by 32")

    store = SafeTensorStore(args.model_dir)
    prefix = f"model.layers.{args.layer}"
    names = {
        "query": f"{prefix}.self_attn.q_proj.weight",
        "key": f"{prefix}.self_attn.k_proj.weight",
        "value": f"{prefix}.self_attn.v_proj.weight",
        "output": f"{prefix}.self_attn.o_proj.weight",
        "gate": f"{prefix}.mlp.gate_proj.weight",
        "up": f"{prefix}.mlp.up_proj.weight",
        "down": f"{prefix}.mlp.down_proj.weight",
    }
    quantized: dict[str, np.ndarray] = {}
    scales: dict[str, float] = {}
    for role, name in names.items():
        quantized[role], scales[role] = quantize_linear(store.read(name))

    source_has_attention_bias = config.get(
        "attention_bias", config.get("model_type") == "qwen2"
    )
    biases: dict[str, np.ndarray] = {}
    if source_has_attention_bias and not args.ignore_attention_bias:
        for role, stem in (("query", "q_proj"), ("key", "k_proj"),
                           ("value", "v_proj")):
            biases[role] = store.read(f"{prefix}.self_attn.{stem}.bias")

    hidden = int(config["hidden_size"])
    kv_width = (
        int(config["num_key_value_heads"])
        * hidden // int(config["num_attention_heads"])
    )
    serialized_biases = {
        "query": biases.get("query", np.zeros(hidden, dtype=np.float32)),
        "key": biases.get("key", np.zeros(kv_width, dtype=np.float32)),
        "value": biases.get("value", np.zeros(kv_width, dtype=np.float32)),
    }

    norm0 = store.read(f"{prefix}.input_layernorm.weight")
    norm1 = store.read(f"{prefix}.post_attention_layernorm.weight")
    embedding = store.read("model.embed_tokens.weight")
    token_ids = np.arange(args.seq_len, dtype=np.int64) % embedding.shape[0]
    if args.input_bf16:
        activation = read_bf16(
            args.input_bf16,
            (args.seq_len, int(config["hidden_size"])),
        )
    else:
        activation = bf16(embedding[token_ids])
    stage_outputs: dict[str, np.ndarray] = {}
    golden = decoder_layer_reference(
        activation, norm0, norm1, quantized, scales, config, biases,
        stage_outputs,
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
    np.asarray(
        [scales[role] for role in names], dtype="<f4"
    ).tofile(args.output_dir / "quant_scales.f32.bin")
    for role, value in serialized_biases.items():
        write_bf16(args.output_dir / f"{role}_bias.bf16.bin", value)

    metadata = {
        "model": args.model_dir.name,
        "architecture": config.get("architectures", ["LlamaForCausalLM"])[0],
        "layer": args.layer,
        "seq_len": args.seq_len,
        "hidden_size": int(config["hidden_size"]),
        "intermediate_size": int(config["intermediate_size"]),
        "query_heads": int(config["num_attention_heads"]),
        "kv_heads": int(config["num_key_value_heads"]),
        "head_dim": int(config["hidden_size"]) // int(config["num_attention_heads"]),
        "rope_theta": float(config["rope_theta"]),
        "rms_norm_eps": float(config["rms_norm_eps"]),
        "scales": scales,
        "source_tensors": names,
        "attention_bias": bool(biases),
        "attention_bias_operands": True,
        "source_attention_bias": bool(source_has_attention_bias),
        "ignored_attention_bias": bool(
            source_has_attention_bias and args.ignore_attention_bias
        ),
        "bias_roles": sorted(biases),
        "bias_operand_roles": sorted(serialized_biases),
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
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
