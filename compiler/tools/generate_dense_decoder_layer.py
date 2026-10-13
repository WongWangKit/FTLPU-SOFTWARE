#!/usr/bin/env python3
"""Generate a seq32 bias-free dense decoder StableHLO fixture."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path


def format_float(value: float) -> str:
    return f"{value:.6e}"


def section_replace(text: str, begin: str, end: str, old: str, new: str) -> str:
    first = text.index(begin)
    last = text.index(end, first)
    return text[:first] + text[first:last].replace(old, new) + text[last:]


def transform_tensor_types(
    text: str,
    *,
    hidden: int,
    intermediate: int,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
) -> str:
    half = head_dim // 2
    kv_width = kv_heads * head_dim
    groups = query_heads // kv_heads

    def transform(match: re.Match[str]) -> str:
        body = match.group(1)
        shaped = re.fullmatch(r"([0-9x]+)x(bf16|f32|i8|i32|i1)", body)
        if shaped is None:
            return match.group(0)
        dims = [int(value) for value in shaped.group(1).split("x")]
        original = list(dims)
        for index, value in enumerate(original):
            if value == 1536:
                dims[index] = hidden
            elif value == 8960:
                dims[index] = intermediate
            elif value == 256:
                dims[index] = kv_width
            elif value == 12:
                dims[index] = query_heads
            elif value == 128:
                dims[index] = head_dim
            elif value == 64:
                dims[index] = half
            elif value == 6:
                dims[index] = groups
            elif value == 2:
                is_rope_pair = index == len(original) - 1 and (
                    index > 0 and original[index - 1] == 64
                )
                if not is_rope_pair:
                    dims[index] = kv_heads
        return "tensor<" + "x".join(str(value) for value in dims) + (
            "x" + shaped.group(2) + ">"
        )

    return re.sub(r"tensor<([^>]+)>", transform, text)


def generate(
    source: Path,
    *,
    name: str,
    hidden: int,
    intermediate: int,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    rope_theta: float,
    rms_norm_eps: float,
    residual_scale: float = 1.0,
    rotary_dim: int | None = None,
    rope_attention_factor: float = 1.0,
) -> str:
    if query_heads % kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    if head_dim % 2:
        raise ValueError("head dimension must be even")
    text = source.read_text(encoding="utf-8")
    text = text.replace("qwen2_5_1_5b_decoder_layer_seq32", name)

    for line in (
        "      %query_bias: tensor<1536xbf16>,\n",
        "      %key_bias: tensor<256xbf16>,\n",
        "      %value_bias: tensor<256xbf16>,\n",
    ):
        text = text.replace(line, "")
    bias_begin = text.index("    %attention_query_bias_2d")
    heads_begin = text.index("    %attention_query_heads", bias_begin)
    text = text[:bias_begin] + text[heads_begin:]
    text = text.replace("%attention_query_biased", "%attention_query_2d")
    text = text.replace("%attention_key_biased", "%attention_key_2d")
    text = text.replace("%attention_value_biased", "%attention_value_2d")

    text = transform_tensor_types(
        text,
        hidden=hidden,
        intermediate=intermediate,
        query_heads=query_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
    )
    query_width = query_heads * head_dim
    if query_width != hidden:
        hidden_square = f"{hidden}x{hidden}"
        query_weight_shape = f"{hidden}x{query_width}"
        output_weight_shape = f"{query_width}x{hidden}"
        text = text.replace(
            f"%query_weight: tensor<{hidden_square}xi8>",
            f"%query_weight: tensor<{query_weight_shape}xi8>",
        )
        text = text.replace(
            f"%output_weight: tensor<{hidden_square}xi8>",
            f"%output_weight: tensor<{output_weight_shape}xi8>",
        )
        text = section_replace(
            text, "%attention_query_weight_bf16",
            "%attention_key_weight_bf16", hidden_square,
            query_weight_shape,
        )
        text = section_replace(
            text, "%attention_output_weight_bf16",
            "%attention_query_2d", hidden_square,
            output_weight_shape,
        )
        query_begin = text.index("    %attention_query_2d")
        key_begin = text.index("    %attention_key_2d", query_begin)
        query_section = text[query_begin:key_begin]
        query_section = query_section.replace(hidden_square, query_weight_shape)
        query_section = query_section.replace(
            f"tensor<32x{hidden}xbf16>",
            f"tensor<32x{query_width}xbf16>",
        )
        # The dot input remains hidden-wide even though its result is wider.
        query_section = query_section.replace(
            f"(tensor<32x{query_width}xbf16>, "
            f"tensor<{query_weight_shape}xbf16>)",
            f"(tensor<32x{hidden}xbf16>, "
            f"tensor<{query_weight_shape}xbf16>)",
        )
        text = text[:query_begin] + query_section + text[key_begin:]

        text = text.replace(
            "%attention_query_heads = stablehlo.reshape %attention_query_2d :\n"
            f"        (tensor<32x{hidden}xbf16>)",
            "%attention_query_heads = stablehlo.reshape %attention_query_2d :\n"
            f"        (tensor<32x{query_width}xbf16>)",
        )
        context_begin = text.index("    %attention_context_shd")
        residual_begin = text.index("    %residual1", context_begin)
        context = text[context_begin:residual_begin]
        context = context.replace(
            f"tensor<32x{hidden}xbf16>",
            f"tensor<32x{query_width}xbf16>",
        )
        context = context.replace(
            f"tensor<{hidden_square}xbf16>",
            f"tensor<{output_weight_shape}xbf16>",
        )
        context = context.replace(
            f"tensor<32x{query_width}xbf16>, "
            f"tensor<{output_weight_shape}xbf16>) -> "
            f"tensor<32x{query_width}xbf16>",
            f"tensor<32x{query_width}xbf16>, "
            f"tensor<{output_weight_shape}xbf16>) -> "
            f"tensor<32x{hidden}xbf16>",
        )
        text = text[:context_begin] + context + text[residual_begin:]
    half = head_dim // 2
    groups = query_heads // kv_heads
    text = text.replace(
        "array<i64: 32, 12, 64,", f"array<i64: 32, {query_heads}, {half},"
    )
    text = text.replace(
        "array<i64: 32, 2, 64,", f"array<i64: 32, {kv_heads}, {half},"
    )
    text = text.replace(
        "dense<1.536000e+03>", f"dense<{format_float(float(hidden))}>"
    )
    text = text.replace(
        "dense<1.000000e+06>", f"dense<{format_float(rope_theta)}>"
    )
    text = text.replace(
        "dense<-1.562500e-02>",
        f"dense<{format_float(-2.0 / head_dim)}>",
    )
    text = text.replace(
        "dense<8.838835e-02>",
        f"dense<{format_float(1.0 / math.sqrt(head_dim))}>",
    )
    text = text.replace(
        "dense<1.000000e-06>", f"dense<{format_float(rms_norm_eps)}>"
    )
    expected = (
        f"tensor<32x{query_heads}x{head_dim}xbf16>",
        f"tensor<32x{kv_heads}x{groups}x{head_dim}xbf16>",
        f"tensor<{hidden}x{query_width}xi8>",
        f"tensor<{query_width}x{hidden}xi8>",
        f"tensor<{hidden}x{intermediate}xi8>",
    )
    missing = [marker for marker in expected if marker not in text]
    if missing:
        raise ValueError(f"generated decoder is missing shape markers: {missing}")
    if residual_scale != 1.0:
        shape = f"tensor<32x{hidden}xbf16>"
        for result, source in (("residual1", "attention_result"), ("result", "ffn_result")):
            marker = f"    %{result} = stablehlo.add"
            offset = text.index(marker)
            scalar = f"{source}_residual_scale"
            scaled = f"{source}_residual_scaled"
            prefix = (
                f"    %{scalar} = stablehlo.constant dense<{format_float(residual_scale)}> : {shape}\n"
                f"    %{scaled} = stablehlo.multiply %{source}, %{scalar} : {shape}\n"
            )
            suffix = text[offset:]
            suffix = suffix.replace(f"%{source}", f"%{scaled}", 1)
            text = text[:offset] + prefix + suffix
    rotary_dim = head_dim if rotary_dim is None else rotary_dim
    if not 0 < rotary_dim <= head_dim or rotary_dim % 2:
        raise ValueError("rotary_dim must be positive, even and <= head_dim")
    if rotary_dim != head_dim:
        start = text.index("    %attention_frequency_index")
        end = text.index("    %attention_query_f32", start)
        table = text[start:end]
        table = re.sub(rf"(?<![0-9]){head_dim // 2}(?=x|xf32|x1|>)", str(rotary_dim // 2), table)
        # Replace the scalar exponent denominator, not the attention scale.
        table = table.replace(format_float(-2.0 / head_dim), format_float(-2.0 / rotary_dim))
        text = text[:start] + table + text[end:]
        start = text.index("    %attention_query_f32")
        end = text.index("    %attention_key_grouped", start)
        ops = []
        half, tail = rotary_dim // 2, head_dim - rotary_dim
        for role, heads, result in (("query", query_heads, "attention_query_rope"),
                                    ("key", kv_heads, "attention_key_rope_kv")):
            stem = f"attention_{role}"
            full = f"tensor<32x{heads}x{head_dim}xf32>"
            part = f"tensor<32x{heads}x{half}xf32>"
            ops.append(f"    %{stem}_f32 = stablehlo.convert %{stem}_heads : (tensor<32x{heads}x{head_dim}xbf16>) -> {full}\n")
            for suffix, begin, stop, width in (("low", 0, half, half), ("high", half, rotary_dim, half), ("tail", rotary_dim, head_dim, tail)):
                ops.append(f'    %{stem}_{suffix} = "stablehlo.slice"(%{stem}_f32) {{start_indices = array<i64: 0, 0, {begin}>, limit_indices = array<i64: 32, {heads}, {stop}>, strides = array<i64: 1, 1, 1>}} : ({full}) -> tensor<32x{heads}x{width}xf32>\n')
            for product, value, trig in (("lc", "low", "cos"), ("hs", "high", "sin"), ("hc", "high", "cos"), ("ls", "low", "sin")):
                ops.append(f"    %{stem}_{product} = stablehlo.multiply %{stem}_{value}, %attention_{trig}_{'q' if role == 'query' else 'k'} : {part}\n")
            ops.append(f"    %{stem}_rl = stablehlo.subtract %{stem}_lc, %{stem}_hs : {part}\n")
            ops.append(f"    %{stem}_rh = stablehlo.add %{stem}_hc, %{stem}_ls : {part}\n")
            ops.append(f'    %{stem}_rope_f32 = "stablehlo.concatenate"(%{stem}_rl, %{stem}_rh, %{stem}_tail) {{dimension = 2 : i64}} : ({part}, {part}, tensor<32x{heads}x{tail}xf32>) -> {full}\n')
            ops.append(f"    %{result} = stablehlo.convert %{stem}_rope_f32 : ({full}) -> tensor<32x{heads}x{head_dim}xbf16>\n")
        text = text[:start] + ''.join(ops) + text[end:]
    if rope_attention_factor != 1.0:
        for trig, operation in (("cos", "cosine"), ("sin", "sine")):
            marker = f"    %attention_{trig} = stablehlo.{operation} %attention_angle : tensor<32x{rotary_dim // 2}xf32>"
            shape = f"tensor<32x{rotary_dim // 2}xf32>"
            text = text.replace(marker,
                f"    %attention_{trig}_raw = stablehlo.{operation} %attention_angle : {shape}\n"
                f"    %attention_{trig}_factor_scalar = stablehlo.constant dense<{format_float(rope_attention_factor)}> : tensor<f32>\n"
                f"    %attention_{trig}_factor = stablehlo.broadcast_in_dim %attention_{trig}_factor_scalar, dims = [] : (tensor<f32>) -> {shape}\n"
                f"    %attention_{trig} = stablehlo.multiply %attention_{trig}_raw, %attention_{trig}_factor : {shape}")
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument(
        "--source",
        type=Path,
        default=(
            here.parent / "examples" / "qwen2_5_1_5b_decoder_layer"
            / "decoder_layer_seq32.stablehlo.mlir"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--hidden-size", type=int, required=True)
    parser.add_argument("--intermediate-size", type=int, required=True)
    parser.add_argument("--query-heads", type=int, required=True)
    parser.add_argument("--kv-heads", type=int, required=True)
    parser.add_argument("--head-dim", type=int, required=True)
    parser.add_argument("--rope-theta", type=float, default=10_000.0)
    parser.add_argument("--rms-norm-eps", type=float, default=1.0e-5)
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--rotary-dim", type=int)
    parser.add_argument("--rope-attention-factor", type=float, default=1.0)
    args = parser.parse_args()
    text = generate(
        args.source,
        name=args.name,
        hidden=args.hidden_size,
        intermediate=args.intermediate_size,
        query_heads=args.query_heads,
        kv_heads=args.kv_heads,
        head_dim=args.head_dim,
        rope_theta=args.rope_theta,
        rms_norm_eps=args.rms_norm_eps,
        residual_scale=args.residual_scale,
        rotary_dim=args.rotary_dim,
        rope_attention_factor=args.rope_attention_factor,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
