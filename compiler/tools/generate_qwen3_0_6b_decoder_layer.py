#!/usr/bin/env python3
"""Generate the Qwen3-0.6B seq32 one-layer StableHLO fixture.

The checked-in Qwen2.5 layer remains the canonical spelling for attention,
softmax, residual, and SwiGLU.  This generator changes the model dimensions,
removes the Qwen2 projection biases, and inserts Qwen3's per-head Q/K
RMSNorms between projection and RoPE.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def section_replace(text: str, begin: str, end: str, old: str, new: str) -> str:
    first = text.index(begin)
    last = text.index(end, first)
    return text[:first] + text[first:last].replace(old, new) + text[last:]


def norm(prefix: str, source: str, rows: int, heads: int) -> str:
    matrix = f"{rows}x128"
    original = f"32x{heads}x128"
    return f"""    %attention_{prefix}_norm_input = stablehlo.reshape %attention_{source}_heads :
        (tensor<{original}xbf16>) -> tensor<{matrix}xbf16>
    %attention_{prefix}_norm_input_f32 = stablehlo.convert %attention_{prefix}_norm_input :
        (tensor<{matrix}xbf16>) -> tensor<{matrix}xf32>
    %attention_{prefix}_norm_weight_f32 = stablehlo.convert %{prefix}_norm_weight :
        (tensor<128xbf16>) -> tensor<128xf32>
    %attention_{prefix}_norm_square = stablehlo.multiply %attention_{prefix}_norm_input_f32, %attention_{prefix}_norm_input_f32 :
        tensor<{matrix}xf32>
    %attention_{prefix}_norm_zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %attention_{prefix}_norm_sum = "stablehlo.reduce"(%attention_{prefix}_norm_square, %attention_{prefix}_norm_zero) ({{
      ^bb0(%attention_{prefix}_norm_lhs: tensor<f32>, %attention_{prefix}_norm_rhs: tensor<f32>):
        %attention_{prefix}_norm_value = stablehlo.add %attention_{prefix}_norm_lhs, %attention_{prefix}_norm_rhs : tensor<f32>
        stablehlo.return %attention_{prefix}_norm_value : tensor<f32>
    }}) {{dimensions = array<i64: 1>}} :
        (tensor<{matrix}xf32>, tensor<f32>) -> tensor<{rows}xf32>
    %attention_{prefix}_norm_hidden = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %attention_{prefix}_norm_hidden_broadcast = stablehlo.broadcast_in_dim %attention_{prefix}_norm_hidden, dims = [] :
        (tensor<f32>) -> tensor<{rows}xf32>
    %attention_{prefix}_norm_mean = stablehlo.divide %attention_{prefix}_norm_sum, %attention_{prefix}_norm_hidden_broadcast : tensor<{rows}xf32>
    %attention_{prefix}_norm_epsilon = stablehlo.constant dense<1.000000e-06> : tensor<f32>
    %attention_{prefix}_norm_epsilon_broadcast = stablehlo.broadcast_in_dim %attention_{prefix}_norm_epsilon, dims = [] :
        (tensor<f32>) -> tensor<{rows}xf32>
    %attention_{prefix}_norm_variance = stablehlo.add %attention_{prefix}_norm_mean, %attention_{prefix}_norm_epsilon_broadcast : tensor<{rows}xf32>
    %attention_{prefix}_norm_inverse_rms = stablehlo.rsqrt %attention_{prefix}_norm_variance : tensor<{rows}xf32>
    %attention_{prefix}_norm_inverse_rms_broadcast = stablehlo.broadcast_in_dim %attention_{prefix}_norm_inverse_rms, dims = [0] :
        (tensor<{rows}xf32>) -> tensor<{matrix}xf32>
    %attention_{prefix}_norm_normalized = stablehlo.multiply %attention_{prefix}_norm_input_f32, %attention_{prefix}_norm_inverse_rms_broadcast :
        tensor<{matrix}xf32>
    %attention_{prefix}_norm_weight_broadcast = stablehlo.broadcast_in_dim %attention_{prefix}_norm_weight_f32, dims = [1] :
        (tensor<128xf32>) -> tensor<{matrix}xf32>
    %attention_{prefix}_norm_scaled = stablehlo.multiply %attention_{prefix}_norm_normalized, %attention_{prefix}_norm_weight_broadcast :
        tensor<{matrix}xf32>
    %attention_{prefix}_norm_result = stablehlo.convert %attention_{prefix}_norm_scaled :
        (tensor<{matrix}xf32>) -> tensor<{matrix}xbf16>
    %attention_{prefix}_norm_heads = stablehlo.reshape %attention_{prefix}_norm_result :
        (tensor<{matrix}xbf16>) -> tensor<{original}xbf16>
"""


def generate(source: Path) -> str:
    text = source.read_text(encoding="utf-8")
    text = text.replace("qwen2_5_1_5b", "qwen3_0_6b")
    text = (text.replace("8960", "3072")
                .replace("1536", "1024")
                .replace("256", "1024"))
    text = text.replace("1.536000e+03", "1.024000e+03")

    text = text.replace("%query_weight: tensor<1024x1024xi8>",
                        "%query_weight: tensor<1024x2048xi8>")
    text = text.replace("%output_weight: tensor<1024x1024xi8>",
                        "%output_weight: tensor<2048x1024xi8>")
    text = text.replace(
        "      %query_bias: tensor<1024xbf16>,\n"
        "      %key_bias: tensor<1024xbf16>,\n"
        "      %value_bias: tensor<1024xbf16>,\n", "")
    text = text.replace(
        "      %post_attention_norm_weight: tensor<1024xbf16>,",
        "      %query_norm_weight: tensor<128xbf16>,\n"
        "      %key_norm_weight: tensor<128xbf16>,\n"
        "      %post_attention_norm_weight: tensor<1024xbf16>,")

    text = section_replace(
        text, "%attention_query_weight_bf16", "%attention_key_weight_bf16",
        "1024x1024", "1024x2048")
    text = section_replace(
        text, "%attention_output_weight_bf16", "%attention_query_2d",
        "1024x1024", "2048x1024")
    text = section_replace(
        text, "%attention_query_2d", "%attention_key_2d",
        "1024x1024", "1024x2048")
    text = section_replace(
        text, "%attention_query_2d", "%attention_key_2d",
        "32x1024", "32x2048")
    text = text.replace(
        "(tensor<32x2048xbf16>, tensor<1024x2048xbf16>) -> "
        "tensor<32x2048xbf16>",
        "(tensor<32x1024xbf16>, tensor<1024x2048xbf16>) -> "
        "tensor<32x2048xbf16>")

    bias_begin = text.index("    %attention_query_bias_2d")
    heads_begin = text.index("    %attention_query_heads", bias_begin)
    text = text[:bias_begin] + text[heads_begin:]
    text = text.replace("%attention_query_biased", "%attention_query_2d")
    text = text.replace("%attention_key_biased", "%attention_key_2d")
    text = text.replace("%attention_value_biased", "%attention_value_2d")
    text = text.replace(
        "%attention_query_heads = stablehlo.reshape %attention_query_2d :\n"
        "        (tensor<32x1024xbf16>)",
        "%attention_query_heads = stablehlo.reshape %attention_query_2d :\n"
        "        (tensor<32x2048xbf16>)")

    text = text.replace("tensor<32x12x", "tensor<32x16x")
    text = text.replace("tensor<12x32x", "tensor<16x32x")
    text = text.replace("tensor<12x32xbf16>", "tensor<16x32xbf16>")
    text = text.replace("array<i64: 32, 12,", "array<i64: 32, 16,")
    text = text.replace("tensor<32x2x", "tensor<32x8x")
    text = text.replace("tensor<2x32x", "tensor<8x32x")
    text = text.replace("tensor<2x32xbf16>", "tensor<8x32xbf16>")
    text = text.replace("array<i64: 32, 2,", "array<i64: 32, 8,")
    text = text.replace("tensor<32x8x6x128", "tensor<32x8x2x128")

    insertion = text.index("    %attention_theta_f32")
    qk_norm = norm("query", "query", 512, 16) + "\n" + norm(
        "key", "key", 256, 8) + "\n"
    text = text[:insertion] + qk_norm + text[insertion:]
    text = text.replace(
        "%attention_query_f32 = stablehlo.convert %attention_query_heads",
        "%attention_query_f32 = stablehlo.convert %attention_query_norm_heads")
    text = text.replace(
        "%attention_key_f32 = stablehlo.convert %attention_key_heads",
        "%attention_key_f32 = stablehlo.convert %attention_key_norm_heads")

    context_begin = text.index("    %attention_context_shd")
    residual_begin = text.index("    %residual1", context_begin)
    context = text[context_begin:residual_begin]
    context = context.replace("tensor<32x1024xbf16>",
                              "tensor<32x2048xbf16>")
    context = context.replace("tensor<1024x1024xbf16>",
                              "tensor<2048x1024xbf16>")
    context = context.replace(
        "tensor<32x2048xbf16>, tensor<2048x1024xbf16>) -> "
        "tensor<32x2048xbf16>",
        "tensor<32x2048xbf16>, tensor<2048x1024xbf16>) -> "
        "tensor<32x1024xbf16>")
    text = text[:context_begin] + context + text[residual_begin:]
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument(
        "--source", type=Path,
        default=here.parent / "examples" / "qwen2_5_1_5b_decoder_layer" /
                "decoder_layer_seq32.stablehlo.mlir")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(generate(args.source), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
