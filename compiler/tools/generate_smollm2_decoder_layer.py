#!/usr/bin/env python3
"""Composes the standard StableHLO RMSNorm, attention, and FFN examples."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


VALUE = re.compile(r"%([A-Za-z_][A-Za-z0-9_]*)")


def function_body(path: Path) -> tuple[str, str]:
    text = path.read_text(encoding="utf-8")
    signature_end = text.index("{", text.index("func.func"))
    body_end = text.rfind("}")
    body_end = text.rfind("}", 0, body_end)
    body = text[signature_end + 1 : body_end]
    matches = list(
        re.finditer(r"(?m)^\s*return\s+(%[A-Za-z_][A-Za-z0-9_]*)", body)
    )
    if not matches:
        raise ValueError(f"{path} has no function return")
    match = matches[-1]
    result = match.group(1)
    body = body[: match.start()]
    return body, result


def inline(path: Path, prefix: str, arguments: dict[str, str]) -> tuple[str, str]:
    body, result = function_body(path)

    def rename(match: re.Match[str]) -> str:
        value = match.group(0)
        return arguments.get(value, f"%{prefix}_{match.group(1)}")

    return VALUE.sub(rename, body).strip(), rename(VALUE.match(result))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rmsnorm", type=Path, required=True)
    parser.add_argument("--attention", type=Path, required=True)
    parser.add_argument("--ffn", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--weight-scales",
        type=Path,
        help="JSON metadata emitted by import_hf_decoder_layer.py",
    )
    args = parser.parse_args()

    rms1, rms1_result = inline(args.rmsnorm, "rms1", {
        "%x": "%x",
        "%weight": "%input_norm_weight",
    })
    attention, attention_result = inline(args.attention, "attention", {
        "%input": rms1_result,
        "%query_weight": "%query_weight",
        "%key_weight": "%key_weight",
        "%value_weight": "%value_weight",
        "%output_weight": "%output_weight",
    })
    rms2, rms2_result = inline(args.rmsnorm, "rms2", {
        "%x": "%residual1",
        "%weight": "%post_attention_norm_weight",
    })
    ffn, ffn_result = inline(args.ffn, "ffn", {
        "%x": rms2_result,
        "%gate_w": "%gate_weight",
        "%up_w": "%up_weight",
        "%down_w": "%down_weight",
    })

    if args.weight_scales:
        metadata = json.loads(args.weight_scales.read_text(encoding="utf-8"))
        scales = metadata["scales"]
        attention_scale_ops = []
        for role, stem in (
            ("query", "query_weight"),
            ("key", "key_weight"),
            ("value", "value_weight"),
            ("output", "output_weight"),
        ):
            attention_scale_ops.append(
                f"""%attention_{stem}_scale = stablehlo.constant dense<{scales[role]:.9e}> : tensor<f16>
    %attention_{stem}_scale_broadcast = stablehlo.broadcast_in_dim %attention_{stem}_scale, dims = [] :
        (tensor<f16>) -> tensor<{576 if role != "output" else 576}x{576 if role in ("query", "output") else 192}xf16>
    %attention_{stem}_scaled = stablehlo.multiply %attention_{stem}_f16, %attention_{stem}_scale_broadcast :
        tensor<{576 if role != "output" else 576}x{576 if role in ("query", "output") else 192}xf16>"""
            )
            attention = attention.replace(
                f"%attention_{stem}_f16,\n        contracting",
                f"%attention_{stem}_scaled,\n        contracting",
            )
        attention = attention.replace(
            "%attention_query_2d =",
            "\n    ".join(attention_scale_ops) + "\n\n    %attention_query_2d =",
        )

        ffn_scale_ops = []
        for role, stem, shape in (
            ("gate", "gate_w", "576x1536"),
            ("up", "up_w", "576x1536"),
            ("down", "down_w", "1536x576"),
        ):
            ffn_scale_ops.append(
                f"""%ffn_{stem}_scale = stablehlo.constant dense<{scales[role]:.9e}> : tensor<f32>
    %ffn_{stem}_scale_broadcast = stablehlo.broadcast_in_dim %ffn_{stem}_scale, dims = [] :
        (tensor<f32>) -> tensor<{shape}xf32>
    %ffn_{stem}_scaled = stablehlo.multiply %ffn_{stem}_f, %ffn_{stem}_scale_broadcast :
        tensor<{shape}xf32>"""
            )
            ffn = ffn.replace(
                f"%ffn_{stem}_f,\n      contracting",
                f"%ffn_{stem}_scaled,\n      contracting",
            )
        ffn = ffn.replace(
            "%ffn_gate =",
            "\n    ".join(ffn_scale_ops) + "\n\n    %ffn_gate =",
        )

    output = f"""module {{
  func.func @smollm2_135m_decoder_layer_seq128(
      %x: tensor<128x576xf16>,
      %input_norm_weight: tensor<576xf16>,
      %query_weight: tensor<576x576xi8>,
      %key_weight: tensor<576x192xi8>,
      %value_weight: tensor<576x192xi8>,
      %output_weight: tensor<576x576xi8>,
      %post_attention_norm_weight: tensor<576xf16>,
      %gate_weight: tensor<576x1536xi8>,
      %up_weight: tensor<576x1536xi8>,
      %down_weight: tensor<1536x576xi8>) -> tensor<128x576xf16> {{
    {rms1}

    {attention}
    %residual1 = stablehlo.add %x, {attention_result} :
        tensor<128x576xf16>

    {rms2}

    {ffn}
    %result = stablehlo.add %residual1, {ffn_result} :
        tensor<128x576xf16>
    return %result : tensor<128x576xf16>
  }}
}}
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
