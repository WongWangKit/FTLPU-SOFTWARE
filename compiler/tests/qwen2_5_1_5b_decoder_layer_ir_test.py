#!/usr/bin/env python3
"""Lowers a Qwen2.5-1.5B decoder layer through the public FTLPU IRs."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path


def lower(tool: Path, target: Path, source: Path, output: Path,
          pipeline: str, weight_bank: int | None,
          mxm_execution: str) -> str:
    command = [
        str(tool), "--input", str(source), "--output", str(output),
        "--pipeline", pipeline, "--mxm-execution", mxm_execution,
        "--ffn-schedule", "tail", "--target-config", str(target),
        "--rmsnorm-strategy", "vxm-feedback",
    ]
    if weight_bank is not None:
        command += ["--weight-bank", str(weight_bank)]
    subprocess.run(command, check=True)
    return output.read_text(encoding="utf-8")


def require(text: str, markers: tuple[str, ...], layer: str) -> None:
    missing = [marker for marker in markers if marker not in text]
    if missing:
        raise AssertionError(f"{layer} IR is missing {missing}")


def parse_placement(body: str) -> dict[str, object]:
    def integer(name: str) -> int:
        match = re.search(rf"\b{name} = (-?\d+) : i64", body)
        if not match:
            raise AssertionError(f"placement is missing {name}: {body}")
        return int(match.group(1))

    kind = re.search(r'\bkind = "([^"]+)"', body)
    slices = re.search(r"\bslices = \[([^\]]*)\]", body)
    if not kind or not slices:
        raise AssertionError(f"malformed placement: {body}")
    return {
        "bank": integer("bank"),
        "base": integer("base_row"),
        "count": integer("instruction_count"),
        "kind": kind.group(1),
        "slices": tuple(int(value) for value in re.findall(r"\d+", slices.group(1))),
    }


def validate_attention_output_allocation(tensor: str) -> None:
    pv = next((line for line in tensor.splitlines()
               if "ftlpu.tensor.batch_matmul_task" in line
               and 'kind = "pv"' in line), None)
    output = next((line for line in tensor.splitlines()
                   if "ftlpu.tensor.projection_task" in line
                   and 'kind = "output"' in line), None)
    if pv is None or output is None:
        raise AssertionError("tensor IR is missing PV or output projection")

    def named(line: str, name: str) -> dict[str, object]:
        match = re.search(rf"\b{name} = \{{([^{{}}]+)\}}", line)
        if not match:
            raise AssertionError(f"memory plan has no {name}")
        return parse_placement(match.group(1))

    context = named(pv, "context")
    result = named(output, "result")
    shared_slices = set(context["slices"]).intersection(result["slices"])
    context_end = int(context["base"]) + int(context["count"])
    result_end = int(result["base"]) + int(result["count"])
    rows_overlap = (int(context["base"]) < result_end
                    and int(result["base"]) < context_end)
    if (context["bank"] == result["bank"] and shared_slices and rows_overlap):
        raise AssertionError(
            "O projection result aliases its still-live PV context: "
            f"context={context}, result={result}"
        )


def validate_paged_weights(tensor: str, target_config: Path,
                           weight_bank: int, mxm_execution: str) -> None:
    placements = [parse_placement(match.group(1)) for match in
                  re.finditer(r"placement = \{([^{}]+)\}", tensor)]
    weights = [placement for placement in placements
               if str(placement["kind"]).startswith("w8a16_")]
    for name in ("query_weight", "key_weight", "value_weight", "output_weight"):
        match = re.search(rf"\b{name} = \{{([^{{}}]+)\}}", tensor)
        if not match:
            raise AssertionError(f"attention memory plan has no {name}")
        weights.append(parse_placement(match.group(1)))
    for line in tensor.splitlines():
        if "ftlpu.tensor.rms_norm_task" not in line:
            continue
        section = line.split("weight_allocations = [", 1)
        if len(section) != 2:
            raise AssertionError("RMSNorm task has no weight allocation")
        match = re.search(r"placement = \{([^{}]+)\}", section[1])
        if not match:
            raise AssertionError("RMSNorm weight has no placement")
        weights.append(parse_placement(match.group(1)))

    expected = Counter({
        ("w8a16_attention_weight_striped", 1920): 2,
        ("w8a16_attention_weight_striped", 1280): 1,
        ("w8a16_mxm_weight_striped", 1920): 1,
        ("w8a16_mxm_weight_wave_striped", 2048): 3,
        ("fp16_vxm_gamma_broadcast", 1536): 2,
    }) if mxm_execution == "vector" else Counter({
        ("w8a16_attention_weight_striped", 4608): 1,
        ("w8a16_attention_weight_striped", 768): 2,
        ("w8a16_mxm_weight_striped", 4608): 1,
        ("w8a16_block8_weight_wave_striped", 26880): 3,
        ("fp16_vxm_row_parallel_8", 1536): 2,
    })
    observed = Counter((str(weight["kind"]), int(weight["count"]))
                       for weight in weights)
    if observed != expected:
        raise AssertionError(
            f"paged weight layouts differ: observed={observed}, expected={expected}")

    target = json.loads(target_config.read_text(encoding="utf-8"))
    bank_rows = int(target.get("memory", {}).get("words_per_bank", 8192))
    for weight in weights:
        if (mxm_execution != "vector"
                and weight["bank"] != weight_bank):
            raise AssertionError(
                f"weight is in bank {weight['bank']}, expected {weight_bank}")
        if weight["base"] < 0 or weight["base"] + weight["count"] > bank_rows:
            raise AssertionError(
                f"weight placement exceeds {bank_rows} rows: {weight}")

    if mxm_execution != "vector":
        for index, lhs in enumerate(weights):
            lhs_slices = set(lhs["slices"])
            lhs_end = int(lhs["base"]) + int(lhs["count"])
            for rhs in weights[index + 1:]:
                if (lhs["bank"] != rhs["bank"]
                        or not lhs_slices.intersection(rhs["slices"])):
                    continue
                rhs_end = int(rhs["base"]) + int(rhs["count"])
                if int(lhs["base"]) < rhs_end and int(rhs["base"]) < lhs_end:
                    raise AssertionError(
                        "paged weights overlap on a shared slice: "
                        f"{lhs} vs {rhs}")

    working_bank = (weight_bank + 1) % int(
        target.get("memory", {}).get("banks_per_slice", 1)
    )
    if working_bank == weight_bank:
        raise AssertionError("paged lowering requires a separate working bank")

    attention_workspaces = (
        "input_staging", "query", "key", "value", "score",
        "score_mxm1", "exp", "exp_mxm1", "causal_mask",
        "causal_mask_mxm1", "fused_score", "fused_score_bank1",
        "fused_causal_mask", "fused_causal_mask_bank1",
        "probability_pack", "probability_diagonal", "rope",
        "rope_staging", "rope_product", "context",
        "output_activation", "result",
    )
    if mxm_execution != "vector":
        for name in attention_workspaces:
            matches = re.findall(rf"\b{name} = \{{([^{{}}]+)\}}", tensor)
            for body in matches:
                placement = parse_placement(body)
                if placement["bank"] != working_bank:
                    raise AssertionError(
                        f"attention workspace {name} is in weight bank: "
                        f"{placement}"
                    )

    for line in tensor.splitlines():
        if "ftlpu.tensor.rms_norm_task" not in line:
            continue
        for section_name, end_marker in (
            ("result_allocations", "scratch_allocations"),
            ("scratch_allocations", "weight_allocations"),
        ):
            section = line.split(f"{section_name} = [", 1)[1]
            section = section.split(f"], {end_marker}", 1)[0]
            for body in re.findall(r"placement = \{([^{}]+)\}", section):
                placement = parse_placement(body)
                if (mxm_execution != "vector"
                        and placement["bank"] != working_bank):
                    raise AssertionError(
                        f"RMSNorm {section_name} is in weight bank: "
                        f"{placement}"
                    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--target-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weight-bank", type=int)
    parser.add_argument("--mxm-execution", choices=("block8", "vector"),
                        default="block8")
    parser.add_argument("--seq-len", type=int, default=128)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.input, args.output_dir / "decoder_layer.stablehlo.mlir")

    stablehlo = args.input.read_text(encoding="utf-8")
    require(stablehlo, (
        f"tensor<{args.seq_len}x1536xbf16>", "tensor<1536x8960xi8>",
        "tensor<8960x1536xi8>",
        f"tensor<{args.seq_len}x12x128xbf16>",
        f"tensor<{args.seq_len}x2x128xbf16>", "dense<1.000000e+06>",
        "dense<1.000000e-06>", "stablehlo.compare GE",
    ), "StableHLO")

    kernel = lower(args.tool, args.target_config, args.input,
                   args.output_dir / "decoder_layer.kernel.mlir",
                   "ftlpu-stablehlo-to-kernel", args.weight_bank,
                   args.mxm_execution)
    require(kernel, (
        "ftlpu.kernel.rms_norm", "ftlpu.kernel.rope",
        "ftlpu.kernel.softmax", "ftlpu.kernel.batch_matmul",
        "ftlpu.kernel.swish", "head_dim = 128 : i64",
        "query_heads = 12 : i64", "kv_heads = 2 : i64",
        "theta = 1.000000e+06 : f32", "n = 8960 : i64",
    ), "Kernel")

    tensor = lower(args.tool, args.target_config, args.input,
                   args.output_dir / "decoder_layer.tensor.mlir",
                   "ftlpu-stablehlo-to-tensor", args.weight_bank,
                   args.mxm_execution)
    require(tensor, (
        "ftlpu.tensor.rms_norm_task", "ftlpu.tensor.projection_task",
        "ftlpu.tensor.rope_task", "ftlpu.tensor.softmax_task",
        "ftlpu.tensor.swish_task", 'kind = "w8a16_mxm_weight_striped"',
        'kind = "fp16_mxm_distributed_16"',
    ), "Tensor")
    if args.mxm_execution == "vector":
        validate_attention_output_allocation(tensor)
    if args.weight_bank is not None:
        validate_paged_weights(tensor, args.target_config, args.weight_bank,
                               args.mxm_execution)

    stream = lower(args.tool, args.target_config, args.input,
                   args.output_dir / "decoder_layer.stream.mlir",
                   "ftlpu-stablehlo-to-stream", args.weight_bank,
                   args.mxm_execution)
    require(stream, (
        "ftlpu.stream.rms_norm_task", "ftlpu.stream.projection_task",
        "ftlpu.stream.rope_task", "ftlpu.stream.softmax_task",
        "ftlpu.stream.swish_task", "ftlpu.stream.batch_matmul_task",
        "head_dim = 128 : i64", "query_heads = 12 : i64",
        "kv_heads = 2 : i64", "rope_theta = 1.000000e+06 : f32",
        "stream_count = 16 : i64",
    ), "Stream")
    for legacy in ("ftlpu.kernel.attention", "ftlpu.tensor.attention",
                   "ftlpu.stream.attention", "ftlpu.stream.ffn"):
        if legacy in stream:
            raise AssertionError(f"legacy compound op remained: {legacy}")


if __name__ == "__main__":
    main()
