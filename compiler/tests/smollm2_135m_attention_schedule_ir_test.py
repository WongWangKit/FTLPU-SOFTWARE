#!/usr/bin/env python3
"""Checks generic attention lowering creates a cycle-bounded Schedule IR."""

import argparse
import re
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        str(args.tool), "--input", str(args.input), "--output", str(args.output),
        "--pipeline", "ftlpu-stablehlo-to-schedule",
    ], check=True)
    text = args.output.read_text(encoding="utf-8")
    required = (
        "ftlpu.schedule.binding",
        "ftlpu.schedule.timeline",
        'name = "qkv"',
        'name = "rope"',
        'name = "qk"',
        'name = "softmax"',
        'name = "pv"',
        'name = "o_proj"',
        'access = "input"',
        'access = "output"',
        'access = "internal"',
        'role = "activation"',
        'role = "weight"',
        'role = "result"',
        'role = "constant"',
        "address = 7600 : i64",
        "address = 7800 : i64",
        'opcode = "max"',
        'opcode = "exp"',
        'opcode = "divide"',
        'opcode = "add"',
        'rhs_immediate = -1.000000e+09 : f32',
        'rhs_kind = "stream_f32"',
        'address = 8128 : i64',
        'kind = "fp16_rope_table", slices = [2, 18, 30, 31]',
        "address = 6000 : i64",
        'ftlpu.schedule.sxm',
        'opcode = "transpose"',
        'opcode = "permute"',
        'destination_streams = [32, 33, 34, 35',
        'weight_layout = "matrix_columns"',
        'accumulator_destination = "sram"',
        'accumulator_destination = "stream"',
        'accumulator_clear = true',
        'output_stream = 8 : i64',
        'output_stream = 20 : i64',
        'output_stream = 22 : i64',
    )
    missing = [item for item in required if item not in text]
    if missing:
        raise AssertionError(f"Attention Schedule IR is missing: {missing}")
    if "ftlpu.schedule.attention" in text:
        raise AssertionError("Attention Schedule IR still contains its compound op")
    if text.count("ftlpu.schedule.binding") != 9:
        raise AssertionError(
            "Attention Schedule IR must expose five inputs, one output, "
            "two causal masks, and one RoPE table")
    if text.count("ftlpu.schedule.timeline") != 6:
        raise AssertionError("Attention Schedule IR must expose six phase timelines")
    if 'opcode = "accumulate"' in text:
        raise AssertionError("attention schedule still uses legacy MEM accumulator commands")
    if 'lhs_kind = "stream_i8"' in text:
        raise AssertionError(
            "attention schedule still uses the post-softmax VXM repack pass"
        )
    accumulator_addresses = [
        int(value) for value in re.findall(
            r"ftlpu\.schedule\.mxm_issue \{[^\n]*"
            r"accumulator_address = (\d+) : i64",
            text,
        )
    ]
    if not accumulator_addresses or max(accumulator_addresses) >= 1024:
        raise AssertionError(
            "attention schedule exceeded the physical Vector MXM "
            "accumulator SRAM"
        )
    minimum_counts = {
        "ftlpu.schedule.mem_transfer": 8000,
        "ftlpu.schedule.mxm_issue": 12000,
        "ftlpu.schedule.vxm": 1400,
        'rhs_immediate = -1.000000e+09 : f32': 1,
    }
    for operation, minimum in minimum_counts.items():
        if text.count(operation) < minimum:
            raise AssertionError(
                f"attention schedule has too few {operation} operations")
    if text.count('opcode = "max"') != 36:
        raise AssertionError("attention schedule did not emit recurrent softmax max commands")
    if text.count('opcode = "exp"') != 36 or text.count('opcode = "divide"') != 36:
        raise AssertionError("attention schedule did not emit complete softmax exp/divide commands")
    if text.count('opcode = "iw"') != 4176:
        raise AssertionError("attention schedule did not emit all projection, QK, and PV IW commands")
    lines = text.splitlines()
    for packed_stream in range(16):
        if not any(
            'opcode = "write"' in line
            and 'address = 6000 : i64' in line
            and f'packed_stream = {packed_stream} : i64' in line
            for line in lines
        ):
            raise AssertionError(
                "softmax does not write its packed distributed16 result directly"
            )
    for direct_address in (0, 7800):
        for packed_stream in range(32, 48):
            if not any(
                'opcode = "write"' in line
                and f'address = {direct_address} : i64' in line
                and f'packed_stream = {packed_stream} : i64' in line
                for line in lines
            ):
                raise AssertionError(
                    "Block8 Q/K FIFO or V output is not written directly "
                    "from all 16 westbound MXM result streams")
    block8_compute = sum(
        'opcode = "compute"' in line
        and 'compute_mode = "block8"' in line
        for line in lines
    )
    vector_compute = sum(
        'opcode = "compute"' in line
        and 'compute_mode = "block8"' not in line
        for line in lines
    )
    block8_iw = sum(
        'opcode = "iw"' in line
        and 'weight_input_mode = "int8_dequant_bf16"' in line
        for line in lines
    )
    if block8_compute != 3456 or block8_iw != 3600:
        raise AssertionError(
            "Q/K/V/O projections did not use complete Block8 execution")
    if any(
        'opcode = "compute"' in line
        and 'compute_mode = "block8"' in line
        and 'activation_stream_base = 16 : i64' not in line
        for line in lines
    ):
        raise AssertionError(
            "Block8 activation producer and consumer stream bases diverged")
    if any(
        'opcode = "accumulator_read"' in line
        and 'compute_mode = "block8"' in line
        for line in lines
    ):
        raise AssertionError(
            "Block8 projections still use a separate accumulator drain")
    final_block8 = [
        line for line in lines
        if 'opcode = "compute"' in line
        and 'compute_mode = "block8"' in line
        and 'accumulator_destination = "stream"' in line
    ]
    if not final_block8 or any(
        'accumulator_clear = true' not in line for line in final_block8
    ):
        raise AssertionError(
            "Block8 final partials must emit and clear the accumulator")
    if vector_compute != 576:
        raise AssertionError("QK and PV must remain on Vector MXM compute")
    if text.count("ftlpu.schedule.mxm_dequant") != 3600:
        raise AssertionError("attention projection local dequant is incomplete")
    if 'compute_mode = "block8"' not in text \
            or 'repeat_count = 4 : i64' not in text:
        raise AssertionError("Block8 compute does not issue four 8-row waves")
    if text.count("ftlpu.schedule.sxm") < 1000:
        raise AssertionError("attention schedule did not emit all probability and V transpose/permute waves")
    phase_matches = {
        name: (int(start), int(end))
        for end, name, start in re.findall(
            r'ftlpu\.schedule\.timeline \{end = (\d+) : i64, '
            r'name = "([^"]+)", start = (\d+) : i64\}',
            text,
        )
    }
    single_mxm = "mxms_per_hemisphere = 1 : i64" in text
    qkv_limit = 70000 if single_mxm else 22862
    output_limit = 30000 if single_mxm else 22428
    if phase_matches["qkv"][1] - phase_matches["qkv"][0] > qkv_limit:
        raise AssertionError("Q/K/V projection and GQA preparation lost their pipeline")
    if phase_matches["o_proj"][1] - phase_matches["o_proj"][0] > output_limit:
        raise AssertionError("O projection lost its weight-prefetch pipeline")
    qk_start, qk_end = phase_matches["qk"]
    pv_start, pv_end = phase_matches["pv"]
    pv_mxm = [
        line for line in lines
        if "ftlpu.schedule.mxm_issue" in line
        and (match := re.search(r"cycle = (\d+) : i64", line))
        and pv_start <= int(match.group(1)) < pv_end
    ]
    if any('opcode = "accumulator_read"' in line for line in pv_mxm):
        raise AssertionError(
            "PV still pre-clears or separately drains its accumulator")
    if any("ftlpu.schedule.vxm" in line for line in lines
           if (match := re.search(r"cycle = (\d+) : i64", line))
           and pv_start <= int(match.group(1)) < pv_end):
        raise AssertionError("PV still uses VXM for its final BF16 result")
    pv_compute = [line for line in pv_mxm if 'opcode = "compute"' in line]
    pv_final_compute = [
        line for line in pv_compute
        if 'accumulator_destination = "stream"' in line
    ]
    if len(pv_compute) != 288 or len(pv_final_compute) != 72:
        raise AssertionError("PV did not emit the expected partial reductions")
    if any('accumulator_clear = true' not in line for line in pv_final_compute):
        raise AssertionError(
            "PV final partials must emit and clear the accumulator")
    if any(
        'accumulator_output_format = "bf16"' not in line
        for line in pv_final_compute
    ):
        raise AssertionError("PV final partials must convert to BF16 in MXM")
    if any(
        'accumulator_destination = "sram"' not in line
        for line in pv_compute
        if line not in pv_final_compute
    ):
        raise AssertionError("PV intermediate partials must remain in SRAM")
    o_start, o_end = phase_matches["o_proj"]
    o_compute_cycles = [
        int(match.group(1))
        for line in lines
        if "ftlpu.schedule.mxm_issue" in line
        and 'opcode = "compute"' in line
        and 'compute_mode = "block8"' in line
        and (match := re.search(r"cycle = (\d+) : i64", line))
        and o_start <= int(match.group(1)) < o_end
    ]
    if not o_compute_cycles:
        raise AssertionError("O projection has no Block8 MXM compute")
    o_compute_start = min(o_compute_cycles)
    for unit_id in (0, 1):
        unit_cycles = sorted({
            int(match.group(1))
            for line in lines
            if "ftlpu.schedule.mxm_issue" in line
            and 'opcode = "compute"' in line
            and 'compute_mode = "block8"' in line
            and f"unit_id = {unit_id} : i64" in line
            and (match := re.search(r"cycle = (\d+) : i64", line))
            and o_start <= int(match.group(1)) < o_end
        })
        if any(
            right - left != 4
            for left, right in zip(unit_cycles, unit_cycles[1:])
        ):
            raise AssertionError(
                "O projection MXM compute stream contains a bubble"
            )
    if any(
        "ftlpu.schedule.vxm" in line
        and (match := re.search(r"cycle = (\d+) : i64", line))
        and o_start <= int(match.group(1)) < o_end
        for line in lines
    ):
        raise AssertionError(
            "O projection still occupies VXM instead of passive routing")
    tap_lines = [
        line for line in lines if 'opcode = "write_tap"' in line
    ]
    expanded_taps = sum(
        int(re.search(r"repeat_count = (\d+)", line).group(1))
        * (
            int(match.group(1))
            if (match := re.search(r"wave_count = (\d+)", line))
            else 1
        )
        for line in tap_lines
    )
    if expanded_taps != 4608:
        raise AssertionError(
            "O-projection input passive routing lost local MEM taps")
    pv_compute_cycles = sorted({
        int(match.group(1))
        for line in pv_compute
        if (match := re.search(r"cycle = (\d+) : i64", line))
    })
    if any(
        right - left > 32
        for left, right in zip(pv_compute_cycles, pv_compute_cycles[1:])
    ):
        raise AssertionError("PV still has an inter-key or inter-wave MXM bubble")
    qk_compute_cycles = sorted({
        int(match.group(1))
        for line in lines
        if "ftlpu.schedule.mxm_issue" in line
        and 'opcode = "compute"' in line
        and 'compute_mode = "block8"' not in line
        and (match := re.search(r"cycle = (\d+) : i64", line))
        and qk_start <= int(match.group(1)) < qk_end
    })
    if not qk_compute_cycles or any(
        right - left != 32
        for left, right in zip(qk_compute_cycles, qk_compute_cycles[1:])
    ):
        raise AssertionError("QK MXM compute waves are no longer contiguous")
    qkv_end = phase_matches["qkv"][1]
    qkv_vxm_passes = [
        line for line in lines
        if "ftlpu.schedule.vxm" in line
        and 'opcode = "pass"' in line
        and (match := re.search(r"cycle = (\d+) : i64", line))
        and int(match.group(1)) < qkv_end
    ]
    if qkv_vxm_passes:
        raise AssertionError(
            "Q placement still uses a post-projection VXM hemisphere pass")

    rope_match = re.search(
        r'ftlpu\.schedule\.vxm %arg1, %arg1 \{[^\n]*'
        r'cycle = (\d+) : i64[^\n]*opcode = "multiply"[^\n]*'
        r'queue = 0 : i64[^\n]*repeat_count = (\d+) : i64[^\n]*'
        r'repeat_interval = (\d+) : i64',
        text,
    )
    if not rope_match:
        raise AssertionError("query RoPE FIFO consumer is missing")
    rope_start, rope_count, rope_interval = map(int, rope_match.groups())
    rope_end = rope_start + (rope_count - 1) * rope_interval
    block8_cycles = [
        int(cycle)
        for cycle in re.findall(
            r'ftlpu\.schedule\.mxm_issue \{[^\n]*'
            r'compute_mode = "block8"[^\n]*cycle = (\d+) : i64[^\n]*'
            r'opcode = "compute"',
            text,
        )
    ]
    if not any(rope_start <= cycle <= rope_end for cycle in block8_cycles):
        raise AssertionError("RoPE drains serially instead of overlapping MXM projection")
    for column in range(4):
        if f"weight_column = {column} : i64" not in text:
            raise AssertionError(f"MXM IW weight column {column} is missing")


if __name__ == "__main__":
    main()
