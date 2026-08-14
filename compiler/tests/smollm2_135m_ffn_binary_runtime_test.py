#!/usr/bin/env python3
"""Compiles the SmolLM2-135M FFN and validates its CModel result."""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opt", type=Path, required=True)
    parser.add_argument("--translate", type=Path, required=True)
    parser.add_argument("--runtime-test", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-config", type=Path, required=True)
    parser.add_argument("--ffn-schedule", choices=("tail", "fused"),
                        default="tail")
    parser.add_argument("--mxm-execution", choices=("auto", "legacy", "block8"),
                        default="legacy")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    commands = args.output_dir / "ffn.command.mlir"
    binary = args.output_dir / "ffn.ftlpu"
    trace = args.output_dir / "ffn.runtime.csv"
    pipeline = args.output_dir / "ffn.pipeline.svg"
    compile_command = [
        str(args.opt), "--input", str(args.input), "--output", str(commands),
        "--pipeline", "ftlpu-stablehlo-to-commands",
        "--ffn-schedule", args.ffn_schedule,
        "--mxm-execution", args.mxm_execution,
    ]
    compile_command.extend(["--target-config", str(args.target_config)])
    subprocess.run(compile_command, check=True)
    command_text = commands.read_text(encoding="utf-8")
    block8_selected = args.mxm_execution in ("auto", "block8")
    if not block8_selected:
        for marker in (
            'weight_input_mode = "int8_dequant_bf16"',
            "ftlpu.command.mxm_dequant",
        ):
            if marker not in command_text:
                raise RuntimeError(
                    f"Vector FFN is missing MXM-local dequant marker: {marker}"
                )
        if 'lhs_kind = "stream_i8"' in command_text:
            raise RuntimeError(
                "Vector FFN must not use VXM for weight dequantization"
            )
    if block8_selected:
        for marker in (
            'weight_input_mode = "int8_dequant_bf16"',
            'compute_mode = "block8"',
            "ftlpu.command.mxm_dequant",
            'kind = "w8a16_block8_weight_wave_striped"',
        ):
            if marker not in command_text:
                raise RuntimeError(
                    f"Block8 FFN is missing Command IR marker: {marker}"
                )
        binding_lines = [
            line for line in command_text.splitlines()
            if "ftlpu.command.binding" in line
        ]
        output_binding = next(
            (line for line in binding_lines
             if 'access = "output"' in line
             and 'kind = "fp16_mxm_block8_distributed_16"' in line),
            None,
        )
        if not output_binding:
            raise RuntimeError(
                "Block8 FFN output must use distributed16 placement"
            )
        shape_match = re.search(
            r'shape = \[(\d+), (\d+)\]', output_binding
        )
        if not shape_match:
            raise RuntimeError("Block8 FFN output binding has no shape")
        base_match = re.search(r'base_row = (\d+) : i64', output_binding)
        if not base_match:
            raise RuntimeError("Block8 FFN output binding has no base row")
        sequence_length = int(shape_match.group(1))
        expected_hidden_rows = (sequence_length // 32) * (1536 // 8)
        if int(base_match.group(1)) < expected_hidden_rows:
            raise RuntimeError(
                "Block8 FFN output overlaps the shape-dependent hidden region"
            )
        activation_binding = next(
            (line for line in binding_lines if 'name = "activation"' in line),
            None,
        )
        result_binding = next(
            (line for line in binding_lines if 'access = "output"' in line),
            None,
        )
        gate_binding = next(
            (line for line in binding_lines if 'index = 1 : i64' in line),
            None,
        )
        up_binding = next(
            (line for line in binding_lines if 'index = 2 : i64' in line),
            None,
        )
        if (not activation_binding or not gate_binding or not up_binding
                or not result_binding):
            raise RuntimeError("Block8 FFN is missing physical bindings")

        def binding_slices(line: str) -> set[int]:
            match = re.search(r"slices = \[([^]]+)\]", line)
            if not match:
                raise RuntimeError("Block8 FFN binding has no physical slices")
            return {int(value.strip()) for value in match.group(1).split(",")}

        activation_slices = binding_slices(activation_binding)
        gate_slices = binding_slices(gate_binding)
        up_slices = binding_slices(up_binding)
        result_slices = binding_slices(result_binding)
        if len(activation_slices) != 16 or len(result_slices) != 16:
            raise RuntimeError("Block8 FFN bindings must use 16 physical slices")
        if activation_slices & result_slices:
            raise RuntimeError(
                "Block8 FFN input and hidden/result lifetimes must not overlap"
            )
        if (activation_slices & gate_slices
                or activation_slices & up_slices
                or gate_slices & up_slices):
            raise RuntimeError(
                "Gate, Up, and activation must use independent MEM slices"
            )

        compute_lines = [
            line for line in command_text.splitlines()
            if 'opcode = "compute"' in line
            and 'compute_mode = "block8"' in line
        ]
        if not compute_lines:
            raise RuntimeError("Block8 FFN has no Block8 MXM compute commands")
        first_cycle = re.search(r"cycle = (\d+) : i64", compute_lines[0])
        if not first_cycle:
            raise RuntimeError("cannot read first Block8 FFN compute cycle")
        first_cycle_lines = [
            line for line in compute_lines
            if f"cycle = {first_cycle.group(1)} : i64" in line
        ]
        first_queues = {
            int(re.search(r"queue = (\d+) : i64", line).group(1))
            for line in first_cycle_lines
        }
        mxms_per_hemisphere_match = re.search(
            r"mxms_per_hemisphere = (\d+) : i64", command_text
        )
        if not mxms_per_hemisphere_match:
            raise RuntimeError("cannot read MXM topology from Command IR")
        mxms_per_hemisphere = int(mxms_per_hemisphere_match.group(1))
        mxm_queue_count = 2 * mxms_per_hemisphere
        if first_queues != set(range(mxm_queue_count)):
            raise RuntimeError(
                "Gate/Up must multicast one Block8 activation to every MXM"
            )
        for line in first_cycle_lines:
            if ('activation_stream_base = 16 : i64' not in line
                    or 'repeat_count = 4 : i64' not in line):
                raise RuntimeError(
                    "Gate/Up Block8 compute overlapping IW must use streams 16..31"
                )
        activation_stream_bases = {
            int(re.search(
                r"activation_stream_base = (\d+) : i64", line
            ).group(1))
            for line in compute_lines
        }
        if not activation_stream_bases.issubset({0, 16}):
            raise RuntimeError(
                "Gate/Up Block8 activation must use the low or high 16-stream window"
            )
        queue0_issues = sorted({
            (int(re.search(r"cycle = (\d+) : i64", line).group(1)),
             int(re.search(r"weight_buffer = (\d+) : i64", line).group(1)))
            for line in compute_lines
            if 'queue = 0 : i64' in line
        })
        first_cycle = queue0_issues[0][0] if queue0_issues else -1
        expected_first_pair = [
            (first_cycle + 4 * issue, issue % 2)
            for issue in range(8)
        ]
        if (len(queue0_issues) < 9
                or queue0_issues[:8] != expected_first_pair
                or queue0_issues[8][0] != first_cycle + 32):
            raise RuntimeError(
                "Gate/Up must sustain the 8-cycle two-buffer wavefront schedule"
            )
        vxm_lines = [
            line for line in command_text.splitlines()
            if "ftlpu.command.vxm" in line
            and 'opcode = "negate"' in line
        ]
        if not vxm_lines:
            raise RuntimeError("Block8 FFN has no SwiGLU commands")
        swiglu_cycles = [
            int(re.search(r"cycle = (\d+) : i64", line).group(1))
            for line in vxm_lines
        ]
        accumulator_reads = [
            line for line in command_text.splitlines()
            if "ftlpu.command.mxm" in line
            and 'opcode = "accumulator_read"' in line
        ]
        wavefront_uses_all_streams = (
            'mxm_block_group_interval = 8 : i64' in command_text
            and 'mxm_weight_activation_overlap_enabled = 1 : i64'
                in command_text
        )
        if args.ffn_schedule == "tail":
            first_swiglu_cycle = min(swiglu_cycles)
            projection_drains = [
                line for line in accumulator_reads
                if int(re.search(r"cycle = (\d+) : i64", line).group(1))
                    < first_swiglu_cycle
            ]
            if projection_drains:
                raise RuntimeError(
                    "Block8 tail must stream the final projection partial "
                    "without separate accumulator reads"
                )
            final_stream_computes = [
                line for line in compute_lines
                if 'accumulator_destination = "stream"' in line
                and 'accumulator_clear = true' in line
                and int(re.search(r"cycle = (\d+) : i64", line).group(1))
                    < first_swiglu_cycle
            ]
            # Count ICU compute commands across the compressed outer pair
            # wave. The inner repeat is the four-block execution width of one
            # command and is already represented in the expected formula.
            expected_stream_computes = (
                24 * 4 * mxm_queue_count * 2
                // mxms_per_hemisphere
            )
            dynamic_stream_computes = 0
            for line in final_stream_computes:
                group = re.search(r"group_count = (\d+) : i64", line)
                dynamic_stream_computes += (
                    int(group.group(1)) if group else 1)
            if dynamic_stream_computes != expected_stream_computes:
                raise RuntimeError(
                    "Block8 tail must stream+clear every final Gate/Up "
                    f"partial: expected {expected_stream_computes}, got "
                    f"{dynamic_stream_computes}"
                )
            output_bases = {
                int(re.search(r"output_stream_base = (\d+) : i64", line)
                    .group(1))
                for line in final_stream_computes
            }
            expected_output_bases = {
                local * 16 for local in range(mxms_per_hemisphere)
            }
            if output_bases != expected_output_bases:
                raise RuntimeError(
                    "Gate/Up direct BF16 outputs must occupy west stream "
                    f"windows {sorted(expected_output_bases)}, got "
                    f"{sorted(output_bases)}"
                )
        else:
            first_swiglu_cycle = min(swiglu_cycles)
            last_swiglu_cycle = max(swiglu_cycles)
            projection_compute_cycles = [
                int(re.search(r"cycle = (\d+) : i64", line).group(1))
                for line in compute_lines
            ]
            if not any(first_swiglu_cycle < cycle < last_swiglu_cycle
                       for cycle in projection_compute_cycles):
                raise RuntimeError(
                    "Block8 fused schedule does not overlap SwiGLU with "
                    "later Gate/Up projection compute"
                )
            swiglu_cast_cycles = {cycle + 5 for cycle in swiglu_cycles}
            swiglu_output_streams = {
                int(re.search(r"output_stream = (\d+) : i64", line).group(1))
                for line in command_text.splitlines()
                if "ftlpu.command.vxm" in line
                and 'opcode = "cast"' in line
                and int(re.search(r"cycle = (\d+) : i64", line).group(1))
                    in swiglu_cast_cycles
            }
            expected_fused_output_base = (
                8 if mxms_per_hemisphere == 1 else 24
            )
            if (not swiglu_output_streams
                    or swiglu_output_streams
                        != {expected_fused_output_base}):
                raise RuntimeError(
                    "Block8 fused SwiGLU output overlaps the projection "
                    f"stream range: expected E{expected_fused_output_base}, "
                    f"got {sorted(swiglu_output_streams)}"
                )
        if accumulator_reads:
            raise RuntimeError(
                "Block8 FFN must direct-stream both projection and down "
                "results without accumulator_read commands"
            )
        last_swiglu_cycle = max(swiglu_cycles)
        down_stream_computes = [
            line for line in compute_lines
            if 'accumulator_destination = "stream"' in line
            and 'accumulator_clear = true' in line
            and int(re.search(r"cycle = (\d+) : i64", line).group(1))
                > last_swiglu_cycle
        ]
        # 18 output blocks x 4 token blocks. Each command repeats over
        # the four 8-row token blocks produced by one Block8 issue.
        expected_down_stream_computes = 18 * 4
        logical_down_stream_computes = sum(
            int(match.group(1)) if (match := re.search(
                r"group_count = (\d+) : i64", line)) else 1
            for line in down_stream_computes
        )
        if logical_down_stream_computes != expected_down_stream_computes:
            raise RuntimeError(
                "Block8 down must stream+clear every final partial: "
                f"expected {expected_down_stream_computes}, got "
                f"{logical_down_stream_computes}"
            )
        down_output_bases = {
            int(re.search(r"output_stream_base = (\d+) : i64", line)
                .group(1))
            for line in down_stream_computes
        }
        expected_down_output_bases = {
            local * 16 for local in range(mxms_per_hemisphere)
        }
        if down_output_bases != expected_down_output_bases:
            raise RuntimeError(
                "Down direct BF16 outputs must occupy west stream "
                f"windows {sorted(expected_down_output_bases)}, got "
                f"{sorted(down_output_bases)}"
            )

        def compute_interval(line: str) -> tuple[int, int]:
            cycle = int(re.search(
                r"cycle = (\d+) : i64", line).group(1))
            count = int(re.search(
                r"repeat_count = (\d+) : i64", line).group(1))
            interval = int(re.search(
                r"repeat_interval = (\d+) : i64", line).group(1))
            return cycle, cycle + (count - 1) * interval + 1

        for queue in range(mxm_queue_count):
            queue_marker = f"queue = {queue} : i64"
            projection_intervals = sorted(
                compute_interval(line) for line in compute_lines
                if queue_marker in line
                and compute_interval(line)[0] < first_swiglu_cycle
            )
            projection_gaps = [
                current[0] - previous[1]
                for previous, current in zip(
                    projection_intervals, projection_intervals[1:])
                if current[0] > previous[1]
            ]
            if projection_gaps:
                raise RuntimeError(
                    "Gate/Up Block8 compute must remain continuous across "
                    f"output pairs; queue {queue} gaps={projection_gaps[:8]}"
                )

            down_intervals = sorted(
                compute_interval(line) for line in compute_lines
                if queue_marker in line
                and compute_interval(line)[0] > last_swiglu_cycle
            )
            down_gaps = [
                current[0] - previous[1]
                for previous, current in zip(
                    down_intervals, down_intervals[1:])
                if current[0] > previous[1]
            ]
            columns_per_wave = 2 * mxms_per_hemisphere * 32
            down_wave_count = (576 + columns_per_wave - 1) // columns_per_wave
            # Only the unpaired final reduction can leave one four-cycle hole
            # per token block and output wave. All ordinary reduction pairs
            # remain fully interleaved.
            if (len(down_gaps) > down_wave_count * 4
                    or any(gap != 4 for gap in down_gaps)):
                raise RuntimeError(
                    "Down Block8 reduction-pair wavefront contains an "
                    f"unexpected compute bubble on queue {queue}: "
                    f"{down_gaps[:12]}"
                )
    subprocess.run([
        str(args.translate), "--input", str(commands), "--output", str(binary),
    ], check=True)
    environment = os.environ.copy()
    environment["FTLPU_SCHEDULE_TRACE"] = str(trace)
    runtime_result = subprocess.run([str(args.runtime_test), str(binary)], env=environment)
    runtime_result.check_returncode()
    if block8_selected:
        return
    renderer = Path(__file__).resolve().parents[1] / "tools" / "render_ffn_pipeline.py"
    source = args.input.read_text(encoding="utf-8")
    sequence_match = re.search(r"tensor<(\d+)x576xbf16>", source)
    if not sequence_match:
        raise RuntimeError("cannot determine FFN sequence length from StableHLO input")
    subprocess.run([
        sys.executable, str(renderer), str(trace), str(pipeline),
        "--sequence-length", sequence_match.group(1),
    ], check=True)


if __name__ == "__main__":
    main()
