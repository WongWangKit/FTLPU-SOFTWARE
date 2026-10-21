#!/usr/bin/env python3
"""Exercise raw Macro through the production Schedule/Command CLI entries.

The fixture deliberately keeps three independent affine launch dimensions,
weight-buffer toggling, terminal accumulator behavior and a blocked MEM outer
address domain.  Native VXM/SXM queues must survive unchanged alongside the
MEM/MXM Macro queues.  No offline conversion or experimental artifacts are used.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess


MACRO_RESOURCES = {"mem", "mxm_load", "mxm_compute", "mxm_dequant"}
NATIVE_RESOURCES = {"vxm", "sxm_permute"}


def run(command: list[str | Path], *, failure: str | None = None) -> str:
    result = subprocess.run(
        [str(item) for item in command], capture_output=True, text=True,
        encoding="utf-8", errors="replace", check=False,
    )
    output = result.stdout + result.stderr
    if failure is not None:
        if result.returncode == 0 or failure not in output:
            raise AssertionError(
                f"expected failure containing {failure!r}:\n"
                f"{' '.join(map(str, command))}\n{output}"
            )
    elif result.returncode:
        raise AssertionError(
            f"command failed ({result.returncode}):\n"
            f"{' '.join(map(str, command))}\n{output}"
        )
    return output


def write_schedule(path: Path) -> None:
    lanes = ", ".join(f"{lane} : i64" for lane in range(32))
    streams = ", ".join(f"{lane} : i64" for lane in range(16))
    path.write_text(
        """module {
  func.func @direct_macro(%arg0: tensor<1xbf16>) {
    ftlpu.schedule.mem_transfer {
      cycle = 100 : i64, hemisphere = 0 : i64, slice = 0 : i64,
      bank = 0 : i64, opcode = "read", address = 100 : i64,
      packed_stream = 0 : i64, repeat_count = 2 : i64,
      repeat_interval = 2 : i64, address_stride = 1 : i64,
      wave_count = 3 : i64, wave_interval = 16 : i64,
      wave_address_stride = 8 : i64, group_count = 2 : i64,
      group_interval = 128 : i64, group_address_stride = 32 : i64
    }
    ftlpu.schedule.mem_transfer {
      cycle = 100 : i64, hemisphere = 0 : i64, slice = 1 : i64,
      bank = 1 : i64, opcode = "write", address = 200 : i64,
      packed_stream = 2 : i64, repeat_count = 2 : i64,
      repeat_interval = 2 : i64, address_stride = 1 : i64,
      wave_count = 2 : i64, wave_interval = 16 : i64,
      wave_address_stride = 8 : i64, group_count = 3 : i64,
      group_interval = 128 : i64, group_address_stride = 0 : i64,
      outer_group_size = 2 : i64, outer_group_stride = 32 : i64,
      outer_inner_stride = 4 : i64
    }
    ftlpu.schedule.mxm_issue {
      cycle = 200 : i64, unit_id = 0 : i64, opcode = "iw",
      weight_buffer = 0 : i64, weight_column = 0 : i64,
      activation_stream_base = 0 : i64, output_stream_base = 0 : i64,
      repeat_count = 2 : i64, repeat_interval = 2 : i64,
      repeat_weight_column_stride = 1 : i64,
      wave_count = 2 : i64, wave_interval = 20 : i64,
      wave_weight_column_stride = 0 : i64,
      group_count = 2 : i64, group_interval = 200 : i64,
      group_weight_column_stride = 0 : i64,
      accumulator_address = 0 : i64, accumulator_row_stride = 1 : i64,
      accumulator_destination = "sram", accumulator_clear = false,
      data_format = "bf16", weight_buffer_mode = "toggle_dim1"
    }
    ftlpu.schedule.mxm_issue {
      cycle = 300 : i64, unit_id = 0 : i64, opcode = "compute",
      weight_buffer = 0 : i64, weight_column = 0 : i64,
      activation_stream_base = 0 : i64, output_stream_base = 0 : i64,
      repeat_count = 4 : i64, repeat_interval = 1 : i64,
      wave_count = 2 : i64, wave_interval = 30 : i64,
      wave_accumulator_address_stride = 4 : i64,
      group_count = 2 : i64, group_interval = 200 : i64,
      group_accumulator_address_stride = 8 : i64,
      accumulator_address = 0 : i64, accumulator_row_stride = 1 : i64,
      accumulator_destination = "sram", accumulator_clear = false,
      data_format = "bf16", weight_buffer_mode = "toggle_dim2",
      terminal_dimension = 1 : i64,
      terminal_accumulator_destination = "stream",
      terminal_accumulator_clear = true,
      terminal_accumulator_output_format = "bf16"
    }
    ftlpu.schedule.mxm_dequant {
      cycle = 100 : i64, unit_id = 0 : i64, scale = 0.5 : f32,
      repeat_count = 2 : i64, repeat_interval = 2 : i64,
      wave_count = 2 : i64, wave_interval = 20 : i64,
      group_count = 2 : i64, group_interval = 200 : i64
    }
    %0 = ftlpu.schedule.vxm %arg0, %arg0 {
      cycle = 600 : i64, queue = 0 : i64, chain_depth = 2 : i64,
      opcode = "multiply", lhs_kind = "stream_bf16", lhs_index = 0 : i64,
      lhs_immediate = 0.0 : f32, rhs_kind = "immediate", rhs_index = 0 : i64,
      rhs_immediate = 0.5 : f32, cast_target = "bf16",
      output_stream = 0 : i64, repeat_count = 2 : i64,
      repeat_interval = 4 : i64, wave_count = 2 : i64,
      wave_interval = 20 : i64,
      input_hemisphere = "east", output_hemisphere = "east"
    } : (tensor<1xbf16>, tensor<1xbf16>) -> tensor<1xbf16>
    ftlpu.schedule.sxm {
      cycle = 700 : i64, hemisphere = 0 : i64, opcode = "permute",
      source_streams = [STREAMS],
      destination_streams = [STREAMS], permute_map = [LANES],
      weight_layout = "native", repeat_count = 2 : i64,
      repeat_interval = 4 : i64, wave_count = 2 : i64,
      wave_interval = 20 : i64
    }
    return
  }
}
""".replace("LANES", lanes).replace("STREAMS", streams), encoding="utf-8",
    )


def require_verification(output: str) -> None:
    if "verified ICU logical issue equivalence:" not in output:
        raise AssertionError(f"CLI did not run ICU issue verification:\n{output}")


def check_binary(args: argparse.Namespace, binary: Path, baseline: Path,
                 *, macro: bool, require_all_resources: bool = True) -> str:
    output = run([args.inspect, binary, "--all-queues", "--compare", baseline])
    binary.with_suffix(".inspect.txt").write_text(output, encoding="utf-8")
    if "binary compare result=equivalent" not in output:
        raise AssertionError(f"logical issues changed for {binary}:\n{output}")
    resources: set[str] = set()
    for line in output.splitlines():
        if not line.startswith("imem queue resource="):
            continue
        fields = dict(re.findall(r"(\w+)=([^ ]+)", line))
        resource = fields["resource"]
        resources.add(resource)
        macros, coarse = int(fields["macro"]), int(fields["coarse_program"])
        if macro and resource in MACRO_RESOURCES:
            if macros == 0 or coarse != 0:
                raise AssertionError(f"MEM/MXM silently retained FU3D: {line}")
        elif macros != 0 or coarse == 0:
            raise AssertionError(f"native FU queue changed encoding: {line}")
    if require_all_resources and not (MACRO_RESOURCES | NATIVE_RESOURCES) <= resources:
        raise AssertionError(f"fixture did not cover every queue kind: {resources}")
    if "mem" not in resources or not any(item.startswith("mxm_") for item in resources):
        raise AssertionError(f"fixture did not cover MEM and MXM: {resources}")
    return output


def check_relocation_clones(args: argparse.Namespace, default_command: Path) -> None:
    # Each binding describes real, nonpaged SRAM storage. The first MEM
    # command's 100..149 rows lie in input 0; input 1 owns the dequant scale.
    # Both source domains have three independent axes and therefore require
    # multiple two-axis Macro descriptors carrying the same relocation owner.
    bindings = """    ftlpu.command.binding {
      index = 0 : i64, access = "input", role = "activation", name = "activation",
      ready_cycle = 0 : i64, initializer = "none", initializer_config = {},
      shape = [4096 : i64], element_type = "bf16", bytes = 8192 : i64,
      placement = {kind = "vector", hemisphere = "east", slices = [0 : i64],
        base_row = 100 : i64, instruction_count = 64 : i64,
        address_stride = 1 : i64, bank = 0 : i64}
    }
    ftlpu.command.binding {
      index = 1 : i64, access = "input", role = "weight", name = "weight",
      ready_cycle = 0 : i64, initializer = "none", initializer_config = {},
      shape = [1 : i64], element_type = "i8", bytes = 1 : i64,
      placement = {kind = "vector", hemisphere = "east", slices = [20 : i64],
        base_row = 300 : i64, instruction_count = 1 : i64,
        address_stride = 1 : i64, bank = 0 : i64}
    }
"""
    text, count = re.subn(r"(func\.func [^\n]*\{\n)",
                          lambda match: match[0] + bindings,
                          default_command.read_text(encoding="utf-8"), count=1)
    if count != 1:
        raise AssertionError("cannot locate the direct Command function")
    for operation, attribute in (
        ("mem_3d", 'address_binding = 0 : i64, address_binding_access = "input", '),
        ("mxm_dequant_3d", "scale_binding = 1 : i64, "),
    ):
        text, count = re.subn(rf"(ftlpu\.command\.{operation} \{{)",
                              lambda match: match[0] + attribute, text, count=1)
        if count != 1:
            raise AssertionError(f"cannot locate relocation owner {operation}")
    command = args.output_dir / "relocations.command.mlir"
    command.write_text(text, encoding="utf-8")
    baseline = args.output_dir / "relocations-default.ftlpu"
    macro = args.output_dir / "relocations-raw-macro.ftlpu"
    for binary, mode in ((baseline, "none"), (macro, "raw-macro")):
        require_verification(run([args.translate, "--input", command,
                                  "--output", binary, "--icu-compression", mode,
                                  "--verify-icu-issues"]))
    baseline_output = check_binary(args, baseline, baseline, macro=False)
    macro_output = check_binary(args, macro, baseline, macro=True)
    for pattern, kind in (
        (r"binary address_relocation access=0 binding=0 count=(\d+)", "mem"),
        (r"binary scale_relocation binding=1 resource=mxm_dequant count=(\d+)",
         "mxm_dequant"),
    ):
        source = re.search(pattern, baseline_output)
        converted = re.search(pattern, macro_output)
        queue = re.search(rf"^imem queue resource={kind} queue=0 .*\bmacro=(\d+)\b",
                          macro_output, re.MULTILINE)
        if (not source or int(source[1]) != 1 or not converted or not queue
                or int(converted[1]) <= 1 or int(converted[1]) != int(queue[1])):
            raise AssertionError(f"{kind} relocation did not clone to every descriptor")
    if args.runtime_test:
        run([args.runtime_test, baseline, macro])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("opt", "compile", "translate", "inspect", "target-config",
                   "output-dir"):
        parser.add_argument(f"--{option}", type=Path, required=True)
    parser.add_argument("--runtime-test", type=Path,
                        help="Replay and compare two binaries through real ICU queues")
    parser.add_argument("--numeric-test", type=Path,
                        help="Run the compiled 32x64x128x64 BF16 FFN numerical test")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    schedule = args.output_dir / "direct.schedule.mlir"
    write_schedule(schedule)
    target = ["--target-config", args.target_config]
    baseline = args.output_dir / "compile-default.ftlpu"
    common = [args.compile, "--input", schedule, "--input-stage", "schedule",
              *target, "--verify-icu-issues"]
    require_verification(run(common + ["--output", baseline]))
    check_binary(args, baseline, baseline, macro=False)

    macro_binary = args.output_dir / "compile-raw-macro.ftlpu"
    require_verification(run(common + ["--output", macro_binary,
                                      "--icu-compression", "raw-macro"]))
    if baseline.read_bytes() == macro_binary.read_bytes():
        raise AssertionError("raw-macro CLI is still producing the default FU3D binary")
    check_binary(args, macro_binary, baseline, macro=True)
    if args.runtime_test:
        run([args.runtime_test, baseline, macro_binary])

    # ftlpu-opt must preserve the requested mode in Command IR. Translation
    # without any mode override must honor it. The final translator and the
    # compiler's direct Command entry must also honor an explicit override.
    default_command = args.output_dir / "default.command.mlir"
    for name, mode in (("default", None), ("raw-macro", "raw-macro")):
        command = args.output_dir / f"{name}.command.mlir"
        opt = [args.opt, "--input", schedule, "--output", command,
               "--pipeline", "ftlpu-schedule-to-commands", *target]
        if mode:
            opt.extend(["--icu-compression", mode])
        run(opt)
        text = command.read_text(encoding="utf-8")
        for operation in ("mem_3d", "mxm_load_3d", "mxm_compute_3d",
                          "mxm_dequant_3d", "vxm_run_2d", "sxm_run_2d"):
            if f"ftlpu.command.{operation}" not in text:
                raise AssertionError(f"direct lowering lost {operation}")
        if mode and 'ftlpu.icu_compression = "raw-macro"' not in text:
            raise AssertionError("ftlpu-opt discarded raw-macro mode")
        binary = args.output_dir / f"opt-{name}.ftlpu"
        require_verification(run([args.translate, "--input", command,
                                  "--output", binary, "--verify-icu-issues"]))
        check_binary(args, binary, baseline, macro=bool(mode))

    for tool, stage in ((args.translate, []),
                        (args.compile, ["--input-stage", "command", *target])):
        name = "translate" if tool == args.translate else "compile-command"
        binary = args.output_dir / f"{name}-raw-macro.ftlpu"
        require_verification(run([tool, "--input", default_command,
                                  "--output", binary, *stage,
                                  "--icu-compression", "raw-macro",
                                  "--verify-icu-issues"]))
        check_binary(args, binary, baseline, macro=True)

    raw_command = args.output_dir / "raw-macro.command.mlir"
    for name, override, macro in (("inherited", [], True),
                                  ("explicit-none", ["--icu-compression", "none"], False)):
        binary = args.output_dir / f"compile-command-{name}.ftlpu"
        require_verification(run([args.compile, "--input", raw_command,
                                  "--input-stage", "command", *target,
                                  "--output", binary, *override,
                                  "--verify-icu-issues"]))
        check_binary(args, binary, baseline, macro=macro)

    check_relocation_clones(args, default_command)

    # A production StableHLO entry covers all compiler stages using the
    # existing small fused SwiGLU FFN and its real CModel arithmetic harness.
    ffn = (Path(__file__).resolve().parents[1] / "examples" / "w8a16_ffn" /
           "ffn_32x64x128x64_bf16.stablehlo.mlir")
    ffn_baseline = args.output_dir / "stablehlo-ffn-default.ftlpu"
    ffn_macro = args.output_dir / "stablehlo-ffn-raw-macro.ftlpu"
    for binary, mode in ((ffn_baseline, []),
                         (ffn_macro, ["--icu-compression", "raw-macro"])):
        require_verification(run([args.compile, "--input", ffn,
                                  "--input-stage", "stablehlo", *target,
                                  "--mxm-execution", "vector", "--ffn-schedule", "fused",
                                  "--output", binary, *mode, "--verify-icu-issues"]))
        if args.numeric_test:
            execution = run([args.numeric_test, binary])
            binary.with_suffix(".numeric.txt").write_text(execution, encoding="utf-8")
    if ffn_baseline.read_bytes() == ffn_macro.read_bytes():
        raise AssertionError("StableHLO raw-macro entry still produced the default binary")
    check_binary(args, ffn_baseline, ffn_baseline, macro=False,
                 require_all_resources=False)
    check_binary(args, ffn_macro, ffn_baseline, macro=True,
                 require_all_resources=False)
    if args.runtime_test:
        run([args.runtime_test, ffn_baseline, ffn_macro])

    # A real paged decoder retains Macro computation plus compiler paging
    # candidates. The runtime final link owns the concrete common barriers
    # and independently loaded MEM transport controls.
    decoder = (Path(__file__).resolve().parents[1] / "examples" /
               "qwen2_5_1_5b_decoder_layer" / "decoder_layer_seq32.stablehlo.mlir")
    paged = args.output_dir / "paged-decoder-raw-macro.ftlpu"
    run([args.compile, "--input", decoder, "--input-stage", "stablehlo",
         "--output", paged, *target, "--weight-bank", "1",
         "--kv-cache-capacity", "256", "--mxm-execution", "vector",
         "--projection-rope-overlap", "off", "--icu-compression", "raw-macro"])
    check_binary(args, paged, paged, macro=True, require_all_resources=False)
    print("direct raw Macro CLI entries, inherited mode/override, relocation clones, "
          "StableHLO FFN, native queue preservation, issue equivalence and paged "
          "decoder candidate emission passed")


if __name__ == "__main__":
    main()
