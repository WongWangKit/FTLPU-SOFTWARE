#!/usr/bin/env python3
"""Exercise the production CLI path for direct FFN Up FU 3-D lowering."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command, check=True, text=True, capture_output=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile", required=True)
    parser.add_argument("--inspect", required=True)
    parser.add_argument("--runtime-test", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--target-config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    binary = output_dir / "qwen2_5_up_direct_3d.ftlpu"
    run([
        args.compile,
        "--input", args.input,
        "--output", str(binary),
        "--input-stage", "stream",
        "--direct-ffn-up-3d",
        "--target-config", args.target_config,
    ])
    inspected = run([args.inspect, str(binary), "--all-queues"]).stdout

    target = json.loads(
        pathlib.Path(args.target_config).read_text(encoding="utf-8")
    )
    expected_mxms = int(target["throughput"]["mxms_per_hemisphere"])
    topology = re.search(r"mxms_per_hemisphere=(\d+)", inspected)
    if not topology or int(topology.group(1)) != expected_mxms:
        raise AssertionError(
            "direct binary did not preserve the target's logical MXM topology\n"
            + inspected
        )

    if "binary binding index=" in inspected:
        raise AssertionError(
            "standalone direct mode must leave host bindings empty; its contract is preloaded SRAM\n"
            + inspected)

    summary = re.search(r"binary queues=(\d+) commands=(\d+)", inspected)
    if not summary or int(summary.group(2)) != 418:
        raise AssertionError(
            "direct CLI path did not emit 312 packet words plus 106 timing NOPs\n"
            + inspected)
    physical = re.search(
        r"imem model=target-physical-v1[^\n]*used_slots=(\d+)",
        inspected,
    )
    if not physical or int(physical.group(1)) != 418:
        raise AssertionError(
            "physical i-MEM accounting did not include timing NOPs\n"
            + inspected)

    loaded = run([args.runtime_test, str(binary)]).stdout
    if "ICU queues loaded" not in loaded:
        raise AssertionError(
            "direct binary did not load into the target-aware CModel ICU\n"
            + loaded
        )
    coarse = sum(
        int(value)
        for value in re.findall(r"imem queue resource=.*?coarse_program=(\d+)", inspected)
    )
    if coarse != 106:
        raise AssertionError(
            f"expected 106 direct coarse commands, found {coarse}\n{inspected}")

    # MXM queues stay dense in the executable's logical topology: east is 0
    # and west is 1 for the one-MXM lpu32 target. The runtime smoke above then
    # proves those logical queues can be mapped onto physical CModel queues
    # 0 and 2 without accepting an out-of-topology binary queue.
    mxm_queues: dict[str, set[int]] = {}
    for resource, queue in re.findall(
            r"imem queue resource=(mxm_(?:load|compute|dequant)) "
            r"queue=(\d+)", inspected):
        mxm_queues.setdefault(resource, set()).add(int(queue))
    expected_logical_queues = {0, expected_mxms}
    for resource in ("mxm_load", "mxm_compute", "mxm_dequant"):
        if mxm_queues.get(resource) != expected_logical_queues:
            raise AssertionError(
                f"{resource} queues are not dense east/west logical ids "
                f"{sorted(expected_logical_queues)}\n{inspected}")

    aggregate = re.search(r"binary aggregate ([^\n]+)", inspected)
    if not aggregate:
        raise AssertionError("binary inspector omitted the aggregate command summary")
    legacy_fields = (
        "instruction", "repeat", "repeat2d", "macro",
        "mem_stream_nd", "mem_slice_program", "mxm_stream_nd",
        "vxm_stream_nd", "sxm_tile_program",
    )
    for field in legacy_fields:
        match = re.search(rf"(?:^| ){field}=(\d+)(?: |$)", aggregate.group(1))
        if not match or int(match.group(1)) != 0:
            raise AssertionError(
                f"direct raw 3-D binary unexpectedly contains legacy {field} commands\n"
                + inspected)
    nop = re.search(r"(?:^| )nop=(\d+)(?: |$)", aggregate.group(1))
    if not nop or int(nop.group(1)) != 106:
        raise AssertionError(
            "direct raw 3-D binary did not carry one timing NOP per nonzero queue start\n"
            + inspected)

    # The production mode must reject an ordinary/full graph instead of
    # silently treating any matmul as the standalone Up contract.
    unmarked = output_dir / "unmarked.stream.mlir"
    source = pathlib.Path(args.input).read_text(encoding="utf-8")
    unmarked.write_text(
        source.replace('projection_kind = "up"', 'projection_kind = "gate"', 1),
        encoding="utf-8",
    )
    rejected = subprocess.run([
        args.compile,
        "--input", str(unmarked),
        "--output", str(output_dir / "unmarked.ftlpu"),
        "--input-stage", "stream",
        "--direct-ffn-up-3d",
        "--target-config", args.target_config,
    ], check=False, text=True, capture_output=True)
    if rejected.returncode == 0 or "projection_kind" not in rejected.stderr:
        raise AssertionError(
            "direct CLI did not clearly reject an unmarked projection\n"
            + rejected.stdout + rejected.stderr)

    # The lpu32 production target exposes local MXM 0 in each hemisphere.
    # A direct program must not address a second logical MXM merely because
    # the CModel binary was built with spare physical capacity.
    invalid_unit = output_dir / "invalid_unit.stream.mlir"
    invalid_unit_id = expected_mxms
    invalid_unit.write_text(
        source.replace("destination_unit_id = 0",
                       f"destination_unit_id = {invalid_unit_id}")
              .replace("unit_ids = [0]",
                       f"unit_ids = [{invalid_unit_id}]"),
        encoding="utf-8",
    )
    rejected_unit = subprocess.run([
        args.compile,
        "--input", str(invalid_unit),
        "--output", str(output_dir / "invalid_unit.ftlpu"),
        "--input-stage", "stream",
        "--direct-ffn-up-3d",
        "--target-config", args.target_config,
    ], check=False, text=True, capture_output=True)
    if (rejected_unit.returncode == 0
            or "one local MXM" not in rejected_unit.stderr):
        raise AssertionError(
            "direct CLI accepted an MXM outside the target's logical topology\n"
            + rejected_unit.stdout + rejected_unit.stderr)

    rejected_verify = subprocess.run([
        args.compile,
        "--input", args.input,
        "--output", str(output_dir / "invalid_verify.ftlpu"),
        "--input-stage", "stream",
        "--direct-ffn-up-3d",
        "--target-config", args.target_config,
        "--verify-icu-issues",
    ], check=False, text=True, capture_output=True)
    if (rejected_verify.returncode == 0
            or "cannot use --verify-icu-issues" not in rejected_verify.stderr):
        raise AssertionError(
            "direct CLI did not reject its inapplicable logical-issue self-check\n"
            + rejected_verify.stdout + rejected_verify.stderr)

    print(
        "standalone_ffn_up_3d_compile_test passed: "
        f"mxms_per_hemisphere={expected_mxms} bindings=0 "
        "coarse=106 packet_words=312 timing_nops=106 physical_words=418 legacy=0"
    )


if __name__ == "__main__":
    main()
