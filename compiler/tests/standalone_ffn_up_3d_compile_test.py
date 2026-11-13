#!/usr/bin/env python3
"""Exercise the production CLI path for direct affine FFN Up Macro lowering."""

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
    physical = re.search(r"imem model=target-physical-v1[^\n]*used_slots=(\d+)", inspected)
    if not summary or not physical or int(physical[1]) == 0:
        raise AssertionError("missing physical packed Macro image accounting")
    loaded = run([args.runtime_test, str(binary)]).stdout
    if "ICU queues loaded" not in loaded:
        raise AssertionError("packed Macro binary did not load into CModel: " + loaded)
    for line in inspected.splitlines():
        if line.startswith("imem queue resource="):
            fields = dict(re.findall(r"(\w+)=([^ ]+)", line))
            if int(fields["macro"]) == 0 or int(fields["coarse_program"]) != 0:
                raise AssertionError("non-Macro computation in direct projection: " + line)

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

    verified = run([args.compile, "--input", args.input, "--output",
                    str(output_dir / "verified.ftlpu"), "--input-stage", "stream",
                    "--direct-ffn-up-3d", "--target-config", args.target_config,
                    "--verify-icu-issues"])
    if "verified ICU logical issue equivalence:" not in verified.stdout:
        raise AssertionError("direct Macro issue verification did not execute")

    print(
        "standalone_ffn_up_3d_compile_test passed: "
        f"mxms_per_hemisphere={expected_mxms} bindings=0 "
        f"packed_words={physical[1]} Macro-only"
    )


if __name__ == "__main__":
    main()
