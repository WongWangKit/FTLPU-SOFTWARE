#!/usr/bin/env python3
"""Check direct cross-output-group MEM domains for Qwen2.5 O projection."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path


def attr(line: str, name: str, default: int | None = None) -> int:
    match = re.search(rf"\b{re.escape(name)} = (-?\d+) : i64", line)
    if match:
        return int(match.group(1))
    if default is not None:
        return default
    raise AssertionError(f"missing {name}: {line.strip()}")


def interval(line: str) -> tuple[int, int]:
    start = attr(line, "cycle")
    end = (start + (attr(line, "repeat_count") - 1)
           * attr(line, "repeat_interval")
           + (attr(line, "wave_count", 1) - 1)
           * attr(line, "wave_interval", 1)
           + (attr(line, "group_count", 1) - 1)
           * attr(line, "group_interval", 1) + 1)
    return start, end


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opt", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--target-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stream = args.output_dir / "decoder_layer.stream.mlir"
    schedule = args.output_dir / "decoder_layer.schedule.mlir"
    common = ["--target-config", str(args.target_config),
              "--weight-bank", "1", "--kv-cache-capacity", "256",
              "--mxm-execution", "vector", "--projection-rope-overlap", "off"]
    for source, destination, pipeline in (
        (args.input, stream, "ftlpu-stablehlo-to-stream"),
        (stream, schedule, "ftlpu-stream-to-schedule"),
    ):
        subprocess.run([str(args.opt), "--input", str(source), "--output",
                        str(destination), "--pipeline", pipeline, *common],
                       check=True)

    lines = schedule.read_text(encoding="utf-8").splitlines()
    timeline = next(line for line in lines
                    if "ftlpu.schedule.timeline" in line
                    and 'name = "o_proj"' in line)
    o_start, o_end = attr(timeline, "start"), attr(timeline, "end")
    pv_timeline = next(line for line in lines
                       if "ftlpu.schedule.timeline" in line
                       and 'name = "pv"' in line)
    pv_start, pv_end = attr(pv_timeline, "start"), attr(pv_timeline, "end")
    qk_timeline = next(line for line in lines
                       if "ftlpu.schedule.timeline" in line
                       and 'name = "qk"' in line)
    qk_start, qk_end = attr(qk_timeline, "start"), attr(qk_timeline, "end")
    qkv_timeline = next(line for line in lines
                        if "ftlpu.schedule.timeline" in line
                        and 'name = "qkv"' in line)
    qkv_start, qkv_end = attr(qkv_timeline, "start"), attr(qkv_timeline, "end")
    ffn_timeline = next(line for line in lines
                        if "ftlpu.schedule.timeline" in line
                        and 'name = "ffn.down.vector"' in line)
    ffn_start, ffn_end = attr(ffn_timeline, "start"), attr(ffn_timeline, "end")
    softmax_timeline = next(line for line in lines
                            if "ftlpu.schedule.timeline" in line
                            and 'name = "softmax"' in line)
    softmax_start = attr(softmax_timeline, "start")
    softmax_end = attr(softmax_timeline, "end")
    rms_timelines = [line for line in lines
                     if "ftlpu.schedule.timeline" in line
                     and 'name = "rmsnorm.feedback"' in line]
    transfers = [line for line in lines
                 if "ftlpu.schedule.mem_transfer" in line]

    for hemisphere in range(2):
        for slice_id in range(16, 20):
            domains = [line for line in transfers
                       if qkv_start <= attr(line, "cycle") < qkv_end
                       and attr(line, "hemisphere") == hemisphere
                       and attr(line, "slice") == slice_id
                       and attr(line, "bank") == 0
                       and 'opcode = "write"' in line
                       and attr(line, "group_count", 1) == 24]
            if len(domains) != 1 or (
                    attr(domains[0], "repeat_count"),
                    attr(domains[0], "wave_count", 1),
                    attr(domains[0], "group_interval", 1),
                    attr(domains[0], "group_address_stride", 0)
            ) != (4, 8, 160, 128):
                raise AssertionError(
                    f"Q RoPE scratch queue {(hemisphere, slice_id)} "
                    f"did not emit one 24-slot direct domain")

    for hemisphere in range(2):
        for bank in range(2):
            for slice_id in range(16):
                domains = [line for line in transfers
                           if qk_start <= attr(line, "cycle") < qk_end
                           and attr(line, "hemisphere") == hemisphere
                           and attr(line, "slice") == slice_id
                           and attr(line, "bank") == bank
                           and 'opcode = "read"' in line]
                # The score writer preempts bank-1 slices 12 and 13 between
                # waves. Those reads must remain separate ICU commands.
                if bank == 1 and slice_id in (12, 13):
                    valid = len(domains) == 6 and all(
                        (attr(line, "repeat_count"),
                         attr(line, "wave_count", 1),
                         attr(line, "group_count", 1)) == (4, 2, 1)
                        for line in domains)
                else:
                    valid = len(domains) == 1 and (
                        attr(domains[0], "repeat_count"),
                        attr(domains[0], "wave_count", 1),
                        attr(domains[0], "group_count", 1),
                        attr(domains[0], "group_interval", 1)
                    ) == (4, 2, 6, 128)
                if not valid:
                    raise AssertionError(
                        f"QK Query-IW queue "
                        f"{(hemisphere, slice_id, bank)} was split")
        for slice_id in range(16, 20):
            domains = [line for line in transfers
                       if qk_start <= attr(line, "cycle") < qk_end
                       and attr(line, "hemisphere") == hemisphere
                       and attr(line, "slice") == slice_id
                       and attr(line, "bank") == 1
                       and 'opcode = "read"' in line]
            if len(domains) != 1 or (
                    attr(domains[0], "repeat_count"),
                    attr(domains[0], "wave_count", 1),
                    attr(domains[0], "group_count", 1),
                    attr(domains[0], "group_interval", 1)
            ) != (32, 2, 6, 128):
                raise AssertionError(
                    f"QK K activation queue {(hemisphere, slice_id)} "
                    f"was split")

    for unit_id in range(2):
        qk_mxm = [line for line in lines
                  if "ftlpu.schedule.mxm_issue" in line
                  and qk_start <= attr(line, "cycle") < qk_end
                  and attr(line, "unit_id") == unit_id]
        loads = [line for line in qk_mxm if 'opcode = "iw"' in line]
        computes = [line for line in qk_mxm
                    if 'opcode = "compute"' in line]
        for kind, domains, count in (("load", loads, (4, 4, 6)),
                                     ("compute", computes, (32, 4, 6))):
            if len(domains) != 1 or (
                    attr(domains[0], "repeat_count"),
                    attr(domains[0], "wave_count", 1),
                    attr(domains[0], "group_count", 1)
            ) != count or 'weight_buffer_mode = "toggle_dim1"' not in domains[0]:
                raise AssertionError(
                    f"QK MXM {kind} queue {unit_id} was split")
        if attr(computes[0], "terminal_dimension") != 1:
            raise AssertionError(
                f"QK MXM terminal axis incorrect on {unit_id}")

    # PV has no other MEM traffic on its context destination queues. The
    # 24 result windows for each physical queue therefore form one direct
    # outer domain, including the local bridge-preserving WRITE_TAP case.
    for hemisphere in range(2):
        for slice_id in range(16, 20):
            domains = [line for line in transfers
                       if pv_start <= attr(line, "cycle") < pv_end
                       and attr(line, "hemisphere") == hemisphere
                       and attr(line, "slice") == slice_id
                       and attr(line, "bank") == 1
                       and ('opcode = "write"' in line
                            or 'opcode = "write_tap"' in line)]
            observed = [(attr(line, "repeat_count"),
                         attr(line, "group_count", 1),
                         attr(line, "group_interval", 1),
                         attr(line, "group_address_stride", 0))
                        for line in domains]
            if observed != [(32, 24, 32, 32)]:
                raise AssertionError(
                    f"PV context queue {(hemisphere, slice_id)} was "
                    f"not lowered to one direct MEM3D domain: {observed}")

    for hemisphere in range(2):
        domains = [line for line in transfers
                   if softmax_start <= attr(line, "cycle") < softmax_end
                   and attr(line, "hemisphere") == hemisphere
                   and attr(line, "slice") == 31
                   and attr(line, "bank") == 0
                   and 'opcode = "write"' in line]
        if len(domains) != 1 or (
                attr(domains[0], "repeat_count"),
                attr(domains[0], "group_count", 1),
                attr(domains[0], "group_interval", 1),
                attr(domains[0], "outer_group_size", 1),
                attr(domains[0], "outer_inner_stride", 0),
                attr(domains[0], "outer_group_stride", 0)
        ) != (4, 12, 236, 2, 24, 4):
            raise AssertionError(
                f"softmax probability-pack queue {hemisphere}:31 "
                f"was split")
        for slice_id in (12, 13):
            scalar_reads = [line for line in transfers
                            if softmax_start <= attr(line, "cycle") < softmax_end
                            and attr(line, "hemisphere") == hemisphere
                            and attr(line, "slice") == slice_id
                            and attr(line, "bank") == 1
                            and 'opcode = "read"' in line
                            and attr(line, "group_count", 1) == 2
                            and attr(line, "group_interval", 1) == 45]
            if len(scalar_reads) != 12 or any(
                    (attr(line, "repeat_count"),
                     attr(line, "group_interval")) != (32, 45)
                    for line in scalar_reads):
                raise AssertionError(
                    f"softmax scalar reads on "
                    f"{(hemisphere, slice_id)} were split")

    if len(rms_timelines) != 2:
        raise AssertionError("expected two RMS feedback stages")
    for rms_timeline in rms_timelines:
        rms_start, rms_end = (attr(rms_timeline, "start"),
                              attr(rms_timeline, "end"))
        for hemisphere in range(2):
            for slice_id in range(2, 16):
                domains = [line for line in transfers
                           if rms_start <= attr(line, "cycle") < rms_end
                           and attr(line, "hemisphere") == hemisphere
                           and attr(line, "slice") == slice_id
                           and attr(line, "bank") == 1
                           and 'opcode = "read"' in line
                           and attr(line, "repeat_count") == 192]
                if len(domains) != 1 or (
                        attr(domains[0], "group_count", 1),
                        attr(domains[0], "group_interval", 1)
                ) != (2, 1551):
                    raise AssertionError(
                        f"RMS input queue {(hemisphere, slice_id)} "
                        f"was split: {len(domains)} domains")


    # FFN output writes retain their tensor SSA edge until command lowering.
    # Each mem_write here becomes exactly one physical MEM WRITE_3D.
    ffn_writes = [line for line in lines
                  if "ftlpu.schedule.mem_write" in line
                  and ffn_start <= attr(line, "cycle") < ffn_end
                  and "binding_slices = [" in line
                  and "wave_count = 12 : i64" in line]
    if len(ffn_writes) != 8 or any(
            (attr(line, "duration"), attr(line, "wave_interval"),
             attr(line, "wave_address_stride")) != (32, 17990, 32)
            for line in ffn_writes):
        raise AssertionError("FFN down result was not lowered to "
                             "one 12-tile MEM domain per physical queue")

    for unit_id in range(2):
        ffn_dequant = [line for line in lines
                       if "ftlpu.schedule.mxm_dequant" in line
                       and o_end <= attr(line, "cycle") < ffn_start
                       and attr(line, "unit_id") == unit_id]
        ffn_loads = [line for line in lines
                     if "ftlpu.schedule.mxm_load" in line
                     and o_end <= attr(line, "cycle") < ffn_start
                     and attr(line, "unit_id") == unit_id]
        if len(ffn_dequant) != 2 or any(
                (attr(line, "repeat_count"),
                 attr(line, "wave_count", 1),
                 attr(line, "group_count", 1),
                 attr(line, "group_interval", 1)) != (4, 48, 140, 1536)
                for line in ffn_dequant):
            raise AssertionError(f"FFN MXM dequant unit {unit_id} "
                                 "still follows weight-page boundaries")
        if len(ffn_loads) != 2 or any(
                (attr(line, "duration"), attr(line, "group_count", 1),
                 attr(line, "group_interval", 1)) != (4, 6720, 32)
                for line in ffn_loads):
            raise AssertionError(f"FFN MXM load unit {unit_id} "
                                 "still follows weight-page boundaries")

    sxm_transposes = [line for line in lines
                      if "ftlpu.schedule.sxm" in line
                      and 'opcode = "transpose"' in line]
    for hemisphere in range(2):
        probability = [line for line in sxm_transposes
                       if softmax_end <= attr(line, "cycle") < pv_start
                       and attr(line, "hemisphere") == hemisphere]
        pv = [line for line in sxm_transposes
              if pv_start <= attr(line, "cycle") < pv_end
              and attr(line, "hemisphere") == hemisphere]
        if len(probability) != 1 or (
                attr(probability[0], "repeat_count"),
                attr(probability[0], "wave_count", 1),
                attr(probability[0], "wave_interval", 1)) != (4, 6, 8):
            raise AssertionError("probability SXM transpose was split")
        if len(pv) > 2 or sum(attr(line, "wave_count", 1)
                              for line in pv) != 24:
            raise AssertionError("PV SXM transpose was split")

    q_combine = [line for line in lines
                 if "ftlpu.schedule.vxm" in line
                 and qkv_start <= attr(line, "cycle") < qkv_end
                 and attr(line, "repeat_count") == 32
                 and attr(line, "wave_count", 1) == 12
                 and attr(line, "wave_interval", 1) == 80]
    if len(q_combine) != 8:
        raise AssertionError("Q RoPE combine VXM domains were split")

    # Each physical context/result MEM queue gets one descriptor for all
    # 24 output groups.  The source-group axis is one on this seq32 layout,
    # leaving the third hardware counter free for outputGroup.
    expected: dict[tuple[int, int, int, str], tuple[int, ...]] = {}
    for hemisphere in range(2):
        for slice_id in range(16, 20):
            expected[(hemisphere, slice_id, 1, "read")] = (
                32, 24, 24, 32, 1578, 0)
    for slice_id in range(4):
        expected[(0, slice_id, 1, "write")] = (
            32, 1, 24, 1, 1578, 32)
    for slice_id in (2, 3):
        expected[(1, slice_id, 1, "write_tap")] = (
            32, 1, 24, 1, 1578, 32)

    matched = 0
    for key, shape in expected.items():
        hemisphere, slice_id, bank, opcode = key
        domains = [line for line in transfers
                   if o_start <= attr(line, "cycle") < o_end
                   and attr(line, "hemisphere") == hemisphere
                   and attr(line, "slice") == slice_id
                   and attr(line, "bank") == bank
                   and f'opcode = "{opcode}"' in line]
        observed = [(
            attr(line, "repeat_count"), attr(line, "wave_count", 1),
            attr(line, "group_count", 1), attr(line, "wave_interval", 1),
            attr(line, "group_interval", 1),
            attr(line, "group_address_stride", 0),
        ) for line in domains]
        if observed != [shape]:
            raise AssertionError(
                f"O projection did not directly emit one maximal {opcode} "
                f"on {key}: {observed}")
        domain = domains[0]
        begin, end = interval(domain)
        interfering = [line for line in transfers if line is not domain
                       and attr(line, "hemisphere") == hemisphere
                       and attr(line, "slice") == slice_id
                       and attr(line, "bank") == bank
                       and max(begin, interval(line)[0])
                       < min(end, interval(line)[1])]
        if interfering:
            raise AssertionError(
                f"non-preemptible O projection domain overlaps {key}: "
                f"{interfering[:2]}")
        matched += 1

    # MXM's three counters now represent weight byte/row, reduction, and
    # output group. The terminal action is on the reduction counter, so it
    # still fires once per output group.
    for unit_id in range(2):
        dequant = [line for line in lines
                   if "ftlpu.schedule.mxm_dequant" in line
                   and o_start <= attr(line, "cycle") < o_end
                   and attr(line, "unit_id") == unit_id]
        issues = [line for line in lines
                  if "ftlpu.schedule.mxm_issue" in line
                  and o_start <= attr(line, "cycle") < o_end
                  and attr(line, "unit_id") == unit_id]
        loads = [line for line in issues if 'opcode = "iw"' in line]
        computes = [line for line in issues
                    if 'opcode = "compute"' in line]
        for kind, domains, repeat in (("dequant", dequant, 4),
                                      ("load", loads, 4),
                                      ("compute", computes, 32)):
            if len(domains) != 1 or (
                    attr(domains[0], "repeat_count"),
                    attr(domains[0], "wave_count", 1),
                    attr(domains[0], "group_count", 1),
                    attr(domains[0], "wave_interval", 1),
                    attr(domains[0], "group_interval", 1)
            ) != (repeat, 48, 24, 32, 1578):
                raise AssertionError(
                    f"O MXM {kind} queue {unit_id} was split")
        if (attr(computes[0], "terminal_dimension") != 1
                or 'weight_buffer_mode = "toggle_dim1"' not in loads[0]
                or 'weight_buffer_mode = "toggle_dim1"' not in computes[0]):
            raise AssertionError(
                f"O MXM terminal/buffer axis incorrect on {unit_id}")
    print("Q RoPE, QK, softmax, PV, O projection, RMS and FFN "
          "direct ICU domains passed")


if __name__ == "__main__":
    main()
