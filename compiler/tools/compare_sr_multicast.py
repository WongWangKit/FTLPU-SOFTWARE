"""Compare two complete multichunk raw-macro runs of the same fixture."""
import argparse
import csv
import json
from pathlib import Path

from summarize_frontend_overlap import snapshot


def mem_issues(directory):
    counts = {"Read": 0, "Write": 0, "WriteTap": 0}
    with (directory / "runtime.pipeline.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            unit, _, op = row["resource"].rpartition(".")
            if unit in ("MEM.W", "MEM.E") and op in counts:
                counts[op] += int(row["detail"].rsplit("issues=", 1)[1].split()[0])
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("off", type=Path)
    parser.add_argument("on", type=Path)
    args = parser.parse_args()
    before, after = (snapshot(directory) for directory in (args.off, args.on))
    def domains(directory):
        plan = json.loads((directory / "graph_plan.json").read_text())
        return [d for d in plan["linker_mem_transports"] if d["route"] == "passive_sr_multicast"]

    taps = domains(args.on)
    old_mem, new_mem = (mem_issues(directory) for directory in (args.off, args.on))
    saved_reads = sum(d["vectors"] * d["byte_planes"] for d in taps)
    old, new = before["run_timing"], after["run_timing"]
    checks = {
        "same_shape_and_scope": (before["seq_len"], before["scope"]) == (after["seq_len"], after["scope"]),
        "multicast_exercised": bool(taps) and not domains(args.off),
        "runtime_reads_eliminated": old_mem["Read"] - new_mem["Read"] == saved_reads,
        "both_destinations_written": old_mem["Write"] + old_mem["WriteTap"] == new_mem["Write"] + new_mem["WriteTap"],
        "runtime_local_taps": new_mem["WriteTap"] - old_mem["WriteTap"] == saved_reads,
        "identical_outputs": before["output_sha256"] == after["output_sha256"],
        "identical_reference_errors": before["reference_errors"] == after["reference_errors"],
        "identical_arithmetic_work": before["issued_arithmetic_cycles_by_resource_opcode"] == after["issued_arithmetic_cycles_by_resource_opcode"],
        "same_external_work": all(old[k] == new[k] for k in ("dynamic_weight_bytes", "dynamic_weight_pages", "host_input_payload_bytes")),
        "hardware_execution": old["host_operations"] == new["host_operations"] == 0,
        "cold_gain": new["cold_cycles_through_compute"] < old["cold_cycles_through_compute"],
        "full_trace_gain": new["full_trace_cycles"] < old["full_trace_cycles"],
    }
    report = dict(checks=checks, multicast_domains=len(taps), old_mem=old_mem, new_mem=new_mem,
                  multicast_vectors=sum(d["vectors"] for d in taps),
                  before=before, after=after)
    (args.on / "sr_multicast_comparison.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(checks))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
