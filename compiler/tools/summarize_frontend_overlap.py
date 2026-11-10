"""Measure MXM Compute / VXM arithmetic intersection in a raw-macro GDN run."""
import argparse
import csv
from collections import Counter
import hashlib
import json
from pathlib import Path

from summarize_projection_solve_overlap import (
    current_metrics, duration, epoch_offsets, expanded, intersection, merge, physical_range,
)


def snapshot(directory):
    load = lambda name: json.loads((directory / name).read_text(encoding="utf-8"))
    summary, plan = load("summary.json"), load("graph_plan.json")
    if summary["status"] != "passed" or summary["icu_compression"] != "raw-macro":
        raise ValueError("A passed raw-macro execution is required")

    metrics_rows, mxm_issues, vxm_issues = [], [], []
    issued_work = Counter()
    first = last = None
    with (directory / "runtime.pipeline.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            start, end = int(row["start"]), int(row["end"])
            end += (int(row["inner_count"]) - 1) * int(row["inner_interval"])
            end += (int(row["outer_count"]) - 1) * int(row["outer_interval"])
            if first is None or start < first[0]:
                first = start, row
            if last is None or end > last[0]:
                last = end, row
            resource = row["resource"]
            if resource in {"C2C.HostInput", "C2C.HostOutput", "Paging.Consumer", "Session.Invocation"}:
                metrics_rows.append(row)
            is_compute = resource.startswith("MXM.") and resource.endswith(".Compute")
            is_arithmetic = resource.startswith("VXM.") and row["detail"].split()[0] != "bypass"
            if is_compute or is_arithmetic:
                copies = int(row["inner_count"]) * int(row["outer_count"]) - int(row["skip_first"])
                key = resource if is_compute else resource + "." + row["detail"].split()[0]
                issued_work[key] += (int(row["end"]) - start) * copies
                issues = mxm_issues if is_compute else vxm_issues
                issues.extend(expanded(row))
    # Keep the full trace endpoints for current_metrics without retaining all MEM domains.
    for _, row in (first, last):
        if row not in metrics_rows:
            metrics_rows.append(row)
    timing = current_metrics(summary, metrics_rows)
    offsets = epoch_offsets(metrics_rows)
    mxm, vxm = merge(mxm_issues), merge(vxm_issues)
    simultaneous = intersection(mxm, vxm)

    def activity(windows):
        windows = merge(windows)
        total = duration(windows)
        m, v, both = [duration(intersection(issues, windows)) for issues in (mxm, vxm, simultaneous)]
        return dict(cycles=total, mxm_compute_cycles=m, vxm_arithmetic_cycles=v,
                    simultaneous_cycles=both, only_mxm=m-both, only_vxm=v-both,
                    neither=total-m-v+both)

    groups = {}
    for node in plan["nodes"]:
        name = node["name"]
        if name.startswith(("in_proj_qkv.page", "conv1d_silu.block")):
            groups.setdefault(tuple(physical_range(node, offsets)), []).append(name)
    pages = [span for span, names in groups.items() if any(n.startswith("in_proj_qkv.") for n in names)]
    conv = [span for span, names in groups.items() if any(n.startswith("conv1d_silu.") for n in names)]
    if not pages or not conv:
        raise ValueError("Missing QKV page or Conv1D execution window in graph plan")
    # Pair by actual overlapping time; either group can move when the plan changes.
    common = intersection(merge(pages), merge(conv))
    start = timing["cold_start_cycle"]
    stop = timing["invocation_range"][1]
    outputs = [f"actual.{i}.bin" for i in range(summary["runtime"]["outputs"])]
    evidence = ["graph_plan.json", "runtime.pipeline.csv", "actual.runtime.ftlpu", "frontend.ftlpu"]
    def digest(filename):
        return hashlib.sha256((directory / filename).read_bytes()).hexdigest()
    return dict(seq_len=summary["seq_len"], scope=summary["scope"], raw_macro=True,
                run_timing=timing, cold=activity([(start, stop)]),
                invocation=activity([timing["invocation_range"]]),
                issued_arithmetic_cycles_by_resource_opcode=dict(sorted(issued_work.items())),
                qkv_conv_common_window=activity(common),
                qkv_conv_groups=[dict(nodes=names, runtime_range=span, activity=activity([span]))
                                 for span, names in sorted(groups.items())],
                output_sha256={name: digest(name) for name in outputs},
                artifact_sha256={name: digest(name) for name in evidence},
                reference_errors=summary["errors"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--capture", action="store_true", help="Save an immutable baseline before rerunning")
    mode.add_argument("--compare", action="store_true", help="Compare final execution with that baseline")
    parser.add_argument("--allow-unchanged", action="store_true",
                        help="Accept equal cold time, full trace duration and overlap for an unaffected case")
    args = parser.parse_args()
    directory = args.result_dir
    base = directory / "baseline_frontend_overlap.json"
    if args.capture and base.exists():
        parser.error(f"Baseline already exists: {base}")
    current = snapshot(directory)
    if args.capture:
        with base.open("x", encoding="utf-8") as out:
            json.dump(current, out, indent=2)
        print(base, current["cold"])
        return
    previous = json.loads(base.read_text(encoding="utf-8"))
    old_t, new_t = previous["run_timing"], current["run_timing"]
    gain = old_t["cold_cycles_through_compute"] - new_t["cold_cycles_through_compute"]
    full_gain = old_t["full_trace_cycles"] - new_t["full_trace_cycles"]
    additional = current["cold"]["simultaneous_cycles"] - previous["cold"]["simultaneous_cycles"]
    frontend = (current["qkv_conv_common_window"]["simultaneous_cycles"]
                - previous["qkv_conv_common_window"]["simultaneous_cycles"])
    checks = dict(same_seq_len=previous["seq_len"] == current["seq_len"],
                  bit_identical_outputs=previous["output_sha256"] == current["output_sha256"],
                  same_dynamic_weight_bytes=old_t["dynamic_weight_bytes"] == new_t["dynamic_weight_bytes"],
                  same_arithmetic_work=(previous["issued_arithmetic_cycles_by_resource_opcode"] ==
                                        current["issued_arithmetic_cycles_by_resource_opcode"]),
                  no_host_compute=new_t["host_operations"] == old_t["host_operations"] == 0,
                  cold_time=gain >= 0 if args.allow_unchanged else gain > 0,
                  full_trace_time=full_gain >= 0 if args.allow_unchanged else full_gain > 0,
                  overlap=additional >= 0 if args.allow_unchanged else additional > 0,
                  qkv_conv_overlap=frontend >= 0 if args.allow_unchanged else frontend > 0)
    report = dict(status="passed" if all(checks.values()) else "failed", checks=checks,
                  saved_cold_cycles=gain, saved_full_trace_cycles=full_gain,
                  added_simultaneous_cycles=additional, added_qkv_conv_simultaneous_cycles=frontend,
                  counting="Half-open runtime issue ranges unioned across hemispheres and columns; "
                           "MXM.Compute and VXM non-bypass only. Cold includes initial ingress "
                           "and exposed page waits. Diagnostic readback is separate.",
                  baseline=previous, current=current)
    target = directory / "frontend_overlap_comparison.json"
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(target, report["status"], checks, "saved cold cycles", gain, "added overlap", additional)
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
