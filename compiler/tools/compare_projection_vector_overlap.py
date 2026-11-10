"""Compare complete GDN raw-macro runs with projection/vector overlap off and on."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from summarize_frontend_overlap import snapshot
from summarize_projection_solve_overlap import (
    duration, epoch_offsets, expanded, intersection, merge, physical_range,
)


def selected_activity(directory, names):
    """Intersect actual FU issue cycles with the selected graph nodes' runtime windows."""
    plan = json.loads((directory / "graph_plan.json").read_text())
    mxm, vxm, paging = [], [], []
    with (directory / "runtime.pipeline.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            resource = row["resource"]
            if resource == "Paging.Consumer":
                paging.append(row)
            if resource.startswith("MXM.") and resource.endswith(".Compute"):
                mxm.extend(expanded(row))
            if resource.startswith("VXM.") and row["detail"].split()[0] != "bypass":
                vxm.extend(expanded(row))
    offsets = epoch_offsets(paging)
    mxm, vxm = merge(mxm), merge(vxm)
    both = intersection(mxm, vxm)
    found = {}
    for node in plan["nodes"]:
        if node["name"] in names:
            window = physical_range(node, offsets)
            found[node["name"]] = dict(
                runtime_range=window,
                mxm_compute_cycles=duration(intersection(mxm, [window])),
                vxm_arithmetic_cycles=duration(intersection(vxm, [window])),
                simultaneous_cycles=duration(intersection(both, [window])),
            )
    if set(found) != set(names):
        raise ValueError(f"Missing selected nodes in {directory}: {set(names) - set(found)}")
    windows = merge(item["runtime_range"] for item in found.values())
    return dict(nodes=found, simultaneous_cycles=duration(intersection(both, windows)))


def executed_icu(directory):
    root = directory / "runtime_icu_programs"
    index = root / "index.csv"
    files = sorted(root.glob("*.icu.csv"))
    if not index.is_file() or not files:
        raise ValueError(f"Missing executed ICU export in {root}")
    return dict(path=str(root.resolve()), queues=len(files),
                index_sha256=hashlib.sha256(index.read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("off", type=Path)
    parser.add_argument("on", type=Path)
    args = parser.parse_args()
    before, after = (snapshot(directory) for directory in (args.off, args.on))
    summaries = [json.loads((d / "summary.json").read_text()) for d in (args.off, args.on)]
    plans = [json.loads((d / "graph_plan.json").read_text()) for d in (args.off, args.on)]
    groups = [group for group in plans[1].get("projection_vector_overlap", [])
              if group["status"] == "enabled"]
    names = {name for group in groups for name in [group["projection"], *group["vectors"]]}
    old_tail, new_tail = (selected_activity(directory, names) for directory in (args.off, args.on))
    old, new = before["run_timing"], after["run_timing"]
    checks = dict(
        switch_off_on=[s.get("projection_vector_overlap") for s in summaries] == ["off", "on"],
        same_shape_and_scope=(before["seq_len"], before["scope"]) == (after["seq_len"], after["scope"]),
        same_precision=summaries[0]["precision"] == summaries[1]["precision"],
        same_other_policies=all(summaries[0].get(key) == summaries[1].get(key) for key in
                               ("projection_solve_overlap", "projection_conv_overlap", "passive_sr_multicast")),
        enabled_groups=bool(groups) and not any(
            group["status"] == "enabled" for group in plans[0].get("projection_vector_overlap", [])),
        identical_outputs=before["output_sha256"] == after["output_sha256"],
        identical_reference_errors=before["reference_errors"] == after["reference_errors"],
        identical_arithmetic_work=before["issued_arithmetic_cycles_by_resource_opcode"] == after["issued_arithmetic_cycles_by_resource_opcode"],
        same_external_work=all(old[key] == new[key] for key in
                               ("dynamic_weight_bytes", "dynamic_weight_pages", "host_input_payload_bytes")),
        hardware_execution=old["host_operations"] == new["host_operations"] == 0,
        cold_gain=new["cold_cycles_through_compute"] < old["cold_cycles_through_compute"],
        full_trace_gain=new["full_trace_cycles"] < old["full_trace_cycles"],
        global_overlap_gain=after["cold"]["simultaneous_cycles"] > before["cold"]["simultaneous_cycles"],
        selected_overlap_gain=new_tail["simultaneous_cycles"] > old_tail["simultaneous_cycles"],
    )
    report = dict(
        status="passed" if all(checks.values()) else "failed", checks=checks,
        counting="Union of real MXM Compute and non-bypass VXM issue cycles; includes initial ingress and exposed DDR waits in cold time.",
        before=before, after=after, selected_groups=groups,
        selected_before=old_tail, selected_after=new_tail,
        executed_icu_before=executed_icu(args.off), executed_icu_after=executed_icu(args.on),
        saved_cold_cycles=old["cold_cycles_through_compute"]-new["cold_cycles_through_compute"],
        saved_full_trace_cycles=old["full_trace_cycles"]-new["full_trace_cycles"],
        added_simultaneous_cycles=after["cold"]["simultaneous_cycles"]-before["cold"]["simultaneous_cycles"],
    )
    (args.on / "projection_vector_comparison.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(checks))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
