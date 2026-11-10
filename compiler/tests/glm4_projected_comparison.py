"""Compare completed projected-attention executions with their frozen overlap baselines."""
import argparse
from collections import Counter, defaultdict
import csv
import json
import os
from pathlib import Path
import re

import numpy as np

from glm4_flash_packed_u_test import graph_digest, sha
from summarize_glm4_flash import phase_timing


def trace_statistics(path, total_cycles):
    resources = {}
    arithmetic = Counter()
    arithmetic_cycles = Counter()
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            resource = row["resource"]
            if not resource.startswith(("MEM.", "MXM.", "VXM.", "SXM.", "C2C.")):
                continue
            detail = row["detail"]
            match = re.search(r"\bsource=runtime issues=(\d+)\b", detail)
            assert match, f"missing actual issue count in {resource}"
            issues = int(match[1])
            start, end = int(row["start"]), int(row["end"])
            inner, outer = (int(row[k] or 1) for k in ("inner_count", "outer_count"))
            inner_step, outer_step = (int(row[k] or 0) for k in ("inner_interval", "outer_interval"))
            skip = int(row.get("skip_first") or 0)
            assert 0 <= skip <= 1 and end > start
            assert end + (inner - 1) * inner_step + (outer - 1) * outer_step <= total_cycles
            stat = resources.setdefault(resource, dict(runtime_issue_count=0, event_cycle_sum=0,
                                                       busy=np.zeros(total_cycles, dtype=np.bool_),
                                                       operations=Counter()))
            stat["runtime_issue_count"] += issues
            stat["event_cycle_sum"] += (end - start) * (inner * outer - skip)
            operation = detail.split()[0] if resource.startswith(("VXM.", "MXM.", "SXM.")) else resource.rsplit(".", 1)[-1]
            stat["operations"][operation] += issues
            # The CSV encodes periodic spans. A strided view marks their exact
            # union without allocating an expanded per-cycle event list.
            view = np.ndarray((outer, inner, end - start), dtype=np.bool_, buffer=stat["busy"],
                              offset=start, strides=(outer_step, inner_step, 1))
            if skip:
                view[0, 1:, :] = True
                view[1:, :, :] = True
            else:
                view[:] = True
            if resource.startswith("VXM.") and operation != "bypass":
                arithmetic["VXM." + operation] += issues
                arithmetic_cycles["VXM." + operation] += (end - start) * (inner * outer - skip)
            elif resource.startswith("MXM.") and resource.endswith(".Compute"):
                arithmetic["MXM.Compute"] += issues
                arithmetic_cycles["MXM.Compute"] += (end - start) * (inner * outer - skip)
    for stat in resources.values():
        stat["busy_cycles"] = int(np.count_nonzero(stat.pop("busy")))
        stat["operations"] = dict(sorted(stat["operations"].items()))
    return dict(resources=dict(sorted(resources.items())), arithmetic_issues=dict(sorted(arithmetic.items())),
                arithmetic_event_cycles=dict(sorted(arithmetic_cycles.items())))


def transport_statistics(path):
    plan = json.loads(path.read_text())
    routes = defaultdict(Counter)
    total = Counter()
    for transfer in plan["linker_mem_transports"]:
        route = transfer["route"]
        vectors = int(transfer["vectors"])
        planes = vectors * int(transfer["byte_planes"])
        local = planes if transfer.get("local_tap") else 0
        values = dict(transports=1, vectors=vectors, vector_byte_planes=planes,
                      read_vector_byte_planes=planes, write_vector_byte_planes=planes + local,
                      local_tap_write_vector_byte_planes=local,
                      scheduled_cycle_sum=int(transfer["end"]) - int(transfer["start"]))
        routes[route].update(values)
        total.update(values)
    return dict(totals=dict(total), routes=dict(routes),
                native_output_archive_copies=plan["native_output_archive_copies"],
                graph_plan_sha256=sha(path))


def icu_statistics(folder):
    totals = Counter()
    resources = defaultdict(Counter)
    queues = {}
    with (folder / "index.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            assert (folder / row["file"]).stat().st_size > 0
            values = {key: int(row[key]) for key in ("coarse_instructions", "imem_words", "expanded_fu_issues")}
            totals.update(values)
            resources[row["resource"]].update(values)
            if values["imem_words"]:
                queues[row["file"]] = values["imem_words"]
    largest = max(queues, key=queues.get)
    return dict(totals=dict(totals), resources=dict(resources), queue_imem_words=queues,
                largest_queue=dict(file=largest, imem_words=queues[largest]))


def verify_phase(folder, phase, clock):
    stamp = json.loads((folder / "completed.json").read_text())
    assert stamp["compiler_sha256"] == phase["compiler_sha256"]
    assert stamp["binary_sha256"] == phase["binary_sha256"]
    assert stamp["semantic_graph_sha256"] == phase["semantic_graph_sha256"]
    assert sha(folder / "actual.runtime.ftlpu") == phase["actual_program_sha256"]
    assert sha(folder / "runtime.pipeline.csv") == phase["runtime_pipeline_sha256"]
    timing = phase_timing(folder, clock)
    assert timing == phase["timing"]
    return timing


def compare_batch(root, batch):
    current_dir = root / f"batch{batch}" / "fused"
    previous_dir = root / f"batch{batch}" / "fused_overlap"
    old_run = previous_dir / "candidate"
    previous = json.loads((previous_dir / "accepted_result.json").read_text())
    proof = json.loads((previous_dir / "result.json").read_text())
    current = json.loads((current_dir / "result.json").read_text())
    assert previous["status"] == proof["status"] == current["status"] == "passed"
    assert current["batch"] == previous["batch"] == proof["batch"] == batch
    assert current["mode"] == previous["mode"] == "fused"
    for key in ("scope", "model", "layer", "past_len", "decode_tokens_per_request", "outputs",
                "icu_encoding", "sxm", "weight_paging_policy", "target_sha256", "clock_mhz",
                "fixture_fingerprint", "token_ids_sha256", "hidden_sha256", "metadata_sha256",
                "config_sha256", "past_kv_sha256", "weights_sha256"):
        assert current[key] == previous[key], f"batch {batch} changed {key}"
    assert current["runner_sha256"] == proof["candidate_runner_sha256"]
    assert current["target_sha256"] == proof["target_sha256"]
    old = proof["phases"]["full_attention"]
    old_phase = previous["phases"]["full_attention"]
    new_phase = current["phases"]["full_attention"]
    assert old["program_sha256"] == old_phase["binary_sha256"]
    assert proof["compiler_sha256"] == previous["compiler_sha256"] == old_phase["compiler_sha256"]
    assert old["candidate"]["runner_sha256"] == current["runner_sha256"]
    assert old["candidate"]["output_sha256"] == previous["output_sha256"]
    assert old["expected_output_sha256"] == previous["output_sha256"]
    assert not any(old["candidate"]["bitwise_mismatches"].values())
    assert new_phase["compiler_sha256"] == current["compiler_sha256"]
    assert old_phase["vector_bf16_rne_spill"] is new_phase["vector_bf16_rne_spill"] is True
    assert graph_digest(current_dir) == new_phase["semantic_graph_sha256"] == old["semantic_graph_sha256"]
    assert old_phase["semantic_graph_sha256"] == old["semantic_graph_sha256"]
    assert sha(current_dir / "program.ftlpu") == new_phase["binary_sha256"]
    assert sha(previous_dir / "inputs/program.ftlpu") == old["program_sha256"]
    input_lines = (previous_dir / "inputs/inputs.txt").read_text().splitlines()
    assert len(input_lines) == len(old["inputs"])
    for line, item in zip(input_lines, old["inputs"]):
        filename, _, scale = line.partition("\t")
        assert (scale or None) == item["scale"]
        assert sha(Path(filename)) == sha(previous_dir / "inputs" / item["name"]) == item["sha256"]
    old_trace_phase = dict(old_phase, actual_program_sha256=old["candidate"]["linked_program_sha256"],
                           runtime_pipeline_sha256=old["candidate"]["runtime_pipeline_sha256"],
                           timing=old["candidate"]["timing"])
    before = verify_phase(old_run, old_trace_phase, current["clock_mhz"])
    after = verify_phase(current_dir, new_phase, current["clock_mhz"])
    assert after["runtime_cycles"] == current["runtime_cycles"]
    mismatches = {}
    for index, name in enumerate(current["outputs"]):
        a, b = old_run / f"actual.{index}.bin", current_dir / current["output_files"][index]
        assert sha(a) == old["candidate"]["output_sha256"][index]
        assert sha(b) == current["output_sha256"][index]
        left, right = np.fromfile(a, "<u2"), np.fromfile(b, "<u2")
        assert left.shape == right.shape
        mismatches[name] = int(np.count_nonzero(left != right))
    assert not any(mismatches.values()), mismatches
    assert current["historical_kv_bit_exact"]
    assert current["projected_metrics"]["relative_l2"] < .03
    assert current["worst_request_relative_l2"] < .08
    assert all(m["relative_l2"] < .03 for m in current["new_cache_metrics"].values())
    old_stats = trace_statistics(old_run / "runtime.pipeline.csv", before["runtime_cycles"])
    new_stats = trace_statistics(current_dir / "runtime.pipeline.csv", after["runtime_cycles"])
    assert old_stats["arithmetic_issues"] == new_stats["arithmetic_issues"], "arithmetic issue counts changed"
    deltas = {}
    for resource in sorted(old_stats["resources"].keys() | new_stats["resources"].keys()):
        left, right = old_stats["resources"].get(resource, {}), new_stats["resources"].get(resource, {})
        deltas[resource] = {key: right.get(key, 0) - left.get(key, 0)
                            for key in ("runtime_issue_count", "busy_cycles", "event_cycle_sum")}
    saved = before["runtime_cycles"] - after["runtime_cycles"]
    return dict(status="passed", batch=batch, runner_sha256=current["runner_sha256"],
                target_sha256=current["target_sha256"], previous_compiler_sha256=previous["compiler_sha256"],
                current_compiler_sha256=current["compiler_sha256"], graph_sha256=new_phase["semantic_graph_sha256"],
                passive_sr_multicast=current.get("passive_sr_multicast"),
                fixture_fingerprint=current["fixture_fingerprint"], bitwise_mismatches=mismatches,
                projected_metrics=current["projected_metrics"], worst_request_relative_l2=current["worst_request_relative_l2"],
                new_cache_metrics=current["new_cache_metrics"], historical_kv_bit_exact=True,
                previous=dict(timing=before, runtime=old_stats, transports=transport_statistics(previous_dir / "accepted_graph_plan.json"),
                              icu=icu_statistics(old_run / "icu_programs")),
                current=dict(timing=after, runtime=new_stats, transports=transport_statistics(current_dir / "graph_plan.json"),
                             icu=icu_statistics(current_dir / "icu_programs")),
                resource_deltas=deltas, cycles_saved=saved,
                reduction=saved / before["runtime_cycles"], speedup=before["runtime_cycles"] / after["runtime_cycles"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path,
                        default=Path("results/glm4_9b_decoder_layer_decode_flash_projected/optimized"))
    args = parser.parse_args()
    result = dict(status="passed", scope="GLM-4-9B W8A16 layer0 projected attention, decode one token, past32",
                  baseline="fused_overlap/candidate", current="fused",
                  notes=[
                      "Total runtime includes initial ingress, device execution, weight waits and output/cache readback.",
                      "Runtime issue counts come from the actual CSV issues field. Busy cycles are interval unions per displayed resource.",
                      "Event cycle sums retain parallel slice activity. They can exceed elapsed runtime.",
                      "Passive transport counts and scheduled cycle sums come from graph_plan.json and are compile-time diagnostics.",
                      "VXM spans cover repeat_count cycles of a dispatched packet. Its issue count counts packet dispatches.",
                      "iMEM loading and replacement costs are excluded by the current runner.",
                  ], batches={})
    for batch in (1, 32):
        result["batches"][str(batch)] = compare_batch(args.results, batch)
        print(f"batch {batch}: passed, saved {result['batches'][str(batch)]['cycles_saved']} cycles", flush=True)
    path = args.results / "comparison.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(temporary, path)
    print(path)


if __name__ == "__main__":
    main()
