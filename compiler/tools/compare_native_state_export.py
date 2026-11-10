"""Compare validated GDN runs with native state export disabled/enabled."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from measure_compute_overlap import measure


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("optimized", type=Path)
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()
    roots = [args.baseline.resolve(), args.optimized.resolve()]
    configs = [read(root / "run_config.json") for root in roots]
    control = [dict(config) for config in configs]
    assert not control[0].pop("native_state_export", False)
    assert control[1].pop("native_state_export", False)
    assert control[0] == control[1], "A/B tools, input scope or hardware differ"
    output_hashes = []
    input_hashes = []
    report = {"hardware": configs[0]["hardware"], "encoding": configs[0]["encoding"], "arms": {}}
    for name, root in zip(("off", "on"), roots):
        folder = root / f"batch{args.batch}/step0"
        summary = read(folder / "summary.json")
        assert summary["status"] == "passed" and len(summary["errors"]) == args.batch
        outputs = {str(p.relative_to(folder)): digest(p) for p in folder.rglob("actual.*.bin")}
        assert len(outputs) == 4*args.batch+4, "Incomplete output/cache/state files"
        output_hashes.append(outputs)
        input_hashes.append([[digest(Path(p)) for p in (folder/"mixer"/f"request{i}"/"inputs.txt").read_text().splitlines()]
                             for i in range(args.batch)])
        activity = measure(folder / "runtime.pipeline.csv")
        assert activity["total"]["cycles"] == summary["cycles"]
        exports = [d for d in read(folder/"mixer/graph_plan.json")["linker_mem_transports"]
                   if d["phase"] == "validation_export.1"]
        heads = [h for error in summary["errors"] for h in error["state_heads"]]
        arm = {
            "cycles": summary["cycles"], "latency_ms": summary["latency_ms"],
            "scope": summary["scope"], "compute_activity": activity,
            "state_gather_domains": len(exports),
            "state_gather_bytes_per_request": sum(d["vectors"]*d["byte_planes"]*32 for d in exports),
            "mixer_io": summary["mixer_execution"], "phase_counters": summary["phase_execution"],
            "worst_nrmse": {key:max(e[key]["nrmse"] for e in summary["errors"])
                            for key in ("attention","state","conv_state")},
            "state_heads_checked": len(heads),
            "worst_state_head_nrmse": max(h["nrmse"] for h in heads),
            "program_hashes": {phase:digest(folder/phase/"program.ftlpu")
                               for phase in ("projections","mixer","output")},
            "reused_measured_stages": summary.get("reused_measured_stages", {}),
        }
        report["arms"][name] = arm
    assert output_hashes[0] == output_hashes[1], "A/B outputs differ"
    assert input_hashes[0] == input_hashes[1], "A/B mixer inputs differ"
    before, after = report["arms"].values()
    assert after["state_gather_domains"] == 0, "Native export retained a state gather"
    report.update(status="passed", outputs_bit_identical=len(output_hashes[0]),
                  cycles_saved=before["cycles"]-after["cycles"],
                  latency_reduction_percent=100*(1-after["cycles"]/before["cycles"]))
    destination = roots[1].parent/"native_export_comparison.json"
    destination.write_text(json.dumps(report,indent=2))
    print(json.dumps({key:report[key] for key in ("status","outputs_bit_identical","cycles_saved","latency_reduction_percent")}))


if __name__ == "__main__":
    main()
