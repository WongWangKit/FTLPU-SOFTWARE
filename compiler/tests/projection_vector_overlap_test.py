"""Check independent FP32 vector outputs and projection with real raw-macro programs."""
import argparse
import json
import os
from pathlib import Path
import re
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from qwen3_5_logical_graph import LogicalGraph
from qwen3_5_compiled_layer import bf16
from qwen3_5_decode_gdn_attention_test import run
from projection_solve_overlap_test import relative_error, trace_metrics, truncate


def execute(args, dependent, enabled):
    out = args.output_dir.resolve() / ("dependent" if dependent else "independent") / ("on" if enabled else "off")
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(dict(status="pending")))
    exe = lambda area, name: args.build_dir.resolve() / area / (name + (".exe" if sys.platform == "win32" else ""))
    rng = np.random.default_rng(35261110)
    old = rng.normal(0, .2, (2048, 128)).astype("f4")
    coefficient = rng.uniform(.7, .99, old.shape).astype("f4")
    increment = rng.normal(0, .05, old.shape).astype("f4")
    x = bf16(rng.normal(0, .2, (32, 1024)).astype("f4"))
    width, hidden = (128, 8192) if dependent else (1024, 1024)
    weight = rng.integers(-12, 13, (width, hidden), dtype=np.int8)
    scale = .015625
    graph = LogicalGraph()
    updated = graph.vector_tiles("updated", [(old, "f32"), (coefficient, "f32"), (increment, "f32")],
        lambda vector: vector.add(vector.add(0, 1), 2, kind="add"))
    if dependent:
        activation = graph.truncate_bf16("dependent_input", updated.reshape(32, 8192))
    else:
        # This ordinary producer is after the two terminal vector blocks.
        # Moving the blocks must preserve the projection's input dependency.
        activation = graph.vector_tiles("activation", [(x, "bf16"), (np.full_like(x, .5), "bf16")],
            lambda vector: vector.add(0, 1, dtype="bf16"))
    projected = graph.project("linear", activation, weight, scale)
    logical = graph.write(out / "logical_graph", [updated, projected])
    kernel, binary = out / "kernel.mlir", out / "program.ftlpu"
    run([exe("compiler", "ftlpu-graph-compile"), "--input", logical, "--output", kernel,
         "--emit-kernel", "--target-config", args.target_config.resolve()], out / "import.log")
    attrs = ("ftlpu.sxm_wide = true, ftlpu.common_barrier_protocol = true, "
             "ftlpu.vector_fp32_ingress = true, ftlpu.vector_bf16_spill = true, "
             "ftlpu.vector_bf16_token_halves = true, "
             f"ftlpu.projection_vector_overlap = {str(enabled).lower()}")
    kernel.write_text(kernel.read_text().replace("module {", f"module attributes {{{attrs}}} {{", 1))
    compiler = args.compiler_executable.resolve() if args.compiler_executable else exe("compiler", "ftlpu-compile")
    run([compiler, "--input", kernel, "--input-stage", "kernel", "--output", binary,
         "--target-config", args.target_config.resolve(), "--icu-compression", "raw-macro",
         "--mxm-execution", "ws"], out / "compile.log")
    os.environ.update(FTLPU_FILE_RUNNER_MODEL_SESSION="1", FTLPU_FILE_RUNNER_SXM_ARCHITECTURE="wide")
    run([exe("runtime", "compiled_stage_file_runner"), binary, out / "inputs.txt",
         out / "actual", out / "runtime.pipeline.csv"], out / "execution.log")
    actual = np.fromfile(out / "actual.0.bin", "<f4").reshape(old.shape)
    expected = old * coefficient + increment
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=5e-7)
    projection = (np.fromfile(out / "actual.1.bin", "<u2").astype("u4") << 16).view("f4").reshape(32, width)
    projection_input = truncate(actual).reshape(32, 8192) if dependent else x * .5
    expected_projection = projection_input.astype("f8") @ (weight.astype("f8") * scale).T
    error = relative_error(projection, expected_projection)
    assert np.isfinite(projection).all() and error < .02, error
    log = (out / "execution.log").read_text()
    stats = {key: int(value) for key, value in re.findall(r"(\w+)=(\d+)", log.splitlines()[-1])}
    assert stats.get("host_operations") == 0, stats
    plan = json.loads((out / "graph_plan.json").read_text())
    run([exe("runtime", "ftlpu_icu_program_export"), out / "actual.runtime.ftlpu",
         out / "runtime_icu_programs", "--pre-execution-dir", out / "actual.pre_execution"], out / "export.log")
    summary = dict(status="numerics_passed", dependent=dependent, projection_vector_overlap="on" if enabled else "off",
                   icu_compression="raw-macro", runtime=stats,
                   scheduling=plan.get("projection_vector_overlap", []),
                   vector_max_abs=float(np.max(np.abs(actual-expected))), projection_nrmse=error,
                   **trace_metrics(out / "runtime.pipeline.csv"))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return out, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("build-dir", "target-config", "output-dir"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--compiler-executable", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(dict(status="pending")))
    checks, runs = [], []
    for dependent in (False, True):
        before, old = execute(args, dependent, False)
        after, new = execute(args, dependent, True)
        assert all((before / f"actual.{i}.bin").read_bytes() == (after / f"actual.{i}.bin").read_bytes()
                   for i in range(2)), dependent
        assert old["runtime"]["weight_page_prefetch_bytes"] == new["runtime"]["weight_page_prefetch_bytes"]
        assert old["mxm_vxm_overlap_cycles"] == 0
        enabled_groups = [group for group in new["scheduling"] if group["status"] == "enabled"]
        if dependent:
            assert not enabled_groups, new
            assert new["mxm_vxm_overlap_cycles"] == 0, new
            assert new["cold_cycles_through_compute"] == old["cold_cycles_through_compute"], (old, new)
        else:
            assert any(len(group["vectors"]) == 2 for group in enabled_groups), new
            assert new["mxm_vxm_overlap_cycles"] > 0, new
            assert new["cold_cycles_through_compute"] < old["cold_cycles_through_compute"], (old, new)
        checks.append(dict(dependent=dependent, bit_identical_outputs=2, before=old, after=new))
        runs.extend(((before, old), (after, new)))
    for directory, result in runs:
        result["status"] = "passed"
        (directory / "summary.json").write_text(json.dumps(result, indent=2))
    summary = dict(status="passed", scope="two FP32 multiply/add output blocks with independent or dependent projection",
                   checks=checks)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
