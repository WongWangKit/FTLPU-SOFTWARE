#!/usr/bin/env python3
"""A retained external activation survives RMSNorm and a capacity-tiled residual."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from qwen3_5_logical_graph import LogicalGraph
from qwen3_5_compiled_layer import bf16


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("graph-compile", "compile", "runner", "target-config", "output-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--retain-norm", action="store_true")
    a = p.parse_args()
    out = a.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(161022)
    x = bf16(rng.normal(0, .3, (32, 1536)).astype(np.float32))
    gamma = bf16(rng.normal(1, .02, 1536).astype(np.float32))
    graph = LogicalGraph()
    norm = graph.norm("norm", x, gamma, 1e-6)
    result = graph.vector_tiles("residual", [(x, "bf16"), (norm, "bf16")],
        lambda v: v.add(0, 1, kind="add", dtype="bf16"))
    logical = graph.write(out / "logical", [result, norm] if a.retain_norm else [result])
    subprocess.run([str(a.graph_compile.resolve()), "--input", str(logical),
        "--output", str(out / "residual.kernel.mlir"), "--emit-kernel"], check=True)
    subprocess.run([str(a.compile.resolve()), "--input", str(out / "residual.kernel.mlir"),
        "--input-stage", "kernel", "--output", str(out / "residual.ftlpu"),
        "--mxm-execution", "auto", "--target-config", str(a.target_config.resolve())], check=True)
    subprocess.run([str(a.runner.resolve()), str(out / "residual.ftlpu"),
        str(out / "inputs.txt"), str(out / "actual")], check=True)
    expected = bf16(x + bf16(x / np.sqrt(np.mean(x*x, axis=-1, keepdims=True) + 1e-6) * gamma))
    actual = (np.fromfile(out / "actual.0.bin", "<u2").astype(np.uint32) << 16).view(np.float32).reshape(x.shape)
    error = float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))
    summary = dict(tokens=32, width=1536, nrmse=error,
                   passed=bool(np.isfinite(actual).all() and error < .01))
    if a.retain_norm:
        retained = (np.fromfile(out / "actual.1.bin", "<u2").astype(np.uint32) << 16).view(np.float32).reshape(x.shape)
        norm_ref = bf16(x / np.sqrt(np.mean(x*x, axis=-1, keepdims=True) + 1e-6) * gamma)
        summary["retained_norm_nrmse"] = float(np.linalg.norm(retained-norm_ref)/np.linalg.norm(norm_ref))
        summary["passed"] &= bool(np.isfinite(retained).all() and summary["retained_norm_nrmse"] < .01)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(summary)
    assert summary["passed"]


if __name__ == "__main__":
    main()
