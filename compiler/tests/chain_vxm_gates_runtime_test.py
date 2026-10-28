"""Exercise BF16 spill legalization for sigmoid and softplus gate branches."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from qwen3_5_logical_graph import LogicalGraph
from qwen3_5_compiled_layer import VectorGraph, bf16


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('compile', 'graph-compile', 'runner', 'target-config', 'output-dir'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    out = a.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / 'summary.json').write_text(json.dumps(dict(status='pending')))
    rng = np.random.default_rng(951)
    x = bf16(rng.normal(0, 1, (32, 16)))
    bias = bf16(rng.normal(0, .2, (1, 16)))
    scale = bf16(-np.exp(rng.uniform(-1, 1, (1, 16))).astype('f4'))
    graph = LogicalGraph()
    g = VectorGraph(graph, 'gates', [(x, 'bf16'), (bias, 'f32'), (scale, 'f32')])
    beta = g.add(0, kind='sigmoid', immediate=0)
    biased = g.add(0, 1, kind='add')
    softplus = g.add(biased, kind='softplus', immediate=0)
    decay = g.add(softplus, 2)
    finals = g.finish((beta, decay))
    logical = graph.write(out / 'graph', finals)
    kernel, binary = out / 'kernel.mlir', out / 'program.ftlpu'
    with (out / 'import.log').open('w') as log:
        subprocess.run([str(a.graph_compile.resolve()), '--input', str(logical), '--output', str(kernel),
                        '--emit-kernel', '--target-config', str(a.target_config.resolve())], stdout=log, stderr=subprocess.STDOUT, check=True)
    kernel.write_text(kernel.read_text().replace('module {', 'module attributes {ftlpu.vector_bf16_spill = true} {', 1))
    with (out / 'compile.log').open('w') as log:
        subprocess.run([str(a.compile.resolve()), '--input', str(kernel), '--input-stage', 'kernel',
                        '--output', str(binary), '--target-config', str(a.target_config.resolve())], stdout=log, stderr=subprocess.STDOUT, check=True)
    with (out / 'execution.log').open('w') as log:
        subprocess.run([str(a.runner.resolve()), str(binary), str(out / 'inputs.txt'),
                        str(out / 'actual'), str(out / 'runtime.pipeline.csv')], stdout=log, stderr=subprocess.STDOUT, check=True)
    expected = [1/(1+np.exp(-x)), np.logaddexp(0, x+bias)*scale]
    errors = []
    for i, reference in enumerate(expected):
        actual = np.fromfile(out / f'actual.{i}.bin', '<f4').reshape(x.shape)
        error = float(np.linalg.norm(actual-reference)/np.linalg.norm(reference))
        assert np.isfinite(actual).all() and error < .03, (i, error)
        errors.append(error)
    report = dict(status='passed', shape=list(x.shape), nrmse=errors,
                  precision='BF16 upper-plane ingress at spill boundaries, FP32 chain arithmetic and output ABI')
    (out / 'summary.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == '__main__':
    main()
