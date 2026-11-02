"""Exercise BF16 spill legalization for sigmoid and softplus gate branches."""
import argparse
import json
import re
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
    p.add_argument('--log-softplus', action='store_true')
    p.add_argument('--token-halves', action='store_true')
    p.add_argument('--with-projection', action='store_true')
    a = p.parse_args()
    out = a.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / 'summary.json').write_text(json.dumps(dict(status='pending')))
    rng = np.random.default_rng(951)
    x = bf16(rng.normal(0, 1, (32, 16)))
    if a.log_softplus:
        x[:8] = bf16(np.broadcast_to(np.linspace(-8, -3, 8, dtype='f4')[:, None], (8,16)).copy())
    bias = bf16(rng.normal(0, .2, (1, 16)))
    if a.log_softplus:
        # Isolate Log tail accuracy from rounding the separate bias add.
        # Real checkpoint tests cover nonzero bias and its BF16 boundary.
        bias.fill(0)
    scale = bf16(-np.exp(rng.uniform(-1, 1, (1, 16))).astype('f4'))
    graph = LogicalGraph()
    operand=x
    if a.with_projection:
        raw=bf16(rng.normal(0,.3,(32,64)))
        weight=rng.integers(-16,17,(16,64),dtype=np.int8)
        operand=graph.project('gate_projection',raw,weight,.01)
        x=bf16(raw @ bf16(weight.astype('f4')*.01).T)
    g = VectorGraph(graph, 'gates', [(operand, 'bf16'), (bias, 'f32'), (scale, 'f32')])
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
    kernel.write_text(kernel.read_text().replace('module {', 'module attributes {ftlpu.vector_bf16_spill = true, ftlpu.vector_log_softplus = ' + str(a.log_softplus).lower() + ', ftlpu.vector_bf16_token_halves = ' + str(a.token_halves).lower() + '} {', 1))
    with (out / 'compile.log').open('w') as log:
        subprocess.run([str(a.compile.resolve()), '--input', str(kernel), '--input-stage', 'kernel',
                        '--output', str(binary), '--icu-compression', 'raw-macro', '--target-config', str(a.target_config.resolve())], stdout=log, stderr=subprocess.STDOUT, check=True)
    with (out / 'execution.log').open('w') as log:
        subprocess.run([str(a.runner.resolve()), str(binary), str(out / 'inputs.txt'),
                        str(out / 'actual'), str(out / 'runtime.pipeline.csv')], stdout=log, stderr=subprocess.STDOUT, check=True)
    expected = [1/(1+np.exp(-x)), np.logaddexp(0, x+bias)*scale]
    errors = []
    for i, reference in enumerate(expected):
        actual = np.fromfile(out / f'actual.{i}.bin', '<f4').reshape(x.shape)
        error = float(np.linalg.norm(actual-reference)/np.linalg.norm(reference))
        assert np.isfinite(actual).all() and error < .03, (i, error)
        if a.log_softplus and i == 1:
            tail = (x+bias < -3) & (np.abs(reference) > 1e-4)
            tail_relative = float(np.max(np.abs((actual-reference)[tail] / reference[tail]), initial=0))
            assert tail_relative < .04, ('softplus negative tail', tail_relative)
            assert np.all(actual[tail] < 0), 'negative scale times positive softplus must stay negative'
        errors.append(error)
    report = dict(status='passed', shape=list(x.shape), nrmse=errors,
                  precision='BF16 upper-plane ingress at spill boundaries, FP32 chain arithmetic and output ABI')
    if a.log_softplus:
        report['tail_max_relative_error'] = tail_relative
        assert 'log_lut' in (out / 'runtime.pipeline.csv').read_text()
    if a.token_halves:
        schedule=(out/'decoder_layer.schedule.mlir').read_text()
        repeats=[int(re.search(r'repeat_count = (\d+)', line).group(1))
                 for line in schedule.splitlines() if 'ftlpu.schedule.vxm ' in line]
        assert repeats and set(repeats)=={16}, repeats
        outputs=[line for line in schedule.splitlines() if 'access = "output"' in line and 'placement =' in line]
        assert len(outputs)==2 and all('token_partition' in line for line in outputs)
        report['tokens_per_hemisphere']=16
    (out / 'summary.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == '__main__':
    main()
