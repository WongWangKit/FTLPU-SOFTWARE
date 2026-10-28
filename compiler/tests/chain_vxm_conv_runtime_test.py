"""Causal Conv1D through Kernel graph lowering and the fixed-port VXM."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from qwen3_5_logical_graph import LogicalGraph
from qwen3_5_compiled_layer import bf16


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('compile', 'graph-compile', 'runner', 'target-config', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--channels', type=int, default=64)
    parser.add_argument('--activation', choices=('silu', 'none'), default='silu')
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / 'summary.json').write_text(json.dumps(dict(status='pending')))
    rng = np.random.default_rng(821)
    x = bf16(rng.normal(0, .4, (32, args.channels)))
    w = bf16(rng.normal(0, .3, (args.channels, 4)))
    graph = LogicalGraph()
    result = graph.conv(x, w)
    if args.activation == 'none':
        graph.nodes[0]['source'] = graph.nodes[0]['source'].replace('activation = "silu"', 'activation = "none"')
    logical = graph.write(out / 'graph', [result])
    kernel, binary = out / 'kernel.mlir', out / 'program.ftlpu'
    commands = [
        [str(args.graph_compile.resolve()), '--input', str(logical), '--output', str(kernel), '--emit-kernel'],
        [str(args.compile.resolve()), '--input', str(kernel), '--input-stage', 'kernel', '--output', str(binary)],
    ]
    for index, command in enumerate(commands):
        with (out / f'compile{index}.log').open('w') as log:
            subprocess.run(command + ['--target-config', str(args.target_config.resolve())], stdout=log, stderr=subprocess.STDOUT, check=True)
    with (out / 'execution.log').open('w') as log:
        subprocess.run([str(args.runner.resolve()), str(binary), str(out / 'inputs.txt'),
                        str(out / 'actual'), str(out / 'runtime.pipeline.csv')], stdout=log, stderr=subprocess.STDOUT, check=True)
    expected = np.zeros_like(x)
    for token in range(len(x)):
        for tap in range(4):
            source = token + tap - 3
            if source >= 0:
                expected[token] += x[source] * w[:, tap]
    expected = bf16(expected)
    if args.activation == 'silu':
        expected = bf16(expected / (1 + np.exp(-expected)))
    actual = (np.fromfile(out / 'actual.0.bin', '<u2').astype(np.uint32) << 16).view(np.float32).reshape(x.shape)
    error = float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))
    assert np.isfinite(actual).all() and error < .01, error
    report = dict(status='passed', shape=list(x.shape), activation=args.activation,
                  nrmse=error, max_abs_error=float(np.max(np.abs(actual-expected))))
    (out / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
