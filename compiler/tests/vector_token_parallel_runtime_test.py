#!/usr/bin/env python3
"""Exercise mirrored token partitions, row broadcasts and ordered readback."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from qwen3_5_logical_graph import LogicalGraph
from qwen3_5_compiled_layer import VectorGraph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('compile', 'graph-compile', 'runner', 'target-config', 'output-dir'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    reports = []
    for rows in (1, 5, 32):
        out = args.output_dir.resolve() / f'rows{rows}'
        out.mkdir(parents=True, exist_ok=True)
        x = np.linspace(-9, 9, rows * 16, dtype=np.float32).reshape(rows, 16)
        a = np.cos(x) * 3
        bias = np.linspace(-2, 2, 16, dtype=np.float32)[None, :]
        scale = np.linspace(-.2, -2, 16, dtype=np.float32)[None, :]
        graph = LogicalGraph()
        vector = VectorGraph(graph, 'pointwise', [(v, 'f32') for v in (x, a, bias, scale)])
        beta = vector.add(0, kind='sigmoid', immediate=0)
        shifted = vector.add(1, 2, kind='add')
        soft = vector.add(shifted, kind='softplus', immediate=0)
        decay = vector.add(soft, 3)
        outputs = vector.finish((beta, decay))
        logical = graph.write(out / 'logical', list(outputs))
        kernel, binary = out / 'test.mlir', out / 'test.ftlpu'
        subprocess.run([str(args.graph_compile.resolve()), '--input', str(logical),
                        '--output', str(kernel), '--emit-kernel', '--target-config',
                        str(args.target_config.resolve())], check=True)
        subprocess.run([str(args.compile.resolve()), '--input', str(kernel),
                        '--input-stage', 'kernel', '--output', str(binary),
                        '--target-config', str(args.target_config.resolve())], check=True)
        env = os.environ.copy()
        env['FTLPU_FILE_RUNNER_MODEL_SESSION'] = '1'
        for key in ('FTLPU_SESSION_LINK_ONLY', 'FTLPU_SESSION_STOP_CYCLE'):
            env.pop(key, None)
        with (out / 'execution.log').open('w') as log:
            subprocess.run([str(args.runner.resolve()), str(binary), str(out / 'inputs.txt'),
                            str(out / 'test'), str(out / 'runtime.pipeline.csv')],
                           env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        errors = []
        for index, reference in enumerate((1 / (1 + np.exp(-x.astype(np.float64))),
                                           scale * np.logaddexp(0, a.astype(np.float64) + bias))):
            actual = np.fromfile(out / f'test.{index}.bin', '<f4').reshape(rows, 16)
            np.testing.assert_allclose(actual, reference, rtol=2e-3, atol=1e-4)
            errors.append(float(np.max(np.abs(actual - reference))))
        plan = json.loads((out / 'graph_plan.json').read_text())['nodes']
        if len(plan) != (2 if rows > 1 else 1):
            raise AssertionError('unexpected token partition count')
        if rows > 1:
            schedule = (out / 'decoder_layer.schedule.mlir').read_text(encoding='utf-8')
            bindings = [line for line in schedule.splitlines()
                        if 'ftlpu.schedule.binding' in line and 'access = "output"' in line]
            if len(bindings) != 2 or any('kind = "token_partition_planar"' not in line
                                         for line in bindings):
                raise AssertionError('native partition output must not be merged before readback')
            if plan[0]['nominal_start'] != plan[1]['nominal_start']:
                raise AssertionError('mirrored partitions must start concurrently')
            with (out / 'runtime.pipeline.csv').open(newline='') as stream:
                trace = list(csv.DictReader(stream))
            heads = {row['resource']: row for row in trace
                     if row['resource'] in ('VXM.C0', 'VXM.C15')
                     and row['detail'].startswith('neg ')}
            # Use the first negate; subsequent negates belong to softplus.
            starts = []
            for resource in ('VXM.C0', 'VXM.C15'):
                starts.append(min(int(row['start']) for row in trace
                                  if row['resource'] == resource and row['detail'].startswith('neg ')))
            if len(heads) != 2 or starts[0] != starts[1]:
                raise AssertionError('actual ICU trace must contain simultaneous mirrored heads')
        reports.append(dict(rows=rows, max_abs=errors))
    (args.output_dir / 'summary.json').write_text(json.dumps(reports, indent=2) + '\n')
    print(json.dumps(reports, indent=2))


if __name__ == '__main__':
    main()
