"""Packed graph producers must populate both hemispheres of decode recurrence."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from qwen3_5_logical_graph import LogicalGraph
from qwen3_5_compiled_layer import VectorGraph
from qwen3_5_decode_gdn_test import fixture, reload_reference


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('compile', 'graph-compile', 'runner', 'target-config', 'output-dir'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    out = a.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    h, d = 2, 32
    values = list(fixture(3, h, d))
    # Nonzero, non-unit gates expose lost scalar broadcasts on either side.
    values[3][:] = [.7, .9]
    values[4][:] = [.3, .6]
    graph = LogicalGraph()
    produced = []
    for i, value in enumerate(values[:5]):
        dtype = 'bf16' if i < 3 else 'f32'
        shape = (1, h*d) if i < 3 else (1, h)
        g = VectorGraph(graph, f'producer{i}', [(value.reshape(shape), dtype)])
        produced.append(g.finish(g.add(0, kind='multiply', immediate=1, dtype=dtype)))
    result = graph.delta(*produced, values[5].reshape(h*d, d),
        SimpleNamespace(num_value_heads=h, key_head_dim=d, value_head_dim=d))
    logical = graph.write(out / 'logical', result)
    kernel, binary = out / 'graph.mlir', out / 'graph.ftlpu'
    commands = [
        [a.graph_compile, '--input', logical, '--output', kernel, '--emit-kernel'],
        [a.compile, '--input', kernel, '--input-stage', 'kernel', '--output', binary,
         '--mxm-execution', 'vector'],
    ]
    for i, command in enumerate(commands):
        with (out / f'compile{i}.log').open('w') as log:
            subprocess.run(list(map(str, command + ['--target-config', a.target_config.resolve()])),
                stdout=log, stderr=subprocess.STDOUT, check=True)
        if i == 0:
            kernel.write_text(kernel.read_text().replace('module {',
                'module attributes {ftlpu.vector_bf16_spill = true} {', 1))
    with (out / 'execution.log').open('w') as log:
        subprocess.run(list(map(str, [a.runner, binary, out/'inputs.txt', out/'actual'])),
            stdout=log, stderr=subprocess.STDOUT, check=True)
    actual = [(np.fromfile(out/'actual.0.bin', '<u2').astype('u4') << 16).view('f4').reshape(h,d),
        np.fromfile(out/'actual.1.bin', '<f4').reshape(h,d,d)]
    expected = reload_reference(*values)
    errors = [float(np.linalg.norm(x-y)/np.linalg.norm(y)) for x,y in zip(actual,expected)]
    assert errors[0] < .0002 and errors[1] < .00001, errors
    (out/'summary.json').write_text(json.dumps(dict(status='passed', reload_errors=errors), indent=2))
    print('passed: packed BF16/FP32 producers feed both decode hemispheres', errors)


if __name__ == '__main__':
    main()
