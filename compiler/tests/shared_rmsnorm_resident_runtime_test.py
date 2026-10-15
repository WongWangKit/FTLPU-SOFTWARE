#!/usr/bin/env python3
"""Check shared RMSNorm layout edges in a continuous native ICU program."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from qwen3_5_compiled_layer import VectorGraph, bf16
from qwen3_5_resident_layer import ResidentGraph
from resident_schedule import ResidentLinker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('opt', 'compile', 'runner', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--hidden-size', type=int, default=1024)
    parser.add_argument('--target-config', type=Path)
    parser.add_argument('--direct-initial-input', action='store_true')
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(35102)
    x = bf16(rng.normal(0, .3, (32, args.hidden_size)).astype(np.float32))
    gamma0 = bf16(rng.uniform(.7, 1.3, args.hidden_size).astype(np.float32))
    gamma1 = bf16(rng.uniform(.7, 1.3, args.hidden_size).astype(np.float32))
    graph = ResidentGraph(args.opt.resolve(), out / 'kernels', args.target_config)
    first = graph.norm('input_norm', x, gamma0, 1e-6)
    second = graph.norm('post_attention_norm', first, gamma1, 1e-5)
    pointwise = VectorGraph(graph, 'residual', [(first, 'bf16'), (second, 'bf16')])
    result = pointwise.finish(pointwise.add(0, 1, kind='add', dtype='bf16'))
    linker = ResidentLinker(graph, [result], direct_initial_inputs=[(0,0)] if args.direct_initial_input else [])
    schedule = out / 'program.schedule.mlir'
    schedule.write_text(linker.build(), encoding='utf-8')
    binary = out / 'program.ftlpu'
    subprocess.run([str(args.compile.resolve()), '--input', str(schedule),
                    '--input-stage', 'verified-schedule', '--output', str(binary)], check=True)
    paths = []
    for index, (value, dtype) in enumerate(linker.inputs):
        path = out / f'input{index}.bin'
        if dtype == 'bf16':
            (bf16(value).view(np.uint32) >> 16).astype('<u2').tofile(path)
        else:
            np.asarray(value).astype('<f4').tofile(path)
        paths.append(str(path))
    listing = out / 'inputs.txt'
    listing.write_text('\n'.join(paths) + '\n', encoding='utf-8')
    env = os.environ.copy()
    env['FTLPU_FILE_RUNNER_MODEL_SESSION'] = '1'
    env.pop('FTLPU_SESSION_LINK_ONLY', None)
    execution = subprocess.run([str(args.runner.resolve()), str(binary), str(listing),
                                str(out / 'program'), str(out / 'runtime.pipeline.csv')],
                               env=env, capture_output=True, text=True)
    (out / 'execution.log').write_text(execution.stdout + execution.stderr, encoding='utf-8')
    if execution.returncode:
        raise RuntimeError(execution.stderr)
    def norm(value, gamma, eps):
        return bf16(value / np.sqrt(np.mean(value * value, axis=1, keepdims=True) + eps) * gamma)
    a = norm(x, gamma0, 1e-6)
    expected = bf16(a + norm(a, gamma1, 1e-5))
    actual = (np.fromfile(out / 'program.0.bin', '<u2').astype(np.uint32) << 16).view(np.float32).reshape(x.shape)
    error = float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))
    if not np.isfinite(actual).all() or error > .015:
        raise AssertionError(f'continuous shared RMSNorm NRMSE={error}')
    summary = dict(status='passed', nrmse=error, program_count=1,
                   host_intermediate_transfers=0, shared_rmsnorm_nodes=2,
                   input_shape=list(x.shape), epsilon=[1e-6, 1e-5])
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
