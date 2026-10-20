#!/usr/bin/env python3
"""Retained residual and norm must survive native FFN bank ping-pong pages."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from qwen3_5_compiled_layer import bf16
from qwen3_5_logical_graph import LogicalGraph
from qwen3_5_reference import Qwen35DeltaConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('graph-compile', 'compile', 'runner', 'target-config', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    # These dimensions fill both banks' weight slice groups. Mini FFNs do not
    # expose an omitted bank1 page or a residual placed in its recyclable rows.
    rng = np.random.default_rng(351015)
    x = bf16(rng.normal(0, .2, (32, 1024)).astype(np.float32))
    delta = bf16(rng.normal(0, .05, x.shape).astype(np.float32))
    gamma = bf16(rng.normal(1, .05, 1024).astype(np.float32))
    packed, scales, weights = {}, {}, {}
    for name, shape in [('gate_proj', (3584, 1024)),
                        ('up_proj', (3584, 1024)), ('down_proj', (1024, 3584))]:
        weight = rng.normal(0, .02, shape).astype(np.float32)
        scales[name] = float(np.max(np.abs(weight)) / 127)
        packed[name] = np.rint(weight / scales[name]).astype(np.int8)
        weights[name] = packed[name].astype(np.float32) * scales[name]

    graph = LogicalGraph()
    # Retain a real-size state reservation across FFN. This forces Down's
    # result away from the standalone row-zero layout, exercising relocation.
    config = Qwen35DeltaConfig(1024, 3584, 16, 16, 128, 128)
    zeros = np.zeros((1, 2048), np.float32)
    _, state = graph.delta(zeros, zeros, zeros, np.ones((1, 16), np.float32),
        np.zeros((1, 16), np.float32), np.zeros((2048, 128), np.float32), config)
    residual = graph.vector_tiles('residual', [(x, 'bf16'), (delta, 'bf16')],
        lambda g: g.add(0, 1, kind='add', dtype='bf16'))
    norm = graph.norm('norm', residual, gamma, 1e-6)
    down = graph.ffn(norm, packed, scales)
    output = graph.vector_tiles('output', [(residual, 'bf16'), (down, 'bf16')],
        lambda g: g.add(0, 1, kind='add', dtype='bf16'))
    logical = graph.write(out / 'logical', [state, output, down, norm, residual])
    subprocess.run([str(args.graph_compile.resolve()), '--input', str(logical),
        '--output', str(out / 'layer.kernel.mlir'), '--emit-kernel'], check=True)
    subprocess.run([str(args.compile.resolve()), '--input', str(out / 'layer.kernel.mlir'),
        '--input-stage', 'kernel', '--output', str(out / 'layer.ftlpu'),
        '--mxm-execution', 'vector', '--ffn-schedule', 'fused',
        '--target-config', str(args.target_config.resolve())], check=True)
    plan = json.loads((out / 'graph_plan.json').read_text())
    ffns = [node for node in plan['nodes'] if node.get('native_ffn')]
    assert len(ffns) == 1
    assert ffns[0]['direct_output_layouts'][0]['base_row'] > 0, \
        'test must exercise FFN output at a nonzero physical base row'
    pages = ffns[0]['weight_page_workspaces']
    assert {p['bank'] for p in pages} == {0, 1}
    for node in plan['nodes']:
        if node['name'] not in ('residual', 'norm'):
            continue
        for value in node['direct_output_layouts']:
            for page in pages:
                same_side = ('both' in (page['hemisphere'], value['hemisphere'])
                             or page['hemisphere'] == value['hemisphere'])
                same_slice = bool(set(page['slices']) & set(value['slices']))
                same_rows = (page['base_row'] < value['base_row'] + value['rows']
                             and value['base_row'] < page['base_row'] + page['rows'])
                assert not (same_side and same_slice and same_rows
                            and page['bank'] == value['bank']), (node['name'], value, page)
    subprocess.run([str(args.runner.resolve()), str(out / 'layer.ftlpu'),
        str(out / 'inputs.txt'), str(out / 'actual')], check=True)
    assert not np.any(np.fromfile(out / 'actual.0.bin', '<f4')), 'retained state was overwritten'
    residual_ref = bf16(x + delta)
    norm_ref = residual_ref / np.sqrt(np.mean(residual_ref ** 2, axis=-1, keepdims=True) + 1e-6) * gamma
    gate = norm_ref @ weights['gate_proj'].T
    up = norm_ref @ weights['up_proj'].T
    down_ref = ((gate / (1 + np.exp(-gate))) * up) @ weights['down_proj'].T
    for i, (name, expected) in enumerate((('output', residual_ref + down_ref),
            ('down', down_ref), ('norm', norm_ref), ('residual', residual_ref))):
        actual = (np.fromfile(out / f'actual.{i + 1}.bin', '<u2').astype(np.uint32) << 16).view(np.float32).reshape(x.shape)
        error = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
        print(f'{name}: nrmse={error:.7f}', flush=True)
        assert np.isfinite(actual).all() and error < .03, (name, error)


if __name__ == '__main__':
    main()
