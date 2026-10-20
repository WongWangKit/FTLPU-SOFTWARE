#!/usr/bin/env python3
"""Full-width FFN: let the compiler choose SRAM batches, then check CModel."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from qwen3_5_logical_graph import LogicalGraph
from qwen3_5_compiled_layer import bf16


def check_retained_sum_workspaces(plan):
    """A partial Down sum must survive the following FFN's SRAM writes."""
    nodes = plan['nodes']
    for i, node in enumerate(nodes):
        if '.sum' not in node['name']:
            continue
        following = next((n for n in nodes[i + 1:] if n.get('native_ffn')), None)
        if following is None:
            continue
        assert following['temporary_workspaces'], 'missing FFN temporary lifetimes'
        for value in node['direct_output_layouts']:
            for scratch in following['temporary_workspaces'] + following['weight_page_workspaces']:
                same_side = ('both' in (value['hemisphere'], scratch['hemisphere'])
                             or value['hemisphere'] == scratch['hemisphere'])
                overlaps = (same_side and value['bank'] == scratch['bank']
                            and bool(set(value['slices']) & set(scratch['slices']))
                            and value['base_row'] < scratch['base_row'] + scratch['rows']
                            and scratch['base_row'] < value['base_row'] + value['rows'])
                assert not overlaps, (node['name'], following['name'], value, scratch)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('graph-compile', 'compile', 'runner', 'target-config', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--hidden', type=int, default=8960)
    parser.add_argument('--max-nrmse', type=float, default=.01)
    parser.add_argument('--with-rmsnorm', action='store_true',
                        help='Execute RMSNorm on the LPU before the batched FFN')
    parser.add_argument('--compile-only', action='store_true')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(161019)
    x = bf16(rng.normal(0, .2, (32, 1536)).astype(np.float32))
    gamma = bf16(rng.normal(1, .02, 1536).astype(np.float32))
    packed = {name: rng.integers(-8, 9, shape, dtype=np.int8) for name, shape in (
        ('gate_proj', (args.hidden, 1536)), ('up_proj', (args.hidden, 1536)),
        ('down_proj', (1536, args.hidden)))}
    scales = {name: .01 for name in packed}
    graph = LogicalGraph()
    norm = (graph.norm('norm', x, gamma, 1e-6) if args.with_rmsnorm else
            bf16(x / np.sqrt(np.mean(x*x, axis=1, keepdims=True) + 1e-6) * gamma))
    result = graph.ffn(norm, packed, scales)
    if not args.check_only:
        logical = graph.write(out / 'logical', [result, norm] if args.with_rmsnorm else [result])
        subprocess.run([str(args.graph_compile.resolve()), '--input', str(logical),
            '--output', str(out / 'ffn.kernel.mlir'), '--emit-kernel'], check=True)
        subprocess.run([str(args.compile.resolve()), '--input', str(out / 'ffn.kernel.mlir'),
            '--input-stage', 'kernel', '--output', str(out / 'ffn.ftlpu'),
            '--mxm-execution', 'auto', '--ffn-schedule', 'tail', '--icu-compression', 'macro',
            '--target-config', str(args.target_config.resolve())], check=True)
    plan = json.loads((out / 'graph_plan.json').read_text())
    batches = [node for node in plan['nodes'] if node.get('native_ffn')]
    print(f'Compiler selected {len(batches)} FFN batches', flush=True)
    check_retained_sum_workspaces(plan)
    if args.with_rmsnorm:
        assert plan['native_output_archive_copies'] == 1, 'expected one planned SRAM retention transfer'
    if args.compile_only:
        return
    if not args.check_only:
        subprocess.run([str(args.runner.resolve()), str(out / 'ffn.ftlpu'),
            str(out / 'inputs.txt'), str(out / 'actual')], check=True)
    norm_ref = bf16(x / np.sqrt(np.mean(x*x, axis=1, keepdims=True) + 1e-6) * gamma)
    gate = norm_ref @ (packed['gate_proj'].T.astype(np.float32) * .01)
    up = norm_ref @ (packed['up_proj'].T.astype(np.float32) * .01)
    expected = ((gate / (1 + np.exp(-gate))) * up) @ (packed['down_proj'].T.astype(np.float32) * .01)
    actual = (np.fromfile(out / 'actual.0.bin', '<u2').astype(np.uint32) << 16).view(np.float32).reshape(x.shape)
    nrmse = float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))
    block_error = np.linalg.norm((actual - expected).reshape(32, -1, 16), axis=-1)
    block_scale = np.linalg.norm(expected.reshape(32, -1, 16), axis=-1)
    worst_block = float(np.max(block_error / np.maximum(block_scale, 1e-8)))
    passed = bool(np.isfinite(actual).all() and nrmse < args.max_nrmse and worst_block < .05)
    norm_error = None
    if args.with_rmsnorm:
        norm_actual = (np.fromfile(out / 'actual.1.bin', '<u2').astype(np.uint32) << 16).view(np.float32).reshape(x.shape)
        norm_error = float(np.linalg.norm(norm_actual - norm_ref) / np.linalg.norm(norm_ref))
        passed &= bool(np.isfinite(norm_actual).all() and norm_error < .01)
    summary = dict(scope='rmsnorm_ffn' if args.with_rmsnorm else 'isolated_ffn',
                   weights='synthetic_nonzero_int8',
                   mxm_execution='auto', intermediate_storage='on_chip',
                   tokens=32, width=1536, hidden=args.hidden,
                   batches=len(batches), passed=passed,
                   max_nrmse=args.max_nrmse,
                   retained_norm_nrmse=norm_error,
                   worst_token_feature_block_nrmse=worst_block if np.isfinite(worst_block) else None,
                   nonfinite_outputs=int((~np.isfinite(actual)).sum()),
                   nrmse=nrmse if np.isfinite(nrmse) else None,
                   correlation=float(np.corrcoef(actual.ravel(), expected.ravel())[0, 1])
                       if np.isfinite(actual).all() else None)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    assert passed, 'on-chip FFN numerical regression failed'


if __name__ == '__main__':
    main()
