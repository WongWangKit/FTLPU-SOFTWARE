"""Chunk32 block16 forward solve: real VXM FP32 and one BF16 MXM product."""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from qwen3_5_logical_graph import LogicalGraph
from qwen3_5_decode_gdn_attention_test import run
from unit_lower_solve_runtime_test import source


def audit_blocked_solve(out, heads, dim, cycle_range=None):
    """Require real bilateral MXM execution and useful FP32 diagonal work."""
    active = {}
    with (out / 'runtime.pipeline.csv').open(newline='', encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            resource = row['resource']
            if not ((resource.startswith('MXM.') and resource.endswith('.Compute')) or
                    (resource in ('VXM.C0', 'VXM.C15') and
                     row['detail'].startswith(('mul chain_depth=4', 'sub chain_depth=4')))):
                continue
            for outer in range(int(row['outer_count'])):
                for inner in range(int(row['inner_count'])):
                    if outer == inner == 0 and int(row['skip_first']):
                        continue
                    shift = outer * int(row['outer_interval']) + inner * int(row['inner_interval'])
                    lo, hi = int(row['start']) + shift, int(row['end']) + shift
                    if cycle_range:
                        lo, hi = max(lo, cycle_range[0]), min(hi, cycle_range[1])
                    if lo < hi:
                        active[resource] = active.get(resource, 0) + hi - lo
    mxm = {k: v for k, v in active.items() if k.startswith('MXM.')}
    assert len(mxm) == 2 and all(v == heads // 2 * (dim // 32) * 16 for v in mxm.values()), mxm
    # Include the initial zero-padding vectors (16 per head/value tile).
    expected = heads // 2 * (dim // 32) * (2 * sum(range(1, 17)) + 16)
    assert active.get('VXM.C0') == expected, (active, expected)
    return dict(mxm_active_cycles=mxm,paired_c0_active_cycles=active['VXM.C0'],
                mxm_products_per_head=dim // 32,weight_load_mode='1vector',
                block_size=16,k_padding=32,useful_k=16,fp32_rhs_subtract_fused=True)


def bf16_trunc(x):
    return (np.ascontiguousarray(x, dtype='f4').view('u4') & np.uint32(0xffff0000)).view('f4')


def mixed_reference(lower, rhs):
    """CPU arithmetic oracle only; never feeds any intermediate to CModel."""
    result = np.zeros_like(rhs)
    remaining = rhs.copy()
    for lo in (0, 16):
        for i in range(lo, lo + 16):
            value = remaining[:, i].copy()
            for j in range(lo, i):
                value -= lower[:, i, j, None] * result[:, j]
            result[:, i] = value
        if lo == 0:
            a, b = bf16_trunc(lower[:, 16:, :16]), bf16_trunc(result[:, :16])
            product = np.zeros_like(remaining[:, 16:])
            for k in range(16):
                product += a[:, :, k, None] * b[:, None, k, :]
            remaining[:, 16:] -= product
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('build-dir', 'target-config', 'output-dir'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--heads', type=int, default=2)
    parser.add_argument('--dim', type=int, default=32)
    parser.add_argument('--fixture-dir', type=Path)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    build = args.build_dir.resolve()
    exe = lambda area, name: build / area / (name + ('.exe' if sys.platform == 'win32' else ''))
    heads, dim = args.heads, args.dim
    rng = np.random.default_rng(35261103)
    lower = np.tril(rng.normal(0, .05, (heads, 32, 32)).astype('f4'), -1)
    rhs = rng.normal(size=(heads, 32, dim)).astype('f4')
    if args.fixture_dir:
        lower = np.fromfile(args.fixture_dir / 'actual.8.bin', '<f4').reshape(heads, 32, 32)
        rhs = np.fromfile(args.fixture_dir / 'actual.9.bin', '<f4').reshape(heads, 32, dim)
    graph = LogicalGraph()
    result = graph.run('solve', source(heads, dim, 'unit_lower_solve_blocked'),
        [(lower.reshape(-1, 32), 'f32'), (rhs.reshape(-1, dim), 'f32')], [((heads * 32, dim), 'f32')])[0]
    logical = graph.write(out / 'logical', [result])
    kernel, binary = out / 'kernel.mlir', out / 'program.ftlpu'
    run([exe('compiler', 'ftlpu-graph-compile'), '--input', logical, '--output', kernel,
         '--emit-kernel', '--target-config', args.target_config.resolve()], out / 'import.log')
    kernel.write_text(kernel.read_text().replace('module {',
        'module attributes {ftlpu.sxm_wide = true, ftlpu.common_barrier_protocol = true, '
        'ftlpu.common_transport_overlap = false, ftlpu.vector_bf16_spill = true, '
        'ftlpu.vector_fp32_ingress = true} {', 1))
    run([exe('compiler', 'ftlpu-compile'), '--input', kernel, '--input-stage', 'kernel',
         '--output', binary, '--target-config', args.target_config.resolve(),
         '--icu-compression', 'raw-macro', '--mxm-execution', 'ws'], out / 'compile.log')
    os.environ['FTLPU_FILE_RUNNER_MODEL_SESSION'] = '1'
    os.environ['FTLPU_FILE_RUNNER_SXM_ARCHITECTURE'] = 'wide'
    run([exe('runtime', 'compiled_stage_file_runner'), binary, out / 'inputs.txt',
         out / 'actual', out / 'runtime.pipeline.csv'], out / 'execution.log')
    actual = np.fromfile(out / 'actual.0.bin', '<f4').reshape(heads, 32, dim)
    expected = np.linalg.solve(np.eye(32)[None] + lower.astype('f8'), rhs.astype('f8'))
    mixed = mixed_reference(lower, rhs)
    local_error = float(np.linalg.norm(actual - mixed) / np.linalg.norm(mixed))
    error = float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))
    residual = float(np.linalg.norm((np.eye(32)[None] + lower) @ actual - rhs) / np.linalg.norm(rhs))
    # Verify the block with no low-precision update retains complete FP32.
    np.testing.assert_allclose(actual[:, :16], expected[:, :16], rtol=2e-5, atol=2e-6)
    np.testing.assert_array_equal(actual[:, 0], rhs[:, 0])
    assert local_error < 2e-5, local_error
    assert error < .01 and residual < .01, (error, residual)
    summary = dict(status='passed', heads=heads, dim=dim, nrmse=error,
        relative_residual=residual, mixed_oracle_nrmse=local_error,
        precision='VXM FP32; one truncated BF16 MXM product, FP32 ACC',
        scheduling=audit_blocked_solve(out, heads, dim))
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
