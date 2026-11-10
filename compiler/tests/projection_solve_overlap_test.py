"""Independent projection/FP32 solve overlap, through real raw-macro execution.

The graph specifies arithmetic and views only. C++ decides whether resources,
dependencies, queues and instruction memory permit overlapping the two kernels.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import re
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from qwen3_5_logical_graph import LogicalGraph
from qwen3_5_compiled_layer import bf16
from qwen3_5_decode_gdn_attention_test import run
from unit_lower_solve_runtime_test import source as solve_source


def merge(spans):
    result = []
    for lo, hi in sorted(spans):
        if result and lo <= result[-1][1]:
            result[-1] = (result[-1][0], max(hi, result[-1][1]))
        else:
            result.append((lo, hi))
    return result


def intersection(a, b):
    i = j = total = 0
    while i < len(a) and j < len(b):
        total += max(0, min(a[i][1], b[j][1]) - max(a[i][0], b[j][0]))
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return total


def trace_metrics(path):
    with path.open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    def spans(predicate):
        found = []
        for row in rows:
            if not predicate(row):
                continue
            for outer in range(int(row['outer_count'])):
                for inner in range(int(row['inner_count'])):
                    if outer == inner == 0 and int(row['skip_first']):
                        continue
                    offset = outer * int(row['outer_interval']) + inner * int(row['inner_interval'])
                    found.append((int(row['start']) + offset, int(row['end']) + offset))
        return merge(found)
    mxm = spans(lambda r: r['resource'].startswith('MXM.') and r['resource'].endswith('.Compute'))
    vxm = spans(lambda r: r['resource'].startswith('VXM.')
                and re.match(r'^(mul|add|sub|multiply|subtract|negate)\b', r['detail']))
    invocation = next(r for r in rows if r['resource'] == 'Session.Invocation')
    host_inputs = [row for row in rows if row['resource'] == 'C2C.HostInput']
    input_spans = spans(lambda r: r['resource'] == 'C2C.HostInput')
    payloads = {}
    for row in host_inputs:
        binding = re.search(r'\bbinding=(\d+)\b', row['detail'])
        size = re.search(r'\bbytes=(\d+)\b', row['detail'])
        assert binding and size, row
        key, value = int(binding[1]), int(size[1])
        assert key not in payloads or payloads[key] == value, row
        payloads[key] = value
    # HostInput's bytes field is the binding payload, not physical DDR bytes.
    # A bank-striped upload can emit two events carrying the same full payload
    # size, so count unique bindings instead of summing their event labels.
    cold_start = min([int(invocation['start'])] + [int(r['start']) for r in host_inputs])
    return dict(mxm_vxm_overlap_cycles=intersection(mxm, vxm),
                invocation_cycles=int(invocation['end']) - int(invocation['start']),
                host_input_payload_bytes=sum(payloads.values()),
                host_input_payload_bytes_scope='unique uploaded bindings; excludes physical layout mirroring/padding and dynamic weight pages',
                host_input_cycles=sum(hi-lo for lo, hi in input_spans),
                host_input_transfer_events=len(host_inputs),
                cold_start_cycle=cold_start,
                cold_cycles_through_compute=int(invocation['end']) - cold_start,
                diagnostic_host_readback_cycles=sum(int(r['end']) - int(r['start'])
                    for r in rows if r['resource'] == 'C2C.HostOutput'))


def truncate(x):
    return (x.astype('f4').view('u4') & np.uint32(0xffff0000)).view('f4')


def relative_error(actual, expected):
    return float(np.linalg.norm(actual.astype('f8') - expected) / max(np.linalg.norm(expected), 1e-30))


def execute(args, heads, dim, dependent, enabled):
    label = f'h{heads}_d{dim}_' + ('dependent' if dependent else 'independent')
    out = args.output_dir.resolve() / label / ('on' if enabled else 'off')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'summary.json').write_text(json.dumps(dict(status='pending'), indent=2))
    exe = lambda area, name: args.build_dir.resolve() / area / (name + ('.exe' if sys.platform == 'win32' else ''))
    rng = np.random.default_rng(35261105 + heads + dim)
    lower = np.tril(rng.normal(0, .025, (heads, 32, 32)).astype('f4'), -1)
    rhs = rng.normal(0, .3, (heads, 32, dim)).astype('f4')
    # Values retain low FP32 mantissa bits; row zero must be copied exactly.
    rhs[0, 0] = -1 - np.arange(1, dim + 1, dtype='f4') * 2**-20
    x = bf16(rng.normal(0, .2, (32, 128)).astype('f4'))
    width = heads * dim
    weight = rng.integers(-16, 17, (width, width if dependent else 128), dtype=np.int8)
    scale = .01
    graph = LogicalGraph()
    solved = graph.run('triangular', solve_source(heads, dim),
        [(lower.reshape(-1, 32), 'f32'), (rhs.reshape(-1, dim), 'f32')],
        [((heads * 32, dim), 'f32')])[0]
    solved_tokens = solved.reshape(heads, 32, dim).transpose(1, 0, 2).reshape(32, width)
    narrowed = graph.truncate_bf16('solve_high_planes', solved_tokens)
    projected = graph.project('linear', narrowed if dependent else x, weight, scale)
    joined = graph.vector_tiles('joined', [(narrowed, 'bf16'), (projected, 'bf16')],
        lambda v: v.add(0, 1, dtype='bf16'))
    logical = graph.write(out / 'logical_graph', [solved, projected, joined])
    kernel, binary = out / 'kernel.mlir', out / 'program.ftlpu'
    run([exe('compiler', 'ftlpu-graph-compile'), '--input', logical, '--output', kernel,
         '--emit-kernel', '--target-config', args.target_config.resolve()], out / 'import.log')
    attributes = ('ftlpu.sxm_wide = true, ftlpu.common_barrier_protocol = true, '
        'ftlpu.vector_fp32_ingress = true, ftlpu.vector_bf16_spill = true, '
        f'ftlpu.projection_solve_overlap = {str(enabled).lower()}')
    kernel.write_text(kernel.read_text().replace('module {', f'module attributes {{{attributes}}} {{', 1))
    compiler = args.compiler_executable.resolve() if args.compiler_executable else exe('compiler', 'ftlpu-compile')
    run([compiler, '--input', kernel, '--input-stage', 'kernel', '--output', binary,
         '--target-config', args.target_config.resolve(), '--icu-compression', 'raw-macro',
         '--mxm-execution', 'ws'], out / 'compile.log')
    os.environ.update(FTLPU_FILE_RUNNER_MODEL_SESSION='1', FTLPU_FILE_RUNNER_SXM_ARCHITECTURE='wide')
    run([exe('runtime', 'compiled_stage_file_runner'), binary, out / 'inputs.txt',
         out / 'actual', out / 'runtime.pipeline.csv'], out / 'execution.log')
    actual_d = np.fromfile(out / 'actual.0.bin', '<f4').reshape(heads, 32, dim)
    read_bf16 = lambda index: (np.fromfile(out / f'actual.{index}.bin', '<u2').astype('u4') << 16).view('f4').reshape(32, width)
    actual_z, actual_tail = read_bf16(1), read_bf16(2)
    reference_d = np.linalg.solve(np.eye(32, dtype='f8')[None] + lower.astype('f8'), rhs.astype('f8'))
    np.testing.assert_allclose(actual_d, reference_d, rtol=2e-5, atol=5e-5)
    np.testing.assert_array_equal(actual_d[:, 0], rhs[:, 0])
    d_tokens = truncate(actual_d.transpose(1, 0, 2).reshape(32, width))
    reference_z = (d_tokens if dependent else x).astype('f8') @ (weight.astype('f8') * scale).T
    projection_error = relative_error(actual_z, reference_z)
    tail_error = relative_error(actual_tail, d_tokens.astype('f8') * actual_z)
    assert np.isfinite(actual_z).all() and projection_error < .02, projection_error
    assert np.isfinite(actual_tail).all() and tail_error < .01, tail_error
    log = (out / 'execution.log').read_text()
    stats = {key: int(value) for key, value in re.findall(r'(\w+)=(\d+)', log.splitlines()[-1])}
    assert stats.get('host_operations') == 0, stats
    plan = json.loads((out / 'graph_plan.json').read_text())
    metrics = trace_metrics(out / 'runtime.pipeline.csv')
    run([exe('runtime', 'ftlpu_icu_program_export'), out / 'actual.runtime.ftlpu',
         out / 'runtime_icu_programs', '--pre-execution-dir', out / 'actual.pre_execution'], out / 'export.log')
    result = dict(status='numerics_passed', heads=heads, dim=dim, projection_depends_on_solve=dependent,
        projection_solve_overlap='on' if enabled else 'off', icu_compression='raw-macro',
        precision='FP32 solve storage/arithmetic; explicit BF16 high-plane tail view; int8 projection weight',
        solve_max_abs=float(np.max(np.abs(actual_d-reference_d))), projection_nrmse=projection_error,
        joined_nrmse=tail_error, runtime=stats, scheduling=plan.get('projection_solve_overlap', []), **metrics)
    (out / 'summary.json').write_text(json.dumps(result, indent=2))
    return out, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('build-dir', 'target-config', 'output-dir'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--compiler-executable', type=Path)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / 'summary.json').write_text(json.dumps(dict(status='pending'), indent=2))
    checks = []
    completed_runs = []
    for heads, dim in ((2, 32), (4, 64)):
        for dependent in (False, True):
            before, old = execute(args, heads, dim, dependent, False)
            after, new = execute(args, heads, dim, dependent, True)
            for index in range(3):
                name = f'actual.{index}.bin'
                assert (before / name).read_bytes() == (after / name).read_bytes(), (heads, dim, dependent, name)
            assert old['mxm_vxm_overlap_cycles'] == 0, old
            assert old['runtime']['weight_page_prefetch_bytes'] == new['runtime']['weight_page_prefetch_bytes']
            if dependent:
                assert new['mxm_vxm_overlap_cycles'] == 0, new
                assert new['runtime']['cycles'] == old['runtime']['cycles'], (old, new)
                assert new['cold_cycles_through_compute'] == old['cold_cycles_through_compute'], (old, new)
            else:
                assert new['mxm_vxm_overlap_cycles'] > 0, new
                assert new['runtime']['cycles'] < old['runtime']['cycles'], (old, new)
                assert new['invocation_cycles'] < old['invocation_cycles'], (old, new)
                assert new['cold_cycles_through_compute'] < old['cold_cycles_through_compute'], (old, new)
            checks.append(dict(heads=heads, dim=dim, dependent=dependent, before=old, after=new,
                               all_outputs_bit_identical=True))
            completed_runs.extend(((before, old), (after, new)))
    # No run is fully accepted until every numerical, dependency, actual-overlap
    # and complete-time comparison in the suite has succeeded.
    for directory, result in completed_runs:
        result['status'] = 'passed'
        (directory / 'summary.json').write_text(json.dumps(result, indent=2))
    summary = dict(status='passed', scope='generic linear projection and FP32 triangular solve with joined tail', checks=checks)
    (args.output_dir.resolve() / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
