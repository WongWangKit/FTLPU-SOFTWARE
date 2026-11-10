"""Multi-token-tile int8/BF16 projection through real raw-macro execution.

Checks distinct token/ACC blocks, continuous MXM issue for even K tiles, and
the unchanged odd-K fallback. Python provides only graph, data and references.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from qwen3_5_compiled_layer import bf16
from qwen3_5_logical_graph import LogicalGraph


def run(command, log):
    with log.open('w', encoding='utf-8') as stream:
        subprocess.run([str(value) for value in command], stdout=stream,
                       stderr=subprocess.STDOUT, check=True)


def merge(spans):
    result = []
    for lo, hi in sorted(spans):
        if result and lo <= result[-1][1]:
            result[-1] = (result[-1][0], max(hi, result[-1][1]))
        else:
            result.append((lo, hi))
    return result


def intersection_cycles(left, right):
    i = j = total = 0
    while i < len(left) and j < len(right):
        total += max(0, min(left[i][1], right[j][1]) - max(left[i][0], right[j][0]))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return total


def pipeline_metrics(path):
    with path.open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    activity = {}
    for row in rows:
        resource = row['resource']
        if not (resource.startswith('MXM.') and resource.endswith(('.Compute', '.Load'))):
            continue
        for outer in range(int(row['outer_count'])):
            for inner in range(int(row['inner_count'])):
                if outer == inner == 0 and int(row['skip_first']):
                    continue
                offset = outer * int(row['outer_interval']) + inner * int(row['inner_interval'])
                activity.setdefault(resource, []).append(
                    (int(row['start']) + offset, int(row['end']) + offset))
    spans = {resource: merge(intervals) for resource, intervals in activity.items()}
    compute = {resource: intervals for resource, intervals in spans.items()
               if resource.endswith('.Compute')}
    loads = {resource: intervals for resource, intervals in spans.items()
             if resource.endswith('.Load')}
    overlap = {resource.removesuffix('.Compute'): intersection_cycles(
        intervals, loads.get(resource.removesuffix('.Compute') + '.Load', []))
        for resource, intervals in compute.items()}
    invocation = next(row for row in rows if row['resource'] == 'Session.Invocation')
    ingress = [int(row['start']) for row in rows if row['resource'] == 'C2C.HostInput']
    return dict(compute_spans=compute, weight_load_spans=loads,
                weight_load_compute_overlap_cycles=overlap,
                invocation_cycles=int(invocation['end']) - int(invocation['start']),
                cold_cycles_through_compute=int(invocation['end']) - min(
                    [int(invocation['start'])] + ingress),
                diagnostic_host_readback_cycles=sum(int(row['end']) - int(row['start'])
                    for row in rows if row['resource'] == 'C2C.HostOutput'))


def execute(args, shape, compiler, label, *, expected_continuous=None,
            overlap_disabled=False):
    m, n, k = shape
    out = args.output_dir.resolve() / f'm{m}_n{n}_k{k}' / label
    out.mkdir(parents=True, exist_ok=True)
    (out / 'summary.json').write_text(json.dumps(dict(status='pending')))
    suffix = '.exe' if sys.platform == 'win32' else ''
    exe = lambda area, name: args.build_dir.resolve() / area / (name + suffix)
    rng = np.random.default_rng(353264 + m + n + k)
    x = bf16(rng.normal(0, .2, (m, k)).astype('f4'))
    # Every token tile has different values; accidental ACC row wrapping must
    # fail numerical validation even when the projection has several N groups.
    x = bf16(x + (np.arange(m, dtype='f4')[:, None] // 32) * .125)
    weight = rng.integers(-16, 17, (n, k), dtype=np.int8)
    scale = .01
    graph = LogicalGraph()
    result = graph.project('projection', x, weight, scale)
    logical = graph.write(out / 'logical_graph', [result])
    kernel, binary = out / 'kernel.mlir', out / 'program.ftlpu'
    run([exe('compiler', 'ftlpu-graph-compile'), '--input', logical,
         '--output', kernel, '--emit-kernel', '--target-config', args.target_config.resolve()],
        out / 'import.log')
    kernel.write_text(kernel.read_text().replace('module {',
        'module attributes {ftlpu.common_barrier_protocol = true} {', 1))
    run([compiler, '--input', kernel, '--input-stage', 'kernel', '--output', binary,
         '--target-config', args.target_config.resolve(), '--icu-compression', 'raw-macro',
         '--mxm-execution', 'ws'], out / 'compile.log')
    run([exe('runtime', 'compiled_stage_file_runner'), binary, out / 'inputs.txt',
         out / 'actual', out / 'runtime.pipeline.csv'], out / 'execution.log')
    actual = (np.fromfile(out / 'actual.0.bin', '<u2').astype('u4') << 16).view('f4').reshape(m, n)
    reference = x.astype('f8') @ (weight.astype('f8') * scale).T
    error = float(np.linalg.norm(actual.astype('f8') - reference) / np.linalg.norm(reference))
    assert np.isfinite(actual).all() and error < .02, (shape, error)
    stats = {key: int(value) for key, value in re.findall(r'(\w+)=(\d+)',
        (out / 'execution.log').read_text().splitlines()[-1])}
    assert stats.get('host_operations') == 0, stats
    metrics = pipeline_metrics(out / 'runtime.pipeline.csv')
    assert len(metrics['compute_spans']) == 2, metrics
    continuous = ((k // 32) % 2 == 0 if expected_continuous is None
                  else expected_continuous)
    if label == 'current' and continuous:
        active_cycles = m * (k // 32) * (n // 64)
        for resource, intervals in metrics['compute_spans'].items():
            assert len(intervals) == 1, (shape, resource, intervals)
            assert intervals[0][1] - intervals[0][0] == active_cycles, (
                shape, resource, intervals, active_cycles)
    if overlap_disabled:
        assert len(metrics['weight_load_spans']) == 2, metrics
        assert all(cycles == 0 for cycles in metrics['weight_load_compute_overlap_cycles'].values()), metrics
        assert all(len(intervals) > 1 for intervals in metrics['compute_spans'].values()), metrics
    run([exe('runtime', 'ftlpu_icu_program_export'), out / 'actual.runtime.ftlpu',
         out / 'runtime_icu_programs', '--pre-execution-dir', out / 'actual.pre_execution'],
        out / 'export.log')
    report = dict(status='numerics_passed', shape=dict(m=m, n=n, k=k),
                  precision='int8 weight; BF16 activation/dequant/result; FP32 ACC',
                  icu_compression='raw-macro', expected_continuous_compute=continuous,
                  weight_activation_overlap_disabled=overlap_disabled,
                  nrmse=error, runtime=stats, **metrics)
    (out / 'summary.json').write_text(json.dumps(report, indent=2))
    return out, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('build-dir', 'target-config', 'output-dir'):
        parser.add_argument('--' + option, type=Path, required=True)
    parser.add_argument('--compiler-executable', type=Path)
    parser.add_argument('--baseline-compiler', type=Path,
        help='Optional pre-change compiler; require byte-identical results and unchanged fallback timing.')
    parser.add_argument('--shapes', default='32x128x128,64x128x128,96x192x128,128x128x128,64x128x96,64x128x32',
        help='Comma-separated MxNxK shapes, within the supplied target ACC capacity.')
    parser.add_argument('--only-overlap-disabled', action='store_true',
        help='Run only M64/N128/K128 with the target weight/activation overlap capability disabled.')
    args = parser.parse_args()
    args.output_dir.resolve().mkdir(parents=True, exist_ok=True)
    summary_file = args.output_dir.resolve() / 'summary.json'
    summary_file.write_text(json.dumps(dict(status='pending')))
    os.environ['FTLPU_FILE_RUNNER_MODEL_SESSION'] = '1'
    suffix = '.exe' if sys.platform == 'win32' else ''
    compiler = (args.compiler_executable.resolve() if args.compiler_executable else
        args.build_dir.resolve() / 'compiler' / ('ftlpu-compile' + suffix))
    completed, checks = [], []
    for shape_string in ([] if args.only_overlap_disabled else args.shapes.split(',')):
        shape = tuple(int(value) for value in shape_string.split('x'))
        assert len(shape) == 3, shape
        out, report = execute(args, shape, compiler, 'current')
        completed.append((out, report))
        check = dict(current=report)
        if args.baseline_compiler:
            before, baseline = execute(args, shape, args.baseline_compiler.resolve(), 'baseline')
            completed.append((before, baseline))
            assert (before / 'actual.0.bin').read_bytes() == (out / 'actual.0.bin').read_bytes(), shape
            assert baseline['runtime']['weight_page_prefetch_bytes'] == report['runtime']['weight_page_prefetch_bytes']
            if shape[0] > 32 and report['expected_continuous_compute']:
                assert report['cold_cycles_through_compute'] < baseline['cold_cycles_through_compute'], shape
            else:
                assert report['runtime']['cycles'] == baseline['runtime']['cycles'], shape
            check.update(baseline=baseline, output_bit_identical=True)
        checks.append(check)
        print(json.dumps(dict(shape=shape, nrmse=report['nrmse'], runtime_cycles=report['runtime']['cycles'])))
    # Disable an existing capability in a local target copy; preserve every
    # hardware size, port, delay and numerical criterion from the input target.
    target_config = json.loads(args.target_config.resolve().read_text())
    target_config.setdefault('throughput', {})['mxm_weight_activation_overlap_enabled'] = 0
    disabled_config = args.output_dir.resolve() / 'target_overlap_disabled.json'
    disabled_config.write_text(json.dumps(target_config, indent=2))
    disabled_args = argparse.Namespace(**vars(args))
    disabled_args.target_config = disabled_config
    disabled_args.output_dir = args.output_dir.resolve() / 'overlap_disabled'
    disabled_out, disabled = execute(disabled_args, (64, 128, 128), compiler,
        'current', expected_continuous=False, overlap_disabled=True)
    completed.append((disabled_out, disabled))
    enabled_output = args.output_dir.resolve() / 'm64_n128_k128' / 'current' / 'actual.0.bin'
    disabled_check = dict(target_weight_activation_overlap_enabled=False, current=disabled)
    if not args.only_overlap_disabled and '64x128x128' in args.shapes.split(','):
        assert enabled_output.read_bytes() == (disabled_out / 'actual.0.bin').read_bytes()
        disabled_check['output_bit_identical_to_continuous'] = True
    checks.append(disabled_check)
    print(json.dumps(dict(shape=(64, 128, 128), overlap_disabled=True,
        nrmse=disabled['nrmse'], runtime_cycles=disabled['runtime']['cycles'])))
    for directory, report in completed:
        report['status'] = 'passed'
        (directory / 'summary.json').write_text(json.dumps(report, indent=2))
    summary_file.write_text(json.dumps(dict(status='passed',
        scope='standalone linear projections; no attention, decoder or persistent state', checks=checks), indent=2))


if __name__ == '__main__':
    main()
