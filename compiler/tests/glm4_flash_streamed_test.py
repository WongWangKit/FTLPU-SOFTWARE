"""Compare real GLM decode attention executions at batch 1 or 32.

Only the new attention invocation is compiled and executed. Previous timings
come from their completed runtime traces. Projection, FFN and batch64 are not
part of this run. An optional frozen previous streamed result enables exact
output and same-input performance comparisons while updating the fixed folder.
"""
import argparse
import csv
import json
from pathlib import Path
import re

import numpy as np
from glm4_flash_packed_u_test import benchmark, graph_digest, phase_timing, sha


def covered_cycles(intervals):
    """Measure union of half-open fixed-FU spans, not the sum of their widths."""
    total, limit = 0, 0
    for start, end in sorted(intervals):
        total += max(0, end-max(start, limit))
        limit = max(end, limit)
    return total


def transpose_diagnostics(folder):
    spans = {'SXM.E.Transpose': [], 'SXM.W.Transpose': []}
    with (folder/'runtime.pipeline.csv').open(newline='') as stream:
        for row in csv.DictReader(stream):
            if row['resource'] not in spans:
                continue
            start, end = int(row['start']), int(row['end'])
            inner = int(row.get('inner_count') or 1)
            outer = int(row.get('outer_count') or 1)
            ii = int(row.get('inner_interval') or 0)
            oi = int(row.get('outer_interval') or 0)
            skip = int(row.get('skip_first') or 0)
            for o in range(outer):
                for i in range(inner):
                    if skip and o == i == 0:
                        continue
                    shift = o*oi+i*ii
                    spans[row['resource']].append((start+shift, end+shift))
    per_resource = {
        resource: dict(fu_issues=len(intervals), covered_fu_span_cycles=covered_cycles(intervals))
        for resource, intervals in spans.items()}
    return dict(per_resource=per_resource, total_fu_issues=sum(map(len, spans.values())),
                either_side_covered_fu_span_cycles=covered_cycles([i for side in spans.values() for i in side]))


def stage_diagnostics(folder):
    """Compiler plan intervals only: do not attribute runtime bubbles to these gaps."""
    plan = json.loads((folder/'graph_plan.json').read_text())
    stages, groups, previous = [], {}, 0
    for node in plan['nodes']:
        if 'nominal_start' not in node:
            continue
        name, start, end = node['name'], node['nominal_start'], node['nominal_end']
        # Aggregate repeated physical GEMM nodes / KV tiles by lowering name.
        match = re.search(r'\.group\d+\.q\d+\.(?:kv\d+\.)?([^.]+)', name)
        kind = match[1] if match else name
        group = groups.setdefault(kind, dict(count=0, nominal_node_cycles=0, preceding_positive_gap_cycles=0))
        group['count'] += 1
        group['nominal_node_cycles'] += end-start
        group['preceding_positive_gap_cycles'] += max(0, start-previous)
        stages.append(dict(name=name, nominal_start=start, nominal_end=end,
                           preceding_nominal_gap=start-previous))
        previous = end
    return dict(by_kind=groups, stages=stages)


def compare_outputs(actual, reference):
    results = {}
    for i, name in enumerate(('context', 'key', 'value')):
        a = np.fromfile(actual/f'actual.{i}.bin', '<u2')
        b = np.fromfile(reference/f'actual.{i}.bin', '<u2')
        assert a.shape == b.shape, f'{name}: incompatible outputs'
        bad = np.flatnonzero(a != b)
        results[name] = dict(mismatching_bf16_words=int(bad.size), first_indices=bad[:8].tolist())
    return results


def load_streamed_result(root, source, expected_record):
    """Verify a completed invocation against artifacts and the original fixture."""
    batch = expected_record['batch']
    folder = root/f'batch{batch}/attention'
    record = json.loads((folder.parent/'result.json').read_text())
    stamp = json.loads((folder/'completed.json').read_text())
    assert record['status'] == 'passed', 'previous streamed execution was not accepted'
    for key, expected in dict(model='GLM-4-9B W8A16', layer=0, batch=batch, past_len=32,
                              icu_encoding='raw-macro', sxm='wide').items():
        assert record[key] == expected, f'previous streamed scope differs: {key}'
    for key in ('target_sha256', 'runner_sha256', 'fixture_fingerprint',
                'projection_output_sha256', 'graph_sha256', 'clock_mhz'):
        assert record[key] == expected_record[key], f'previous streamed provenance differs: {key}'
    assert record['graph_sha256'] == graph_digest(folder) == graph_digest(source/f'batch{batch}/attention')
    assert record['binary_sha256'] == stamp['binary_sha256'] == sha(folder/'program.ftlpu')
    assert record['compiler_sha256'] == stamp['compiler_sha256']
    assert record['graph_sha256'] == stamp['semantic_graph_sha256']
    runtime_sha = sha(folder/'actual.runtime.ftlpu')
    assert runtime_sha == record['actual_runtime_program_sha256']['streamed_flash']
    timing = phase_timing(folder, record['clock_mhz'])
    assert timing == record['timing']['streamed_flash'], 'previous timing does not match its actual trace'
    inputs = {p.name: sha(p) for p in sorted(folder.glob('input*.bin'))}
    assert inputs, 'previous snapshot is missing raw input files'
    return dict(folder=folder, record=record, timing=timing, inputs=inputs,
                runtime_sha256=runtime_sha)


def previous_streamed_result(args, source, expected_record):
    if args.previous_results is None:
        return None
    return load_streamed_result(args.previous_results.resolve(), source, expected_record)


def compare_previous_execution(previous, folder, record):
    differences = compare_outputs(folder, previous['folder'])
    inputs = {p.name: sha(p) for p in sorted(folder.glob('input*.bin'))}
    before = previous['timing']['runtime_cycles']
    after = record['timing']['streamed_flash']['runtime_cycles']
    record.update(previous_streamed_results=str(previous['folder'].parent.parent),
        bitwise_differences_from_previous_streamed=differences,
        cycles_saved_vs_previous_streamed=before-after,
        latency_reduction_percent_vs_previous_streamed=100*(before-after)/before,
        speedup_vs_previous_streamed=before/after, faster_than_previous_streamed=after<before,
        input_sha256=inputs, inputs_match_previous_streamed=inputs==previous['inputs'])
    record['timing']['previous_streamed_flash'] = previous['timing']
    record['actual_runtime_program_sha256']['previous_streamed_flash'] = previous['runtime_sha256']
    assert record['inputs_match_previous_streamed'], 'previous raw inputs differ'
    assert all(x['mismatching_bf16_words'] == 0 for x in differences.values()), differences


def benchmark_batch32(args, source, folder, record):
    historical_baseline = source/'baseline_same_build/batch32'
    historical = json.loads((historical_baseline/'result.json').read_text())
    assert historical['status'] == 'passed' and historical['batch'] == 32
    assert historical['target_sha256'] == sha(args.target_config)
    assert historical['binary_sha256'] == sha(historical_baseline/'attention/program.ftlpu')
    assert historical['flash_binary_sha256'] == sha(source/'batch32/attention/program.ftlpu')
    benchmark(args, source, folder, record, timing_key='streamed_flash')
    record['timing']['historical_baseline'] = record['timing'].pop('baseline')
    record['faster_than_historical_baseline'] = record.pop('faster_than_baseline')
    record['actual_runtime_program_sha256'] = dict(streamed_flash=sha(folder/'actual.runtime.ftlpu'))
    record['input_sha256'] = {p.name: sha(p) for p in sorted(folder.glob('input*.bin'))}
    if args.previous_results is not None:
        compare_previous_execution(previous_streamed_result(args, source, record), folder, record)


def compare(args, source, packed_root, folder, record):
    scalar = source/'batch1/attention'
    packed = packed_root/'batch1/attention'
    baseline = source/'baseline_same_build/batch1/attention'
    ref = json.loads((packed_root/'batch1/result.json').read_text())
    stamp = json.loads((packed/'completed.json').read_text())
    signature = json.loads((source/'signature.json').read_text())
    assert ref['status'] == 'passed', 'packed-U execution was not accepted'
    assert ref['target_sha256'] == signature['target'] == sha(args.target_config)
    assert ref['runner_sha256'] == signature['runner'], 'different hardware runners'
    assert ref['binary_sha256'] == stamp['binary_sha256'] == sha(packed/'program.ftlpu')
    assert ref['previous_program_sha256'] == sha(scalar/'program.ftlpu'), 'different scalar Flash comparator'
    assert ref['graph_sha256'] == graph_digest(scalar) == graph_digest(packed), 'incompatible graph or constants'
    baseline_record = json.loads((baseline.parent/'result.json').read_text())
    assert baseline_record['status'] == 'passed' and baseline_record['target_sha256'] == signature['target']
    assert baseline_record['binary_sha256'] == sha(baseline/'program.ftlpu')
    assert baseline_record['flash_binary_sha256'] == ref['previous_program_sha256']
    previous = previous_streamed_result(args, source, ref)
    record['comparison_notes'] = [
        'Baseline uses the recorded baseline_same_build run on the same inputs, hardware target and runner; it is not rebuilt here.',
        'Plan stages/gaps are nominal compiler intervals, not runtime bubbles or automatically recoverable cycles.',
        'SXM coverage is the union of issued FU spans on each resource; sums across overlapping hemispheres are not elapsed latency.',
    ]

    # Reuse the exact packed-U test driver, fixture, independent numerical
    # oracle, raw-macro execution and per-ICU export. This emits only the new
    # binary. Collect extra diagnostics before returning a failed acceptance.
    try:
        benchmark(args, source, folder, record)
    finally:
        if 'timing' in record:
            record['timing']['streamed_flash'] = record['timing'].pop('packed_flash')
            record['timing']['packed_u_flash'] = phase_timing(packed, record['clock_mhz'])
            record['bitwise_differences_from_packed_u'] = compare_outputs(folder, packed)
            cycles = {k: v['runtime_cycles'] for k, v in record['timing'].items()}
            record['cycles_saved_vs_packed_u'] = cycles['packed_u_flash']-cycles['streamed_flash']
            record['cycles_saved_vs_baseline'] = cycles['baseline']-cycles['streamed_flash']
            record['faster_than_packed_u'] = record['cycles_saved_vs_packed_u'] > 0
            record['faster_than_baseline'] = record['cycles_saved_vs_baseline'] > 0
            variants = {'scalar_flash': scalar, 'packed_u_flash': packed, 'streamed_flash': folder}
            if previous is not None:
                variants['previous_streamed_flash'] = previous['folder']
            record['nominal_plan_diagnostics'] = {k: stage_diagnostics(v) for k, v in variants.items()}
            record['runtime_transpose_diagnostics'] = {
                k: transpose_diagnostics(v) for k, v in {**variants, 'baseline': baseline}.items()}
            record['actual_runtime_program_sha256'] = {k: sha(v/'actual.runtime.ftlpu') for k, v in variants.items()}
            if previous is not None:
                compare_previous_execution(previous, folder, record)
    assert record['graph_sha256'] == ref['graph_sha256']
    assert record['fixture_fingerprint'] == ref['fixture_fingerprint']
    assert record['projection_output_sha256'] == ref['projection_output_sha256']
    differences = record['bitwise_differences_from_packed_u']
    assert not any(x['mismatching_bf16_words'] for x in differences.values()), differences
    assert record['faster_than_packed_u'], 'streamed run did not improve on packed U'
    if previous is not None:
        assert record['faster_than_previous_streamed'], 'no execution-time improvement over previous streamed execution'
    # Beating baseline is a measured outcome, not a numerical acceptance criterion.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path('models/official_dense/glm4_9b'))
    parser.add_argument('--flash-results', type=Path, default=Path('results/glm4_9b_decoder_layer_decode_flash'))
    parser.add_argument('--packed-results', type=Path, default=Path('results/glm4_9b_decoder_layer_decode_flash_packed'))
    parser.add_argument('--batch', type=int, choices=(1,32), default=1)
    parser.add_argument('--previous-results', type=Path,
                        help='completed previous streamed root containing the requested batch and its attention artifacts')
    parser.add_argument('--compare-only', action='store_true',
                        help='compare a completed streamed invocation in --output-dir with --previous-results')
    parser.add_argument('--output-dir', type=Path, default=Path('results/glm4_9b_decoder_layer_decode_flash_streamed'))
    parser.add_argument('--build-dir', type=Path, help='compiler and runtime snapshot for a new execution')
    parser.add_argument('--target-config', type=Path, required=True)
    args = parser.parse_args()
    if not args.compare_only and args.build_dir is None:
        parser.error('--build-dir is required for a new execution')
    scalar_root, packed_root = args.flash_results.resolve(), args.packed_results.resolve()
    folder = args.output_dir.resolve()/f'batch{args.batch}/attention'
    if any(folder.is_relative_to(ref) for ref in (scalar_root, packed_root)):
        parser.error('select an output directory outside both reference results')
    if args.previous_results is not None:
        previous = args.previous_results.resolve()
        output = args.output_dir.resolve()
        if output.is_relative_to(previous) or previous.is_relative_to(output):
            parser.error('previous streamed snapshot and output must be disjoint directories')
    if args.compare_only:
        if args.previous_results is None:
            parser.error('--compare-only requires --previous-results')
        result = folder.parent/'result.json'
        record = json.loads(result.read_text())
        assert record['batch'] == args.batch and record['target_sha256'] == sha(args.target_config)
        load_streamed_result(args.output_dir.resolve(), scalar_root, record)
        compare_previous_execution(previous_streamed_result(args, scalar_root, record), folder, record)
        result.write_text(json.dumps(record, indent=2))
        print(f'{result}: exact context/K/V and inputs; {record["cycles_saved_vs_previous_streamed"]} cycles saved')
        return
    folder.mkdir(parents=True, exist_ok=True)
    result = folder.parent/'result.json'
    record = dict(status='running', model='GLM-4-9B W8A16', layer=0, batch=args.batch, past_len=32,
                  scope='one-token attention only; no projection, FFN or whole-layer rerun',
                  icu_encoding='raw-macro', sxm='wide', scalar_results=str(scalar_root))
    result.write_text(json.dumps(record, indent=2))
    try:
        if args.batch == 1:
            record['packed_u_results'] = str(packed_root)
            compare(args, scalar_root, packed_root, folder, record)
        else:
            benchmark_batch32(args, scalar_root, folder, record)
        record['status'] = 'passed'
    except Exception as error:
        record.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        result.write_text(json.dumps(record, indent=2))
    if args.batch == 1:
        print(f'{result}: passed; {record["cycles_saved_vs_packed_u"]} cycles saved vs packed U; '
              f'{record["cycles_saved_vs_baseline"]} cycles saved vs baseline', flush=True)
    else:
        print(f'{result}: passed; {record["timing"]["streamed_flash"]["runtime_cycles"]} runtime cycles', flush=True)
    if args.previous_results is not None:
        print(f'{record["cycles_saved_vs_previous_streamed"]} cycles saved vs previous streamed; '
              f'{record["speedup_vs_previous_streamed"]:.4f}x speedup; exact context/K/V match', flush=True)


if __name__ == '__main__':
    main()
