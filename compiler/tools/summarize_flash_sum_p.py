"""Audit the shared exponent read against accepted separate-pass runtime data."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re


def exponent_read_bytes(directory):
    total = 0
    with (directory / 'runtime.pipeline.csv').open(newline='') as stream:
        for row in csv.DictReader(stream):
            if row['resource'] not in ('MEM.E.Read', 'MEM.W.Read'):
                continue
            address = re.search(r'slice=(\d+) bank=(\d+)', row['detail'])
            if address and 12 <= int(address[1]) <= 15 and int(address[2]) == 0:
                assert int(row['skip_first']) == 0
                total += (int(row['end']) - int(row['start'])) * 32 \
                    * int(row['inner_count']) * int(row['outer_count'])
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    args = parser.parse_args()
    read = lambda path: json.loads(path.read_text())
    before = read(args.baseline / 'flash/result.json')
    after = read(args.results / 'flash/result.json')
    comparison = read(args.results / 'scheduling_comparison.json')
    case = comparison['cases']['flash']
    regression = read(args.results / 'regression/flash_tile/summary.json')
    for key in ('runner_sha256', 'target_sha256', 'fixture_sha256', 'precision',
                'q_shape', 'k_shape', 'v_shape', 'icu_encoding'):
        assert before[key] == after[key], key
    assert case['bit_identical_to_baseline']
    assert all(c['status'] == 'passed' and c['exponent_read_passes'] == 1
               and c['bit_identical_to_separate_passes'] for c in regression['cases'])
    old_reads = exponent_read_bytes(args.baseline / 'flash')
    new_reads = exponent_read_bytes(args.results / 'flash')
    qshape = after['q_shape']
    expected = qshape[0] * qshape[2] * qshape[1] * after['k_shape'][1] * 4
    assert old_reads == 2 * new_reads and new_reads == expected
    for audit in case['physical_audits']:
        assert audit['fits'] and audit['peak_fu_contexts'] <= 1
        assert audit['compute_reload_segments'] == 0
    report = dict(
        status='passed', scope=comparison['scope'],
        mechanism='One FP32 exponent read feeds C0/C1 BF16 P and C2/C3 FP32 sum; terminal sum uses existing C3-to-C2 feedback for reciprocal.',
        precision=after['precision'], runner_sha256=after['runner_sha256'],
        target_sha256=after['target_sha256'], fixture_sha256=after['fixture_sha256'],
        compiler_sha256=after['compiler_sha256'],
        exponent_read_bytes=dict(before=old_reads, after=new_reads),
        bit_identical_to_baseline=True,
        context_relative_l2=case['context_relative_l2'],
        end_to_end_context_relative_l2=case['end_to_end_context_relative_l2'],
        before={k: case['before'][k] for k in ('timing', 'diagnostic_timing', 'mxm_compute')},
        after={k: case['after'][k] for k in ('timing', 'diagnostic_timing', 'mxm_compute', 'layout_copies', 'program_sha256', 'pipeline_sha256')},
        runtime_cycles_saved=case['runtime_cycles_saved'],
        runtime_reduction_fraction=case['runtime_reduction_fraction'],
        physical_audits=case['physical_audits'],
        regression=[{k: c[k] for k in ('case', 'shape', 'causal', 'metrics',
            'exponent_read_bytes', 'exponent_read_passes', 'bit_identical_to_separate_passes')}
                    for c in regression['cases']],
        artifacts=dict(pipeline='flash/runtime.pipeline.csv',
                       executed_icu_index='flash/icu_programs/index.csv',
                       raw_macro='flash/actual.runtime.ftlpu',
                       build_manifest='scheduling_build.json'),
        limitations=after['limits'])
    report['audit_script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (args.results / 'sum_p_fusion.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f"passed: exponent read {old_reads} -> {new_reads} bytes; "
          f"saved {case['runtime_cycles_saved']} cycles; bit-identical outputs")


if __name__ == '__main__':
    main()
