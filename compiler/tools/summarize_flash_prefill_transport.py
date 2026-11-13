"""Compare executed Flash prefill artifacts; never generates model schedules."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import re

from summarize_flash_probability_streaming import audit, fused_edges, mem_issues


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def transport_bytes(folder):
    plan = json.loads((folder/'graph_plan.json').read_text())
    counts = collections.Counter()
    for transfer in plan['linker_mem_transports']:
        phase = transfer['phase']
        category = ('context_export' if phase.startswith('validation_export.')
                    else phase.rsplit('.', 1)[-1])
        # Count source reads plus destination writes, at 32 bytes/plane/word.
        counts[category] += transfer['vectors'] * transfer['byte_planes'] * 32 * 2
    return dict(counts)


def coverage(folder, resource):
    with (folder/'runtime.pipeline.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    invocation = [r for r in rows if r['resource'] == 'Session.Invocation']
    assert len(invocation) == 1
    begin, end = (int(invocation[0][k]) for k in ('start', 'end'))
    intervals = []
    for row in rows:
        if row['resource'] != resource:
            continue
        assert row['induction'] == 'none'
        duration = int(row['end'])-int(row['start'])
        for outer in range(int(row['outer_count'])):
            for inner in range(int(row['inner_count'])):
                if int(row['skip_first']) and not outer and not inner:
                    continue
                at = int(row['start']) + outer*int(row['outer_interval']) + inner*int(row['inner_interval'])
                lo, hi = max(begin, at), min(end, at+duration)
                if lo < hi:
                    intervals.append((lo, hi))
    occupied, previous = 0, begin
    for lo, hi in sorted(intervals):
        occupied += max(0, hi-max(previous, lo))
        previous = max(previous, hi)
    return dict(active_cycles=occupied, invocation_cycles=end-begin,
                busy_fraction=occupied/(end-begin))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path,
                        default=Path('results/glm4_9b_prefill_attention_comparison'))
    parser.add_argument('--build-dir', type=Path, required=True)
    args = parser.parse_args()
    root, build = args.results.resolve(), args.build_dir.resolve()
    before_dir, after_dir = root/'flash_before_transport_cleanup', root/'flash'
    before = json.loads((before_dir/'result.json').read_text())
    after = json.loads((after_dir/'result.json').read_text())
    assert before['status'] == after['status'] == 'passed'
    for key in ('runner_sha256', 'target_sha256', 'fixture_sha256', 'precision', 'softmax_precision'):
        assert before[key] == after[key], key
    assert before['phase']['semantic_graph_sha256'] == after['phase']['semantic_graph_sha256']
    assert before['phase']['output_sha256'] == after['phase']['output_sha256']
    for folder, record in ((before_dir, before), (after_dir, after)):
        assert sha(folder/'actual.0.bin') == record['phase']['output_sha256'][0]
        assert sha(folder/'runtime.pipeline.csv') == record['phase']['runtime_pipeline_sha256']
    plan = json.loads((after_dir/'graph_plan.json').read_text())
    normalizers = [n for n in plan['nodes'] if n['name'].endswith('.normalize')]
    schedule = (after_dir/'decoder_layer.schedule.mlir').read_text()
    windows = re.findall(r'ftlpu.schedule.timeline \{end = (\d+) : i64, name = "([^"\n]*vector\.native_row_scale)", start = (\d+) : i64\}', schedule)
    assert len(windows) == len(normalizers) == after['native_row_scale_count']
    assert len(fused_edges(plan)) == 16  # GLM seq32, two KV groups, 16 query blocks.
    for n in normalizers:
        assert n['tight_serial_schedule']
        assert all('distributed' not in p['kind'] for p in n['direct_output_layouts'])
    audit_summaries = {}
    for relative in ('dense', 'flash', 'synthetic/dense', 'synthetic/flash',
                     'regression/prefill_streaming', 'regression/single_tile'):
        folder = root/relative
        is_regression = relative.startswith('regression/')
        record = json.loads((folder/('summary.json' if is_regression else 'result.json')).read_text())
        assert record['status'] == 'passed'
        result = audit(folder, build, None if is_regression else record)
        audit_summaries[relative] = dict(status='passed',
            compute_reload_segment_count=result['compute_reload_segment_count'],
            source=result['source'], relative_l2=(record['relative_l2'] if is_regression
                                                else record['context_metrics']['relative_l2']))
    old, new = before['phase']['timing'], after['phase']['timing']
    old_bytes, new_bytes = transport_bytes(before_dir), transport_bytes(after_dir)
    assert old_bytes.get('normalize', 0) > 0 and new_bytes.get('normalize', 0) == 0
    old_mem, new_mem = mem_issues(before_dir), mem_issues(after_dir)
    report = dict(status='passed', scope=after['scope'], model=after['model'],
        shape=dict(q=after['q_shape'], k=after['k_shape'], v=after['v_shape']),
        before=old, after=new, output_bitexact=True, inputs_runner_target_unchanged=True,
        precision=after['precision'], softmax_precision=after['softmax_precision'],
        compiler_sha256=after['compiler_sha256'], runner_sha256=after['runner_sha256'],
        target_sha256=after['target_sha256'], fixture_sha256=after['fixture_sha256'],
        actual_program_sha256=after['phase']['actual_program_sha256'],
        runtime_pipeline_sha256=after['phase']['runtime_pipeline_sha256'],
        context_metrics=after['context_metrics'], end_to_end_context_metrics=after['end_to_end_context_metrics'],
        total_latency_reduction_percent=100*(old['runtime_cycles']-new['runtime_cycles'])/old['runtime_cycles'],
        device_window_reduction_percent=100*(old['device_execution_window_cycles']-new['device_execution_window_cycles'])/old['device_execution_window_cycles'],
        layout_copy_read_write_bytes_before=old_bytes, layout_copy_read_write_bytes_after=new_bytes,
        layout_copy_read_write_bytes_saved=sum(old_bytes.values())-sum(new_bytes.values()),
        mem_issues_before=old_mem, mem_issues_after=new_mem,
        runtime_coverage={r:dict(before=coverage(before_dir, r), after=coverage(after_dir, r))
                          for r in ('VXM.C0', 'MXM.E0.Compute', 'MXM.W0.Compute')},
        native_row_scale_windows=len(windows), probability_streaming_windows=len(fused_edges(plan)),
        physical_capacity='passed; one active FU 3D context per ICU queue',
        audits=audit_summaries,
        notes=['Core QK/causal online softmax/PV only; excludes O projection, residual and FFN.',
               'Includes initial C2C and context output readback. No separate diagnostic output.',
               'iMEM image transfer/reload latency is not modeled by the runner.',
               'Remaining layout copies establish mirrored time-major softmax input and contiguous context export.',
               'Reduction dependencies, SXM tile collection and explicit iMEM drain boundaries remain.',
               'Prepared K/V reuse reduces graph nodes; this external-input fixture already deduplicated their physical bindings.'])
    path = root/'flash_transport_optimization.json'
    path.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('status','total_latency_reduction_percent',
        'device_window_reduction_percent','layout_copy_read_write_bytes_saved','runtime_coverage')}, indent=2))


if __name__ == '__main__':
    main()
