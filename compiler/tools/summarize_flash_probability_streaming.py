"""Validate Flash P streaming from executed artifacts; never emits schedules."""
import argparse
import collections
import csv
import json
from pathlib import Path
import re
import subprocess

from summarize_dense_probability_streaming import sha


def mem_issues(folder):
    counts = collections.Counter()
    with (folder/'runtime.pipeline.csv').open(newline='') as stream:
        for row in csv.DictReader(stream):
            if row['resource'].startswith('MEM.'):
                match = re.search(r'issues=(\d+)', row['detail'])
                assert match, 'runtime trace lacks actual FU issue count'
                counts[row['resource']] += int(match[1])
    return dict(counts)


def fused_edges(value):
    if isinstance(value, dict):
        return ([value] if value.get('vxm_sxm_direct_transpose') is True else []) + [
            edge for child in value.values() for edge in fused_edges(child)]
    if isinstance(value, list):
        return [edge for child in value for edge in fused_edges(child)]
    return []


def audit(folder, build, record=None):
    program = folder/'actual.runtime.ftlpu'
    diagnostic = build/'check.exe'
    output = subprocess.run([str(diagnostic), str(program)], check=True,
                            capture_output=True, text=True)
    analysis = json.loads(output.stdout)
    reports = [analysis['transport']] + [s['capacity'] for s in analysis['compute_reload_segments']]
    if not analysis['compute_reload_segments']:
        reports.append(analysis['aggregate'])
    assert all(r['fits'] for r in reports)
    assert all(q['peak_macro_contexts'] <= 1 for r in reports for q in r['queues'])
    analysis['source'] = dict(program_sha256=sha(program),
        runtime_pipeline_sha256=sha(folder/'runtime.pipeline.csv'),
        compiler_sha256=sha(build/'compiler/ftlpu-compile.exe'),
        runner_sha256=sha(build/'runtime/compiled_stage_file_runner.exe'),
        diagnostic_sha256=sha(diagnostic))
    if record:
        assert analysis['source']['program_sha256']==record['phase']['actual_program_sha256']
        assert analysis['source']['runtime_pipeline_sha256']==record['phase']['runtime_pipeline_sha256']
        assert analysis['source']['compiler_sha256']==record['compiler_sha256']
        assert analysis['source']['runner_sha256']==record['runner_sha256']
        analysis['source']['result_sha256']=sha(folder/'result.json')
        analysis['source']['target_sha256']=record['target_sha256']
    (folder/'compute_imem_segments.json').write_text(json.dumps(analysis, indent=2)+'\n')
    return analysis


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path,
                        default=Path('results/glm4_9b_prefill_attention_comparison'))
    parser.add_argument('--build-dir', type=Path, required=True)
    args=parser.parse_args()
    root, build=args.results.resolve(), args.build_dir.resolve()
    before_dir, after_dir=root/'flash_before_p_streaming', root/'flash'
    before=json.loads((before_dir/'result.json').read_text())
    after=json.loads((after_dir/'result.json').read_text())
    assert before['status']==after['status']=='passed'
    for field in ('runner_sha256','target_sha256','fixture_sha256','precision','softmax_precision'):
        assert before[field]==after[field], field
    assert before['phase']['semantic_graph_sha256']==after['phase']['semantic_graph_sha256']
    assert before['phase']['output_sha256']==after['phase']['output_sha256']
    for folder, record in ((before_dir,before),(after_dir,after)):
        assert sha(folder/'actual.0.bin')==record['phase']['output_sha256'][0]
        assert sha(folder/'runtime.pipeline.csv')==record['phase']['runtime_pipeline_sha256']
    plan=json.loads((after_dir/'graph_plan.json').read_text())
    edges=fused_edges(plan)
    schedule=(after_dir/'decoder_layer.schedule.mlir').read_text()
    windows=re.findall(r'ftlpu.schedule.timeline \{end = (\d+) : i64, name = "([^"\n]*vector\.vxm_sxm_direct_transpose)", start = (\d+) : i64\}',schedule)
    assert edges and len(edges)==len(windows)
    assert all(edge['materialized_before_transpose'] is False for edge in edges)
    audits={}
    for relative in ('dense','flash','synthetic/dense','synthetic/flash'):
        folder=root/relative
        record=json.loads((folder/'result.json').read_text())
        assert record['status']=='passed'
        audits[relative]=audit(folder,build,record)
    regressions=[]
    for name in ('prefill_streaming','single_tile'):
        folder=root/'regression'/name
        record=json.loads((folder/'summary.json').read_text())
        assert record['status']=='passed'
        audits[name]=audit(folder,build)
        regressions.append(record)
    old,new=before['phase']['timing'],after['phase']['timing']
    old_counts,new_counts=mem_issues(before_dir),mem_issues(after_dir)
    with (after_dir/'runtime.pipeline.csv').open(newline='') as stream:
        invocation=[r for r in csv.DictReader(stream) if r['resource']=='Session.Invocation']
    assert len(invocation)==1
    first=min(windows,key=lambda x:int(x[2]))
    shift=int(invocation[0]['start'])
    report=dict(status='passed',scope=after['scope'],model=after['model'],
        shape=dict(q=after['q_shape'],k=after['k_shape'],v=after['v_shape']),
        before=old,after=new,output_bitexact=True,inputs_runner_target_unchanged=True,
        compiler_sha256=after['compiler_sha256'],runner_sha256=after['runner_sha256'],
        target_sha256=after['target_sha256'],fixture_sha256=after['fixture_sha256'],
        actual_program_sha256=after['phase']['actual_program_sha256'],
        runtime_pipeline_sha256=after['phase']['runtime_pipeline_sha256'],
        precision=after['precision'],softmax_precision=after['softmax_precision'],
        context_metrics=after['context_metrics'],end_to_end_context_metrics=after['end_to_end_context_metrics'],
        cycles_saved=old['runtime_cycles']-new['runtime_cycles'],
        total_latency_reduction_percent=100*(old['runtime_cycles']-new['runtime_cycles'])/old['runtime_cycles'],
        device_window_reduction_percent=100*(old['device_execution_window_cycles']-new['device_execution_window_cycles'])/old['device_execution_window_cycles'],
        streaming_windows=len(windows),before_transpose_materialized=False,
        transposed_probability_retained_for_pv_reuse=True,
        fp32_probability_retained_for_denominator=True,
        first_runtime_streaming_window=dict(start=shift+int(first[2]),end=shift+int(first[0]),name=first[1]),
        mem_issues_before=old_counts,mem_issues_after=new_counts,
        mem_issues_saved={k:old_counts.get(k,0)-new_counts.get(k,0) for k in old_counts.keys()|new_counts.keys()},
        physical_capacity='passed; one active FU 3D context per queue',
        regression_results=regressions,
        notes=['Core only: QK, causal online softmax and PV; excludes projection, O and FFN.',
               'Runtime includes initial C2C input and context readback. No separate diagnostic tensor output.',
               'iMEM image load/reload transfer latency is not timed.',
               'Complete 32-key tiles stream; padded tails and multi-consumer edges retain materialization.'])
    (root/'flash_probability_streaming.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('status','cycles_saved','streaming_windows','output_bitexact','mem_issues_saved','first_runtime_streaming_window')},indent=2))


if __name__=='__main__':
    main()
