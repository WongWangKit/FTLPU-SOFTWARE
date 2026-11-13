"""Compare two executed attention schedules without modifying either trace."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def coverage(folder, resource, interval=None):
    spans = []
    with (folder/'runtime.pipeline.csv').open(newline='') as stream:
        for row in csv.DictReader(stream):
            if row['resource'] != resource:
                continue
            # The selected MXM compute rows are rectangular runtime domains.
            assert int(row['skip_first']) == 0
            for outer in range(int(row['outer_count'])):
                for inner in range(int(row['inner_count'])):
                    offset = outer*int(row['outer_interval']) + inner*int(row['inner_interval'])
                    a, b = int(row['start'])+offset, int(row['end'])+offset
                    if interval:
                        a, b = max(a, interval[0]), min(b, interval[1])
                    if a < b:
                        spans.append((a, b))
    merged = []
    for a, b in sorted(spans):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(b, merged[-1][1])
        else:
            merged.append([a, b])
    if not merged:
        return dict(active_cycles=0)
    active = sum(b-a for a, b in merged)
    span = merged[-1][1]-merged[0][0]
    return dict(first_issue=merged[0][0], last_issue_end=merged[-1][1],
                active_cycles=active, first_to_last_span=span,
                active_fraction=active/span, gaps=len(merged)-1,
                max_gap=max((b[0]-a[1] for a, b in zip(merged, merged[1:])), default=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline', 'results', 'imem-check'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    report = dict(scope='GLM-4-9B seq32 prefill attention cores, excludes projection execution',
                  source='actual raw-macro runtime traces; diagnostic output compared separately', cases={})
    for strategy in ('dense', 'flash'):
        before, after = args.baseline/strategy, args.results/strategy
        old, new = (json.loads((folder/'result.json').read_text()) for folder in (before, after))
        assert old['status'] == new['status'] == 'passed'
        for key in ('fixture_sha256', 'runner_sha256', 'target_sha256', 'precision'):
            assert old[key] == new[key], key
        assert sha(before/'diagnostic/actual.0.bin') == sha(after/'diagnostic/actual.0.bin')
        entry = dict(bit_identical_to_baseline=True, context_relative_l2=new['context_metrics']['relative_l2'],
                     end_to_end_context_relative_l2=new['end_to_end_context_metrics']['relative_l2'])
        for label, folder, record in (('before', before, old), ('after', after, new)):
            phase = record['phase']
            assert sha(folder/'actual.runtime.ftlpu') == phase['actual_program_sha256']
            assert sha(folder/'runtime.pipeline.csv') == phase['runtime_pipeline_sha256']
            data = dict(timing=phase['timing'], diagnostic_timing=phase['diagnostic_phase']['timing'],
                        pipeline_sha256=phase['runtime_pipeline_sha256'], program_sha256=phase['actual_program_sha256'])
            data['mxm_compute'] = {unit: coverage(folder, unit) for unit in ('MXM.E0.Compute', 'MXM.W0.Compute')}
            plan = json.loads((folder/'graph_plan.json').read_text())
            copies = [t for t in plan['linker_mem_transports']
                      if not t['phase'].startswith('validation_export.')]
            copied_bytes = sum(t['vectors']*t['byte_planes']*32 for t in copies)
            data['layout_copies'] = dict(domains=len(copies), payload_bytes=copied_bytes,
                                         read_plus_write_bytes=2*copied_bytes)
            data['graph_nodes'] = len(plan['nodes'])
            data['vxm'] = {unit: coverage(folder, unit) for unit in ('VXM.C0', 'VXM.C1', 'VXM.C5', 'VXM.C7')}
            if strategy == 'flash' and any(n.get('attention_core_schedule') == 'flash_tile_pipeline'
                                           for n in plan['nodes']):
                assert not copies
                schedule = (folder/'decoder_layer.schedule.mlir').read_text()
                data['tiles'] = []
                offset = phase['timing']['initial_ingress_window_cycles']
                for stage in ('qk.continuous_k', 'pv.mxm_vxm_direct'):
                    intervals = re.findall(r'ftlpu.schedule.timeline \{end = (\d+) : i64, name = "[^"]+\.flash\.'+
                                           re.escape(stage)+r'", start = (\d+) : i64', schedule)
                    for index, (end, start) in enumerate(intervals):
                        # The selected stream window includes result drain.
                        # Compute's own first-to-last interval must have no gap.
                        bounds = (int(start)+offset, int(end)+offset)
                        occupancy = {unit: coverage(folder, unit, bounds) for unit in ('MXM.E0.Compute', 'MXM.W0.Compute')}
                        data['tiles'].append(dict(stage=stage, tile=index, bounds=bounds, mxm_compute=occupancy))
            if strategy == 'dense':
                schedule = (folder/'decoder_layer.schedule.mlir').read_text()
                data['stages'] = {}
                offset = phase['timing']['initial_ingress_window_cycles']
                for stage in ('qk', 'softmax', 'pv'):
                    end, start = map(int, re.search(r'ftlpu.schedule.timeline \{end = (\d+) : i64, name = "[^"]+\.'+stage+r'", start = (\d+) : i64', schedule).groups())
                    data['stages'][stage] = dict(cycles=end-start, runtime_start=start+offset, runtime_end=end+offset)
                    if stage != 'softmax':
                        data['stages'][stage]['mxm_compute'] = {
                            unit: coverage(folder, unit, (start+offset, end+offset))
                            for unit in ('MXM.E0.Compute', 'MXM.W0.Compute')}
            entry[label] = data
        entry['runtime_cycles_saved'] = old['phase']['timing']['runtime_cycles'] - new['phase']['timing']['runtime_cycles']
        entry['runtime_reduction_fraction'] = entry['runtime_cycles_saved']/old['phase']['timing']['runtime_cycles']
        entry['physical_audits'] = []
        for folder in (after, after/'diagnostic'):
            audit = json.loads(subprocess.check_output([str(args.imem_check.resolve()), str((folder/'actual.runtime.ftlpu').resolve())], text=True))
            assert audit['aggregate']['fits'] and audit['transport']['fits']
            assert max(q['peak_macro_contexts'] for q in audit['aggregate']['queues']) <= 1
            (folder/'imem_capacity.json').write_text(json.dumps(audit, indent=2)+'\n')
            entry['physical_audits'].append(dict(directory=str(folder), fits=True,
                physical_words=audit['aggregate']['physical_words_total'],
                physical_bytes=sum(q['physical_words']*q['word_bits']//8 for q in audit['aggregate']['queues']),
                peak_fu_contexts=max(q['peak_macro_contexts'] for q in audit['aggregate']['queues']),
                compute_reload_segments=audit['compute_reload_segment_count']))
        report['cases'][strategy] = entry
    report['status'] = 'passed'
    (args.results/'scheduling_comparison.json').write_text(json.dumps(report, indent=2)+'\n')
    for strategy, case in report['cases'].items():
        print(strategy, 'saved cycles', case['runtime_cycles_saved'], 'bit identical', case['bit_identical_to_baseline'])


if __name__ == '__main__':
    main()
