"""Compare completed SWA runs; consumes artifacts without generating schedules."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from summarize_attention_schedule import coverage


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def phases(folder, summary):
    plan = json.loads((folder / 'graph_plan.json').read_text())
    cores = [n for n in plan['nodes'] if n.get('attention_core_schedule') == 'flash_tile_pipeline']
    begin = max(n['end'] for n in plan['nodes'] if n.get('native_planar_rotary'))
    end = max(n['nominal_end'] for n in cores)
    offset = summary['timing']['initial_ingress_cycles'] + summary['timing']['weight_page_wait_cycles']
    bounds = (begin + offset, end + offset)
    copies = [c for c in plan['linker_mem_transports'] if c['phase'] in {n['name'] for n in cores}]
    exports = [c for c in plan['linker_mem_transports'] if c['reason'] == 'graph_result_export']
    native_span = sum(n['nominal_end'] - n['nominal_start'] for n in cores)
    assert all(c['route'].startswith('passive_sr') for c in copies)
    occupancy = {}
    for resource in ['MXM.E0.Compute', 'MXM.W0.Compute', *[f'VXM.C{i}' for i in range(8)]]:
        item = coverage(folder, resource, bounds)
        item['swa_window_issue_fraction'] = item['active_cycles'] / (end - begin)
        occupancy[resource] = item
    return dict(groups=len(cores), native_core_cycles=native_span,
                swa_with_input_preparation_cycles=end - begin, runtime_bounds=bounds,
                input_passive_copy_domains=len(copies), input_passive_copy_vectors=sum(c['vectors'] for c in copies),
                diagnostic_pack_domains=len(exports), diagnostic_pack_vectors=sum(c['vectors'] for c in exports),
                diagnostic_pack_span=max(c['end'] for c in exports) - min(c['start'] for c in exports) if exports else 0,
                occupancy=occupancy,
                mxm_mean_issue_fraction=sum(occupancy[x]['swa_window_issue_fraction'] for x in ('MXM.E0.Compute', 'MXM.W0.Compute')) / 2,
                mxm_native_core_issue_fraction=sum(occupancy[x]['active_cycles'] for x in ('MXM.E0.Compute', 'MXM.W0.Compute')) / (2 * native_span),
                vxm_mean_issue_fraction=sum(occupancy[f'VXM.C{i}']['swa_window_issue_fraction'] for i in range(8)) / 8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    args = parser.parse_args()
    before, after = args.before, args.after
    old, new = [json.loads((p / 'summary.json').read_text()) for p in (before, after)]
    assert old['status'] == new['status'] == 'passed'
    for key in ('scope', 'revision', 'input_sha256', 'weight_sha256', 'gamma_sha256', 'limits',
                'precision', 'raw_macro', 'sequence', 'batch', 'swa', 'outputs'):
        assert old[key] == new[key], key
    for key in ('target', 'runner', 'exporter'):
        assert old['signatures'][key] == new['signatures'][key], key
    identical = {name: sha(before / f'actual.{i}.bin') == sha(after / f'actual.{i}.bin')
                 for i, name in enumerate(new['outputs'])}
    assert all(identical.values()), identical
    stages = [phases(p, s) for p, s in ((before, old), (after, new))]
    assert stages[1]['swa_with_input_preparation_cycles'] < stages[0]['swa_with_input_preparation_cycles']
    segments = list(csv.DictReader((after / 'runtime_icu_programs/compute_imem_segments.csv').open(newline='')))
    assert all(s['capacity_checked'] == '1' for s in segments)
    report = dict(status='passed', comparison='same scope, inputs, weights, precision, thresholds, target and runtime',
                  before=str(before), after=str(after), all_outputs_bitwise_unchanged=identical,
                  metrics=new['metrics']['swa_context'],
                  timing_cycles={key: dict(before=old['timing'][key], after=new['timing'][key],
                                           saved=old['timing'][key]-new['timing'][key]) for key in old['timing']},
                  swa_before=stages[0], swa_after=stages[1],
                  compute_imem_segments=len(segments), max_segment_slots=max(int(s['used_slots']) for s in segments),
                  notes=['SWA occupancy includes input preparation, excludes host I/O and external-memory waits.',
                         'VXM.C0..C7 describe mirrored ALU pairs; the reported fraction counts issued vectors, not FLOP efficiency.',
                         'Passive copy totals exclude SXM layout transforms; they are not a zero-copy claim.',
                         'Invocation timing includes diagnostic context packing; HostOutput readback is separate.',
                         'Instruction-image DMA is not timed; no inverse RoPE, output projection, cache persistence, mHC or FFN.'])
    output = after / 'scheduling_comparison.json'
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(status=report['status'], timing=report['timing_cycles'],
                         swa_before={k:v for k,v in stages[0].items() if k != 'occupancy'},
                         swa_after={k:v for k,v in stages[1].items() if k != 'occupancy'}), indent=2))


if __name__ == '__main__':
    main()
