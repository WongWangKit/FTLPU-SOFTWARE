"""Summarize actual GLM decode traces; this tool performs no model lowering."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re


def phase_timing(folder, clock_mhz):
    log = (folder/'execution.log').read_text(errors='replace')
    final = re.search(r'^cycles=(\d+) .*weight_page_runtime_wait_cycles=(\d+).*$', log, re.M)
    if not final:
        raise ValueError(f'incomplete execution: {folder}')
    total, input_end, output_start, output_end, invocation_end = 0, 0, None, 0, None
    with (folder/'runtime.pipeline.csv').open(newline='') as stream:
        for row in csv.DictReader(stream):
            start, end = int(row['start']), int(row['end'])
            end += (int(row.get('inner_count') or 1)-1)*int(row.get('inner_interval') or 0)
            end += (int(row.get('outer_count') or 1)-1)*int(row.get('outer_interval') or 0)
            total = max(total, end)
            if row['resource'] == 'C2C.HostInput':
                input_end = max(input_end, end)
            if row['resource'] == 'C2C.HostOutput':
                output_start = start if output_start is None else min(output_start, start)
                output_end = max(output_end, end)
            if row['resource'] == 'Session.Invocation':
                invocation_end = end if invocation_end is None else max(invocation_end, end)
    if output_start is None:
        if invocation_end is None:
            raise ValueError(f'missing completed invocation: {folder}')
        # A resident-only graph legitimately has no external result transfer.
        # Count its actual completed device window, not a fabricated readback.
        output_start = output_end = invocation_end
    result = dict(runtime_cycles=total, runtime_ms=total/(clock_mhz*1000),
                  initial_ingress_window_cycles=input_end,
                  device_execution_window_cycles=output_start-input_end,
                  static_schedule_cycles=int(final[1]),
                  weight_page_wait_cycles=int(final[2]),
                  output_readback_including_diagnostics_cycles=output_end-output_start,
                  isolated_diagnostic_readback_cycles=None)
    # This partition measures disjoint actual timeline windows. Static schedule
    # and weight waits are diagnostics within execution, not additional totals.
    result['other_timeline_cycles'] = total-input_end-(output_start-input_end)-(output_end-output_start)
    if (folder/'completed.json').exists():
        stamp = json.loads((folder/'completed.json').read_text())
        # Older reused phases recorded binary/input hashes only. Do not
        # misattribute them to the current compiler or root manifest.
        result['compiler_sha256'] = stamp.get('compiler_sha256')
        result['program_sha256'] = stamp['binary_sha256']
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--flash-results', type=Path, required=True)
    parser.add_argument('--baseline-results', type=Path, required=True)
    parser.add_argument('--target-config', type=Path, required=True)
    args = parser.parse_args()
    signature = json.loads((args.flash_results/'signature.json').read_text())
    assert signature['target'] == hashlib.sha256(args.target_config.read_bytes()).hexdigest(), 'timing target differs from executed target'
    config = json.loads(args.target_config.read_text())['external_memory']
    clock = config['lpu_clock_mhz']
    report = dict(clock_mhz=clock,
                  target_sha256=signature['target'], runner_sha256=signature['runner'],
                  peak_ddr_mbytes_per_second=config['ddr_peak_bandwidth_mbytes_per_second'],
                  ddr_scheduling_efficiency_percent=config['ddr_scheduling_efficiency_percent'],
                  scope='GLM-4-9B layer0 decode past32; three device invocations',
                  notes=[
                      'Runtime includes input transfer, weight waits, and output/diagnostic readback.',
                      'Static schedule cycles are not pure active-FU cycles.',
                      'Weight wait is within execution window; do not add it to runtime again.',
                      'HostOutput aggregates required outputs and KV diagnostics; their exact split is unavailable.',
                      'Attention comparison uses one compiler; original whole-layer baseline has per-phase build provenance.',
                  ], batches=[])
    for batch in (1, 32, 64):
        folder = args.flash_results/f'batch{batch}'
        record = json.loads((folder/'result.json').read_text())
        entry = dict(batch=batch, status=record['status'])
        comparison = args.flash_results/'baseline_same_build'/f'batch{batch}'
        if (comparison/'result.json').exists():
            entry['attention_same_build'] = dict(
                metrics=json.loads((comparison/'result.json').read_text()),
                baseline=phase_timing(comparison/'attention', clock),
                flash=phase_timing(folder/'attention', clock))
        if record['status'] == 'passed':
            entry['flash_phases'] = {name: phase_timing(folder/name, clock)
                                      for name in ('projections', 'attention', 'tail')}
            runtime = sum(p['runtime_cycles'] for p in entry['flash_phases'].values())
            assert runtime == record['runtime_cycles']
            entry.update(flash_runtime_cycles=runtime, flash_runtime_ms=runtime/(clock*1000),
                         relative_l2=record['metrics']['relative_l2'],
                         worst_request_relative_l2=max(r['relative_l2'] for r in record['requests']))
            baseline = json.loads((args.baseline_results/f'batch{batch}'/'result.json').read_text())
            entry.update(previous_baseline_runtime_cycles=baseline['runtime_cycles'],
                         previous_baseline_relative_l2=baseline['metrics']['relative_l2'])
        report['batches'].append(entry)
    destination = args.flash_results/'comparison.json'
    destination.write_text(json.dumps(report, indent=2))
    print(destination)
    for entry in report['batches']:
        print({k: v for k, v in entry.items() if k not in ('flash_phases', 'attention_same_build')})


if __name__ == '__main__':
    main()
