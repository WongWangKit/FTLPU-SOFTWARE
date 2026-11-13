"""Audit measured dense-decode transport changes; performs no model computation."""
import argparse
import collections
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path(
        'results/glm4_9b_decoder_layer_decode_flash_streamed/baseline_current'))
    args = parser.parse_args()
    rows = []
    for batch in (1, 32):
        folder = args.results/f'batch{batch}'
        before = json.loads((folder/'before_transport_optimization.json').read_text())
        after = json.loads((folder/'result.json').read_text())
        assert before['status'] == after['status'] == 'passed'
        for field in ('runner_sha256', 'target_sha256', 'fixture_fingerprint',
                      'projection_output_sha256', 'past_kv_sha256'):
            assert before[field] == after[field], f'batch{batch}: {field} changed'
        assert before['phase']['semantic_graph_sha256'] == after['phase']['semantic_graph_sha256']
        assert before['phase']['output_sha256'] == after['phase']['output_sha256'], 'output bytes changed'
        active = collections.Counter()
        with (folder/'attention/runtime.pipeline.csv').open(newline='') as stream:
            for row in csv.DictReader(stream):
                if row['resource'].startswith('MXM.') and row['resource'].endswith('.Compute'):
                    active[row['resource']] += ((int(row['end'])-int(row['start']))
                        * int(row['inner_count'] or 1)*int(row['outer_count'] or 1))
        old, new = before['phase']['timing'], after['phase']['timing']
        if 'mxm_compute_issue_cycles' in before:
            assert dict(active) == before['mxm_compute_issue_cycles'], 'MXM work changed'
        result = dict(batch=batch, outputs_bit_exact=True, inputs_runner_target_unchanged=True,
            before=old, after=new,
            before_transports=before['linker_transport_summary'],
            after_transports=after['phase']['linker_transport_summary'],
            latency_reduction_percent=100*(old['runtime_cycles']-new['runtime_cycles'])/old['runtime_cycles'],
            mxm_compute_issue_cycles=dict(active),
            mxm_compute_timeline_coverage_before={k:v/old['runtime_cycles'] for k,v in active.items()},
            mxm_compute_timeline_coverage_after={k:v/new['runtime_cycles'] for k,v in active.items()})
        rows.append(result)
        print(batch, new['runtime_cycles'], result['latency_reduction_percent'],
              result['after_transports'], flush=True)
    report = dict(scope='dense GLM-4-9B decode attention core plus KV append, past32, decode1',
        results=rows, notes=[
            'Linker transport counts exclude native softmax intermediate storage and SXM buffers.',
            'MXM coverage counts Compute issue cycles in the runtime timeline, not useful MAC-lane utilization.',
            'Runtime includes C2C input and full-cache/context readback, but not instruction-image load latency.'])
    (args.results/'transport_optimization.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
