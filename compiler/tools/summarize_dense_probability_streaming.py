"""Check numerical identity and actual traffic for dense VXM-to-SXM P streaming."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import re


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path(
        'results/glm4_9b_decoder_layer_decode_flash_streamed/baseline_current'))
    args = parser.parse_args()
    results = []
    for batch in (1, 32):
        folder = args.results / f'batch{batch}'
        before = json.loads((folder / 'before_p_streaming.json').read_text())
        after = json.loads((folder / 'result.json').read_text())
        assert before['status'] == after['status'] == 'passed'
        for field in ('runner_sha256', 'target_sha256', 'fixture_fingerprint',
                      'projection_output_sha256', 'past_kv_sha256', 'precision'):
            assert before[field] == after[field], field
        assert before['phase']['semantic_graph_sha256'] == after['phase']['semantic_graph_sha256']
        assert before['phase']['output_sha256'] == after['phase']['output_sha256']
        phase = folder / 'attention'
        assert sha(phase / 'actual.runtime.ftlpu') == after['phase']['actual_program_sha256']
        assert sha(phase / 'runtime.pipeline.csv') == after['phase']['runtime_pipeline_sha256']
        for i, expected in enumerate(after['phase']['output_sha256']):
            assert sha(phase / f'actual.{i}.bin') == expected
        schedule = (phase / 'decoder_layer.schedule.mlir').read_text()
        windows = re.findall(r'name = "[^"]*softmax\.vxm_sxm_direct"', schedule)
        assert len(windows) == batch * 32, 'missing streamed probability head'
        assert 'name = "attention.probability_pack"' not in schedule
        assert not re.search(r'name = "[^"]*\.probability_transpose"', schedule)
        issues = collections.Counter()
        with (phase / 'runtime.pipeline.csv').open(newline='') as source:
            for row in csv.DictReader(source):
                if row['resource'].startswith('MEM.'):
                    count = re.search(r'issues=(\d+)', row['detail'])
                    assert count, 'missing actual runtime issue count'
                    issues[row['resource']] += int(count[1])
        old, new = (r['phase']['timing'] for r in (before, after))
        result = dict(batch=batch, output_and_cache_bit_exact=True,
                      inputs_runner_target_unchanged=True, before=old, after=new,
                      streaming_head_windows=len(windows), probability_pack_materialized=False,
                      mem_runtime_issues=dict(issues),
                      latency_reduction_percent=100 * (old['runtime_cycles'] - new['runtime_cycles']) / old['runtime_cycles'])
        if batch == 32:
            saved = json.loads((args.results / 'before_p_streaming_mem_counts.json').read_text())
            assert saved['runtime_pipeline_sha256'] == before['phase']['runtime_pipeline_sha256']
            assert saved['actual_program_sha256'] == before['phase']['actual_program_sha256']
            old_counts = saved['counts']
            result['mem_runtime_issues_before'] = old_counts
            result['mem_runtime_issues_saved'] = {
                key: old_counts[key] - issues[key] for key in old_counts}
        results.append(result)
        print(batch, old['runtime_cycles'], new['runtime_cycles'],
              result['latency_reduction_percent'], 'bit-exact', flush=True)
    report = dict(scope='GLM-4-9B dense decode attention core + KV append; past32, decode1',
                  results=results,
                  notes=['Actual runtime issue counts, including native operator MEM activity; not linker-copy counts.',
                         'P remains BF16. The transposed P allocation is retained for PV reuse.',
                         'Runtime includes C2C input and context/full-cache readback; iMEM load/reload transfer is not timed.'])
    (args.results / 'probability_streaming.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
