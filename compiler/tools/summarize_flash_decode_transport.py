"""Audit Flash decode transport cleanup against the saved accepted run."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def traffic(transfers):
    # Each byte plane is one 32-byte physical MEM word. A linker transfer
    # reads and writes it; this is on-chip traffic, not external DDR bytes.
    phases = Counter()
    for entry in transfers:
        phases[entry['phase']] += entry['vectors'] * entry['byte_planes'] * 32 * 2
    return dict(read_plus_write_bytes=sum(phases.values()), by_phase=dict(phases))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, default=Path('results/glm4_9b_decoder_layer_decode_flash_streamed'))
    p.add_argument('--before', type=Path, default=Path('tmp/glm4_flash_decode_tight_1111/before.json'))
    args = p.parse_args()
    before = json.loads(args.before.read_text())
    rows = []
    for batch in (1, 32):
        folder = args.results / f'batch{batch}'
        saved = before[str(batch)]
        old = saved['result']
        new = json.loads((folder / 'result.json').read_text())
        assert old['status'] == new['status'] == 'passed'
        for key in ('runner_sha256', 'target_sha256', 'fixture_fingerprint',
                    'projection_output_sha256', 'past_kv_sha256', 'precision'):
            assert old[key] == new[key], key
        assert old['phase']['semantic_graph_sha256'] == new['phase']['semantic_graph_sha256']
        assert old['phase']['output_sha256'] == new['phase']['output_sha256']
        phase = folder / 'attention'
        assert sha(phase / 'actual.runtime.ftlpu') == new['phase']['actual_program_sha256']
        assert sha(phase / 'runtime.pipeline.csv') == new['phase']['runtime_pipeline_sha256']
        for i, digest in enumerate(new['phase']['output_sha256']):
            assert sha(phase / f'actual.{i}.bin') == digest
        plan = json.loads((phase / 'graph_plan.json').read_text())
        assert any(node.get('tight_serial_schedule') for node in plan['nodes'])
        transports = plan['linker_mem_transports']
        assert not any(e['phase'].endswith(('.pv.outer', '.normalize')) for e in transports), \
            'coefficient/repeated-V staging unexpectedly returned'
        a, b = (r['phase']['timing'] for r in (old, new))
        ta, tb = traffic(saved['transports']), traffic(transports)
        assert b['runtime_cycles'] < a['runtime_cycles']
        assert tb['read_plus_write_bytes'] < ta['read_plus_write_bytes']
        rows.append(dict(batch=batch, outputs_and_caches_bit_exact=True,
                         before=a, after=b, linker_mem_transports_before=ta,
                         linker_mem_transports_after=tb,
                         total_latency_reduction_percent=100*(1-b['runtime_cycles']/a['runtime_cycles']),
                         linker_transport_reduction_percent=100*(1-tb['read_plus_write_bytes']/ta['read_plus_write_bytes'])))
        (folder / 'before_transport_cleanup.json').write_text(json.dumps(saved, indent=2)+'\n')
        print(batch, a['runtime_cycles'], b['runtime_cycles'], ta['read_plus_write_bytes'], tb['read_plus_write_bytes'])
    report = dict(scope='GLM-4-9B Flash decode attention core + KV append, past32/decode1, batch1/32',
                  results=rows,
                  notes=['Raw-macro numerical execution; BF16 storage boundaries and FP32 accumulation unchanged.',
                         'Transport byte counts cover linker MEM-to-MEM copies, including validation exports, not every native MEM instruction.',
                         'Runtime includes initial C2C ingress and output/cache diagnostic readback. iMEM load/reload transfer latency is not modeled.',
                         'Native FU/stream drain and dependency waits remain; this is not a claim of zero idle cycles.'])
    (args.results / 'transport_cleanup.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
