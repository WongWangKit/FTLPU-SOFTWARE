"""Independent causal/GQA/hemisphere checks of the planar Flash tile scheduler."""
import argparse
import csv
import json
from pathlib import Path
import re
import sys
import numpy as np

from activation_batch_matmul_test import LogicalGraph, bf16
from attention_graph import flash_attention
from dense_attention_core_test import reference
from glm4_attention_comparison_test import execute
from glm4_decode_test import errors
from glm4_flash_packed_u_test import sha
from model_test_outputs import OutputContract, TestOutput


def verify_shared_exponent_read(directory, batch, heads):
    # Native FlashTileMemory's FP32 exponent home is slices 12..15, bank 0.
    # Count actual FU reads (both hemispheres), not merely a compiler marker.
    # One read per exponent must supply BOTH BF16 P and the FP32 reduction.
    read_bytes = 0
    with (directory/'runtime.pipeline.csv').open(newline='') as stream:
        for row in csv.DictReader(stream):
            if row['resource'] not in ('MEM.E.Read', 'MEM.W.Read'):
                continue
            address = re.search(r'slice=(\d+) bank=(\d+)', row['detail'])
            if not address or not 12 <= int(address[1]) <= 15 or int(address[2]) != 0:
                continue
            assert int(row['skip_first']) == 0
            read_bytes += (int(row['end'])-int(row['start'])) * 32 \
                * int(row['inner_count']) * int(row['outer_count'])
    expected = batch * heads * 32 * 32 * 4
    assert read_bytes == expected, (read_bytes, expected)
    return dict(exponent_read_bytes=read_bytes, exponent_read_passes=1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ('build-dir', 'target-config', 'output-dir'):
        p.add_argument('--'+arg, type=Path, required=True)
    p.add_argument('--cases', nargs='+', default=['gqa128', 'mha32', 'gqa64', 'batch2'])
    p.add_argument('--baseline-dir', type=Path,
                   help='optional accepted separate-pass results for bitwise comparison')
    a = p.parse_args()
    a.build_dir, a.target_config, a.output_dir = (
        v.resolve() for v in (a.build_dir, a.target_config, a.output_dir))
    ext = '.exe' if sys.platform == 'win32' else ''
    provenance = dict(compiler_sha256=sha(a.build_dir/'compiler'/('ftlpu-compile'+ext)),
                      runner_sha256=sha(a.build_dir/'runtime'/('compiled_stage_file_runner'+ext)),
                      clock_mhz=json.loads(a.target_config.read_text())['external_memory']['lpu_clock_mhz'])
    shapes = dict(gqa128=(1, 4, 2, 128, True), mha32=(1, 2, 2, 32, False),
                  gqa64=(1, 8, 4, 64, True), batch2=(2, 4, 2, 128, True))
    records = []
    for case in a.cases:
        batch, heads, kvheads, dim, causal = shapes[case]
        rng = np.random.default_rng(32128)
        q = bf16(rng.normal(0, .5, (batch, 32, heads, dim)).astype('f4'))
        k = bf16(rng.normal(0, .5, (batch, 32, kvheads, dim)).astype('f4'))
        # Distinct tokens, KV groups and requests expose routing or mask errors.
        v = bf16(rng.normal(0, .5, k.shape).astype('f4')
                 + np.arange(32, dtype='f4')[None, :, None, None]/4
                 + np.arange(kvheads, dtype='f4')[None, None, :, None]
                 + np.arange(batch, dtype='f4')[:, None, None, None]*8)
        expected = reference(q, k, v, dim**-.5, causal, 0)
        graph = LogicalGraph()
        result = flash_attention(graph, 'attention', q, k, v, causal=causal, query_position=0)
        directory = a.output_dir/case
        (actual,), phase = execute(a, batch, 'flash',
            (graph, [result.reshape(batch*32, -1)]), directory, provenance,
            output_contract=OutputContract(TestOutput('context', 'diagnostic')))
        actual = actual.reshape(expected.shape)
        metrics = errors(actual, expected)
        assert np.isfinite(actual).all() and metrics['relative_l2'] < .03, metrics
        if causal:
            first = errors(actual[:, 0], np.repeat(v[:, 0], heads//kvheads, axis=1))
            assert first['relative_l2'] < .01, first
        plan = json.loads((directory/'graph_plan.json').read_text())
        assert len(plan['nodes']) == 1
        assert all(n['attention_core_schedule'] == 'flash_tile_pipeline' for n in plan['nodes'])
        assert not plan['linker_mem_transports'], plan['linker_mem_transports']
        record = dict(status='passed', case=case, shape=[batch, 32, heads, kvheads, dim],
                      causal=causal, metrics=metrics, phase=phase, **provenance)
        record.update(verify_shared_exponent_read(directory, batch, heads))
        if a.baseline_dir:
            previous = a.baseline_dir/case/'diagnostic/actual.0.bin'
            current = directory/'diagnostic/actual.0.bin'
            assert current.read_bytes() == previous.read_bytes(), case
            record['bit_identical_to_separate_passes'] = True
        (directory/'result.json').write_text(json.dumps(record, indent=2)+'\n')
        records.append(record)
        print(case, metrics, flush=True)
    (a.output_dir/'summary.json').write_text(json.dumps(dict(status='passed', cases=records), indent=2)+'\n')


if __name__ == '__main__':
    main()
