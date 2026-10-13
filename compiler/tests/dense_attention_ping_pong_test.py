#!/usr/bin/env python3
"""Check bank residency and runtime-prefetch policy through direct lowering."""
import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from generate_dense_decoder_layer import generate


def integer(line, name):
    return int(re.search(rf'\b{name} = (\d+) : i64', line)[1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tool', type=Path, required=True)
    parser.add_argument('--target-config', type=Path, required=True)
    args = parser.parse_args()
    template = ROOT / 'examples/qwen2_5_1_5b_decoder_layer/decoder_layer_seq32.stablehlo.mlir'
    with tempfile.TemporaryDirectory(prefix='attention_ping_pong_') as temporary:
        directory = Path(temporary)
        for name, hidden, intermediate, heads, kv_heads, head_dim in (
            ('llama', 2048, 8192, 32, 8, 64),
            ('phi', 3072, 8192, 24, 8, 128),
        ):
            source = directory / (name + '.stablehlo.mlir')
            source.write_text(generate(template, name=name, hidden=hidden,
                intermediate=intermediate, query_heads=heads, kv_heads=kv_heads,
                head_dim=head_dim, rope_theta=10000, rms_norm_eps=1e-5), encoding='utf-8')
            common = [str(args.tool), '--target-config', str(args.target_config),
                      '--weight-bank', '1', '--ffn-schedule', 'fused',
                      '--attention-schedule', 'fused', '--projection-rope-overlap', 'off']
            stream, schedule, command = [directory / (name + '.' + stage + '.mlir')
                                         for stage in ('stream', 'schedule', 'command')]
            for before, after, pipeline in (
                (source, stream, 'ftlpu-stablehlo-to-stream'),
                (stream, schedule, 'ftlpu-stream-to-schedule'),
                (schedule, command, 'ftlpu-schedule-to-commands'),
            ):
                subprocess.run(common + ['--input', str(before), '--output', str(after),
                                         '--pipeline', pipeline], check=True)
            text = stream.read_text(encoding='utf-8')
            for role, bank in (('query', 1), ('value', 0), ('key', 1), ('output', 0)):
                placements = re.findall(rf'\b{role}_weight = \{{([^{{}}]+)\}}', text)
                assert placements, (name, role)
                assert all(integer(p, 'bank') == bank and 'page_bank_ping_pong = true' in p
                           for p in placements), (name, role, placements)
            pages = [line for line in command.read_text().splitlines()
                     if 'ftlpu.command.weight_page ' in line]
            inputs = {integer(line, 'binding_index'): line for line in pages
                      if integer(line, 'binding_index') in (2, 3, 4, 5)}
            assert len(inputs) == 4, (name, inputs)
            for binding, bank in ((2, 1), (4, 0), (3, 1), (5, 0)):
                line = inputs[binding]
                assert integer(line, 'bank') == bank, line
                assert ('runtime_prefetch = true' in line) == (binding != 2), line
            assert integer(inputs[2], 'release_cycle') < integer(inputs[3], 'ready_cycle')
            assert integer(inputs[4], 'release_cycle') < integer(inputs[5], 'ready_cycle')
            print(name + ': bank alternation and direct-domain prefetch policy passed', flush=True)


if __name__ == '__main__':
    main()
