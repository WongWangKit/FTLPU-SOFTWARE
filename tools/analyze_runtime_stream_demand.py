"""Count LPU32 vector-prefill instruction SR demand from runtime CSV + ICU exports.

This is an instruction-demand census, not a live SR-cell/valid-bit simulation.
It expands the CSV repetition patterns, separates the two hemisphere fabrics
from the central VXM fabric, and reports C2C reservation separately from FU use.
SXM block transpose/permute is counted by its external SR interface, not by the
internal transpose-buffer names. VXM operands/results are decoded from its
explicit-SR control word, including byte widths and accumulator emit gating.
"""
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        yield from csv.DictReader(line for line in f if not line.startswith('#'))


def attrs(text):
    return dict(re.findall(r'(\w+)=([^ ]+)', text))


def span(base, width):
    assert 0 <= base < 64 and base % 32 + width <= 32, (base, width)
    return ((1 << width) - 1) << base


def windows(row):
    start, end = int(row['start']), int(row['end'])
    for j in range(int(row['outer_count'])):
        for i in range(int(row['inner_count'])):
            if i == j == 0 and int(row['skip_first']):
                continue
            offset = j * int(row['outer_interval']) + i * int(row['inner_interval'])
            yield start + offset, end + offset


def names(mask):
    return [('E' if i < 32 else 'W') + str(i % 32) for i in range(64) if mask >> i & 1]


def count_bits(array):
    lut = np.array([i.bit_count() for i in range(256)], dtype=np.uint8)
    return lut[array.view(np.uint8).reshape(-1, 8)].sum(axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result_dir', type=Path)
    args = parser.parse_args()
    root = args.result_dir
    programs = {}
    for p in (root / 'icu_programs').glob('*.icu.csv'):
        programs[p.name] = {int(r['pc_word']): r for r in read_csv(p)
                            if r['phase'] == 'execution'}
    rows = list(read_csv(root / 'runtime.pipeline.csv'))
    end = max(b for r in rows for _, b in windows(r)) + 64
    active = np.zeros((3, end), dtype=np.uint64)
    compute = np.zeros_like(active)
    reserved = np.zeros_like(active)
    records = []

    def add(row, fabric, mask, category, *, last_only=False, reserve=False):
        if not mask:
            return
        target = reserved if reserve else active
        for a, b in windows(row):
            if last_only:
                a = b - 1
            target[fabric, a:b] |= np.uint64(mask)
            if not reserve and not category.endswith('.WriteSync'):
                compute[fabric, a:b] |= np.uint64(mask)
        records.append((row, fabric, mask, category, last_only, reserve))

    for r in rows:
        resource = r['resource']
        d = attrs(r['detail'])
        if resource.startswith('MEM.') and 'stream' in d:
            side = int(resource.split('.')[1] == 'W')
            s = d['stream']
            add(r, side, span(int(s[1:]) + (32 if s[0] == 'W' else 0), 1), resource)
        elif resource.startswith('SR.') and resource.endswith('.C2C.Shared'):
            lo, hi = map(int, d['streams'].split('..'))
            add(r, int(resource.split('.')[1] == 'W'), span(32 + lo, hi - lo + 1),
                'C2C reservation', reserve=True)
        elif resource.startswith('MXM.') and not resource.endswith('.Dequant'):
            side, unit = resource.split('.')[1][0], resource.split('.')[1][1:]
            kind = resource.split('.')[2].lower()
            inst = programs[f'mxm_{kind}_{side}_mxm{unit}.icu.csv'][int(d['pc'])]
            x = attrs(inst['details'])
            fabric = int(side == 'W')
            if kind == 'load':
                assert inst['opcode'] in ('LOAD_1D', 'LOAD_2D', 'LOAD_3D', 'LOAD_COLUMN_3D'), inst
                assert 'weight_stream_base' in x, inst
                width = (1 if x['weight_input_mode'] == '0' else 2)
                if inst['opcode'] != 'LOAD_COLUMN_3D':
                    width *= 8
                add(r, fabric, span(int(x['weight_stream_base']), width), resource)
            else:
                if r['detail'].startswith('Compute'):
                    add(r, fabric, span(int(x['activation_stream_base']), 2), resource + '.activation')
                # MEM consumers already identify actual result stream accesses.
                # Include the producer's footprint during stream-destination
                # Compute / accumulator-read instruction activity too.
                if 'destination=stream' in r['detail'] or r['detail'].startswith('AccumulatorRead'):
                    fmt = int(x.get('regular_output_format', x.get('output_format', '0')))
                    if 'destination=stream' in r['detail'] and 'terminal_output_format' in x:
                        fmt = int(x['terminal_output_format'])
                    base = int(x.get('result_stream_base', x.get('stream_base', '0')))
                    add(r, fabric, span(32 + base, 2 if fmt == 1 else 4), resource + '.result')
        elif resource.startswith('SXM.'):
            _, side, kind = resource.split('.')
            inst = programs[f'sxm_{kind.lower()}_{side}.icu.csv'][int(d['pc'])]
            x = attrs(inst['details'])
            if kind == 'Transpose':
                width = 16 if x['input_row'] == 'all' else 2
                add(r, int(side == 'W'), span(int(x['src_base']), width), resource + '.input')
            else:
                width = 16 if x['output_row'] == 'all' else 2
                # all rows in this trace use the block transpose-buffer path.
                assert 'permute_map' in x
                add(r, int(side == 'W'), span(int(x['dst_base']), width), resource + '.output')
        elif resource.startswith('VXM.'):
            alu = int(resource.split('C')[1])
            inst = programs[f'vxm_alu{alu}.icu.csv'][int(d['pc'])]
            control = int(attrs(inst['details'])['compact_control'], 16)
            mask = 0
            op = control & 15
            for kshift, sshift in [(4, 7)] + ([(13, 16)] if op in (1, 2, 3, 5) else []):
                kind = (control >> kshift) & 7
                width = {0: 2, 1: 2, 2: 4, 5: 1}.get(kind, 0)
                if width:
                    mask |= span((control >> sshift) & 63, width)
            add(r, 2, mask, resource + '.input')
            acc_write, acc_emit = (control >> 32) & 1, (control >> 33) & 1
            if not acc_write or acc_emit:
                width = {0: 4, 1: 2, 2: 1, 3: 2}[(control >> 29) & 3]
                add(r, 2, span((control >> 22) & 63, width), resource + '.result',
                    last_only=bool((control >> 59) & 1))

    def witness(fabric, cycle, include_reservations):
        found = []
        for row, f, mask, category, last_only, reserve in records:
            if f != fabric or (reserve and not include_reservations):
                continue
            if any((b - 1 if last_only else a) <= cycle < b for a, b in windows(row)):
                found.append(dict(resource=category, streams=names(mask), detail=row['detail']))
        return found

    report = {'metric': 'runtime active-instruction stream footprint, distinct byte-SR IDs per fabric',
              'source': str(root.resolve()),
              'trace_sha256': hashlib.sha256((root / 'runtime.pipeline.csv').read_bytes()).hexdigest(),
              'limitations': ['Not an exact live-cell occupancy counter: excludes passive in-flight SR data and detailed FU pipeline latency/skew.',
                              'C2C full-window reservation is reported separately; MEM WriteSync reflects executed FU writes.',
                              'Peaks do not prove a smaller physical stream count can execute this program.'],
              'fabrics': {}}
    for f, label in enumerate(('hemisphere_E', 'hemisphere_W', 'VXM')):
        info = {}
        for reserve in (False, True):
            masks = active[f] | reserved[f] if reserve else active[f]
            counts = count_bits(masks)
            cycle = int(counts.argmax())
            info['with_c2c_reservation' if reserve else 'instruction_footprint'] = {
                'peak': int(counts[cycle]), 'first_cycle': cycle,
                'streams': names(int(masks[cycle])),
                'east_peak': int(count_bits(masks & np.uint64(0xffffffff)).max()),
                'west_peak': int(count_bits(masks & np.uint64(0xffffffff00000000)).max()),
                'witness': witness(f, cycle, reserve)}
        info['ever_used'] = names(int(np.bitwise_or.reduce(active[f] | reserved[f])))
        compute_counts = count_bits(compute[f])
        compute_cycle = int(compute_counts.argmax())
        info['excluding_c2c_writes'] = dict(peak=int(compute_counts[compute_cycle]),
            first_cycle=compute_cycle, streams=names(int(compute[f, compute_cycle])))
        report['fabrics'][label] = info
    summed = sum(count_bits(a | r) for a, r in zip(active, reserved))
    cycle = int(summed.argmax())
    report['sum_of_separate_fabric_demands'] = dict(peak=int(summed[cycle]), first_cycle=cycle,
        note='Counts physical segments separately; not the number of SR IDs required by a single fabric.')
    out = root / 'stream_demand.json'
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
