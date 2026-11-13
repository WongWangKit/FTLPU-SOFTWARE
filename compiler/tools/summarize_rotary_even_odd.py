"""Audit completed RoPE runs and their actual runtime traces; no scheduling."""
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def merged_intervals(path, first, last):
    intervals = {'VXM.C0': [], 'VXM.C1': []}
    with path.open(newline='', encoding='utf-8-sig') as stream:
        for row in csv.DictReader(stream):
            if row['resource'] not in intervals:
                continue
            start, end = int(row['start']), int(row['end'])
            n, m = int(row['inner_count']), int(row['outer_count'])
            di, do = int(row['inner_interval']), int(row['outer_interval'])
            if start >= last or end + (n-1)*di + (m-1)*do <= first:
                continue
            for outer in range(m):
                for inner in range(n):
                    if outer*n + inner < int(row['skip_first']):
                        continue
                    delta = outer*do + inner*di
                    a, b = max(first, start+delta), min(last, end+delta)
                    if a < b:
                        intervals[row['resource']].append((a, b))
    result = {}
    for resource, ranges in intervals.items():
        merged = []
        for a, b in sorted(ranges):
            if merged and a <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
            else:
                merged.append((a, b))
        result[resource] = merged
    return result


def occupied(intervals, start, end):
    return sum(max(0, min(b, end)-max(a, start)) for a, b in intervals)


def analyze(directory):
    summary = load(directory/'summary.json')
    assert summary['status'] == 'passed'
    plan = load(directory/'graph_plan.json')
    nodes = [n for n in plan['nodes'] if n.get('native_planar_rotary')]
    assert nodes and all(n.get('tile_pipeline') for n in nodes)
    # This connected test finishes all weight-page consumers before RoPE.
    # Standalone RoPE has no weight-page wait. Do not silently reuse this
    # translation for graphs that page weights inside a measured RoPE region.
    consumers = [int(x) for x in re.findall(r'consumer=(\d+)',
                   (directory/'execution.log').read_text(encoding='utf-8'))]
    assert not consumers or max(consumers) < min(n['start'] for n in nodes)
    timing = summary['timing']
    offset = timing['initial_ingress_cycles'] + timing.get('weight_page_wait_cycles', 0)
    first = min(n['start'] for n in nodes) + offset
    last = max(n['end'] for n in nodes) + offset
    intervals = merged_intervals(directory/'runtime.pipeline.csv', first, last)
    x_domains = x_vectors = 0
    for path in (directory/'runtime_icu_programs').glob('mem_*.icu.csv'):
        with path.open(encoding='utf-8-sig') as stream:
            for row in csv.DictReader(line for line in stream if not line.startswith('#')):
                if row['opcode'] != 'MEM_READ_MACRO':
                    continue
                cycle = int(row['start_cycle'])
                if not any(n['start'] <= cycle < n['end'] for n in nodes):
                    continue
                shape = tuple(int(row[k]) for k in ('count0', 'count1', 'count2'))
                if shape not in ((2, 1, 1), (32, 2, 1)):
                    continue
                x_domains += 1
                x_vectors += int(row['expanded_fu_issues'])
    tiles = []
    for node in nodes:
        for tile in node['tiles']:
            # Runtime VXM rows record the actual ICU configuration start;
            # the C0/C1 chain is launched one cycle before vxm_start.
            start, end = tile['vxm_start']-1+offset, tile['vxm_end']-1+offset
            active = {r: occupied(v, start, end) for r, v in intervals.items()}
            tiles.append(dict(group=node['name'], token_tile=tile['token_tile'],
                slot=tile['slot'], start=start, end=end, span_cycles=end-start,
                active_cycles=active, utilization={r: v/(end-start) for r, v in active.items()}))
    stage_cycles = sum(n['end']-n['start'] for n in nodes)
    stage_active = {r: sum(occupied(v, n['start']+offset, n['end']+offset) for n in nodes)
                    for r, v in intervals.items()}
    return dict(rotary_cycles=stage_cycles,
        stage_active_cycles=stage_active,
        stage_utilization={r: v/stage_cycles for r, v in stage_active.items()},
        result_pitch=sorted(set(n['feature_interval'] for n in nodes)),
        transpose_vectors=sum(n['transpose_input_vectors']+n['transpose_output_vectors'] for n in nodes),
        intermediate_product_mem_writes=sum(n['intermediate_product_mem_writes'] for n in nodes),
        linker_copy_domains=len(plan.get('linker_mem_transports', [])),
        x_read_macro_domains=x_domains, x_read_vectors=x_vectors,
        tiles=tiles, runtime_cycle_offset=offset,
        tile_compute_span_cycles=sum(t['span_cycles'] for t in tiles),
        tile_active_cycles={r: sum(t['active_cycles'][r] for t in tiles) for r in intervals},
        native_rotary_groups=len(nodes))


def compare(before, after):
    old, new = load(before/'summary.json'), load(after/'summary.json')
    for key in ('revision', 'input_sha256', 'weight_sha256', 'gamma_sha256',
                'limits', 'precision', 'raw_macro', 'sequence', 'batch', 'rope_tile_overlap'):
        assert old[key] == new[key], f'changed comparison condition: {key}'
    for key in ('target', 'runner', 'exporter'):
        assert old['signatures'][key] == new['signatures'][key], f'changed {key}'
    stats = [analyze(p) for p in (before, after)]
    assert stats[1]['result_pitch'] == [2]
    assert all(t['span_cycles'] == 64 and all(v == 64 for v in t['active_cycles'].values())
               for t in stats[1]['tiles']), 'runtime VXM has gaps inside a tile'
    for key in ('transpose_vectors', 'intermediate_product_mem_writes', 'linker_copy_domains'):
        assert stats[0][key] == stats[1][key], f'added data movement: {key}'
    assert stats[0]['x_read_vectors'] == stats[1]['x_read_vectors']
    assert stats[0]['x_read_macro_domains'] == 32*stats[1]['x_read_macro_domains']
    assert stats[1]['x_read_macro_domains'] == 4*len(stats[1]['tiles'])
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    repo = Path(__file__).resolve().parents[2]
    command = new['command']
    target = (repo/command[command.index('--target-config')+1]).resolve()
    assert digest(target) == new['signatures']['target'], 'target config changed since execution'
    external_memory = load(target)['external_memory']
    clock = external_memory['lpu_clock_mhz']
    unchanged = {name: digest(before/f'actual.{i}.bin') == digest(after/f'actual.{i}.bin')
                 for i, name in enumerate(new['outputs'])}
    assert all(unchanged.values()), 'optimized outputs changed from the baseline'
    timing = {key: dict(before=old['timing'][key], after=new['timing'][key],
                       saved=old['timing'][key]-new['timing'][key]) for key in old['timing']}
    return dict(status='passed', before=str(before), after=str(after),
        scope=new['scope'], precision=new['precision'], limits=new['limits'], metrics=new['metrics'],
        same_inputs_weights_target_runtime=True, outputs_bitwise_unchanged=unchanged,
        timing_cycles=timing, rotary_before=stats[0], rotary_after=stats[1],
        clock_mhz=clock, external_memory_config=external_memory,
        cold_latency_ms_before=old['timing']['cold_cycles_through_compute']/(clock*1000),
        cold_latency_ms_after=new['timing']['cold_cycles_through_compute']/(clock*1000),
        runtime_pipeline=str(after/'runtime.pipeline.csv'),
        runtime_icu_index=str(after/'runtime_icu_programs/index.csv'),
        reproduction_cwd=str(repo), reproduction_command=command,
        limitations=['Tile boundaries still obey SXM timing and ping-pong buffer reuse.',
            'Instruction-image DMA latency is not timed by the existing runtime in either run.',
            'The test does not include mHC, output projection, FFN or a full decoder layer.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.before.resolve(), args.after.resolve())
    output = args.after/'rope_even_odd_comparison.json'
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(output)
