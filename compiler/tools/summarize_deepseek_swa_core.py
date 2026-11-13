"""Analyze completed SWA artifacts; never generates device layouts or schedules."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--frontend', type=Path, required=True)
    args = parser.parse_args()
    result, before = args.result, args.frontend
    read = lambda path: json.loads(path.read_text(encoding='utf-8'))
    current, previous = read(result/'summary.json'), read(before/'summary.json')
    assert current['status'] == previous['status'] == 'passed'
    for key in ('input_sha256', 'weight_sha256', 'gamma_sha256', 'batch', 'sequence', 'precision'):
        assert current[key] == previous[key], key
    for key in ('target', 'runner', 'exporter'):
        assert current['signatures'][key] == previous['signatures'][key], key
    unchanged = {}
    for index, name in enumerate(previous['outputs']):
        unchanged[name] = (result/f'actual.{index}.bin').read_bytes() == (before/f'actual.{index}.bin').read_bytes()
    assert all(unchanged.values()), unchanged
    plan = read(result/'graph_plan.json')
    cores = [node for node in plan['nodes'] if node.get('attention_core_schedule') == 'flash_tile_pipeline']
    exports = [item for item in plan['linker_mem_transports'] if item['reason'] == 'graph_result_export']
    rope_end = max(node['end'] for node in plan['nodes'] if node.get('native_planar_rotary'))
    core_end = max(node['nominal_end'] for node in cores)
    ranges = dict(after_rope=rope_end, first_core_compute=min(node['nominal_start'] for node in cores),
                  last_core_compute_end=core_end,
                  core_native_span_sum=sum(node['nominal_end']-node['nominal_start'] for node in cores),
                  rope_to_core_end_including_transports=core_end-rope_end)
    if exports:
        ranges['diagnostic_context_pack_start'] = min(item['start'] for item in exports)
        ranges['diagnostic_context_pack_end'] = max(item['end'] for item in exports)
        ranges['diagnostic_context_pack_span'] = ranges['diagnostic_context_pack_end']-ranges['diagnostic_context_pack_start']
    offset = current['timing']['initial_ingress_cycles'] + current['timing']['weight_page_wait_cycles']
    record = dict(status='passed', scope=current['scope'],
        comparison_note='The after test adds SWA; its total time is not an optimization speedup comparison.',
        upstream_outputs_bitwise_unchanged=unchanged, core_metrics=current['metrics']['swa_context'],
        timing=current['timing'], compiler_cycle_boundaries=ranges,
        runtime_core_zoom=dict(start=rope_end+offset, end=core_end+offset),
        runtime_offset_note='All dynamic weight-page waits precede SWA; offset adds ingress and those waits to compiler boundaries.',
        passive_transport_domains=len(plan['linker_mem_transports']),
        diagnostic_context_pack_domains=len(exports),
        diagnostic_context_pack_vectors=sum(item['vectors'] for item in exports),
        caveats=['No inverse RoPE, output projection, cache persistence, mHC or FFN.',
                 f'KV is prepared once per native core group ({len(cores)} groups); cross-group residency is not yet implemented.',
                 'Runtime invocation includes diagnostic context packing; HostOutput readback is listed separately.',
                 'Instruction-image DMA is not timed.'])
    (result/'core_comparison.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
