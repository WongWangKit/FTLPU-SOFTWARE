"""Compare a saved baseline with an executed projection/solve schedule.

This tool reads artifacts only; it never imports, lowers or executes a model.
By default it writes projection_solve_overlap_comparison.json in --result-dir.
Use --baseline and --output-name to compare subsequent optimization rounds.
"""
import argparse
from bisect import bisect_right
import csv
import hashlib
import json
from pathlib import Path
import re


def merge(spans):
    result = []
    for begin, end in sorted(spans):
        if result and begin <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((begin, end))
    return result


def intersection(left, right):
    i = j = 0
    result = []
    while i < len(left) and j < len(right):
        begin, end = max(left[i][0], right[j][0]), min(left[i][1], right[j][1])
        if begin < end:
            result.append((begin, end))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return merge(result)


def expanded(row):
    for outer in range(int(row['outer_count'])):
        for inner in range(int(row['inner_count'])):
            if outer == inner == 0 and int(row['skip_first']):
                continue
            shift = outer * int(row['outer_interval']) + inner * int(row['inner_interval'])
            yield int(row['start']) + shift, int(row['end']) + shift


def issue_spans(rows, predicate, window=None):
    result = []
    for row in rows:
        if not predicate(row):
            continue
        for begin, end in expanded(row):
            if window is not None:
                begin, end = max(begin, window[0]), min(end, window[1])
            if begin < end:
                result.append((begin, end))
    return merge(result)


def duration(spans):
    return sum(end - begin for begin, end in spans)


def trace_end(rows):
    # The stored end belongs to the first issue in a repeated domain.
    return max(int(row['end']) + (int(row['inner_count']) - 1) * int(row['inner_interval'])
               + (int(row['outer_count']) - 1) * int(row['outer_interval']) for row in rows)


def epoch_offsets(rows):
    result = []
    for row in rows:
        if row['resource'] != 'Paging.Consumer':
            continue
        logical = re.search(r'\blogical_cycle=(\d+)\b', row['detail'])
        if not logical:
            raise ValueError('Paging.Consumer is missing its logical cycle')
        cycle = int(logical[1])
        result.append((cycle, int(row['start']) - cycle))
    if not result:
        raise ValueError('Actual Paging.Consumer events are required for stage clock mapping')
    return sorted(result)


def physical_range(node, offsets):
    def physical(cycle):
        for consumer, delay in reversed(offsets):
            if consumer <= cycle:
                return cycle + delay
        raise ValueError(f'No runtime clock offset before logical cycle {cycle}')
    begin, end = node['nominal_start'], node['nominal_end']
    # Do not charge the next consumer's wait to a half-open previous interval.
    return [physical(begin), physical(end - 1) + 1 if end > begin else physical(end)]


def baseline_metrics(summary):
    runtime = summary['runtime']
    logical = int(runtime['cycles'])
    wait = int(runtime['weight_page_runtime_wait_cycles'])
    solve = summary.get('chunkwise_solve', {})
    diagnostic = summary.get('diagnostic_host_readback_cycles',
                             solve.get('diagnostic_host_readback_cycles'))
    if diagnostic is None:
        raise ValueError('Saved baseline has no diagnostic readback timing')
    if 'device_session_end_cycle' in solve:
        cold = int(solve['device_session_end_cycle'])
        source = 'saved chunkwise_solve.device_session_end_cycle'
    else:
        cold = int(summary['runtime_trace_cycles']) - int(diagnostic)
        source = 'saved runtime_trace_cycles minus diagnostic_host_readback_cycles'
    initial = cold - logical - wait
    if initial < 0:
        raise ValueError('Saved baseline clock fields are inconsistent')
    return dict(logical_cycles=logical, external_wait_cycles=wait,
        invocation_cycles=logical + wait, cold_cycles_through_compute=cold,
        cold_cycles_source=source,
        initial_prefix_cycles=initial,
        initial_prefix_cycles_source='cold minus logical cycles minus runtime page waits',
        host_input_active_cycles=None, host_input_payload_bytes=None,
        missing_baseline_evidence='Baseline snapshot did not retain HostInput events or input byte counts; initial prefix is reconstructed, not an independent HostInput sum',
        diagnostic_host_readback_cycles=int(diagnostic),
        full_trace_cycles=int(summary['runtime_trace_cycles']),
        dynamic_weight_bytes=int(runtime['weight_page_prefetch_bytes']),
        dynamic_weight_pages=int(runtime['weight_page_prefetches']),
        execution_epochs=int(runtime['execution_epochs']),
        host_operations=int(runtime['host_operations']))


def current_metrics(summary, rows):
    runtime = summary['runtime']
    invocations = [row for row in rows if row['resource'] == 'Session.Invocation']
    if len(invocations) != 1:
        raise ValueError('Expected one continuous GDN attention invocation')
    invocation = invocations[0]
    begin, end = int(invocation['start']), int(invocation['end'])
    cold_start = min(int(row['start']) for row in rows)
    host_inputs = [row for row in rows if row['resource'] == 'C2C.HostInput']
    payloads = {}
    for row in host_inputs:
        binding = re.search(r'\bbinding=(\d+)\b', row['detail'])
        size = re.search(r'\bbytes=(\d+)\b', row['detail'])
        if not binding or not size:
            raise ValueError('HostInput event is missing its binding/payload bytes')
        key, value = int(binding[1]), int(size[1])
        if key in payloads and payloads[key] != value:
            raise ValueError('Repeated HostInput binding has inconsistent payload size')
        payloads[key] = value
    logical, wait = int(runtime['cycles']), int(runtime['weight_page_runtime_wait_cycles'])
    if end - begin != logical + wait:
        raise ValueError('Actual invocation time does not equal logical time plus runtime waits')
    return dict(logical_cycles=logical, external_wait_cycles=wait,
        invocation_cycles=end - begin, cold_cycles_through_compute=end - cold_start,
        cold_start_cycle=cold_start, invocation_range=[begin, end],
        cold_cycles_source='actual trace origin to Session.Invocation.end',
        initial_prefix_cycles=begin - cold_start,
        host_input_active_cycles=duration(issue_spans(rows, lambda row: row['resource'] == 'C2C.HostInput')),
        host_input_payload_bytes=sum(payloads.values()),
        host_input_payload_bytes_scope='Unique uploaded binding payloads; excludes physical layout mirroring/padding and dynamic weight pages',
        host_input_transfer_events=len(host_inputs),
        diagnostic_host_readback_cycles=duration(issue_spans(rows, lambda row: row['resource'] == 'C2C.HostOutput')),
        full_trace_cycles=trace_end(rows) - cold_start,
        dynamic_weight_bytes=int(runtime['weight_page_prefetch_bytes']),
        dynamic_weight_pages=int(runtime['weight_page_prefetches']),
        execution_epochs=int(runtime['execution_epochs']),
        host_operations=int(runtime['host_operations']))


def pair_metrics(plan, rows):
    offsets = epoch_offsets(rows)
    nodes = {node['name']: node for node in plan['nodes']}
    result = []
    for pair in plan.get('projection_solve_overlap', []):
        if pair.get('projection') not in nodes or pair.get('solve') not in nodes:
            raise ValueError(f'Overlap plan references an unknown node: {pair}')
        projection = nodes[pair['projection']]
        solve = nodes[pair['solve']]
        projection_range, solve_range = physical_range(projection, offsets), physical_range(solve, offsets)
        compute = lambda row: row['resource'].startswith('MXM.') and row['resource'].endswith('.Compute')
        arithmetic = lambda row: row['resource'].startswith('VXM.') and re.match(
            r'^(mul|multiply|add|sub|subtract|negate)\b', row['detail'])
        mxm = issue_spans(rows, compute, projection_range)
        vxm = issue_spans(rows, arithmetic, solve_range)
        overlap = intersection(mxm, vxm)
        units = sorted({row['resource'] for row in rows if compute(row)})
        per_unit = {}
        for unit in units:
            spans = issue_spans(rows, lambda row: row['resource'] == unit, projection_range)
            if spans:
                per_unit[unit] = dict(active_cycles=duration(spans),
                    envelope=[spans[0][0], spans[-1][1]], segments=spans)
        result.append(dict(projection=pair['projection'], solve=pair['solve'],
            compiler_status=pair.get('status'), projection_runtime_range=projection_range,
            solve_runtime_range=solve_range,
            stage_window_overlap_cycles=max(0, min(projection_range[1], solve_range[1])
                - max(projection_range[0], solve_range[0])),
            projection_mxm_union_active_cycles=duration(mxm),
            solve_vxm_arithmetic_union_active_cycles=duration(vxm),
            actual_arithmetic_overlap_cycles=duration(overlap),
            actual_arithmetic_overlap_envelope=[overlap[0][0], overlap[-1][1]] if overlap else None,
            actual_arithmetic_overlap_segments=overlap,
            projection_mxm_by_unit=per_unit,
            arithmetic_scope='Physical MXM Compute issues in projection window intersected with VXM arithmetic issues in solve window; bypass excluded; hemispheres/columns unioned'))
    return result


def phase_metrics(plan, rows):
    """Account for QKV pages and incoming-State GEMMs, including their staging.

    Stage names identify compiler graph operations, not test paths or cycles.
    FU counts are activity inside each actual stage window, not an assertion
    that every issue in an overlapping window belongs exclusively to that node.
    """
    offsets = epoch_offsets(rows)
    groups = {}
    for node in plan['nodes']:
        name = node['name']
        if re.search(r'(?:^|\.)in_proj_qkv(?:\.page\d+)?$', name):
            group, kind = name, 'qkv_projection'
        else:
            match = re.match(r'^(.*\bincoming_(qs|ks))(?:\.|$)', name)
            if not match:
                continue
            group, kind = match[1], 'incoming_state_' + match[2]
        entry = groups.setdefault(group, dict(kind=kind, nodes=[], transports=[]))
        entry['nodes'].append(node)
    for name, entry in groups.items():
        entry['transports'] = [item for item in plan.get('linker_mem_transports', [])
            if item['phase'] == name or item['phase'].startswith(name + '.')]

    result = []
    is_compute = lambda row: row['resource'].startswith('MXM.') and row['resource'].endswith('.Compute')
    is_arithmetic = lambda row: row['resource'].startswith('VXM.') and not re.match(r'^bypass\b', row['detail'])
    units = sorted({row['resource'] for row in rows if is_compute(row)})
    def group_start(entry):
        return min([n['nominal_start'] for n in entry['nodes'] if 'nominal_start' in n]
                   + [item['start'] for item in entry['transports']])
    for name, entry in sorted(groups.items(), key=lambda item: group_start(item[1])):
        nodes, transports = entry['nodes'], entry['transports']
        scheduled_nodes = [node for node in nodes if 'nominal_start' in node and 'nominal_end' in node]
        compute_begin = min(node['nominal_start'] for node in scheduled_nodes)
        compute_end = max(node['nominal_end'] for node in scheduled_nodes)
        begin = min([compute_begin] + [item['start'] for item in transports])
        end = max([compute_end] + [item['end'] for item in transports])
        runtime_range = physical_range(dict(nominal_start=begin, nominal_end=end), offsets)
        # Avoid expanding repeated issue domains wholly outside this phase.
        window_rows = [row for row in rows if int(row['start']) < runtime_range[1]
            and int(row['end']) + (int(row['inner_count']) - 1) * int(row['inner_interval'])
            + (int(row['outer_count']) - 1) * int(row['outer_interval']) > runtime_range[0]]
        active = issue_spans(window_rows, is_compute, runtime_range)
        by_unit = {}
        for unit in units:
            spans = issue_spans(window_rows, lambda row: row['resource'] == unit, runtime_range)
            if spans:
                by_unit[unit] = dict(active_cycles=duration(spans),
                    envelope=[spans[0][0], spans[-1][1]],
                    inactive_cycles_inside_envelope=spans[-1][1] - spans[0][0] - duration(spans),
                    interval_count=len(spans))
        result.append(dict(name=name, kind=entry['kind'], node_count=len(nodes),
            scheduled_node_count=len(scheduled_nodes),
            nominal_node_range=[compute_begin, compute_end],
            nominal_range_including_staging=[begin, end], nominal_cycles_including_staging=end-begin,
            runtime_range_including_staging=runtime_range,
            runtime_cycles_including_staging=runtime_range[1]-runtime_range[0],
            linker_mem_transport_domains=len(transports),
            linker_mem_transport_vectors=sum(int(item['vectors']) for item in transports),
            linker_mem_transport_byte_plane_vectors=sum(int(item['vectors'])*int(item['byte_planes']) for item in transports),
            mxm_compute_union_active_cycles_in_window=duration(active),
            vxm_non_bypass_union_active_cycles_in_window=duration(issue_spans(window_rows, is_arithmetic, runtime_range)),
            mxm_by_unit_in_window=by_unit))
    return result


def state_tile_preparation(plan):
    """Account for compiler-created operand views and explicit SRAM copies per head pair."""
    tile_name = re.compile(r'^(.*\bincoming_(?:qs|ks))(?:\.col(\d+))?\.batch(\d+)\.row(\d+)\.k(\d+)(?:\.(lhs|rhs))?$')
    copies = {}
    for transfer in plan.get('linker_mem_transports', []):
        copies.setdefault(transfer['phase'], []).append(transfer)
    stages = {}
    for node in plan['nodes']:
        match = tile_name.fullmatch(node['name'])
        if not match:
            continue
        name, column, head, row, reduction, operand = match.groups()
        stage = stages.setdefault(name, dict(by_head_pair={}, direct_lhs_tiles=[]))
        counts = stage['by_head_pair'].setdefault(head, dict(
            products=0, direct_lhs_views=0, lhs_preparations=0, rhs_preparations=0,
            lhs_copy_domains=0, rhs_copy_domains=0, lhs_copy_vectors=0, rhs_copy_vectors=0))
        if operand is None:
            counts['products'] += 1
            continue
        transfers = copies.get(node['name'], [])
        counts[operand + '_preparations'] += 1
        counts[operand + '_copy_domains'] += len(transfers)
        counts[operand + '_copy_vectors'] += sum(item['vectors'] for item in transfers)
        if node.get('native_mxm_lhs_view'):
            if operand != 'lhs' or transfers:
                raise ValueError(f"Direct LHS tile unexpectedly has a staging copy: {node['name']}")
            counts['direct_lhs_views'] += 1
            stage['direct_lhs_tiles'].append(dict(
                head=int(head), row=int(row), k=int(reduction), column=int(column or 0),
                evidence={key: value for key, value in node.items()
                          if key not in ('name', 'node', 'layout_join')}))
    for stage in stages.values():
        members = list(stage['by_head_pair'].values())
        stage['totals'] = {key: sum(member[key] for member in members) for key in members[0]}
    return stages


def state_tile_changes(baseline_phases, current_phases, before_tiles, after_tiles):
    previous = {phase['name']: phase for phase in baseline_phases}
    current = {phase['name']: phase for phase in current_phases}
    def changed(before, after):
        return dict(before=before, after=after, change=after-before)
    delta = {}
    for name in sorted(before_tiles.keys() & after_tiles.keys()):
        if name not in previous or name not in current:
            continue
        old, new = before_tiles[name], after_tiles[name]
        head_delta = {}
        for head in sorted(old['by_head_pair'].keys() & new['by_head_pair'].keys(), key=int):
            a, b = old['by_head_pair'][head], new['by_head_pair'][head]
            head_delta[head] = {key: b[key]-value for key, value in a.items()}
        delta[name] = dict(
            totals={key: changed(value, new['totals'][key]) for key, value in old['totals'].items()},
            change_by_head_pair=head_delta,
            runtime_cycles=changed(previous[name]['runtime_cycles_including_staging'],
                                   current[name]['runtime_cycles_including_staging']),
            physical_mxm_compute_cycles=changed(previous[name]['mxm_compute_union_active_cycles_in_window'],
                                                current[name]['mxm_compute_union_active_cycles_in_window']))
    return delta


def audit_direct_lhs(result_dir, plan, operands):
    """Cross-check planned aliases against MEM read ports, slices, rows and 3D strides."""
    windows = []
    for stage, entry in operands.items():
        views = {(t['head'], t['row'], t['k']): t['evidence'] for t in entry['direct_lhs_tiles']}
        if not views:
            continue
        for node in plan['nodes']:
            match = re.fullmatch(re.escape(stage) + r'(?:\.col\d+)?\.batch(\d+)\.row(\d+)\.k(\d+)', node['name'])
            if match and (key := tuple(map(int, match.groups()))) in views:
                windows.append((node['nominal_start'], node['nominal_end'], stage, views[key], set()))
    if not windows:
        return {}
    windows.sort(key=lambda window: window[0])
    starts = [window[0] for window in windows]
    numbers = re.compile(r'(\w+) = (-?\d+) : i64')
    issues = {}
    with (result_dir/'decoder_layer.schedule.mlir').open(encoding='utf8') as schedule:
        for line in schedule:
            if ('ftlpu.schedule.mem_transfer' not in line or 'opcode = "read"' not in line or
                not ('packed_stream = 4 :' in line or 'packed_stream = 5 :' in line)):
                continue
            attrs = {key: int(val) for key, val in numbers.findall(line)}
            index = bisect_right(starts, attrs['cycle'])-1
            if index < 0:
                continue
            _, end, stage, source, observed = windows[index]
            if attrs['cycle'] >= end:
                continue
            plane = attrs['packed_stream']-4
            if (attrs['slice'] not in source['source_slices'][plane::2] or
                attrs['bank'] != source['source_bank'] or attrs['address'] not in source['head_base_rows'] or
                (attrs['repeat_count'],attrs['repeat_interval'],attrs['address_stride']) != (4,8,1) or
                (attrs['wave_count'] > 1 and (attrs['wave_interval'],attrs['wave_address_stride']) != (32,0))):
                raise ValueError(f'Unexpected direct LHS read in {stage} at logical cycle {attrs["cycle"]}: {attrs}')
            observed.add((attrs['hemisphere'], attrs['slice'], attrs['address']))
            issues[stage] = issues.get(stage, 0) + attrs['repeat_count']*attrs['wave_count']*attrs['group_count']
    audit = {}
    for begin, _, stage, source, observed in windows:
        wanted_slices = set(source['source_slices'])
        for side in (0,1):
            side_reads = [(s,a) for h,s,a in observed if h == side]
            if {s for s,_ in side_reads} != wanted_slices or len({a for _,a in side_reads}) != 1:
                raise ValueError(f'Direct LHS read coverage differs from producer layout in {stage} at {begin}')
        if {a for _,_,a in observed} != set(source['head_base_rows']):
            raise ValueError(f'Direct LHS reads do not cover both source heads in {stage} at {begin}')
        sums = audit.setdefault(stage, dict(products_checked=0, source_read_domains_checked=0))
        sums['products_checked'] += 1
        sums['source_read_domains_checked'] += len(observed)
        sums['scheduled_source_read_issues'] = issues[stage]
    return audit


def count_runtime_source_reads(audit, operands, phases, rows):
    """Count real read issues on the planned LHS slice/bank/port in each stage."""
    windows = {p['name']: p['runtime_range_including_staging'] for p in phases}
    location = re.compile(r'\bslice=(\d+) bank=(\d+).*?\bstream=E([45])\b')
    for stage, check in audit.items():
        ports = set()
        for view in operands[stage]['direct_lhs_tiles']:
            source = view['evidence']
            ports.update((s, source['source_bank'], 4 + i%2) for i,s in enumerate(source['source_slices']))
        start,end = windows[stage]
        reads = {}
        for row in rows:
            if row['resource'] not in ('MEM.E.Read', 'MEM.W.Read') or int(row['start']) >= end:
                continue
            match = location.search(row['detail'])
            if not match or (port := tuple(map(int, match.groups()))) not in ports:
                continue
            spans = reads.setdefault((row['resource'], port), [])
            spans.extend((max(start,a), min(end,b)) for a,b in expanded(row) if a < end and b > start)
        check['runtime_source_read_issues_in_window'] = sum(duration(merge(spans)) for spans in reads.values())
        check['runtime_issue_count_matches_schedule'] = check['runtime_source_read_issues_in_window'] == check['scheduled_source_read_issues']


def compare(result_dir, baseline_name='baseline_projection_solve_overlap.json', allow_unchanged_time=False):
    baseline_path = result_dir / baseline_name
    baseline = json.loads(baseline_path.read_text())
    summary = json.loads((result_dir / 'summary.json').read_text())
    if baseline['summary'].get('status') != 'passed' or summary.get('status') != 'passed':
        raise ValueError('Both baseline and current numerical runs must be passed before comparison')
    plan = json.loads((result_dir / 'graph_plan.json').read_text())
    with (result_dir / 'runtime.pipeline.csv').open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    hashes = []
    for entry in baseline['output_sha256']:
        path = result_dir / entry['file']
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes.append(dict(file=entry['file'], baseline_bytes=entry['bytes'], current_bytes=path.stat().st_size,
            baseline_sha256=entry['sha256'], current_sha256=digest,
            identical=entry['bytes'] == path.stat().st_size and entry['sha256'] == digest))
    before = baseline_metrics(baseline['summary'])
    saved_trace = baseline.get('comparison', {}).get('current')
    if saved_trace:
        for key in ('logical_cycles', 'external_wait_cycles', 'cold_cycles_through_compute',
                    'diagnostic_host_readback_cycles', 'full_trace_cycles'):
            if before[key] != saved_trace[key]:
                raise ValueError(f'Saved baseline summary and runtime trace disagree on {key}')
        before = saved_trace
    after = current_metrics(summary, rows)
    deltas = {}
    for key, old in before.items():
        new = after.get(key)
        if isinstance(old, int) and isinstance(new, int):
            deltas[key] = dict(before=old, after=new, change=new - old,
                reduction_percent=(old - new) / old * 100 if old else None)
    pairs = pair_metrics(plan, rows)
    checks = dict(all_outputs_bit_identical=all(item['identical'] for item in hashes),
        output_count_matches=len(hashes) == int(summary['runtime']['outputs']),
        dynamic_weight_bytes_unchanged=before['dynamic_weight_bytes'] == after['dynamic_weight_bytes'],
        no_host_compute=before['host_operations'] == after['host_operations'] == 0,
        actual_overlap_observed=bool(pairs) and all(pair['actual_arithmetic_overlap_cycles'] > 0 for pair in pairs))
    if allow_unchanged_time:
        checks['cold_compute_time_not_increased'] = after['cold_cycles_through_compute'] <= before['cold_cycles_through_compute']
    else:
        checks['cold_compute_time_reduced'] = after['cold_cycles_through_compute'] < before['cold_cycles_through_compute']
    phases = phase_metrics(plan, rows)
    operands = state_tile_preparation(plan)
    direct_lhs_evidence = audit_direct_lhs(result_dir, plan, operands)
    count_runtime_source_reads(direct_lhs_evidence, operands, phases, rows)
    if direct_lhs_evidence:
        checks['direct_lhs_runtime_reads_match_schedule'] = all(
            stage['runtime_issue_count_matches_schedule'] for stage in direct_lhs_evidence.values())
    lowering = json.loads((result_dir / 'lowering_path.json').read_text()) if (result_dir / 'lowering_path.json').exists() else {}
    tile_changes = state_tile_changes(baseline.get('comparison', {}).get('current_phases', []),
        phases, baseline.get('state_tile_preparation', {}), operands)
    return dict(status='passed' if all(checks.values()) else 'failed',
        scope=summary.get('scope'), seq_len=summary.get('seq_len',
            summary.get('chunkwise_prefix', {}).get('shape', [None])[0]),
        comparison='saved baseline versus current actual raw-macro execution',
        baseline_file=str(baseline_path),
        timing_acceptance='no_regression' if allow_unchanged_time else 'strict_improvement',
        baseline=before, current=after, changes=deltas, checks=checks, outputs=hashes,
        overlap_pairs=pairs, current_phases=phases, timing_unit='cycles',
        state_tile_preparation=operands, state_tile_changes=tile_changes,
        direct_lhs_schedule_audit=direct_lhs_evidence,
        compiler_shared_native_rhs_reuses=lowering.get('shared_native_rhs_reuses'),
        phase_note='Phase windows include compiler-recorded staging transports. FU activity is a union within each physical window and may include a concurrent operator. Transport vectors sum explicit copies. Byte-plane vectors count their storage planes. Phase changes use only saved baseline measurements.',
        timing_note='Cold timing includes initial ingress and runtime external-memory waits. Diagnostic HostOutput is reported separately. Static logical cycles are not standalone hardware arithmetic utilization.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-dir', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, default=Path('baseline_projection_solve_overlap.json'),
                        help='Saved JSON baseline; relative paths are resolved against --result-dir')
    parser.add_argument('--output-name', type=Path, default=Path('projection_solve_overlap_comparison.json'),
                        help='Comparison JSON filename inside --result-dir')
    parser.add_argument('--allow-unchanged-time', action='store_true',
                        help='Accept equal cold time for an unaffected-shape regression run; bit-exact checks remain mandatory')
    args = parser.parse_args()
    if args.output_name.is_absolute() or len(args.output_name.parts) != 1:
        parser.error('--output-name must be a filename, not a directory path')
    result_dir = args.result_dir.resolve()
    result = compare(result_dir, args.baseline, args.allow_unchanged_time)
    destination = result_dir / args.output_name
    if destination.resolve() == (result_dir / args.baseline).resolve():
        parser.error('--output-name must not overwrite the saved baseline')
    destination.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(dict(status=result['status'], output=str(destination), checks=result['checks'],
        cold_cycles=result['changes']['cold_cycles_through_compute'],
        actual_overlap_cycles=[pair['actual_arithmetic_overlap_cycles'] for pair in result['overlap_pairs']]), indent=2))
    if result['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
