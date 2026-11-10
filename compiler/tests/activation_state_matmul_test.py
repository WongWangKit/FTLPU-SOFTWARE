"""Native BF16 activation GEMM from a device-produced FP32 State high-plane view.

Exercises K=128, N=128, odd/even head counts and full/tail token tiles. Python
provides operands and a fixed 32-K FP32 reference; C++ owns all layout choices.
"""
import argparse
import json
from pathlib import Path
import re

import numpy as np

from activation_batch_matmul_test import LogicalGraph, bf16, compile_graph, run


def truncated_bf16(value):
    return (np.ascontiguousarray(value, dtype='f4').view('u4') & np.uint32(0xffff0000)).view('f4')


def matmul_fp32(graph, lhs, rhs, name='contraction'):
    shape = (lhs.shape[0], lhs.shape[1], rhs.shape[2])
    a, b = graph.type(lhs.shape, 'bf16'), graph.type(rhs.shape, 'bf16')
    output = graph.type(shape, 'f32')
    source = f'''module {{ func.func @stage(%a: {a}, %b: {b}) -> {output} {{
      %o = ftlpu.kernel.batch_matmul %a, %b {{transpose_rhs = false, role = "pv"}}
        : ({a}, {b}) -> {output}
      return %o : {output}
    }} }}'''
    return graph.run(name, source, [(lhs, 'bf16'), (rhs, 'bf16')], [(shape, 'f32')])[0]


def fp32_chunked_reference(lhs, rhs):
    """32 sequential products per partial, then FP32 adds in ascending K order."""
    result = None
    for first in range(0, lhs.shape[2], 32):
        partial = np.zeros((lhs.shape[0], lhs.shape[1], rhs.shape[2]), dtype='f4')
        for k in range(first, first + 32):
            product = lhs[:, :, k, None] * rhs[:, None, k, :]
            partial = np.add(partial, product, dtype=np.float32)
        result = partial if result is None else np.add(result, partial, dtype=np.float32)
    return result


def audit_preparation(plan, heads, rows, schedule):
    nodes = [node for node in plan['nodes'] if node['name'].startswith('contraction.')]
    lhs = [node for node in nodes if node['name'].endswith('.lhs')]
    rhs = [node for node in nodes if node['name'].endswith('.rhs')]
    pairs, row_tiles, k_tiles, n_tiles = (heads + 1) // 2, (rows + 31) // 32, 4, 2
    expected_lhs = pairs * row_tiles * k_tiles
    assert len(lhs) == expected_lhs, (len(lhs), expected_lhs)
    assert len(rhs) == expected_lhs * n_tiles, (len(rhs), expected_lhs * n_tiles)
    # A join describes a byte-preserving MEM/SXM layout operation. The former
    # multiply-by-one conversion had an arithmetic module and no layout_join.
    assert all(node.get('layout_join') for node in lhs + rhs), lhs + rhs
    assert all('.col0.' in node['name'] for node in lhs), 'LHS must be prepared once, then reused by col64'
    unique_lhs = {re.search(r'\.batch(\d+)\.row(\d+)\.k(\d+)\.lhs$', node['name']).groups() for node in lhs}
    assert len(unique_lhs) == expected_lhs
    products = [node for node in nodes if re.search(r'\.batch\d+\.row\d+\.k\d+$', node['name'])]
    # Later 32-K products feed an FP32 add in the same node. The prior sum
    # may already have the required native layout, so a .prior join is not
    # required. Check the emitted FP32 add itself, rather than its old name.
    priors = [node for node in nodes if node['name'].endswith('.prior')]
    adds = [node for node in nodes if node['name'].endswith('.accumulate')]
    expected_adds = pairs * row_tiles * (k_tiles - 1) * n_tiles
    assert len(products) == expected_lhs * n_tiles, len(products)
    assert len(priors) <= expected_adds and all(node.get('layout_join') for node in priors), priors
    assert not adds, 'The FP32 partial sum must be added directly after MXM in its fused node'
    emitted_adds = []
    for line in schedule.read_text().splitlines():
        if ('ftlpu.schedule.vxm ' in line and 'opcode = "add"' in line
                and 'lhs_kind = "stream_f32"' in line and 'rhs_kind = "stream_f32"' in line):
            assert 'cast_target = "fp32"' in line, line
            emitted_adds.append(int(re.search(r'\bcycle = (\d+) : i64', line).group(1)))
    assert len(emitted_adds) == expected_adds, (len(emitted_adds), expected_adds)
    for node in products:
        reduction = int(re.search(r'\.k(\d+)$', node['name']).group(1))
        count = sum(node['nominal_start'] <= cycle < node['nominal_end'] for cycle in emitted_adds)
        assert count == int(reduction != 0), (node['name'], count)
    return dict(lhs_preparations=len(lhs), lhs_preparations_without_n_reuse=expected_lhs*n_tiles,
        rhs_preparations=len(rhs), arithmetic_operand_conversions=0,
        k32_partial_products=len(products), fused_fp32_partial_sum_adds=expected_adds,
        prior_layout_joins=len(priors),
        standalone_fp32_partial_sum_add_nodes=len(adds))


def audit_state_high_planes(plan, schedule):
    producer = next(node for node in plan['nodes'] if node['name'] == 'state_producer')
    layouts = producer['direct_output_layouts']
    assert len(layouts) == 1 and layouts[0]['kind'] == 'fp32_quad_planar', layouts
    layout = layouts[0]
    low, high = layout['slices'][:2], layout['slices'][2:]
    assert len(low) == len(high) == 2 and high[1] == high[0] + 1
    transfers = [item for item in plan['linker_mem_transports']
                 if item['phase'].startswith('contraction.') and item['phase'].endswith('.rhs')]
    assert transfers, 'The produced State view must feed the RHS preparation'
    reads = set()
    for line in schedule.read_text().splitlines():
        if 'ftlpu.schedule.mem_transfer' not in line or 'opcode = "read"' not in line:
            continue
        attrs = {key: int(value) for key, value in re.findall(r'(\w+) = (-?\d+) : i64', line)}
        reads.add((attrs['cycle'], attrs['hemisphere'], attrs['bank'], attrs['slice'], attrs['address']))
    for item in transfers:
        assert item['byte_planes'] == 2 and item['source_slice'] == high[0], item
        assert item['source_bank'] == layout['bank'], item
        assert layout['base_row'] <= item['source_row'] < layout['base_row'] + layout['rows'], item
        side = {'east': 0, 'west': 1}[item['source_side']]
        for plane in high:
            assert (item['start'], side, item['source_bank'], plane, item['source_row']) in reads, item
        for plane in low:
            assert (item['start'], side, item['source_bank'], plane, item['source_row']) not in reads, item
    return dict(status='passed', producer_fp32_slices=layout['slices'], rhs_read_slices=high,
                rhs_read_byte_planes=2, audited_transport_domains=len(transfers),
                validation='Compiler layout and corresponding scheduled MEM read instructions')


def execute(args, heads, rows):
    out = args.output_dir.resolve() / f'h{heads}_m{rows}_k128_n128'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'summary.json').write_text(json.dumps(dict(status='pending'), indent=2))
    rng = np.random.default_rng(35261106 + heads * 100 + rows)
    bias = np.float32(2**-12)
    source_state = rng.normal(0, .15, (heads, 128, 128)).astype('f4')
    lhs = bf16(rng.normal(0, .15, (heads, rows, 128)).astype('f4'))
    # This cell distinguishes the specified 32-K grouping from accumulating
    # all 128 products in one FP32 chain: 2^24 + (1+1) versus (2^24+1)+1.
    source_state[:, :, 0] = -bias
    source_state[:, 0, 0] = np.float32(2**24)
    source_state[:, 32:34, 0] = np.float32(1) - bias
    lhs[:, 0, :] = 0
    lhs[:, 0, (0, 32, 33)] = 1
    graph = LogicalGraph()
    state = graph.vector_tiles('state_producer', [(source_state.reshape(heads*128, 128), 'f32')],
        lambda v: v.add(0, kind='add', immediate=float(bias), dtype='f32'))
    state_view = graph.truncate_bf16('state_high_planes', state).reshape(heads, 128, 128)
    product = matmul_fp32(graph, lhs, state_view)
    # Export the original FP32 State and GEMM result. The BF16 high-plane
    # alias stays an internal input: its FP32 home is not a standalone BF16
    # C2C binding. Inspect the actual source-plane reads below instead.
    exe, env, binary = compile_graph(graph,
        [state, product.reshape(heads*rows, 128)],
        out, args.build_dir.resolve(), args.target_config.resolve(), fp32_spills=True)
    plan = json.loads((out / 'graph_plan.json').read_text())
    preparation = audit_preparation(plan, heads, rows, out / 'decoder_layer.schedule.mlir')
    high_plane_audit = audit_state_high_planes(plan, out / 'decoder_layer.schedule.mlir')
    run([exe('runtime', 'compiled_stage_file_runner'), binary, out / 'inputs.txt',
         out / 'actual', out / 'runtime.pipeline.csv'], out / 'execution.log', env)
    actual_state = np.fromfile(out / 'actual.0.bin', '<f4').reshape(heads, 128, 128)
    actual = np.fromfile(out / 'actual.1.bin', '<f4').reshape(heads, rows, 128)
    expected_state = np.add(source_state, bias, dtype=np.float32)
    expected_view = truncated_bf16(expected_state)
    np.testing.assert_array_equal(actual_state, expected_state)
    reference = fp32_chunked_reference(lhs, expected_view)
    # Tight FP32 criteria are fixed independently of the model-level BF16
    # threshold. The dedicated sentinel additionally enforces 32-K grouping.
    np.testing.assert_allclose(actual, reference, rtol=2e-6, atol=2e-7)
    np.testing.assert_array_equal(actual[:, 0, 0], np.full(heads, np.float32(2**24+2)))
    for head in range(heads):
        others = [i for i in range(heads) if i != head]
        if others:
            # Exclude the common sentinel column from the head-isolation test.
            local = np.linalg.norm(actual[head, :, 1:] - reference[head, :, 1:])
            assert all(local < np.linalg.norm(actual[head, :, 1:] - reference[i, :, 1:]) for i in others)
    log = (out / 'execution.log').read_text()
    stats = {key: int(value) for key, value in re.findall(r'(\w+)=(\d+)', log.splitlines()[-1])}
    assert stats.get('host_operations') == 0
    run([exe('runtime', 'ftlpu_icu_program_export'), out / 'actual.runtime.ftlpu',
         out / 'runtime_icu_programs', '--pre-execution-dir', out / 'actual.pre_execution'], out / 'export.log', env)
    report = dict(status='passed', heads=heads, rows=rows, k=128, n=128,
        icu_compression='raw-macro', runtime=stats, preparation=preparation,
        precision='Device-produced FP32 State; explicit truncated BF16 high-plane view; BF16 LHS; FP32 32-K partials and ordered FP32 sums',
        reference_tolerance=dict(rtol=2e-6, atol=2e-7),
        state_bit_exact=True, high_plane_read_audit=high_plane_audit, k32_grouping_sentinel_exact=True,
        matmul_bit_exact=bool(np.array_equal(actual, reference)),
        matmul_max_abs=float(np.max(np.abs(actual.astype('f8') - reference.astype('f8')))))
    (out / 'summary.json').write_text(json.dumps(report, indent=2))
    return report


def check_view_isolation(args):
    """Views of one FP32 producer that select different heads need distinct RHS tiles."""
    out = args.output_dir.resolve() / 'same_producer_swapped_heads'
    heads, rows, dim = 2, 16, 64
    rng = np.random.default_rng(35261107)
    source = rng.normal(0, .25, (heads, dim, dim)).astype('f4')
    lhs = bf16(rng.normal(0, .125, (heads, rows, dim)).astype('f4'))
    graph = LogicalGraph()
    state = graph.vector_tiles('state', [(source.reshape(heads*dim, dim), 'f32')],
        lambda v: v.add(0, kind='add', immediate=.125, dtype='f32'))
    view = graph.truncate_bf16('state_high_planes', state).reshape(heads, dim, dim)
    results = [matmul_fp32(graph, lhs, value, name).reshape(heads*rows, dim)
               for name, value in [('original_heads', view), ('swapped_heads', view[::-1])]]
    exe, env, binary = compile_graph(graph, results, out, args.build_dir.resolve(),
        args.target_config.resolve(), fp32_spills=True)
    path = json.loads((out / 'lowering_path.json').read_text())
    assert path['shared_native_rhs_reuses'] == 0, path
    run([exe('runtime', 'compiled_stage_file_runner'), binary, out / 'inputs.txt',
         out / 'actual', out / 'runtime.pipeline.csv'], out / 'execution.log', env)
    assert 'host_operations=0' in (out / 'execution.log').read_text().splitlines()[-1]
    expected_state = truncated_bf16(source + np.float32(.125))
    for index, rhs in enumerate((expected_state, expected_state[::-1])):
        actual = np.fromfile(out / f'actual.{index}.bin', '<f4').reshape(heads, rows, dim)
        np.testing.assert_array_equal(actual, fp32_chunked_reference(lhs, rhs))
    run([exe('runtime', 'ftlpu_icu_program_export'), out / 'actual.runtime.ftlpu',
         out / 'runtime_icu_programs', '--pre-execution-dir', out / 'actual.pre_execution'],
        out / 'export.log', env)
    report = dict(status='passed', output_count=2, both_bit_exact=True, shared_native_rhs_reuses=0)
    (out / 'summary.json').write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('build-dir', 'target-config', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / 'summary.json').write_text(json.dumps(dict(status='pending'), indent=2))
    cases = [execute(args, heads, rows) for heads, rows in ((2, 32), (3, 16), (4, 32))]
    report = dict(status='passed', scope='activation GEMM with a produced FP32 State BF16 view',
                  cases=cases, view_isolation=check_view_isolation(args))
    (out / 'summary.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
