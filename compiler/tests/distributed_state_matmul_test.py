"""Two different normalized queries read the same device-produced FP32 State."""
import argparse
import json
from pathlib import Path

import numpy as np

from activation_batch_matmul_test import LogicalGraph, compile_graph, run
from activation_state_matmul_test import fp32_chunked_reference, matmul_fp32, truncated_bf16


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('build-dir', 'target-config', 'output-dir'):
        parser.add_argument('--' + arg, type=Path, required=True)
    args = parser.parse_args()
    result_dir = args.output_dir.resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    summary = result_dir / 'summary.json'
    summary.write_text(json.dumps({'status': 'pending'}))

    heads, tokens, keys, values = 4, 32, 64, 128
    rng = np.random.default_rng(751113)
    graph = LogicalGraph()
    source = rng.normal(0, 0.2, (heads * keys, values)).astype('f4')
    bias = np.float32(2**-12)
    state = graph.vector_tiles('state', [(source, 'f32')],
        lambda v: v.add(0, kind='add', immediate=float(bias), dtype='f32'))
    view = graph.truncate_bf16('state_high', state).reshape(heads, keys, values)
    rhs_reference = truncated_bf16(source + bias).reshape(heads, keys, values)

    products, references, norms, norm_references = [], [], [], []
    for name in ('query', 'key'):
        # Each token has sixteen signed unit features, so its L2 norm is
        # exactly four in FP32. Different features on every head/token expose
        # misrouted lanes and reuse of the wrong left operand.
        samples = np.zeros((tokens, heads, keys), dtype=np.float32)
        for token in range(tokens):
            for head in range(heads):
                indices = rng.choice(keys, 4, replace=False)
                samples[token, head, indices] = rng.choice([-1, 1], 4)
        lhs = graph.norm(name + '_norm', samples.reshape(tokens, heads * keys),
            np.ones(heads * keys), 1e-6, mean=False)
        norms.append(lhs)
        norm_references.append(samples.reshape(tokens, heads * keys) * 0.25)
        lhs = lhs.reshape(tokens, heads, keys).transpose(1, 0, 2)
        product = matmul_fp32(graph, lhs, view, name + '_product')
        products.append(product.reshape(heads * tokens, values))
        references.append(fp32_chunked_reference(samples.transpose(1, 0, 2) * 0.25, rhs_reference))

    exe, env, image = compile_graph(graph, products + norms, result_dir, args.build_dir.resolve(),
                                    args.target_config.resolve(), fp32_spills=True)
    plan = json.loads((result_dir / 'graph_plan.json').read_text())
    lhs_nodes = [node for node in plan['nodes'] if node['name'].endswith('.lhs')]
    expected_lhs = len(products) * (heads // 2) * (keys // 32)
    assert len(lhs_nodes) == expected_lhs and all(node.get('native_mxm_lhs_view') for node in lhs_nodes)
    lhs_names = {node['name'] for node in lhs_nodes}
    assert not any(copy['phase'] in lhs_names for copy in plan['linker_mem_transports'])
    path = json.loads((result_dir / 'lowering_path.json').read_text())
    assert path['shared_native_rhs_reuses'] == 1, path
    run([exe('runtime', 'compiled_stage_file_runner'), image, result_dir / 'inputs.txt',
         result_dir / 'actual', result_dir / 'runtime.pipeline.csv'], result_dir / 'execution.log', env)
    for index, reference in enumerate(references):
        actual = np.fromfile(result_dir / f'actual.{index}.bin', '<f4').reshape(reference.shape)
        np.testing.assert_array_equal(actual, reference)
    for index, reference in enumerate(norm_references, start=len(products)):
        words = np.fromfile(result_dir / f'actual.{index}.bin', '<u2').astype('u4')
        np.testing.assert_array_equal((words << 16).view('f4').reshape(reference.shape), reference)
    assert 'host_operations=0' in (result_dir / 'execution.log').read_text().splitlines()[-1].split()
    run([exe('runtime', 'ftlpu_icu_program_export'), result_dir / 'actual.runtime.ftlpu',
         result_dir / 'runtime_icu_programs', '--pre-execution-dir', result_dir / 'actual.pre_execution'],
        result_dir / 'export.log', env)
    report = dict(status='passed', heads=heads, tokens=tokens, k=keys, n=values,
                  products_bit_exact=len(products), normalized_operands_bit_exact=len(norms),
                  native_lhs_views=len(lhs_nodes), shared_rhs_reuses=path['shared_native_rhs_reuses'],
                  scope='two L2-normalized operands times one State high-plane view')
    summary.write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == '__main__':
    main()
