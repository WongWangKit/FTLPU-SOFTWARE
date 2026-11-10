"""Compare paged linear/causal depthwise convolution with temporal overlap off/on."""
import argparse
import json
from pathlib import Path

import numpy as np

from activation_batch_matmul_test import LogicalGraph, bf16, compile_graph, run
from projection_solve_overlap_test import trace_metrics
from qwen3_5_qkv_conv_prefill_test import compare
from qwen3_5_reference import causal_depthwise_conv1d


def execute(args, enabled):
    out = args.output_dir.resolve() / ('on' if enabled else 'off')
    tokens, hidden, channels = 64, 256, 3072
    rng = np.random.default_rng(7411)
    x = bf16(rng.normal(0, .2, (tokens, hidden)).astype('f4'))
    weight = rng.integers(-12, 13, (channels, hidden), dtype=np.int8)
    taps = bf16(rng.normal(0, .3, (channels, 4)).astype('f4'))
    graph = LogicalGraph()
    projected = graph.project('input_linear', x, weight, .015625)
    conv = graph.conv(projected, taps)
    # A head norm is an ordinary consumer of distributed Conv results.
    group = conv[:, :256].reshape(tokens * 2, 128)
    norm = graph.norm('head_norm', group, np.ones(128), 1e-6, mean=False)
    outputs = [projected, conv, norm]
    attrs = ('ftlpu.l2_head_partition = true', 'ftlpu.weight_page_rows = 512 : i64',
             f'ftlpu.projection_conv_overlap = {str(enabled).lower()}')
    exe, env, image = compile_graph(graph, outputs, out, args.build_dir.resolve(),
                                    args.target_config.resolve(), module_attrs=attrs)
    run([exe('runtime', 'compiled_stage_file_runner'), image, out / 'inputs.txt',
         out / 'actual', out / 'runtime.pipeline.csv'], out / 'execution.log', env)
    projected_ref = x @ weight.astype('f4').T * .015625
    conv_ref, _ = causal_depthwise_conv1d(projected_ref, taps, np.zeros_like(taps))
    errors = {}
    paths = [out / f'actual.{i}.bin' for i in range(len(outputs))]
    for label, path, ref in zip(('projection', 'conv_silu'), paths, (projected_ref, conv_ref)):
        words = np.fromfile(path, '<u2').astype('u4')
        errors[label] = compare((words << 16).view('f4').reshape(ref.shape), ref)
        assert errors[label]['nrmse'] < .03 and errors[label]['cosine'] > .999, errors
    assert 'host_operations=0' in (out / 'execution.log').read_text().splitlines()[-1].split()
    plan = json.loads((out / 'graph_plan.json').read_text())
    pages = [n for n in plan['nodes'] if n['name'].startswith('input_linear.page')]
    assert len(pages) == 3
    metrics = trace_metrics(out / 'runtime.pipeline.csv')
    run([exe('runtime', 'ftlpu_icu_program_export'), out / 'actual.runtime.ftlpu',
         out / 'runtime_icu_programs', '--pre-execution-dir', out / 'actual.pre_execution'],
        out / 'export.log', env)
    return paths, dict(errors=errors, runtime=metrics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('build-dir', 'target-config', 'output-dir'):
        parser.add_argument('--' + key, required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = args.output_dir / 'summary.json'
    summary.write_text(json.dumps({'status': 'pending'}))
    before, old = execute(args, False)
    after, new = execute(args, True)
    assert all(a.read_bytes() == b.read_bytes() for a, b in zip(before, after))
    assert old['runtime']['mxm_vxm_overlap_cycles'] == 0
    assert new['runtime']['mxm_vxm_overlap_cycles'] > 0
    assert new['runtime']['cold_cycles_through_compute'] < old['runtime']['cold_cycles_through_compute']
    result = dict(status='passed', bit_identical_outputs=len(after), off=old, on=new)
    summary.write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
