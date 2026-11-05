"""Check mirrored BF16 recurrent readout -> shared native RMSNorm ingress."""
import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from qwen3_5_logical_graph import LogicalGraph
from qwen3_5_compiled_layer import bf16
from qwen3_5_decode_gdn_attention_test import run
from qwen3_5_qkv_conv_prefill_test import compare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('build-dir', 'target-config', 'output-dir'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--value-dim', type=int, choices=(32, 128), default=128)
    args = parser.parse_args()
    dim = args.value_dim
    # Bound the synthetic State footprint; the real seq32 test covers 128x128.
    key_dim = 128 if dim == 32 else 32
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    build = args.build_dir.resolve()
    suffix = '.exe' if sys.platform == 'win32' else ''
    exe = lambda area, name: build / area / (name + suffix)
    rng = np.random.default_rng(351104)
    c = SimpleNamespace(num_value_heads=16, num_key_heads=16, key_head_dim=key_dim, value_head_dim=dim)
    q, k = [bf16(rng.normal(0, .1, (4, 16 * key_dim)).astype('f4')) for _ in range(2)]
    v = bf16(rng.normal(0, .1, (4, 16 * dim)).astype('f4'))
    eg = rng.uniform(.7, .99, (4, 16)).astype('f4')
    beta = rng.uniform(.1, .8, (4, 16)).astype('f4')
    state = rng.normal(0, .02, (16 * key_dim, dim)).astype('f4')
    gamma = bf16(rng.uniform(.5, 1.5, dim).astype('f4'))
    graph = LogicalGraph()
    core, _ = graph.delta(q, k, v, eg, beta, state, c, state_dtype='f32')
    norm = graph.norm('output_norm', core.reshape(64, dim), gamma, 1e-6)
    logical = graph.write(out / 'logical', [core, norm])
    kernel, binary = out / 'kernel.mlir', out / 'program.ftlpu'
    run([exe('compiler', 'ftlpu-graph-compile'), '--input', logical, '--output', kernel,
         '--emit-kernel', '--target-config', args.target_config.resolve()], out / 'import.log')
    kernel.write_text(kernel.read_text().replace('module {', 'module attributes {'
        'ftlpu.gdn_prefill_strategy = "tokenwise", ftlpu.vector_fp32_ingress = true, '
        'ftlpu.vector_bf16_spill = true, ftlpu.sxm_wide = true, '
        'ftlpu.l2_head_partition = true, ftlpu.common_transport_overlap = false} {', 1))
    run([exe('compiler', 'ftlpu-compile'), '--input', kernel, '--input-stage', 'kernel',
         '--output', binary, '--target-config', args.target_config.resolve(),
         '--icu-compression', 'raw-macro', '--mxm-execution', 'ws'], out / 'compile.log')
    os.environ.update(FTLPU_FILE_RUNNER_MODEL_SESSION='1', FTLPU_FILE_RUNNER_SXM_ARCHITECTURE='wide')
    run([exe('runtime', 'compiled_stage_file_runner'), binary, out / 'inputs.txt',
         out / 'actual', out / 'runtime.pipeline.csv'], out / 'execution.log')
    read = lambda i: (np.fromfile(out / f'actual.{i}.bin', '<u2').astype('u4') << 16).view('f4').reshape(64, dim)
    source, actual = read(0), read(1)
    expected = source / np.sqrt(np.mean(source * source, axis=1, keepdims=True) + 1e-6) * gamma
    error = compare(actual, expected)
    assert error['nrmse'] < .01 and error['cosine'] > .9999, error
    plan = json.loads((out / 'graph_plan.json').read_text())
    nodes = [n for n in plan['nodes'] if n['name'].startswith('output_norm.rows')]
    if dim == 128:
        assert len(nodes) == 2 and nodes[1]['norm_weight_reused'] and nodes[1]['norm_tight_start']
    else:
        assert not nodes  # Single-vector rows keep the existing generic reduction.
    transports = [t for t in plan['linker_mem_transports'] if t['phase'].startswith('output_norm')]
    if dim == 128:
        assert len(transports) == 8 and all(t['phase'] == nodes[0]['name'] for t in transports), transports
    report = dict(status='passed', errors=error, native_norm_tiles=len(nodes), norm_weight_preparations=1 if dim == 128 else None,
                  readout_layout_copies=0 if dim == 128 else None, raw_macro=True, value_dim=dim, key_dim=key_dim)
    (out / 'summary.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
