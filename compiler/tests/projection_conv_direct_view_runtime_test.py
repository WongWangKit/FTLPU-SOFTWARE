"""Native projection addresses feed bilateral Conv1D and word-view L2Norm."""
import argparse
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from qwen3_5_logical_graph import LogicalGraph
from qwen3_5_compiled_layer import bf16


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('compile', 'graph-compile', 'runner', 'target-config', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--channels', type=int, default=2048)
    parser.add_argument('--wide-sxm', action='store_true')
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / 'summary.json').write_text(json.dumps(dict(status='pending')))
    rng = np.random.default_rng(821)
    x = bf16(rng.normal(0, .4, (32, args.channels)))
    w = bf16(rng.normal(0, .3, (args.channels, 4)))
    graph = LogicalGraph()
    pw = rng.integers(-16, 17, (args.channels, 64), dtype=np.int8)
    raw = bf16(rng.normal(0, .2, (32, 64)))
    projected = graph.project('projection', raw, pw, .01)
    result = graph.conv(projected, w)
    normalized = graph.norm('l2', result.reshape(-1, 128), np.ones(128), 1e-6, False)
    x = bf16(raw @ bf16(pw.astype(np.float32) * .01).T)
    logical = graph.write(out / 'graph', [result, normalized])
    kernel, binary = out / 'kernel.mlir', out / 'program.ftlpu'
    commands = [
        [str(args.graph_compile.resolve()), '--input', str(logical), '--output', str(kernel), '--emit-kernel'],
        [str(args.compile.resolve()), '--input', str(kernel), '--input-stage', 'kernel', '--output', str(binary), '--icu-compression', 'raw-macro', '--mxm-execution', 'ws'],
    ]
    if args.wide_sxm:
        os.environ['FTLPU_FILE_RUNNER_SXM_ARCHITECTURE']='wide'
        os.environ['FTLPU_FILE_RUNNER_MODEL_SESSION']='1'
    for index, command in enumerate(commands):
        with (out / f'compile{index}.log').open('w') as log:
            subprocess.run(command + ['--target-config', str(args.target_config.resolve())], stdout=log, stderr=subprocess.STDOUT, check=True)
        if index == 0 and args.wide_sxm:
            kernel.write_text(kernel.read_text().replace('module {','module attributes {ftlpu.sxm_wide = true, ftlpu.l2_head_partition = true} {',1))
    with (out / 'execution.log').open('w') as log:
        subprocess.run([str(args.runner.resolve()), str(binary), str(out / 'inputs.txt'),
                        str(out / 'actual'), str(out / 'runtime.pipeline.csv')], stdout=log, stderr=subprocess.STDOUT, check=True)
    expected = np.zeros_like(x)
    for token in range(len(x)):
        for tap in range(4):
            source = token + tap - 3
            if source >= 0:
                expected[token] += x[source] * w[:, tap]
    expected = bf16(expected)
    expected = bf16(expected / (1 + np.exp(-expected)))
    actual = (np.fromfile(out / 'actual.0.bin', '<u2').astype(np.uint32) << 16).view(np.float32).reshape(x.shape)
    error = float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))
    norm_actual = (np.fromfile(out / 'actual.1.bin', '<u2').astype(np.uint32) << 16).view(np.float32).reshape(-1,128)
    norm_ref = expected.reshape(-1,128)
    norm_ref = norm_ref / np.sqrt(np.sum(norm_ref*norm_ref,axis=1,keepdims=True)+1e-6)
    norm_error = float(np.linalg.norm(norm_actual-norm_ref)/np.linalg.norm(norm_ref))
    assert norm_error < .02, norm_error
    schedule=(out/'decoder_layer.schedule.mlir').read_text()
    assert ('l2.l2norm.sxm_vxm_streaming' if args.wide_sxm else 'l2.l2norm.transpose_native_group') in schedule
    if args.wide_sxm:
        assert schedule.count('name = "l2.l2norm.sxm_vxm_streaming"') == args.channels // 256
        assert 'l2norm.prepare_sxm_pair' not in schedule
        assert 'l2norm.replicate_feedback_tile' not in schedule
    assert 'l2.l2norm.transpose_input' not in schedule
    primary = schedule.split('name = "conv1d_silu.block0.operand"',1)[1].split('name = "conv1d_silu.block0.depthwise_conv1d"',1)[0]
    assert not re.search(r'opcode = "write"[^\n]*slice = 3[45] :', primary), 'Conv ingress copy returned'
    print('L2 NRMSE',norm_error)
    assert np.isfinite(actual).all() and error < .01, error
    report = dict(status='passed', shape=list(x.shape), l2_nrmse=norm_error,
                  nrmse=error, max_abs_error=float(np.max(np.abs(actual-expected))))
    (out / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
