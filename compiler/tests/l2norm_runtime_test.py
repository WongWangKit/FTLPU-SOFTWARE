#!/usr/bin/env python3
"""Check native L2 feedback semantics and the shared RMSNorm path on ICU/FU."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from common_kernel_graph import l2_norm_source, rms_norm_source
from qwen3_5_compiled_layer import bf16
from qwen3_5_logical_graph import LogicalGraph
from qwen3_5_reference import causal_depthwise_conv1d


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('compile', 'runner', 'target-config', 'output-dir'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument("--native-only", action="store_true", help="Test normalization without the convolution producer")
    a = p.parse_args()
    results = []
    # Ones distinguish sum from mean; tiny inputs test the unscaled epsilon.
    # Non-power-of-two scale must remain an FP32 immediate, not BF16 gamma.
    for kind, width, eps, scale in (
            ('l2', 32, 1e-6, 1.), ('l2', 128, 1e-6, 1/np.sqrt(128)),
            ('l2', 256, .25, .37), ('rms', 128, 1e-6, 1.)):
        out = a.output_dir.resolve() / f'{kind}_{width}'
        out.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(35128)
        x = rng.normal(0, .3, (32, width)).astype(np.float32)
        x[0] = 0
        x[1] *= .001
        x[2] = 1
        x[3] *= 5
        x = bf16(x)
        gamma = bf16(rng.uniform(.7, 1.3, width).astype(np.float32))
        source = l2_norm_source(x.shape, eps, scale) if kind == 'l2' else rms_norm_source(x.shape, eps)
        kernel = out / 'norm.kernel.mlir'
        kernel.write_text(source, encoding='utf-8')
        binary = out / 'norm.ftlpu'
        subprocess.run([str(a.compile.resolve()), '--input', str(kernel),
                        '--input-stage', 'kernel', '--output', str(binary),
                        '--target-config', str(a.target_config.resolve())], check=True)
        inputs = [x] if kind == 'l2' else [x, gamma]
        paths = []
        for i, v in enumerate(inputs):
            path = out / f'input{i}.bin'
            (v.view(np.uint32) >> 16).astype('<u2').tofile(path)
            paths.append(str(path))
        listing = out / 'inputs.txt'
        listing.write_text('\n'.join(paths) + '\n', encoding='utf-8')
        env = os.environ.copy()
        env['FTLPU_FILE_RUNNER_MODEL_SESSION'] = '1'
        for key in ('FTLPU_SESSION_LINK_ONLY', 'FTLPU_SESSION_STOP_CYCLE'):
            env.pop(key, None)
        with (out / 'execution.log').open('w', encoding='utf-8') as log:
            subprocess.run([str(a.runner.resolve()), str(binary), str(listing),
                            str(out / 'norm'), str(out / 'runtime.pipeline.csv')],
                           env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        actual = (np.fromfile(out / 'norm.0.bin', '<u2').astype(np.uint32) << 16).view(np.float32).reshape(x.shape)
        sq = x.astype(np.float64)**2
        expected = x / np.sqrt((sq.sum(axis=1, keepdims=True) if kind == 'l2'
                               else sq.mean(axis=1, keepdims=True)) + eps)
        expected *= scale if kind == 'l2' else gamma
        row_error = np.linalg.norm(actual-expected, axis=1) / np.maximum(np.linalg.norm(expected, axis=1), 1e-30)
        if not np.isfinite(actual).all() or row_error.max() > .012:
            raise AssertionError(f'{kind}/{width}: per-row NRMSE {row_error.tolist()}')
        results.append(dict(kind=kind, width=width, epsilon=eps, scale=scale,
                            max_row_nrmse=float(row_error.max())))
    if a.native_only:
        (a.output_dir / "summary.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(results, indent=2))
        return
    # Exercise a reshaped native producer, not just an externally packed
    # L2 input. Distinct groups detect head/token address permutations.
    for width, groups, eps, scale in ((32, 4, .25, .37), (128, 2, 1e-6, 1.)):
        out = a.output_dir.resolve() / f'grouped_{width}'
        out.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(width)
        x = bf16(rng.normal(0, .3, (32, width * groups)).astype(np.float32))
        w = bf16(rng.normal(0, .2, (width * groups, 4)).astype(np.float32))
        graph = LogicalGraph()
        conv = graph.conv(x, w)
        normalized = graph.norm('grouped_l2', conv.reshape(-1, width),
                                np.ones(width, np.float32), eps, False, scale)
        logical = graph.write(out / 'logical_graph', [normalized.reshape(x.shape)])
        kernel, binary = out / 'norm.kernel.mlir', out / 'norm.ftlpu'
        graph_compile = a.compile.resolve().with_name(
            'ftlpu-graph-compile.exe' if sys.platform == 'win32' else 'ftlpu-graph-compile')
        subprocess.run([str(graph_compile), '--input', str(logical), '--output', str(kernel),
                        '--emit-kernel', '--target-config', str(a.target_config.resolve())], check=True)
        kernel.write_text(kernel.read_text().replace('module {',
            'module attributes {ftlpu.common_barrier_protocol = true} {', 1))
        subprocess.run([str(a.compile.resolve()), '--input', str(kernel), '--input-stage', 'kernel',
                        '--output', str(binary), '--target-config', str(a.target_config.resolve())], check=True)
        schedule = (out / 'decoder_layer.schedule.mlir').read_text(encoding='utf-8')
        if 'l2norm.transpose_native_group' not in schedule:
            raise AssertionError('grouped input unexpectedly used the generic materialization path')
        with (out / 'execution.log').open('w', encoding='utf-8') as log:
            subprocess.run([str(a.runner.resolve()), str(binary), str(out / 'inputs.txt'),
                            str(out / 'norm'), str(out / 'runtime.pipeline.csv')],
                           env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        actual = (np.fromfile(out / 'norm.0.bin', '<u2').astype(np.uint32) << 16).view(np.float32).reshape(-1, width)
        expected, _ = causal_depthwise_conv1d(x, w, np.zeros_like(w))
        expected = expected.reshape(-1, width).astype(np.float64)
        expected = scale * expected / np.sqrt(np.sum(expected**2, axis=1, keepdims=True) + eps)
        error = np.linalg.norm(actual-expected, axis=1) / np.maximum(np.linalg.norm(expected, axis=1), 1e-30)
        if not np.isfinite(actual).all() or error.max() > .015:
            raise AssertionError(f'grouped/{width}: max per-row NRMSE {error.max()}')
        results.append(dict(kind='grouped_l2', width=width, groups=groups,
                            epsilon=eps, scale=scale, max_row_nrmse=float(error.max())))
    (a.output_dir / 'summary.json').write_text(json.dumps(results, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
