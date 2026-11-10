"""Exercise mixed variance ranges within the same BF16 raw-macro RMSNorm."""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from common_kernel_graph import rms_norm_source
from deepseek_v41_qkv_norm_test import run
from import_hf_decoder_layer import bf16, read_bf16, write_bf16


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('build-dir', 'target-config', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--epsilon', type=float, default=1e-20)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    summary = out / 'summary.json'
    summary.write_text(json.dumps({'status': 'pending'}))
    suffix = '.exe' if os.name == 'nt' else ''
    def exe(area, name):
        return args.build_dir.resolve() / area / (name + suffix)

    width, epsilon = 512, args.epsilon
    # Powers of two straddle several FP16 LUT ranges and the epsilon floor.
    scales = np.asarray([0, 2**-50, 2**-34, 2**-24, 2**-12, 1, 2**12, 2**36] * 4, dtype='f4')
    rng = np.random.default_rng(7491)
    x = bf16(rng.normal(size=(32, width)).astype('f4') * scales[:, None])
    if epsilon < 2**-140:
        # Exactly representable squared values survive binary32 underflow.
        # This row requires a lookup scale larger than one finite FP32 word.
        scales[1] = 2**-74
        x[1] = np.where(np.arange(width) % 2, scales[1], -scales[1])
    gamma = bf16(rng.uniform(.75, 1.25, size=width).astype('f4'))
    write_bf16(out / 'x.bin', x)
    write_bf16(out / 'gamma.bin', gamma)
    listing = out / 'inputs.txt'
    listing.write_text(f'{out / "x.bin"}\n{out / "gamma.bin"}\n')
    kernel = out / 'kernel.mlir'
    kernel.write_text(rms_norm_source(x.shape, epsilon).replace('module {',
        'module attributes {ftlpu.sxm_wide = true} {', 1))
    binary = out / 'program.ftlpu'
    run([exe('compiler', 'ftlpu-compile'), '--input', kernel, '--input-stage', 'kernel',
         '--output', binary, '--target-config', args.target_config.resolve(), '--icu-compression', 'raw-macro'],
        out / 'compile.log')
    env = dict(os.environ, FTLPU_FILE_RUNNER_MODEL_SESSION='1', FTLPU_FILE_RUNNER_SXM_ARCHITECTURE='wide')
    env.pop('FTLPU_SESSION_LINK_ONLY', None)
    run([exe('runtime', 'compiled_stage_file_runner'), binary, listing, out / 'actual',
         out / 'runtime.pipeline.csv'], out / 'execution.log', env)
    run([exe('runtime', 'ftlpu_icu_program_export'), out / 'actual.runtime.ftlpu',
         out / 'runtime_icu_programs', '--pre-execution-dir', out / 'actual.pre_execution'], out / 'export.log')
    actual = read_bf16(out / 'actual.0.bin', x.shape).astype('f8')
    reference = bf16(x * (1.0 / np.sqrt(np.mean(x*x, axis=1, keepdims=True) + epsilon)) * gamma).astype('f8')
    metrics = []
    for a, r, scale in zip(actual, reference, scales):
        if scale:
            cosine = float(a @ r / (np.linalg.norm(a) * np.linalg.norm(r))) if np.any(a) else 0.
            nrmse = float(np.linalg.norm(a - r) / np.linalg.norm(r))
            passed = np.isfinite(a).all() and nrmse <= .01 and cosine >= .9999
            metrics.append(dict(input_scale=float(scale), nrmse=nrmse, cosine=cosine, passed=bool(passed)))
        else:
            metrics.append(dict(input_scale=0, passed=bool(np.all(a == 0))))
    passed = all(case['passed'] for case in metrics) and 'host_operations=0' in (out/'execution.log').read_text()
    summary.write_text(json.dumps(dict(status='passed' if passed else 'failed', epsilon=epsilon,
                                       limits=dict(nrmse_max=.01, cosine_min=.9999), rows=metrics), indent=2))
    if not passed:
        raise AssertionError(f'RMSNorm variance-range regression failed, see {summary}')
    print(f'RMSNorm variance-range regression passed, {len(metrics)} rows')


if __name__ == '__main__':
    main()
