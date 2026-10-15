#!/usr/bin/env python3
"""Execute one real-size input RMSNorm through native ICU/FU and C2C I/O."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from qwen3_5_compiled_layer import bf16, load_imported_layer
from qwen3_5_resident_layer import ResidentGraph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('opt', 'compile', 'runner', 'package', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--seq-len', type=int, default=32)
    parser.add_argument('--target-config', type=Path,
                        default=Path(__file__).resolve().parents[3] / 'FTLPU-CMODEL' / 'config' / 'ftlpu-lpu32.json',
                        help='Hardware configuration shared with dense tests')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / 'summary.json').unlink(missing_ok=True)
    x, weights, _, _, config = load_imported_layer(args.package)
    if not 1 <= args.seq_len <= len(x):
        raise ValueError('seq-len must fit the imported activation')
    x = x[:args.seq_len]
    gamma = bf16(1 + weights.input_norm).reshape(1, -1)
    graph = ResidentGraph(args.opt.resolve(), output / 'kernels', args.target_config)
    graph.norm('input_rmsnorm', x, gamma, config.rms_norm_eps)
    # A standalone shared operator already owns its complete physical layout.
    # Keep the compiler's distributed input/output and feedback scratch plan.
    node = graph.nodes[0]
    schedule = output / 'rmsnorm.schedule.mlir'
    schedule.write_text(node['schedule'].read_text(encoding='utf-8'), encoding='utf-8')
    program = output / 'rmsnorm.ftlpu'
    subprocess.run([str(args.compile.resolve()), '--input', str(schedule),
                    '--input-stage', 'verified-schedule', '--output', str(program)], check=True)
    files = []
    for index, (value, dtype) in enumerate(node['inputs']):
        path = output / f'input{index}.bin'
        if dtype == 'bf16':
            (bf16(value).view(np.uint32) >> 16).astype('<u2').tofile(path)
        else:
            np.asarray(value).astype('<f4' if dtype == 'f32' else 'i1').tofile(path)
        files.append(str(path))
    listing = output / 'inputs.txt'
    listing.write_text('\n'.join(files) + '\n', encoding='utf-8')
    prefix = output / 'rmsnorm'
    shutil.rmtree(output / 'rmsnorm.pre_execution', ignore_errors=True)
    for name in ('rmsnorm.runtime.ftlpu', 'rmsnorm.0.bin', 'runtime.pipeline.csv'):
        (output / name).unlink(missing_ok=True)
    env = os.environ.copy()
    env['FTLPU_FILE_RUNNER_MODEL_SESSION'] = '1'
    # Link-only diagnostics must never count as an arithmetic test.
    env.pop('FTLPU_SESSION_LINK_ONLY', None)
    execution = subprocess.run([str(args.runner.resolve()), str(program), str(listing),
                                str(prefix), str(output / 'runtime.pipeline.csv')],
                               env=env, capture_output=True, text=True)
    (output / 'execution.log').write_text(execution.stdout + execution.stderr, encoding='utf-8')
    if execution.returncode:
        raise RuntimeError(execution.stderr)
    runtime_program = output / 'rmsnorm.runtime.ftlpu'
    if not runtime_program.exists():
        raise RuntimeError('runner did not produce an actual ModelSession runtime image')
    actual = (np.fromfile(output / 'rmsnorm.0.bin', '<u2').astype(np.uint32) << 16).view(np.float32).reshape(x.shape)
    expected = x * (1 / np.sqrt(np.mean(x * x, axis=1, keepdims=True) + config.rms_norm_eps)) * gamma
    diff = actual.astype(np.float64) - expected.astype(np.float64)
    norm = np.linalg.norm(expected.astype(np.float64))
    nrmse = float(np.linalg.norm(diff) / max(norm, 1e-30))
    cosine = float(np.dot(actual.ravel().astype(np.float64), expected.ravel().astype(np.float64)) /
                   max(np.linalg.norm(actual.astype(np.float64)) * norm, 1e-30))
    if not np.isfinite(actual).all() or nrmse > .01 or cosine < .9999:
        raise AssertionError(f'RMSNorm nrmse={nrmse} cosine={cosine}')
    stats = dict((key, int(value)) for key, value in re.findall(r'(\w+)=(\d+)', execution.stdout))
    summary = dict(status='passed',operator='input_rmsnorm',model='Qwen/Qwen3.5-0.8B',
                   shape=list(x.shape),epsilon=config.rms_norm_eps,logical_operators=1,
                   lowering='shared kernel.rms_norm / vxm_feedback',
                   target_config=str(args.target_config.resolve()) if args.target_config else 'compiler default',
                   sram_tiles=1,output_layout_joins=0,program_count=1,host_intermediate_transfers=0,
                   arithmetic_backend='CModel ICU/FU',external_io='C2C',
                   gamma='BF16(1 + checkpoint input_layernorm.weight)',
                   reference='FP32 RMSNorm using the same BF16 activation and gamma',
                   nrmse=nrmse,cosine=cosine,max_abs=float(np.max(np.abs(diff))),
                   pass_limits=dict(nrmse=.01,cosine=.9999),execution_stats=stats,
                   compiler_bytes=program.stat().st_size,runtime_bytes=runtime_program.stat().st_size,
                   runtime_sha256=hashlib.sha256(runtime_program.read_bytes()).hexdigest())
    (output / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
