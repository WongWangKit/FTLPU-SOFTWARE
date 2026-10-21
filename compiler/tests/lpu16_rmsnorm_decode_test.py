#!/usr/bin/env python3
"""Compile and numerically execute a 1536-wide RMSNorm on LPU16."""

import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np


def bf16_bits(values: np.ndarray) -> np.ndarray:
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    return ((bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) >> 16).astype('<u2')


def bf16_values(path: Path) -> np.ndarray:
    return (np.fromfile(path, dtype='<u2').astype(np.uint32) << 16).view(np.float32)


def run(command: list[str], log: Path, *, env: dict[str, str] | None = None) -> None:
    result = subprocess.run(command, capture_output=True, text=True, env=env)
    log.write_text(result.stdout + result.stderr, encoding='utf-8')
    result.check_returncode()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('opt', 'compile', 'runner', 'target-config', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    example = Path(__file__).resolve().parents[1] / 'examples/rmsnorm/rmsnorm_32x1536.stablehlo.mlir'
    stablehlo = output / 'rmsnorm_1x1536.stablehlo.mlir'
    source = example.read_text(encoding='utf-8')
    source = source.replace('32x1536', '1x1536').replace('tensor<32x', 'tensor<1x')
    stablehlo.write_text(source, encoding='utf-8')
    schedule = output / 'rmsnorm_1x1536.schedule.mlir'
    binary = output / 'rmsnorm_1x1536.ftlpu'
    run([str(args.opt.resolve()), '--input', str(stablehlo), '--output', str(schedule),
         '--pipeline', 'ftlpu-stablehlo-to-schedule', '--target-config',
         str(args.target_config.resolve()), '--weight-bank', '1'],
        output / 'compile-schedule.log')
    run([str(args.compile.resolve()), '--input', str(schedule), '--input-stage',
         'verified-schedule', '--output', str(binary), '--target-config',
         str(args.target_config.resolve())], output / 'compile-binary.log')

    logical_x = np.linspace(-1, 1, 1536, dtype=np.float32)
    logical_gamma = 0.75 + 0.5 * np.arange(1536, dtype=np.float32) / 1536
    x_file, gamma_file = output / 'input.bin', output / 'gamma.bin'
    bf16_bits(logical_x).tofile(x_file)
    bf16_bits(logical_gamma).tofile(gamma_file)
    inputs = output / 'inputs.txt'
    inputs.write_text(f'{x_file}\n{gamma_file}\n', encoding='utf-8')
    prefix = output / 'rmsnorm'
    env = os.environ.copy()
    env['FTLPU_FILE_RUNNER_MODEL_SESSION'] = '1'
    env.pop('FTLPU_SESSION_LINK_ONLY', None)
    run([str(args.runner.resolve()), str(binary), str(inputs), str(prefix),
         str(output / 'runtime.pipeline.csv')], output / 'run.log', env=env)
    actual = bf16_values(output / 'rmsnorm.0.bin')
    x, gamma = bf16_values(x_file), bf16_values(gamma_file)
    expected = x / np.sqrt(np.mean(x * x) + 1e-6) * gamma
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise AssertionError('RMSNorm output shape or finiteness failed')
    difference = actual - expected
    nrmse = float(np.linalg.norm(difference) / np.linalg.norm(expected))
    tail_nrmse = float(np.linalg.norm(difference[1024:]) / np.linalg.norm(expected[1024:]))
    cosine = float(np.dot(actual, expected)
                   / (np.linalg.norm(actual) * np.linalg.norm(expected)))
    summary = {'status': 'passed' if nrmse < .01 and tail_nrmse < .01
               and cosine > .9999 else 'failed', 'nrmse': nrmse,
               'last512_nrmse': tail_nrmse, 'cosine': cosine,
               'max_abs': float(np.max(np.abs(difference))),
               'target_config': str(args.target_config.resolve())}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n',
                                         encoding='utf-8')
    print(json.dumps(summary, indent=2))
    if summary['status'] != 'passed':
        raise AssertionError('LPU16 segmented gamma RMSNorm numerical check failed')


if __name__ == '__main__':
    main()
