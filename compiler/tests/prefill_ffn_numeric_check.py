#!/usr/bin/env python3
"""Check a BF16 RMSNorm/INT8 SwiGLU FFN fixture against NumPy.

Input manifest order: activation, gamma, Gate, Up, Down. Weights have logical
KxN layout. This checker does not imply that a reduced FFN is a full decoder.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def bf16(values):
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    bits = bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)
    return ((bits >> 16).astype(np.uint32) << 16).view(np.float32)


def read_bf16(path, shape):
    return (np.fromfile(path, dtype='<u2').astype(np.uint32) << 16).view(np.float32).reshape(shape)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--tokens', type=int, default=32)
    parser.add_argument('--width', type=int, default=1536)
    parser.add_argument('--hidden', type=int, required=True)
    parser.add_argument('--weight-scale', type=float, default=.01)
    parser.add_argument('--max-nrmse', type=float, default=.01)
    args = parser.parse_args()
    m, k, h = args.tokens, args.width, args.hidden
    x = read_bf16(args.inputs / 'input0.bin', (m, k))
    gamma = read_bf16(args.inputs / 'input1.bin', (k,))
    weights = [np.fromfile(args.inputs / f'input{i + 2}.bin', np.int8)
               .reshape(shape).astype(np.float32) * args.weight_scale
               for i, shape in enumerate(((k, h), (k, h), (h, k)))]
    norm = bf16(x / np.sqrt(np.mean(x * x, axis=1, keepdims=True) + 1e-6) * gamma)
    gate, up = norm @ weights[0], norm @ weights[1]
    expected = bf16(((gate / (1 + np.exp(-gate))) * up) @ weights[2])
    actual = read_bf16(args.output, (m, k))
    delta = actual - expected
    nrmse = float(np.linalg.norm(delta) / max(float(np.linalg.norm(expected)), 1e-12))
    # Check each token and output block too: an average alone can conceal a
    # dropped token tile or a wrong physical hemisphere/result-slot mapping.
    actual_blocks = actual.reshape(m, k // 16, 16)
    reference_blocks = expected.reshape(m, k // 16, 16)
    block_error = np.linalg.norm(actual_blocks - reference_blocks, axis=2)
    block_scale = np.maximum(np.linalg.norm(reference_blocks, axis=2), 1e-6)
    worst_block_nrmse = float(np.max(block_error / block_scale))
    passed = bool(np.isfinite(actual).all() and nrmse < args.max_nrmse
                  and worst_block_nrmse < 5 * args.max_nrmse)
    summary = dict(scope='isolated_rmsnorm_ffn', tokens=m, width=k, hidden=h,
                   output=str(args.output.resolve()), passed=passed,
                   finite_outputs=int(np.isfinite(actual).sum()),
                   nonzero_outputs=int(np.count_nonzero(actual)),
                   correlation=float(np.corrcoef(actual.ravel(), expected.ravel())[0, 1]),
                   rmse=float(np.sqrt(np.mean(delta * delta))),
                   max_abs=float(np.max(np.abs(delta))), nrmse=nrmse,
                   worst_token_feature_block_nrmse=worst_block_nrmse,
                   max_nrmse=args.max_nrmse)
    args.summary.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))
    if not passed:
        raise SystemExit('FFN numerical regression failed')


if __name__ == '__main__':
    main()
