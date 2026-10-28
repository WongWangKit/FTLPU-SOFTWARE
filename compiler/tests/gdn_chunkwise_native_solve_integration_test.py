"""GDN numerical integration experiment, NOT an end-to-end hardware lowering.

Only coefficient expansion and triangular solve execute on native MEM/ICU/SXM/
VXM. NumPy supplies the other GDN stages as an oracle. Explicit intermediate
host transfers are intentional here and must never be reported as full GDN
runtime performance. The C++ runner receives data only, not Python schedules.
"""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from gated_delta_chunkwise_runtime_test import bf16, recurrent


def nrmse(actual, expected):
    return float(np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-30))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runner', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2364)
    tokens, dim, chunk = 64, 128, 32
    q, k = [rng.normal(size=(tokens, dim)).astype(np.float32) for _ in range(2)]
    q = bf16(q / np.linalg.norm(q, axis=1, keepdims=True) / np.sqrt(dim))
    k = bf16(k / np.linalg.norm(k, axis=1, keepdims=True))
    v = bf16(rng.normal(0, .2, size=(tokens, dim)).astype(np.float32))
    alpha = rng.uniform(.90, .999, (tokens, 1)).astype(np.float32)
    beta = rng.uniform(.05, .95, (tokens, 1)).astype(np.float32)
    initial_state = rng.normal(0, .02, size=(dim, dim)).astype(np.float32)
    reference_output, reference_state = recurrent(q, k, v, alpha, beta, initial_state)
    state = initial_state.copy()
    output, chunks = [], []
    for first in range(0, tokens, chunk):
        folder = out / f'chunk{first // chunk}'
        folder.mkdir(parents=True, exist_ok=True)
        qs, ks, vs = [x[first:first+chunk] for x in (q, k, v)]
        a, b = alpha[first:first+chunk], beta[first:first+chunk]
        gamma = np.cumprod(a[:, 0])[:, None]
        decay = np.eye(chunk, dtype=np.float32)
        for row in range(1, chunk):
            decay[row, :row] = decay[row-1, :row] * a[row, 0]
        lower = np.tril((ks @ ks.T) * decay * b, -1)
        rhs = b * (vs - gamma * (ks @ state))
        lower.astype('<f4').tofile(folder / 'lower.f32.bin')
        rhs.astype('<f4').tofile(folder / 'rhs.f32.bin')
        with (folder / 'runner.log').open('w') as log:
            subprocess.run([str(args.runner.resolve()), str(folder),
                            '--packed-coefficients', str(folder)],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        hardware = json.loads((folder / 'summary.json').read_text())
        assert hardware['status'] == 'passed'
        d = np.fromfile(folder / 'actual.f32.bin', '<f4').reshape(chunk, dim)
        exact_d = np.linalg.solve(np.eye(chunk, dtype=np.float32) + lower, rhs)
        # From this point onward d is the actual SRAM readback. No replacement
        # with oracle values is permitted, including the next chunk's state.
        output.append(gamma * (qs @ state) + ((qs @ ks.T) * decay) @ d)
        state = gamma[-1, 0] * state + ks.T @ (decay[-1, :, None] * d)
        chunks.append(dict(index=first // chunk, solve_nrmse=nrmse(d, exact_d),
                           hardware_cycles=hardware['cycles'],
                           native_sram_mismatches=hardware['mismatches']))
    output = np.concatenate(output)
    errors = dict(output_nrmse=nrmse(output, reference_output),
                  state_nrmse=nrmse(state, reference_state))
    threshold = .003
    passed = all(x < threshold for x in errors.values()) and all(c['solve_nrmse'] < threshold for c in chunks)
    summary = dict(status='passed' if passed else 'failed',
                   backend='mixed NumPy oracle / native CModel triangular solve',
                   end_to_end_hardware=False, real_model_weights=False,
                   tokens=tokens, chunk_size=chunk, heads=1, key_dim=dim, value_dim=dim,
                   nonzero_initial_state=True, state_carried_from_native_solve=True,
                   hardware_stages=['coefficient_expansion', 'triangular_solve'],
                   reference_stages=['decay_prefix', 'gram', 'rhs', 'output', 'state_update'],
                   precision='BF16 L/RHS/D storage; FP32 VXM accumulation; other stages FP32 oracle',
                   threshold_nrmse=threshold, chunks=chunks, errors=errors)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    output.astype('<f4').tofile(out / 'output.f32.bin')
    state.astype('<f4').tofile(out / 'state.f32.bin')
    print(json.dumps(summary, indent=2))
    assert passed, 'GDN native-solve precision budget exceeded'


if __name__ == '__main__':
    main()
