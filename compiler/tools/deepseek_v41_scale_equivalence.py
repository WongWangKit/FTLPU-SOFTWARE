"""Audit BF16 dequantization and FP32 K reduction order on the pinned Q/KV fixtures."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from deepseek_v41_qkv_reference import CASE, BLOCK, REVISION, E4M3FN, scales, tensor
from import_hf_decoder_layer import bf16


def operand_stats():
    return dict(elements=0, zero_to_zero=0, nonfinite=0, underflow_to_zero=0, subnormal=0,
                bf16_inexact=0, min_nonzero_abs=None, max_abs=0.)


def dequant_bf16(raw, multiplier, stats):
    # FP64 covers the entire finite E4M3FN * UE8M0 exponent range.
    exact = raw.astype(np.float64) * multiplier
    actual = bf16(exact.astype(np.float32))
    nonzero = abs(exact[exact != 0])
    stats['elements'] += exact.size
    stats['zero_to_zero'] += int(np.count_nonzero((exact == 0) & (actual == 0)))
    stats['nonfinite'] += int(np.count_nonzero(~np.isfinite(actual)))
    stats['underflow_to_zero'] += int(np.count_nonzero((exact != 0) & (actual == 0)))
    stats['subnormal'] += int(np.count_nonzero((actual != 0) & (abs(actual) < 2. ** -126)))
    stats['bf16_inexact'] += int(np.count_nonzero(actual != exact))
    if nonzero.size:
        minimum = float(nonzero.min())
        previous = stats['min_nonzero_abs']
        stats['min_nonzero_abs'] = minimum if previous is None else min(minimum, previous)
        stats['max_abs'] = max(stats['max_abs'], float(nonzero.max()))
    return actual


def errors(expected, actual):
    delta = expected.astype(np.float64) - actual.astype(np.float64)
    return dict(count=delta.size, unequal=int(np.count_nonzero(delta)),
                bf16_bit_differences=int(np.count_nonzero(
                    bf16(expected).view(np.uint32) != bf16(actual).view(np.uint32))),
                max_abs=float(np.max(np.abs(delta))),
                relative_l2=float(np.linalg.norm(delta) / np.linalg.norm(expected.astype(np.float64))))


def run(case, role, activation, result, weights, reference):
    def load(name):
        info = reference['tensors'][name]
        path = case / 'reference' / info['file']
        if hashlib.sha256(path.read_bytes()).hexdigest() != info['sha256']:
            raise ValueError(f'SHA256 mismatch in reference tensor {name}')
        return np.load(path, allow_pickle=False)

    original = load(activation)
    a = load(activation + '_fp8')
    sa = scales(load(activation + '_scale'))
    values = E4M3FN[a].reshape(*sa.shape, BLOCK)
    a_stats, w_stats = operand_stats(), operand_stats()
    physical = dequant_bf16(values, sa[..., None], a_stats)
    rows, k = original.shape
    n = weights['tensors'][role]['shape'][0]
    groups = k // BLOCK
    w = tensor(case / 'weights', weights, role, 'F8_E4M3', (n, k))
    sw = scales(tensor(case / 'weights', weights, role + '_scale', 'F8_E8M0', (n // BLOCK, groups)))
    a_blocks = values.transpose(1, 0, 2).copy()
    physical_blocks = physical.transpose(1, 0, 2).copy()
    original_blocks = original.reshape(rows, groups, BLOCK).transpose(1, 0, 2).copy()
    # Early and late scaling use the same FP32 reduction implementation for each K32.
    late, early, grouped_scalar, across, skipped = [np.empty((rows, n), dtype=np.float32) for _ in range(5)]
    lhs = physical.reshape(rows, k).T.copy()
    partial_differences, max_partial_error = 0, 0.
    for first in range(0, n, 256):
        stop = min(n, first + 256)
        width = stop - first
        stripe = slice(first, stop)
        raw_w = E4M3FN[w[stripe]].reshape(width, groups, BLOCK)
        stripe_scales = sw[np.arange(first, stop) // BLOCK, :]
        bf_w = dequant_bf16(raw_w, stripe_scales[..., None], w_stats)
        physical_w_blocks = bf_w.transpose(1, 2, 0).copy()
        late_blocks = a_blocks @ raw_w.transpose(1, 2, 0).copy()
        late_blocks *= sa.T[:, :, None]
        late_blocks *= stripe_scales.T[:, None, :]
        early_blocks = physical_blocks @ physical_w_blocks
        partial_differences += int(np.count_nonzero(late_blocks.view(np.uint32) != early_blocks.view(np.uint32)))
        max_partial_error = max(max_partial_error, float(np.max(abs(late_blocks - early_blocks))))
        skipped_blocks = original_blocks @ physical_w_blocks
        for destination, blocks in ((late, late_blocks), (early, early_blocks), (skipped, skipped_blocks)):
            total = np.zeros((rows, width), dtype=np.float32)
            for group in range(groups):
                total += blocks[group]
            destination[:, stripe] = total
        # One continuous FP32 accumulation in ascending individual K order. Each
        # product is exactly representable because both operands originate in E4M3FN.
        rhs = bf_w.reshape(width, k).T.copy()
        total = np.zeros((rows, width), dtype=np.float32)
        grouped = np.zeros_like(total)
        for start in range(0, k, BLOCK):
            partial = np.zeros_like(total)
            for index in range(start, start + BLOCK):
                product = lhs[index, :, None] * rhs[index, :]
                partial += product
                total += product
            grouped += partial
        grouped_scalar[:, stripe] = grouped
        across[:, stripe] = total
    golden = load(result).reshape(rows, n)
    lo = float(np.min(sa)) * float(np.min(sw))
    hi = float(np.max(sa)) * float(np.max(sw))
    grid = lo * 2. ** -18
    upper = hi * k * 448. ** 2
    rounded_continuous, rounded_grouped = bf16(across), bf16(grouped_scalar)
    positions = np.argwhere(rounded_continuous.view(np.uint32) != rounded_grouped.view(np.uint32))
    examples = [dict(token=int(i), output_channel=int(j),
                     grouped_fp32=float(grouped_scalar[i, j]), continuous_fp32=float(across[i, j]),
                     grouped_bf16=float(rounded_grouped[i, j]), continuous_bf16=float(rounded_continuous[i, j]))
                for i, j in positions[:3]]
    return dict(role=role, input=activation, shape_mnk=[rows, n, k], k32_count=groups,
        activation_scale_exponent_range=[int(np.log2(np.min(sa))), int(np.log2(np.max(sa)))],
        weight_scale_exponent_range=[int(np.log2(np.min(sw))), int(np.log2(np.max(sw)))],
        scaled_operands={'activation': a_stats, 'weight': w_stats},
        product_grid_and_sum_bounds={'minimum_exact_nonzero_lattice_spacing': grid,
            'max_sum_of_abs_products_bound': upper,
            'inside_fp32_normal_range': 2. ** -126 <= grid <= upper < 2. ** 127},
        grouped_partial_count=groups * rows * n, grouped_partial_fp32_bit_differences=partial_differences,
        grouped_partial_max_abs=max_partial_error,
        grouped_output_early_vs_late=errors(late, early),
        grouped_early_round_bf16_vs_saved=errors(golden, bf16(early)),
        continuous_k_order_vs_grouped=errors(late, across),
        grouped_scalar_vs_grouped_blas=errors(early, grouped_scalar),
        continuous_vs_grouped_identical_scalar_mac_order=errors(grouped_scalar, across),
        bf16_examples_from_changing_group_boundaries=examples,
        input_quantization_error=errors(original, physical.reshape(rows, k)),
        skip_activation_quant_vs_grouped=errors(late, skipped))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, default=CASE)
    args = parser.parse_args()
    case = args.package.resolve()
    weights = json.loads((case / 'weights/provenance.json').read_text())
    ref = json.loads((case / 'reference/summary.json').read_text())
    if weights['revision'] != REVISION or ref['revision'] != REVISION:
        raise ValueError('Expected pinned DeepSeek V4.1 Flash checkpoint')
    roles = [('wq_a', 'x', 'q_a'), ('wkv', 'x', 'kv_raw'), ('wq_b', 'q_norm', 'q_pre_rope')]
    checks = [run(case, *role, weights, ref) for role in roles]
    passed = all(x['grouped_partial_fp32_bit_differences'] == 0 and
        x['grouped_output_early_vs_late']['unequal'] == 0 and
        x['grouped_early_round_bf16_vs_saved']['bf16_bit_differences'] == 0 and
        x['product_grid_and_sum_bounds']['inside_fp32_normal_range'] and
        all(v['bf16_inexact'] == v['subnormal'] == v['nonfinite'] == 0 for v in x['scaled_operands'].values())
        for x in checks)
    report = dict(status='passed' if passed else 'failed', evidence='CPU numerical reference only',
        revision=REVISION, numpy_version=np.__version__, fixture_seed=ref['seed'],
        acceptance=[
            'Match pinned E4M3FN and UE8M0 quantizer codes exactly before dequantization.',
            'For these fixtures every scaled E4M3FN value must equal its BF16 input to the MAC exactly; no underflow, overflow, or FTZ.',
            'For a fixed per-block dot evaluation order, moving power-of-two scales to the operands preserves each FP32 K32 partial exactly, away from exponent limits.',
            'Reset the FP32 partial accumulator every 32 K values; add completed partials to a separate FP32 accumulator in ascending K32 order. Round only the final output to BF16.',
            'With the same within-block reduction, require identical final BF16 results. Different within-block reduction requires separate numerical validation against the reference.'
        ],
        note='Checkpoint pre-quantization floating weights are unavailable. The reported quantization error isolates activation rounding and holds the checkpoint weights fixed.',
        checks=checks)
    dest = case / 'scale_equivalence_cpu'
    dest.mkdir(exist_ok=True)
    (dest / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    for item in checks:
        print(item['role'], 'partial mismatches:', item['grouped_partial_fp32_bit_differences'],
              'continuous FP32/BF16 differences:', item['continuous_k_order_vs_grouped'])
    if not passed:
        raise ValueError('See scale_equivalence_cpu/summary.json')


if __name__ == '__main__':
    main()
