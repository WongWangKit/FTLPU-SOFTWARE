"""One unscaled K32 Wkv partial product from already-quantized layer-0 operands."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from activation_batch_matmul_test import LogicalGraph, compile_graph, run
from activation_state_matmul_test import matmul_fp32
from deepseek_v41_qkv_reference import E4M3FN, REVISION, tensor
from import_hf_decoder_layer import bf16


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('package', 'build-dir', 'target-config'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    case = args.package.resolve()
    out = case / 'isolated_dot32'
    out.mkdir(parents=True, exist_ok=True)
    summary = out / 'summary.json'
    summary.write_text(json.dumps({'status': 'pending'}))
    weights = json.loads((case / 'weights/provenance.json').read_text())
    ref = json.loads((case / 'reference/summary.json').read_text())
    if weights['revision'] != REVISION or ref['revision'] != REVISION:
        raise ValueError('Expected pinned DeepSeek V4.1 Flash checkpoint')
    spec = ref['tensors']['x_fp8']
    source = case / 'reference' / spec['file']
    if hashlib.sha256(source.read_bytes()).hexdigest() != spec['sha256']:
        raise ValueError('Input activation SHA mismatch')
    codes = np.load(source, allow_pickle=False)
    if codes.shape != (32, 5120) or codes.dtype != np.uint8:
        raise ValueError('Expected 32 rows of native E4M3FN codes')
    w = tensor(case / 'weights', weights, 'wkv', 'F8_E4M3', (512, 5120))
    # Checkpoint W has [N,K] order. Hand the compiler a logical [K,N] operand.
    lhs = E4M3FN[codes[:, :32]]
    rhs = E4M3FN[w[:32, :32]].T.copy()
    for operand in (lhs, rhs):
        np.testing.assert_array_equal(bf16(operand), operand)
    graph = LogicalGraph()
    result = matmul_fp32(graph, lhs[None, :, :], rhs[None, :, :], 'wkv_dot32')
    exe, env, binary = compile_graph(graph, [result.reshape(32, 32)], out,
        args.build_dir.resolve(), args.target_config.resolve(), fp32_spills=True)
    trace = out / 'runtime.pipeline.csv'
    run([exe('runtime', 'compiled_stage_file_runner'), binary, out / 'inputs.txt',
         out / 'actual', trace], out / 'execution.log', env)
    run([exe('runtime', 'ftlpu_icu_program_export'), out / 'actual.runtime.ftlpu',
         out / 'runtime_icu_programs', '--pre-execution-dir', out / 'actual.pre_execution'],
        out / 'export.log', env)
    schedule = (out / 'decoder_layer.schedule.mlir').read_text()
    compute_ops = [line for line in schedule.splitlines()
                   if 'ftlpu.schedule.mxm_issue' in line and 'opcode = "compute"' in line]
    direct_fp32 = bool(compute_ops) and all(
        'weight_input_mode = "direct16"' in op and 'accumulator_output_format = "fp32"' in op
        for op in compute_ops)
    actual = np.fromfile(out / 'actual.0.bin', '<f4').reshape(32, 32)
    products = lhs.astype(np.float64)[:, :, None] * rhs.astype(np.float64)[None, :, :]
    reference = np.sum(products, axis=1)
    bounds = np.sum(np.abs(products), axis=1) * 1.e-6
    np.save(out / 'expected.npy', reference)
    np.save(out / 'absolute_error_bounds.npy', bounds)
    difference = np.abs(actual - reference)
    stats = dict((k, int(v)) for k, v in re.findall(r'(\w+)=(\d+)',
        (out / 'execution.log').read_text().splitlines()[-1]))
    with trace.open(newline='') as f:
        activity = list(csv.DictReader(f))
    invocation = next(r for r in activity if r['resource'] == 'Session.Invocation')
    begins = [int(r['start']) for r in activity if r['resource'] == 'C2C.HostInput']
    cold = int(invocation['start']) - min(begins)
    def active(unit):
        return sum((int(r['end']) - int(r['start'])) *
                   (int(r['inner_count']) * int(r['outer_count']) - int(r['skip_first']))
                   for r in activity if r['resource'] == unit)
    mismatches = int(np.count_nonzero(difference > bounds))
    non_bf16 = int(np.count_nonzero(bf16(actual) != actual))
    mxm = {unit: active(unit) for unit in ('MXM.E0.Compute', 'MXM.W0.Compute')}
    passed = (np.isfinite(actual).all() and direct_fp32 and non_bf16 > 0 and mismatches == 0
              and all(mxm.values()) and stats.get('host_operations') == 0)
    report = dict(status='passed' if passed else 'failed', revision=REVISION, m=32, n=32, k=32,
        scope='Unscaled dot only. Input FP8 codes expanded losslessly to BF16 in the test fixture.',
        weights=weights['tensors']['wkv']['name'], x_fp8_file_sha256=spec['sha256'],
        wkv_file_sha256=weights['tensors']['wkv']['sha256'], first_k=0, first_output_channel=0,
        target=args.target_config.name, compute_schedule_direct16_fp32=direct_fp32,
        precision='BF16 inputs; Direct16 MXM multiplication and FP32 accumulator output',
        elements=actual.size, outputs_not_representable_in_bf16=non_bf16, max_abs=float(np.max(difference)),
        mismatches=mismatches, tolerance='1e-6 * sum_k(abs(x[k]*w[k])) per output, FP64 oracle',
        identical_to_fp64=int(np.count_nonzero(actual == reference)), runtime=stats,
        invocation_cycles=int(invocation['end']) - int(invocation['start']),
        cold_ingress_cycles=cold, cold_cycles_through_compute=int(invocation['end']) - min(begins),
        active_cycles_per_mxm=mxm, diagnostic_readback_cycles=active('C2C.HostOutput'))
    summary.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))
    if not passed:
        raise RuntimeError(f'See {summary}')


if __name__ == '__main__':
    main()
