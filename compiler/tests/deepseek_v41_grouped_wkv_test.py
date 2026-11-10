"""Raw-macro K32 partials, ordered FP32 sum and final BF16 Wkv output on 32 channels."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from activation_batch_matmul_test import LogicalGraph, compile_graph, run
from activation_state_matmul_test import matmul_fp32
from deepseek_v41_qkv_reference import BLOCK, E4M3FN, REVISION, dequant_act, project, scales, tensor
from import_hf_decoder_layer import bf16, read_bf16


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('package', 'build-dir', 'target-config'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--k', type=int, default=5120)
    args = parser.parse_args()
    k = args.k
    if k % BLOCK or not 64 <= k <= 5120:
        raise ValueError('Expected at least two complete K32 groups within Wkv width 5120')
    case = args.package.resolve()
    out = case / f'isolated_grouped_wkv/k{k}_n32'
    out.mkdir(parents=True, exist_ok=True)
    report_file = out / 'summary.json'
    report_file.write_text(json.dumps(dict(status='pending', k=k)))
    weight_meta = json.loads((case / 'weights/provenance.json').read_text())
    reference = json.loads((case / 'reference/summary.json').read_text())
    if reference['revision'] != REVISION or weight_meta['revision'] != REVISION:
        raise ValueError('Unexpected model revision')
    def load(name):
        spec = reference['tensors'][name]
        path = case / 'reference' / spec['file']
        if hashlib.sha256(path.read_bytes()).hexdigest() != spec['sha256']:
            raise ValueError(f'Digest mismatch in {name}')
        return np.load(path, allow_pickle=False)

    groups = k // 32
    codes, scale = load('x_fp8')[:, :k], load('x_scale')[:, :groups]
    w = tensor(case / 'weights', weight_meta, 'wkv', 'F8_E4M3', (512, 5120))[:32, :k]
    sw = tensor(case / 'weights', weight_meta, 'wkv_scale', 'F8_E8M0', (16, 160))[:1, :groups]
    lhs = dequant_act(codes, scale)
    rhs = (E4M3FN[w].reshape(32, groups, 32) * scales(sw)[..., None]).reshape(32, k).T.copy()
    for operand in (lhs, rhs):
        np.testing.assert_array_equal(operand, bf16(operand))
    official = project(codes, scale, w, sw)
    if k == 5120:
        np.testing.assert_array_equal(official, load('kv_raw')[:, :32])

    graph = LogicalGraph()
    summed = matmul_fp32(graph, lhs[None, :, :], rhs[None, :, :], 'wkv').reshape(32, 32)
    rounded = graph.vector_tiles('output_bf16', [(summed, 'f32')],
                                 lambda alu: alu.add(0, kind='cast', immediate=0, dtype='bf16'))
    exe, env, binary = compile_graph(graph, [rounded, summed], out, args.build_dir.resolve(),
                                    args.target_config.resolve(), fp32_spills=True)
    trace = out / 'runtime.pipeline.csv'
    run([exe('runtime', 'compiled_stage_file_runner'), binary, out / 'inputs.txt', out / 'actual', trace],
        out / 'execution.log', env)
    run([exe('runtime', 'ftlpu_icu_program_export'), out / 'actual.runtime.ftlpu',
        out / 'runtime_icu_programs', '--pre-execution-dir', out / 'actual.pre_execution'], out / 'export.log', env)
    actual = read_bf16(out / 'actual.0.bin', (32, 32))
    pre_round = np.fromfile(out / 'actual.1.bin', '<f4').reshape(32, 32)
    # Independent CPU math directly on native codes. Mirror the K-order of the
    # target MAC and keep both scales after each dot, as in the publisher kernel.
    expected_fp32 = np.zeros_like(pre_round)
    a, b = E4M3FN[codes], E4M3FN[w]
    for group in range(groups):
        partial = np.zeros_like(pre_round)
        for j in range(group * 32, group * 32 + 32):
            partial += a[:, j, None] * b[:, j]
        expected_fp32 += partial * scales(scale[:, group, None]) * scales(sw[:, group])
    np.save(out / 'expected_native_fp32.npy', expected_fp32)
    np.save(out / 'expected_native_bf16.npy', official)
    schedule = (out / 'decoder_layer.schedule.mlir').read_text().splitlines()
    dots = [line for line in schedule if 'schedule.mxm_issue' in line and 'opcode = "compute"' in line]
    additions = [line for line in schedule if 'ftlpu.schedule.vxm ' in line and 'opcode = "add"' in line
                 and 'lhs_kind = "stream_f32"' in line and 'rhs_kind = "stream_f32"' in line]
    indices = [int(re.search(r'\.k(\d+)$', node['name']).group(1))
               for node in json.loads((out / 'graph_plan.json').read_text())['nodes']
               if re.fullmatch(r'wkv\.batch0\.row0\.k\d+', node['name'])]
    stats = dict((key, int(v)) for key, v in re.findall(r'(\w+)=(\d+)',
        (out / 'execution.log').read_text().splitlines()[-1]))
    with trace.open(newline='') as f:
        rows = list(csv.DictReader(f))
    invocation = next(row for row in rows if row['resource'] == 'Session.Invocation')
    start = min(int(row['start']) for row in rows if row['resource'] == 'C2C.HostInput')
    def active(unit):
        return sum((int(r['end']) - int(r['start'])) *
                   (int(r['inner_count']) * int(r['outer_count']) - int(r['skip_first']))
                   for r in rows if r['resource'] == unit)
    mismatches = int(np.count_nonzero(actual.view(np.uint32) != official.view(np.uint32)))
    fp32_mismatches = int(np.count_nonzero(pre_round.view(np.uint32) != expected_fp32.view(np.uint32)))
    cast_errors = int(np.count_nonzero(actual.view(np.uint32) != bf16(pre_round).view(np.uint32)))
    correct_path = (indices == list(range(0, k, BLOCK)) and len(additions) == groups - 1 and bool(dots)
                    and all('cast_target = "fp32"' in line for line in additions)
                    and all('weight_input_mode = "direct16"' in line and
                            'accumulator_output_format = "fp32"' in line for line in dots))
    passed = (correct_path and mismatches == fp32_mismatches == cast_errors == 0
              and stats.get('host_operations') == 0 and all(np.isfinite(pre_round.ravel())))
    report = dict(status='passed' if passed else 'failed', scope='Wkv channels [0,32), fixture supplies losslessly dequantized BF16 operands',
        revision=REVISION, m=32, n=32, k=k, groups=groups, target=args.target_config.name,
        acceptance='FP32 before rounding matches sequential K32/native-scale oracle exactly. Final BF16 bits match native FP8 projection.',
        official_bf16_mismatches=mismatches, native_order_fp32_mismatches=fp32_mismatches,
        max_abs_fp32_error=float(np.max(abs(expected_fp32 - pre_round))), final_cast_mismatches=cast_errors,
        fp32_native_layout_and_order=correct_path, scheduled_fp32_adds=len(additions), elements=1024,
        precision='FP32 accumulator reset at each K32, FP32 ordered fold, final BF16 RNE',
        activation_and_weight_scale_placement='applied losslessly outside device as input fixture',
        invocation_cycles=int(invocation['end'])-int(invocation['start']),
        cold_ingress_cycles=int(invocation['start'])-start, cold_cycles_through_compute=int(invocation['end'])-start,
        diagnostic_readback_cycles=active('C2C.HostOutput'), readback_outputs=['BF16 result', 'FP32 preceding final cast'],
        active_cycles_per_unit={u: active(u) for u in ('MXM.E0.Compute','MXM.W0.Compute','VXM.C0')}, runtime=stats)
    report_file.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))
    if not passed:
        raise RuntimeError(f'See {report_file}')


if __name__ == '__main__':
    main()
