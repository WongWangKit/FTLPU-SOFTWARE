"""Layer-0 Wkv K32 projection with original FP8 weight bytes on the WS port."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np

from deepseek_v41_qkv_norm_test import bf16_input, reference_tensor, run, trace_times
from deepseek_v41_qkv_reference import E4M3FN, REVISION, act_quant, dequant_act, project, scales, tensor
from import_hf_decoder_layer import bf16_bits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('build-dir', 'target-config', 'package'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--k-group', type=int, default=0)
    parser.add_argument('--projection-only', action='store_true',
                        help='Exercise the MXM primitive with already quantized/dequantized BF16 inputs.')
    parser.add_argument('--fp32-output', action='store_true', help='Verify the K32 partial before final rounding.')
    args = parser.parse_args()
    case = args.package.resolve()
    out = case / 'fp8_iw_wkv' / f'kgroup_{args.k_group}'
    if args.projection_only:
        out = out / 'projection_only'
    if args.fp32_output:
        out = out / 'fp32'
    out.mkdir(parents=True, exist_ok=True)
    report = out / 'summary.json'
    report.write_text(json.dumps(dict(status='running')))
    weights = case / 'weights'
    meta = json.loads((weights / 'provenance.json').read_text())
    reference = json.loads((case / 'reference/summary.json').read_text())
    if reference['revision'] != REVISION or meta['revision'] != REVISION or not 0 <= args.k_group < 160:
        raise ValueError('Expected pinned layer 0 and K group 0..159')
    cols = slice(32 * args.k_group, 32 * (args.k_group + 1))
    x = reference_tensor(case/'reference', reference, 'x', (32, 5120))[:, cols]
    w = tensor(weights, meta, 'wkv', 'F8_E4M3', (512, 5120))[:64, cols].copy()
    s = tensor(weights, meta, 'wkv_scale', 'F8_E8M0', (16, 160))[:2, args.k_group:args.k_group+1]
    se, sw = scales(s).ravel()
    a, b, c = 'tensor<32x32xbf16>', 'tensor<32x64xf8E4M3FN>', 'tensor<32x64xbf16>'
    if args.fp32_output:
        c = 'tensor<32x64xf32>'
    codes, sx = act_quant(x)
    qdq = '' if args.projection_only else (
        f'%xq = ftlpu.kernel.block_fp8_qdq %x {{ftlpu.logical_region=0:i64}} : ({a}) -> {a}')
    operand = '%x' if args.projection_only else '%xq'
    attrs = '' if args.projection_only else (
        f'attributes {{ftlpu.logical_inputs=["{(out/"x.bin").as_posix()}","{(out/"w.bin").as_posix()}"], '
        f'ftlpu.artifact_directory="{out.as_posix()}"}}')
    ir = f'''module attributes {{ftlpu.sxm_wide = true}} {{
      func.func @main(%x: {a}, %w: {b}) -> {c} {attrs} {{
        {qdq}
        %y = "ftlpu.kernel.matmul"({operand}, %w) {{m=32:i64, k=32:i64, n=64:i64, unit="MXM",
            ftlpu.logical_region=1:i64, rhs_scale={se:.9e}:f32,
            rhs_scale_west={sw:.9e}:f32}} : ({a}, {b}) -> {c}
        return %y : {c}
      }}
    }}'''
    (out/'kernel.mlir').write_text(ir)
    bf16_input(dequant_act(codes, sx) if args.projection_only else x, (32,32)).tofile(out/'x.bin')
    w.T.copy().tofile(out/'w.bin')
    if args.projection_only:
        (out/'inputs.txt').write_text(f'{out/"x.bin"}\n{out/"w.bin"}\n')
    ext = '.exe' if os.name == 'nt' else ''
    build = args.build_dir.resolve()
    env = {k:v for k,v in os.environ.items() if not k.startswith('FTLPU_')}
    env.update(FTLPU_FILE_RUNNER_MODEL_SESSION='1', FTLPU_FILE_RUNNER_SXM_ARCHITECTURE='wide')
    run([build/'compiler'/('ftlpu-compile'+ext), '--input',out/'kernel.mlir','--input-stage','kernel',
         '--target-config',args.target_config.resolve(),'--icu-compression','raw-macro','--mxm-execution','ws',
         '--output',out/'program.ftlpu'],out/'compile.log',env)
    trace = out/'runtime.pipeline.csv'
    run([build/'runtime'/('compiled_stage_file_runner'+ext),out/'program.ftlpu',out/'inputs.txt',out/'actual',
         trace],out/'execution.log',env)
    run([build/'runtime'/('ftlpu_icu_program_export'+ext),out/'actual.runtime.ftlpu',out/'runtime_icu_programs',
         '--pre-execution-dir',out/'actual.pre_execution'],out/'export.log',env)
    expected = bf16_bits(project(codes, sx, w, s))
    dtype = '<u2'
    if args.fp32_output:
        left = dequant_act(codes, sx)
        right = E4M3FN[w] * scales(s)[np.arange(64) // 32]
        expected = np.zeros((32,64), 'f4')
        for column in range(32):
            expected += left[:,column,None] * right[:,column]
        dtype = '<f4'
    actual = np.fromfile(out/'actual.0.bin',dtype).reshape(expected.shape)
    mismatches = int(np.count_nonzero(actual != expected))
    stats = dict(re.findall(r'(\w+)=(\d+)',(out/'execution.log').read_text().splitlines()[-1]))
    with trace.open(newline='') as stream:
        loads = [row for row in csv.DictReader(stream)
                 if row['resource'] in ('MXM.E0.Load', 'MXM.W0.Load')]
    live = ({row['resource'] for row in loads} == {'MXM.E0.Load', 'MXM.W0.Load'}
            and all('mode=fp8_dequant_bf16' in row['detail'] for row in loads))
    passed = mismatches == 0 and live and stats.get('host_operations') == '0'
    report.write_text(json.dumps(dict(status='passed' if passed else 'failed', k_group=args.k_group,
        input_shape=[32,32], weight_shape=[64,32], revision=REVISION, scale_bytes=s.ravel().tolist(),
        weight_input_sha256=hashlib.sha256((out/'w.bin').read_bytes()).hexdigest(),
        observed_fp8_iw=live, device_activation_qdq=not args.projection_only,
        output_dtype='f32' if args.fp32_output else 'bf16',
        bf16_mismatches=mismatches, count=actual.size,
        timing=trace_times(trace), runtime=stats),indent=2))
    if not passed:
        raise ValueError(report.read_text())
    print(report.read_text())


if __name__ == '__main__':
    main()
