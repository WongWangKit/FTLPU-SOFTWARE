"""Checkpoint Wkv projection through compiler-owned K32 decomposition."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import sys

import numpy as np

from deepseek_v41_qkv_norm_test import bf16_input, reference_tensor, run, trace_times
from deepseek_v41_qkv_reference import REVISION, act_quant, project, tensor
from import_hf_decoder_layer import bf16_bits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('build-dir', 'target-config', 'package'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--k', type=int, default=5120)
    parser.add_argument('--n', type=int, default=512)
    parser.add_argument('--prepare-only', action='store_true', help='Compile without claiming runtime or numerical validation.')
    parser.add_argument('--run-prepared', action='store_true', help='Run a previously compiled case after checking its input and program digests.')
    args = parser.parse_args()
    if args.prepare_only and args.run_prepared:
        parser.error('--prepare-only and --run-prepared are mutually exclusive')
    if args.k < 32 or args.k > 5120 or args.k % 32 or args.n < 64 or args.n > 512 or args.n % 64:
        raise ValueError('Expected K32 aligned within 5120, N64 aligned within 512')
    package = args.package.resolve()
    out = package / 'fp8_wkv_projection' / f'k{args.k}_n{args.n}'
    out.mkdir(parents=True, exist_ok=True)
    report = out / 'summary.json'
    report.write_text(json.dumps({'status': 'running'}))
    meta = json.loads((package / 'weights/provenance.json').read_text())
    refs = json.loads((package / 'reference/summary.json').read_text())
    if refs['revision'] != REVISION or meta['revision'] != REVISION:
        raise ValueError('Expected pinned DeepSeek-V4.1 checkpoint')
    x = reference_tensor(package / 'reference', refs, 'x', (32, 5120))[:, :args.k].copy()
    w = tensor(package / 'weights', meta, 'wkv', 'F8_E4M3', (512, 5120))[:args.n, :args.k].copy()
    scales = tensor(package / 'weights', meta, 'wkv_scale', 'F8_E8M0', (16, 160))[
        :args.n // 32, :args.k // 32].copy()
    bf16_input(x, x.shape).tofile(out / 'x.bin')
    w.tofile(out / 'w.bin')
    scales.tofile(out / 'scales.bin')
    a = f'tensor<32x{args.k}xbf16>'
    b = f'tensor<{args.n}x{args.k}xf8E4M3FN>'
    c = f'tensor<{args.n // 32}x{args.k // 32}xf8E8M0FNU>'
    d = f'tensor<32x{args.n}xbf16>'
    files = ','.join(json.dumps((out / name).as_posix()) for name in ('x.bin', 'w.bin', 'scales.bin'))
    (out / 'kernel.mlir').write_text(f'''module attributes {{
        ftlpu.sxm_wide=true, ftlpu.vector_fp32_ingress=true, ftlpu.vector_bf16_rne_spill=false,
        ftlpu.common_barrier_protocol=true}} {{
      func.func @wkv(%x:{a}, %w:{b}, %s:{c}) -> {d}
        attributes {{ftlpu.logical_inputs=[{files}], ftlpu.artifact_directory="{out.as_posix()}"}} {{
        %y = ftlpu.kernel.block_fp8_projection %x, %w, %s
          {{ftlpu.logical_name="wkv"}} : ({a},{b},{c}) -> {d}
        return %y : {d}
      }}
    }}''')
    build = args.build_dir.resolve()
    ext = '.exe' if os.name == 'nt' else ''
    exe = lambda area, name: build / area / (name + ext)
    env = {k: v for k, v in os.environ.items() if not k.startswith('FTLPU_')}
    env.update(FTLPU_FILE_RUNNER_MODEL_SESSION='1', FTLPU_FILE_RUNNER_SXM_ARCHITECTURE='wide',
               FTLPU_GRAPH_PROGRESS='1', FTLPU_SESSION_PROGRESS='1')
    try:
        def prepared_digests():
            files = [out / name for name in ('x.bin', 'w.bin', 'scales.bin', 'kernel.mlir', 'program.ftlpu')]
            files += [args.target_config.resolve(), exe('compiler', 'ftlpu-compile')]
            return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        prepared = out / 'prepared.json'
        if args.run_prepared:
            if json.loads(prepared.read_text()) != prepared_digests():
                raise ValueError('Prepared projection inputs, program, config, or compiler changed; compile again')
        else:
            run([exe('compiler', 'ftlpu-compile'), '--input', out / 'kernel.mlir', '--input-stage', 'kernel',
                 '--target-config', args.target_config.resolve(), '--icu-compression', 'raw-macro',
                 '--mxm-execution', 'ws', '--output', out / 'program.ftlpu'], out / 'compile.log', env)
            prepared.write_text(json.dumps(prepared_digests(), indent=2))
        if args.prepare_only:
            report.write_text(json.dumps(dict(status='compiled_only', input_shape=list(x.shape),
                weight_shape=list(w.shape), output_shape=[32, args.n], revision=REVISION,
                command=[sys.executable, *sys.argv]), indent=2))
            print(report.read_text())
            return
        trace = out / 'runtime.pipeline.csv'
        run([exe('runtime', 'compiled_stage_file_runner'), out / 'program.ftlpu', out / 'inputs.txt',
             out / 'actual', trace], out / 'execution.log', env)
        run([exe('runtime', 'ftlpu_icu_program_export'), out / 'actual.runtime.ftlpu',
             out / 'runtime_icu_programs', '--pre-execution-dir', out / 'actual.pre_execution'],
            out / 'export.log', env)
        codes, act_scales = act_quant(x)
        expected = bf16_bits(project(codes, act_scales, w, scales))
        actual = np.fromfile(out / 'actual.0.bin', '<u2').reshape(expected.shape)
        mismatches = int(np.count_nonzero(actual != expected))
        with trace.open(newline='') as stream:
            loads = [r for r in csv.DictReader(stream) if r['resource'] in ('MXM.E0.Load', 'MXM.W0.Load')]
        fp8 = bool(loads) and all('mode=fp8_dequant_bf16' in r['detail'] for r in loads)
        stats = dict(re.findall(r'(\w+)=(\d+)', (out / 'execution.log').read_text().splitlines()[-1]))
        passed = not mismatches and fp8 and stats.get('host_operations') == '0'
        result = dict(status='passed' if passed else 'failed', scope='Wkv projection including device activation QDQ',
                      revision=REVISION, input_shape=list(x.shape), weight_shape=list(w.shape),
                      output_shape=list(expected.shape), fp8_iw=fp8, bf16_mismatches=mismatches,
                      count=actual.size, timing=trace_times(trace), runtime=stats,
                      precision=dict(input='BF16', activation_quantization='device E4M3FN RNE, UE8M0 per row K32',
                          weight='checkpoint E4M3FN with UE8M0 per N32/K32, IW dequantization to BF16',
                          partial_and_k_reduction='FP32', output='BF16 RNE'),
                      raw_macro=True, persistent_state=None,
                      artifacts=dict(pipeline='runtime.pipeline.csv', executed_program='actual.runtime.ftlpu',
                          icu_programs='runtime_icu_programs/index.csv', graph_plan='graph_plan.json'),
                      executable_sha256={name: hashlib.sha256(exe(area, name).read_bytes()).hexdigest()
                          for area, name in [('compiler', 'ftlpu-compile'), ('runtime', 'compiled_stage_file_runner'),
                                             ('runtime', 'ftlpu_icu_program_export')]},
                      command=[sys.executable, *sys.argv])
        report.write_text(json.dumps(result, indent=2))
        if not passed:
            raise ValueError(f'Wkv validation failed with {mismatches} mismatches, FP8 IW={fp8}')
        print(report.read_text())
    except Exception as error:
        current = json.loads(report.read_text())
        current.update(status='failed', error=str(error))
        report.write_text(json.dumps(current, indent=2))
        raise


if __name__ == '__main__':
    main()
