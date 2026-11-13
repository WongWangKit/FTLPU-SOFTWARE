"""Checkpoint Wkv through the shared linear projection lowering and FP8 IW."""
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
from deepseek_v41_qkv_reference import E4M3FN, REVISION, act_quant, dequant_act, project, scales as decode_scales, tensor
from import_hf_decoder_layer import bf16_bits


def projection_structure(out, k, n):
    """Check native task dimensions, not just stage names or timeline labels."""
    plan = json.loads((out / 'graph_plan.json').read_text())
    stages = []
    projections = []
    legacy_intermediates = []
    for node in plan['nodes']:
        native = node.get('native_projections', [])
        stage = {key: node[key] for key in ('node', 'name', 'nominal_start', 'nominal_end') if key in node}
        stage['native_projections'] = native
        stages.append(stage)
        projections.extend(native)
        if re.search(r'\.n\d+\.k\d+\.accumulate$|\.round\.n\d+$', node.get('name', '')):
            legacy_intermediates.append(node['name'])
    dimensions_valid = bool(projections) and all(
        p.get('kind') == 'linear' and p.get('m') == 32 and p.get('k') == k
        and p.get('n', 0) > 0 and p['n'] % 64 == 0
        and 'e4m3' in str(p.get('weight_dtype', '')).lower()
        and p.get('rhs_scale_block_count') == (p['n'] // 32) * (k // 32)
        for p in projections)
    full_n = sum(p.get('n', 0) for p in projections) == n
    transports = plan.get('linker_mem_transports', [])
    compute_transports = [t for t in transports if not t.get('phase', '').startswith('validation_export.')]
    result = dict(
        passed=dimensions_valid and full_n and not legacy_intermediates,
        native_projection_stage_count=sum(bool(s['native_projections']) for s in stages),
        native_projection_count=len(projections), native_projections=projections,
        all_projection_stages_keep_full_k=dimensions_valid,
        projection_output_columns=sum(p.get('n', 0) for p in projections),
        legacy_partial_accumulate_or_round_stages=legacy_intermediates,
        graph_stage_count=len(stages),
        native_output_archive_copies=plan.get('native_output_archive_copies'),
        linker_mem_transport_count=len(transports),
        computation_mem_transport_count=len(compute_transports),
        computation_mem_transport_byte_plane_vectors=sum(
            t['byte_planes'] * t['vectors'] for t in compute_transports),
        static_program_bytes=(out / 'program.ftlpu').stat().st_size)
    (out / 'stages.json').write_text(json.dumps(dict(structure=result, stages=stages), indent=2))
    return result


def grouped_fp32_reference(codes, act_scales, weight, weight_scales):
    """Independent serial K32 partials followed by ascending-group FP32 adds."""
    lhs = dequant_act(codes, act_scales)
    decoded = E4M3FN[weight]
    scale = decode_scales(weight_scales)
    total = np.zeros((len(lhs), len(weight)), dtype=np.float32)
    for group in range(lhs.shape[1] // 32):
        partial = np.zeros_like(total)
        rhs_scale = scale[np.arange(len(weight)) // 32, group]
        for column in range(group * 32, (group + 1) * 32):
            rhs = decoded[:, column] * rhs_scale
            partial += lhs[:, column, None] * rhs[None, :]
        total += partial
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('build-dir', 'target-config', 'package'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--k', type=int, default=5120)
    parser.add_argument('--n', type=int, default=512)
    parser.add_argument('--prepare-only', action='store_true', help='Compile without claiming runtime or numerical validation.')
    parser.add_argument('--run-prepared', action='store_true', help='Run a previously compiled case after checking its input and program digests.')
    parser.add_argument('--projection-only', action='store_true',
                        help='Primitive only: fixture supplies already-QDQ BF16 activation; does not validate device activation quantization.')
    parser.add_argument('--fp32-output', action='store_true',
                        help='With --projection-only, compare FP32 K32 partial accumulation bitwise before final BF16 rounding.')
    args = parser.parse_args()
    if args.prepare_only and args.run_prepared:
        parser.error('--prepare-only and --run-prepared are mutually exclusive')
    if args.fp32_output and not args.projection_only:
        parser.error('--fp32-output is restricted to the explicitly marked --projection-only primitive')
    if args.k < 32 or args.k > 5120 or args.k % 32 or args.n < 64 or args.n > 512 or args.n % 64:
        raise ValueError('Expected K32 aligned within 5120, N64 aligned within 512')
    package = args.package.resolve()
    out = package / 'fp8_wkv_projection' / f'k{args.k}_n{args.n}'
    if args.projection_only:
        out = out / 'projection_only'
        if args.fp32_output:
            out = out / 'fp32'
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
    codes, act_scales = act_quant(x)
    bf16_input(dequant_act(codes, act_scales) if args.projection_only else x, x.shape).tofile(out / 'x.bin')
    # This is the logical KxN matrix operand of matmul, not a physical MEM layout.
    (w.T.copy() if args.projection_only else w).tofile(out / 'w.bin')
    scales.tofile(out / 'scales.bin')
    a = f'tensor<32x{args.k}xbf16>'
    b = f'tensor<{args.n}x{args.k}xf8E4M3FN>'
    c = f'tensor<{args.n // 32}x{args.k // 32}xf8E8M0FNU>'
    d = f'tensor<32x{args.n}x' + ('f32>' if args.fp32_output else 'bf16>')
    names = ('x.bin', 'w.bin') if args.projection_only else ('x.bin', 'w.bin', 'scales.bin')
    files = ','.join(json.dumps((out / name).as_posix()) for name in names)
    operands = f'%x:{a}, %w:{b}, %s:{c}'
    operation = f'''%y = ftlpu.kernel.block_fp8_projection %x, %w, %s
          {{ftlpu.logical_name="wkv"}} : ({a},{b},{c}) -> {d}'''
    if args.projection_only:
        b = f'tensor<{args.k}x{args.n}xf8E4M3FN>'
        operands = f'%x:{a}, %w:{b}'
        scale_values = ','.join(f'{value:.9e}' for value in decode_scales(scales).ravel())
        operation = f'''%y = "ftlpu.kernel.matmul"(%x, %w)
          {{m=32:i64, k={args.k}:i64, n={args.n}:i64, unit="MXM", ftlpu.logical_name="wkv",
            rhs_scale=1.0:f32, rhs_block_scales=array<f32: {scale_values}>}}
          : ({a},{b}) -> {d}'''
    (out / 'kernel.mlir').write_text(f'''module attributes {{
        ftlpu.sxm_wide=true, ftlpu.vector_fp32_ingress=true, ftlpu.vector_bf16_rne_spill=false,
        ftlpu.common_barrier_protocol=true}} {{
      func.func @wkv({operands}) -> {d}
        attributes {{ftlpu.logical_inputs=[{files}], ftlpu.artifact_directory="{out.as_posix()}"}} {{
        {operation}
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
        structure = projection_structure(out, args.k, args.n)
        report.write_text(json.dumps(dict(status='compiled', structure=structure,
            device_activation_qdq=not args.projection_only, projection_only=args.projection_only,
            output_dtype='f32' if args.fp32_output else 'bf16'), indent=2))
        if not structure['passed']:
            raise ValueError('Expected shared full-K linear projection stages covering N, with no split-K graph accumulation')
        if args.prepare_only:
            report.write_text(json.dumps(dict(status='compiled_only', input_shape=list(x.shape),
                weight_shape=list(w.shape), output_shape=[32, args.n], revision=REVISION,
                structure=structure, device_activation_qdq=not args.projection_only,
                projection_only=args.projection_only, output_dtype='f32' if args.fp32_output else 'bf16',
                command=[sys.executable, *sys.argv]), indent=2))
            print(report.read_text())
            return
        trace = out / 'runtime.pipeline.csv'
        run([exe('runtime', 'compiled_stage_file_runner'), out / 'program.ftlpu', out / 'inputs.txt',
             out / 'actual', trace], out / 'execution.log', env)
        run([exe('runtime', 'ftlpu_icu_program_export'), out / 'actual.runtime.ftlpu',
             out / 'runtime_icu_programs', '--pre-execution-dir', out / 'actual.pre_execution'],
            out / 'export.log', env)
        expected = bf16_bits(project(codes, act_scales, w, scales))
        reference_bf16_mismatches = 0
        if args.fp32_output:
            fp32_expected = grouped_fp32_reference(codes, act_scales, w, scales)
            reference_bf16_mismatches = int(np.count_nonzero(bf16_bits(fp32_expected) != expected))
            expected = fp32_expected.view(np.uint32)
        actual = np.fromfile(out / 'actual.0.bin', '<u4' if args.fp32_output else '<u2').reshape(expected.shape)
        mismatches = int(np.count_nonzero(actual != expected))
        with trace.open(newline='') as stream:
            loads = [r for r in csv.DictReader(stream) if r['resource'] in ('MXM.E0.Load', 'MXM.W0.Load')]
        fp8 = bool(loads) and all('mode=fp8_dequant_bf16' in r['detail'] for r in loads)
        stats = dict(re.findall(r'(\w+)=(\d+)', (out / 'execution.log').read_text().splitlines()[-1]))
        passed = not mismatches and not reference_bf16_mismatches and fp8 and stats.get('host_operations') == '0'
        scope = ('Wkv projection primitive with fixture-QDQ activation; no device QDQ validation'
                 if args.projection_only else 'Wkv projection including device activation QDQ')
        result = dict(status='passed' if passed else 'failed', scope=scope,
                      revision=REVISION, input_shape=list(x.shape), weight_shape=list(w.shape),
                      output_shape=list(expected.shape), fp8_iw=fp8,
                      bf16_mismatches=None if args.fp32_output else mismatches,
                      fp32_bit_mismatches=mismatches if args.fp32_output else None,
                      grouped_reference_vs_official_bf16_mismatches=reference_bf16_mismatches if args.fp32_output else None,
                      count=actual.size, timing=trace_times(trace), runtime=stats,
                      structure=structure, device_activation_qdq=not args.projection_only,
                      projection_only=args.projection_only,
                      precision=dict(input='BF16', activation_quantization=(
                          'fixture-only E4M3FN QDQ, UE8M0 per row K32' if args.projection_only
                          else 'device E4M3FN RNE, UE8M0 per row K32'),
                          weight='checkpoint E4M3FN with UE8M0 per N32/K32, IW dequantization to BF16',
                          partial_and_k_reduction='independent K32 FP32 partials, ordered FP32 MXM ACC',
                          output='FP32' if args.fp32_output else 'BF16 RNE'),
                      raw_macro=True, persistent_state=None,
                      artifacts=dict(pipeline='runtime.pipeline.csv', executed_program='actual.runtime.ftlpu',
                          icu_programs='runtime_icu_programs/index.csv', graph_plan='graph_plan.json', stages='stages.json'),
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
