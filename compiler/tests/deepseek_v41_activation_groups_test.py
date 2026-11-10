"""Execute independent 32-wide BF16 activation groups with the LPU compiler and runner."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re

import numpy as np

from deepseek_v41_qkv_norm_test import bf16_input, reference_tensor, run, trace_times


def group_source(rows, width, output):
    matrix = f'tensor<{rows}x{width}xbf16>'
    group = f'tensor<{rows}x1xf32>'
    if output in ('scale', 'qdq'):
        result = group if output == 'scale' else matrix
        return f'''module attributes {{ftlpu.sxm_wide = true}} {{
  func.func @group_{output}(%x: {matrix}) -> {result} {{
    %o = ftlpu.kernel.block_fp8_{output} %x : ({matrix}) -> {result}
    return %o : {result}
  }}
}}
'''
    return f'''module attributes {{ftlpu.sxm_wide = true, ftlpu.vector_fp32_ingress = true}} {{
  func.func @group_absmax(%x: {matrix}) -> {group} {{
    %neg = ftlpu.kernel.vector_alu %x, %x {{kind = "negate", use_immediate = true,
      immediate = 0.0 : f32, reduce = false}} : ({matrix}, {matrix}) -> {matrix}
    %abs = ftlpu.kernel.vector_alu %x, %neg {{kind = "max", use_immediate = false,
      immediate = 0.0 : f32, reduce = false}} : ({matrix}, {matrix}) -> {matrix}
    %amax = ftlpu.kernel.vector_alu %abs, %abs {{kind = "max", use_immediate = true,
      immediate = 0.0 : f32, reduce = true, reduction_kind = "max"}} : ({matrix}, {matrix}) -> {group}
    %o = ftlpu.kernel.vector_alu %amax, %amax {{kind = "add", use_immediate = true,
      immediate = 0.0 : f32, reduce = false}} : ({group}, {group}) -> {group}
    return %o : {group}
  }}
}}
'''


def expected_cases(directory, manifest, output):
    from deepseek_v41_qkv_reference import E4M3FN, act_quant, scales

    def archived(name):
        meta = manifest['tensors'][name]
        path = directory/meta['file']
        if hashlib.sha256(path.read_bytes()).hexdigest() != meta['sha256']:
            raise ValueError(f'Reference checksum mismatch for {name}')
        return np.load(path)

    for name, columns, scale_name, code_name in (
        ('x', 5120, 'x_scale', 'x_fp8'),
        ('q_norm', 1280, 'q_norm_scale', 'q_norm_fp8'),
        ('kv_rope', 512, 'kv_scale', 'kv_fp8'),
    ):
        x = reference_tensor(directory, manifest, name, (32, columns))[:, :32]
        y = np.max(np.abs(x), axis=1)
        if output != 'absmax':
            y = scales(archived(scale_name)[:, 0])
        if output == 'qdq':
            y = bf16_input(E4M3FN[archived(code_name)[:, :32]] * y[:, None], (32, 32)).ravel()
        yield name, x, y, dict(input_slice='all 32 tokens, features [0,32)',
                               input_npy_sha256=manifest['tensors'][name]['sha256'])

    # Include adjacent BF16 maxima on both sides of the power-of-two
    # scale boundaries, sign-zero bytes, E4M3 subnormals and halfway values.
    x = np.zeros((32, 32), dtype='<f4')
    x[:, 1::2] = -0.
    for row in range(1, 32):
        k = (-22, -18, -8, -2, 0, 8, 24, 42, 74, 105, 119)[(row-1)//3]
        sign = (-1.) ** row
        x[row, 0] = math.ldexp(448 + 2 * ((row-1) % 3 - 1), k) * sign
        for col, value in enumerate((1.0625, 1.1875, -1.0625, -1.1875, 2**-10,
                                     -2**-10, 3*2**-10, -3*2**-10), start=2):
            x[row, col] = math.ldexp(value, k)
    x[1, -2:] = np.array([0x00010000, 0x80010000], dtype='<u4').view('<f4')
    x[-1, -2:] = np.array([0x7f7f0000, 0xff7f0000], dtype='<u4').view('<f4')
    y = np.max(np.abs(x), axis=1)
    if output != 'absmax':
        codes, encoded = act_quant(x)
        y = scales(encoded).reshape(32)
        if output == 'qdq':
            # Rounding the largest finite BF16 values to E4M3 and scaling
            # back can overflow BF16/FP32 to a signed infinity.
            with np.errstate(over='ignore'):
                decoded = E4M3FN[codes] * y[:, None]
            bits = decoded.view('<u4')
            if np.isnan(decoded).any() or (bits & 65535).any():
                raise ValueError('Expected exactly BF16-representable boundary output')
            y = (bits >> 16).astype('<u2').ravel()
    yield 'boundaries', x, y, dict(input_slice='constructed BF16 scale and E4M3 rounding boundary groups')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', type=Path, required=True)
    parser.add_argument('--target-config', type=Path, required=True)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--output', choices=('absmax', 'scale', 'qdq'), default='absmax')
    parser.add_argument('--case', choices=('x', 'q_norm', 'kv_rope', 'boundaries'))
    parser.add_argument('--verify-recorded', action='store_true',
                        help='Recheck output, recorded program and inputs from a previous execution')
    args = parser.parse_args()
    package = args.package.resolve()
    names = {'absmax': 'isolated_activation_groups', 'scale': 'isolated_activation_scales',
             'qdq': 'isolated_activation_qdq'}
    dest = package / names[args.output]
    dest.mkdir(parents=True, exist_ok=True)
    build = args.build_dir.resolve()
    suffix = '.exe' if os.name == 'nt' else ''
    kernel = dest / 'kernel.mlir'
    program = dest / 'program.ftlpu'
    report_dir = dest/args.case if args.case else dest
    report_dir.mkdir(exist_ok=True)
    report = report_dir/'summary.json'
    source = group_source(32, 32, args.output)
    if args.verify_recorded and kernel.read_text(encoding='utf-8') != source:
        raise ValueError('Recorded Kernel IR differs from the requested entry point')
    report.write_text(json.dumps(dict(status='running', operator=args.output)), encoding='utf-8')
    if not args.verify_recorded:
        kernel.write_text(source, encoding='utf-8')
        run([build/'compiler'/('ftlpu-compile'+suffix), '--input', kernel, '--input-stage', 'kernel',
             '--output', program, '--target-config', args.target_config.resolve(),
             '--icu-compression', 'raw-macro'], dest/'compile.log')
    environment = {k: v for k, v in os.environ.items() if not k.startswith('FTLPU_')}
    environment.update(FTLPU_FILE_RUNNER_MODEL_SESSION='1', FTLPU_FILE_RUNNER_SXM_ARCHITECTURE='wide')
    manifest = json.loads((package/'reference'/'summary.json').read_text())
    reports = []
    for name, x, expected, provenance in expected_cases(package/'reference', manifest, args.output):
        if args.case and name != args.case:
            continue
        directory = dest/name
        directory.mkdir(exist_ok=True)
        input_path = directory/'input.bin'
        input_data = bf16_input(x, (32, 32)).tobytes()
        listing = directory/'inputs.txt'
        trace = directory/'runtime.pipeline.csv'
        if not args.verify_recorded:
            input_path.write_bytes(input_data)
            listing.write_text(str(input_path)+'\n', encoding='utf-8')
            run([build/'runtime'/('compiled_stage_file_runner'+suffix), program, listing,
                 directory/'actual', trace], directory/'execution.log', environment)
            run([build/'runtime'/('ftlpu_icu_program_export'+suffix), directory/'actual.runtime.ftlpu',
                 directory/'runtime_icu_programs', '--pre-execution-dir', directory/'actual.pre_execution'],
                directory/'export.log')
        if input_path.read_bytes() != input_data or (directory/'actual.runtime.ftlpu').read_bytes() != program.read_bytes():
            raise ValueError(f'Recorded {name} input or program has changed')
        dtype = '<u2' if args.output == 'qdq' else '<f4'
        unsigned = '<u2' if args.output == 'qdq' else '<u4'
        actual = np.fromfile(directory/'actual.0.bin', dtype)
        mismatches = int(np.count_nonzero(actual.view(unsigned) != expected.view(unsigned)))
        stats = {k: int(v) for k, v in re.findall(r'(\w+)=(\d+)',
                 (directory/'execution.log').read_text().splitlines()[-1])}
        reports.append(dict(input=name, **provenance, input_sha256=hashlib.sha256(input_path.read_bytes()).hexdigest(),
                            mismatches=mismatches,
                            groups=32, compared_elements=expected.size, timing=trace_times(trace), runtime=stats))
    passed = all(r['mismatches'] == 0 and r['runtime'].get('host_operations') == 0 for r in reports)
    operator = {'absmax': 'BF16 group absmax', 'scale': 'dynamic UE8M0 scale decoded to FP32',
                'qdq': 'Dynamic E4M3FN quantize and dequantize to BF16'}[args.output]
    result = dict(status='passed' if passed else 'failed', operator=operator,
                  input_shape=[32, 32], output_shape=[32, 32 if args.output == 'qdq' else 1],
                  output_dtype='bf16' if args.output == 'qdq' else 'f32', raw_macro=True,
                  entry='isolated vector_alu composition' if args.output == 'absmax' else f'ftlpu.kernel.block_fp8_{args.output}',
                  revision=manifest['revision'], target=args.target_config.name,
                  program_sha256=hashlib.sha256(program.read_bytes()).hexdigest(), cases=reports)
    report.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    if not args.case:
        for case in reports:
            status = 'passed' if case['mismatches'] == 0 and case['runtime'].get('host_operations') == 0 else 'failed'
            record = dict(result, status=status, cases=[case])
            (dest/case['input']/'summary.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result))
    if not passed:
        raise AssertionError(f'{operator} must match bitwise for every test input')


if __name__ == '__main__':
    main()
