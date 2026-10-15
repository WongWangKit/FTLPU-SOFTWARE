#!/usr/bin/env python3
"""Real Qwen3.5 RMSNorm -> QKV projection -> Conv1D/SiLU in one ICU program."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from qwen3_5_compiled_layer import bf16, load_imported_layer
from qwen3_5_reference import causal_depthwise_conv1d


def metrics(actual, expected):
    a, e = actual.astype(np.float64).ravel(), expected.astype(np.float64).ravel()
    if a.shape != e.shape or not np.isfinite(a).all():
        raise AssertionError('invalid output shape or non-finite arithmetic')
    return dict(nrmse=float(np.linalg.norm(a-e) / max(np.linalg.norm(e), 1e-30)),
                cosine=float(np.dot(a, e) / max(np.linalg.norm(a)*np.linalg.norm(e), 1e-30)),
                max_abs=float(np.max(np.abs(a-e))))


def mirrored_runtime_intervals(pipeline):
    """Verify both halves issue MAC and SiLU together in the actual trace."""
    with pipeline.open(newline='', encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    def end(row):
        return int(row['end']) + (int(row['inner_count'])-1)*int(row['inner_interval']) \
            + (int(row['outer_count'])-1)*int(row['outer_interval'])
    projection_end = max(end(r) for r in rows if r['resource'].startswith('MXM.')
                         and r['resource'].endswith('.Compute'))
    intervals = {}
    for opcode in ('mul', 'neg'):
        for queue in (0, 15):
            intervals[opcode, queue] = sorted((int(r['start']), end(r)) for r in rows
                if r['resource'] == f'VXM.C{queue}'
                and r['detail'].startswith(opcode+' ')
                and int(r['start']) >= projection_end)
        if len(intervals[opcode, 0]) != 3 or intervals[opcode, 0] != intervals[opcode, 15]:
            raise AssertionError(f'mirrored {opcode} domains are not simultaneous: {intervals}')
    return [dict(group=i, channels_per_hemisphere=1024,
                 mac_start=mac[0], mac_end=mac[1], silu_start=silu[0], silu_end=silu[1])
            for i,(mac,silu) in enumerate(zip(intervals['mul',0],intervals['neg',0]))]


def conv_bypass_intervals(pipeline):
    """Record the necessary numerical casts and reject pure-copy ALU domains."""
    with pipeline.open(newline='', encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    projection_end = max(int(r['end']) + (int(r['inner_count'])-1)*int(r['inner_interval'])
                         + (int(r['outer_count'])-1)*int(r['outer_interval'])
                         for r in rows if r['resource'].startswith('MXM.')
                         and r['resource'].endswith('.Compute'))
    uses = {'VXM.C1':'causal_zero_fill_fp32_to_bf16', 'VXM.C14':'causal_zero_fill_fp32_to_bf16',
            'VXM.C7':'convolution_result_fp32_to_bf16', 'VXM.C8':'convolution_result_fp32_to_bf16',
            'VXM.C5':'silu_result_fp32_to_bf16', 'VXM.C10':'silu_result_fp32_to_bf16'}
    result=[]
    for row in rows:
        if not row['resource'].startswith('VXM.') or not row['detail'].startswith('bypass ') \
                or int(row['start']) < projection_end:continue
        if row['resource'] not in uses:
            raise AssertionError(f'unexpected Conv ALU copy: {row}')
        result.append(dict(start=int(row['start']), end=int(row['end']), resource=row['resource'],
                           purpose=uses[row['resource']], conversion='FP32 to BF16'))
    if len(result) != 18 or any(sum(r['purpose']==purpose for r in result)!=6 for purpose in set(uses.values())):
        raise AssertionError(f'expected three numerical casts per Conv channel domain: {result}')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('opt', 'compile', 'runner', 'package', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--target-config', type=Path,
                        default=Path(__file__).resolve().parents[3] / 'FTLPU-CMODEL' / 'config' / 'ftlpu-lpu32.json')
    parser.add_argument('--compile-only', action='store_true')
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / 'summary.json').unlink(missing_ok=True)
    x, weights, packed, scales, config = load_imported_layer(args.package)
    metadata = json.loads((args.package / 'metadata.json').read_text(encoding='utf-8'))
    provenance = metadata.get('checkpoint_provenance', {})
    if provenance.get('repo') != 'Qwen/Qwen3.5-0.8B' or x.shape != (32,1024) or config.conv_width != 6144:
        raise ValueError('requires the real seq32 Qwen3.5-0.8B layer-0 package')
    gamma = bf16(1 + weights.input_norm)
    binary, prefix = out / 'frontend.ftlpu', out / 'frontend'
    if not args.validate_only:
        # Describe the complete logical graph. Tile sizes, pages, SRAM/SR
        # placements, mirroring and cycle domains belong to the C++ compiler.
        kernel = out / 'frontend.kernel.mlir'
        kernel.write_text(f'''module {{
  func.func @prefill_frontend(%x: tensor<32x1024xbf16>, %gamma: tensor<1024xbf16>,
      %weight: tensor<1024x6144xi8>, %conv_weight: tensor<6144x4xbf16>)
      -> (tensor<32x6144xbf16>, tensor<32x1024xbf16>, tensor<32x6144xbf16>) {{
    %normalized = ftlpu.kernel.rms_norm %x, %gamma {{axis = 1 : i64,
      epsilon = {float(config.rms_norm_eps):.9e} : f32}}
      : (tensor<32x1024xbf16>, tensor<1024xbf16>) -> tensor<32x1024xbf16>
    %projected = "ftlpu.kernel.matmul"(%normalized, %weight) {{m = 32 : i64,
      n = 6144 : i64, k = 1024 : i64, unit = "MXM",
      rhs_scale = {float(scales['in_proj_qkv']):.9e} : f32}}
      : (tensor<32x1024xbf16>, tensor<1024x6144xi8>) -> tensor<32x6144xbf16>
    %mixed = ftlpu.kernel.depthwise_conv1d %projected, %conv_weight
      {{kernel_size = 4 : i64, causal = true, activation = "silu"}}
      : (tensor<32x6144xbf16>, tensor<6144x4xbf16>) -> tensor<32x6144xbf16>
    return %mixed, %normalized, %projected
      : tensor<32x6144xbf16>, tensor<32x1024xbf16>, tensor<32x6144xbf16>
  }}
}}
''', encoding='utf-8')
        schedule = out / 'frontend.schedule.mlir'
        subprocess.run([str(args.opt.resolve()), '--input', str(kernel), '--output', str(schedule),
                        '--pipeline', 'ftlpu-stablehlo-to-schedule',
                        '--target-config', str(args.target_config.resolve())], check=True)
        source=schedule.read_text(encoding='utf-8')
        if 'ftlpu.connected_graph_schedule' not in source or 'projection_archive_copies = 0' not in source:
            raise AssertionError('C++ connected graph planning did not execute')
        (out / 'graph.json').write_text(json.dumps(dict(
            operators=['rms_norm','matmul','depthwise_conv1d_silu'],
            input_shape=[32,1024], projection_shape=[32,6144],
            lowering='C++ connected graph planner and shared native operator lowerers',
            kernel=str(kernel), schedule=str(schedule), projection_archive_copies=0), indent=2)+'\n', encoding='utf-8')
        subprocess.run([str(args.compile.resolve()), '--input', str(schedule),
                        '--input-stage', 'verified-schedule', '--output', str(binary)], check=True)
        paths = []
        inputs=((x,'bf16'),(gamma,'bf16'),(packed['in_proj_qkv'].T,'i8'),(weights.conv1d,'bf16'))
        for index, (value, dtype) in enumerate(inputs):
            path = out / f'input{index}.bin'
            if dtype == 'bf16':
                (bf16(value).view(np.uint32) >> 16).astype('<u2').tofile(path)
            else:
                np.asarray(value).astype('<f4' if dtype == 'f32' else 'i1').tofile(path)
            paths.append(str(path))
        listing = out / 'inputs.txt'
        listing.write_text('\n'.join(paths)+'\n', encoding='utf-8')
        if args.compile_only:
            print(f'compiled {binary} ({binary.stat().st_size} bytes)', flush=True)
            return
        shutil.rmtree(out / 'frontend.pre_execution', ignore_errors=True)
        for name in ('frontend.runtime.ftlpu', 'runtime.pipeline.csv', 'frontend.0.bin', 'frontend.1.bin', 'frontend.2.bin'):
            (out / name).unlink(missing_ok=True)
        env = os.environ.copy()
        env['FTLPU_FILE_RUNNER_MODEL_SESSION'] = '1'
        env['FTLPU_SESSION_PROGRESS'] = '1'
        env.pop('FTLPU_SESSION_LINK_ONLY', None)
        print('Executing continuous ICU program with DDR/C2C weight paging', flush=True)
        with (out / 'execution.log').open('w', encoding='utf-8') as log:
            execution = subprocess.run([str(args.runner.resolve()), str(binary), str(listing),
                                        str(prefix), str(out / 'runtime.pipeline.csv')],
                                       env=env, stdout=log, stderr=subprocess.STDOUT)
        if execution.returncode:
            raise RuntimeError((out / 'execution.log').read_text(encoding='utf-8')[-8000:])
    actual = []
    for index, shape in enumerate(((32,6144), (32,1024), (32,6144))):
        actual.append((np.fromfile(out / f'frontend.{index}.bin', '<u2').astype(np.uint32) << 16).view(np.float32).reshape(shape))
    # The reference uses the same W8A16 projection and BF16 gamma/conv weights.
    expected_norm = x / np.sqrt(np.mean(x*x, axis=1, keepdims=True) + config.rms_norm_eps) * gamma
    expected_qkv = expected_norm @ weights.in_proj_qkv.T
    expected_conv, _ = causal_depthwise_conv1d(expected_qkv, weights.conv1d, np.zeros_like(weights.conv1d))
    checks = {name:metrics(a,e) for name,a,e in (
        ('input_rmsnorm',actual[1],expected_norm), ('qkv_projection',actual[2],expected_qkv),
        ('conv1d_silu',actual[0],expected_conv))}
    passed = all(m['nrmse'] <= .03 and m['cosine'] >= .999 for m in checks.values())
    execution_log = (out / 'execution.log').read_text(encoding='utf-8')
    stats_line = [line for line in execution_log.splitlines() if line.startswith('cycles=')][-1]
    runtime_binary = out / 'frontend.runtime.ftlpu'
    if not runtime_binary.exists():
        raise RuntimeError('missing actually loaded ICU program')
    mirrored = mirrored_runtime_intervals(out / 'runtime.pipeline.csv')
    bypasses = conv_bypass_intervals(out / 'runtime.pipeline.csv')
    with (out / 'conv_bypass_runtime.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(bypasses[0]))
        writer.writeheader()
        writer.writerows(bypasses)
    with (out / 'conv_mirrored_runtime.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(mirrored[0]))
        writer.writeheader()
        writer.writerows(mirrored)
    summary = dict(status='passed' if passed else 'failed', model=provenance['repo'],
                   revision=provenance['revision'], layer=metadata['layer'], seq_len=32, hidden_size=1024,
                   qkv_width=6144, q_width=2048, k_width=2048, v_width=2048, conv_kernel_size=4,
                   operators=['input_rmsnorm', 'qkv_projection', 'causal_depthwise_conv1d_silu'],
                   initial_conv_state='zero', program_count=1, host_intermediate_transfers=0,
                   initial_activation_placement='C2C directly to RMSNorm bank0 rows0..127',
                   initial_activation_materialization=False,
                   rmsnorm_to_projection='compiler-planned consumer layout via passive MEM/SR copy',
                   lowering='C++ connected graph planner; no Python hardware lowering',
                   projection_archive_copies=0,
                   conv_execution='three pairs of independent mirrored 1024-channel domains',
                   mirrored_conv_runtime=mirrored,
                   conv_bypass_domains=len(bypasses),
                   conv_data_copies='MEM/SR only; no ALU input-copy or layout-copy domains',
                   arithmetic_backend='CModel ICU/FU', external_io='DDR/C2C', projection_format=metadata['projection_format'],
                   reference='FP32 equations with the same W8A16 projection, BF16 gamma and convolution weights',
                   metrics=checks, pass_limits=dict(nrmse=.03, cosine=.999),
                   execution_stats={k:int(v) for k,v in re.findall(r'(\w+)=(\d+)', stats_line)},
                   cycles_note='static cycles plus drain; real ingress, waits, computation and egress are in runtime.pipeline.csv',
                   compiler_bytes=binary.stat().st_size, runtime_bytes=runtime_binary.stat().st_size,
                   runtime_sha256=hashlib.sha256(runtime_binary.read_bytes()).hexdigest())
    (out / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)
    if not passed:
        raise AssertionError('frontend numerical validation failed')


if __name__ == '__main__':
    main()
