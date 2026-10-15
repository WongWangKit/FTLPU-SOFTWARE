#!/usr/bin/env python3
"""Validate official layer weights through the direct ICU compiler and CModel.

Inputs are unchanged BF16 layer-0 subsets downloaded by
download_official_dense_layer.py, with pinned revisions and tensor hashes.
This test uses full model dimensions and embedding token IDs 0..31, converts
Linear weights to per-tensor symmetric W8A16, and executes one decoder prefill
layer. It does not run token generation or the complete multi-layer model.

The pass limits apply to CModel versus the hardware-aware quantized NumPy
reference. Error versus an independent BF16 NumPy checkpoint reference is
reported separately, including quantization and arithmetic approximation.
All run modes update results/dense_transformers_official_weights/<model>.
--verify-only rechecks saved outputs against the runtime/package hashes;
--execute-only runs already prepared packages. Llama may be explicitly
skipped with --skip-models llama3_2_1b when official access is unavailable.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import html
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

from download_official_dense_layer import OFFICIAL_REPOS, sha256
from generate_dense_decoder_layer import generate
from import_hf_decoder_layer import (SafeTensorStore, bf16, decoder_config,
    checkpoint_prefixes, load_decoder_weights, attention_head_dim,
    rope_parameters, rope_theta_value, residual_scale, layer_uses_rope,
    rope_at_positions, read_bf16, write_bf16)

ROOT = Path(__file__).resolve().parents[2]

def write_summary(directory):
    reports = [json.loads(path.read_text(encoding='utf-8'))
               for path in sorted(directory.glob('*/result.json'))]
    (directory / 'acceptance.json').write_text(json.dumps(reports, indent=2), encoding='utf-8')
    rows = []
    for report in reports:
        comparison = report.get('hardware_vs_quantized_reference', {})
        native = report.get('hardware_vs_native_checkpoint', {})
        name = html.escape(report['model'])
        pipeline = (directory / report['model'] / 'pipeline.html').is_file()
        label = f'<a href="{name}/pipeline.html">{name}</a>' if pipeline else name
        fields = [label, html.escape(report['status'])]
        for key in ('max_error', 'mean_error', 'relative_l2', 'cosine'):
            fields.append(f'{comparison[key]:.6g}' if key in comparison else '—')
        fields.append(f"{native['relative_l2']:.6g}" if 'relative_l2' in native else '—')
        rows.append('<tr>' + ''.join('<td>' + field + '</td>' for field in fields) + '</tr>')
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>官方权重单层验收</title><style>
body{font:16px system-ui;max-width:1150px;margin:40px auto;color:#182333}
table{border-collapse:collapse;width:100%}th,td{padding:12px;text-align:left;border-bottom:1px solid #ccd5df}
a{color:#1267ae}p{line-height:1.6}</style>
<h1>官方权重单层验收</h1>
<p>seq32，第 0 层，完整模型尺寸。输入使用官方 embedding 的 token 0–31。
官方 BF16 权重转换为对称 per-tensor W8A16。表中 max/mean/L2/cosine
比较真实 CModel 输出与硬件算术参考；最后一列与原始 BF16 checkpoint 的独立 NumPy 参考比较，包含权重量化与算术近似误差。
通过条件：max ≤ 0.35、mean ≤ 0.035、相对 L2 ≤ 0.05、cosine ≥ 0.99。</p>
<p>点击模型查看真实运行的流水图。每个模型目录包含来源与 SHA-256、config、result.json、日志和输出。</p>
<table><tr><th>模型 / pipeline</th><th>结果</th><th>max</th><th>mean</th><th>相对 L2</th><th>cosine</th><th>vs BF16 相对 L2</th></tr>
''' + ''.join(rows) + '</table></html>'
    (directory / 'index.html').write_text(page, encoding='utf-8')
    return reports

def metrics(actual, expected):
    a, b = actual.astype(np.float64).reshape(-1), expected.astype(np.float64).reshape(-1)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('non-finite numerical output')
    delta = a - b
    denominator = np.linalg.norm(b)
    return dict(max_error=float(np.max(np.abs(delta))), mean_error=float(np.mean(np.abs(delta))),
        rmse=float(np.sqrt(np.mean(delta ** 2))),
        relative_l2=float(np.linalg.norm(delta) / max(denominator, 1e-30)),
        cosine=float(np.dot(a, b) / max(np.linalg.norm(a) * denominator, 1e-30)),
        reference_rms=float(np.sqrt(np.mean(b ** 2))))

def native_bf16_reference(checkpoint, seq_len=32):
    """BF16 checkpoint math before W8A16 conversion, independent of ICU code.

    Exact rsqrt/exp are used here. The existing hardware-aware quantized
    reference remains a separate comparison; native error is reported rather
    than being confused with implementation error or lossless acceptance.
    """
    root_config = json.loads((checkpoint / 'config.json').read_text())
    config = decoder_config(root_config)
    prefix, embedding_name = checkpoint_prefixes(root_config, 0)
    store = SafeTensorStore(checkpoint)
    weights, _ = load_decoder_weights(store, prefix, config)
    hidden, heads, kv_heads = int(config['hidden_size']), int(config['num_attention_heads']), int(config['num_key_value_heads'])
    dim = attention_head_dim(config)
    x = bf16(store.read(embedding_name)[:seq_len] * float(config.get('scale_emb', 1)))
    def norm(value, gamma):
        variance = np.mean(value * value, axis=-1, keepdims=True, dtype=np.float32)
        return bf16(bf16(value / np.sqrt(variance + np.float32(config['rms_norm_eps']))) * gamma)
    def linear(value, role):
        return bf16(value @ weights[role].T)
    normalized = norm(x, store.read(prefix + '.input_layernorm.weight'))
    q = linear(normalized, 'query').reshape(seq_len, heads, dim)
    k = linear(normalized, 'key').reshape(seq_len, kv_heads, dim)
    v = linear(normalized, 'value').reshape(seq_len, kv_heads, dim)
    inv, amplitude, rotary = rope_parameters(config, seq_len)
    if config['model_type'] == 'glm':
        def rotate(value):
            angle = np.arange(seq_len, dtype=np.float32)[:, None] * inv[None, :]
            c, s = bf16(np.cos(angle) * amplitude)[:, None, :], bf16(np.sin(angle) * amplitude)[:, None, :]
            low, high = value[..., 0::2], value[..., 1::2]
            return bf16(np.stack((low*c-high*s, high*c+low*s), axis=-1).reshape(value.shape))
        q, k = rotate(q), rotate(k)
    else:
        def rotate(value):
            return rope_at_positions(value, np.arange(seq_len, dtype=np.float32),
                rope_theta_value(config), inverse_frequencies=inv,
                attention_factor=amplitude, rotary_dim=rotary,
                use_rope=layer_uses_rope(config, 0))
        q, k = rotate(q), rotate(k)
    q = q.transpose(1, 0, 2)
    k = np.repeat(k, heads // kv_heads, axis=1).transpose(1, 0, 2)
    v = np.repeat(v, heads // kv_heads, axis=1).transpose(1, 0, 2)
    scores = bf16(bf16(q @ k.transpose(0, 2, 1)) * np.float32(1 / np.sqrt(dim)))
    scores = np.where(np.triu(np.ones((seq_len, seq_len), dtype=bool), 1), -np.inf, scores)
    exp = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    probability = bf16(exp / np.sum(exp, axis=-1, keepdims=True))
    context = bf16((probability @ v).transpose(1, 0, 2).reshape(seq_len, heads * dim))
    scale = residual_scale(config)
    residual = bf16(x + bf16(linear(context, 'output') * scale))
    normalized = norm(residual, store.read(prefix + '.post_attention_layernorm.weight'))
    gate, up = linear(normalized, 'gate'), linear(normalized, 'up')
    with np.errstate(over='ignore'):
        swiglu = bf16(bf16(gate / (1 + np.exp(-gate))) * up)
    return bf16(residual + bf16(linear(swiglu, 'down') * scale))

def source_ir(config, name):
    head = attention_head_dim(config)
    inv, amplitude, rotary = rope_parameters(config, 32)
    text = generate(ROOT / 'compiler/examples/qwen2_5_1_5b_decoder_layer/decoder_layer_seq32.stablehlo.mlir',
        name=name, hidden=int(config['hidden_size']), intermediate=int(config['intermediate_size']),
        query_heads=int(config['num_attention_heads']), kv_heads=int(config['num_key_value_heads']),
        head_dim=head, rope_theta=rope_theta_value(config), rms_norm_eps=float(config['rms_norm_eps']),
        rotary_dim=rotary, rope_attention_factor=amplitude, residual_scale=residual_scale(config))
    # Emit the exact configured frequency vector in the mathematical graph.
    # The compiler extracts it into native constant-table binding metadata.
    values = ', '.join(f'{float(v):.9e}' for v in inv)
    text, count = re.subn(r'%attention_inverse_frequency = stablehlo.power[^\n]+\n\s*tensor<\d+xf32>',
        f'%attention_inverse_frequency = stablehlo.constant dense<[{values}]> : tensor<{len(inv)}xf32>', text)
    if count != 1:
        raise ValueError('could not emit exact model RoPE frequency vector')
    return text

def run_model(a, model):
    d, work, checkpoint = (a.result_dir or a.results / model), a.work / model, (a.model_dir or a.checkpoints / model)
    d.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    report = dict(model=model, source_repo=OFFICIAL_REPOS[model], checkpoint_kind='official_layer_subset',
        sequence_length=32, layers=1, layer=0, quantization='symmetric_per_tensor_w8a16',
        started_at=datetime.now(timezone.utc).isoformat(), status='running',
        limits=dict(max_error=0.35, mean_error=0.035, relative_l2=0.05, cosine_min=0.99))
    def save():
        temporary = d / 'result.json.part'
        temporary.write_text(json.dumps(report, indent=2), encoding='utf-8')
        temporary.replace(d / 'result.json')
    save()
    if model in a.skip_models:
        report.update(status='skipped_access', checkpoint_kind='unavailable',
            reason='official Llama repo returned GatedRepo 401 after retry; user authorized skipping')
        save()
        return report
    try:
        provenance = json.loads((checkpoint / 'provenance.json').read_text())
        if (provenance['repo'] != OFFICIAL_REPOS[model]
                or sha256(checkpoint / 'model.safetensors') != provenance['subset_sha256']
                or sha256(checkpoint / 'config.json') != provenance['config_sha256']):
            raise ValueError('official source provenance mismatch')
        report['checkpoint_provenance'] = provenance
        report['input_token_ids'] = provenance['embedding_token_ids'][:32]
        shutil.copy2(checkpoint / 'provenance.json', d / 'provenance.json')
        shutil.copy2(checkpoint / 'config.json', d / 'config.json')
        config = decoder_config(json.loads((checkpoint / 'config.json').read_text()))
        source = work / 'decoder.stablehlo.mlir'
        source.write_text(source_ir(config, model), encoding='utf-8')
        golden, program, package = work / 'golden', work / 'decoder.ftlpu', work / (model + '.ftlpum')
        environment = os.environ.copy()
        environment['FTLPU_SESSION_PROGRESS'] = '1'
        if a.no_pipeline:
            environment.pop('FTLPU_DENSE_PIPELINE_CSV', None)
        else:
            environment['FTLPU_DENSE_PIPELINE_CSV'] = str((d / 'runtime.pipeline.csv').resolve())
        environment['FTLPU_DENSE_ACTUAL_BF16'] = str((d / 'actual.bf16.bin').resolve())
        phases = [
            ('reference', [sys.executable, '-B', str(ROOT/'compiler/tools/import_hf_decoder_layer.py'),
                '--model-dir', str(checkpoint), '--output-dir', str(golden), '--seq-len', '32']),
            ('compile', [str(a.compiler), '--input', str(source), '--output', str(program),
                '--target-config', str(a.target), '--weight-bank', '1', '--ffn-schedule', 'fused',
                '--attention-schedule', 'fused', '--projection-rope-overlap', 'off']),
            ('package', [sys.executable, '-B', str(ROOT/'compiler/tools/build_decoder_layer_package.py'),
                '--golden-dir', str(golden), '--executable', str(program), '--output', str(package)]),
            ('cmodel', [str(a.runtime), str(package)]),
        ]
        runtime_returncode = 0
        if not a.verify_only:
            for phase, cmd in phases:
                if a.prepare_only and phase == "cmodel":
                    break
                if a.execute_only and phase != "cmodel":
                    continue
                print(model + ': ' + phase, flush=True)
                report['phase'] = phase
                save()
                if phase == 'cmodel':
                    # Never accept a previous execution's output after a failed run.
                    (d / 'actual.bf16.bin').unlink(missing_ok=True)
                with (d / (phase + '.log')).open('w', encoding='utf-8') as out:
                    completed = subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT, env=environment)
                if phase == 'cmodel':
                    (d / 'execution.json').write_text(json.dumps(dict(
                        returncode=completed.returncode,
                        runtime_sha256=sha256(a.runtime),
                        package_sha256=sha256(package)), indent=2), encoding='utf-8')
                if completed.returncode:
                    if phase == 'cmodel' and (d / 'actual.bf16.bin').is_file():
                        runtime_returncode = completed.returncode
                        report['runtime_returncode'] = runtime_returncode
                        continue  # Preserve metrics and trace for numerical failures.
                    raise RuntimeError(phase + ' failed; see ' + str(d / (phase + '.log')))
        if a.prepare_only:
            report.update(status='prepared', finished_at=datetime.now(timezone.utc).isoformat())
            save()
            return report
        execution = json.loads((d / 'execution.json').read_text())
        runtime_returncode = execution['returncode']
        if execution['runtime_sha256'] != sha256(a.runtime) or execution['package_sha256'] != sha256(package):
            raise ValueError('execution belongs to a different runtime or model package; rerun cmodel')
        report['runtime_returncode'] = runtime_returncode
        metadata = json.loads((golden / 'metadata.json').read_text())
        shape = (32, int(config['hidden_size']))
        actual = read_bf16(d / 'actual.bf16.bin', shape)
        expected = read_bf16(golden / 'golden.bf16.bin', shape)
        comparison = metrics(actual, expected)
        report['hardware_vs_quantized_reference'] = comparison
        # Keep source-model precision error separate from implementation error.
        native = native_bf16_reference(checkpoint)
        write_bf16(d / 'native_checkpoint_reference.bf16.bin', native)
        report['quantized_reference_vs_native_checkpoint'] = metrics(expected, native)
        report['hardware_vs_native_checkpoint'] = metrics(actual, native)
        report['native_reference'] = 'NumPy BF16 checkpoints; exact rsqrt and softmax; official per-model RoPE semantics'
        query = read_bf16(golden / 'golden.query.bf16.bin', (32, int(config['num_attention_heads']) * attention_head_dim(config)))
        key = read_bf16(golden / 'golden.key.bf16.bin', (32, int(config['num_key_value_heads']) * attention_head_dim(config)))
        probabilities = read_bf16(golden / 'golden.probability.bf16.bin', (int(config['num_attention_heads']), 32, 32))
        report['attention_coverage'] = dict(query_nonzero_elements=int(np.count_nonzero(query)), key_nonzero_elements=int(np.count_nonzero(key)),
            nonuniform_heads=int(np.count_nonzero(np.ptp(probabilities[:, -1, :], axis=-1) > 0.001)))
        if not np.isfinite(query).all() or not np.isfinite(key).all() or not np.count_nonzero(query) or not np.count_nonzero(key):
            raise ValueError('official Q/K coverage missing')
        if not np.isfinite(probabilities).all() or not report['attention_coverage']['nonuniform_heads']:
            raise ValueError('official nonuniform attention coverage missing')
        for file in ('metadata.json', 'golden.bf16.bin', 'input.bf16.bin'):
            shutil.copy2(golden / file, d / file)
        shutil.copy2(program, d / 'decoder.ftlpu')
        shutil.copy2(str(package) + '.linked.ftlpu', d / 'linked.ftlpu')
        report['runtime_sha256'] = sha256(a.runtime)
        report['compiler_sha256'] = sha256(a.compiler)
        report['executable_sha256'] = sha256(d / 'linked.ftlpu')
        report['actual_sha256'] = sha256(d / 'actual.bf16.bin')
        report['quantized_reference_sha256'] = sha256(d / 'golden.bf16.bin')
        if not a.no_pipeline:
            sys.path.insert(0, str(ROOT / 'tools'))
            from run_dense_pipelines import embedded_viewer, inspect_trace
            report['pipeline'] = inspect_trace(d / 'runtime.pipeline.csv')
            report['pipeline_sha256'] = sha256(d / 'runtime.pipeline.csv')
            embedded_viewer(ROOT / 'tools/pipeline_viewer', d / 'runtime.pipeline.csv', d / 'pipeline.html', model + ' · 官方权重')
        report['numeric_passed'] = runtime_returncode == 0 and comparison['max_error'] <= 0.35 and comparison['mean_error'] <= 0.035 and comparison['relative_l2'] <= 0.05 and comparison['cosine'] >= 0.99
        report['status'] = 'passed' if report['numeric_passed'] else 'numeric_failed'
    except Exception as error:
        report.update(status='failed', error=str(error))
    report['finished_at'] = datetime.now(timezone.utc).isoformat()
    save()
    print(model + ': ' + report['status'], flush=True)
    return report

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models', nargs='+', choices=OFFICIAL_REPOS, default=list(OFFICIAL_REPOS))
    p.add_argument('--skip-models', nargs='*', choices=('llama3_2_1b',), default=[])
    p.add_argument('--checkpoints', type=Path, default=ROOT/'models/official_dense')
    p.add_argument('--work', type=Path, default=ROOT/'tmp/dense_official_weights')
    p.add_argument('--results', type=Path, default=ROOT/'results/dense_transformers_official_weights')
    p.add_argument('--model-dir', type=Path, help='Exact checkpoint directory for a single model')
    p.add_argument('--result-dir', type=Path, help='Exact result directory for a single model')
    p.add_argument('--no-pipeline', action='store_true', help='Run numerical validation without recording the optional pipeline')
    p.add_argument('--compiler', type=Path, default=ROOT/'build-ftlpu-vs2026-direct/compiler/ftlpu-compile.exe')
    p.add_argument('--runtime', type=Path, default=ROOT/'build-ftlpu-vs2026-direct/runtime/hf_decoder_layer_model_session_test.exe')
    p.add_argument('--target', type=Path, default=ROOT.parent/'FTLPU-CMODEL/config/ftlpu-lpu32.json')
    p.add_argument('--jobs', type=int, choices=(1, 2), default=2)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--verify-only', action='store_true')
    mode.add_argument('--prepare-only', action='store_true')
    mode.add_argument('--execute-only', action='store_true')
    a = p.parse_args()
    if (a.model_dir or a.result_dir) and len(a.models) != 1:
        p.error('--model-dir/--result-dir require exactly one model')
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as pool:
        reports = list(pool.map(lambda model: run_model(a, model), a.models))
    # Partial reruns update their model directory and retain the other results.
    if a.result_dir is None:
        write_summary(a.results)
    if any(r['status'] not in ('passed', 'skipped_access', 'prepared') for r in reports):
        raise SystemExit(1)

if __name__ == '__main__':
    main()
