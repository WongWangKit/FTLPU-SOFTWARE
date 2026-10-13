#!/usr/bin/env python3
"""Record dense-model CModel pipelines and embed them in the existing viewer."""
from __future__ import annotations

import argparse
import base64
import csv
import gzip
import hashlib
import html
import json
import os
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

MODELS = ['llama3_2_1b', 'ministral3_3b', 'phi4_mini', 'smollm3_3b',
          'glm_edge_1_5b', 'minicpm4_0_5b']
LABELS = dict(zip(MODELS, ['Llama3.2-1B', 'Ministral3-3B', 'Phi-4-mini',
                         'SmolLM3-3B', 'GLM-Edge-1.5B', 'MiniCPM4-0.5B']))


def embedded_viewer(viewer: Path, trace: Path, output: Path, name: str):
    source = (viewer / 'index.html').read_text(encoding='utf-8')
    css = (viewer / 'style.css').read_text(encoding='utf-8')
    js = (viewer / 'app.js').read_text(encoding='utf-8')
    payload = base64.b64encode(gzip.compress(trace.read_bytes(), mtime=0)).decode('ascii')
    loader = '''
(async () => {
  dom.interactionState.textContent = "正在载入实际运行流水";
  const packed = Uint8Array.from(atob(PAYLOAD), ch => ch.charCodeAt(0));
  const stream = new Blob([packed]).stream().pipeThrough(new DecompressionStream("gzip"));
  const blob = await new Response(stream).blob();
  await loadFile(new File([blob], TRACE_NAME, {type: "text/csv"}));
  dom.interactionState.textContent = "实际运行 · 包含最初数据读入";
})().catch(error => { dom.interactionState.textContent = error.message; });
'''.replace('PAYLOAD', json.dumps(payload)).replace('TRACE_NAME', json.dumps(name + ' / runtime.pipeline.csv'))
    source = source.replace('<link rel="stylesheet" href="style.css">', '<style>' + css + '</style>')
    source = source.replace('<script src="app.js"></script>',
                            '<script>' + (js + loader).replace('</script', '<\\/script') + '</script>')
    source = source.replace('<title>FTLPU Pipeline Viewer</title>',
                            '<title>' + html.escape(name) + ' · 实际运行流水</title>')
    source = source.replace('<h1>FTLPU Pipeline Viewer</h1>',
                            '<h1>' + html.escape(name) + ' · Pipeline</h1>')
    output.write_text(source, encoding='utf-8')


def inspect_trace(path: Path):
    families, resources = {}, set()
    start, end, count, logical_count = None, 0, 0, 0
    with path.open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        if not {'start', 'end', 'resource', 'detail'}.issubset(reader.fieldnames or []):
            raise ValueError('unexpected pipeline CSV columns')
        for row in reader:
            a, b = int(row['start']), int(row['end'])
            if b <= a:
                raise ValueError('invalid event interval')
            start = a if start is None else min(start, a)
            inner = int(row.get('inner_count') or 1)
            outer = int(row.get('outer_count') or 1)
            pattern_end = b + (inner - 1) * int(row.get('inner_interval') or 0)
            pattern_end += (outer - 1) * int(row.get('outer_interval') or 0)
            end = max(end, pattern_end)
            count += 1
            logical_count += inner * outer - int(row.get('skip_first') or 0)
            resource = row['resource']
            resources.add(resource)
            family = resource.split('.')[0]
            families[family] = families.get(family, 0) + 1
    for family in ('MEM', 'MXM', 'VXM', 'SXM', 'C2C'):
        if not families.get(family):
            raise ValueError('missing functional-unit family: ' + family)
    if start != 0:
        raise ValueError('trace must include initial data loading from physical cycle zero')
    return {'start_cycle': start, 'end_cycle': end, 'events': count, 'logical_events': logical_count,
            'families': families, 'resources': sorted(resources)}


def run_model(args, model):
    directory = args.results / model
    directory.mkdir(parents=True, exist_ok=True)
    package = (args.work / model / (model + '.ftlpum')).resolve()
    trace = directory / 'runtime.pipeline.csv'
    temporary = directory / 'runtime.pipeline.csv.part'
    report_path = directory / 'pipeline.run.json'
    prior_report = json.loads(report_path.read_text(encoding='utf-8')) if report_path.exists() else {}
    started = datetime.now(timezone.utc).isoformat()
    report = {'model': model, 'status': 'running', 'started_at': started,
              'source': 'CModel actual physical-cycle ICU/FU issue trace',
              'sequence_length': 32, 'layers': 1, 'qk_pattern': 'nonzero',
              'package': str(package), 'package_sha256': hashlib.sha256(package.read_bytes()).hexdigest()}
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    if not args.render_only:
        environment = os.environ.copy()
        environment['FTLPU_DENSE_PIPELINE_CSV'] = str(temporary.resolve())
        environment['FTLPU_SESSION_PROGRESS'] = '1'
        temporary.unlink(missing_ok=True)
        print(model + ': running CModel', flush=True)
        with (directory / 'pipeline.cmodel.log').open('w', encoding='utf-8') as log:
            completed = subprocess.run([str(args.runtime.resolve()), str(package)],
                                       stdout=log, stderr=subprocess.STDOUT, env=environment)
        report['returncode'] = completed.returncode
        if temporary.exists():
            temporary.replace(trace)
        output = (directory / 'pipeline.cmodel.log').read_text(encoding='utf-8', errors='replace')
        metrics = re.search(r'max_error=([\d.e+-]+) mean_error=([\d.e+-]+)', output)
        if metrics:
            report.update(max_error=float(metrics[1]), mean_error=float(metrics[2]))
        report['numeric_passed'] = completed.returncode == 0
        linked = Path(str(package) + '.linked.ftlpu')
        if linked.exists():
            shutil.copy2(linked, directory / 'pipeline.linked.ftlpu')
    else:
        report.update(prior_report)
        report['rendered_at'] = started
        log = directory / 'pipeline.cmodel.log'
        if log.exists():
            metrics = re.search(r'max_error=([\d.e+-]+) mean_error=([\d.e+-]+)', log.read_text(encoding='utf-8', errors='replace'))
            if metrics:
                report.update(max_error=float(metrics[1]), mean_error=float(metrics[2]))
    try:
        report.update(inspect_trace(trace))
        embedded_viewer(args.viewer, trace, directory / 'pipeline.html', LABELS[model])
        report['csv_sha256'] = hashlib.sha256(trace.read_bytes()).hexdigest()
        report['status'] = 'passed' if report.get('numeric_passed') else 'numeric_failed'
        report.pop('error', None)
    except Exception as error:
        report.update(status='failed', error=str(error))
    report['finished_at'] = datetime.now(timezone.utc).isoformat()
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(model + ': ' + report['status'], flush=True)
    return report


def overview(results: Path):
    cards = []
    reports = []
    for model in MODELS:
        path = results / model / 'pipeline.run.json'
        if not path.exists():
            continue
        report = json.loads(path.read_text(encoding='utf-8'))
        reports.append(report)
        ready = (results / model / 'pipeline.html').exists()
        title = html.escape(LABELS[model])
        link = f'<a href="{model}/pipeline.html">{title} →</a>' if ready else title
        detail = (f"{report.get('events', 0):,} 条记录 · physical cycle 0–{report.get('end_cycle', 0):,}"
                  if ready else '正在记录完整单层流水')
        cards.append(f'<article><h2>{link}</h2><p>{detail}</p><p>{html.escape(report["status"])}</p>'
                     f'<a href="{model}/runtime.pipeline.csv">下载 CSV</a></article>')
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Dense Transformer Pipelines</title>
<style>body{font:16px system-ui;background:#f3f6fa;color:#203347;margin:40px auto;max-width:1100px;padding:0 24px}
h1{font-size:30px}p{line-height:1.7}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:20px}
article{background:white;border:1px solid #d4dee8;border-radius:12px;padding:24px}h2{font-size:21px;margin-top:0}
a{color:#176b93;text-decoration:none}a:hover{text-decoration:underline}.intro{margin-bottom:28px}</style>
<h1>Dense Transformer · 实际运行流水</h1><p class="intro">seq32 · 单层 prefill · 合成非零 Q/K<br>
从最初数据读入开始，包含 C2C、同步等待和 MEM / MXM / VXM / SXM。点击模型打开原 Pipeline Viewer；
支持缩放、拖动、功能单元筛选和 A/B 游标。各图使用实际 physical cycle，保留真实等待。</p><main>CARDS</main></html>'''
    (results / 'pipelines.html').write_text(page.replace('CARDS', ''.join(cards)), encoding='utf-8')
    (results / 'pipelines.json').write_text(json.dumps(reports, indent=2), encoding='utf-8')


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=MODELS, default=MODELS)
    parser.add_argument('--work', type=Path, default=root / 'tmp/dense_nonzero_qk')
    parser.add_argument('--results', type=Path, default=root / 'results/dense_transformers_nonzero_qk')
    parser.add_argument('--runtime', type=Path, default=root / 'build-ftlpu-vs2026-direct/runtime/hf_decoder_layer_model_session_test.exe')
    parser.add_argument('--viewer', type=Path, default=root / 'tools/pipeline_viewer')
    parser.add_argument('--jobs', type=int, choices=(1, 2), default=2)
    parser.add_argument('--render-only', action='store_true')
    args = parser.parse_args()
    args.results.mkdir(parents=True, exist_ok=True)
    # Smaller models finish first, so a browsable result is available early.
    ordered = sorted(args.models, key=lambda model: MODELS.index(model))
    if not args.render_only:
        ordered = sorted(ordered, key=lambda model: (args.work / model / (model + '.ftlpum')).stat().st_size)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(run_model, args, model) for model in ordered]
        reports = []
        for future in as_completed(futures):
            reports.append(future.result())
            overview(args.results)
    overview(args.results)
    if any(report['status'] != 'passed' for report in reports):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
