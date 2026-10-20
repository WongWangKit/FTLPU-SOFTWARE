#!/usr/bin/env python3
"""Run the resident-MEM LPU32 projection experiment and export actual traces.

This tests existing column-IW hardware through distributed ICU queues; it does
not change the compiler's whole-layer weight loading strategy or SR capacity.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import subprocess
from pathlib import Path

from run_dense_pipelines import embedded_viewer


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', type=Path, default=root.parent / 'FTLPU-CMODEL'
                        / 'build-vs2026/Release/prefill_weight_streams_test.exe')
    parser.add_argument('--output', type=Path, default=root / 'results/lpu32_prefill_weight_streams')
    args = parser.parse_args()
    if not args.exe.is_file():
        parser.error(f'Build CModel target prefill_weight_streams_test first: {args.exe}')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    # Avoid retaining a previous passing summary if this run fails early.
    (out / 'summary.json').write_text(json.dumps({'status': 'running'}), encoding='utf-8')
    run = subprocess.run([str(args.exe.resolve()), str(out)], text=True,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out / 'test.log').write_text(run.stdout, encoding='utf-8')
    print(run.stdout, end='')
    if run.returncode:
        (out / 'summary.json').write_text(json.dumps({
            'status': 'failed', 'exit_code': run.returncode, 'log': 'test.log'}), encoding='utf-8')
        raise SystemExit(run.returncode)
    summary = json.loads((out / 'summary.json').read_text(encoding='utf-8'))
    if summary.get('status') != 'passed' or len(summary.get('cases', [])) != 6:
        raise RuntimeError('Missing six-case successful numerical report')
    rows = []
    for case in summary['cases']:
        name = f"seq{case['tokens']}_s{case['weight_streams']}"
        trace = out / f'{name}.pipeline.csv'
        with trace.open(encoding='utf-8', newline='') as source:
            reader = csv.DictReader(source)
            if reader.fieldnames != ['start', 'end', 'resource', 'detail']:
                raise ValueError(f'Unexpected trace header: {trace}')
            resources = set()
            for event in reader:
                if int(event['end']) <= int(event['start']):
                    raise ValueError(f'Invalid trace interval: {event}')
                resources.add(event['resource'])
            if not {'MXM.E0.Load', 'MXM.E0.Compute'}.issubset(resources):
                raise ValueError(f'Missing MXM runtime events: {trace}')
        viewer = out / f'{name}.pipeline.html'
        embedded_viewer(root / 'tools/pipeline_viewer', trace, viewer,
                        f'LPU32 {name} — resident-MEM projection')
        page = viewer.read_text(encoding='utf-8').replace(
            '实际运行 · 包含最初数据读入',
            '实际运行 · MEM 已驻留 · 单 MXM projection · 不含 DDR/C2C')
        viewer.write_text(page, encoding='utf-8')
        rows.append('<tr>' + ''.join(f'<td>{case[key]}</td>' for key in (
            'tokens', 'weight_streams', 'first_compute', 'compute_cycles',
            'internal_bubbles', 'max_abs_error')) +
            f'<td><a href="{html.escape(viewer.name)}">流水图</a> / '
            f'<a href="{html.escape(trace.name)}">CSV</a></td></tr>')
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>LPU32 prefill 权重 stream 对照</title>
<style>body{font:16px system-ui;margin:40px;color:#20344f;max-width:1200px}
table{border-collapse:collapse;width:100%}th,td{padding:12px;border-bottom:1px solid #ccd6e0;text-align:left}
th{background:#edf2f7}a{color:#24589a}p{line-height:1.7}</style>
<h1>LPU32 prefill：1 / 8 条权重 stream 对照</h1>
<p>单 MXM、W8A16、K=128、N=64，8 个权重块，两个 weight buffer 交替复用。
真实 MEM → SR → MXM → ACC → MEM 数据通路；全部输出与 CPU 参考以及 8-stream 基线比较。</p>
<p>seq32 / seq64：单 stream 没有增加稳态空泡，首次计算晚 28 cycles。
seq16：单 stream 每次块切换等待 16 cycles，7 次切换共 112 cycles。</p>
<table><tr><th>Tokens</th><th>权重 streams</th><th>首次计算 cycle</th><th>计算发射 cycles</th>
<th>内部空泡 cycles</th><th>最大绝对误差</th><th>实际流水</th></tr>ROWS</table>
<p>范围：输入预先驻留 MEM，不含 DDR/C2C、完整 decoder 层或编译器 direct lowering。
单 stream 使用 LOAD_COLUMN_3D，8-stream 使用 FU 兼容路径；未裁减物理 SR 数量。流水中的 MXM Compute 表示 tile0 计算发射，
MEM 行展示所有 tile 的实际传输。</p><p><a href="summary.json">数值与周期报告</a> / <a href="test.log">运行日志</a></p></html>'''
    (out / 'index.html').write_text(page.replace('ROWS', ''.join(rows)), encoding='utf-8')
    print(out / 'index.html')


if __name__ == '__main__':
    main()
