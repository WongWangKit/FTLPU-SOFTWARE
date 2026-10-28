#!/usr/bin/env python3
"""Reproduce the seq32 LPU32 Native / Raw Macro single-layer comparison.

Uses the full-shape, nonzero-Q/K synthetic numerical harness. Official checkpoint
acceptance is a separate test. Common Native and Raw use identical barrier
boundaries; default Native retains its existing overlapping paging policy.

Build ftlpu_compile, ftlpu_binary_inspect, ftlpu_compare_compute_issues and
compiled_qwen2_5_1_5b_decoder_layer_seq32_runtime_test first. Then run this script
with --jobs 3. Use --phase report to regenerate the dashboard without rerunning
the simulator. Existing case directories are updated in place.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import re
import subprocess
import time

from run_dense_pipelines import embedded_viewer, inspect_trace

ROOT = Path(__file__).resolve().parents[1]
LABELS = {'native': 'Native 3D / 默认分页', 'native_common': 'Native 3D / 公共屏障',
          'raw': 'Raw Macro / 公共屏障'}


def execute(command, log, env=None):
    started = time.time()
    with log.open('w', encoding='utf-8') as output:
        result = subprocess.run(list(map(str, command)), cwd=ROOT, env=env,
                                stdout=output, stderr=subprocess.STDOUT)
    return {'command': list(map(str, command)), 'exit_code': result.returncode,
            'wall_seconds': round(time.time() - started, 3)}


def compile_programs(a):
    source = ROOT / 'compiler/examples/qwen2_5_1_5b_decoder_layer/decoder_layer_seq32.stablehlo.mlir'
    common = a.results / 'common.stablehlo.mlir'
    text = source.read_text(encoding='utf-8')
    assert 'module {' in text
    common.write_text(text.replace('module {',
        'module attributes {ftlpu.common_barrier_protocol = true} {', 1), encoding='utf-8')
    for mode in LABELS:
        command = [a.build / 'compiler/ftlpu-compile.exe', '--input',
                   common if mode == 'native_common' else source,
                   '--input-stage', 'stablehlo', '--target-config',
                   ROOT.parent / 'FTLPU-CMODEL/config/ftlpu-lpu32.json',
                   '--weight-bank', '1', '--kv-cache-capacity', '256',
                   '--mxm-execution', 'vector', '--projection-rope-overlap', 'off',
                   '--ffn-schedule', 'tail', '--icu-compression',
                   'raw-macro' if mode == 'raw' else 'none', '--output', a.results / f'{mode}.ftlpu']
        outcome = execute(command, a.results / f'{mode}.compile.log')
        (a.results / f'{mode}.compile.json').write_text(json.dumps(outcome, indent=2), encoding='utf-8')
        if outcome['exit_code']: raise RuntimeError(f'{mode} compilation failed')


def run_case(a, case):
    mode, bandwidth = case
    directory = a.results / f'{mode}_bw{bandwidth}'
    directory.mkdir(exist_ok=True)
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(('FTLPU_QWEN_', 'FTLPU_CMODEL_LOG', 'FTLPU_LPU64_')) or key in (
                'FTLPU_SKIP_QWEN_NUMERIC_CHECK', 'FTLPU_TRACE_ONLY', 'FTLPU_SCHEDULE_TRACE'):
            del env[key]
    env.update(FTLPU_DDR_BANDWIDTH_MBPS=str(bandwidth), FTLPU_SESSION_BENCHMARK='1',
               FTLPU_SESSION_PROGRESS='1', FTLPU_QWEN_DECODER_LAYERS='1',
               FTLPU_QWEN_C2C_PIPELINE_CSV=str(directory / 'runtime.pipeline.csv'),
               FTLPU_QWEN_C2C_LINKED_BINARY=str(directory / 'linked.ftlpu'),
               FTLPU_QWEN_C2C_PRE_EXECUTION_DIR=str(directory / 'pre_execution_programs'))
    command = [a.build / 'runtime/compiled_qwen2_5_1_5b_decoder_layer_seq32_runtime_test.exe',
               a.results / 'native.ftlpu']
    if mode != 'native': command.append(a.results / f'{mode}.ftlpu')
    print(f'Start {mode} / {bandwidth}', flush=True)
    outcome = execute(command, directory / 'test.log', env)
    outcome.update(mode=mode, bandwidth_mbps=bandwidth, clock_mhz=500,
                   baseline_sha256=hashlib.sha256((a.results / 'native.ftlpu').read_bytes()).hexdigest(),
                   executable_sha256=hashlib.sha256((a.results / f'{mode}.ftlpu').read_bytes()).hexdigest())
    (directory / 'run.json').write_text(json.dumps(outcome, indent=2), encoding='utf-8')
    print(f'Finish {mode} / {bandwidth}: exit={outcome["exit_code"]}', flush=True)
    return outcome


def fields(line):
    return {k: float(v) if '.' in v else int(v)
            for k, v in re.findall(r'([a-zA-Z_][a-zA-Z_0-9]*)=([0-9]+(?:\.[0-9]+)?)', line)}


def report(a):
    rows = []
    for mode in LABELS:
        for bandwidth in a.bandwidth:
            d = a.results / f'{mode}_bw{bandwidth}'
            if not (d / 'test.log').exists(): continue
            log = (d / 'test.log').read_text(encoding='utf-8-sig', errors='replace')
            passed = 'Complete Qwen2.5-1.5B-seq32 decoder layer passed:' in log
            row = dict(mode=mode, bandwidth_mbps=bandwidth, passed=passed)
            for prefix, key in [('Session benchmark:', 'execution'), ('Session frontend:', 'frontend'),
                                ('Qwen dynamic C2C pages passed:', 'paging'),
                                ('Complete Qwen2.5', 'numerical')]:
                matches = [line for line in log.splitlines() if line.startswith(prefix)]
                if matches: row[key] = fields(matches[-1])
            if passed:
                trace = d / 'runtime.pipeline.csv'
                row['trace'] = inspect_trace(trace)
                row['trace_ms'] = row['trace']['end_cycle'] / 500000
                row['including_frontend_prefill_ms'] = (row['trace']['end_cycle']
                    + row.get('execution', {}).get('instruction_prefill_cycles', 0)) / 500000
                embedded_viewer(ROOT / 'tools/pipeline_viewer', trace, d / 'pipeline.html', LABELS[mode])
                outcome = execute([a.build / 'runtime/ftlpu_binary_inspect.exe', d / 'linked.ftlpu'], d / 'linked.inspect.txt')
                if outcome['exit_code']: raise RuntimeError('linked image inspection failed')
                inspection = (d / 'linked.inspect.txt').read_text(encoding='utf-8', errors='replace')
                for line in inspection.splitlines():
                    if line.startswith('binary common_barrier_storage '): row['storage'] = fields(line)
                    if line.startswith('imem model=target-physical-v1 '): row['physical_imem'] = fields(line)
                    if line.startswith('imem physical_resource='):
                        resource = re.search(r'physical_resource=(\S+)', line).group(1)
                        row.setdefault('resources', {})[resource] = fields(line)
                startup = []
                for image in sorted((d / 'pre_execution_programs').glob('*.ftlpu')):
                    inspection_path = image.with_suffix('.inspect.txt')
                    outcome = execute([a.build / 'runtime/ftlpu_binary_inspect.exe', image, '--physical-only'], inspection_path)
                    if outcome['exit_code']: raise RuntimeError('startup image inspection failed')
                    for line in inspection_path.read_text(encoding='utf-8').splitlines():
                        if line.startswith('binary common_barrier_storage '):
                            startup.append(dict(image=image.name, **fields(line)))
                row['startup_images'] = startup
                row['startup_imem_upload_bytes'] = sum(x['total_imem_bytes'] for x in startup)
                row['startup_peak_imem_bytes'] = max((x['total_imem_bytes'] for x in startup), default=0)
                # ICU issue occupancy is distinct from useful FLOP utilization.
                occupied = {}
                with trace.open(encoding='utf-8', newline='') as stream:
                    for event in csv.DictReader(stream):
                        resource = event['resource']
                        if not (resource.startswith('MXM.') and resource.endswith('.Compute')): continue
                        count = int(event.get('inner_count') or 1) * int(event.get('outer_count') or 1)
                        count -= int(event.get('skip_first') or 0)
                        occupied[resource] = occupied.get(resource, 0) + count * (int(event['end']) - int(event['start']))
                row['mxm_issue_cycles'] = occupied
            else:
                row['last_log_line'] = log.splitlines()[-1] if log.splitlines() else 'running'
            rows.append(row)
    (a.results / 'summary.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    images = {}
    for mode in LABELS:
        binary = a.results / f'{mode}.ftlpu'
        if not binary.exists(): continue
        log_path = a.results / f'{mode}.inspect.txt'
        outcome = execute([a.build / 'runtime/ftlpu_binary_inspect.exe', binary], log_path)
        if outcome['exit_code']: raise RuntimeError('compiled image inspection failed')
        images[mode] = {'sha256': hashlib.sha256(binary.read_bytes()).hexdigest()}
        for line in log_path.read_text(encoding='utf-8').splitlines():
            if line.startswith('binary common_barrier_storage '): images[mode]['storage'] = fields(line)
            if line.startswith('imem physical_resource='):
                resource = re.search(r'physical_resource=(\S+)', line).group(1)
                images[mode].setdefault('resources', {})[resource] = fields(line)
        images[mode]['largest_compute_icu_bytes'] = max(
            (v['max_used_slots'] * (16 if resource.startswith('mxm_') else 12)
             for resource, v in images[mode].get('resources', {}).items()), default=0)
    (a.results / 'compiled_images.json').write_text(json.dumps(images, indent=2), encoding='utf-8')
    body = []
    for r in rows:
        d = f'{r["mode"]}_bw{r["bandwidth_mbps"]}'
        storage, execution, paging = (r.get(x, {}) for x in ('storage', 'execution', 'paging'))
        def value(obj, key): return f'{obj[key]:,}' if key in obj else '—'
        ms = f'{r["trace_ms"]:.4f}' if 'trace_ms' in r else '—'
        body.append(f'<tr><td>{LABELS[r["mode"]]}</td><td>{r["bandwidth_mbps"]/1000:g}</td>'
                    f'<td>{"通过" if r["passed"] else "未通过/进行中"}</td><td>{ms}</td>'
                    f'<td>{value(execution,"logical_cycles")}</td><td>{value(execution,"physical_cycles")}</td>'
                    f'<td>{value(paging,"runtime_wait_cycles")}</td><td>{value(storage,"total_imem_bytes")}</td>'
                    f'<td><a href="{d}/test.log">日志</a> '
                    + (f'<a href="{d}/pipeline.html">流水图</a>' if r['passed'] else '') + '</td></tr>')
    image_rows = []
    for mode, data in images.items():
        s = data['storage']
        image_rows.append(f'<tr><td>{LABELS[mode]}</td><td>{s["compute_imem_bytes"]:,}</td>'
                          f'<td>{s["transport_native_bytes"]:,}</td><td>{data["largest_compute_icu_bytes"]:,}</td>'
                          f'<td>{s["total_file_bytes"]:,}</td></tr>')
    page = '''<!doctype html><meta charset="utf-8"><title>Qwen2.5 ICU 指令方案对比</title>
<style>body{font:16px system-ui;background:#f5f7fb;color:#172338;margin:40px;line-height:1.6}table{border-collapse:collapse;background:white;width:100%}td,th{padding:12px;text-align:left;border-bottom:1px solid #dde2ea}th{background:#e6edf7}a{color:#1763b1}p{max-width:1100px}</style>
<h1>Qwen2.5-1.5B · seq32 单层 prefill</h1>
<p>LPU32，500 MHz，W8 / BF16，完整尺寸合成权重，非零 Q/K；不是官方权重验收。DDR 效率 90%，读延迟35拍、抖动15拍；写延迟25拍、抖动10拍，种子固定。</p>
<p>Native 公共屏障与 Raw Macro 公共屏障用于同调度比较；Native 默认分页用于当前实现对比。公共屏障版本在 HOLD 中搬入动态权重页，时延差不能直接解释为指令解码速度差。</p>
<p>总时间来自实际运行流水，包含初始数据读入、动态等待、KV 写回和输出；指令前端预填另见 summary.json。iMEM 列是本次运行主程序的计算及传输指令，不含启动程序、屏障表或前端状态；JSON 另列全部启动镜像的上传字节数及峰值。算子数值通过还要求动态输出与原生基准逐字节一致。</p>
<p>不计主机编译时间、指令镜像下载时间；这不是完整 28 层模型的端到端 TTFT。</p>
<p>Native 默认版的逻辑时钟会在队列 SYNC 等待期间继续走，不能将其逻辑周期列理解为纯计算周期。公共屏障版在等待时暂停计算逻辑时钟。MXM issue 占用也不能直接当成有效 FLOP 利用率。</p>
<p>原生 3D 每个队列只有 1 个活动域；Raw Macro 当前配置为 MEM 32 / MXM 40 个 context，静态需求峰值分别是 8 / 25。两者的前端硬件复杂度不同，iMEM 缩小并不等价于整体硬件面积更小。</p>
<p>这里的 context 属于 ICU 解码与调度，与 MXM accumulator 的 4 个 context 不同。前端 DDB 等待计数是各队列周期之和，包含正常回压，不能直接换算成整层额外时延。</p>
<table><tr><th>方案</th><th>GB/s</th><th>数值</th><th>流水总 ms</th><th>计算逻辑周期</th><th>计算物理周期</th><th>运行等待周期</th><th>主程序 iMEM B</th><th>证据</th></tr>ROWS</table>
<h2>编译镜像</h2><p>公共屏障 Native / Raw 两行的计算图、布局、分页边界一致，可比较纯计算 iMEM。默认 Native 的主队列已包含队列内同步，不是纯计算口径。容器文件还包含绑定及重定位等软件元数据。</p>
<table><tr><th>方案</th><th>主计算队列 B</th><th>独立传输控制 B</th><th>最大单计算 ICU B</th><th>容器文件 B</th></tr>IMAGE_ROWS</table>
<h2>本轮修复与验证</h2><p>修复 V projection 的 bias 与结果写入早 2 拍的问题；增加独立于指令编码的公共屏障选择。原生 3D 的 SRAM 安全校验只在校验阶段归一化仿射域，实际执行仍保留 3D 指令。未放宽存储覆盖检查，也未跳过数值测试。</p>
<p>同调度比较逐 ICU、逐周期核对 FU 操作及操作数。另通过公共屏障、安全存储覆盖、VXM RUN2D 和 Raw Macro CLI 回归；详见 <a href="regression.log">回归日志</a>、<a href="metadata.json">配置与版本</a>。</p>
<p><a href="summary.json">完整测量 JSON</a> · <a href="compute_equivalence.log">同调度 FU 指令逐周期比较</a></p>'''
    (a.results / 'index.html').write_text(page.replace('IMAGE_ROWS', ''.join(image_rows)).replace('ROWS', ''.join(body)), encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build', type=Path, default=ROOT / 'build-ftlpu-vs2026')
    p.add_argument('--results', type=Path, default=ROOT / 'results/qwen2_5_prefill_instruction_comparison')
    p.add_argument('--phase', choices=['all', 'compile', 'run', 'report'], default='all')
    p.add_argument('--bandwidth', type=int, nargs='+', default=[25600, 51200])
    p.add_argument('--modes', choices=list(LABELS), nargs='+', default=list(LABELS))
    p.add_argument('--jobs', type=int, choices=[1, 2, 3], default=1)
    a = p.parse_args()
    a.build, a.results = a.build.resolve(), a.results.resolve()
    a.results.mkdir(parents=True, exist_ok=True)
    if a.phase in ('all', 'compile'):
        compile_programs(a)
        outcome = execute([a.build / 'runtime/ftlpu_compare_compute_issues.exe',
            a.results / 'native_common.ftlpu', a.results / 'raw.ftlpu'], a.results / 'compute_equivalence.log')
        if outcome['exit_code']: raise RuntimeError('common-schedule computation differs')
    if a.phase in ('all', 'run'):
        with ThreadPoolExecutor(max_workers=a.jobs) as pool:
            outcomes = list(pool.map(lambda case: run_case(a, case),
                                    [(m, b) for m in a.modes for b in a.bandwidth]))
        report(a)
        if any(r['exit_code'] for r in outcomes): raise SystemExit(1)
    elif a.phase == 'report': report(a)


if __name__ == '__main__': main()
