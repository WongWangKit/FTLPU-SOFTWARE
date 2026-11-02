"""Bundle a standalone operator explorer; read existing traces, never lower/run ops."""
from pathlib import Path
import csv
import json
import re

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

def build():
    td = ROOT / 'compiler/include/ftlpu/compiler/Dialect/Kernel/IR/KernelOps.td'
    ops = re.findall(r'def \w+Op : FTLPU_Kernel_Op<"([^"]+)"', td.read_text())
    sources = [
        ('rms_norm', 'current_vxm/native_norm/rms_128', 'VXM'),
        ('l2_norm', 'current_vxm/native_norm/l2_32', 'VXM'),
        ('depthwise_conv1d', 'conv_chain_mirrored_test', 'VXM'),
        ('swish', 'chain_gates_test', 'VXM'),
        ('matmul', 'distributed_projection_staging_test', 'MXM'),
        ('gated_delta_rule', 'gdn_mxm_chunk32', 'MXM'),
    ]
    traces = []
    for op, name, unit in sources:
        p = ROOT / 'results' / name / 'runtime.pipeline.csv'
        if not p.exists():
            continue
        with p.open(encoding='utf-8-sig', newline='') as f:
            rows = list(csv.DictReader(f))
        pivot = min(int(r['start']) for r in rows if r['resource'].startswith(unit))
        lo, hi = max(0, pivot-24), pivot+256
        # Keep source patterns intersecting this window; JS materializes precisely.
        kept = []
        for r in rows:
            end = int(r['end']) + (int(r.get('inner_count') or 1)-1)*int(r.get('inner_interval') or 0) + (int(r.get('outer_count') or 1)-1)*int(r.get('outer_interval') or 0)
            if end > lo and int(r['start']) < hi:
                kept.append(r)
        traces.append(dict(op=op, name=name, source=str(p.relative_to(ROOT)).replace('\\','/'), start=lo, end=hi, rows=kept, mtime=p.stat().st_mtime))
    config = json.loads((ROOT.parent/'FTLPU-CMODEL/config/ftlpu-lpu32.json').read_text())
    data = dict(ops=ops, traces=traces, hardware=config)
    html = (HERE/'template.html').read_text(encoding='utf-8')
    html = html.replace('/*__DATA__*/', 'const DATA='+json.dumps(data, ensure_ascii=False).replace('</','<\\/')+';')
    html = html.replace('/*__APP__*/', '\n'.join((HERE/name).read_text(encoding='utf-8') for name in ['app.js','physical.js','sxm_variants.js']))
    target = ROOT.parent/'lpu-operator-explorer.html'
    target.write_text(html, encoding='utf-8')
    print(f'{target}: {len(ops)} Kernel IR ops, {len(traces)} recorded trace windows, {target.stat().st_size} bytes')

if __name__ == '__main__':
    build()
