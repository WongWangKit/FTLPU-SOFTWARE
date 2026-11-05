"""Render the real moe_ffn_demo execution with the repository pipeline viewer.

Semantic stage intervals are joined from actual MXM issues / VXM execution /
SRAM commits. No planned schedule is substituted for an execution trace.
"""
from __future__ import annotations
import argparse
import base64
import csv
import gzip
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'ftlpu-mpl-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
from run_dense_pipelines import embedded_viewer

ROOT = Path(__file__).resolve().parents[1]
COLORS = dict(Gate='#377eb8', Up='#24a4ac', SwiGLU='#e88a36', Down='#8561aa')


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def expanded(rows, resource):
    result = []
    for r in rows:
        if r['resource'] != resource:
            continue
        for outer in range(int(r.get('outer_count') or 1)):
            for inner in range(int(r.get('inner_count') or 1)):
                delta = outer * int(r.get('outer_interval') or 0) + inner * int(r.get('inner_interval') or 0)
                result.append((int(r['start'])+delta, int(r['end'])+delta, r['detail']))
    return sorted(result)


def groups(events):
    result = []
    for a, b, detail in sorted(events):
        if result and a <= result[-1][1]:
            result[-1][1] = max(result[-1][1], b)
        else:
            result.append([a, b])
    return result


def event(start, end, resource, detail):
    assert end > start
    return dict(start=start, end=end, resource=resource, detail=detail,
                pattern='single', inner_count=1, inner_interval=0, inner_stride=0,
                outer_count=1, outer_interval=0, outer_stride=0, skip_first=0,
                induction='none', base_delta=0)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output', type=Path)
    a=p.parse_args(); out=a.output.resolve()
    report=json.loads((out/'summary.json').read_text())
    assert report['status']=='passed' and report['bit_mismatches']==0
    rows=read_csv(out/'runtime.pipeline.csv')
    tasks=[{k:int(v) for k,v in r.items()} for r in read_csv(out/'expert_tasks.csv')]
    ops=read_csv(out/'rcu_operations.csv')
    headers=list(rows[0]); commits=[]
    # Keep actual transfers and issues for an immediately usable, sparse MEM
    # view. The full raw trace remains available beside this filtered file.
    with (out/'runtime.mem.csv').open(encoding='utf-8',newline='') as src, (out/'runtime.mem.active.csv').open('w',encoding='utf-8',newline='') as dst:
        reader=csv.DictReader(src); writer=csv.DictWriter(dst,fieldnames=reader.fieldnames);writer.writeheader()
        for r in reader:
            if r['stage']!='icu' or r['action'] in ('issue','macro_issue','repeat_issue','repeat2d_issue','sync_issue'):
                writer.writerow(r)
            if r['stage']=='sram' and r['action']=='write_commit':
                commits.append(dict(cycle=int(r['cycle']), side=r['hemisphere'],
                                    slice=int(r['slice']), bank=int(r['bank']), address=int(r['address'])))
    compute=expanded(rows,'MXM.E0.Compute')
    heads=expanded(rows,'VXM.Datapath.C0')
    semantic=[]; stages=[]
    gathers=[r for r in ops if r['opcode']=='ROUTE_ADDR']
    assert len(gathers)==len(tasks)
    for task,gather in zip(tasks,gathers):
        lo,hi=task['start'],task['end']
        launches=groups([e for e in compute if lo<=e[0]<hi and e[2].startswith('Compute ')])
        assert len(launches)==3, (task,launches)
        local=[c for c in commits if lo<=c['cycle']<hi]
        gate=[c['cycle'] for c in local if c['side']=='E' and c['slice'] in (40,41) and 200<=c['address']<200+task['rows']]
        hidden=[c['cycle'] for c in local if c['side']=='E' and c['slice'] in (48,49) and 220<=c['address']<220+task['rows']]
        output=[c['cycle'] for c in local if c['side']=='W' and c['slice'] in (0,1) and task['assignment_base']<=c['address']<task['assignment_base']+task['rows']]
        assert len(gate)==len(hidden)==len(output)==task['rows']*8
        head=min(e[0] for e in heads if launches[1][0]<=e[0]<launches[2][0])
        for name,start,end in [('Gate',launches[0][0],max(gate)+1),
                               ('Up',launches[1][0],head),
                               ('SwiGLU',head,max(hidden)+1),
                               ('Down',launches[2][0],max(output)+1)]:
            entry=dict(task=task['task'], expert=task['expert'], rows=task['rows'], stage=name, start=start,end=end)
            stages.append(entry)
            semantic.append(event(start,end,'MoE.FFN.'+name,
                f"expert={task['expert']} task={task['task']} rows={task['rows']} observed issues through commit; source=runtime"))
        if task['row_offset']==0:
            weights=[c['cycle'] for c in commits if int(gather['end'])<=c['cycle']<lo and c['bank']==1
                     and ((c['side']=='E' and c['slice']<16 and 64<=c['address']<76)
                          or (c['side']=='W' and c['slice'] in (4,5,6,7) and c['address']==64))]
            assert len(weights)==196*4,(task,len(weights))
            semantic.append(event(min(weights),max(weights)+1,'MoE.WeightsToSRAM',
                f"expert={task['expert']} vectors=196 actual first-to-last SRAM weight commit; source=runtime"))
    router_compute=expanded(rows,'MXM.W0.Compute')
    router_writes=[c['cycle'] for c in commits if c['side']=='W' and c['slice'] in (40,41,42,43) and 500<=c['address']<505]
    assert len(router_writes)==5*4*4
    semantic.append(event(min(e[0] for e in router_compute if e[2].startswith('Compute ')),max(router_writes)+1,
                          'MoE.Router.Projection','West MXM X @ W_router -> FP32 logits SRAM; source=runtime'))
    rows.extend(semantic); rows.sort(key=lambda r:(int(r['start']),r['resource']))
    with (out/'pipeline.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=headers);writer.writeheader();writer.writerows(rows)
    (out/'stages.json').write_text(json.dumps(stages,indent=2),encoding='utf-8')
    embedded_viewer(ROOT/'tools/pipeline_viewer',out/'pipeline.csv',out/'pipeline.html','小型 SwiGLU MoE · 实际执行')
    page=(out/'pipeline.html').read_text(encoding='utf-8')
    page=page.replace('实际运行 · 包含最初数据读入','实际执行 · 含程序与专家权重 DMA；输入预置')
    page=page.replace('await loadFile(new File([blob], TRACE_NAME, {type: "text/csv"}));',
                      'await loadFile(new File([blob], TRACE_NAME, {type: "text/csv"}));\n  dom.searchInput.value="MoE"; state.query="MoE"; applyFilters(); fitAll();')
    page=page.replace('function eventColor(event) {','''function eventColor(event) {
  if(event.resource.startsWith("MoE.")) {
    if(event.resource.endsWith("Gate")) return "#377eb8";
    if(event.resource.endsWith("Up")) return "#24a4ac";
    if(event.resource.endsWith("SwiGLU")) return "#e88a36";
    if(event.resource.endsWith("Down")) return "#8561aa";
    if(event.resource.includes("Weight")) return "#caa052";
    if(event.resource.includes("Gather")) return "#469b80";
    if(event.resource.includes("Combine")) return "#be5984";
    return "#738ea8";
  }''')
    (out/'pipeline.html').write_text(page,encoding='utf-8')
    # Embed the active-only real MEM trace in the existing MEM viewer too.
    viewer=ROOT/'tools/pipeline_viewer'
    mem=(viewer/'mem.html').read_text(encoding='utf-8')
    for css in ('style.css','mem_style.css'):
        mem=mem.replace(f'<link rel="stylesheet" href="{css}">','<style>'+(viewer/css).read_text(encoding='utf-8')+'</style>')
    payload=base64.b64encode(gzip.compress((out/'runtime.mem.active.csv').read_bytes(),mtime=0)).decode()
    loader='''
(async()=>{const packed=Uint8Array.from(atob(PAYLOAD),c=>c.charCodeAt(0));
const blob=await new Response(new Blob([packed]).stream().pipeThrough(new DecompressionStream("gzip"))).blob();
loadFile(new File([blob],"MoE / actual MEM activity.csv",{type:"text/csv"}));})().catch(e=>els.interactionState.textContent=e.message);
'''.replace('PAYLOAD',json.dumps(payload))
    mem=mem.replace('<script src="mem_viewer.js"></script>','<script>'+((viewer/'mem_viewer.js').read_text(encoding='utf-8')+loader).replace('</script','<\\/script')+'</script>')
    mem=mem.replace('href="index.html"','href="pipeline.html"')
    (out/'mem.html').write_text(mem,encoding='utf-8')
    # Publication/export preview uses ordinary plotting, with actual cycles.
    font=Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus']=False
    labels=['程序 DMA / 前端准备','Router 投影','Top-2 / PLAN','专家输入 Gather','专家权重落 SRAM','专家 0','专家 1','专家 2','专家 3','加权回并']
    fig,ax=plt.subplots(figsize=(15.6,6.2));fig.patch.set_facecolor('#f7f9fb');ax.set_facecolor('#ffffff')
    semantic_rows={'MoE.ProgramLoad':0,'MoE.Router.Projection':1,'MoE.Router.ROUTE_SELECT':2,'MoE.Router.ROUTE_PLAN':2,'MoE.InputGather':3,'MoE.WeightsToSRAM':4,'MoE.Combine':9}
    for resource,y in semantic_rows.items():
        for a0,b0,detail in expanded(rows,resource):
            color=['#7891ae','#3a7b9e','#4b9b86','#4b9b86','#caa052'][min(y,4)] if y<5 else '#be5984'
            ax.broken_barh([(a0,b0-a0)],(y-.32,.64),facecolors=color)
    for task in tasks:
        y=5+task['expert'];ax.broken_barh([(task['start'],task['end']-task['start'])],(y-.34,.68),facecolors='#e4eaf0')
        ax.text((task['start']+task['end'])/2,y-.48,f"{task['rows']} 行",ha='center',fontsize=8,color='#354656')
    for s in stages:ax.broken_barh([(s['start'],s['end']-s['start'])],(5+s['expert']-.30,.60),facecolors=COLORS[s['stage']])
    ax.set_yticks(range(len(labels)),labels);ax.invert_yaxis();ax.set_xlim(0,report['physical_cycles']);ax.set_xlabel('实际物理 cycle（包含程序装载）')
    ax.set_title('设备端 MoE 全流程 · 5 token / 4 experts / Top-2 / SwiGLU 32→32→32',loc='left',fontsize=15,pad=18)
    ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
    ax.legend(handles=[Patch(color=c,label=n) for n,c in COLORS.items()],loc='lower right',ncol=4,frameon=False)
    fig.text(.015,.015,f"{report['physical_cycles']:,} cycles  ·  6 专家输入块  ·  160 个 BF16 输出逐位一致  ·  路由权重 = 选中正分数（无归一化）",fontsize=10,color='#465564')
    fig.tight_layout(rect=(0,.04,1,1));fig.savefig(out/'pipeline_overview.png',dpi=150);fig.savefig(out/'pipeline_overview.svg');plt.close(fig)
    task=tasks[0];origin=task['start'];fig,ax=plt.subplots(figsize=(15.6,7.0));fig.patch.set_facecolor('#f7f9fb')
    zoom_labels=['MXM 权重 IW','Gate：MXM→BF16','Up：MXM→VXM','SwiGLU→BF16','Down：MXM→输出']+[f'VXM C{s} 实际运算' for s in range(8)]
    for start,end,detail in expanded(rows,'MXM.E0.Load'):
        if origin<=start<task['end']:ax.broken_barh([(start-origin,end-start)],(-.3,.6),facecolors='#83a9c9')
    for s in stages:
        if s['task']==task['task']:ax.broken_barh([(s['start']-origin,s['end']-s['start'])],(1+list(COLORS).index(s['stage'])-.32,.64),facecolors=COLORS[s['stage']])
    for stage in range(8):
        for start,end,detail in expanded(rows,f'VXM.Datapath.C{stage}'):
            if origin<=start<task['end']:ax.broken_barh([(start-origin,end-start)],(5+stage-.28,.56),facecolors='#d78055')
    ax.set_yticks(range(len(zoom_labels)),zoom_labels);ax.invert_yaxis();ax.set_xlim(0,task['end']-origin+5)
    ax.set_title(f"专家 {task['expert']} · task {task['task']} · {task['rows']} 行 · 实际流水放大",loc='left',fontsize=15,pad=16)
    ax.set_xlabel(f"任务局部 cycle（共同起点 C0 = 物理 cycle {origin}）");ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
    fig.tight_layout();fig.savefig(out/'expert_pipeline.png',dpi=150);plt.close(fig)
    print(json.dumps(dict(status='rendered',html=str(out/'pipeline.html'),phases=len(stages),tasks=len(tasks),mem_commits=len(commits)),ensure_ascii=False))


if __name__=='__main__':main()
