"""Package physical-cycle instruction records and verify route-state replay."""
from __future__ import annotations
import argparse
import base64
from collections import Counter
import csv
import gzip
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

ROOT=Path(__file__).resolve().parents[1]
VIEWER=ROOT/'tools/pipeline_viewer'


def read(path):
    with path.open(encoding='utf-8',newline='') as f:
        return list(csv.DictReader(f))


def pack_events(data):
    """Dictionary-pack repeated strings; decode with plain JavaScript only."""
    strings=[];string_ids={};payloads=[];payload_ids={};events=[]
    def intern(value):
        if value not in string_ids:
            string_ids[value]=len(strings);strings.append(value)
        return string_ids[value]
    for event in data['events']:
        payload=tuple(v for k,value in sorted(event[7].items()) for v in (intern(k),intern(value)))
        if payload not in payload_ids:
            payload_ids[payload]=len(payloads);payloads.append(payload)
        events.append([event[0],*(intern(v) for v in event[1:7]),payload_ids[payload]])
    return dict(meta={k:v for k,v in data.items() if k!='events'},strings=strings,payloads=payloads,events=events)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path)
    out=p.parse_args().output.resolve();summary=json.loads((out/'summary.json').read_text())
    rows=read(out/'instructions.cycles.csv')
    events=[[int(r['cycle']),r['unit'],r['resource'],r['stage'],r['opcode'],r['pc'],r['detail'],json.loads(r['payload'])] for r in rows]
    events.sort(key=lambda e:(e[0],e[1],e[2],e[3]))
    assert events and min(e[0] for e in events)>=0 and max(e[0] for e in events)<summary['physical_cycles']
    steps=Counter(e[6] for e in events if e[2]=='RCU.Core' and e[3]=='internal')
    assert steps['ChoiceRead']==20 and steps['Compare']==40 and steps['StoreId']==10 and steps['StoreRaw']==10
    assert steps['CountWrite']==steps['AssignInverse']==10
    weight_dma=[e for e in events if e[1]=='DMA' and e[2] in ('DMA.E','DMA.W')]
    assert sum(e[3]=='activate' for e in weight_dma)==80
    assert sum(e[3]=='complete' for e in weight_dma)==80
    assert sum(e[3]=='data_return' for e in weight_dma)==784
    assert sum(e[3]=='control_read' for e in events)==80
    assert sum(e[1]=='VXM' and e[3]=='controller_issue' for e in events)==20
    assert sum(e[1]=='RCU' and e[3]=='issue' for e in events)==4
    # Semantic intervals use the same observed issue/commit clock as the
    # detailed events. Keep them above the narrow per-cycle inspection window.
    phases=[]
    phase_resources={
        'MoE.ProgramLoad':(0,'程序 / 描述符 DMA','#7891ae'),
        'MoE.Router.Projection':(1,'Router MXM','#3a7b9e'),
        'MoE.Router.ROUTE_SELECT':(2,'Top-K','#4b9b86'),
        'MoE.Router.ROUTE_PLAN':(2,'PLAN','#348674'),
        'MoE.WeightsToSRAM':(4,'权重落 SRAM','#caa052'),
        'MoE.Combine':(9,'恢复 token 顺序 / 加权求和','#be5984')}
    for row in read(out/'pipeline.csv'):
        if row['resource'] not in phase_resources:continue
        lane,label,color=phase_resources[row['resource']]
        phases.append(dict(start=int(row['start']),end=int(row['end']),row=lane,label=label,color=color))
    gathers=[r for r in read(out/'rcu_operations.csv') if r['opcode']=='ROUTE_ADDR']
    tasks=read(out/'expert_tasks.csv');assert len(gathers)==len(tasks)==6
    for gather,task in zip(gathers,tasks):
        phases.append(dict(start=int(gather['start']),end=int(gather['end']),row=3,
                           label=f"E{task['expert']} 输入 {task['rows']} 行",color='#469b80'))
    colors=dict(Gate='#377eb8',Up='#24a4ac',SwiGLU='#e88a36',Down='#8561aa')
    for stage in json.loads((out/'stages.json').read_text(encoding='utf-8')):
        phases.append(dict(start=stage['start'],end=stage['end'],row=5+stage['expert'],
                           label=f"E{stage['expert']} task{stage['task']} {stage['stage']} {stage['rows']}行",color=colors[stage['stage']]))
    phases.sort(key=lambda p:(p['start'],p['row']))
    assert {p['row'] for p in phases}==set(range(10))
    assert all(0<=p['start']<p['end']<=summary['physical_cycles'] for p in phases)
    data=dict(events=events,phases=phases,**{k:summary[k] for k in ('tokens','experts','top_k','physical_cycles')})
    packed_data=pack_events(data)
    payload=json.dumps(data,separators=(',',':'),ensure_ascii=False).encode('utf-8')
    (out/'instructions.cycles.json.gz').write_bytes(gzip.compress(payload,mtime=0))
    checks=dict(events=len(events),units=dict(Counter(e[1] for e in events)),rcu_steps=dict(steps),weight_dma_commands=80,weight_dma_vectors=784,physical_cycles=summary['physical_cycles'])
    # Test the actual browser data model (including state before scores exist).
    data_path=out/'instructions.cycles.json';data_path.write_bytes(payload)
    script=r'''
const fs=require('fs'),assert=require('assert');
const api=require(process.argv[1]);const data=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));
const model=api.prepare(data),final=model.routeAt(model.count-1);
assert.deepStrictEqual(final.counts,[3,2,2,3]);
assert.deepStrictEqual(final.offsets,[0,3,5,7]);
const expected=[[0,2],[1,3],[0,3],[1,2],[0,3]];
assert.deepStrictEqual(final.selected.map(row=>row.map(x=>x.expert)),expected);
assert.deepStrictEqual(final.inverse,[[0,5],[3,7],[1,8],[4,6],[2,9]]);
const first=model.events.find(e=>e.resource==='RCU.Core' && e.detail==='ChoiceRead');
assert(model.routeAt(first.cycle-1).scores.every(row=>row.every(x=>x===null)));
assert.strictEqual(model.routeAt(first.cycle).scores[0][0],0.5625);
assert(model.at(model.jumps.select).some(e=>e.detail==='Compare'));
assert(model.at(model.jumps.router).some(e=>e.unit==='MXM' && e.opcode==='COMPUTE'));
assert(model.at(model.jumps.weights).some(e=>e.unit==='DMA' && e.stage==='activate'));
    assert(model.at(model.jumps.plan).some(e=>e.resource==='RCU.Core' && e.opcode==='ROUTE_PLAN'));
    assert(model.at(model.jumps.input).some(e=>e.resource==='RCU.Core' && e.detail==='AddressExpert'));
assert(model.at(model.jumps.combine).some(e=>e.resource==='RCU.Combine'));
    assert.strictEqual(model.data.phases.length,39);
    assert.deepStrictEqual([...new Set(model.data.phases.map(p=>p.row))].sort((a,b)=>a-b),[0,1,2,3,4,5,6,7,8,9]);
const firstCompare=model.events.find(e=>e.resource==='RCU.Core'&&e.detail==='Compare');
assert.strictEqual(model.routeAt(firstCompare.cycle+1).compare.incomingValid,false);
// Exercise the real UI's load, jump, filter and drawing handlers without
// claiming a browser screenshot or substituting a schedule for trace data.
const vm=require('vm'),elements={},buttons={};
function element(id){return elements[id]??=( {value:id==='unit'?'ALL':id==='span'?'128':'',checked:false,innerHTML:'',textContent:'',clientWidth:1200,
 querySelectorAll:()=>[],getBoundingClientRect:()=>({left:0,width:1200}),
 getContext:()=>({scale(){},clearRect(){},fillText(){},beginPath(){},moveTo(){},lineTo(){},stroke(){},fillRect(){}})} );}
const page={MoeCycleModel:api,console,setInterval:()=>1,clearInterval(){},
 document:{getElementById:element,addEventListener(){},querySelectorAll:()=>Object.keys(model.jumps).map(k=>buttons[k]??={dataset:{jump:k}})},
 window:{devicePixelRatio:1,addEventListener(){}}};
vm.createContext(page);vm.runInContext(fs.readFileSync(process.argv[3],'utf8'),page);
page.window.loadMoeCycles(data);assert.strictEqual(Number(elements.cycle.value),model.jumps.router);
buttons.select.onclick();assert(elements.events.innerHTML.includes('ROUTE_SELECT'));
buttons.router.onclick();assert.strictEqual(Number(elements.cycle.value),model.jumps.router);assert(elements.events.innerHTML.includes('COMPUTE'));
buttons.input.onclick();assert.strictEqual(Number(elements.cycle.value),model.jumps.input);assert(elements.events.innerHTML.includes('ROUTE_ADDR'));
elements.overview.onclick({clientX:130+(1200-142)*.7});assert(Math.abs(Number(elements.cycle.value)-Math.floor(model.count*.7))<=1);
buttons.weights.onclick();elements.unit.value='DMA';elements.issues.checked=true;elements.issues.onchange();assert(elements.events.innerHTML.includes('DDR_TO_C2C'));
console.log(JSON.stringify({status:'replay passed',jumps:model.jumps,final:final.selected}));
'''
    subprocess.run(['node','-e',script,str(VIEWER/'cycle_model.js'),str(data_path),str(VIEWER/'cycle_app.js')],check=True)
    html=(VIEWER/'cycles.html').read_text(encoding='utf-8')
    html=html.replace('href="index.html"','href="pipeline.html"')
    html=html.replace('<link rel="stylesheet" href="cycle_style.css">','<style>'+(VIEWER/'cycle_style.css').read_text(encoding='utf-8')+'</style>')
    for name in ('cycle_model.js','cycle_app.js'):
        html=html.replace(f'<script src="{name}"></script>','<script>'+(VIEWER/name).read_text(encoding='utf-8').replace('</script','<\\/script')+'</script>')
    packed_json=json.dumps(packed_data,separators=(',',':'),ensure_ascii=False).replace('<','\\u003c')
    loader='''<script type="application/json" id="cycleData">PAYLOAD</script><script>
try {
  document.getElementById('loadStatus').textContent='正在读取内嵌周期数据…';
  window.loadMoeCycles(window.MoeCycleModel.unpack(JSON.parse(document.getElementById('cycleData').textContent)));
} catch(error) {
  document.getElementById('loadStatus').textContent='交互视图加载失败：'+error.message+'；静态总览仍可查看。';
  console.error(error);
}
</script>'''.replace('PAYLOAD',packed_json)
    static='<details open style="margin:12px 0"><summary>全流程静态预览（交互就绪后可折叠）</summary><img alt="完整 MoE 流程：程序装载、Router、Top-K、专家输入、权重、FFN、回并" style="width:100%;height:auto" src="data:image/png;base64,'+base64.b64encode((out/'pipeline_overview.png').read_bytes()).decode('ascii')+'"></details>'
    html=html.replace('<!--STATIC_OVERVIEW-->',static).replace('</body>',loader+'</body>')
    (out/'cycles.html').write_text(html,encoding='utf-8')
    (out/'moe_full_cycles.html').write_text(html,encoding='utf-8')
    # Run every executable inline script from the final HTML in one global
    # environment, including the real embedded-data loader and its status.
    # A DOM stub verifies packaging, not an actual browser screenshot.
    packaged_test=r'''
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const html=fs.readFileSync(process.argv[1],'utf8'),original=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));
const scripts=[...html.matchAll(/<script([^>]*)>([\s\S]*?)<\/script>/g)];
const dataScript=scripts.find(s=>s[1].includes('application/json'));assert(dataScript);
const elements={};
function element(id){return elements[id]??={value:id==='unit'?'ALL':id==='span'?'128':'',checked:false,innerHTML:'',textContent:id==='cycleData'?dataScript[2]:'',clientWidth:1200,
 querySelectorAll:()=>[],getBoundingClientRect:()=>({left:0,top:0,width:1200}),getContext:()=>({scale(){},clearRect(){},fillText(){},beginPath(){},moveTo(){},lineTo(){},stroke(){},fillRect(){}})};}
const context={console,setInterval:()=>1,clearInterval(){},addEventListener(){},document:{getElementById:element,addEventListener(){},querySelectorAll:()=>[]}};
context.window=context;vm.createContext(context);
for(const script of scripts)if(!script[1].includes('application/json'))vm.runInContext(script[2],context);
assert(elements.loadStatus.textContent.includes('已就绪'));
assert.strictEqual(Number(elements.cycle.value),5043);assert(elements.events.innerHTML.includes('COMPUTE'));
const decoded=context.MoeCycleModel.unpack(JSON.parse(dataScript[2]));
assert.deepStrictEqual(JSON.parse(JSON.stringify(decoded)),original);
assert(!html.includes('DecompressionStream'));assert(!html.includes('<script src='));
console.log('Final HTML loader passed: complete trace preserved; no fetch or decompression API required.');
'''
    subprocess.run(['node','-e',packaged_test,str(out/'cycles.html'),str(data_path)],check=True)
    # Keep the original pipeline as the overview. Add the missing actual RCU
    # comparison/plan operations and indirect DMA work; all other detail stays
    # directly accessible in the physical-cycle inspector.
    overview=read(out/'pipeline.csv');headers=list(overview[0])
    for e in events:
        selected=(e[2]=='RCU.Core' and e[6] in ('ChoiceRead','Compare','StoreId','CountWrite','AssignInverse')) or (e[1]=='DMA' and e[3] in ('activate','ddr_read_request','data_return','complete'))
        if not selected:continue
        resource=('RCU.'+e[4]+'.'+e[6]) if e[1]=='RCU' else e[2]+'.'+e[3]
        detail=f"{e[4]} {e[6]} {json.dumps(e[7],separators=(',',':'))} source=runtime"
        overview.append(dict(start=e[0],end=e[0]+1,resource=resource,detail=detail,pattern='single',inner_count=1,inner_interval=0,inner_stride=0,outer_count=1,outer_interval=0,outer_stride=0,skip_first=0,induction='none',base_delta=0))
    overview.sort(key=lambda e:(int(e['start']),e['resource']))
    with (out/'pipeline.instructions.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=headers);writer.writeheader();writer.writerows(overview)
    page=(out/'pipeline.html').read_text(encoding='utf-8')
    encoded=base64.b64encode(gzip.compress((out/'pipeline.instructions.csv').read_bytes(),mtime=0)).decode('ascii')
    page,n=re.subn(r'atob\("[A-Za-z0-9+/=]+"\)',f'atob("{encoded}")',page,count=1);assert n==1
    if 'href="cycles.html"' not in page:
        page=page.replace('<a class="viewer-link" href="mem.html">MEM Cycle Viewer</a>', '<a class="viewer-link" href="cycles.html">逐周期 / 指令级</a><a class="viewer-link" href="mem.html">MEM Cycle Viewer</a>')
    page=page.replace('<option value="RCU">RCU</option>','').replace('<option value="DMA">DMA</option>','')
    page=page.replace('<option value="SXM">SXM</option>','<option value="SXM">SXM</option><option value="RCU">RCU</option><option value="DMA">DMA</option>')
    extra_colors='\n  if(event.resource.startsWith("RCU."))return "#668bba";\n  if(event.resource.startsWith("DMA."))return "#bd9848";'
    page=page.replace(extra_colors,'').replace('function eventColor(event) {','function eventColor(event) {'+extra_colors)
    (out/'pipeline.html').write_text(page,encoding='utf-8')
    (out/'cycle_trace_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    # Compact static companion: the actual first-token decision interval and
    # the first expert's instruction/configuration pipeline.
    os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'ftlpu-mpl-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font=Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus']=False
    compare=[e for e in events if e[2]=='RCU.Core' and e[6]=='Compare' and e[7]['token']=='0']
    lo,hi=compare[0][0]-15,compare[-1][0]+15
    figure,axes=plt.subplots(2,1,figsize=(15,7),gridspec_kw={'height_ratios':[1.3,1]},layout='constrained')
    families=['RCU','MXM','VXM','DMA','MEM'];colors={'RCU':'#668bba','MXM':'#377eb8','VXM':'#dd8754','DMA':'#bd9848','MEM':'#499b81'}
    for j,unit in enumerate(families):
        points=sorted({e[0] for e in events if lo<=e[0]<hi and e[1]==unit})
        axes[0].broken_barh([(v,1) for v in points],(j-.32,.64),facecolors=colors[unit])
    axes[0].set_yticks(range(5),families);axes[0].invert_yaxis();axes[0].set_xlim(lo,hi);axes[0].set_title('token 0 · Top-2 实际逐周期活动（包括 RCU 控制步骤）',loc='left');axes[0].grid(axis='x',alpha=.2)
    incoming_valid=True
    for e in compare:
        pp=e[7];slot=int(pp['slot'])
        if slot==0:incoming_valid=True
        axes[1].scatter([e[0]],[slot],color='#dc8752' if pp['exchanged']=='1' else '#668bba',s=50)
        label=f"E{pp['incoming_id']} {'交换' if pp['exchanged']=='1' else '保留'}" if incoming_valid else '无有效输入'
        axes[1].annotate(label,(e[0],slot),xytext=(0,12),textcoords='offset points',ha='center',fontsize=9)
        if pp['exchanged']=='1':incoming_valid=pp['candidate_valid']=='1'
    axes[1].set_yticks([0,1],['Top-K slot 0','Top-K slot 1']);axes[1].set_ylim(-.4,1.5);axes[1].set_xlim(lo,hi);axes[1].set_xlabel('实际物理 cycle');axes[1].grid(axis='x',alpha=.2)
    figure.savefig(out/'routing_cycles.png',dpi=150);plt.close(figure)
    print(json.dumps(dict(status='passed',html=str(out/'cycles.html'),**checks),ensure_ascii=False))


if __name__=='__main__':main()
