'use strict';
const $=id=>document.getElementById(id), esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const input=Array.from({length:4},(_,t)=>Array.from({length:32},(_,c)=>Math.sin((c+1)*.31+t*.7)));
const map=(a,f)=>a.map((r,t)=>r.map((v,c)=>f(v,c,t))), clone=a=>a.map(r=>r.slice()), sum=a=>a.reduce((x,y)=>x+y,0), sig=x=>1/(1+Math.exp(-x));
const S=(label,unit,formula,lat=1,apply=null)=>({label,unit,formula,lat,apply});
const unary=(label,formula,fn,lat=1)=>S(label,'VXM',formula,lat,a=>map(a,fn));
const recipes={};
function add(id,name,formula,steps,note='',extra={}){recipes[id]={id,name,formula,steps,note,...extra};}
const square=()=>unary('平方','u = x × x',x=>x*x,2);
const reduce=(name,kind)=>S(name,'VXM',kind==='max'?'m = max_j x[j]':'s = Σ_j x[j]',5,a=>a.map(r=>Array(32).fill(kind==='max'?Math.max(...r):sum(r))));
const projection=(name='投影',source=input)=>S(name,'MXM','Y[t,n] = Σ_k X[t,k] W[k,n]',8,a=>a.map(row=>Array.from({length:32},(_,n)=>sum(row.map((v,k)=>v*Math.cos((k+1)*(n+1)*.07)/32)))));
add('matmul','Projection / Matmul','Y = XW',[S('W 进入权重 buffer','MXM','MEM → SR → MXM weight buffer',4),projection(),S('ACC 输出 / 反量化','MXM','acc × scale → 输出布局',2)],'这里用 32×32 稠密矩阵说明点积；实际 m/n/k 分块、量化及 ACC 排空以 trace 为准。');
add('batch_matmul','Batch Matmul','Y[b] = A[b] B[b]',[projection('分 batch 矩阵乘'),S('各 batch 输出','MXM','batch 的累加状态分别维护',2)],'教学例子展示一个 batch；其他 batch 独立，权重是否复用取决于 rhs。');
add('bias_add','Bias Add','Y[t,c] = X[t,c] + bias[c]',[unary('广播 bias 并相加','x + bias[c]',(x,c)=>x+(c-16)/32)]);
add('elementwise','Residual / Elementwise','Y = X + R',[unary('读取双输入并相加','x + residual',x=>x+.25)],'演示 kind=add；具体 kind 由 IR 属性选择。');
add('vector_alu','Vector ALU','Y = X × scalar',[unary('标量广播与乘法','x × 0.5',x=>x*.5,2)],'演示 broadcast multiply；也可配置逐元素算术、归约。');
add('rms_norm','RMSNorm','Y = X · rsqrt(mean(X²) + ε) · γ',[square(),reduce('平方和归约','sum'),unary('除以 D 并加 ε','s / 32 + 1e−5',x=>x/32+1e-5),unary('倒数平方根','r = rsqrt(s/32+ε)',x=>1/Math.sqrt(x),5),S('重读 X 并乘 r','VXM','x × r',2,a=>map(a,(x,c,t)=>input[t][c]*x)),unary('乘 γ','y × γ[c]',(x,c)=>x*(1+c/64),2)],'归约跨 lane 的路由/树仅以阶段表示；不能把该阶段直接等同于单个 ALU op。');
add('l2_norm','L2Norm','Y = scale · X · rsqrt(Σ X² + ε)',[square(),reduce('平方和归约','sum'),unary('加 ε','s + 1e−5',x=>x+1e-5),unary('倒数平方根','r = rsqrt(s+ε)',x=>1/Math.sqrt(x),5),S('乘回 X','VXM','y = x × r (scale=1)',2,a=>map(a,(x,c,t)=>input[t][c]*x))],'与 RMSNorm 不同：不除以 D、不乘 γ。');
add('softmax','Softmax','Y = exp(X − max X) / Σ exp(X − max X)',[reduce('行最大值','max'),S('减去最大值','VXM','u = x − max(x)',1,a=>map(a,(m,c,t)=>input[t][c]-m)),unary('指数','e = exp(u)',Math.exp,5),reduce('指数和','sum'),unary('取倒数','r = 1 / Σe',x=>1/x,5),S('重读 e 并归一化','VXM','y = e × r',2,a=>map(a,(v,c,t)=>Math.exp(input[t][c]-Math.max(...input[t]))*v))],'示例不加 causal mask，scale=1；实际 attention 配置会包含缩放和 mask。');
function sigmoidSteps(){return [unary('取负','u = −x',x=>-x),unary('指数','e = exp(u)',Math.exp,5),unary('加一','d = 1 + e',x=>x+1),unary('倒数','s = 1 / d',x=>1/x,5)];}
add('swish','SiLU / Swish','Y = X · sigmoid(X)',[...sigmoidSteps(),S('乘保留的 X','VXM','y = x × sigmoid(x)',2,a=>map(a,(s,c,t)=>s*input[t][c]))],'原始 X 必须保留或重读；不能在 exp 链中消耗后丢失。');
add('sigmoid_multiply','Sigmoid Multiply','Y = input · sigmoid(gate)',[...sigmoidSteps(),unary('乘另一路 input','input=0.5；y=0.5×sigmoid(gate)',x=>.5*x,2)]);
add('swiglu','SwiGLU','gate=XWg; up=XWu; Y=SiLU(gate)×up',[projection('Gate projection'),projection('Up projection（独立读取 X）'),S('Gate 激活 × Up','VXM','SiLU(XWg) × XWu',8,()=>{let gate=projection().apply(input);return map(gate,x=>x*sig(x)*(x*.75));})],'数值例子 Wu=0.75 Wg；第二路 projection 独立读取 X。量化输出步骤在此省略。');
recipes.swiglu.steps[1].apply=()=>map(projection().apply(input),x=>x*.75);
add('rope','RoPE','Ylo=Xlo cosθ−Xhi sinθ; Yhi=Xhi cosθ+Xlo sinθ',[S('读取高低半部与 sin/cos','SR','Xlo / Xhi / sin / cos → VXM',1),S('两路乘加旋转','VXM','每对维度旋转，θ=0.3',5,a=>map(a,(x,c,t)=>c<16?x*Math.cos(.3)-a[t][c+16]*Math.sin(.3):x*Math.cos(.3)+a[t][c-16]*Math.sin(.3)))],'示例所有 pair 使用相同 θ，真实 RoPE 角度依位置和频率变化。布局是否需搬运由 compiler 决定。');
add('depthwise_conv1d','Depthwise Conv1D','Y[t,c] = Σ_i w[i,c] X[t−i,c]',[S('读取因果窗口','MEM','当前 token 与前两个 token；越界补零',3),S('逐 channel 乘加','VXM','w=[0.5,0.3,0.2]；无跨 channel 求和',8,a=>map(a,(x,c,t)=>.5*x+.3*(input[t-1]?.[c]??0)+.2*(input[t-2]?.[c]??0)))],'演示 kernel=3；SiLU 是可选后续激活，未在此默认打开。');
add('reshape','Reshape','只改变逻辑 shape',[S('更改逻辑视图','CONTROL','无数据移动，无 FU 指令',0)],'可由地址/布局解释实现；不能仅凭 reshape 就安排 MEM copy。',{view:true});
add('indexed_view','Indexed View','用 sources / indices 引用已有元素',[S('建立索引视图','CONTROL','逻辑引用，本身不分配 storage',0)],'消费者若不能直接读取该布局，才需要另外产生物理转换。',{view:true});
add('transpose','Transpose / Layout','Y[r,c] = X[c,r]',[S('局部 transpose','SXM','每 SL 内 8×8',8),S('跨 superlane block permute','SXM','重组局部 block',4)],'这里展示现有 SXM 路径；完整跨 SL 大 transpose 是另一个方案。数值预览显示独立 8×8 教学矩阵。',{special:'transpose'});
add('gqa_broadcast','GQA Broadcast','KV heads → 重复匹配 query heads',[S('复用 head 引用 / fanout','SXM','一个 KV head 可供多个 Q head 使用',4)],'是否需要物理 fanout 或只改引用取决于布局与消费者；示例展示有物理 fanout 的情况。');
add('gated_delta_rule','Gated DeltaNet','S̄=αS; δ=β(v−S̄ᵀk); S′=S̄+kδᵀ; o=S′ᵀq',[S('State 衰减','VXM','S̄ = α × S',2),S('旧状态预测','MXM','p = S̄ᵀ k',8),S('纠错向量','VXM','δ = β × (v − p)',3),S('秩一更新','MXM','S′ = S̄ + k δᵀ',8),S('输出投影','MXM','o = S′ᵀ q',8)],'这是数学依赖示意，不宣称当前 recurrence lowering 用 MXM 完成所有阶段；真实 trace 例子来自 chunkwise MXM 测试，二者不能混为同一调度。',{gdn:true});
const primitives=[['Bypass',x=>x,1],['Add',x=>x+.25,1],['Subtract',x=>x-.25,1],['Multiply',x=>x*.5,2],['Negate',x=>-x,1],['Max',x=>Math.max(x,0),1],['Exp',Math.exp,5],['Reciprocal',x=>1/(x+2),5],['Rsqrt',x=>1/Math.sqrt(x+2),5],['Log',x=>Math.log(x+2),5]];
for(const [name,fn,lat] of primitives)add('fu_'+name,'VXM '+name,name+'(x)',[unary(name,name+' · latency='+lat+', II=1',fn,lat)],'独立 FU 演示。Reciprocal / Rsqrt / Log 用 x+2 保证输入定义域；Max 比较 0；二元算术使用示例常量。',{primitive:true});
for(const [name,unit,formula] of [['Read','MEM','MEM[row] → SR'],['Write','MEM','SR → MEM[row]'],['Gather','MEM','map 指定读取地址'],['Scatter','MEM','map 指定写入地址'],['ShiftSelect','SXM','选择当前 / North / South shifted 数据'],['Distribute','SXM','在 superlane 内按 map 选择 lane'],['Permute','SXM','跨 superlane 的完整 block 重排'],['C2CReceive','C2C','DDR → C2C → WRITE_SYNC → MEM'],['C2CSend','C2C','MEM → READ_SYNC → C2C → DDR']])add('fu_'+name,name,formula,[S(name,unit,formula,1)],'只演示传输方向；该示意阶段长度不是 C2C 带宽或 SXM 实际时延。',{primitive:true,transfer:true});
const missing=DATA.ops.filter(id=>!recipes[id]);if(missing.length)throw Error('缺少 Kernel 演示 '+missing);
let selected='l2_norm',mode='demo',cycle=0,timer=null,events=[],steps=[],snapshots=[],finalValue=[],minCycle=0,maxCycle=0,patterns=[],traceId=0,selectedUnit=null;
function makeGdn(){let state=Array.from({length:32},()=>Array(32).fill(0)),outputs=[];for(let t=0;t<4;t++){let k=input[t].map(x=>x/Math.sqrt(sum(input[t].map(x=>x*x))+1e-5)),q=k,v=input[t];state=state.map(r=>r.map(x=>x*.9));let pred=Array.from({length:32},(_,j)=>sum(k.map((x,i)=>x*state[i][j])));let delta=v.map((x,j)=>.6*(x-pred[j]));state=state.map((r,i)=>r.map((x,j)=>x+k[i]*delta[j]));outputs.push(Array.from({length:32},(_,j)=>sum(q.map((x,i)=>x*state[i][j]))));}return {outputs,state};}
function makeDemo(recipe){
 if(recipe.id==='transpose'&&$('sxmVariant').value==='wide')return makeWideTranspose();
 let ev=[],st=[],sn=[],a=clone(input),cursor=0;
 if(recipe.view)return {events:[{start:0,end:1,unit:'CONTROL',sl:null,detail:recipe.steps[0].formula,stage:0}],steps:[{...recipe.steps[0],start:0,end:1}],snapshots:[a],output:a};
 if(recipe.transfer){
  const name=recipe.id.slice(3),paths={Read:['MEM','SR'],Gather:['MEM','SR'],Write:['SR','MEM'],Scatter:['SR','MEM'],C2CReceive:['C2C','CONTROL','MEM'],C2CSend:['MEM','CONTROL','C2C']},path=paths[name]||['SR','SXM','SR'];
  for(let token=0;token<4;token++)for(let sl=0;sl<4;sl++)for(let i=0;i<path.length;i++){
   const unit=path[i],start=token+sl+i;let detail=recipe.formula+` · token ${token} / SL${sl}`;
   if(unit==='CONTROL')detail=(name==='C2CReceive'?'WRITE_SYNC 等待 RX 完成，再允许 MEM 写入':'READ_SYNC 协调 MEM 读出和 C2C 发送')+'（示意 ready；未模拟 DDR 时延）';
   ev.push({start,end:start+1,unit,sl,token,stage:0,detail,from:i===0?path[0]:path[i-1],to:path[Math.min(i+1,path.length-1)]});
  }
  return {events:ev,steps:[{...recipe.steps[0],start:0,end:6+path.length}],snapshots:[a],output:a};
 }
 const passes=recipe.gdn?4:1;
 for(let pass=0;pass<passes;pass++)for(let i=0;i<recipe.steps.length;i++){
  const step=recipe.steps[i],sid=st.length,start=cursor,lat=step.lat,alu=i%8,nt=recipe.gdn?1:4;
  // Map VXM/MXM routes through the configured physical SR boundaries.
  if(['VXM','MXM'].includes(step.unit)){
   const group=Number($('memGroup').value||6),route=physicalStage(step,start,nt,recipe.gdn?pass:0,sid,alu,group);
   ev.push(...route.events);cursor=route.end;st.push({...step,start,end:cursor,pass:recipe.gdn?pass:null});
   if(step.apply)a=step.apply(a);sn.push(clone(a));continue;
  }
  // Other teaching phases have no claimed physical SR placement.
  for(let token=0;token<nt;token++)for(let sl=0;sl<4;sl++){
   let t=start+token+sl,row=recipe.gdn?pass:token;
   const emit=(s,e,unit,detail,from,to)=>ev.push({start:s,end:e,unit,sl,token:row,stage:sid,alu,detail,from,to});
   emit(t,t+1,'MEM',`读取 ${i?'中间 tensor':'X'}[${row},${sl*8}:${sl*8+8}]；8 元素`,'MEM','SR');
   emit(t+1,t+2,'SR',`传送 token ${row} / SL${sl}；示意 1 段 SR 路程`,'SR',step.unit);
   emit(t+2,t+2+lat,step.unit,`${step.label} · ${step.formula} · token ${row} / SL${sl}`);
   emit(t+2+lat,t+3+lat,'SR','结果进入输出 SR',step.unit,'MEM');
   emit(t+3+lat,t+4+lat,'MEM','结果写回教学 scratch；实际可直接衔接消费者','SR','MEM');
  }
  cursor=start+(nt-1)+3+4+lat;
  st.push({...step,start,end:cursor,pass:recipe.gdn?pass:null});
  if(step.apply)a=step.apply(a);sn.push(clone(a));
 }
 if(recipe.gdn)a=makeGdn().outputs;
 if(recipe.special==='transpose')a=Array.from({length:8},(_,r)=>Array.from({length:8},(_,c)=>c*8+r));
 return {events:ev,steps:st,snapshots:sn,output:a};
}
function normalizeRow(r){const num=(key,d)=>r[key]===undefined||r[key]===''?d:Number(r[key]);let p={start:num('start',0),end:num('end',0),resource:r.resource||'',detail:r.detail||'',ic:num('inner_count',1),ii:num('inner_interval',0),is:num('inner_stride',0),oc:num('outer_count',1),oi:num('outer_interval',0),os:num('outer_stride',0),skip:num('skip_first',0)!==0,ind:r.induction||'none',base:num('base_delta',0)};
 for(const k of ['start','end','ic','ii','is','oc','oi','os','base'])if(!Number.isSafeInteger(p[k]))throw Error('CSV 包含非整数 / 过大周期');
 if(p.start<0||p.end<=p.start||p.ic<1||p.oc<1||p.ii<0||p.oi<0||p.ic>1000000||p.oc>1000000||(p.ic>1&&!p.ii)||(p.oc>1&&!p.oi))throw Error('CSV repeat / 周期参数无效');
 if(r.pattern&&!['single','repeat','repeat2d'].includes(r.pattern))throw Error('暂不支持 CSV pattern='+r.pattern);
 p.last=p.end+(p.ic-1)*p.ii+(p.oc-1)*p.oi;return p;
}
function induced(p,i,o){let detail=p.detail,delta=p.base+i*p.is+o*p.os,key={mem_address:'addr',mxm_weight_column:'column',mxm_accumulator_address:'acc'}[p.ind];if(key)detail=detail.replace(new RegExp('\\b'+key+'=(-?\\d+)'),(_,v)=>key+'='+(+v+delta));return detail;}
function realAt(t){const out=[];for(const p of patterns){if(t<p.start||t>=p.last)continue;let dur=p.end-p.start,span=(p.ic-1)*p.ii+dur,of=p.oc===1?0:Math.max(0,Math.floor((t-p.start-span)/p.oi)+1),ol=p.oc===1?0:Math.min(p.oc-1,Math.floor((t-p.start)/p.oi));for(let o=of;o<=ol;o++){let s=p.start+o*p.oi,first=p.ic===1?0:Math.max(0,Math.floor((t-s-dur)/p.ii)+1),last=p.ic===1?0:Math.min(p.ic-1,Math.floor((t-s)/p.ii));for(let i=first;i<=last;i++){if(p.skip&&i===0&&o===0)continue;const start=s+i*p.ii;if(!(start<=t&&t<start+dur))continue;const detail=induced(p,i,o),sl=detail.match(/\b(?:tile|superlane)=(\d+)/);out.push({start,end:start+dur,unit:p.resource.split('.')[0],resource:p.resource,detail,sl:sl?+sl[1]:null});if(out.length>10000)throw Error('本拍超过 10000 个并发记录，请缩小输入 trace。');}}}return out;}
function activeAt(t){return mode==='demo'?events.filter(e=>e.start<=t&&t<e.end):realAt(t);}
function stop(){clearInterval(timer);timer=null;$('play').textContent='播放';}
function setCycle(t){cycle=Math.max(minCycle,Math.min(maxCycle,Math.round(t)||0));render();}
function load(){stop();$('error').textContent='';selectedUnit=null;const recipe=recipes[selected];$('title').textContent=recipe.name;$('formula').textContent=recipe.formula;$('description').textContent=recipe.note;$('realOptions').hidden=mode!=='real';
 if(mode==='demo'){const demo=makeDemo(recipe);events=demo.events;steps=demo.steps;snapshots=demo.snapshots;finalValue=demo.output;minCycle=0;maxCycle=Math.max(...events.map(e=>e.end))-1;$('badge').textContent='机制演示 · 教学调度，不是 compiler/cmodel 的执行周期';$('scope').textContent='覆盖仓库全部 '+DATA.ops.length+' 类 Kernel IR 算子，以及 10 种 VXM 算术和 MEM/SXM/C2C 传输原语。每个示例只说明选定形状与算子属性。';}
 else {const tr=DATA.traces[traceId];patterns=tr.rows.map(normalizeRow);minCycle=tr.start;maxCycle=tr.end-1;steps=[];$('badge').textContent='真实 runtime CSV · 保留绝对周期，未运行新测试';$('traceSource').textContent=tr.source+' · '+(tr.imported?'导入完整文件':'内置窗口，非整个测试')+' · ['+tr.start+', '+tr.end+')';$('scope').textContent='当前回放 '+tr.name+'。这份 trace 是具体测试的记录，不表示当前选择算子的所有形状或最新代码都验证通过。';}
 cycle=minCycle;$('seek').min=minCycle;$('seek').max=maxCycle;$('cycleNumber').min=minCycle;$('cycleNumber').max=maxCycle;renderCatalog();render();
}
function renderCatalog(){const query=$('search').value.toLowerCase();$('catalog').innerHTML=['Kernel 算子','FU 原语'].map((group,gi)=>'<p class="eyebrow">'+group+'</p>'+Object.values(recipes).filter(r=>!!r.primitive===!!gi&&(r.name+' '+r.id).toLowerCase().includes(query)).map(r=>`<button class="op ${r.id===selected?'active':''}" data-op="${r.id}">${esc(r.name)}<small>${r.id}${DATA.traces.some(t=>t.op===r.id)?' · 有 trace':''}</small></button>`).join('')).join('');$('coverage').textContent=DATA.ops.length+' 类 Kernel · '+Object.values(recipes).filter(r=>r.primitive).length+' 种原语';}
const colors={MEM:'#52dfbc',SR:'#62cfe5',SXM:'#b6a4ff',MXM:'#ffbd69',VXM:'#ff9eaa',C2C:'#8eafff',CONTROL:'#c6cdd6'};
function drawLegacyFabric(active){
 let s='<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0 0L10 5L0 10Z" fill="#4a6989"/></marker></defs>';
 const txt=(x,y,t,size=12,fill='#dfeaf8')=>`<text x="${x}" y="${y}" font-size="${size}" fill="${fill}">${esc(t)}</text>`;
 s+=txt(12,22,mode==='demo'?'教学路径：每个 vector 的 SL0/1/2/3 错 0/1/2/3 拍':'真实记录：只点亮已记录的资源；未知 SL 用 ? 标注');
 const columns=[['MEM',20,125],['SR',165,105],['SXM',290,135],['MXM',445,170],['VXM',635,210],['C2C',865,115]];
 for(let j=0;j<columns.length;j++){let [unit,x,w]=columns[j];let hit=active.filter(e=>e.unit===unit);s+=`<g data-unit="${unit}" style="cursor:pointer"><rect x="${x}" y="45" width="${w}" height="270" rx="9" fill="${hit.length?'#213d4d':'#152238'}" stroke="${hit.length?colors[unit]:'#30425c'}" stroke-width="${selectedUnit===unit?3:1}"/>`+txt(x+10,68,unit,16,colors[unit]);
  for(let sl=0;sl<4;sl++){let ev=hit.filter(e=>e.sl===sl),y=86+sl*53;s+=`<rect x="${x+8}" y="${y}" width="${w-16}" height="44" rx="4" fill="${ev.length?'#3b5861':'#1c2c42'}"/>`+txt(x+14,y+16,'SL'+sl,11)+txt(x+14,y+33,ev.length?ev.length+' 活动':'·',10,ev.length?colors[unit]:'#7186a3');if(ev.length&&mode==='demo')s+=txt(x+w-46,y+16,'t'+ev[0].token,10);}
  let unknown=hit.filter(e=>e.sl===null);s+=txt(x+10,308,unknown.length?'? SL 未记录：'+unknown.length:'',10,colors[unit])+'</g>';
 }
 // Only illustrative mode has inferred wiring. Real mode must not invent SR traffic.
 if(mode==='demo')for(const e of active.filter(e=>e.from&&e.to&&e.unit==='SR')){let a=columns.find(c=>c[0]===e.from),b=columns.find(c=>c[0]===e.to);if(a&&b){let y=101+e.sl*53;s+=`<path d="M${a[1]+a[2]/2} ${y} L${b[1]+b[2]/2} ${y}" stroke="#52dfbc" stroke-width="2" opacity=".7" marker-end="url(#arrow)"/><circle cx="${(a[1]+a[2]/2+b[1]+b[2]/2)/2}" cy="${y}" r="4" fill="#52dfbc"/>`;}}
 s+=txt(20,348,'VXM stage 活动（真实模式保留 C 编号；示意模式只表示链步骤）',13);
 for(let c=0;c<16;c++){let x=20+(c%8)*121,y=364+Math.floor(c/8)*65;let hits=active.filter(e=>e.unit==='VXM'&&(mode==='demo'?e.alu===c:new RegExp('\\.C'+c+'(?:\\.|$)').test(e.resource||'')));s+=`<rect x="${x}" y="${y}" width="110" height="52" rx="6" fill="${hits.length?'#5e3d4c':'#1b2a40'}" stroke="${hits.length?'#ff9eaa':'#30425c'}"/>`+txt(x+8,y+19,'C'+c,12)+txt(x+8,y+38,hits.length?(mode==='demo'?'流水阶段':'日志活动'):'空闲 / 未记录',10,hits.length?'#ffbec7':'#7e93af');}
 const hemi=h=>active.filter(e=>(e.resource||'').includes('.'+h+'.')).length;
 s+=txt(20,523,mode==='real'?`半球显式记录：E ${hemi('E')} 条 / W ${hemi('W')} 条。无半球字段的 VXM 不强行归属。`:'示意不绑定某个 MEM slice / bank；不同算子的真实地址分配请切换 runtime。');s+=txt(20,550,'点击功能单元可筛选右侧事件；再次点击取消。',12,'#a0b2c9');$('fabric').innerHTML=s;
}
function displayTensor(label,a){const token=+$('token').value,sl=+$('sl').value,row=a[Math.min(token,a.length-1)]||[],values=row.slice(sl*8,sl*8+8);return `<p>${esc(label)} <code>[${a.length}, ${row.length}]</code></p><div class="valuegrid">${values.map(v=>`<span title="${v}">${Number.isFinite(v)?v.toFixed(3):String(v)}</span>`).join('')||'<p class="tiny">该 tensor 在此 SL 无元素</p>'}</div>`;}
function render(){const recipe=recipes[selected],active=activeAt(cycle);$('seek').value=cycle;$('cycleNumber').value=cycle;$('clockInfo').textContent=`cycle ${cycle} / ${maxCycle} · ${active.length} 条活动记录 · 区间语义 [start,end)`;
 $('steps').innerHTML=steps.map((s,i)=>`<button class="step ${cycle>=s.start&&cycle<s.end?'on':''} ${cycle>=s.end?'done':''}" data-stage="${i}">${s.pass===null||s.pass===undefined?'':'token '+s.pass+' · '}${esc(s.label)}<br><small>${s.start}–${s.end-1}</small></button>`).join('');
 drawFabric(active);let filtered=selectedUnit?active.filter(e=>e.unit===selectedUnit):active;
 $('events').innerHTML=filtered.slice(0,160).map(e=>`<tr><td style="color:${colors[e.unit]||'#c6cdd6'}">${esc(e.resource||e.unit)}${e.sl===null?'':' / SL'+e.sl}<br>${e.start}–${e.end-1}</td><td>${esc(e.detail)}</td></tr>`).join('')||'<tr><td colspan="2">本拍无匹配事件</td></tr>';$('eventCount').textContent=filtered.length>160?`共 ${filtered.length} 条，表格仅列前 160 条；单元统计包含全部。`:selectedUnit?'筛选：'+selectedUnit:'';
 if(mode==='demo'){let idx=steps.findIndex(s=>s.start<=cycle&&cycle<s.end),step=steps[idx];$('explain').innerHTML=step?`<p><b>${esc(step.label)}</b> · ${esc(step.formula)}</p>`:'<p>本拍处于阶段边界。</p>';
 $('mapNote').textContent='按配置的 slice 连接与 SR hop 展开；VXM/MXM 教学阶段逐列传播，stream 16/17 输入、18/19 输出是示例分配。其余阶段无确证 SR 路径时显示未知。横向距离是 hop 数，不是版图毫米尺寸。';
 let completed=steps.reduce((r,s,i)=>cycle>=s.end-1?i:r,-1),preview=recipe.gdn||recipe.special?finalValue:(snapshots[Math.max(0,idx)]||input);
 $('tensors').innerHTML=displayTensor('输入 X',input)+displayTensor(recipe.gdn?'最终 o（预先计算参考，不表示已完成）':recipe.special?'8×8 转置参考':`当前阶段预期结果${step?'：'+step.label:''}（参考）`,preview)+`<p class="tiny">已完成阶段：${completed+1} / ${steps.length}</p>`;
 if(recipe.gdn){const g=makeGdn();$('tensors').innerHTML+='<p>最终 State 左上角 4×4（FP32 理想参考）</p><pre>'+g.state.slice(0,4).map(r=>r.slice(0,4).map(x=>x.toFixed(4)).join('  ')).join('\n')+'</pre>';}
 $('tensorNote').textContent=recipe.view?'纯逻辑视图，没有硬件数据传输。':'数值使用 JavaScript 理想浮点，不模拟 BF16 舍入、量化或 LUT 误差；数值预览与周期调度独立，未冒充 runtime payload。';
 }else{$('explain').innerHTML='<p>以下是原始 trace 记录覆盖此周期的事件。记录可能是 issue/repeat 区间，不等于内部 ALU 结果已经 ready。</p>';$('mapNote').textContent='SR 未记录则保持空白，不因 MEM Read 就推断 stream 的具体位置。SL 未记录的事件计入 ? 行。';$('tensors').innerHTML='<p>这份 pipeline CSV 没有逐 lane 的数值 payload。</p><p>地址、stream、PC 等字段见上方事件表；如需真正跟踪某个元素，需要扩展 CModel 数据 trace。</p>';$('tensorNote').textContent='不生成或猜测真实数值。';}
 drawTimeline();renderSxmPanel();
}
function drawTimeline(){const cv=$('timeline'),w=Math.max(600,cv.clientWidth||600),h=150;cv.width=w*2;cv.height=h*2;const c=cv.getContext('2d');c.scale(2,2);c.clearRect(0,0,w,h);const units=Object.keys(colors),start=Math.max(minCycle,cycle-12),end=Math.min(maxCycle+1,start+72),scale=(w-65)/(end-start);for(let row=0;row<units.length;row++){c.fillStyle='#9db0ca';c.font='10px monospace';c.fillText(units[row],0,20+row*18);}for(let t=start;t<end;t++){const hits=new Set(activeAt(t).map(e=>e.unit));for(const unit of hits){let y=units.indexOf(unit);if(y<0)continue;c.fillStyle=colors[unit];c.fillRect(62+(t-start)*scale,9+y*18,Math.max(1,scale-.8),12);}}c.strokeStyle='#fff';c.beginPath();c.moveTo(62+(cycle-start)*scale,0);c.lineTo(62+(cycle-start)*scale,137);c.stroke();c.fillStyle='#fff';c.fillText(start,62,149);c.fillText(end-1,w-40,149);cv.dataset.start=start;cv.dataset.end=end;cv.dataset.width=w;}
function choose(id){selected=id;let trace=DATA.traces.findIndex(t=>t.op===id);if(trace>=0){traceId=trace;$('traceSelect').value=trace;}else if(mode==='real'){mode='demo';$('mode').value=mode;}load();}
function populateTraces(){$('traceSelect').innerHTML=DATA.traces.map((t,i)=>`<option value="${i}">${esc(t.name)}</option>`).join('');$('traceSelect').value=traceId;}
function parseCSV(text){let rows=[],row=[],cell='',quoted=false;for(let i=0;i<text.length;i++){let c=text[i];if(c==='"'){if(quoted&&text[i+1]==='"'){cell+='"';i++;}else quoted=!quoted;}else if(!quoted&&(c===','||c==='\n')){row.push(cell);cell='';if(c==='\n'){rows.push(row);row=[];}}else if(c!=='\r'||quoted)cell+=c;}if(quoted)throw Error('CSV 引号未闭合');if(cell||row.length){row.push(cell);rows.push(row);}const head=(rows.shift()||[]).map(s=>s.replace(/^\uFEFF/,''));for(const key of ['start','end','resource'])if(!head.includes(key))throw Error('CSV 缺少列 '+key);return rows.filter(r=>r.some(Boolean)).map(r=>Object.fromEntries(head.map((h,i)=>[h,r[i]??''])));}
function nextEvent(){if(mode==='demo'){let future=events.flatMap(e=>[e.start,e.end]).filter(t=>t>cycle);setCycle(future.length?Math.min(...future):maxCycle);return;}let next=maxCycle;for(const p of patterns){if(p.start>cycle){next=Math.min(next,p.start);continue;}if(p.last<=cycle)continue;let o=p.oc===1?0:Math.max(0,Math.floor((cycle-p.start-(p.ic-1)*p.ii-(p.end-p.start))/p.oi));for(;o<p.oc;o++){let s=p.start+o*p.oi;if(s>next)break;for(const off of [0,p.end-p.start]){let i=p.ic===1?0:Math.max(0,Math.floor((cycle-s-off)/p.ii)+1);if(p.skip&&o===0&&i===0)i=1;if(i<p.ic){let t=s+i*p.ii+off;if(t>cycle)next=Math.min(next,t);}}}}setCycle(next);}
$('catalog').onclick=e=>{const b=e.target.closest('[data-op]');if(b)choose(b.dataset.op);};$('search').oninput=renderCatalog;
$('mode').onchange=()=>{mode=$('mode').value;load();};$('traceSelect').onchange=()=>{traceId=+$('traceSelect').value;selected=DATA.traces[traceId].op||selected;load();};
$('prev').onclick=()=>{stop();setCycle(cycle-1);};$('next').onclick=()=>{stop();setCycle(cycle+1);};$('eventNext').onclick=()=>{stop();nextEvent();};$('reset').onclick=()=>{stop();setCycle(minCycle);};$('seek').oninput=()=>{stop();setCycle(+$('seek').value);};$('cycleNumber').onchange=()=>{stop();setCycle(+$('cycleNumber').value);};
$('play').onclick=()=>{if(timer){stop();return;}if(cycle===maxCycle)cycle=minCycle;$('play').textContent='暂停';timer=setInterval(()=>{if(cycle>=maxCycle){stop();return;}setCycle(cycle+1);},+$('speed').value);};$('speed').onchange=stop;
$('steps').onclick=e=>{const b=e.target.closest('[data-stage]');if(b){stop();setCycle(steps[+b.dataset.stage].start);}};
$('fabric').onclick=e=>{const g=e.target.closest('[data-unit]');if(g){selectedUnit=selectedUnit===g.dataset.unit?null:g.dataset.unit;render();}};
$('sl').onchange=render;$('token').onchange=render;
$('timeline').onclick=e=>{const cv=$('timeline'),r=cv.getBoundingClientRect(),x=(e.clientX-r.left)/r.width*(+cv.dataset.width);stop();setCycle(+cv.dataset.start+(x-62)/(+cv.dataset.width-65)*(+cv.dataset.end-+cv.dataset.start));};
$('file').onchange=async()=>{stop();try{const file=$('file').files[0];if(!file)return;if(file.size>160*1024*1024)throw Error('文件超过 160 MiB，请导出一个算子片段后导入。');const rows=parseCSV(await file.text());if(!rows.length)throw Error('CSV 没有事件');let lo=Infinity,hi=0;for(const r of rows){const p=normalizeRow(r);lo=Math.min(lo,p.start);hi=Math.max(hi,p.last);}DATA.traces.push({op:selected,name:file.name,source:file.name,start:lo,end:hi,rows,imported:true});traceId=DATA.traces.length-1;populateTraces();mode='real';$('mode').value=mode;load();}catch(e){$('error').textContent=e.message;}};
$('export').onclick=()=>{let content=mode==='demo'?{kind:'teaching_schedule_not_runtime',op:selected,events}: {kind:'runtime_patterns',source:DATA.traces[traceId].source,rows:DATA.traces[traceId].rows};let url=URL.createObjectURL(new Blob([JSON.stringify(content,null,2)],{type:'application/json'})),a=document.createElement('a');a.href=url;a.download='lpu-'+selected+'-'+mode+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
function selfTest(){for(const op of DATA.ops)if(!recipes[op])throw Error('catalog');for(const r of Object.values(recipes)){const d=makeDemo(r);if(d.events.some(e=>!Number.isFinite(e.start)||e.end<=e.start))throw Error('bad schedule '+r.id);if(d.output.flat().some(x=>!Number.isFinite(x)))throw Error('nonfinite '+r.id);}const l2=makeDemo(recipes.l2_norm).output;for(const row of l2)if(Math.abs(sum(row.map(x=>x*x))-1)>1e-5)throw Error('L2');const soft=makeDemo(recipes.softmax).output;if(soft.some(row=>Math.abs(sum(row)-1)>1e-12))throw Error('softmax');return '内置检查通过：18 类 IR 覆盖、全部示例有限数值、L2 范数与 Softmax 行和、事件区间有效。';}
$('checks').textContent=selfTest();populateTraces();choose(selected);
