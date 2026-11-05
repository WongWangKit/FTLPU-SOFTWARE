(function(){
"use strict";
const el=Object.fromEntries(["summary","cycle","slider","span","max","badges","wave","overview","cycleTitle","unit","issues","events","scoresHead","scores","comparison","counts","inverse","detail","file","play"].map(id=>[id,document.getElementById(id)]));
const units=["RCU","MXM","VXM","DMA","MEM","ICU"],colors={RCU:"#668bba",MXM:"#377eb8",VXM:"#dd8754",DMA:"#bd9848",MEM:"#499b81",ICU:"#8c79ab"};
let model=null,cycle=0,playing=null,waveStart=0;
const escapeHTML=v=>String(v??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const fmt=v=>v===null||v===undefined?"—":Number.isFinite(v)?Number(v.toPrecision(7)):String(v);
const stageNames={buffered:"FIFO 等待",issue:"指令发射",controller_issue:"设备配置发射",internal:"内部步骤",execute:"执行",pipeline:"流水中",result:"结果产生",activate:"DMA 启动",active:"传输在途",ddr_read_request:"DDR 读请求",ddr_write_request:"DDR 写请求",data_return:"DDR 返回",write_complete:"DDR 写完成",complete:"DMA 完成",credit_wait:"信用等待",wait:"等待",fetch:"描述符装载",read_commit:"SRAM 读",write_commit:"写入提交",control_read:"控制口读",control_write:"控制口写"};
function load(data){model=MoeCycleModel.prepare(data);el.summary.textContent=`${model.T} token · ${model.E} experts · Top-${model.K} · ${model.count.toLocaleString()} 物理 cycles · ${model.events.length.toLocaleString()} 条实际记录`;
  el.cycle.max=el.slider.max=model.count-1;el.max.textContent=`/ ${model.count-1}`;
  el.scoresHead.innerHTML="<tr><th>token</th>"+Array.from({length:model.E},(_,i)=>`<th>E${i} 分数</th>`).join("")+"<th>Top-K 寄存器</th><th>已写入选择结果</th></tr>";
  seek(model.jumps.router);
  document.getElementById('loadStatus').textContent='交互视图已就绪：可点击总览、逐周期查看，或使用阶段跳转。';
}
function seek(value){if(!model)return;cycle=Math.max(0,Math.min(model.count-1,Math.round(Number(value)||0)));el.cycle.value=el.slider.value=cycle;render();}
function keyFields(e){const p=e.payload;if(e.resource==="RCU.Core"&&e.detail==="Compare"){const c=model.routeAt(e.cycle).compare;return `token=${p.token} ${c.incomingValid?`E${p.incoming_id}`:"无剩余有效输入"} slot=${p.slot} ${p.exchanged==="1"?"交换":"保留"} → E${p.winner_id}`;}
  return Object.entries(p).filter(([k])=>["token","expert","slot","tile","lane","row","stream","ddr_address","vector_index","result","lhs","rhs","descriptor","assignment","value"].includes(k)).map(([k,v])=>`${k}=${v}`).join(" · ")||e.detail;}
function render(){
  const rows=model.at(cycle);el.cycleTitle.textContent=`物理 cycle ${cycle.toLocaleString()} · ${rows.length} 条记录`;
  el.badges.innerHTML=units.map(unit=>{const status=model.activity(cycle,unit),n=rows.filter(e=>e.unit===unit).length;return `<div class="badge" style="border-left-color:${status==="idle"?"#c8d2dd":colors[unit]}"><strong>${unit}</strong><small>${status==="active"?"活动":status==="waiting"?"等待／在途":"空闲"} · ${n}</small></div>`;}).join("");
  const filtered=rows.filter(e=>(el.unit.value==="ALL"||e.unit===el.unit.value)&&(!el.issues.checked||["issue","controller_issue","activate"].includes(e.stage)));
  el.events.innerHTML=filtered.length?filtered.map((e,i)=>`<tr data-event="${i}"><td><b>${e.unit}</b><br>${escapeHTML(e.resource)}</td><td>${escapeHTML(stageNames[e.stage]||e.stage)}</td><td>${escapeHTML(e.opcode)}<br><small>${escapeHTML(e.detail)}</small></td><td>${escapeHTML(e.pc||"—")}</td><td class="key">${escapeHTML(keyFields(e))}</td></tr>`).join(""):"<tr><td class='empty' colspan='5'>本周期无符合筛选条件的记录</td></tr>";
  el.events.querySelectorAll("tr[data-event]").forEach(row=>row.onclick=()=>{el.detail.textContent=JSON.stringify(filtered[Number(row.dataset.event)],null,2);});
  const state=model.routeAt(cycle);
  const chips=(list,final)=>list.map((v,i)=>v?`<span class="chip">${i}: E${v.expert} · ${escapeHTML(fmt(final?v.weight:v.score))}</span>`:`<span class="chip">${i}: —</span>`).join(" ");
  el.scores.innerHTML=state.scores.map((scores,t)=>`<tr${state.compare?.cycle===cycle&&state.compare.token===t?" class='current'":""}><td>${t}</td>${scores.map(v=>`<td>${fmt(v)}</td>`).join("")}<td>${chips(state.top[t],false)}</td><td>${chips(state.selected[t],true)}</td></tr>`).join("");
  const c=state.compare;el.comparison.innerHTML=c?`<b>${c.cycle===cycle?"本周期比较":"最近一次比较"} · cycle ${c.cycle}</b><br>token ${c.token}，Top-K slot ${c.slot}：${c.incomingValid?`输入 E${c.incoming} (${fmt(c.score)})`:"本轮已无剩余有效输入"}，候选 ${c.candidate===null?"空":`E${c.candidate} (${fmt(c.candidateScore)})`}<br>${c.exchanged?"发生比较交换":"保留原候选"}。有效候选见上表；该步骤属于 ROUTE_SELECT 内部执行。`:"尚未执行 Top-K 比较";
  el.counts.innerHTML=state.counts.map((v,i)=>`<tr><td>E${i}</td><td>${fmt(v)}</td><td>${fmt(state.offsets[i])}</td></tr>`).join("");
  el.inverse.innerHTML=state.inverse.map((v,t)=>`<div>token ${t}：${v.map((x,s)=>`slot ${s} → ${x??"—"}`).join("　")}</div>`).join("");
  draw();
}
function drawOverview(){
  const canvas=el.overview,dpr=window.devicePixelRatio||1,w=canvas.clientWidth,h=350,left=130,right=12,top=28,rowHeight=29,span=w-left-right;
  canvas.width=w*dpr;canvas.height=h*dpr;const ctx=canvas.getContext("2d");ctx.scale(dpr,dpr);ctx.clearRect(0,0,w,h);ctx.font='11px "Segoe UI",sans-serif';
  const labels=['程序 / 描述符装载','Router 打分','Top-K / PLAN','专家输入 Gather','专家权重落 SRAM','专家 0','专家 1','专家 2','专家 3','加权回并'];
  if(!model.data.phases?.length){ctx.fillStyle='#62738b';ctx.fillText('此数据未附带阶段区间；可使用功能单元波形和阶段跳转。',left,45);return;}
  for(let i=0;i<=6;++i){const x=left+span*i/6;ctx.fillStyle='#62738b';ctx.fillText(String(Math.round(model.count*i/6)),x,15);ctx.strokeStyle='#e8edf3';ctx.beginPath();ctx.moveTo(x,top);ctx.lineTo(x,top+rowHeight*10);ctx.stroke();}
  labels.forEach((label,i)=>{ctx.fillStyle='#38516e';ctx.fillText(label,4,top+i*rowHeight+18);});
  for(const p of model.data.phases){const x=left+span*p.start/model.count,width=span*(p.end-p.start)/model.count;ctx.fillStyle=p.color;ctx.fillRect(x,top+p.row*rowHeight+3,Math.max(1,width),22);if(width>42){ctx.fillStyle='#fff';ctx.fillText(p.label,x+3,top+p.row*rowHeight+18);}}
  const x=left+span*cycle/model.count;ctx.strokeStyle='#20384f';ctx.beginPath();ctx.moveTo(x,top-4);ctx.lineTo(x,top+rowHeight*10);ctx.stroke();
  ctx.fillStyle='#62738b';ctx.fillText('专家计算颜色：蓝 Gate · 青 Up · 橙 SwiGLU · 紫 Down；横轴为实际物理 cycle',left,h-6);
}
function draw(){if(!model)return;drawOverview();const canvas=el.wave,dpr=window.devicePixelRatio||1,w=canvas.clientWidth,h=240;canvas.width=w*dpr;canvas.height=h*dpr;const ctx=canvas.getContext("2d");ctx.scale(dpr,dpr);ctx.clearRect(0,0,w,h);
  const left=60,right=12,top=28,rowHeight=32,span=Number(el.span.value),cw=(w-left-right)/span;
  waveStart=Math.max(0,Math.min(model.count-span,cycle-Math.floor(span/2)));
  ctx.font='11px "Segoe UI",sans-serif';ctx.fillStyle="#62738b";
  for(let i=0;i<span;++i){const t=waveStart+i;if(i%Math.max(1,Math.floor(span/8))===0){ctx.fillText(String(t),left+i*cw,15);ctx.strokeStyle="#e8edf3";ctx.beginPath();ctx.moveTo(left+i*cw,top);ctx.lineTo(left+i*cw,h-10);ctx.stroke();}
    units.forEach((unit,j)=>{const status=model.activity(t,unit);if(status==="idle")return;ctx.globalAlpha=status==="waiting"?.25:.9;ctx.fillStyle=colors[unit];ctx.fillRect(left+i*cw,top+j*rowHeight+4,Math.max(.8,cw-.7),26);ctx.globalAlpha=1;});}
  units.forEach((unit,j)=>{ctx.fillStyle="#38516e";ctx.fillText(unit,4,top+j*rowHeight+21);});
  const x=left+(cycle-waveStart+.5)*cw;ctx.strokeStyle="#20384f";ctx.lineWidth=1.5;ctx.beginPath();ctx.moveTo(x,top-4);ctx.lineTo(x,h-11);ctx.stroke();
}
document.getElementById("first").onclick=()=>seek(0);document.getElementById("prev").onclick=()=>seek(cycle-1);document.getElementById("next").onclick=()=>seek(cycle+1);
el.cycle.onchange=()=>seek(el.cycle.value);el.slider.oninput=()=>seek(el.slider.value);el.span.onchange=draw;el.unit.onchange=render;el.issues.onchange=render;
document.querySelectorAll("[data-jump]").forEach(button=>button.onclick=()=>{if(model)seek(model.jumps[button.dataset.jump]);});
el.play.onclick=()=>{if(playing){clearInterval(playing);playing=null;el.play.textContent="播放";}else{playing=setInterval(()=>{seek(cycle+1);if(model&&cycle>=model.count-1)el.play.click();},80);el.play.textContent="暂停";}};
el.wave.onclick=event=>{if(!model)return;const rect=el.wave.getBoundingClientRect();const x=event.clientX-rect.left;seek(waveStart+Math.floor((x-60)/(rect.width-72)*Number(el.span.value)));};
el.overview.onclick=event=>{if(!model)return;const rect=el.overview.getBoundingClientRect();seek(Math.floor((event.clientX-rect.left-130)/(rect.width-142)*model.count));};
el.overview.onmousemove=event=>{if(!model)return;const rect=el.overview.getBoundingClientRect(),t=(event.clientX-rect.left-130)/(rect.width-142)*model.count,row=Math.floor((event.clientY-rect.top-28)/29);const phases=(model.data.phases||[]).filter(p=>p.row===row&&p.start<=t&&t<p.end);el.overview.title=phases.map(p=>`${p.label} [${p.start}, ${p.end})`).join('\n');};
document.addEventListener("keydown",e=>{if(e.target.matches("input,select"))return;if(e.key==="ArrowRight"){e.preventDefault();seek(cycle+1);}if(e.key==="ArrowLeft"){e.preventDefault();seek(cycle-1);}});
window.addEventListener("resize",draw);
el.file.onchange=async()=>{try{const file=el.file.files[0],text=await file.text();load(file.name.endsWith('.json')?JSON.parse(text):MoeCycleModel.parseCSV(text));}catch(e){el.summary.textContent=e.message;}};
window.loadMoeCycles=load;
if(window.MOE_CYCLE_DATA)load(window.MOE_CYCLE_DATA);
})();
