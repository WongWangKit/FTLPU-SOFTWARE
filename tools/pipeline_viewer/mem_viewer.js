"use strict";

const ROW_H = 32;
const ALL_SLICES = 32;
const ALL_BANKS = 2;
const els = Object.fromEntries([
  "fileInput", "sampleButton", "fitButton", "zoomInButton", "zoomOutButton",
  "hemisphereFilter", "sliceFilter", "bankFilter", "portFilter", "stageFilter",
  "searchInput", "allPortsToggle", "firstButton", "prevButton", "playButton",
  "nextButton", "lastButton", "cycleInput", "speedSelect", "viewportReadout",
  "eventReadout", "portReadout", "traceSummary", "resourceLabels", "waveScroller",
  "waveTrack", "waveCanvas", "rulerCanvas", "dropOverlay", "inspectorCycle",
  "cycleSummary", "cycleTableBody", "eventDetail", "fileName", "interactionState"
].map(id => [id, document.getElementById(id)]));

const state = {
  events: [], filtered: [], rows: [], rowEvents: new Map(), cycleEvents: new Map(),
  minCycle: 0, maxCycle: 16, viewStart: 0, viewSpan: 16, cycle: 0,
  playing: false, timer: null, selected: null, drag: null, fileName: "built-in.mem.csv"
};

function memCsvWorkerMain() {
  const required = ["cycle", "hemisphere", "slice", "bank", "port", "stage", "action"];
  function parseLine(line) {
    const fields=[]; let field="", quoted=false;
    for(let i=0;i<line.length;++i){const ch=line[i];if(ch==='"'){if(quoted&&line[i+1]==='"'){field+='"';++i;}else quoted=!quoted;}else if(ch===","&&!quoted){fields.push(field);field="";}else field+=ch;}
    fields.push(field); return fields;
  }
  self.onmessage=async({data})=>{const reader=data.file.stream().getReader(),decoder=new TextDecoder();let carry="",columns=null,batch=[],count=0;
    try{for(;;){const {value,done}=await reader.read();carry+=decoder.decode(value||new Uint8Array(),{stream:!done});const lines=carry.split(/\r?\n/);carry=done?"":lines.pop();
        for(const line of lines){if(!line.trim())continue;if(!columns){columns=parseLine(line).map(v=>v.trim());const missing=required.filter(v=>!columns.includes(v));if(missing.length)throw new Error(`不是 MEM trace CSV，缺少字段: ${missing.join(", ")}`);continue;}
          const fields=parseLine(line),row={};for(let i=0;i<columns.length;++i)row[columns[i]]=fields[i]??"";batch.push(row);++count;if(batch.length>=5000){self.postMessage({type:"batch",rows:batch,count});batch=[];}}
        if(done)break;}
      if(!columns)throw new Error("CSV 文件为空");if(batch.length)self.postMessage({type:"batch",rows:batch,count});self.postMessage({type:"done",count});
    }catch(error){self.postMessage({type:"error",message:error.message||String(error)});}
  };
}

function createMemCsvWorker() {
  const url=URL.createObjectURL(new Blob([`(${memCsvWorkerMain.toString()})()`],{type:"text/javascript"}));
  const worker=new Worker(url); worker.sourceUrl=url; return worker;
}
function disposeMemCsvWorker(worker) { worker.terminate(); URL.revokeObjectURL(worker.sourceUrl); }

function sampleEvents() {
  const out = [];
  const add = (cycle, h, slice, bank, port, tile, stage, action, opcode, address, dir, stream, extra = {}) =>
    out.push({cycle, hemisphere:h, slice, bank, port, tile, stage, action, opcode,
      address, stream_direction:dir, stream_index:stream, sr_column:"", vector_tag:"",
      data_hex:"", source:"program", pc:"", iq_before:"", iq_after:"", ...extra});
  add(2,"E",2,1,"read","","icu","functional_issue","Read",17,"E",3,{pc:1,iq_before:1,iq_after:0});
  for (let tile=0; tile<4; ++tile) {
    add(3+tile,"E",2,1,"read",tile,"pipeline","execute","Read",17,"E",3);
    add(3+tile,"E",2,1,"read",tile,"sram","read_to_sr","Read",17,"E",3,
      {sr_column:1,vector_tag:40+tile,data_hex:[...Array(8)].map((_,i)=>(0x10+tile*8+i).toString(16).padStart(2,"0")).join("")});
  }
  add(5,"W",7,0,"write","","icu","sync_data_wait","Write",83,"E",9,{pc:4,iq_before:1,iq_after:1});
  add(6,"W",7,0,"write","","icu","sync_issue","Write",83,"E",9,{pc:4,iq_before:1,iq_after:0,source:"c2c"});
  for (let tile=0; tile<4; ++tile) {
    add(7+tile,"W",7,0,"write",tile,"pipeline","execute","Write",83,"E",9,{source:"c2c"});
    add(7+tile,"W",7,0,"write",tile,"sram","write_commit","Write",83,"E",9,
      {sr_column:3,vector_tag:90+tile,data_hex:"a0a1a2a3a4a5a6a7",source:"c2c"});
  }
  return out;
}

function numeric(value, fallback = -1) {
  if (value === "" || value === null || value === undefined) return fallback;
  const n = Number(value); return Number.isFinite(n) ? n : fallback;
}

function normalize(row) {
  return {
    cycle: numeric(row.cycle, 0), hemisphere: String(row.hemisphere || "E").toUpperCase(),
    slice: numeric(row.slice, 0), bank: numeric(row.bank, 0), port: String(row.port || "read").toLowerCase(),
    tile: numeric(row.tile), stage: String(row.stage || ""), action: String(row.action || ""),
    opcode: String(row.opcode || ""), address: numeric(row.address),
    stream_direction: String(row.stream_direction || ""), stream_index: numeric(row.stream_index),
    sr_column: numeric(row.sr_column), vector_tag: row.vector_tag === "" ? "" : String(row.vector_tag),
    data_hex: String(row.data_hex || ""), source: String(row.source || ""), pc: numeric(row.pc),
    iq_before: numeric(row.iq_before), iq_after: numeric(row.iq_after)
  };
}

function portKey(e) { return `MEM.${e.hemisphere}.S${String(e.slice).padStart(2,"0")}.B${e.bank}.${e.port === "write" ? "W" : "R"}`; }
function rowSort(a,b) { return a.localeCompare(b, undefined, {numeric:true}); }
function colorFor(e) {
  if (e.stage === "icu") return /wait|gated|underflow|delay/.test(e.action) ? "#a45b65" : "#167a72";
  if (e.stage === "pipeline") return ["#87a7d0","#6f96c7","#5a85bd","#4773aa"][Math.max(0,e.tile)%4];
  return e.action === "write_commit" ? "#d38c35" : "#55a270";
}
function stageBand(stage) { return stage === "icu" ? [3,7] : stage === "pipeline" ? [12,7] : [21,7]; }
function streamName(e) { return e.stream_index < 0 ? "—" : `${e.stream_direction}${e.stream_index}`; }
function eventText(e) {
  const address = e.address < 0 ? "" : ` a=${e.address}`;
  const stream = e.stream_index < 0 ? "" : ` s=${streamName(e)}`;
  const tile = e.tile < 0 ? "" : ` t${e.tile}`;
  return `${e.action}${tile}${address}${stream}`;
}

function filterEvents() {
  const h = els.hemisphereFilter.value, s = els.sliceFilter.value, b = els.bankFilter.value;
  const p = els.portFilter.value, stage = els.stageFilter.value;
  const search = els.searchInput.value.trim().toLowerCase();
  state.filtered = state.events.filter(e =>
    (h === "ALL" || e.hemisphere === h) && (s === "ALL" || e.slice === Number(s)) &&
    (b === "ALL" || e.bank === Number(b)) && (p === "ALL" || e.port === p) &&
    (stage === "ALL" || e.stage === stage) && (!search ||
      `${portKey(e)} ${e.stage} ${e.action} ${e.opcode} ${e.address} ${streamName(e)} ${e.source}`.toLowerCase().includes(search)));

  const activeRows = new Set(state.filtered.map(portKey));
  if (els.allPortsToggle.checked) {
    const hemispheres = h === "ALL" ? ["E","W"] : [h];
    const slices = s === "ALL" ? [...Array(ALL_SLICES).keys()] : [Number(s)];
    const banks = b === "ALL" ? [...Array(ALL_BANKS).keys()] : [Number(b)];
    const ports = p === "ALL" ? ["read","write"] : [p];
    state.rows = [];
    for (const side of hemispheres) for (const slice of slices) for (const bank of banks) for (const port of ports)
      state.rows.push(portKey({hemisphere:side,slice,bank,port}));
  } else state.rows = [...activeRows].sort(rowSort);

  state.rowEvents = new Map(state.rows.map(row => [row, []]));
  state.cycleEvents = new Map();
  for (const e of state.filtered) {
    state.rowEvents.get(portKey(e))?.push(e);
    if (!state.cycleEvents.has(e.cycle)) state.cycleEvents.set(e.cycle, []);
    state.cycleEvents.get(e.cycle).push(e);
  }
  for (const events of state.rowEvents.values()) events.sort((a,b)=>a.cycle-b.cycle);
  els.waveTrack.style.height = `${Math.max(state.rows.length * ROW_H, els.waveScroller.clientHeight)}px`;
  els.waveScroller.scrollTop = Math.min(els.waveScroller.scrollTop, Math.max(0,state.rows.length*ROW_H-els.waveScroller.clientHeight));
  els.eventReadout.value = state.filtered.length.toLocaleString();
  els.portReadout.value = state.rows.length.toLocaleString();
  render(); updateInspector();
}

function populateDimensions() {
  const slices = [...new Set(state.events.map(e=>e.slice))].sort((a,b)=>a-b);
  const banks = [...new Set(state.events.map(e=>e.bank))].sort((a,b)=>a-b);
  els.sliceFilter.innerHTML = '<option value="ALL">全部</option>' + slices.map(v=>`<option value="${v}">${v}</option>`).join("");
  els.bankFilter.innerHTML = '<option value="ALL">全部</option>' + banks.map(v=>`<option value="${v}">${v}</option>`).join("");
}

function loadEvents(events, name, alreadyNormalized = false) {
  stop(); state.events = (alreadyNormalized ? events : events.map(normalize)).filter(e=>Number.isFinite(e.cycle)); state.fileName = name;
  if (!state.events.length) { state.minCycle=0; state.maxCycle=16; }
  else {
    state.minCycle=Number.POSITIVE_INFINITY; state.maxCycle=Number.NEGATIVE_INFINITY;
    for (const event of state.events) { state.minCycle=Math.min(state.minCycle,event.cycle); state.maxCycle=Math.max(state.maxCycle,event.cycle); }
  }
  state.cycle=state.minCycle; state.selected=null; els.cycleInput.min=state.minCycle; els.cycleInput.max=state.maxCycle;
  els.fileName.textContent=name; els.traceSummary.textContent=`${name} · ${state.events.length.toLocaleString()} events · cycles ${state.minCycle}–${state.maxCycle}`;
  populateDimensions(); fit(); filterEvents();
}

function fit() { state.viewStart=state.minCycle; state.viewSpan=Math.max(8,state.maxCycle-state.minCycle+1); render(); }
function clampView() {
  state.viewSpan=Math.max(2,Math.min(Math.max(8,state.maxCycle-state.minCycle+1)*4,state.viewSpan));
  const pad=state.viewSpan*.2; state.viewStart=Math.max(state.minCycle-pad,Math.min(state.maxCycle+1-state.viewSpan+pad,state.viewStart));
}
function zoom(factor, anchor=0.5) { const at=state.viewStart+state.viewSpan*anchor; state.viewSpan*=factor; state.viewStart=at-state.viewSpan*anchor; clampView(); render(); }
function pan(cycles) { state.viewStart+=cycles; clampView(); render(); }

function resizeCanvas(canvas) {
  const rect=canvas.getBoundingClientRect(), dpr=window.devicePixelRatio||1;
  const w=Math.max(1,Math.floor(rect.width*dpr)), h=Math.max(1,Math.floor(rect.height*dpr));
  if (canvas.width!==w || canvas.height!==h) { canvas.width=w; canvas.height=h; }
  const ctx=canvas.getContext("2d"); ctx.setTransform(dpr,0,0,dpr,0,0); return {ctx,w:rect.width,h:rect.height};
}
function xFor(cycle,width) { return (cycle-state.viewStart)/state.viewSpan*width; }
function cycleFor(x,width) { return Math.floor(state.viewStart+x/width*state.viewSpan); }

function drawRuler() {
  const {ctx,w,h}=resizeCanvas(els.rulerCanvas); ctx.clearRect(0,0,w,h); ctx.fillStyle="#f3f6f7";ctx.fillRect(0,0,w,h);
  const rough=state.viewSpan/Math.max(2,Math.floor(w/90)); const power=10**Math.floor(Math.log10(Math.max(1,rough)));
  const step=[1,2,5,10].map(v=>v*power).find(v=>v>=rough)||10*power;
  const first=Math.ceil(state.viewStart/step)*step; ctx.font='10px Consolas, monospace';ctx.textAlign="center";ctx.fillStyle="#65737a";ctx.strokeStyle="#cbd3d6";
  for(let c=first;c<=state.viewStart+state.viewSpan;c+=step){const x=xFor(c,w);ctx.beginPath();ctx.moveTo(x,h-8);ctx.lineTo(x,h);ctx.stroke();ctx.fillText(String(c),x,h-12);}
  const cx=xFor(state.cycle,w);ctx.strokeStyle="#d24f45";ctx.beginPath();ctx.moveTo(cx,0);ctx.lineTo(cx,h);ctx.stroke();
}

function renderLabels(first,count) {
  els.resourceLabels.innerHTML="";
  for(let i=first;i<Math.min(state.rows.length,first+count);++i){
    const div=document.createElement("div");div.className="resource-label";div.style.top=`${i*ROW_H-els.waveScroller.scrollTop}px`;
    const row=state.rows[i], port=row.endsWith(".W")?"WRITE":"READ";div.innerHTML=`<span>${row}</span><span class="port-badge">${port}</span>`;els.resourceLabels.appendChild(div);
  }
}

function lowerBound(events, cycle) { let lo=0,hi=events.length;while(lo<hi){const m=(lo+hi)>>1;if(events[m].cycle<cycle)lo=m+1;else hi=m;}return lo; }
function drawWave() {
  els.waveCanvas.style.height=`${els.waveScroller.clientHeight}px`;
  const {ctx,w,h}=resizeCanvas(els.waveCanvas);ctx.clearRect(0,0,w,h);ctx.font='9px Consolas, monospace';
  const first=Math.floor(els.waveScroller.scrollTop/ROW_H), count=Math.ceil(h/ROW_H)+1;
  renderLabels(first,count);
  for(let v=0;v<count && first+v<state.rows.length;++v){
    const rowIndex=first+v,y=rowIndex*ROW_H-els.waveScroller.scrollTop;
    ctx.fillStyle=rowIndex%2?"#f7f9fa":"#ffffff";ctx.fillRect(0,y,w,ROW_H);ctx.strokeStyle="#e5e9eb";ctx.beginPath();ctx.moveTo(0,y+ROW_H-.5);ctx.lineTo(w,y+ROW_H-.5);ctx.stroke();
    const events=state.rowEvents.get(state.rows[rowIndex])||[];let ix=lowerBound(events,Math.floor(state.viewStart)-1);
    for(;ix<events.length;++ix){const e=events[ix];if(e.cycle>state.viewStart+state.viewSpan+1)break;if(e.cycle<state.viewStart-1)continue;
      const x=xFor(e.cycle,w), next=xFor(e.cycle+1,w), ew=Math.max(2,next-x-.5), [off,eh]=stageBand(e.stage);
      ctx.fillStyle=colorFor(e);ctx.fillRect(x,y+off,ew,eh);
      if(ew>18){ctx.fillStyle="#fff";ctx.textAlign="left";ctx.fillText(e.stage==="pipeline"?`t${e.tile}`:e.action,x+2,y+off+6);}
    }
  }
  const cx=xFor(state.cycle,w);if(cx>=0&&cx<=w){ctx.fillStyle="rgba(210,79,69,.10)";ctx.fillRect(cx,0,Math.max(2,w/state.viewSpan),h);ctx.strokeStyle="#d24f45";ctx.beginPath();ctx.moveTo(cx,0);ctx.lineTo(cx,h);ctx.stroke();}
}
function render() { clampView(); drawRuler(); drawWave(); els.viewportReadout.value=`${Math.floor(state.viewStart)} – ${Math.ceil(state.viewStart+state.viewSpan)}`; }

function selectEvent(e) { state.selected=e; updateEventDetail(); updateInspector(); render(); }
function updateEventDetail() {
  const e=state.selected;if(!e){els.eventDetail.innerHTML="<dt>状态</dt><dd>单击时间线或周期表中的事件</dd>";return;}
  const values=[["物理端口",portKey(e)],["Cycle",e.cycle],["阶段 / 行为",`${e.stage} / ${e.action}`],["指令",e.opcode||"—"],["Tile",e.tile<0?"—":e.tile],["SRAM row",e.address<0?"—":e.address],["Stream",streamName(e)],["SR column",e.sr_column<0?"—":e.sr_column],["Vector tag",e.vector_tag||"—"],["来源",e.source||"—"],["PC",e.pc<0?"—":e.pc],["IQ before → after",e.iq_before<0?"—":`${e.iq_before} → ${e.iq_after}`],["8-byte data",e.data_hex||"—"]];
  els.eventDetail.innerHTML=values.map(([k,v])=>`<dt>${k}</dt><dd>${escapeHtml(String(v))}</dd>`).join("");
}
function escapeHtml(s){return s.replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));}
function setCycle(cycle, keepVisible=true) {
  state.cycle=Math.max(state.minCycle,Math.min(state.maxCycle,Math.round(cycle)));els.cycleInput.value=state.cycle;
  if(keepVisible&&(state.cycle<state.viewStart||state.cycle>=state.viewStart+state.viewSpan)){state.viewStart=state.cycle-state.viewSpan*.25;clampView();}
  updateInspector();render();
}
function updateInspector() {
  els.inspectorCycle.value=state.cycle;const events=(state.cycleEvents.get(state.cycle)||[]).slice().sort((a,b)=>portKey(a).localeCompare(portKey(b))||a.stage.localeCompare(b.stage));
  els.cycleSummary.textContent=events.length?`${events.length} 个 MEM 行为；未列出的 ${Math.max(0,state.rows.length-new Set(events.map(portKey)).size)} 个显示端口为空闲。`:`该周期没有 MEM 行为，${state.rows.length} 个显示端口均为空闲。`;
  els.cycleTableBody.innerHTML=events.map((e,i)=>`<tr data-index="${i}" class="${e===state.selected?"selected":""}"><td>${portKey(e)}</td><td><span class="stage-pill" style="background:${colorFor(e)}">${e.stage}</span></td><td>${escapeHtml(e.action)}</td><td>${e.tile<0?"—":e.tile}</td><td>${e.address<0?"—":e.address} / ${streamName(e)}</td></tr>`).join("");
  els.cycleTableBody.querySelectorAll("tr").forEach(row=>row.addEventListener("click",()=>selectEvent(events[Number(row.dataset.index)])));
}

function start() { if(state.playing)return;state.playing=true;els.playButton.textContent="❚❚";els.playButton.classList.add("active");state.timer=setInterval(()=>{const step=Number(els.speedSelect.value);if(state.cycle+step>state.maxCycle){stop();setCycle(state.maxCycle);}else setCycle(state.cycle+step);},100); }
function stop(){state.playing=false;clearInterval(state.timer);state.timer=null;if(els.playButton){els.playButton.textContent="▶";els.playButton.classList.remove("active");}}

function loadFile(file) {
  stop();els.interactionState.textContent="Parsing…";state.events=[];
  const worker=createMemCsvWorker();const rows=[];
  worker.onmessage=message=>{const data=message.data;if(data.type==="batch"){for(const row of data.rows)rows.push(normalize(row));els.interactionState.textContent=`Parsing ${data.count.toLocaleString()}…`;}
    else if(data.type==="done"){disposeMemCsvWorker(worker);loadEvents(rows,file.name,true);els.interactionState.textContent="Ready";}
    else {disposeMemCsvWorker(worker);els.interactionState.textContent=data.message;alert(data.message);}};
  worker.onerror=error=>{disposeMemCsvWorker(worker);els.interactionState.textContent="Parse failed";alert(error.message);};worker.postMessage({file});
}

for(const select of [els.hemisphereFilter,els.sliceFilter,els.bankFilter,els.portFilter,els.stageFilter,els.allPortsToggle]) select.addEventListener("change",filterEvents);
let searchTimer;els.searchInput.addEventListener("input",()=>{clearTimeout(searchTimer);searchTimer=setTimeout(filterEvents,120);});
els.fileInput.addEventListener("change",()=>{if(els.fileInput.files[0])loadFile(els.fileInput.files[0]);});
els.sampleButton.addEventListener("click",()=>loadEvents(sampleEvents(),"built-in.mem.csv"));els.fitButton.addEventListener("click",fit);
els.zoomInButton.addEventListener("click",()=>zoom(.6));els.zoomOutButton.addEventListener("click",()=>zoom(1.67));
els.firstButton.addEventListener("click",()=>setCycle(state.minCycle));els.prevButton.addEventListener("click",()=>setCycle(state.cycle-1));els.playButton.addEventListener("click",()=>state.playing?stop():start());els.nextButton.addEventListener("click",()=>setCycle(state.cycle+1));els.lastButton.addEventListener("click",()=>setCycle(state.maxCycle));els.cycleInput.addEventListener("change",()=>setCycle(Number(els.cycleInput.value)));
els.waveScroller.addEventListener("scroll",drawWave,{passive:true});
els.waveScroller.addEventListener("wheel",event=>{if(event.ctrlKey){event.preventDefault();const rect=els.waveCanvas.getBoundingClientRect();zoom(event.deltaY>0?1.18:.85,(event.clientX-rect.left)/rect.width);}else if(event.shiftKey){event.preventDefault();pan(event.deltaY/Math.max(1,els.waveCanvas.clientWidth)*state.viewSpan);}}, {passive:false});
els.waveCanvas.addEventListener("pointerdown",event=>{els.waveCanvas.setPointerCapture(event.pointerId);state.drag={x:event.clientX,start:state.viewStart,moved:false};els.waveCanvas.classList.add("panning");});
els.waveCanvas.addEventListener("pointermove",event=>{if(!state.drag)return;const dx=event.clientX-state.drag.x;if(Math.abs(dx)>3)state.drag.moved=true;state.viewStart=state.drag.start-dx/els.waveCanvas.clientWidth*state.viewSpan;clampView();render();});
els.waveCanvas.addEventListener("pointerup",event=>{if(!state.drag)return;const moved=state.drag.moved;state.drag=null;els.waveCanvas.classList.remove("panning");if(moved)return;const rect=els.waveCanvas.getBoundingClientRect(),cycle=cycleFor(event.clientX-rect.left,rect.width);setCycle(cycle,false);const row=Math.floor((event.clientY-rect.top+els.waveScroller.scrollTop)/ROW_H),events=state.rowEvents.get(state.rows[row])||[];const hits=events.filter(e=>e.cycle===state.cycle);if(hits.length)selectEvent(hits.find(e=>e.stage==="sram")||hits[0]);});
for(const type of ["dragenter","dragover"]) document.addEventListener(type,event=>{event.preventDefault();els.dropOverlay.style.display="grid";});
document.addEventListener("dragleave",event=>{if(event.relatedTarget===null)els.dropOverlay.style.display="none";});document.addEventListener("drop",event=>{event.preventDefault();els.dropOverlay.style.display="none";const file=event.dataTransfer.files[0];if(file)loadFile(file);});
new ResizeObserver(()=>{els.waveTrack.style.height=`${Math.max(state.rows.length*ROW_H,els.waveScroller.clientHeight)}px`;render();}).observe(els.waveScroller);
document.addEventListener("keydown",event=>{if(event.target.matches("input,select"))return;if(event.code==="Space"){event.preventDefault();state.playing?stop():start();}else if(event.key==="ArrowLeft")setCycle(state.cycle-1);else if(event.key==="ArrowRight")setCycle(state.cycle+1);});

loadEvents(sampleEvents(),"built-in.mem.csv");
