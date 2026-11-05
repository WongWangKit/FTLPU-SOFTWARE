/* Pure data model: the browser view and the Node verification share this. */
(function (root) {
  "use strict";
  const floatBits = value => {
    const buffer=new ArrayBuffer(4), view=new DataView(buffer);
    view.setUint32(0,Number(value),true);return view.getFloat32(0,true);
  };
  function prepare(data) {
    const events=data.events.map(r=>Array.isArray(r)?{cycle:Number(r[0]),unit:r[1],resource:r[2],stage:r[3],opcode:r[4],pc:r[5],detail:r[6],payload:r[7]}:r)
      .sort((a,b)=>a.cycle-b.cycle || a.unit.localeCompare(b.unit));
    const T=data.tokens||5,E=data.experts||4,K=data.top_k||2;
    const count=data.physical_cycles || events.reduce((n,e)=>Math.max(n,e.cycle+1),0);
    const byCycle=new Map();
    for(const e of events){if(!byCycle.has(e.cycle))byCycle.set(e.cycle,[]);byCycle.get(e.cycle).push(e);}
    const state={scores:Array.from({length:T},()=>Array(E).fill(null)),
      top:Array.from({length:T},()=>Array(K).fill(null)),selected:Array.from({length:T},()=>Array(K).fill(null)),
      counts:Array(E).fill(null),offsets:Array(E).fill(null),inverse:Array.from({length:T},()=>Array(K).fill(null)),compare:null};
    const snapshots=[{cycle:-1,state:JSON.parse(JSON.stringify(state))}];
    const valid=(token,slot)=>token>=0 && token<T && slot>=0 && slot<K;
    for(const event of events){
      if(event.resource!=="RCU.Core" || event.stage!=="internal")continue;
      const p=event.payload,t=Number(p.token),e=Number(p.expert),slot=Number(p.slot);let changed=true;
      switch(event.detail){
      case "ChoiceRead":if(t<T&&e<E)state.scores[t][e]=floatBits(p.score_bits);break;
      case "SelectClear":if(t<T&&slot===0)state.top[t]=Array(K).fill(null);break;
      case "Compare": {
        const previous=state.compare;
        const incomingValid=slot===0 || (previous && previous.token===t && previous.expert===e && previous.slot===slot-1
          ? (previous.exchanged?previous.candidate!==null:previous.incomingValid):true);
        if(valid(t,slot))state.top[t][slot]=p.exchanged==="1"||p.candidate_valid==="1"?{
          expert:Number(p.winner_id),score:floatBits(p.exchanged==="1"?p.score_bits:p.candidate_bits)}:null;
        state.compare={cycle:event.cycle,token:t,expert:e,slot,incomingValid,incoming:Number(p.incoming_id),score:floatBits(p.score_bits),
          candidate:p.candidate_valid==="1"?Number(p.candidate_id):null,candidateScore:p.candidate_valid==="1"?floatBits(p.candidate_bits):null,
          exchanged:p.exchanged==="1",winner:Number(p.winner_id)};break;}
      case "StoreId":if(valid(t,slot))state.selected[t][slot]={expert:Number(p.value),weight:null};break;
      case "StoreRaw":if(valid(t,slot)&&state.selected[t][slot])state.selected[t][slot].weight=floatBits(p.raw_bits);break;
      case "ClearCounts":if(Number(p.position)<E)state.counts[Number(p.position)]=0;break;
      case "CountWrite":if(e<E)state.counts[e]=Number(p.value);break;
      case "PrefixOffset":if(e<E)state.offsets[e]=Number(p.value);break;
      case "AssignInverse":if(valid(t,slot))state.inverse[t][slot]=Number(p.value);break;
      default:changed=false;
      }
      if(changed)snapshots.push({cycle:event.cycle,state:JSON.parse(JSON.stringify(state))});
    }
    function routeAt(cycle){let lo=0,hi=snapshots.length;while(lo<hi){const mid=(lo+hi)>>1;if(snapshots[mid].cycle<=cycle)lo=mid+1;else hi=mid;}return snapshots[Math.max(0,lo-1)].state;}
    function at(cycle){return byCycle.get(cycle)||[];}
    function activity(cycle,unit){
      const rows=at(cycle).filter(e=>e.unit===unit);if(!rows.length)return "idle";
      const work=rows.some(e=>!["wait","active","credit_wait","fetch","buffered"].includes(e.stage)
        && !(e.stage==="internal" && ["Io","Acquire","AddressDrain","Collect","Compute","Commit","Drain"].includes(e.detail)));
      return work?"active":"waiting";
    }
    const first=predicate=>events.find(predicate)?.cycle??0;
    const jumps={load:first(e=>e.resource==="DMA.ProgramLoader"&&e.stage==="activate"),
      router:first(e=>e.unit==="MXM"&&e.opcode==="COMPUTE"&&e.resource.startsWith("MXM.W0")),
      select:first(e=>e.resource==="RCU.Core"&&e.detail==="Compare"),
      plan:first(e=>e.resource==="RCU.Core"&&e.opcode==="ROUTE_PLAN"&&e.stage==="internal"),
      input:first(e=>e.resource==="RCU.Core"&&e.detail==="AddressExpert"),
      weights:first(e=>e.unit==="DMA"&&e.resource!=="DMA.ProgramLoader"&&e.stage==="activate"),
      expert:first(e=>e.unit==="MXM"&&e.opcode==="COMPUTE"&&e.resource.startsWith("MXM.E0")),
      combine:first(e=>e.resource==="RCU.Combine")};
    return {data,events,at,activity,routeAt,jumps,count,T,E,K};
  }
  function parseCSV(text){
    const rows=[];let row=[],field="",quoted=false;
    for(let i=0;i<text.length;++i){const ch=text[i];if(ch==='"'){if(quoted&&text[i+1]==='"'){field+='"';++i;}else quoted=!quoted;}
      else if(ch===','&&!quoted){row.push(field);field="";}else if(ch==='\n'&&!quoted){row.push(field.replace(/\r$/,""));if(row.some(Boolean))rows.push(row);row=[];field="";}else field+=ch;}
    if(field||row.length){row.push(field.replace(/\r$/,""));rows.push(row);}
    const header=rows.shift();if(!header || !header.includes("payload") || !header.includes("opcode"))throw new Error("需要 instructions.cycles.csv");
    const events=rows.map(r=>{const o=Object.fromEntries(header.map((k,i)=>[k,r[i]??""]));return [Number(o.cycle),o.unit,o.resource,o.stage,o.opcode,o.pc,o.detail,JSON.parse(o.payload||"{}")];});
    const scores=events.filter(e=>e[2]==="RCU.Core"&&e[6]==="ChoiceRead");
    return {events,tokens:scores.length?Math.max(...scores.map(e=>Number(e[7].token)))+1:5,
      experts:scores.length?Math.max(...scores.map(e=>Number(e[7].expert)))+1:4,
      top_k:Math.max(1,...events.filter(e=>e[2]==="RCU.Core"&&e[6]==="Compare").map(e=>Number(e[7].slot)+1))};
  }
  function unpack(packed){
    const strings=packed.strings;
    const payloads=packed.payloads.map(flat=>{const result={};for(let i=0;i<flat.length;i+=2)result[strings[flat[i]]]=strings[flat[i+1]];return result;});
    return {...packed.meta,events:packed.events.map(row=>[row[0],...row.slice(1,7).map(i=>strings[i]),payloads[row[7]]])};
  }
  const api={prepare,parseCSV,floatBits,unpack};root.MoeCycleModel=api;
  if(typeof module!=="undefined")module.exports=api;
})(typeof globalThis!=="undefined"?globalThis:this);
