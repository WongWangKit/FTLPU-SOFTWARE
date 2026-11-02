// Pure JS model/parser/UI-binding checks; no browser, network, or hardware run.
const fs=require('fs'),vm=require('vm'),path=require('path');
const file=path.resolve(__dirname,'../../..','lpu-operator-explorer.html');
const html=fs.readFileSync(file,'utf8'),els={};
const ctx2d=new Proxy({}, {get:()=>()=>{},set:()=>true});
const document={getElementById(id){return els[id]??={value:({search:'',mode:'demo',sl:'0',token:'0',speed:'500',memGroup:'6',fabricSelect:'E',physicalSL:'0',zoom:'fit'})[id]??'0',style:{},dataset:{},clientWidth:700,getContext:()=>ctx2d};}};
const ctx={document,console,clearInterval,setInterval,setTimeout,Blob,URL};vm.createContext(ctx);
for(const script of html.matchAll(/<script>([\s\S]*?)<\/script>/g))vm.runInContext(script[1],ctx);
vm.runInContext(`
function assert(v,msg){if(!v)throw Error(msg);}
// Every recipe render, and verify explicit input SL arrival skew.
for(const r of Object.values(recipes)){
 mode='demo';choose(r.id);
 for(const t of [minCycle,Math.floor(maxCycle/2),maxCycle])setCycle(t);
 if(!r.view&&!r.transfer){const first=events.filter(e=>e.stage===0&&e.unit==='MEM'&&e.token===0&&e.from==='MEM');for(let sl=0;sl<4;sl++)assert(first.some(e=>e.sl===sl&&e.start===sl),'SL skew '+r.id);}
}
assert(makeDemo(recipes.fu_Write).events.filter(e=>e.start===0).every(e=>e.unit==='SR'),'write starts at SR');
assert(makeDemo(recipes.fu_Read).events.filter(e=>e.start===0).every(e=>e.unit==='MEM'),'read starts at MEM');
// Every physical group, both endpoint types: exact hop timing and no SR collisions.
for(const unit of ['VXM','MXM'])for(let g=0;g<13;g++){
 const p=physicalStage({unit,lat:2,label:'test',formula:'x'},0,4,0,0,0,g);
 const origin=unit==='VXM'?g:g+1,target=unit==='VXM'?0:15,hop=Math.abs(origin-target),keys=new Set();
 for(const e of p.events){if(e.unit==='SR'){
  const key=[e.start,e.fabric,e.sl,e.column,e.flow,e.stream].join(':');assert(!keys.has(key),'SR collision');keys.add(key);
  const isInput=e.stream===16||e.stream===17,expected=isInput?e.token+e.sl+1+Math.abs(e.column-origin):e.token+e.sl+2+hop+2+Math.abs(e.column-target);
  assert(e.start===expected,'physical hop timing');
 }}
 assert(p.events.find(e=>e.unit===unit&&e.token===0&&e.sl===0).start===2+hop,'consumer latency');
}
mode='demo';choose('l2_norm');assert((document.getElementById('fabric').innerHTML.match(/data-sr=/g)||[]).length===1024,'32 streams x 2 directions x 16 columns');
assert(srFact({unit:'SR',resource:'SR.E.W16',detail:'column=6 tile=0 valid=0',sl:0}).valid===false,'explicit invalid SR');
assert(srFact({unit:'MEM',resource:'MEM.E.Read',detail:'stream=W16',sl:0})===null,'never infer SR validity from MEM');
// New SXM selection is separate from the legacy recipe and all runtime records.
document.getElementById('sxmVariant').value='wide';mode='demo';choose('transpose');
assert(maxCycle===133,'three wide tiles drain at 133');
assert(events.filter(e=>e.wide==='arrival').length===3*32*4*2,'all byte-plane arrivals');
assert(events.filter(e=>e.wide==='departure').length===3*32*4*2,'all byte-plane departures');
for(const e of events){
 if(e.wide==='arrival')assert(e.start===e.tile*32+e.index+e.sl,'wide input skew');
 if(e.wide==='departure')assert(e.start===e.tile*32+35+e.index+e.sl,'wide output skew');
}
for(let r=0;r<32;r++)for(let c=0;c<32;c++)assert(finalValue[r][c]===c*32+r,'wide transpose values');
for(let t=0;t<=133;t++)setCycle(t);
assert(document.getElementById('sxmPanel').innerHTML.includes('输入 0/96 B，输出 0/96 B'),'FIFO drain displayed');
document.getElementById('sxmVariant').value='legacy';choose('transpose');assert(!events.some(e=>e.wide),'legacy preserved');
assert(makeDemo(recipes.reshape).events.every(e=>e.unit==='CONTROL'),'reshape has no transfers');
assert(makeDemo(recipes.indexed_view).events.every(e=>e.unit==='CONTROL'),'view has no transfers');
// repeat2d interpreter versus independent full expansion at every cycle.
for(let skip of [0,1]){const p=normalizeRow({start:3,end:6,resource:'MEM.E.Read',detail:'addr=20',inner_count:3,inner_interval:4,outer_count:4,outer_interval:17,inner_stride:2,outer_stride:10,skip_first:skip,induction:'mem_address',base_delta:1});patterns=[p];for(let t=0;t<70;t++){const expected=[];for(let o=0;o<4;o++)for(let i=0;i<3;i++){if(skip&&o===0&&i===0)continue;let st=3+o*17+i*4;if(st<=t&&t<st+3)expected.push('addr='+(21+o*10+i*2));}assert(JSON.stringify(realAt(t).map(e=>e.detail))===JSON.stringify(expected),'repeat expansion cycle '+t);}}
// CSV quoted commas, escaped quotes, newline and BOM.
const parsed=parseCSV('\uFEFFstart,end,resource,detail\\r\\n0,1,VXM.C0,"a,b \\"\\"q\\"\\""\\r\\n');
assert(parsed.length===1&&parsed[0].detail==='a,b "q"','CSV quoting');
// All bundled traces normalize and render, using their original cycle bounds.
for(let k=0;k<DATA.traces.length;k++){mode='real';traceId=k;selected=DATA.traces[k].op;load();for(let t=minCycle;t<=maxCycle;t+=7)setCycle(t);assert(patterns.length>0,'trace empty');}
console.log('PASS: 37 recipes, 26 physical routes, 1024 SR cells, collision-free skew, explicit/unknown SR state, repeat2d, quoted CSV, six real trace windows.');
`,ctx);
