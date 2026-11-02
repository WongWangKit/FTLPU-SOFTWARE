// Physical coordinates are modeled SR-column hops, not silicon millimeters.
function physicalStage(step,start,nt,tokenBase,stage,alu,group){
 const events=[],toMxm=step.unit==='MXM',origin=toMxm?group+1:group,target=toMxm?15:0,
       flow=toMxm?'E':'W',back=toMxm?'W':'E',hop=Math.abs(origin-target),fabric=$('fabricSelect').value||'E';
 let end=start;
 for(let token=0;token<nt;token++)for(let sl=0;sl<4;sl++){
  const row=tokenBase+token,t=start+token+sl;
  const put=(at,span,unit,detail,extra={})=>{events.push({start:at,end:at+span,unit,detail,sl,token:row,stage,alu,fabric,group,...extra});end=Math.max(end,at+span);};
  put(t,1,'MEM',`G${group} slice ${group*4}/${group*4+1} bank0 读出 BF16 两个 byte plane；token ${row}`,{from:'MEM',to:'SR',slice:group*4,action:'read'});
  const route=(at,a,b,dir,ids,label)=>{
   const sign=b>=a?1:-1;for(let j=0;j<=Math.abs(b-a);j++)for(const stream of ids)
    put(at+j,1,'SR',`${label} ${dir}${stream} · column=${a+j*sign} · SL${sl} · token ${row} · valid=1`,{column:a+j*sign,flow:dir,stream,valid:true,from:'SR',to:step.unit});
  };
  route(t+1,origin,target,flow,[16,17],'输入');
  const compute=t+2+hop;
  put(compute,step.lat,step.unit,`${step.label} · ${step.formula} · 输入经过 ${hop} 个 SR hop`);
  route(compute+step.lat,target,origin,back,[18,19],'输出');
  put(compute+step.lat+hop+1,1,'MEM',`G${group} slice ${group*4}/${group*4+1} bank1 写回；token ${row}`,{from:'SR',to:'MEM',slice:group*4,action:'write'});
 }
 return {events,end};
}

function srFact(event){
 if(event.unit!=='SR')return null;
 if(event.column!==undefined)return event;
 // Optional explicit SR observation rows can be imported in pipeline CSV format.
 // Resource SR.E.E16; detail column=6 tile=0 valid=1 (E fabric, East stream 16).
 const m=(event.resource||'').match(/^SR\.([EW])\.([EW])(\d+)$/),col=event.detail.match(/\bcolumn=(\d+)/),valid=event.detail.match(/\bvalid=([01])/);
 if(!m||!col||!valid||event.sl===null)return null;
 const c=+col[1],s=+m[3];if(c>15||s>31||event.sl>3)return null;
 return {...event,fabric:m[1],flow:m[2],stream:s,column:c,valid:valid[1]==='1'};
}

function drawFabric(active){
 const hw=DATA.hardware,columns=hw.stream_topology.fabric_template.columns,
   fabric=$('fabricSelect').value||'E',sl=+$('physicalSL').value||0,zoom=$('zoom').value||'fit',svg=$('fabric');
 svg.style.width=zoom==='fit'?'100%':zoom+'px';svg.style.minWidth=zoom==='fit'?'0':zoom+'px';
 const currentStage=mode==='demo'?steps.find(s=>s.start<=cycle&&cycle<s.end):null,
   fullyKnown=mode==='demo'&&(recipes[selected].view||['VXM','MXM'].includes(currentStage?.unit)||(selected==='transpose'&&$('sxmVariant').value==='wide'));
 const facts=active.map(srFact).filter(x=>x&&x.fabric===fabric&&x.sl===sl),lookup=new Map();
 for(const e of facts)lookup.set(`${e.flow}:${e.stream}:${e.column}`,e);
 const x0=485,dx=82,x=c=>x0+c*dx,top=76,bottom=664,dy=10,unitY=431;
 const text=(a,b,t,size=12,color='#dce8f8',attrs='')=>`<text x="${a}" y="${b}" font-size="${size}" fill="${color}" ${attrs}>${esc(t)}</text>`;
 let out='<defs><pattern id="srUnknown" width="6" height="6" patternUnits="userSpaceOnUse"><rect width="6" height="6" fill="#1d263d"/><path d="M0 6L6 0" stroke="#45435f" stroke-width="1"/></pattern></defs>';
 out+=text(12,24,`${fabric} fabric · SL${sl} · 每个亮格 = 该物理列中一个 stream 的 8-lane valid 数据`,17);
 out+=text(12,46,'SR 邻列间距 = 配置中的 1 hop / 1 cycle；slice 与 ALU 图形宽度不是毫米尺寸。',12,'#a4b5cc');
 for(let col=0;col<16;col++){
  out+=`<path d="M${x(col)} 60V996" stroke="#26364e" stroke-dasharray="3 5"/>`;
  out+=text(x(col),63,columns[col].replace('mem.',''),10,'#9ab1cc','text-anchor="middle"');
 }
 for(const [dir,ybase,color,label] of [['E',top,'#52dfbc','东向 E0…E31 →'],['W',bottom,'#81b2ff','← 西向 W0…W31']]){
  out+=text(300,ybase-10,label,13,color);
  for(let stream=0;stream<32;stream++){
   const y=ybase+stream*dy;
   out+=text(x0-43,y+7,dir+stream,9,color);
   out+=`<path d="M${x(0)} ${y+3}H${x(15)}" stroke="#263e4b" stroke-width="1"/>`;
   for(let col=0;col<16;col++){
    const e=lookup.get(`${dir}:${stream}:${col}`),known=!!e||fullyKnown,valid=e?.valid===true,fill=valid?color:known?'#0c1724':'url(#srUnknown)';
    let desc=`${fabric} fabric / SL${sl} / ${dir}${stream} / ${columns[col]}：${valid?'valid':known?'无数据':'未知，日志未记录'}`;
    if(e)desc+=' · '+e.detail;
    out+=`<rect data-sr="${dir}${stream}" x="${x(col)-19}" y="${y}" width="38" height="7" rx="1.5" fill="${fill}" stroke="${valid?color:'#304155'}" stroke-width=".5"><title>${esc(desc)}</title></rect>`;
   }
  }
 }
 const hit=(unit,group=null)=>active.filter(e=>{
  if(e.unit!==unit)return false;
  if(mode==='demo')return (e.fabric||fabric)===fabric&&(group===null||e.group===group);
  if(!(e.resource||'').includes('.'+fabric+'.'))return false;
  if(group!==null){const slice=e.detail.match(/\bslice=(\d+)/);return slice&&Math.floor(+slice[1]/4)===group;}
  return true;
 });
 out+=text(22,414,'VXM · 16 ALU（内部不增加 SR 列）',13,colors.VXM);
 for(let c=0;c<16;c++){
  const xx=20+(c%8)*54,yy=unitY+Math.floor(c/8)*74,ev=active.filter(e=>e.unit==='VXM'&&(mode==='demo'?e.alu===c:new RegExp('\\.C'+c+'(?:\\.|$)').test(e.resource||'')));
  out+=`<g data-unit="VXM" style="cursor:pointer"><rect x="${xx}" y="${yy}" width="47" height="60" rx="4" fill="${ev.length?'#623e4e':'#1b293d'}" stroke="${ev.length?colors.VXM:'#3b4a64'}"/>`+text(xx+7,yy+23,'C'+c,12)+text(xx+5,yy+44,ev.length?'active':'·',9)+'</g>';
 }
 out+=`<path d="M440 470H${x(0)} M440 550H${x(0)}" stroke="#b2869f" stroke-width="2"/>`;
 out+=text(20,603,'输入 W@b0；输出 E@b0',12,'#a6b7d0');
 for(let g=0;g<13;g++){
  const xx=x(g)+10,w=dx-20,ev=hit('MEM',g),fill=ev.length?'#254e47':'#182c3a';
  out+=`<g data-unit="MEM" style="cursor:pointer"><rect x="${xx}" y="${unitY}" width="${w}" height="159" rx="4" fill="${fill}" stroke="${ev.length?colors.MEM:'#3a526b'}"/>`+text(xx+7,unitY+18,'G'+g,13,colors.MEM);
  for(let j=0;j<4;j++){
   const slice=g*4+j,yy=unitY+38+j*27;
   out+=text(xx+4,yy,'M'+slice,10);
   for(let bank=0;bank<2;bank++){
    const bankHit=ev.some(e=>mode==='demo'?(e.slice===slice||e.slice+1===slice)&&((e.action==='write'?1:0)===bank):new RegExp('\\bslice='+slice+'\\b').test(e.detail)&&new RegExp('\\bbank='+bank+'\\b').test(e.detail));
    out+=`<rect x="${xx+30+bank*13}" y="${yy-9}" width="10" height="13" rx="1" fill="${bankHit?'#52dfbc':'#101d2c'}"><title>MEM slice ${slice} bank ${bank}：${bankHit?'有活动':'未记录活动'}</title></rect>`;
   }
  }
  out+='</g>'+text(xx+2,unitY+177,`${g*4}–${g*4+3}`,10,'#91a8c1');
 }
 // C2C is attached to b13; SXM spans column 14->15; MXM terminates at 15.
 for(const [unit,xx,w,label,port] of [['C2C',x(13)+9,64,'C2C','b13'],['SXM',x(14)+9,64,'SXM','14 ↔ 15'],['MXM',x(15)+25,125,'MXM','in E15 / out W15']]){
  const ev=hit(unit);out+=`<g data-unit="${unit}" style="cursor:pointer"><rect x="${xx}" y="${unitY}" width="${w}" height="159" rx="5" fill="${ev.length?'#3e4156':'#202c40'}" stroke="${ev.length?colors[unit]:'#4a5772'}"/>`+text(xx+7,unitY+23,label,14,colors[unit])+text(xx+4,unitY+147,port,9);
  if(unit==='SXM')out+=text(xx+4,unitY+130,mode==='demo'&&$('sxmVariant').value==='wide'?'大转置*':'现有',10,colors.SXM);
  for(let s=0;s<4;s++)out+=text(xx+6,unitY+47+s*21,'SL'+s+(ev.some(e=>e.sl===s)?' ●':''),11);
  out+='</g>';
 }
 out+=text(20,1024,'上 / 下 SR 的 East/West 是流向，不是半球名称；使用 Fabric 选择器查看两个半球。b0 的跨半球 passive bridge 延迟 1 拍。',13,'#a6b7d0');
 svg.innerHTML=out;
 $('physicalStatus').textContent=`13 MEM groups × 4 slices / hemisphere；16 SR 列 × 32 streams × 2 方向。当前 SL${sl}：${facts.filter(e=>e.valid).length} 个有效位置。${fullyKnown?'亮=示意调度中有数据，暗=该示意调度无数据。':'斜纹=未知；没有完整 SR valid trace，不能当成空闲。'}`;
 if(mode==='demo')$('mapNote').textContent='VXM/MXM 示例使用选定 MEM group 的真实端口和每 hop 1 拍；示例 tensor 的地址、stream ID（16/17 输入，18/19 输出）仍是教学配置，不是实际 compiler 分配。';
}

$('fabricSelect').onchange=()=>mode==='demo'?load():render();
$('memGroup').onchange=()=>mode==='demo'?load():render();
$('physicalSL').onchange=render;
$('zoom').onchange=render;
