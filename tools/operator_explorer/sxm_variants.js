// Experimental architecture only; legacy compiler/runtime traces stay unchanged.
function makeWideTranspose(){
 const n=32,tiles=3,events=[],fabric=$('fabricSelect').value||'E';
 const memory=Array.from({length:2},()=>Array.from({length:n},()=>Array(n).fill(null)));
 let align=0,skew=0;
 for(let t=0;t<134;t++){
  const put=(detail,extra)=>events.push({start:t,end:t+1,stage:0,unit:'SXM',sl:null,fabric,detail,...extra});
  for(let sl=0;sl<4;sl++){
   const incoming=t-sl,tile=Math.floor(incoming/32),row=incoming%32;
   if(incoming>=0&&tile<tiles){
    align+=8;
    for(const stream of [16,17])put(`输入 T${tile}[${row},${sl*8}:${sl*8+8}] → 输入对齐 FIFO`,{unit:'SR',sl,token:row,column:14,flow:'E',stream,valid:true,wide:'arrival',tile,index:row});
   }
   const outgoing=t-35-sl,ot=Math.floor(outgoing/32),col=outgoing%32;
   if(outgoing>=0&&ot<tiles){
    skew-=8;
    for(const stream of [18,19])put(`输出 T${ot}[${sl*8}:${sl*8+8},${col}]，转置结果行 ${col}`,{unit:'SR',sl,token:col,column:15,flow:'E',stream,valid:true,wide:'departure',tile:ot,index:col});
   }
  }
  const readIndex=t-35,rt=Math.floor(readIndex/32),col=readIndex%32;
  if(readIndex>=0&&rt<tiles){
   for(let r=0;r<32;r++){if(memory[rt%2][r][col]!==rt*1024+r*32+col)throw Error('大 transpose 读出错误');memory[rt%2][r][col]=null;}
   skew+=32;put(`buffer ${rt%2?'B':'A'} 列 ${col} → 输出错拍 FIFO；32 个 bank 各读 1 元素`,{wide:'read',tile:rt,index:col,buf:rt%2});
  }
  const writeIndex=t-3,wt=Math.floor(writeIndex/32),row=writeIndex%32;
  if(writeIndex>=0&&wt<tiles){
   for(let c=0;c<32;c++){if(memory[wt%2][row][c]!==null)throw Error('大 transpose 覆盖存活数据');memory[wt%2][row][c]=wt*1024+row*32+c;}
   align-=32;put(`对齐后的 T${wt} 行 ${row} → buffer ${wt%2?'B':'A'}；bank=(r+c)%32`,{wide:'write',tile:wt,index:row,buf:wt%2});
  }
  if(align<0||align>48||skew<0||skew>48)throw Error('大 transpose FIFO 超容量');
 }
 if(align||skew||memory.flat(2).some(x=>x!==null))throw Error('大 transpose 未排空');
 const output=Array.from({length:32},(_,r)=>Array.from({length:32},(_,c)=>c*32+r));
 return {events,steps:[{label:'跨 SL 32×32 · 连续 3 tiles',unit:'SXM',formula:'输入对齐 → ping-pong bank → 输出错拍；无独立 permute 指令',start:0,end:134}],snapshots:[output],output};
}

function renderSxmPanel(){
 const isWide=$('sxmVariant').value==='wide';$('sxmVariant').disabled=mode==='real';
 if(mode==='real'){$('sxmPanel').innerHTML='<p class="tiny">真实回放保持原有 SXM 指令语义；新架构尚无对应 runtime trace。</p>';return;}
 if(selected!=='transpose'){$('sxmPanel').innerHTML='<p class="tiny">所选 SXM 方案：'+(isWide?'新大 transpose（待实现）':'现有 SXM')+'。点击“查看所选 SXM 的 transpose”进入比较。</p>';return;}
 if(!isWide){$('sxmPanel').innerHTML='<p><b>现有方案保留：</b>4 × 局部 8×8 transpose，再由 block permute 重组。CModel TransposeBank 数据载荷为 512 B（BF16）。下方仍是原有机制演示，不是精确 runtime pass 数。</p>';return;}
 $('badge').textContent='新 SXM 方案模型 · 未接入 compiler / CModel · 不代表现有映射';
 $('formula').textContent='32×32 transpose：输入每 SL 错一拍 → 3/2/1/0 拍对齐 → 32-bank ping-pong → 0/1/2/3 拍输出错拍';
 $('description').textContent='一次输入一个逻辑向量的启动间隔为 1 拍，四个 superlane 依次到达；连续 3 个 tile。替代局部 transpose + 独立 block permute，内部仍需跨 SL 路由。';
 const past=events.filter(e=>e.start<=cycle),writes=past.filter(e=>e.wide==='write'),reads=past.filter(e=>e.wide==='read'),arrivals=past.filter(e=>e.wide==='arrival'&&e.stream===16),departures=past.filter(e=>e.wide==='departure'&&e.stream===18);
 const inputBytes=(arrivals.length*8-writes.length*32)*2,outputBytes=(reads.length*32-departures.length*8)*2;
 let grids='';
 for(let b=0;b<2;b++){
  const wr=writes.filter(e=>e.buf===b),rd=reads.filter(e=>e.buf===b),last=wr.at(-1),tile=last?.tile,fill=last?.index??-1,drained=rd.filter(e=>e.tile===tile).at(-1)?.index??-1;
  const live=tile===undefined?0:(fill+1)*(32-drained-1);
  let cells='';for(let r=0;r<32;r++)for(let c=0;c<32;c++){
   const currentWrite=wr.some(e=>e.start===cycle&&e.index===r),currentRead=rd.some(e=>e.start===cycle&&e.index===c),valid=r<=fill&&c>drained;
   cells+=`<span style="background:${currentRead?'#ffbd69':currentWrite?'#52dfbc':valid?'#466b94':'#0b1728'};aspect-ratio:1" title="${tile===undefined?'未使用':'T'+tile} [${r},${c}]；bank ${(r+c)%32}, addr ${r}；${valid?'有效':'空'}"></span>`;
  }
  grids+=`<div><b>buffer ${b?'B':'A'} · ${live*2} / 2048 B</b><div style="display:grid;grid-template-columns:repeat(32,1fr);gap:1px;max-width:320px;margin-top:8px">${cells}</div></div>`;
 }
 $('sxmPanel').innerHTML=`<p><b>总数据存储 4288 B</b> = 2 × 2048 B 矩阵 + 输入对齐 96 B + 输出错拍 96 B。首个输出 SL0 在 cycle 35，SL3 在 38，tile 间隔 32 拍。</p><p>拍末 FIFO：输入 ${inputBytes}/96 B，输出 ${outputBytes}/96 B。绿=本拍写；橙=本拍读；蓝=有效数据。</p><div style="display:grid;grid-template-columns:1fr 1fr;gap:18px;max-width:700px">${grids}</div><p class="tiny">容量不含控制/ECC/额外 SRAM 读延迟；此模型从 SXM 的 column 14 输入端口开始，输出在 column 15，未杜撰上游 MEM 的读取。SR rail 只显示所选 SL。单个 32×32 tile 比现有 4 个 8×8 block 覆盖更多数据，不能用这两个示例的周期直接比较性能。</p>`;
 const ref=Array.from({length:32},(_,r)=>Array.from({length:32},(_,c)=>r*32+c));
 $('tensors').innerHTML=displayTensor('T0 输入，值=r×32+c',ref)+displayTensor('T0 转置结果（独立参考）',finalValue);
 $('tensorNote').textContent='数值索引用于验证转置；buffer 中连续处理 T0/T1/T2。输入输出 BF16 按两个 byte-plane stream 表示。';
 $('mapNote').textContent='新 SXM 从 E16/E17@column14 接收，向 E18/E19@column15 输出。内部输入对齐和输出错拍延迟均显式计入；不是当前硬件的 trace。';
}
$('sxmVariant').onchange=()=>{stop();if(selected==='transpose')load();else render();};
$('showTranspose').onclick=()=>{stop();mode='demo';$('mode').value=mode;choose('transpose');};
