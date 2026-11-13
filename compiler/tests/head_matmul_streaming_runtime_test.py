"""Direct BF16 head matmul: SXM-to-MXM routing, continuous compute, no scratch."""
import argparse
import csv
import json
import os
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from qwen3_5_logical_graph import LogicalGraph
from qwen3_5_decode_gdn_attention_test import run
from unit_lower_mxm_runtime_test import source
from import_hf_decoder_layer import bf16


def audit(path, heads, dim, interval=None, reduction=32):
    with Path(path).open() as f: rows=list(csv.DictReader(f))
    if interval: rows=[r for r in rows if interval[0]<=int(r['start'])<interval[1]]
    assert not any(r['resource'].startswith('VXM.') for r in rows)
    active={}
    for unit in ('MXM.E0.Compute','MXM.W0.Compute'):
        spans=[]
        for r in rows:
            if r['resource']!=unit:continue
            for outer in range(int(r['outer_count'])):
                for inner in range(int(r['inner_count'])):
                    delta=outer*int(r['outer_interval'])+inner*int(r['inner_interval'])
                    spans.append((int(r['start'])+delta,int(r['end'])+delta))
        spans.sort()
        count=sum(b-a for a,b in spans)
        assert count==(heads//2)*(dim//32)*reduction,(unit,count)
        assert all(spans[i][1]==spans[i+1][0] for i in range(len(spans)-1)),(unit,spans)
        active[unit]=dict(start=spans[0][0],end=spans[-1][1],active_cycles=count,idle_cycles=0)
    writes=[r for r in rows if r['resource'].startswith('MEM.') and r['resource'].endswith('.Write')]
    words=sum((int(r['end'])-int(r['start']))*int(r['inner_count'])*int(r['outer_count']) for r in writes)
    # Four FP32 byte planes, both hemispheres. Only the final product is stored.
    assert words==heads*32*(dim//32)*4,(words,heads,dim)
    return dict(mxm_compute=active,mem_write_vectors=words,transpose_mem_spill=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('build-dir','target-config','output-dir'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--heads',type=int,default=4);p.add_argument('--dim',type=int,default=128)
    p.add_argument('--k',type=int,default=32,help='K>32 exercises native BF16 ACC retention (heads=2, dim=32)')
    a=p.parse_args();out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    build=a.build_dir.resolve();exe=lambda area,n:build/area/(n+('.exe' if sys.platform=='win32' else ''))
    h,d,k=a.heads,a.dim,a.k;rng=np.random.default_rng(35261104)
    lhs=rng.normal(size=(h,32,k)).astype('f4');rhs=rng.normal(size=(h,k,d)).astype('f4')
    graph=LogicalGraph()
    kernel_source=source(h,d,'head_matmul').replace('kind = "head_matmul",','kind = "head_matmul", mxm_precision = "bf16",')
    dtype='f32'
    if k!=32:
        if h!=2 or d!=32 or k<=0 or k%32:raise ValueError('Native K test requires heads=2, dim=32, K32')
        lhs=bf16(lhs);rhs=bf16(rhs);dtype='bf16'
        at=f'tensor<{h*32}x{k}xbf16>';bt=f'tensor<{h*k}x{d}xbf16>';ot=f'tensor<{h*32}x{d}xf32>'
        kernel_source=f'''module {{ func.func @stage(%a: {at}, %b: {bt}) -> {ot} {{
          %r = ftlpu.kernel.vector_alu %a, %b {{kind = "head_matmul", mxm_precision = "bf16", use_immediate = false, immediate = 0.0 : f32, reduce = false}} : ({at}, {bt}) -> {ot}
          return %r : {ot} }} }}'''
    output=graph.run('direct_matmul',kernel_source,[(lhs.reshape(-1,k),dtype),(rhs.reshape(-1,d),dtype)],[((h*32,d),'f32')])[0]
    logical=graph.write(out/'logical',[output]);kernel=out/'kernel.mlir';binary=out/'program.ftlpu'
    run([exe('compiler','ftlpu-graph-compile'),'--input',logical,'--output',kernel,'--emit-kernel','--target-config',a.target_config.resolve()],out/'import.log')
    kernel.write_text(kernel.read_text().replace('module {','module attributes {ftlpu.sxm_wide = true, ftlpu.common_barrier_protocol = true, ftlpu.common_transport_overlap = false, ftlpu.vector_fp32_ingress = true} {',1))
    run([exe('compiler','ftlpu-compile'),'--input',kernel,'--input-stage','kernel','--output',binary,'--target-config',a.target_config.resolve(),'--icu-compression','raw-macro','--mxm-execution','ws'],out/'compile.log')
    os.environ['FTLPU_FILE_RUNNER_MODEL_SESSION']='1';os.environ['FTLPU_FILE_RUNNER_SXM_ARCHITECTURE']='wide'
    run([exe('runtime','compiled_stage_file_runner'),binary,out/'inputs.txt',out/'actual',out/'runtime.pipeline.csv'],out/'execution.log')
    actual=np.fromfile(out/'actual.0.bin','<f4').reshape(h,32,d)
    trunc=lambda x:(x.view('u4')&np.uint32(0xffff0000)).view('f4').astype('f8')
    ref=trunc(lhs)@trunc(rhs);error=float(np.linalg.norm(actual-ref)/np.linalg.norm(ref))
    assert np.isfinite(actual).all() and error<1e-6,error
    summary=dict(status='passed',nrmse=error,heads=h,dim=d,k=k,**audit(out/'runtime.pipeline.csv',h,d,reduction=k))
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(summary)


if __name__=='__main__':main()
