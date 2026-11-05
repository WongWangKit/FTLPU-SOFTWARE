"""Weighted per-head K^T D, using existing final-row coefficients."""
import argparse,csv,json,os,re,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from qwen3_5_logical_graph import LogicalGraph
from qwen3_5_compiled_layer import bf16
from qwen3_5_decode_gdn_attention_test import run


def source(h,k,v,with_output=False):
    kt,rt,dt,wt,st=(LogicalGraph.type(shape,dtype) for shape,dtype in (
        ((32,h*k),'bf16'),((h*32,32),'bf16'),((h*32,v),'f32'),((h*k,32),'bf16'),((h*k,v),'f32')))
    ht=LogicalGraph.type((h*32,32),'f32')
    extra=f', %h: {ht}' if with_output else ''
    returns=f'({dt}, {st})' if with_output else st
    output=f'''%o = ftlpu.kernel.vector_alu %h, %d {{kind = "head_matmul", mxm_precision = "bf16", use_immediate = false, immediate = 0.0 : f32, reduce = false}} : ({ht}, {dt}) -> {dt}''' if with_output else ''
    ret=f'%o, %s : {dt}, {st}' if with_output else f'%s : {st}'
    return f'''module {{ func.func @stage(%k: {kt}, %r: {rt}, %d: {dt}{extra}) -> {returns} {{
      %w = ftlpu.kernel.vector_alu %k, %r {{kind = "head_last_row_scale_transpose", use_immediate = false, immediate = 0.0 : f32, reduce = false}} : ({kt}, {rt}) -> {wt}
      %s = ftlpu.kernel.vector_alu %w, %d {{kind = "head_cross_product", use_immediate = false, immediate = 0.0 : f32, reduce = false}} : ({wt}, {dt}) -> {st}
      {output}
      return {ret}
    }} }}'''


def reference(k,r,d):
    weighted=bf16(k*r[-1][:,:,None])
    truncated=(d.astype('f4').view('u4')&np.uint32(0xffff0000)).view('f4')
    return weighted.astype('f8').transpose(0,2,1)@truncated.astype('f8')


def audit(path,h,k,v,interval=None,with_output=False):
    with Path(path).open() as f:rows=list(csv.DictReader(f))
    if interval:rows=[r for r in rows if interval[0]<=int(r['start'])<interval[1]]
    counts={};compute_bursts={};load_overlap={}
    for unit in ('MXM.E0.Compute','MXM.W0.Compute'):
        counts[unit]=sum((int(r['end'])-int(r['start']))*int(r['inner_count'])*int(r['outer_count']) for r in rows if r['resource']==unit)
        assert counts[unit]==h//2*(v//32)*(k+(32 if with_output else 0)),counts
        spans=[]
        for row in rows:
            if row['resource']!=unit:continue
            for outer in range(int(row['outer_count'])):
                for inner in range(int(row['inner_count'])):
                    offset=outer*int(row['outer_interval'])+inner*int(row['inner_interval'])
                    spans.append((int(row['start'])+offset,int(row['end'])+offset))
        merged=[]
        for a,b in sorted(spans):
            assert not merged or a>=merged[-1][1],(unit,merged[-1],(a,b))
            if merged and a==merged[-1][1]:merged[-1][1]=b
            else:merged.append([a,b])
        if with_output:
            jobs=h//2*(v//32)
            assert len(merged)==(jobs+1)//2,(unit,merged)
            assert all(b-a==min(2,jobs-2*i)*(k+32) for i,(a,b) in enumerate(merged)),(unit,merged)
        compute_bursts[unit]=dict(continuous_spans=merged,
            idle_cycles=merged[-1][1]-merged[0][0]-counts[unit],
            max_gap=max((merged[i+1][0]-merged[i][1] for i in range(len(merged)-1)),default=0))
        def events(resource):
            result=[]
            for row in rows:
                if row['resource']!=resource:continue
                buffer=int(re.search(r'buffer=(\d+)',row['detail']).group(1))
                for outer in range(int(row['outer_count'])):
                    for inner in range(int(row['inner_count'])):
                        shift=outer*int(row['outer_interval'])+inner*int(row['inner_interval'])
                        result.append((int(row['start'])+shift,int(row['end'])+shift,buffer))
            return result
        overlap=0
        for a,b,load_buffer in events(unit.replace('.Compute','.Load')):
            for c,d,compute_buffer in events(unit):
                common=max(0,min(b,d)-max(a,c))
                assert not common or load_buffer!=compute_buffer,(unit,'live weight overwrite',a,b,c,d)
                overlap+=common
        if with_output and h//2*(v//32)>2:assert overlap>0,(unit,'missing weight prefetch overlap')
        load_overlap[unit]=overlap
    writes=sum((int(r['end'])-int(r['start']))*int(r['inner_count'])*int(r['outer_count']) for r in rows if r['resource'].startswith('MEM.') and r['resource'].endswith('.Write'))
    assert writes==h*(k+(32 if with_output else 0))*(v//32)*4,writes
    loads={unit:sum((int(r['end'])-int(r['start']))*int(r['inner_count'])*int(r['outer_count']) for r in rows if r['resource']==unit) for unit in ('MXM.E0.Load','MXM.W0.Load')}
    assert all(n==h//2*(v//32)*32 for n in loads.values()),loads
    transposes={unit:sum(int(r['inner_count'])*int(r['outer_count']) for r in rows if r['resource']==unit) for unit in ('SXM.E.Transpose','SXM.W.Transpose')}
    assert all(n==h//2*(v//32)*(1+k//32) for n in transposes.values()),transposes
    return dict(mxm_active_cycles=counts,weight_load_cycles=loads,sxm_transpose_tiles=transposes,
                mxm_compute_bursts=compute_bursts,load_compute_overlap_cycles=load_overlap,
                mem_write_vectors=writes,intermediate_mem_spill=False,shared_d_tile=with_output)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('build-dir','target-config','output-dir'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--heads',type=int,default=4);p.add_argument('--key-dim',type=int,default=128);p.add_argument('--value-dim',type=int,default=128)
    p.add_argument('--with-output',action='store_true')
    a=p.parse_args();h,k,v=a.heads,a.key_dim,a.value_dim;out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    build=a.build_dir.resolve();exe=lambda area,n:build/area/(n+('.exe' if sys.platform=='win32' else ''))
    rng=np.random.default_rng(35261104);keys=bf16(rng.normal(size=(h,32,k)).astype('f4'));coeff=bf16(rng.uniform(.1,1,(32,h,32)).astype('f4'));values=rng.normal(size=(h,32,v)).astype('f4')
    grouped=np.concatenate((values[::2],values[1::2]),axis=0)
    matrices=rng.normal(size=(h,32,32)).astype('f4')
    inputs=[(keys.transpose(1,0,2).reshape(32,-1),'bf16'),(coeff.reshape(-1,32),'bf16'),(grouped.reshape(-1,v),'f32')]
    outputs=[((h*k,v),'f32')]
    if a.with_output:
        inputs.append((np.concatenate((matrices[::2],matrices[1::2]),axis=0).reshape(-1,32),'f32'))
        outputs.insert(0,((h*32,v),'f32'))
    graph=LogicalGraph();results=graph.run('weighted_product',source(h,k,v,a.with_output),inputs,outputs)
    logical=graph.write(out/'logical',results);kernel=out/'kernel.mlir';binary=out/'program.ftlpu'
    run([exe('compiler','ftlpu-graph-compile'),'--input',logical,'--output',kernel,'--emit-kernel','--target-config',a.target_config.resolve()],out/'import.log')
    kernel.write_text(kernel.read_text().replace('module {','module attributes {ftlpu.sxm_wide = true, ftlpu.common_barrier_protocol = true, ftlpu.common_transport_overlap = false, ftlpu.vector_fp32_ingress = true} {',1))
    run([exe('compiler','ftlpu-compile'),'--input',kernel,'--input-stage','kernel','--output',binary,'--target-config',a.target_config.resolve(),'--icu-compression','raw-macro','--mxm-execution','ws'],out/'compile.log')
    os.environ['FTLPU_FILE_RUNNER_MODEL_SESSION']='1';os.environ['FTLPU_FILE_RUNNER_SXM_ARCHITECTURE']='wide'
    run([exe('runtime','compiled_stage_file_runner'),binary,out/'inputs.txt',out/'actual',out/'runtime.pipeline.csv'],out/'execution.log')
    actual=np.fromfile(out/f'actual.{int(a.with_output)}.bin','<f4').reshape(2,h//2,k,v).transpose(1,0,2,3).reshape(h,k,v)
    ref=reference(keys,coeff,values);err=float(np.linalg.norm(actual-ref)/np.linalg.norm(ref))
    assert np.isfinite(actual).all() and err<1e-6,err
    summary=dict(status='passed',nrmse=err,**audit(out/'runtime.pipeline.csv',h,k,v,with_output=a.with_output))
    if a.with_output:
        actual_o=np.fromfile(out/'actual.0.bin','<f4').reshape(2,h//2,32,v).transpose(1,0,2,3).reshape(h,32,v)
        trunc=lambda x:(x.view('u4')&np.uint32(0xffff0000)).view('f4').astype('f8')
        ref_o=trunc(matrices)@trunc(values)
        error_o=float(np.linalg.norm(actual_o-ref_o)/np.linalg.norm(ref_o))
        assert error_o<1e-6,error_o
        summary['output_nrmse']=error_o
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(summary)


if __name__=='__main__':main()
