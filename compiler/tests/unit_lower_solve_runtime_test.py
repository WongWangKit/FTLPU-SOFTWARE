"""FP32 unit-lower solve through compiler-generated raw ICU and real SRAM."""
import argparse, csv, json, os, re, sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from qwen3_5_logical_graph import LogicalGraph
from qwen3_5_decode_gdn_attention_test import run

def audit_forward_solve(out, heads, dim, cycle_range=None, node_name='solve'):
    """Audit actual useful issues and gap-free long-row vector blocks."""
    spans=[]
    with (out/'runtime.pipeline.csv').open(newline='',encoding='utf-8') as stream:
        for r in csv.DictReader(stream):
            if r['resource']!='VXM.C0' or not r['detail'].startswith('mul chain_depth=4'):continue
            for outer in range(int(r['outer_count'])):
                for inner in range(int(r['inner_count'])):
                    if outer==inner==0 and int(r['skip_first']):continue
                    shift=outer*int(r['outer_interval'])+inner*int(r['inner_interval'])
                    lo,hi=int(r['start'])+shift,int(r['end'])+shift
                    if cycle_range is not None:lo,hi=max(lo,cycle_range[0]),min(hi,cycle_range[1])
                    if lo<hi:spans.append((lo,hi))
    spans.sort()
    active=sum(b-a for a,b in spans)
    assert active==(heads//2)*(dim//32)*sum(range(1,33)),active
    assert all(a[1]<=b[0] for a,b in zip(spans,spans[1:])), 'overlapping VXM issues'
    lines=(out/'decoder_layer.schedule.mlir').read_text().splitlines()
    windows=[];domains=[]
    for line in lines:
        fields={k:int(v) for k,v in re.findall(r'(\w+) = (-?\d+) : i64',line)}
        if 'name = "'+node_name+'.unit_lower_solve.forward_row"' in line:
            windows.append((fields['start']-1,fields['end']))
        if 'ftlpu.schedule.vxm' in line and 'opcode = "multiply"' in line and fields.get('queue')==0:
            domains.append((fields['cycle'],fields['repeat_count'],fields['repeat_interval']))
    windows.sort();assert len(windows)==heads//2*32,len(windows)
    for index,(begin,end) in enumerate(windows):
        issues=sorted(t+k*ii for t,n,ii in domains if begin<=t<end for k in range(n))
        row=index%32
        assert len(issues)==(row+1)*(dim//32),(row,len(issues))
        if row>=8:assert issues==list(range(issues[0],issues[0]+len(issues))), (row,'vector-block bubble')
    duration=(cycle_range[1]-cycle_range[0]) if cycle_range else spans[-1][1]-spans[0][0]
    return dict(paired_c0_active_cycles=active,measured_cycles=duration,
                paired_c0_utilization=active/duration,long_row_vector_block_idle_cycles=0,
                coefficient_banks=2)

def source(heads,dim,kind='unit_lower_solve'):
    shapes=(((heads*32,32),'f32'),((heads*32,dim),'f32')) if kind in ('unit_lower_solve','unit_lower_solve_blocked') else (((32,heads*dim),'bf16'),((32,heads),'f32'))
    a,b=[LogicalGraph.type(*s) for s in shapes];o=LogicalGraph.type((heads*32,dim),'f32')
    return f'''module {{ func.func @stage(%a: {a}, %b: {b}) -> {o} {{
      %o = ftlpu.kernel.vector_alu %a, %b {{kind = "{kind}", use_immediate = false,
        immediate = 0.0 : f32, reduce = false}} : ({a}, {b}) -> {o}
      return %o : {o}
    }} }}'''

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('build-dir','target-config','output-dir'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--heads',type=int,default=2);p.add_argument('--dim',type=int,default=32)
    p.add_argument('--produced-rhs',action='store_true',help='Use an FP32 ALU producer for B, as in nonzero-State chunk correction')
    a=p.parse_args();out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    build=a.build_dir.resolve();exe=lambda area,n:build/area/(n+('.exe' if sys.platform=='win32' else ''))
    heads,dim=a.heads,a.dim;rng=np.random.default_rng(35261103)
    lower=np.tril(rng.normal(0,.05,(heads,32,32)).astype('f4'),-1)
    b=rng.normal(size=(heads,32,dim)).astype('f4')
    # Preserve low FP32 mantissa bits and exercise a nontrivial row dependency.
    b[0,0]=-128-np.arange(1,dim+1,dtype='f4')*2**-16
    graph=LogicalGraph()
    rhs=b.reshape(-1,dim)
    if a.produced_rhs:
        rhs=graph.vector_tiles('corrected_rhs',[(rhs,'f32')],lambda v:v.add(0,kind='multiply',immediate=1))
    d=graph.run('solve',source(heads,dim),[(lower.reshape(-1,32),'f32'),(rhs,'f32')],[((heads*32,dim),'f32')])[0]
    logical=graph.write(out/'logical',[d]);kernel=out/'kernel.mlir';binary=out/'program.ftlpu'
    run([exe('compiler','ftlpu-graph-compile'),'--input',logical,'--output',kernel,'--emit-kernel','--target-config',a.target_config.resolve()],out/'import.log')
    kernel.write_text(kernel.read_text().replace('module {','module attributes {ftlpu.sxm_wide = true, ftlpu.common_barrier_protocol = true, ftlpu.vector_bf16_spill = true, ftlpu.vector_fp32_ingress = true} {',1))
    run([exe('compiler','ftlpu-compile'),'--input',kernel,'--input-stage','kernel','--output',binary,'--target-config',a.target_config.resolve(),'--icu-compression','raw-macro'],out/'compile.log')
    if a.produced_rhs:
        plan=json.loads((out/'graph_plan.json').read_text())
        producer=next(node for node in plan['nodes'] if node['name']=='corrected_rhs')
        assert producer['direct_output_layouts'][0]['kind']=='fp32_quad_planar',producer
    os.environ['FTLPU_FILE_RUNNER_MODEL_SESSION']='1';os.environ['FTLPU_FILE_RUNNER_SXM_ARCHITECTURE']='wide'
    run([exe('runtime','compiled_stage_file_runner'),binary,out/'inputs.txt',out/'actual',out/'runtime.pipeline.csv'],out/'execution.log')
    actual=np.fromfile(out/'actual.0.bin','<f4').reshape(heads,32,dim)
    ref=np.linalg.solve(np.eye(32,dtype='f8')[None]+lower.astype('f8'),b.astype('f8'))
    np.testing.assert_allclose(actual,ref,rtol=2e-5,atol=5e-5)
    np.testing.assert_array_equal(actual[:,0],b[:,0])
    residual=(np.eye(32)[None]+lower)@actual-b
    summary=dict(status='passed',max_abs=float(np.max(np.abs(actual-ref))),relative_residual=float(np.linalg.norm(residual)/np.linalg.norm(b)))
    summary['scheduling']=audit_forward_solve(out,heads,dim)
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(summary)
if __name__=='__main__':main()
