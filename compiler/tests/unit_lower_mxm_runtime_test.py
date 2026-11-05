"""Nilpotent chunk32 solve on real MXM, with FP64 residual/reference checks."""
import argparse, json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from qwen3_5_logical_graph import LogicalGraph
from qwen3_5_decode_gdn_attention_test import run

def source(heads,dim,kind):
    a=LogicalGraph.type((heads*32,32),'f32')
    b=LogicalGraph.type((heads*32,dim),'f32')
    return f'''module {{ func.func @stage(%a: {a}, %b: {b}) -> {b} {{
      %o = ftlpu.kernel.vector_alu %a, %b {{kind = "{kind}", use_immediate = false,
        immediate = 0.0 : f32, reduce = false}} : ({a}, {b}) -> {b}
      return %o : {b}
    }} }}'''

def polynomial(lower):
    t=np.eye(32)[None]-lower.astype('f8');p=lower.astype('f8')@lower.astype('f8')
    for i in range(4):
        t=t+p@t
        if i!=3:p=p@p
    return t

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('build-dir','target-config','output-dir'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--fixture-dir',type=Path)
    p.add_argument('--heads',type=int,default=2);p.add_argument('--dim',type=int,default=32)
    a=p.parse_args();out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    build=a.build_dir.resolve();exe=lambda area,n:build/area/(n+('.exe' if sys.platform=='win32' else ''))
    heads,dim=a.heads,a.dim;rng=np.random.default_rng(35261103)
    lower=np.tril(rng.normal(0,.05,(heads,32,32)).astype('f4'),-1)
    b=rng.normal(size=(heads,32,dim)).astype('f4')
    if a.fixture_dir:
        lower=np.fromfile(a.fixture_dir/'actual.8.bin','<f4').reshape(heads,32,32)
        b=np.fromfile(a.fixture_dir/'actual.9.bin','<f4').reshape(heads,32,dim)
    eye=np.broadcast_to(np.eye(32,dtype='f4'),(heads,32,32)).copy()
    graph=LogicalGraph()
    t=graph.run('inverse',source(heads,32,'unit_lower_inverse'),[(lower.reshape(-1,32),'f32'),(eye.reshape(-1,32),'f32')],[((heads*32,32),'f32')])[0]
    d=graph.run('matmul',source(heads,dim,'head_matmul'),[(t,'f32'),(b.reshape(-1,dim),'f32')],[((heads*32,dim),'f32')])[0]
    logical=graph.write(out/'logical',[t,d]);kernel=out/'kernel.mlir';binary=out/'program.ftlpu'
    run([exe('compiler','ftlpu-graph-compile'),'--input',logical,'--output',kernel,'--emit-kernel','--target-config',a.target_config.resolve()],out/'import.log')
    kernel.write_text(kernel.read_text().replace('module {','module attributes {ftlpu.sxm_wide = true, ftlpu.common_barrier_protocol = true, ftlpu.common_transport_overlap = false, ftlpu.vector_bf16_spill = true, ftlpu.vector_fp32_ingress = true} {',1))
    run([exe('compiler','ftlpu-compile'),'--input',kernel,'--input-stage','kernel','--output',binary,'--target-config',a.target_config.resolve(),'--icu-compression','raw-macro','--mxm-execution','ws'],out/'compile.log')
    os.environ['FTLPU_FILE_RUNNER_MODEL_SESSION']='1';os.environ['FTLPU_FILE_RUNNER_SXM_ARCHITECTURE']='wide'
    run([exe('runtime','compiled_stage_file_runner'),binary,out/'inputs.txt',out/'actual',out/'runtime.pipeline.csv'],out/'execution.log')
    actual_t=np.fromfile(out/'actual.0.bin','<f4').reshape(heads,32,32)
    actual=np.fromfile(out/'actual.1.bin','<f4').reshape(heads,32,dim)
    ref=np.linalg.solve(np.eye(32)[None]+lower.astype('f8'),b.astype('f8'))
    oracle_t=polynomial(lower)
    np.testing.assert_allclose(oracle_t@b,ref,rtol=1e-12,atol=1e-12)
    err=float(np.linalg.norm(actual-ref)/np.linalg.norm(ref))
    residual=float(np.linalg.norm((np.eye(32)[None]+lower)@actual-b)/np.linalg.norm(b))
    t_error=float(np.linalg.norm(actual_t-oracle_t)/np.linalg.norm(oracle_t))
    summary=dict(status='passed',nrmse=err,relative_residual=residual,inverse_nrmse=t_error,
        precision='FP32 workspaces/accumulation; three BF16 components; six MXM partial products',heads=heads,dim=dim)
    limit=.003 if a.fixture_dir else 1e-6
    assert err<limit and residual<limit and t_error<limit,summary
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(summary)
if __name__=='__main__':main()
