"""MXM Gram with streamed VXM weighting; no Gram spill or host arithmetic."""
import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from qwen3_5_logical_graph import LogicalGraph
from qwen3_5_compiled_layer import bf16
from qwen3_5_decode_gdn_attention_test import run
from qwen3_5_qkv_conv_prefill_test import compare


def weighted_gram_source(heads,dim,transposed=False):
    weight=((32*heads,32),'bf16') if transposed else ((1024,heads),'f32')
    k,r,a=(LogicalGraph.type(shape,dtype) for shape,dtype in (
        ((32,heads*dim),'bf16'),weight,((heads*32,32),'f32')))
    return f'''module {{ func.func @stage(%k: {k}, %w: {r}) -> {a} {{
      %a = ftlpu.kernel.vector_alu %k, %w {{kind = "weighted_gram", use_immediate = false,
        immediate = 0.0 : f32, reduce = false}} : ({k}, {r}) -> {a}
      return %a : {a}
    }} }}'''


def reference(k,weights):
    heads=weights.shape[-1]
    keys=k.reshape(32,heads,-1).transpose(1,0,2)
    gram=keys@keys.transpose(0,2,1)
    return gram*weights.reshape(32,32,heads).transpose(2,0,1)


def read_head_groups(path, heads):
    # Native output ABI: even heads on East followed by odd heads on West.
    # This is output unpacking only, not runtime model arithmetic.
    return np.fromfile(path,'<f4').reshape(2,heads//2,32,32).transpose(1,0,2,3).reshape(heads,32,32)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('build-dir','target-config','output-dir'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();build=a.build_dir.resolve()
    exe=lambda area,name:build/area/(name+('.exe' if sys.platform=='win32' else ''))
    cases=[]
    for heads,dim in ((2,32),(4,128)):
        out=a.output_dir.resolve()/f'h{heads}_d{dim}';out.mkdir(parents=True,exist_ok=True)
        rng=np.random.default_rng(heads*1000+dim)
        k=rng.normal(size=(32,heads,dim)).astype('f4')
        k=bf16(k/np.linalg.norm(k,axis=-1,keepdims=True)).reshape(32,-1)
        g=-np.cumsum(rng.uniform(.01,.2,(32,heads)).astype('f4'),axis=0)
        weights=np.where(np.tri(32,dtype=bool)[:,:,None],np.exp(g[:,None]-g[None,:]),0).astype('f4').reshape(1024,heads)
        graph=LogicalGraph()
        value=graph.run('weighted_gram',weighted_gram_source(heads,dim),[(k,'bf16'),(weights,'f32')],[((heads*32,32),'f32')])[0]
        logical=graph.write(out/'logical',[value]);kernel=out/'kernel.mlir';binary=out/'program.ftlpu'
        run([exe('compiler','ftlpu-graph-compile'),'--input',logical,'--output',kernel,'--emit-kernel',
             '--target-config',a.target_config.resolve()],out/'import.log')
        kernel.write_text(kernel.read_text().replace('module {','module attributes {ftlpu.vector_bf16_spill = true, '
            'ftlpu.sxm_wide = true, ftlpu.common_barrier_protocol = true} {',1))
        run([exe('compiler','ftlpu-compile'),'--input',kernel,'--input-stage','kernel','--output',binary,
             '--target-config',a.target_config.resolve(),'--icu-compression','raw-macro','--mxm-execution','ws'],out/'compile.log')
        os.environ['FTLPU_FILE_RUNNER_MODEL_SESSION']='1';os.environ['FTLPU_FILE_RUNNER_SXM_ARCHITECTURE']='wide'
        run([exe('runtime','compiled_stage_file_runner'),binary,out/'inputs.txt',out/'actual',out/'runtime.pipeline.csv'],out/'execution.log')
        actual=read_head_groups(out/'actual.0.bin',heads)
        expected=reference(k,weights)
        per_head=[compare(actual[h],expected[h]) for h in range(heads)]
        assert all(e['nrmse']<.01 and e['cosine']>.9999 for e in per_head),per_head
        assert np.all(actual[:,np.triu_indices(32,1)[0],np.triu_indices(32,1)[1]]==0)
        cases.append(dict(heads=heads,dim=dim,errors=per_head))
    (a.output_dir/'summary.json').write_text(json.dumps(dict(status='passed',cases=cases),indent=2));print(cases)


if __name__=='__main__':main()
