"""Independent Q/K contraction and inclusive causal epilogue on raw macro ICU."""
import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np

from weighted_gram_runtime_test import scaled_weighted_gram_source, weighted_cross_gram_source, read_head_groups
from causal_exp_difference_runtime_test import causal_exp_source, read_transposed
from qwen3_5_logical_graph import LogicalGraph
from qwen3_5_compiled_layer import bf16
from qwen3_5_decode_gdn_attention_test import run
from qwen3_5_qkv_conv_prefill_test import compare


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('build-dir','target-config','output-dir'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument("--row-scale",action="store_true")
    args=parser.parse_args()
    build=args.build_dir.resolve()
    exe=lambda area,name:build/area/(name+('.exe' if sys.platform=='win32' else ''))
    cases=[]
    for heads,dim in ((2,32),(4,128)):
        out=args.output_dir.resolve()/f'h{heads}_d{dim}'
        out.mkdir(parents=True,exist_ok=True)
        rng=np.random.default_rng(35261103+heads)
        q,k=[bf16(rng.normal(size=(32,heads*dim)).astype('f4')/np.sqrt(dim)) for _ in range(2)]
        g=-np.cumsum(rng.uniform(.01,.1,(32,heads)).astype('f4'),axis=0)
        graph=LogicalGraph()
        decay=graph.run('causal_decay',causal_exp_source(32,heads,True),[(g,'f32')],[((heads*32,32),'bf16')])[0]
        beta=rng.uniform(.1,.9,(32,heads)).astype('f4')
        if args.row_scale:
            result=graph.run('scaled_gram',scaled_weighted_gram_source(heads,dim),
                [(k,'bf16'),(decay,'bf16'),(beta,'f32'),(np.tri(32,k=-1,dtype='f4'),'bf16')],[((heads*32,32),'f32')])[0]
        else:
            result=graph.run('weighted_cross_gram',weighted_cross_gram_source(heads,dim),
                [(q,'bf16'),(k,'bf16'),(decay,'bf16')],[((heads*32,32),'f32')])[0]
        logical=graph.write(out/'logical',[decay,result])
        kernel=out/'kernel.mlir';binary=out/'program.ftlpu'
        run([exe('compiler','ftlpu-graph-compile'),'--input',logical,'--output',kernel,'--emit-kernel',
             '--target-config',args.target_config.resolve()],out/'import.log')
        kernel.write_text(kernel.read_text().replace('module {','module attributes {ftlpu.vector_bf16_spill = true, '
            'ftlpu.vector_fp32_ingress = true, ftlpu.sxm_wide = true, ftlpu.common_barrier_protocol = true} {',1))
        run([exe('compiler','ftlpu-compile'),'--input',kernel,'--input-stage','kernel','--output',binary,
             '--target-config',args.target_config.resolve(),'--icu-compression','raw-macro','--mxm-execution','ws'],out/'compile.log')
        os.environ['FTLPU_FILE_RUNNER_MODEL_SESSION']='1'
        os.environ['FTLPU_FILE_RUNNER_SXM_ARCHITECTURE']='wide'
        run([exe('runtime','compiled_stage_file_runner'),binary,out/'inputs.txt',out/'actual',out/'runtime.pipeline.csv'],out/'execution.log')
        decay_actual=read_transposed(out/'actual.0.bin',heads).transpose(2,0,1)
        actual=read_head_groups(out/'actual.1.bin',heads)
        queries=q.reshape(32,heads,dim).transpose(1,0,2)
        keys=k.reshape(32,heads,dim).transpose(1,0,2)
        expected=(queries@keys.transpose(0,2,1))*decay_actual
        if args.row_scale:
            expected=(keys@keys.transpose(0,2,1))*decay_actual*beta.T[:,:,None]*np.tri(32,k=-1)[None]
        errors=[compare(actual[h],expected[h]) for h in range(heads)]
        assert all(e['nrmse']<.01 and e['cosine']>.9999 for e in errors),errors
        assert np.all(actual[:,np.triu_indices(32,1)[0],np.triu_indices(32,1)[1]]==0)
        np.testing.assert_allclose(np.diagonal(actual,axis1=1,axis2=2),np.diagonal(expected,axis1=1,axis2=2),rtol=.009,atol=2e-5)
        assert ('weighted_gram.beta_broadcast' if args.row_scale else 'cross_gram.mxm_to_vxm') in (out/'decoder_layer.schedule.mlir').read_text()
        cases.append(dict(heads=heads,dim=dim,errors=errors))
    (args.output_dir/'summary.json').write_text(json.dumps(dict(status='passed',cases=cases),indent=2))
    print(cases)


if __name__=='__main__':main()
