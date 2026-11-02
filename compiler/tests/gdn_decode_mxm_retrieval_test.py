"""Compare real raw-macro MXM and VXM State retrieval for a decode head."""
import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from qwen3_5_decode_gdn_attention_test import run
from qwen3_5_decode_gdn_batched_test import encode, read
from qwen3_5_qkv_conv_prefill_test import compare


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('compile','runner','target-config','output-dir'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(12832)
    q=rng.normal(0,.1,(1,128)).astype('f4');k=rng.normal(0,.1,(1,128)).astype('f4')
    q/=np.linalg.norm(q)*np.sqrt(128);k/=np.linalg.norm(k)
    values=[q,k,rng.normal(0,.03,(1,128)).astype('f4'),
            np.array([[.83]],'f4'),np.array([[.61]],'f4'),
            rng.normal(0,.02,(128,128)).astype('f4')]
    paths=[]
    for i,value in enumerate(values):
        path=out/f'input{i}.bin';path.write_bytes(encode(value,'bf16' if i<3 else 'f32'));paths.append(str(path))
    (out/'inputs.txt').write_text('\n'.join(paths)+'\n')
    source=Path(__file__).with_name('Inputs').joinpath('gated_delta_rule_seq1.kernel.mlir').read_text()
    source=source.replace('1x64','1x128').replace('1x2x','1x1x').replace('64x32','128x128')
    source=source.replace('heads = 2','heads = 1').replace('dim = 32','dim = 128')
    os.environ['FTLPU_FILE_RUNNER_MODEL_SESSION']='1'
    for mode in ('vxm','mxm'):
        kernel=out/f'{mode}.mlir';binary=out/f'{mode}.ftlpu'
        kernel.write_text(source.replace('module {','module attributes {ftlpu.vector_bf16_spill = true, '
            'ftlpu.common_barrier_protocol = true, ftlpu.gdn_decode_retrieval_mxm = '
            +str(mode=='mxm').lower()+'} {'))
        run([a.compile.resolve(),'--input',kernel,'--input-stage','kernel','--output',binary,
             '--target-config',a.target_config.resolve(),'--icu-compression','raw-macro'],out/f'{mode}.compile.log')
        run([a.runner.resolve(),binary,out/'inputs.txt',out/mode,out/f'{mode}.pipeline.csv'],out/f'{mode}.execution.log')
    errors={}
    for index,shape,dtype in ((0,(1,128),'bf16'),(1,(128,128),'f32')):
        errors[str(index)]=compare(read(out/f'mxm.{index}.bin',shape,dtype),read(out/f'vxm.{index}.bin',shape,dtype))
    assert all(v['nrmse']<.002 and v['cosine']>.99999 for v in errors.values()),errors
    assert 'MXM.' in (out/'mxm.pipeline.csv').read_text()
    (out/'summary.json').write_text(json.dumps(dict(status='passed',errors=errors),indent=2))
    print(json.dumps(errors,indent=2))


if __name__=='__main__':main()
