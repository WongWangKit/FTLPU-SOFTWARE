"""RMSNorm distributed16 output to token-lane QDQ, directly and through partial RoPE."""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
from deepseek_v41_swa_test import Model
from deepseek_v41_qkv_norm_test import run, trace_times
from deepseek_v41_qkv_reference import tensor, norm, rope, act_quant, dequant_act
from import_hf_decoder_layer import bf16_bits, read_bf16


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('build-dir','target-config','package','output-dir'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    meta=json.loads((a.package/'weights/provenance.json').read_text())
    gamma=tensor(a.package/'weights',meta,'kv_norm','BF16',(512,))
    x=np.load(a.package/'reference/kv_raw.npy')
    m=Model(out);m.input('x',x);m.input('gamma',gamma)
    y=m.op('norm','rms_norm',['x','gamma'],(32,512),'axis=1:i64,epsilon=1.000000e-20:f32')
    z=m.rotary('rope',m.reshape('head',y,(32,1,512)),1)
    z=m.reshape('rotated',z,(32,512))
    q=m.op('norm_quant','block_fp8_qdq',[y],(32,512))
    r=m.op('rope_quant','block_fp8_qdq',[z],(32,512))
    m.write([y,z,q,r])
    gold_norm=norm(x,gamma,1e-20);gold_rope=rope(gold_norm,64,10000)
    expected=[gold_norm,gold_rope,dequant_act(*act_quant(gold_norm)),dequant_act(*act_quant(gold_rope))]
    ext='.exe' if os.name=='nt' else '';exe=lambda area,name:a.build_dir.resolve()/area/(name+ext)
    env={k:v for k,v in os.environ.items() if not k.startswith('FTLPU_')}
    env.update(FTLPU_FILE_RUNNER_MODEL_SESSION='1',FTLPU_FILE_RUNNER_SXM_ARCHITECTURE='wide',
               FTLPU_GRAPH_PROGRESS='1',FTLPU_SESSION_PROGRESS='1')
    record=dict(status='running',scope=__doc__,raw_macro=True,command=[sys.executable,*sys.argv],
                limits=dict(relative_l2=.01,local_qdq_bit_mismatches=0))
    try:
        run([exe('compiler','ftlpu-compile'),'--input',out/'kernel.mlir','--input-stage','kernel',
             '--target-config',a.target_config.resolve(),'--icu-compression','raw-macro',
             '--output',out/'program.ftlpu'],out/'compile.log',env)
        run([exe('runtime','compiled_stage_file_runner'),out/'program.ftlpu',out/'inputs.txt',
             out/'actual',out/'runtime.pipeline.csv'],out/'execution.log',env)
        values=[read_bf16(out/f'actual.{i}.bin',(32,512)) for i in range(4)]
        errors=[float(np.linalg.norm(actual.astype('f8')-gold)/np.linalg.norm(gold))
                for actual,gold in zip(values,expected)]
        mismatches=[int(np.count_nonzero(bf16_bits(values[i+2])!=
                    bf16_bits(dequant_act(*act_quant(values[i]))))) for i in range(2)]
        prefix=int(np.count_nonzero(bf16_bits(values[0][:,:448])!=bf16_bits(values[1][:,:448])))
        record.update(relative_l2=errors,local_qdq_bit_mismatches=mismatches,
                      rope_prefix_bit_mismatches=prefix,timing=trace_times(out/'runtime.pipeline.csv'))
        assert all(e<.01 for e in errors) and mismatches==[0,0] and prefix==0,record
        record['status']='passed'
    except Exception as e:record.update(status='failed',error=str(e));raise
    finally:(out/'summary.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)


if __name__=='__main__':main()
