"""Native MXM hemisphere-interleaved results feeding VXM reduction and token-lane QDQ."""
import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np
from deepseek_v41_swa_test import Model
from deepseek_v41_qkv_norm_test import run, trace_times
from deepseek_v41_qkv_reference import tensor, act_quant, dequant_act, project, scales
from import_hf_decoder_layer import bf16_bits, read_bf16


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('build-dir','target-config','package','output-dir'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    directory=a.package/'weights';meta=json.loads((directory/'provenance.json').read_text())
    w=tensor(directory,meta,'wo_a','F8_E4M3',(8192,4096))[:128,:64].copy()
    s=tensor(directory,meta,'wo_a_scale','F8_E8M0',(256,128))[:4,:2].copy()
    q,scale=act_quant(np.random.default_rng(41111).normal(size=(32,64)).astype('f4'))
    x=dequant_act(q,scale);expected=project(q,scale,w,s)
    m=Model(out);m.input('x',x);m.input('w',w.T.copy(),'f8E4M3FN')
    y=m.op('projection','matmul',['x','w'],(32,128),'m=32:i64,n=128:i64,k=64:i64,unit="MXM",rhs_scale=1.0:f32,rhs_block_scales=array<f32:'+
        ','.join(f'{v:.9e}' for v in scales(s).ravel())+'>',generic=True)
    reduced=m.op('sum','vector_alu',[y,y],(32,1),
        'kind="multiply",use_immediate=true,immediate=1.0:f32,reduce=true,ftlpu.vector_require_fp32',dtype='f32')
    quantized=m.op('requantized','block_fp8_qdq',[y],(32,128))
    m.write([y,reduced,quantized])
    env={k:v for k,v in os.environ.items() if not k.startswith('FTLPU_')}
    env.update(FTLPU_FILE_RUNNER_MODEL_SESSION='1',FTLPU_FILE_RUNNER_SXM_ARCHITECTURE='wide',FTLPU_GRAPH_PROGRESS='1',FTLPU_SESSION_PROGRESS='1')
    ext='.exe' if os.name=='nt' else '';exe=lambda area,name:a.build_dir.resolve()/area/(name+ext)
    record=dict(status='running',scope='projection-to-reduction and projection-to-QDQ transport primitive; fixture activation already QDQ',raw_macro=True,command=[sys.executable,*sys.argv])
    try:
        run([exe('compiler','ftlpu-compile'),'--input',out/'kernel.mlir','--input-stage','kernel','--target-config',a.target_config.resolve(),
             '--icu-compression','raw-macro','--output',out/'program.ftlpu'],out/'compile.log',env)
        run([exe('runtime','compiled_stage_file_runner'),out/'program.ftlpu',out/'inputs.txt',out/'actual',out/'runtime.pipeline.csv'],out/'execution.log',env)
        run([exe('runtime','ftlpu_icu_program_export'),out/'actual.runtime.ftlpu',out/'runtime_icu_programs','--pre-execution-dir',out/'actual.pre_execution'],out/'export.log',env)
        mismatch=int(np.count_nonzero(bf16_bits(read_bf16(out/'actual.0.bin',expected.shape))!=bf16_bits(expected)))
        actual=np.fromfile(out/'actual.1.bin','<f4');gold=expected.astype('f8').sum(1)
        error=float(np.linalg.norm(actual-gold)/np.linalg.norm(gold))
        quant_gold=dequant_act(*act_quant(expected))
        quant_mismatch=int(np.count_nonzero(bf16_bits(read_bf16(out/'actual.2.bin',expected.shape))!=bf16_bits(quant_gold)))
        record.update(bf16_mismatches=mismatch,reduction_relative_l2=error,
                      requantized_bf16_mismatches=quant_mismatch,timing=trace_times(out/'runtime.pipeline.csv'))
        assert mismatch==0 and quant_mismatch==0 and error<1e-5,record
        record['status']='passed'
    except Exception as e:record.update(status='failed',error=str(e));raise
    finally:(out/'summary.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)


if __name__=='__main__':main()
