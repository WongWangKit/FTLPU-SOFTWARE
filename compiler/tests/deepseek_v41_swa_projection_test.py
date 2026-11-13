"""Real output-weight slices exercise grouped projection and independent K32 QDQ."""
import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np
from deepseek_v41_swa_test import Model
from deepseek_v41_qkv_norm_test import run, trace_times
from deepseek_v41_qkv_reference import tensor, act_quant, dequant_act, project
from import_hf_decoder_layer import bf16_bits, read_bf16


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('build-dir','target-config','package','output-dir'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    weights=a.package/'weights';manifest=json.loads((weights/'provenance.json').read_text())
    original=tensor(weights,manifest,'wo_a','F8_E4M3',(8192,4096))
    original_scale=tensor(weights,manifest,'wo_a_scale','F8_E8M0',(256,128))
    # Distinct original groups and scales catch group-offset and K32-axis errors.
    w=np.concatenate((original[:64,:64],original[1024:1088,:64]))
    s=np.concatenate((original_scale[:2,:2],original_scale[32:34,:2]))
    rng=np.random.default_rng(41111)
    x=(bf16_bits(rng.normal(size=(32,128)).astype('f4')*
        np.repeat(np.array([.01,.125,1.,32.],dtype='f4'),32)).astype('u4')<<16).view('f4')
    m=Model(out);m.input('x',x);m.input('w',w,'f8E4M3FN');m.input('s',s,'f8E8M0FNU')
    y=m.projection('grouped','x','w','s',32,128,2)
    # Ten K32 groups exceed one broadcast-layout batch on the LPU32 target.
    cache_x=np.concatenate((x,x*2,x[:,:64]*4),axis=1)
    m.input('cache_x',cache_x)
    z=m.op('kv_precision','block_fp8_qdq',['cache_x'],(32,320));m.write([y,z])
    gold=np.concatenate([project(*act_quant(x[:,g*64:(g+1)*64]),w[g*64:(g+1)*64],s[g*2:(g+1)*2]) for g in range(2)],axis=1)
    expected=[gold,dequant_act(*act_quant(cache_x))]
    env={k:v for k,v in os.environ.items() if not k.startswith('FTLPU_')}
    env.update(FTLPU_FILE_RUNNER_MODEL_SESSION='1',FTLPU_FILE_RUNNER_SXM_ARCHITECTURE='wide',FTLPU_GRAPH_PROGRESS='1')
    ext='.exe' if os.name=='nt' else '';exe=lambda area,name:a.build_dir.resolve()/area/(name+ext)
    record=dict(status='running',scope='two grouped output-projection weight slices and wide K32 QDQ',raw_macro=True,
        command=[sys.executable,*sys.argv],bf16_mismatch_limit=0)
    try:
        run([exe('compiler','ftlpu-compile'),'--input',out/'kernel.mlir','--input-stage','kernel','--target-config',a.target_config.resolve(),
             '--icu-compression','raw-macro','--output',out/'program.ftlpu'],out/'compile.log',env)
        run([exe('runtime','compiled_stage_file_runner'),out/'program.ftlpu',out/'inputs.txt',out/'actual',out/'runtime.pipeline.csv'],out/'execution.log',env)
        run([exe('runtime','ftlpu_icu_program_export'),out/'actual.runtime.ftlpu',out/'runtime_icu_programs','--pre-execution-dir',out/'actual.pre_execution'],out/'export.log',env)
        differences=[int(np.count_nonzero(bf16_bits(read_bf16(out/f'actual.{i}.bin',gold.shape))!=bf16_bits(gold))) for i,gold in enumerate(expected)]
        record.update(bf16_mismatches=differences,timing=trace_times(out/'runtime.pipeline.csv'))
        assert not any(differences),differences
        record['status']='passed'
    except Exception as e:record.update(status='failed',error=str(e));raise
    finally:(out/'summary.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)


if __name__=='__main__':main()
