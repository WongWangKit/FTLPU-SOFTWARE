"""Native planar RMSNorm: tile boundary, multi-tile replay and epsilon paths."""
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
from deepseek_v41_swa_test import Model
from deepseek_v41_qkv_norm_test import run
from import_hf_decoder_layer import bf16, read_bf16


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('build-dir','target-config','output-dir'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();root=a.output_dir.resolve();root.mkdir(parents=True,exist_ok=True)
    ext='.exe' if os.name=='nt' else ''
    exe=lambda area,name:a.build_dir.resolve()/area/(name+ext)
    env={k:v for k,v in os.environ.items() if not k.startswith('FTLPU_')}
    env.update(FTLPU_FILE_RUNNER_MODEL_SESSION='1',FTLPU_FILE_RUNNER_SXM_ARCHITECTURE='wide')
    cases=[]
    for dtype,width,epsilon in [('bf16',32,1e-20),('bf16',96,1e-20),('bf16',512,1e-5),('f16',96,1e-5)]:
        out=root/f'{dtype}_{width}';out.mkdir(exist_ok=True)
        rng=np.random.default_rng(2032+width)
        quant=bf16 if dtype=='bf16' else lambda x:x.astype('<f2')
        x=rng.normal(size=(32,width)).astype('f4');x[0]=0
        if epsilon==1e-20:
            x[1]*=2**-20;x[2]*=128
        x=quant(x);g=quant(rng.uniform(.5,1.5,size=width).astype('f4'))
        m=Model(out);m.input('x',x,dtype);m.input('gamma',g,dtype)
        y=m.op('norm','rms_norm',['x','gamma'],x.shape,
               f'axis=1:i64,epsilon={epsilon:.9e}:f32',dtype=dtype);m.write([y])
        if dtype=='f16':
            # The logical model importer supports BF16 but not FP16. Exercise
            # the same native RMSNorm lowering through its typed kernel API.
            (out/'kernel.mlir').write_text('module attributes {ftlpu.sxm_wide=true} {\n'
                +'func.func @norm('+','.join(m.args)+') -> '+m.types[y]+' {\n'
                +'\n'.join(m.ops)+'\nreturn %'+y+' : '+m.types[y]+'\n}}\n')
            (out/'inputs.txt').write_text('\n'.join(m.files)+'\n')
        run([exe('compiler','ftlpu-compile'),'--input',out/'kernel.mlir','--input-stage','kernel',
             '--target-config',a.target_config.resolve(),'--icu-compression','raw-macro',
             '--output',out/'program.ftlpu'],out/'compile.log',env)
        if dtype=='bf16':
            schedule=(out/'decoder_layer.schedule.mlir').read_text()
            assert 'distributed_16' not in schedule and 'rmsnorm.vxm_sxm_direct' in schedule
        run([exe('runtime','compiled_stage_file_runner'),out/'program.ftlpu',out/'inputs.txt',
             out/'actual',out/'runtime.pipeline.csv'],out/'execution.log',env)
        actual=read_bf16(out/'actual.0.bin',x.shape) if dtype=='bf16' else np.fromfile(out/'actual.0.bin','<f2').reshape(x.shape)
        xd=x.astype('f8');gold=quant((xd/np.sqrt(np.mean(xd*xd,axis=1,keepdims=True)+epsilon)*g).astype('f4')).astype('f8')
        error=float(np.linalg.norm(actual-gold)/np.linalg.norm(gold))
        # Check every nonzero row, including the tiny/large variance cases.
        row_errors=np.linalg.norm(actual[1:]-gold[1:],axis=1)/np.linalg.norm(gold[1:],axis=1)
        assert np.isfinite(actual).all() and np.count_nonzero(actual[0])==0
        assert error<.01 and row_errors.max()<.01,(dtype,width,error,row_errors)
        run([exe('runtime','ftlpu_icu_program_export'),out/'actual.runtime.ftlpu',out/'icu'],out/'export.log',env)
        packed=0
        for path in (out/'icu').glob('mem_*.icu.csv'):
            for row in csv.DictReader(line for line in path.read_text().splitlines() if not line.startswith('#')):
                assert row['opcode']!='EXTENDED',row
                if row['opcode'].endswith('_MACRO'):
                    packed+=1
                    assert not row['pc_word'] and not row['imem_words']
                    image=out/'icu'/row['raw_words'].removeprefix('file:')
                    assert image.stat().st_size%12==0
        assert packed>=16,packed
        cases.append(dict(dtype=dtype,width=width,epsilon=epsilon,relative_l2=error,
                          worst_row_relative_l2=float(row_errors.max()),zero_row_exact=True,packed_mem_domains=packed))
    (root/'summary.json').write_text(json.dumps(dict(status='passed',cases=cases),indent=2)+'\n')
    print(json.dumps(cases))


if __name__=='__main__':main()
