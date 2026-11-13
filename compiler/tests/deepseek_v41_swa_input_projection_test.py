"""Step 1: real layer-0 Q_A/KV projections, FP8 weights and unquantized BF16 X.

Default: compile only. --run explicitly enables raw-macro CModel execution.
No RMSNorm, RoPE, attention, cache, mHC or MoE is part of this test.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys

import numpy as np

from deepseek_v41_swa_test import Model
from deepseek_v41_qkv_norm_test import run, trace_times
from deepseek_v41_qkv_reference import REVISION, tensor, project_bf16_activation, project, act_quant
from import_hf_decoder_layer import bf16_bits, read_bf16


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('build-dir','target-config','package','output-dir'):
        parser.add_argument('--'+name,type=Path,required=True)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--run',action='store_true')
    mode.add_argument('--emit-only',action='store_true',help='Write logical graph/reference only, without compiling or executing')
    args=parser.parse_args();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    pkg=args.package.resolve();meta=json.loads((pkg/'weights/provenance.json').read_text())
    if meta['revision']!=REVISION:raise ValueError('checkpoint revision mismatch')
    x=np.load(pkg/'reference/x.npy')
    if x.shape!=(32,5120):raise ValueError('Step 1 requires the batch1 seq32 layer-input fixture')
    model=Model(out,quantize_activation=False);model.input('x',x)
    expected=[];outputs=[];official_difference=[]
    for name,role,width in [('q_a','wq_a',1280),('kv_raw','wkv',512)]:
        w=tensor(pkg/'weights',meta,role,'F8_E4M3',(width,5120))
        s=tensor(pkg/'weights',meta,role+'_scale','F8_E8M0',(width//32,160))
        model.input(role,w,'f8E4M3FN');model.input(role+'_scale',s,'f8E8M0FNU')
        outputs.append(model.projection(name,'x',role,role+'_scale',32,width))
        gold=project_bf16_activation(x,w,s);expected.append(gold);np.save(out/(name+'.expected.npy'),gold)
        original=project(*act_quant(x),w,s)
        official_difference.append(dict(output=name,relative_l2=float(
            np.linalg.norm(gold.astype('f8')-original)/np.linalg.norm(original.astype('f8')))))
    model.write(outputs)
    record=dict(status='logical_graph_only',scope='layer 0 SWA input projections only',
        batch=1,sequence=32,inputs={'x':[32,5120]},outputs={'q_a':[32,1280],'kv_raw':[32,512]},
        persistent_state=None,revision=REVISION,precision_policy='bf16-activation',
        precision='Original E4M3FN weights/E8M0 block32 scales dequantized to BF16 at MXM IW; '
                  'unquantized BF16 X; ordered FP32 K32 partial/total accumulation; BF16 RNE output',
        raw_macro=True,limits={'bf16_bit_mismatches':0},
        cpu_reference_difference_from_official_fp8_activation=official_difference,
        command=[sys.executable,*sys.argv])
    env={k:v for k,v in os.environ.items() if not k.startswith('FTLPU_')}
    env.update(FTLPU_FILE_RUNNER_MODEL_SESSION='1',FTLPU_FILE_RUNNER_SXM_ARCHITECTURE='wide',
               FTLPU_GRAPH_PROGRESS='1',FTLPU_SESSION_PROGRESS='1')
    ext='.exe' if os.name=='nt' else ''
    exe=lambda area,name:args.build_dir.resolve()/area/(name+ext)
    digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    try:
        if args.emit_only:return
        run([exe('compiler','ftlpu-compile'),'--input',out/'kernel.mlir','--input-stage','kernel',
             '--target-config',args.target_config.resolve(),'--icu-compression','raw-macro','--mxm-execution','ws',
             '--output',out/'program.ftlpu'],out/'compile.log',env)
        plan=json.loads((out/'graph_plan.json').read_text());roles={}
        for name,width in [('q_a',1280),('kv_raw',512)]:
            tasks=[p for node in plan['nodes'] if node['name'].startswith(name+'.') for p in node.get('native_projections',[])]
            roles[name]=dict(pages=len(tasks),full_k=5120,columns=sum(p['n'] for p in tasks))
            assert tasks and sum(p['n'] for p in tasks)==width
            assert all(p['kind']=='linear' and p['m']==32 and p['k']==5120
                       and p['rhs_scale_block_count']==p['n']//32*160 for p in tasks)
        assert not any('.qdq' in node['name'] for node in plan['nodes']),'Unexpected activation QDQ'
        record.update(status='compiled_only',structure=dict(roles=roles,graph_nodes=len(plan['nodes']),
            native_output_archive_copies=plan.get('native_output_archive_copies'),activation_qdq_nodes=0),
            signatures={k:digest(p) for k,p in dict(kernel=out/'kernel.mlir',program=out/'program.ftlpu',
                target=args.target_config,compiler=exe('compiler','ftlpu-compile'),
                runner=exe('runtime','compiled_stage_file_runner')).items()})
        if not args.run:return
        record['status']='executing';(out/'summary.json').write_text(json.dumps(record,indent=2)+'\n')
        run([exe('runtime','compiled_stage_file_runner'),out/'program.ftlpu',out/'inputs.txt',
             out/'actual',out/'runtime.pipeline.csv'],out/'execution.log',env)
        run([exe('runtime','ftlpu_icu_program_export'),out/'actual.runtime.ftlpu',out/'runtime_icu_programs',
             '--pre-execution-dir',out/'actual.pre_execution','--compute-imem-segments'],out/'export.log',env)
        metrics=[]
        for index,gold in enumerate(expected):
            actual=read_bf16(out/f'actual.{index}.bin',gold.shape)
            metrics.append(dict(output=outputs[index],bf16_bit_mismatches=int(np.count_nonzero(
                bf16_bits(actual)!=bf16_bits(gold))),relative_l2=float(
                np.linalg.norm(actual.astype('f8')-gold)/np.linalg.norm(gold.astype('f8')))))
        stats=dict(re.findall(r'(\w+)=(\d+)',(out/'execution.log').read_text().splitlines()[-1]))
        timing=trace_times(out/'runtime.pipeline.csv')
        timing['weight_page_wait_cycles']=int(stats['weight_page_runtime_wait_cycles'])
        timing['invocation_excluding_weight_page_wait_cycles']=timing['invocation_cycles']-timing['weight_page_wait_cycles']
        record.update(metrics=metrics,runtime=stats,timing=timing)
        assert all(m['bf16_bit_mismatches']==0 for m in metrics),metrics
        assert stats['host_operations']=='0'
        record['status']='passed'
    except KeyboardInterrupt:
        record['status']='stopped_by_user';raise
    except Exception as exc:
        record.update(status='failed',error=str(exc));raise
    finally:
        (out/'summary.json').write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps(record),flush=True)


if __name__=='__main__':main()
