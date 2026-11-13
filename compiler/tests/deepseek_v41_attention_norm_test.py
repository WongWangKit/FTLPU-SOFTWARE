"""The step after mHC: native attention RMSNorm on an explicit BF16 hc_pre fixture.

This does not execute mHC or projection. C++ owns layout, ingress and scheduling.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import sys

import numpy as np

from deepseek_v41_swa_test import Model
from deepseek_v41_qkv_norm_test import run, trace_times, bf16_input
from deepseek_v41_qkv_reference import REVISION, tensor
from import_hf_decoder_layer import bf16, read_bf16


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('build-dir','target-config','output-dir','input-npy','model-config'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((out/'weights/provenance.json').read_text())
    digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if manifest['revision']!=REVISION or digest(args.model_config)!=manifest['config_sha256']:
        raise ValueError('pinned model config/provenance mismatch')
    config=json.loads(args.model_config.read_text())['text_config'];width=config['hidden_size']
    epsilon=config['rms_norm_eps'];shape=(32,width)
    x=np.load(args.input_npy,allow_pickle=False)
    bf16_input(x,shape)  # Require exact BF16 values at the hc_pre output interface.
    gamma=tensor(out/'weights',manifest,'attn_norm','BF16',(width,))
    model=Model(out,quantize_activation=False)
    model.input('hc_pre_output',x);model.input('gamma',gamma)
    y=model.op('attention_rmsnorm','rms_norm',['hc_pre_output','gamma'],shape,
               f'axis=1:i64,epsilon={epsilon:.9e}:f32')
    model.write([y])
    # Independent FP64 formula with only the declared final BF16 RNE boundary.
    xd=x.astype('f8');gd=gamma.astype('f8')
    reference=xd/np.sqrt(np.mean(xd*xd,axis=1,keepdims=True)+epsilon)*gd
    expected=bf16(reference.astype('f4'));np.save(out/'expected_output.npy',expected)
    limit=dict(relative_l2=.01,cosine=.9999)
    record=dict(status='preparing',scope='layer 0 attention RMSNorm only; hc_pre output supplied as a fixture',
        input_shape=list(shape),output_shape=list(shape),persistent_state=None,
        precision='BF16 input/original gamma/output with FP32 normalization arithmetic; no FP8 activation QDQ',
        epsilon=epsilon,revision=REVISION,raw_macro=True,limits=limit,
        input_npy=str(args.input_npy.resolve()),input_sha256=digest(args.input_npy),
        gamma_sha256=manifest['tensors']['attn_norm']['sha256'],command=[sys.executable,*sys.argv])
    env={k:v for k,v in os.environ.items() if not k.startswith('FTLPU_')}
    env.update(FTLPU_FILE_RUNNER_MODEL_SESSION='1',FTLPU_FILE_RUNNER_SXM_ARCHITECTURE='wide',
               FTLPU_GRAPH_PROGRESS='1',FTLPU_SESSION_PROGRESS='1')
    ext='.exe' if os.name=='nt' else '';exe=lambda area,name:args.build_dir.resolve()/area/(name+ext)
    try:
        run([exe('compiler','ftlpu-compile'),'--input',out/'kernel.mlir','--input-stage','kernel',
             '--target-config',args.target_config.resolve(),'--icu-compression','raw-macro',
             '--output',out/'program.ftlpu'],out/'compile.log',env)
        plan=json.loads((out/'graph_plan.json').read_text())
        active=[n for n in plan['nodes'] if not n.get('layout_join')]
        assert len(active)==1 and active[0]['name']=='attention_rmsnorm',active
        assert not active[0].get('native_projections'),active
        schedule=(out/'decoder_layer.schedule.mlir').read_text()
        assert 'distributed_16' not in schedule, 'RMSNorm must use planar storage including replay'
        for phase in ('direct_planar_ingress','sxm_vxm_streaming','vxm_sxm_direct','direct_planar_output'):
            assert 'rmsnorm.'+phase in schedule, phase
        assert not plan.get('linker_mem_transports') and not plan.get('native_output_archive_copies'),plan
        record.update(status='compiled_only',structure=dict(graph_nodes=len(plan['nodes']),native_rmsnorm_nodes=1,
            linker_mem_transports=plan.get('linker_mem_transports',[]),
            output_layouts=active[0].get('direct_output_layouts'),
            native_output_archive_copies=plan.get('native_output_archive_copies'),
            layout='planar input/output and planar transposed-x replay',
            direct_edges=['SXM -> VXM reduction','VXM normalization -> SXM'],
            normalized_output_spill=False),
            signatures={k:digest(p) for k,p in dict(kernel=out/'kernel.mlir',program=out/'program.ftlpu',
                target=args.target_config,compiler=exe('compiler','ftlpu-compile'),
                runner=exe('runtime','compiled_stage_file_runner'),
                exporter=exe('runtime','ftlpu_icu_program_export')).items()})
        if args.prepare_only:return
        record['status']='executing';(out/'summary.json').write_text(json.dumps(record,indent=2)+'\n')
        run([exe('runtime','compiled_stage_file_runner'),out/'program.ftlpu',out/'inputs.txt',
             out/'actual',out/'runtime.pipeline.csv'],out/'execution.log',env)
        run([exe('runtime','ftlpu_icu_program_export'),out/'actual.runtime.ftlpu',out/'runtime_icu_programs',
             '--pre-execution-dir',out/'actual.pre_execution','--compute-imem-segments'],out/'export.log',env)
        packed=0
        for path in (out/'runtime_icu_programs').glob('mem_*.icu.csv'):
            for row in csv.DictReader(line for line in path.read_text().splitlines() if not line.startswith('#')):
                assert row['opcode']!='EXTENDED',row
                if row['opcode'].endswith('_MACRO'):
                    packed+=1
                    image=out/'runtime_icu_programs'/row['raw_words'].removeprefix('file:')
                    assert image.stat().st_size%12==0
        assert packed>=20,packed
        record['structure']['exported_packed_mem_domains']=packed
        actual=read_bf16(out/'actual.0.bin',shape);av=actual.astype('f8').ravel();rv=expected.astype('f8').ravel()
        error=float(np.linalg.norm(av-rv)/np.linalg.norm(rv))
        cosine=float(np.vdot(av,rv)/(np.linalg.norm(av)*np.linalg.norm(rv)))
        stats={k:int(v) for k,v in re.findall(r'(\w+)=(\d+)',(out/'execution.log').read_text().splitlines()[-1])}
        timing=trace_times(out/'runtime.pipeline.csv')
        timing['weight_page_wait_cycles']=stats['weight_page_runtime_wait_cycles']
        timing['invocation_excluding_weight_page_wait_cycles']=timing['invocation_cycles']-timing['weight_page_wait_cycles']
        record.update(metrics=dict(relative_l2=error,cosine=cosine,finite=bool(np.isfinite(av).all()),
            max_absolute_error=float(np.max(np.abs(av-rv)))),runtime=stats,timing=timing,
            runtime_pipeline=str(out/'runtime.pipeline.csv'),runtime_icu_index=str(out/'runtime_icu_programs/index.csv'))
        assert np.isfinite(av).all() and error<limit['relative_l2'] and cosine>limit['cosine'],record['metrics']
        assert stats['host_operations']==0 and stats['outputs']==1,stats
        record['status']='passed'
        baseline=out/'baseline_before_planar_streaming/summary.json'
        if baseline.exists():
            before=json.loads(baseline.read_text())
            for key in ('input_sha256','gamma_sha256','epsilon','limits','precision'):
                assert before[key]==record[key],f'comparison changed {key}'
            assert before['signatures']['target']==record['signatures']['target']
            record['comparison']={key:dict(before=before['timing'][key],after=timing[key],
                reduction_percent=100*(1-timing[key]/before['timing'][key]))
                for key in ('invocation_cycles','cold_cycles_through_compute','initial_ingress_cycles','diagnostic_readback_cycles')}
    except KeyboardInterrupt:
        record['status']='stopped_by_user';raise
    except Exception as exc:
        record.update(status='failed',error=str(exc));raise
    finally:
        (out/'summary.json').write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps(record),flush=True)


if __name__=='__main__':main()
