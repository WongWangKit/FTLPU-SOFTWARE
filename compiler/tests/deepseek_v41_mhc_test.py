"""Standalone mHC: real BF16-converted weights, four unequal streams, raw-macro execution.

Y is an independent sublayer-output fixture, not a host implementation of attention.
Python defines the logical graph and independent reference only.
"""
import argparse
import csv
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import numpy as np

from deepseek_v41_swa_test import Model
from deepseek_v41_qkv_norm_test import run, trace_times
from deepseek_v41_qkv_reference import REVISION
from import_hf_decoder_layer import bf16, bf16_bits, read_bf16


def reference(x,w,scale,base,iterations=20,epsilon=1e-6,norm_epsilon=1e-20,narrow=True):
    tokens,streams,hidden=x.shape
    flat=x.reshape(tokens,-1).astype('f4')
    inverse=1/np.sqrt(np.mean(flat*flat,axis=-1,keepdims=True)+np.float32(norm_epsilon))
    projection=(flat @ w.T)*inverse
    if narrow:projection=bf16(projection)
    sigmoid=lambda a:1/(1+np.exp(-a.astype('f8')))
    pre=(sigmoid(projection[:,:streams]*scale[0]+base[:streams])+epsilon).astype('f4')
    post=(2*sigmoid(projection[:,streams:2*streams]*scale[1]+base[streams:2*streams])).astype('f4')
    logits=(projection[:,2*streams:]*scale[2]+base[2*streams:]).reshape(tokens,streams,streams)
    comb=np.exp(logits.astype('f8')-np.max(logits,axis=-1,keepdims=True))
    comb=comb/comb.sum(axis=-1,keepdims=True)+epsilon
    comb=comb/(comb.sum(axis=-2,keepdims=True)+epsilon)
    for _ in range(iterations-1):
        comb=comb/(comb.sum(axis=-1,keepdims=True)+epsilon)
        comb=comb/(comb.sum(axis=-2,keepdims=True)+epsilon)
    if narrow:return [bf16(pre),bf16(post),bf16(comb.astype('f4')),projection]
    return [pre,post,comb.astype('f4'),projection]


def collapse(x,pre):
    return bf16(np.sum(x.astype('f4')*pre[:,:,None],axis=1))


def expand(y,x,post,comb):
    # Official comb index order: first stream is the source, second the destination.
    return bf16(post[:,:,None]*y[:,None,:]+np.einsum('tid,tij->tjd',x,comb,optimize=True))


def metric(actual,gold):
    a=np.asarray(actual,dtype='f8').ravel();b=np.asarray(gold,dtype='f8').ravel()
    norm=float(np.linalg.norm(b));denom=float(np.linalg.norm(a))*norm
    return dict(finite=bool(np.isfinite(a).all()),relative_l2=float(np.linalg.norm(a-b)/max(norm,1e-30)),
                cosine=float(np.vdot(a,b)/denom) if denom else float(np.array_equal(a,b)),
                max_abs=float(np.max(np.abs(a-b))))


def write_structure(out,projection_mapping):
    plan=json.loads((out/'graph_plan.json').read_text())
    schedule=(out/'decoder_layer.schedule.mlir').read_text()
    layouts=Counter(layout['kind'] for node in plan['nodes']
                    for layout in node.get('direct_output_layouts',[]))
    structure=dict(scope='compiler structure; runtime numerical status is in summary.json',
        nodes=len(plan['nodes']),output_layout_kinds=dict(layouts),
        uses_distributed16='distributed16' in json.dumps(plan),
        native_output_archive_copies=plan['native_output_archive_copies'],
        linker_mem_transport_records=len(plan['linker_mem_transports']),
        linker_mem_transport_vectors=sum(x['vectors'] for x in plan['linker_mem_transports']),
        access_decisions=dict(Counter(x['decision'] for n in plan['nodes'] for x in n.get('access_plans',[]))),
        arithmetic_fusion_chains=dict(Counter(re.findall(
            r'name = "[^"]*\.vector\.(square_sum_chain|scale_add_chain|add_scale_chain|fold_chain)"',schedule))),
        projection_mapping=projection_mapping)
    (out/'structure.json').write_text(json.dumps(structure,indent=2)+'\n')
    assert not structure['uses_distributed16'],'mHC requires planar intermediate layouts'
    return structure


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('build-dir','target-config','weights','output-dir'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--hidden',type=int,default=5120);p.add_argument('--tokens',type=int,default=32)
    p.add_argument('--stage',choices=('all','mixes','pre','post'),default='all')
    p.add_argument('--fixture',choices=('unequal-streams','layer0-entry'),default='unequal-streams')
    p.add_argument('--baseline',type=Path,help='Compare a saved passing run with identical logical inputs, runtime and hardware')
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--reuse-prepared',action='store_true')
    a=p.parse_args();out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    config_bytes=a.target_config.read_bytes()
    a.target_config=out/'target_config.json';a.target_config.write_bytes(config_bytes)
    if not 0<a.hidden<=5120 or a.hidden%32 or a.tokens<=0 or a.tokens%32:raise ValueError('Expected D32 within 5120 and complete token32 tiles')
    manifest=json.loads((a.weights/'provenance.json').read_text())
    if manifest['revision']!=REVISION:raise ValueError('checkpoint revision mismatch')
    params={};original={}
    for role,entry in manifest['tensors'].items():
        path=a.weights/entry['file'];raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=entry['sha256']:raise ValueError('BF16 weight checksum')
        params[role]=read_bf16(path,entry['shape'])
        raw=(a.weights/entry['original_file']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=entry['original_sha256']:raise ValueError('original weight checksum')
        original[role]=np.frombuffer(raw,dtype='<f4').reshape(entry['shape'])
    for collection in (params,original):collection['fn']=collection['fn'].reshape(24,4,5120)[:,:,:a.hidden].reshape(24,4*a.hidden)
    rng=np.random.default_rng(411120)
    x=bf16(rng.normal(size=(a.tokens,4,a.hidden)).astype('f4')*np.array([.7,1.,1.3,1.6],dtype='f4')[None,:,None])
    y=bf16(rng.normal(size=(a.tokens,a.hidden)).astype('f4')*.25)
    incoming=rng.uniform(.1,1.,size=(a.tokens,4)).astype('f4');incoming=bf16(incoming/incoming.sum(axis=-1,keepdims=True))
    if a.fixture=='layer0-entry':
        x=np.repeat(x[:,:1,:],4,axis=1)
        incoming=np.zeros((a.tokens,4),dtype='f4');incoming[:,0]=1
    gold=reference(x,params['fn'],params['scale'],params['base'])
    legacy=reference(x,original['fn'],original['scale'],original['base'],narrow=False)
    all_gold=gold+[collapse(x,incoming),expand(y,x,gold[1],gold[2])]
    model=Model(out,quantize_activation=False);model.input('x',x)
    if a.stage in ('all','mixes'):
        for role in ('fn','scale','base'):model.input(role,params[role])
        names=['pre','post','comb','projection'];shapes=[(a.tokens,4),(a.tokens,4),(a.tokens,4,4),(a.tokens,24)]
        for name,shape in zip(names,shapes):model.types[name]=model.ty(shape)
        model.ops.append('%pre, %post, %comb, %projection = "ftlpu.kernel.hc_mixes"(%x, %fn, %scale, %base) '
            '{iterations=20:i64,epsilon=1.000000e-06:f32,norm_epsilon=1.000000e-20:f32,ftlpu.logical_name="hc_mixes"} : ('+
            ','.join(model.types[n] for n in ['x','fn','scale','base'])+') -> ('+','.join(model.types[n] for n in names)+')')
    if a.stage in ('all','pre'):
        model.input('incoming_pre',incoming)
        model.op('collapsed','hc_pre',['x','incoming_pre'],(a.tokens,a.hidden),generic=True)
    if a.stage in ('all','post'):
        model.input('y',y)
        if a.stage=='post':model.input('post',gold[1]);model.input('comb',gold[2])
        model.op('expanded','hc_post',['y','x','post','comb'],x.shape,generic=True)
    indices={'all':list(range(6)),'mixes':list(range(4)),'pre':[4],'post':[5]}[a.stage]
    names=['pre','post','comb','projection','collapsed','expanded'];outputs=[names[i] for i in indices]
    # Diagnostic output views retain token columns. This changes only the
    # test's readback order; hc_pre/post still consume the semantic [T,H,...]
    # tensors above. It avoids forcing short H=4 rows through a padded SXM
    # transpose merely to compare coefficients on the host.
    diagnostic=[]
    for i in indices:
        width=int(np.prod(all_gold[i].shape[1:]));flat=names[i]
        if all_gold[i].ndim!=2:flat=model.reshape(names[i]+'_flat',flat,(a.tokens,width))
        diagnostic.append(model.op(names[i]+'_token_columns','transpose',[flat],(width,a.tokens),
                                    'permutation=array<i64:1,0>',generic=True))
    model.write(diagnostic)
    kernel=out/'kernel.mlir'
    kernel.write_text(kernel.read_text().replace('module attributes {',
        'module attributes {ftlpu.native_output_pieces=true,',1))
    for i in indices:np.save(out/(names[i]+'.expected.npy'),all_gold[i])
    record=dict(status='preparing',scope='independent mHC '+a.stage,model='DeepSeek-V4.1-Flash-1017',
        revision=REVISION,layer=manifest['layer'],branch=manifest['branch'],tokens=a.tokens,streams=4,hidden=a.hidden,
        real_full_shape=a.hidden==5120,raw_macro=True,persistent_state=None,fixture=a.fixture,
        projection_mapping='BF16 MXM W*X^T, K partitioned across hemispheres; bounded continuous K32 domains retain FP32 ACC, FP32 partial-page sums and normalization before BF16 output',
        external_memory=json.loads(config_bytes)['external_memory'],
        timing_scope='Data ingress + simulated execution; diagnostic readback separate; instruction-image DMA loading is not timed by ModelSession.',
        precision='BF16 X/Y/incoming mix/weights/scale/base; FP32 projection sums and Sinkhorn intermediates; '
                  'BF16 normalized projection, pre/post/comb and final output boundaries',
        sinkhorn_iterations=20,limits=dict(relative_l2=.01,cosine=.9999,row_column_sum_error_vs_reference=.01),
        outputs=outputs,diagnostic_output_view='[flattened feature,token]; transposed back for comparison only',command=[sys.executable,*sys.argv],
        cpu_bf16_policy_difference_from_original_fp32={names[i]:metric(gold[i],legacy[i]) for i in range(4)})
    env={k:v for k,v in os.environ.items() if not k.startswith('FTLPU_')}
    env.update(FTLPU_FILE_RUNNER_MODEL_SESSION='1',FTLPU_FILE_RUNNER_SXM_ARCHITECTURE='wide',
               FTLPU_GRAPH_PROGRESS='1',FTLPU_SESSION_PROGRESS='1')
    ext='.exe' if os.name=='nt' else '';exe=lambda area,name:a.build_dir.resolve()/area/(name+ext)
    digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    try:
        sig={k:digest(p) for k,p in dict(kernel=out/'kernel.mlir',target=a.target_config,
            compiler=exe('compiler','ftlpu-compile'),runner=exe('runtime','compiled_stage_file_runner')).items()}
        if a.reuse_prepared:
            prepared=json.loads((out/'prepared.json').read_text())
            assert prepared['signatures']==sig and prepared['program']==digest(out/'program.ftlpu'),'Prepared program/config changed'
        else:
            run([exe('compiler','ftlpu-compile'),'--input',out/'kernel.mlir','--input-stage','kernel',
                 '--target-config',a.target_config.resolve(),'--icu-compression','raw-macro',
                 '--output',out/'program.ftlpu'],out/'compile.log',env)
            (out/'prepared.json').write_text(json.dumps(dict(signatures=sig,program=digest(out/'program.ftlpu')),indent=2)+'\n')
        record.update(status='compiled_only',signatures=sig)
        write_structure(out,record['projection_mapping'])
        if a.prepare_only:return
        record['status']='executing';(out/'summary.json').write_text(json.dumps(record,indent=2)+'\n')
        run([exe('runtime','compiled_stage_file_runner'),out/'program.ftlpu',out/'inputs.txt',out/'actual',
             out/'runtime.pipeline.csv'],out/'execution.log',env)
        run([exe('runtime','ftlpu_icu_program_export'),out/'actual.runtime.ftlpu',out/'runtime_icu_programs',
             '--pre-execution-dir',out/'actual.pre_execution','--compute-imem-segments'],out/'export.log',env)
        actual=[read_bf16(out/f'actual.{j}.bin',(int(np.prod(all_gold[i].shape[1:])),a.tokens)).T.reshape(all_gold[i].shape)
                for j,i in enumerate(indices)]
        metrics={names[i]:metric(v,all_gold[i]) for i,v in zip(indices,actual)}
        stats=dict(re.findall(r'(\w+)=(\d+)',(out/'execution.log').read_text().splitlines()[-1]))
        record.update(metrics=metrics,runtime=stats,timing=trace_times(out/'runtime.pipeline.csv'))
        record['timing']['weight_page_runtime_wait_cycles']=int(stats['weight_page_runtime_wait_cycles'])
        compute={}
        with (out/'runtime.pipeline.csv').open(newline='') as stream:
            for row in csv.DictReader(stream):
                if row['resource'].startswith('MXM.') and row['resource'].endswith('.Compute'):
                    count=int(row['inner_count'])*int(row['outer_count'])-int(row['skip_first'])
                    compute[row['resource']]=compute.get(row['resource'],0)+count
        # Each trace block spans a K32 matrix operation, not one FU issue.
        record['mxm_compute_blocks']=compute
        if a.stage in ('all','mixes'):
            assert any(k.startswith('MXM.E') and v>0 for k,v in compute.items()),'mHC projection must execute east MXM'
            assert any(k.startswith('MXM.W') and v>0 for k,v in compute.items()),'mHC projection must execute west MXM'
        if a.stage in ('all','mixes'):
            comb=actual[2].astype('f4')
            record['sinkhorn_checks']=dict(row_sum_max_abs=float(abs(comb.sum(-1)-1).max()),
                column_sum_max_abs=float(abs(comb.sum(-2)-1).max()),nonnegative=bool((comb>=0).all()),
                reference_row_sum_max_abs=float(abs(gold[2].sum(-1)-1).max()),
                reference_column_sum_max_abs=float(abs(gold[2].sum(-2)-1).max()),
                row_sum_error_vs_reference=float(abs(comb.sum(-1)-gold[2].sum(-1)).max()),
                column_sum_error_vs_reference=float(abs(comb.sum(-2)-gold[2].sum(-2)).max()),
                note='Finite 20-iteration Sinkhorn does not guarantee convergence within 1% of unity; compare the same algorithm, and report convergence separately.')
            assert record['sinkhorn_checks']['nonnegative']
            assert max(record['sinkhorn_checks']['row_sum_error_vs_reference'],record['sinkhorn_checks']['column_sum_error_vs_reference'])<.01
        if a.stage=='all':
            local=metric(actual[5],expand(y,x,actual[1],actual[2]))
            record['local_post_using_device_coefficients']=local
            assert local['finite'] and local['relative_l2']<.01 and local['cosine']>.9999,local
        assert all(m['finite'] and m['relative_l2']<.01 and m['cosine']>.9999 for m in metrics.values()),metrics
        if a.fixture=='layer0-entry' and 'collapsed' in outputs:
            assert np.array_equal(actual[outputs.index('collapsed')],x[:,0,:]),'one-hot initial pre must preserve the embedding exactly'
        assert stats['host_operations']=='0'
        if a.baseline:
            previous=json.loads((a.baseline/'summary.json').read_text())
            assert previous['status']=='passed'
            for field in ('revision','layer','branch','tokens','streams','hidden','fixture','precision','sinkhorn_iterations','external_memory','outputs'):
                assert previous[field]==record[field],f'baseline differs: {field}'
            for field in ('target','runner'):
                assert previous['signatures'][field]==sig[field],f'baseline executable/config differs: {field}'
            logical_inputs=['x.bin']
            if a.stage in ('all','mixes'):logical_inputs+=['fn.bin','scale.bin','base.bin']
            if a.stage in ('all','pre'):logical_inputs+=['incoming_pre.bin']
            if a.stage in ('all','post'):logical_inputs+=['y.bin']
            if a.stage=='post':logical_inputs+=['post.bin','comb.bin']
            for file in logical_inputs:
                assert digest(a.baseline/file)==digest(out/file),f'baseline input differs: {file}'
            comparison=dict(status='passed',baseline=str(a.baseline.resolve()),same_conditions=True,
                timing={k:dict(before=previous['timing'][k],after=v,
                    speedup=previous['timing'][k]/v if v else None) for k,v in record['timing'].items()},
                metrics_before=previous['metrics'],metrics_after=metrics,
                output_bit_identical={name:(a.baseline/f'actual.{i}.bin').read_bytes()==(out/f'actual.{i}.bin').read_bytes()
                    for i,name in enumerate(outputs) if (a.baseline/f'actual.{i}.bin').exists()})
            (out/'comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
            record['comparison']=comparison
        record['status']='passed'
    except KeyboardInterrupt:record['status']='stopped_by_user';raise
    except Exception as exc:record.update(status='failed',error=str(exc));raise
    finally:
        (out/'summary.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)


if __name__=='__main__':main()
