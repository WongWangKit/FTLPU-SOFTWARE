"""Real layer-0 SWA equations; all lowering, paging and ICU scheduling are C++."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys

import numpy as np

from deepseek_v41_qkv_norm_test import run, trace_times
from deepseek_v41_qkv_reference import tensor, act_quant, project, project_bf16_activation, norm, rope, E4M3FN, scales, dequant_act, REVISION
from deepseek_v41_swa_core_test import reference as attention_reference
from import_hf_decoder_layer import bf16, bf16_bits, read_bf16


def check_native_projections(folder,stage,precision_policy):
    plan=json.loads((folder/'graph_plan.json').read_text())
    expected={'q_a':(5120,1280),'q_b':(1280,32768),'kv_raw':(5120,512)}
    if stage=='full':expected.update(o_a=(4096,8192),o_b=(8192,5120))
    roles={}
    for name,(k,n) in expected.items():
        tasks=[p for node in plan['nodes'] if node['name'].startswith(name+'.') for p in node.get('native_projections',[])]
        valid=bool(tasks) and sum(p['n'] for p in tasks)==n and all(
            p['kind']=='linear' and p['m']==32 and p['k']==k and p['n']%64==0
            and p['rhs_scale_block_count']==p['n']//32*(k//32) for p in tasks)
        roles[name]=dict(passed=valid,full_k=k,output_columns=n,page_count=len(tasks))
    qdq_count=sum('.qdq' in node['name'] for node in plan['nodes'])
    precision_valid=precision_policy!='bf16-activation' or qdq_count==0
    result=dict(passed=all(x['passed'] for x in roles.values()) and precision_valid,roles=roles,
        precision_policy=precision_policy,activation_qdq_nodes=qdq_count,
        graph_nodes=len(plan['nodes']),native_output_archive_copies=plan.get('native_output_archive_copies'))
    (folder/'structure.json').write_text(json.dumps(result,indent=2)+'\n')
    assert result['passed'],result
    return result


class Model:
    """Only logical model tensors and equations; no physical layouts or tile loops."""
    def __init__(self, folder, quantize_activation=True):
        self.folder=folder;self.args=[];self.files=[];self.ops=[];self.types={}
        self.quantize_activation=quantize_activation

    @staticmethod
    def ty(shape,dtype='bf16'):return 'tensor<'+'x'.join(map(str,shape))+'x'+dtype+'>'

    def input(self,name,data,dtype='bf16'):
        path=self.folder/(name+'.bin')
        (bf16_bits(data) if dtype=='bf16' else np.asarray(data)).tofile(path)
        self.types[name]=self.ty(data.shape,dtype);self.args.append('%'+name+':'+self.types[name]);self.files.append(path.as_posix())
        return name

    def op(self,name,opcode,operands,shape,attrs='',dtype='bf16',generic=False):
        self.types[name]=self.ty(shape,dtype)
        arguments=', '.join('%'+a for a in operands)
        operation=(f'"ftlpu.kernel.{opcode}"('+arguments+')') if generic else f'ftlpu.kernel.{opcode} '+arguments
        self.ops.append(f'%{name} = '+operation+
            ' {'+attrs+f', ftlpu.logical_name="{name}"'+'} : ('+','.join(self.types[a] for a in operands)+') -> '+self.types[name])
        return name

    def reshape(self,name,x,shape):return self.op(name,'reshape',[x],shape)

    def projection(self,name,x,w,s,m,n,groups=1):
        return self.op(name,'block_fp8_projection',[x,w,s],(m,n),
            f'groups={groups}:i64,quantize_activation={str(self.quantize_activation).lower()}')

    def rotary(self,name,x,heads,inverse=False):
        return self.op(name,'rope',[x],(32,heads,512),f'heads={heads}:i64,head_dim=512:i64,theta=1.000000e+04:f32,'
            f'rotary_dim=64:i64,rotary_offset=448:i64,interleaved=true,inverse={str(inverse).lower()}')

    def write(self,outputs,rope_tile_overlap=False):
        outtypes=','.join(self.types[n] for n in outputs)
        source='module attributes {ftlpu.sxm_wide=true,ftlpu.vector_fp32_ingress=true,ftlpu.vector_bf16_rne_spill=false,ftlpu.common_barrier_protocol=true,'
        source+='ftlpu.rope_tile_overlap='+str(rope_tile_overlap).lower()+'} {\n'
        source+='func.func @deepseek_swa('+','.join(self.args)+') -> ('+outtypes+') attributes {'
        source+='ftlpu.logical_inputs='+json.dumps(self.files)+',ftlpu.artifact_directory='+json.dumps(self.folder.as_posix())+'} {\n'
        source+='\n'.join(self.ops)+'\nreturn '+','.join('%'+n for n in outputs)+' : '+outtypes+'\n}}\n'
        # Empty attribute lists have no leading comma.
        source=source.replace('{, ftlpu.logical_name','{ftlpu.logical_name')
        (self.folder/'kernel.mlir').write_text(source)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('build-dir','target-config','package','output-dir'):p.add_argument('--'+arg,type=Path,required=True)
    p.add_argument('--stage',choices=('qkv','attention','full'),default='full')
    p.add_argument('--precision-policy',choices=('bf16-activation','official-fp8'),default='bf16-activation',
                   help='BF16 activations and KV by default; official-fp8 retains projection and cache QDQ')
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--reuse-prepared',action='store_true',help='Execute a previously compiled program after verifying its source, target and executable hashes')
    a=p.parse_args();out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    pkg=a.package.resolve();meta=json.loads((pkg/'weights/provenance.json').read_text())
    extra=out/'weights';extra_meta=json.loads((extra/'provenance.json').read_text())
    if meta['revision']!=REVISION or extra_meta['revision']!=REVISION:raise ValueError('checkpoint revision mismatch')
    x=np.load(pkg/'reference/x.npy')
    official_fp8=a.precision_policy=='official-fp8'
    model=Model(out,quantize_activation=official_fp8);model.input('x',x)
    weights={}
    specs={'wq_a':(1280,5120),'wq_b':(32768,1280),'wkv':(512,5120)}
    if a.stage=='full':specs.update(wo_a=(8192,4096),wo_b=(5120,8192))
    for role,shape in specs.items():
        directory,manifest=(extra,extra_meta) if role.startswith('wo_') else (pkg/'weights',meta)
        w=tensor(directory,manifest,role,'F8_E4M3',shape)
        s=tensor(directory,manifest,role+'_scale','F8_E8M0',(shape[0]//32,shape[1]//32))
        model.input(role,w,'f8E4M3FN');model.input(role+'_scale',s,'f8E8M0FNU');weights[role]=(w,s)
    for role,width in [('q_norm_weight',1280),('kv_norm_weight',512)]:
        gamma=tensor(pkg/'weights',meta,role.removesuffix('_weight'),'BF16',(width,))
        model.input(role,gamma);weights[role]=gamma
    sinks=np.fromfile(extra/'attn_sink.bin','<f4')
    if hashlib.sha256((extra/'attn_sink.bin').read_bytes()).hexdigest()!=extra_meta['tensors']['attn_sink']['sha256']:raise ValueError('sink checksum')
    qa=model.projection('q_a','x','wq_a','wq_a_scale',32,1280)
    qn=model.op('q_norm','rms_norm',[qa,'q_norm_weight'],(32,1280),'axis=1:i64,epsilon=1.000000e-20:f32')
    qb=model.projection('q_b',qn,'wq_b','wq_b_scale',32,32768)
    q=model.rotary('q_rope',model.reshape('q_heads',qb,(32,64,512)),64)
    kv=model.projection('kv_raw','x','wkv','wkv_scale',32,512)
    kv=model.op('kv_norm','rms_norm',[kv,'kv_norm_weight'],(32,512),'axis=1:i64,epsilon=1.000000e-20:f32')
    kv=model.rotary('kv_rope',model.reshape('kv_head',kv,(32,1,512)),1)
    kv_flat=model.reshape('post_rope_kv',kv,(32,512))
    kv_output=(model.op('window_kv','block_fp8_qdq',[kv_flat],(32,512)) if official_fp8
               else model.reshape('window_kv',kv_flat,(32,512)))
    q4=model.reshape('query_batch',q,(1,32,64,512));kv4=model.reshape('kv_batch',kv_output,(1,32,1,512))
    outputs=[model.reshape('query_out',q,(32,32768)),kv_output]
    if a.stage!='qkv':
        c=model.op('swa','dense_attention',[q4,kv4,kv4],(1,32,64,512),
            f'scale={512**-.5:.9e}:f32,causal=true,query_position=0:i64,sliding_window=128:i64,'+
            'attention_sinks=array<f32:'+','.join(f'{v:.9e}' for v in sinks)+'>')
        c=model.rotary('context_inverse_rope',model.reshape('context_heads',c,(32,64,512)),64,True)
        c=model.reshape('context',c,(32,32768));outputs=[c,kv_output]
        if a.stage=='full':
            o=model.projection('o_a',c,'wo_a','wo_a_scale',32,8192,groups=8)
            o=model.projection('o_b',o,'wo_b','wo_b_scale',32,5120);outputs=[o,kv_output]
    model.write(outputs)
    record=dict(status='preparing',scope='layer 0 causal encoder SWA '+a.stage,
        batch=1,sequence=32,hidden=5120,query_heads=64,head_dim=512,window=128,
        initial_cache_tokens=0,final_cache_tokens=32,revision=REVISION,raw_macro=True,
        limits=dict(relative_l2=.03,cosine=.999),
        precision_policy=a.precision_policy,
        precision=('Official FP8 activation/cache QDQ; ' if official_fp8 else 'Unquantized BF16 activation/cache; ')+
                  'checkpoint FP8 weight blocks32 dequantized to BF16; FP32 accumulators/sink, BF16 output.')
    record['command']=[sys.executable,*sys.argv]
    (out/'summary.json').write_text(json.dumps(record,indent=2)+'\n')
    try:
        # Independent reference from the original logical weights, no device intermediates.
        def linear(z,role):
            return project(*act_quant(z),*weights[role]) if official_fp8 else project_bf16_activation(z,*weights[role])
        qa_ref=linear(x,'wq_a');qn_ref=norm(qa_ref,weights['q_norm_weight'],1e-20)
        q_ref=rope(linear(qn_ref,'wq_b').reshape(32,64,512),64,10000)
        kv_ref=rope(norm(linear(x,'wkv'),weights['kv_norm_weight'],1e-20),64,10000)
        if official_fp8:kv_ref=dequant_act(*act_quant(kv_ref))
        expected=q_ref.reshape(32,32768)
        if a.stage!='qkv':
            context=bf16(attention_reference(q_ref,kv_ref,sinks))
            # Conjugate the positional rotation, keeping its one BF16 boundary.
            tail=context[...,-64:].copy();inv=context.copy()
            freq=10000**(-np.arange(0,64,2,dtype='f4')/64)
            angle=np.arange(32,dtype='f4')[:,None,None]*freq[None,None,:]
            inv[...,-64::2]=bf16(tail[...,::2]*np.cos(angle)+tail[...,1::2]*np.sin(angle))
            inv[...,-63::2]=bf16(tail[...,1::2]*np.cos(angle)-tail[...,::2]*np.sin(angle))
            expected=inv.reshape(32,32768)
            if a.stage=='full':
                w,s=weights['wo_a']
                if official_fp8:
                    decoded=E4M3FN[w]*np.repeat(np.repeat(scales(s),32,0),32,1)
                    quant=dequant_act(*act_quant(expected)).reshape(32,8,4096)
                    # Official wo_a runs BF16 dequantized weights through grouped einsum.
                    oa=bf16(np.einsum('tgd,grd->tgr',quant,decoded.reshape(8,1024,4096),optimize=True)).reshape(32,8192)
                else:
                    oa=np.concatenate([project_bf16_activation(expected[:,g*4096:(g+1)*4096],
                        w[g*1024:(g+1)*1024],s[g*32:(g+1)*32]) for g in range(8)],axis=1)
                expected=linear(oa,'wo_b')
        np.save(out/'expected_output.npy',expected);np.save(out/'expected_window_kv.npy',kv_ref)
        env={k:v for k,v in os.environ.items() if not k.startswith('FTLPU_')}
        env.update(FTLPU_FILE_RUNNER_MODEL_SESSION='1',FTLPU_FILE_RUNNER_SXM_ARCHITECTURE='wide',FTLPU_GRAPH_PROGRESS='1',FTLPU_SESSION_PROGRESS='1')
        ext='.exe' if os.name=='nt' else '';exe=lambda area,name:a.build_dir.resolve()/area/(name+ext)
        digest=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
        signatures={name:digest(path) for name,path in {
            'kernel':out/'kernel.mlir','target':a.target_config,
            'compiler':exe('compiler','ftlpu-compile'),'runner':exe('runtime','compiled_stage_file_runner')}.items()}
        if a.reuse_prepared:
            prepared=json.loads((out/'prepared.json').read_text())
            assert prepared['signatures']==signatures,'Prepared source/config/executable mismatch; compile again'
            assert prepared['program_sha256']==digest(out/'program.ftlpu'),'Prepared program mismatch'
        else:
            run([exe('compiler','ftlpu-compile'),'--input',out/'kernel.mlir','--input-stage','kernel',
                '--target-config',a.target_config.resolve(),'--icu-compression','raw-macro','--mxm-execution','ws','--pass-timing',
                '--output',out/'program.ftlpu'],out/'compile.log',env)
            (out/'prepared.json').write_text(json.dumps(dict(signatures=signatures,
                program_sha256=digest(out/'program.ftlpu')),indent=2)+'\n')
        record['status']='compiled_only'
        record['structure']=check_native_projections(out,a.stage,a.precision_policy)
        if a.prepare_only:return
        record.update(status='executing',signatures=signatures)
        (out/'summary.json').write_text(json.dumps(record,indent=2)+'\n')
        run([exe('runtime','compiled_stage_file_runner'),out/'program.ftlpu',out/'inputs.txt',out/'actual',out/'runtime.pipeline.csv'],out/'execution.log',env)
        export_command=[exe('runtime','ftlpu_icu_program_export'),out/'actual.runtime.ftlpu',out/'runtime_icu_programs',
                        '--pre-execution-dir',out/'actual.pre_execution','--compute-imem-segments']
        record['icu_export_command']=list(map(str,export_command))
        run(export_command,out/'export.log',env)
        metrics=[]
        for i,gold in enumerate([expected,kv_ref]):
            actual=read_bf16(out/f'actual.{i}.bin',gold.shape).astype('f8');gold=gold.astype('f8')
            error=float(np.linalg.norm(actual-gold)/np.linalg.norm(gold))
            cosine=float(np.vdot(actual,gold)/(np.linalg.norm(actual)*np.linalg.norm(gold)))
            metrics.append(dict(relative_l2=error,cosine=cosine,finite=bool(np.isfinite(actual).all())))
        stats=dict(re.findall(r'(\w+)=(\d+)',(out/'execution.log').read_text().splitlines()[-1]))
        timing=trace_times(out/'runtime.pipeline.csv')
        timing['weight_page_wait_cycles']=int(stats['weight_page_runtime_wait_cycles'])
        timing['invocation_excluding_weight_page_wait_cycles']=(
            timing['invocation_cycles']-timing['weight_page_wait_cycles'])
        record.update(metrics=metrics,runtime=stats,timing=timing)
        assert all(m['finite'] and m['relative_l2']<.03 and m['cosine']>.999 for m in metrics),metrics
        assert stats['host_operations']=='0'
        record['status']='passed'
    except KeyboardInterrupt:
        record['status']='stopped_by_user';raise
    except Exception as e:
        record.update(status='failed',error=str(e));raise
    finally:
        record['command']=[sys.executable,*sys.argv]
        (out/'summary.json').write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps(record),flush=True)


if __name__=='__main__':main()
