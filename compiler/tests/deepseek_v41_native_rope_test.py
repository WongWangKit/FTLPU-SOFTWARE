"""Standalone interleaved RoPE through the same C++ lowering as connected SWA."""
import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np
from deepseek_v41_swa_test import Model
from deepseek_v41_qkv_norm_test import run, trace_times
from import_hf_decoder_layer import bf16_bits, read_bf16

def validate_rotary_pipeline(plan, enabled):
    groups=[n for n in plan['nodes'] if n.get('native_planar_rotary')]
    assert groups
    overlaps=0;tile_gaps=0
    for node in groups:
        assert bool(node.get('tile_pipeline'))==enabled
        assert node['feature_interval']==2
        assert node['feature_order']=='even_then_odd_per_tile'
        assert node['x_read_domains_per_tile_per_plane']==1
        if not enabled:continue
        assert len(node['buffers'])==4
        queues=set()
        for buf in node['buffers']:
            assert buf['words_per_plane']==32
            for sl in buf['slices']:
                key=(sl,buf['bank']);assert key not in queues, 'ping-pong buffers share a MEM ICU'
                queues.add(key)
        per_token={}
        for tile in node['tiles']:per_token.setdefault(tile['token_tile'],[]).append(tile)
        for tiles in per_token.values():
            tiles.sort(key=lambda t:t['slot'])
            pitch=node['feature_interval']
            assert pitch==2, 'each output requires two consecutive product beats'
            assert node['tile_compute_interval']==32*pitch
            for tile in tiles:
                assert tile['vxm_end']-tile['vxm_start']==32*pitch
                assert tile['input_sxm']<tile['vxm_start'], 'input transpose starts after computation'
                assert tile['output_sxm']>=tile['vxm_end'], 'output transpose starts before results are ready'
            for a,b in zip(tiles,tiles[1:]):
                assert b['vxm_start']>=a['vxm_end'], 'VXM tile executions overlap'
                tile_gaps+=b['vxm_start']-a['vxm_end']
                overlaps+=int(b['vxm_start']<=a['output_sxm']<b['vxm_end'])
    # Faster VXM makes fixed-latency SXM/dual-buffer reuse visible at tile
    # boundaries. Report these gaps instead of retaining the old 192-cycle
    # arithmetic pitch solely to hide transport behind slower computation.
    if enabled and any(len(n.get('tiles',[]))>1 for n in groups):
        assert overlaps>0, 'all output transposes became serialized'
    return dict(status='passed',enabled=enabled,groups=len(groups),overlapped_output_tiles=overlaps,
                inter_tile_gap_cycles=tile_gaps)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('build-dir','target-config','reference-dir','output-dir'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--synthetic',action='store_true')
    p.add_argument('--sequence',type=int,default=32)
    p.add_argument('--heads',type=int,default=2)
    p.add_argument('--head-dim',type=int,default=512)
    p.add_argument('--rotary-dim',type=int,default=64)
    p.add_argument('--rotary-offset',type=int,default=448)
    p.add_argument('--rope-tile-overlap',action='store_true')
    a=p.parse_args();out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    x=(np.random.default_rng(1907).normal(size=(a.sequence,a.heads,a.head_dim)).astype('f4') if a.synthetic
       else np.load(a.reference_dir/'q_pre_rope.npy')[:,:a.heads].copy())
    x=(bf16_bits(x).astype('u4')<<16).view('f4')
    m=Model(out);m.input('x',x)
    def rotary(name,value,inverse=False):
        return m.op(name,'rope',[value],x.shape,f'heads={a.heads}:i64,head_dim={a.head_dim}:i64,theta=1.000000e+04:f32,'
            f'rotary_dim={a.rotary_dim}:i64,rotary_offset={a.rotary_offset}:i64,interleaved=true,inverse={str(inverse).lower()}')
    y=rotary('rope','x');z=rotary('inverse',y,True)
    m.write([m.reshape('forward_output',y,(a.sequence,a.heads*a.head_dim)),m.reshape('inverse_output',z,(a.sequence,a.heads*a.head_dim))],rope_tile_overlap=a.rope_tile_overlap)
    def reference(value,sign):
        angle=np.arange(a.sequence,dtype='f8')[:,None,None]*10000**(-np.arange(0,a.rotary_dim,2,dtype='f8')/a.rotary_dim)[None,None,:]
        result=value.copy();tail=value[...,a.rotary_offset:a.rotary_offset+a.rotary_dim]
        rotated=(tail[...,::2]+1j*tail[...,1::2])*np.exp(sign*1j*angle)
        result[...,a.rotary_offset:a.rotary_offset+a.rotary_dim:2]=(bf16_bits(rotated.real.astype('f4')).astype('u4')<<16).view('f4')
        result[...,a.rotary_offset+1:a.rotary_offset+a.rotary_dim:2]=(bf16_bits(rotated.imag.astype('f4')).astype('u4')<<16).view('f4')
        return result
    expected=reference(x,1);inverse=reference(expected,-1)
    ext='.exe' if os.name=='nt' else '';exe=lambda area,name:a.build_dir.resolve()/area/(name+ext)
    env={k:v for k,v in os.environ.items() if not k.startswith('FTLPU_')}
    env.update(FTLPU_FILE_RUNNER_MODEL_SESSION='1',FTLPU_FILE_RUNNER_SXM_ARCHITECTURE='wide',FTLPU_GRAPH_PROGRESS='1')
    record=dict(status='running',relative_l2_limit=.001,scope='forward and inverse partial RoPE',
                synthetic=a.synthetic,shape=list(x.shape),rotary_dim=a.rotary_dim,rotary_offset=a.rotary_offset,
                rope_tile_overlap=a.rope_tile_overlap)
    try:
        run([exe('compiler','ftlpu-compile'),'--input',out/'kernel.mlir','--input-stage','kernel','--target-config',a.target_config.resolve(),
             '--icu-compression','raw-macro','--output',out/'program.ftlpu'],out/'compile.log',env)
        record['pipeline_structure']=validate_rotary_pipeline(json.loads((out/'graph_plan.json').read_text()),a.rope_tile_overlap)
        run([exe('runtime','compiled_stage_file_runner'),out/'program.ftlpu',out/'inputs.txt',out/'actual',out/'runtime.pipeline.csv'],out/'execution.log',env)
        run([exe('runtime','ftlpu_icu_program_export'),out/'actual.runtime.ftlpu',out/'runtime_icu_programs','--pre-execution-dir',out/'actual.pre_execution','--compute-imem-segments'],out/'export.log',env)
        metrics=[]
        for i,gold in enumerate([expected,inverse]):
            actual=read_bf16(out/f'actual.{i}.bin',gold.shape)
            untouched=np.ones(a.head_dim,dtype=bool);untouched[a.rotary_offset:a.rotary_offset+a.rotary_dim]=False
            prefix=int(np.count_nonzero(bf16_bits(actual[...,untouched])!=bf16_bits(x[...,untouched])))
            error=float(np.linalg.norm(actual-gold)/np.linalg.norm(gold))
            metrics.append(dict(relative_l2=error,prefix_bit_mismatches=prefix))
            assert prefix==0 and error<.001,metrics
        record.update(status='passed',metrics=metrics,timing=trace_times(out/'runtime.pipeline.csv'))
    except Exception as e:record.update(status='failed',error=str(e));raise
    finally:(out/'summary.json').write_text(json.dumps(record,indent=2)+'\n')

if __name__=='__main__':main()
