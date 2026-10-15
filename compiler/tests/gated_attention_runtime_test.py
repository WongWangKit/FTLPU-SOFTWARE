#!/usr/bin/env python3
"""Native GQA: wide Q/K norm, partial RoPE and sigmoid context gate."""
import argparse
from pathlib import Path
import subprocess
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from qwen3_5_full_attention import (attention_source,quantize,reference_attention,
    graph_layer,reference,bf16,LogicalGraph)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('compile','graph-compile','runner','output-dir','target-config'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--hidden-size',type=int,default=64)
    p.add_argument('--full-layer',action='store_true')
    a=p.parse_args();out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    hidden=a.hidden_size
    c=dict(hidden_size=hidden,intermediate_size=128,num_attention_heads=2,num_key_value_heads=1,
        head_dim=256,rms_norm_eps=1e-6,rope_parameters=dict(rope_theta=1e7,partial_rotary_factor=.25))
    rng=np.random.default_rng(35083)
    w={name:bf16(rng.normal(0,.12,shape).astype(np.float32)) for name,shape in (
        ('self_attn.q_proj',(1024,hidden)),('self_attn.k_proj',(256,hidden)),
        ('self_attn.v_proj',(256,hidden)),('self_attn.o_proj',(hidden,512)))}
    for name in ('q','k'):w['self_attn.'+name+'_norm']=bf16(rng.normal(0,.05,256).astype(np.float32))
    x=bf16(rng.normal(0,.5,(32,hidden)).astype(np.float32))
    if a.full_layer:
        c['intermediate_size']=384
        for name,shape in (('mlp.gate_proj',(384,hidden)),('mlp.up_proj',(384,hidden)),('mlp.down_proj',(hidden,384))):
            w[name]=bf16(rng.normal(0,.04,shape).astype(np.float32))
        for name in ('input_layernorm','post_attention_layernorm'):
            w[name]=bf16(rng.normal(0,.05,hidden).astype(np.float32))
    q,s,e=quantize(w,c)
    if a.full_layer:
        graph,result=graph_layer(x,w,q,s,c)
        expected=reference(x,e,c)
    else:
        graph=LogicalGraph()
        normalized=graph.norm('input_norm',x,np.ones(hidden,np.float32),1e-6)
        gate=graph.project('gate',normalized,q['attention_gate'],s['attention_gate'])
        inputs=[(normalized,'bf16')]+[(q[n].T.copy(),'i8') for n in ('query','key','value','output')]
        inputs += [(bf16(1+w['self_attn.'+n+'_norm']),'bf16') for n in ('q','k')]+[(gate,'bf16')]
        result=graph.run('gated_attention',attention_source(32,c,s),inputs,[(x.shape,'bf16')])[0]
        expected=reference_attention(x/np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+1e-6),e,c)
    logical=graph.write(out/'logical_graph',[result])
    expected.astype('<f4').tofile(out/'expected.f32.bin')
    subprocess.run([str(a.graph_compile.resolve()),'--input',str(logical),'--output',str(out/'schedule.mlir'),
        '--inputs-dir',str(out),'--target-config',str(a.target_config.resolve())],check=True)
    subprocess.run([str(a.compile.resolve()),'--input',str(out/'schedule.mlir'),'--input-stage','verified-schedule',
        '--output',str(out/'attention.ftlpu'),'--target-config',str(a.target_config.resolve())],check=True)
    subprocess.run([str(a.runner.resolve()),str(out/'attention.ftlpu'),str(out/'inputs.txt'),str(out/'actual'),str(out/'runtime.pipeline.csv')],check=True)
    actual=(np.fromfile(out/'actual.0.bin','<u2').astype(np.uint32)<<16).view(np.float32).reshape(x.shape)
    error=np.linalg.norm(actual-expected)/np.linalg.norm(expected)
    cosine=np.dot(actual.ravel(),expected.ravel())/(np.linalg.norm(actual)*np.linalg.norm(expected))
    print(f'gated attention: nrmse={error:.7f}, cosine={cosine:.7f}',flush=True)
    assert np.isfinite(actual).all() and error<.03 and cosine>.999, (error,cosine)
    print(f'gated attention passed: nrmse={error:.7f}, cosine={cosine:.7f}')


if __name__=='__main__':main()
