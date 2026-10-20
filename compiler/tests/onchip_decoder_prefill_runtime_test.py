#!/usr/bin/env python3
"""Qwen2.5-sized dense decoder with on-chip intermediates and matrix prefill."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from onchip_attention_prefill_runtime_test import attention_source
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from qwen3_5_logical_graph import LogicalGraph
from qwen3_5_compiled_layer import bf16


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('graph-compile', 'compile', 'runner', 'target-config', 'output-dir'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--compile-only', action='store_true')
    p.add_argument('--check-only', action='store_true')
    a = p.parse_args()
    out = a.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    m, h, q, k, d, hidden = 32, 1536, 12, 2, 128, 8960
    rng = np.random.default_rng(161021)
    x = bf16(rng.normal(0, .3, (m, h)).astype(np.float32))
    gammas = [bf16(rng.normal(1, .02, h).astype(np.float32)) for _ in range(2)]
    shapes = [(h,q*d),(h,k*d),(h,k*d),(q*d,h)]
    weights = [rng.integers(-8, 9, shape, dtype=np.int8) for shape in shapes]
    biases = [bf16(rng.normal(0,.02,n*d).astype(np.float32)) for n in (q,k,k)]
    ffn = {name:rng.integers(-8,9,shape,dtype=np.int8) for name,shape in (
        ('gate_proj',(hidden,h)),('up_proj',(hidden,h)),('down_proj',(h,hidden)))}
    scale = .005
    if not a.check_only:
        g = LogicalGraph()
        norm = g.norm('input_norm', x, gammas[0], 1e-6)
        attn = g.run('attention', attention_source(m,h,q,k,d,scale,True),
            [(norm,'bf16')]+[(w,'i8') for w in weights]+[(b,'bf16') for b in biases],
            [((m,h),'bf16')])[0]
        residual = g.vector_tiles('attention_residual',[(x,'bf16'),(attn,'bf16')],
            lambda v:v.add(0,1,kind='add',dtype='bf16'))
        normalized = g.norm('post_attention_norm',residual,gammas[1],1e-6)
        down = g.ffn(normalized,ffn,{name:scale for name in ffn})
        result = g.vector_tiles('ffn_residual',[(residual,'bf16'),(down,'bf16')],
            lambda v:v.add(0,1,kind='add',dtype='bf16'))
        logical = g.write(out/'logical',[result])
        subprocess.run([str(a.graph_compile.resolve()),'--input',str(logical),
            '--output',str(out/'decoder.kernel.mlir'),'--emit-kernel'],check=True)
        subprocess.run([str(a.compile.resolve()),'--input',str(out/'decoder.kernel.mlir'),
            '--input-stage','kernel','--output',str(out/'decoder.ftlpu'),
            '--mxm-execution','auto','--ffn-schedule','tail','--projection-rope-overlap','off',
            '--target-config',str(a.target_config.resolve())],check=True)
    if a.compile_only:
        return
    if not a.check_only:
        subprocess.run([str(a.runner.resolve()),str(out/'decoder.ftlpu'),
            str(out/'inputs.txt'),str(out/'actual')],check=True)
    def rms(v,gamma):
        return bf16(v/np.sqrt(np.mean(v*v,axis=-1,keepdims=True)+1e-6)*gamma)
    norm = rms(x,gammas[0])
    qw,kw,vw,ow = [w.astype(np.float32)*scale for w in weights]
    query = (norm@qw+biases[0]).reshape(m,q,d)
    key = (norm@kw+biases[1]).reshape(m,k,d)
    value = (norm@vw+biases[2]).reshape(m,k,d)
    angle = np.arange(m,dtype=np.float32)[:,None]/np.float32(1e6)**(np.arange(0,d,2,dtype=np.float32)/d)
    co,si = np.cos(angle)[:,None,:],np.sin(angle)[:,None,:]
    def rope(v):
        lo,hi = v[:,:,:d//2],v[:,:,d//2:]
        return np.concatenate([lo*co-hi*si,hi*co+lo*si],axis=-1)
    query = rope(query).transpose(1,0,2)
    key = np.repeat(rope(key),q//k,axis=1).transpose(1,0,2)
    value = np.repeat(value,q//k,axis=1).transpose(1,0,2)
    score = (query@key.transpose(0,2,1))*np.float32(d**-.5)
    score = np.where(np.triu(np.ones((m,m),bool),1),-np.inf,score)
    prob = np.exp(score-np.max(score,axis=-1,keepdims=True))
    prob /= np.sum(prob,axis=-1,keepdims=True)
    attn = bf16((prob@value).transpose(1,0,2).reshape(m,q*d)@ow)
    residual = bf16(x+attn)
    norm = rms(residual,gammas[1])
    gate,up = [norm@(ffn[n].T.astype(np.float32)*scale) for n in ('gate_proj','up_proj')]
    down = bf16((gate/(1+np.exp(-gate))*up)@(ffn['down_proj'].T.astype(np.float32)*scale))
    expected = bf16(residual+down)
    actual = (np.fromfile(out/'actual.0.bin','<u2').astype(np.uint32)<<16).view(np.float32).reshape(m,h)
    error = float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))
    passed = bool(np.isfinite(actual).all() and error < .03)
    summary = dict(scope='complete_decoder_layer',weights='synthetic_nonzero_qk',
        mxm_execution='auto',intermediate_storage='on_chip',tokens=m,width=h,
        hidden=hidden,heads=q,kv_heads=k,head_dim=d,nrmse=error,passed=passed)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(summary)
    assert passed, 'decoder numerical check failed'


if __name__ == '__main__':
    main()
