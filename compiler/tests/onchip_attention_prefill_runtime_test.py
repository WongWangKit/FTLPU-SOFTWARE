#!/usr/bin/env python3
"""On-chip matrix-prefill attention regression with nonzero Q/K weights."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from qwen3_5_logical_graph import LogicalGraph
from qwen3_5_compiled_layer import bf16


def attention_source(m, hidden, heads, kvheads, dim, scale, with_bias=False):
    ty = lambda *shape: LogicalGraph.type(shape, 'bf16')
    wty = lambda *shape: LogicalGraph.type(shape, 'i8')
    qwidth, kvwidth = heads*dim, kvheads*dim
    a = ty(m,hidden)
    args = [('x',a),('qw',wty(hidden,qwidth)),('kw',wty(hidden,kvwidth)),
            ('vw',wty(hidden,kvwidth)),('ow',wty(qwidth,hidden))]
    if with_bias:
        args += [(n+'b',ty(heads_*dim)) for n,heads_ in [('q',heads),('k',kvheads),('v',kvheads)]]
    lines = ['module { func.func @attention('+', '.join('%'+n+': '+t for n,t in args)+') -> '+a+' {']
    def op(n,k,operands,types,out,attrs=''):
        lines.append(f' %{n} = "ftlpu.kernel.{k}"('+', '.join('%'+v for v in operands)+f') {{{attrs}}} : ('+', '.join(types)+f') -> {out}')
    for n,heads_ in [('q',heads),('k',kvheads),('v',kvheads)]:
        width=heads_*dim
        op(n,'matmul',['x',n+'w'],[a,wty(hidden,width)],ty(m,width),
           f'm = {m} : i64, n = {width} : i64, k = {hidden} : i64, unit = "MXM", rhs_scale = {scale:.9e} : f32')
        if with_bias:
            op(n+'biased','bias_add',[n,n+'b'],[ty(m,width),ty(width)],ty(m,width))
        op(n+'h','reshape',[n+'biased' if with_bias else n],[ty(m,width)],ty(m,heads_,dim))
        if n!='v':
            op(n+'r','rope',[n+'h'],[ty(m,heads_,dim)],ty(m,heads_,dim),
               f'heads = {heads_} : i64, head_dim = {dim} : i64, theta = 1.0e6 : f32, rotary_dim = {dim} : i64')
    for n,src in [('k','kr'),('v','vh')]:
        op(n+'rep','gqa_broadcast',[src],[ty(m,kvheads,dim)],ty(m,heads,dim),
           f'query_heads = {heads} : i64, kv_heads = {kvheads} : i64')
    for n,src in [('q','qr'),('k','krep'),('v','vrep')]:
        op(n+'t','transpose',[src],[ty(m,heads,dim)],ty(heads,m,dim),'permutation = array<i64: 1, 0, 2>')
    op('score','batch_matmul',['qt','kt'],[ty(heads,m,dim)]*2,ty(heads,m,m),'role = "qk", transpose_rhs = true')
    op('prob','softmax',['score'],[ty(heads,m,m)],ty(heads,m,m),f'axis = -1 : i64, causal = true, scale = {dim**-.5:.9e} : f32')
    op('pv','batch_matmul',['prob','vt'],[ty(heads,m,m),ty(heads,m,dim)],ty(heads,m,dim),'role = "pv", transpose_rhs = false')
    op('ct','transpose',['pv'],[ty(heads,m,dim)],ty(m,heads,dim),'permutation = array<i64: 1, 0, 2>')
    op('c','reshape',['ct'],[ty(m,heads,dim)],ty(m,qwidth))
    op('o','matmul',['c','ow'],[ty(m,qwidth),wty(qwidth,hidden)],a,
       f'm = {m} : i64, n = {hidden} : i64, k = {qwidth} : i64, unit = "MXM", rhs_scale = {scale:.9e} : f32')
    return '\n'.join(lines+[f' return %o : {a}', '} }'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('graph-compile','compile','runner','target-config','output-dir'):
        p.add_argument('--'+n,type=Path,required=True)
    for n,v in [('width',256),('heads',2),('kv-heads',1),('head-dim',64),('seq-len',32)]:
        p.add_argument('--'+n,type=int,default=v)
    p.add_argument('--input-std',type=float,default=.3)
    p.add_argument('--retain-norm',action='store_true')
    p.add_argument('--with-rmsnorm',action='store_true')
    p.add_argument('--with-bias',action='store_true')
    p.add_argument('--compile-only',action='store_true')
    p.add_argument('--check-only',action='store_true')
    a=p.parse_args(); out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    m,h,q,k,d=a.seq_len,a.width,a.heads,a.kv_heads,a.head_dim
    rng=np.random.default_rng(161020); scale=.005
    x=bf16(rng.normal(0,a.input_std,(m,h)).astype(np.float32))
    packed=[rng.integers(-8,9,shape,dtype=np.int8) for shape in [(h,q*d),(h,k*d),(h,k*d),(q*d,h)]]
    biases=[bf16(rng.normal(0,.02,n*d).astype(np.float32)) if a.with_bias else np.zeros(n*d,np.float32) for n in (q,k,k)]
    gamma=bf16(rng.normal(1,.02,h).astype(np.float32))
    if not a.check_only:
        g=LogicalGraph()
        norm=g.norm('input_norm',x,gamma,1e-6) if a.with_rmsnorm else x
        result=g.run('attention',attention_source(m,h,q,k,d,scale,a.with_bias),[(norm,'bf16')]+[(w,'i8') for w in packed]+([(v,'bf16') for v in biases] if a.with_bias else []),[((m,h),'bf16')])[0]
        logical=g.write(out/'logical',[result,norm] if a.retain_norm and a.with_rmsnorm else [result])
        subprocess.run([str(a.graph_compile.resolve()),'--input',str(logical),'--output',str(out/'attention.kernel.mlir'),'--emit-kernel'],check=True)
        subprocess.run([str(a.compile.resolve()),'--input',str(out/'attention.kernel.mlir'),'--input-stage','kernel','--output',str(out/'attention.ftlpu'),'--mxm-execution','auto','--projection-rope-overlap','off','--target-config',str(a.target_config.resolve())],check=True)
    if a.compile_only:return
    if not a.check_only:
        subprocess.run([str(a.runner.resolve()),str(out/'attention.ftlpu'),str(out/'inputs.txt'),str(out/'actual')],check=True)
    if a.with_rmsnorm:
        x=bf16(x/np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+1e-6)*gamma)
    norm_error=None
    if a.retain_norm and a.with_rmsnorm:
        norm_actual=(np.fromfile(out/'actual.1.bin','<u2').astype(np.uint32)<<16).view(np.float32).reshape(m,h)
        norm_error=float(np.linalg.norm(norm_actual-x)/np.linalg.norm(x))
    w=[v.astype(np.float32)*scale for v in packed]
    query=(x@w[0]+biases[0]).reshape(m,q,d);key=(x@w[1]+biases[1]).reshape(m,k,d);value=(x@w[2]+biases[2]).reshape(m,k,d)
    angle=np.arange(m,dtype=np.float32)[:,None]/np.float32(1e6)**(np.arange(0,d,2,dtype=np.float32)/d)
    co,si=np.cos(angle)[:,None,:],np.sin(angle)[:,None,:]
    def rope(v):
        lo,hi=v[:,:,:d//2],v[:,:,d//2:]
        return np.concatenate([lo*co-hi*si,hi*co+lo*si],axis=-1)
    query=rope(query).transpose(1,0,2);key=np.repeat(rope(key),q//k,axis=1).transpose(1,0,2)
    value=np.repeat(value,q//k,axis=1).transpose(1,0,2)
    score=(query@key.transpose(0,2,1))*np.float32(d**-.5)
    score=np.where(np.triu(np.ones((m,m),bool),1),-np.inf,score)
    prob=np.exp(score-np.max(score,axis=-1,keepdims=True));prob/=np.sum(prob,axis=-1,keepdims=True)
    expected=(prob@value).transpose(1,0,2).reshape(m,q*d)@w[3]
    actual=(np.fromfile(out/'actual.0.bin','<u2').astype(np.uint32)<<16).view(np.float32).reshape(m,h)
    err=float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))
    result=dict(scope='rmsnorm_attention' if a.with_rmsnorm else 'isolated_attention',weights='synthetic_nonzero_qk',projection_bias=a.with_bias,mxm_execution='auto',intermediate_storage='on_chip',input_std=a.input_std,retained_norm_nrmse=norm_error,tokens=m,width=h,heads=q,kv_heads=k,head_dim=d,nrmse=err if np.isfinite(err) else None,passed=bool(np.isfinite(actual).all() and err<.03 and (norm_error is None or norm_error<.01)))
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
    assert result['passed'], 'attention numerical check failed'

if __name__=='__main__':main()
