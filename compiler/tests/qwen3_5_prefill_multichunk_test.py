"""Real-weight chunk32 GDN attention, with device-resident inter-chunk State.

Python builds equations/views and independent references only. The production
C++ graph compiler owns layouts, tiling, paging and raw-macro scheduling.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import re
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from qwen3_5_logical_graph import LogicalGraph
from qwen3_5_compiled_layer import load_imported_layer, bf16
from qwen3_5_reference import causal_depthwise_conv1d
from qwen3_5_decode_gdn_attention_test import run
from qwen3_5_qkv_conv_prefill_test import compare
from causal_exp_difference_runtime_test import causal_exp_source
from weighted_gram_runtime_test import scaled_weighted_gram_source, weighted_cross_gram_source
from unit_lower_solve_runtime_test import source as solve_source
from head_weighted_product_runtime_test import source as product_source


def grouped(value, heads, rows, cols):
    """Native head-product ABI: even heads then odd heads."""
    return value.reshape(heads//2, 2, rows, cols).transpose(1, 0, 2, 3).reshape(heads*rows, cols)


def canonical(value, heads, rows, cols):
    return value.reshape(2, heads//2, rows, cols).transpose(1, 0, 2, 3).reshape(heads, rows, cols)


def gemm(graph, name, a, b):
    shape = (a.shape[0], a.shape[1], b.shape[2])
    at, bt, ot = graph.type(a.shape, 'bf16'), graph.type(b.shape, 'bf16'), graph.type(shape, 'f32')
    src = f'''module {{ func.func @stage(%a: {at}, %b: {bt}) -> {ot} {{
      %o = ftlpu.kernel.batch_matmul %a, %b {{transpose_rhs = false, role = "pv"}} : ({at}, {bt}) -> {ot}
      return %o : {ot}
    }} }}'''
    return graph.run(name, src, [(a, 'bf16'), (b, 'bf16')], [(shape, 'f32')])[0]


def build(graph, x, w, packed, scales, c):
    h, k, d = c.num_value_heads, c.key_head_dim, c.value_head_dim
    qkv = graph.project('in_proj_qkv', x, packed['in_proj_qkv'], scales['in_proj_qkv'])
    # One causal convolution over the entire sequence retains token-31 history.
    conv = graph.conv(qkv, w.conv1d)
    finals, manifest, state = [], [], None
    def export(name, value, layout='logical'):
        manifest.append(dict(name=name, shape=list(value.shape), dtype=value.dtype, layout=layout))
        finals.append(value)
    for first in range(0, len(x), 32):
        start_node = len(graph.nodes)
        name = f'chunk{first//32}'
        mixed = conv[first:first+32]
        q = graph.norm('query_l2', mixed[:, :h*k].reshape(-1, k), np.ones(k), 1e-6, False, 1/np.sqrt(k)).reshape(32, h*k)
        key = graph.norm('key_l2', mixed[:, h*k:2*h*k].reshape(-1, k), np.ones(k), 1e-6, False).reshape(32, h*k)
        v = mixed[:, 2*h*k:]
        beta, g = graph.gates(x[first:first+32], packed, scales, w.dt_bias, w.a_log)
        gt = graph.vector_tiles('g_prefix_sum', [(g, 'f32')], lambda a:a.add(0, kind='cumsum_rows', immediate=0))
        gamma = graph.vector_tiles('chunk_gamma', [(gt, 'f32')], lambda a:a.add(0, kind='exp', immediate=0))
        r = graph.run('chunk_causal_decay', causal_exp_source(32, h, True, False, False), [(gt, 'f32')], [((32*h, 32), 'bf16')])[0]
        lower = graph.run('chunk_weighted_gram', scaled_weighted_gram_source(h,k),
            [(key,'bf16'),(r,'bf16'),(beta,'f32'),(np.tri(32,k=-1,dtype='f4'),'bf16')], [((h*32,32),'f32')])[0]
        rhs = graph.run('chunk_rhs_beta_v', solve_source(h,d,'head_scale'),
            [(v,'bf16'),(beta,'f32')], [((h*32,d),'f32')])[0]
        old_output = None
        if state is not None:
            # State remains FP32 in MEM; only the two MXM contractions consume
            # its BF16 high planes. No host readback or reset between chunks.
            old = canonical(graph.truncate_bf16('state_mxm_view', state),h,k,d)
            ks = gemm(graph,'incoming_ks',key.reshape(32,h,k).transpose(1,0,2),old)
            qs = gemm(graph,'incoming_qs',q.reshape(32,h,k).transpose(1,0,2),old)
            coeff = graph.vector_tiles('beta_gamma',[(beta,'f32'),(gamma,'f32')],lambda a:a.add(0,1))
            def expand(t):return grouped(t.T.reshape(h,32,1).repeat(d,2),h,32,d)
            correction=graph.vector_tiles('rhs_old_state',[(grouped(ks,h,32,d),'f32'),(expand(coeff),'f32')],lambda a:a.add(0,1))
            rhs=graph.vector_tiles('rhs_corrected',[(rhs,'f32'),(correction,'f32')],lambda a:a.add(0,1,kind='subtract'))
            old_output=graph.vector_tiles('output_old_state',[(grouped(qs,h,32,d),'f32'),(expand(gamma),'f32')],lambda a:a.add(0,1))
        solved=graph.run('chunk_unit_lower_solve',solve_source(h,d),[(lower,'f32'),(rhs,'f32')],[((h*32,d),'f32')])[0]
        hh=graph.run('chunk_output_cross_gram',weighted_cross_gram_source(h,k),[(q,'bf16'),(key,'bf16'),(r,'bf16')],[((h*32,32),'f32')])[0]
        output,update=graph.run('chunk_output_state',product_source(h,k,d,True),
            [(key,'bf16'),(r,'bf16'),(solved,'f32'),(hh,'f32')],[((h*32,d),'f32'),((h*k,d),'f32')])
        if state is not None:
            output=graph.vector_tiles('output_add_old',[(old_output,'f32'),(output,'f32')],lambda a:a.add(0,1,kind='add'))
            last=grouped(gamma[-1:].reshape(h,1,1).repeat(k,1).repeat(d,2),h,k,d)
            state=graph.vector_tiles('state_add_old',[(state,'f32'),(last,'f32'),(update,'f32')],lambda a:a.add(a.add(0,1),2,kind='add'))
        else:
            state=update
        z=graph.project('in_proj_z',x[first:first+32],packed['in_proj_z'],scales['in_proj_z'])
        core=canonical(output,h,32,d).transpose(1,0,2).reshape(32*h,d)
        norm=graph.norm('delta_output_norm',graph.truncate_bf16('output_bf16',core),w.delta_norm,c.rms_norm_eps)
        gated=graph.vector_tiles('delta_output_gate',[(norm,'bf16'),(z.reshape(32*h,d),'bf16')],
            lambda a:a.add(0,a.add(1,a.sigmoid(1)),dtype='bf16'))
        attn=graph.project('out_proj',gated.reshape(32,h*d),packed['out_proj'],scales['out_proj'])
        for label,value,layout in [('core',output,'grouped'),('state',state,'grouped'),('attention',attn,'logical')]:
            export(name+'.'+label,value,layout)
        # Unique names do not prescribe placement or cycle decisions.
        for node in graph.nodes[start_node:]:node['name']=name+'.'+node['name']
    return finals,manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('package','build-dir','target-config','output-dir'):p.add_argument('--'+arg,type=Path,required=True)
    p.add_argument('--seq-len',type=int,default=64)
    modes=p.add_mutually_exclusive_group()
    modes.add_argument('--validate-only',action='store_true')
    modes.add_argument('--compile-only',action='store_true')
    modes.add_argument('--run-only',action='store_true')
    p.add_argument('--compiler-executable',type=Path)
    a=p.parse_args()
    assert a.seq_len in (32,64)
    out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    x,w,packed,scales,c=load_imported_layer(a.package)
    assert c.num_key_heads==c.num_value_heads==16
    # Reuse the existing BF16 fixture for chunk0. The independent second half
    # prevents accidental replay of chunk0 from satisfying seq64 validation.
    if a.seq_len>len(x):
        rng=np.random.default_rng(3526110464)
        x=np.concatenate((x,bf16(rng.normal(float(x.mean()),float(x.std()),(a.seq_len-len(x),c.hidden_size)).astype('f4'))))
    x=bf16(x[:a.seq_len]);h,k,d=c.num_value_heads,c.key_head_dim,c.value_head_dim
    exe=lambda area,n:a.build_dir.resolve()/area/(n+('.exe' if sys.platform=='win32' else ''))
    if not a.validate_only:
        (out/'summary.json').write_text(json.dumps(dict(status='pending',seq_len=a.seq_len)))
    if not a.validate_only and not a.run_only:
        graph=LogicalGraph();finals,manifest=build(graph,x,w,packed,scales,c)
        (out/'outputs.json').write_text(json.dumps(manifest,indent=2));x.astype('<f4').tofile(out/'fixture.f32.bin')
        logical=graph.write(out/'logical_graph',finals);kernel=out/'frontend.kernel.mlir';binary=out/'frontend.ftlpu'
        run([exe('compiler','ftlpu-graph-compile'),'--input',logical,'--output',kernel,'--emit-kernel','--target-config',a.target_config.resolve()],out/'import.log')
        kernel.write_text(kernel.read_text().replace('module {','module attributes {ftlpu.vector_bf16_spill = true, ftlpu.common_barrier_protocol = true, ftlpu.vector_fp32_ingress = true, ftlpu.vector_bf16_token_halves = true, ftlpu.vector_log_softplus = true, ftlpu.l2_head_partition = true, ftlpu.sxm_wide = true} {',1))
        run([a.compiler_executable.resolve() if a.compiler_executable else exe('compiler','ftlpu-compile'),'--input',kernel,'--input-stage','kernel','--output',binary,'--target-config',a.target_config.resolve(),'--icu-compression','raw-macro','--mxm-execution','ws'],out/'compile.log')
        if a.compile_only:return
    if not a.validate_only:
        os.environ.update(FTLPU_FILE_RUNNER_MODEL_SESSION='1',FTLPU_FILE_RUNNER_SXM_ARCHITECTURE='wide',FTLPU_SESSION_PROGRESS='1')
        run([exe('runtime','compiled_stage_file_runner'),out/'frontend.ftlpu',out/'inputs.txt',out/'actual',out/'runtime.pipeline.csv'],out/'execution.log')
    manifest=json.loads((out/'outputs.json').read_text());actual={}
    for i,item in enumerate(manifest):
        raw=np.fromfile(out/f'actual.{i}.bin','<f4' if item['dtype']=='f32' else '<u2')
        if item['dtype']=='bf16':raw=(raw.astype('u4')<<16).view('f4')
        value=raw.reshape(item['shape'])
        if item['layout']=='grouped':value=canonical(value,h,k if item['name'].endswith('state') else 32,d)
        actual[item['name']]=value
    conv,_=causal_depthwise_conv1d(x@w.in_proj_qkv.T,w.conv1d,np.zeros_like(w.conv1d))
    q=conv[:,:h*k].reshape(a.seq_len,h,k);key=conv[:,h*k:2*h*k].reshape(a.seq_len,h,k);v=conv[:,2*h*k:].reshape(a.seq_len,h,d)
    q=q/np.sqrt(np.sum(q*q,axis=-1,keepdims=True)+1e-6)/np.sqrt(k)
    key=key/np.sqrt(np.sum(key*key,axis=-1,keepdims=True)+1e-6)
    beta=1/(1+np.exp(-(x@w.in_proj_b.T)))
    g=-np.exp(w.a_log)*np.logaddexp(0,x@w.in_proj_a.T+w.dt_bias)
    state=np.zeros((h,k,d),dtype='f8');core=np.empty((a.seq_len,h,d),dtype='f8');states=[]
    for t in range(a.seq_len):
        state*=np.exp(g[t]).astype('f8')[:,None,None]
        delta=beta[t,:,None]*(v[t]-np.einsum('hk,hkv->hv',key[t],state))
        state+=np.einsum('hk,hv->hkv',key[t],delta)
        core[t]=np.einsum('hk,hkv->hv',q[t],state)
        if t%32==31:states.append(state.copy())
    z=(x@w.in_proj_z.T).reshape(a.seq_len,h,d)
    norm=core/np.sqrt(np.mean(core*core,axis=-1,keepdims=True)+c.rms_norm_eps)*bf16(w.delta_norm)
    attention=(norm*(z/(1+np.exp(-z)))).reshape(a.seq_len,h*d)@w.out_proj.T
    errors={}
    for i,first in enumerate(range(0,a.seq_len,32)):
        refs=dict(query=q[first:first+32].reshape(32,-1),key=key[first:first+32].reshape(32,-1),beta=beta[first:first+32],g=g[first:first+32],
                  core=core[first:first+32].transpose(1,0,2),state=states[i],attention=attention[first:first+32])
        for label,ref in refs.items():
            name=f'chunk{i}.{label}'
            if name in actual:errors[name]=compare(actual[name],ref)
        for label in ('core','state'):
            name=f'chunk{i}.{label}'
            errors[name]['max_head_nrmse']=max(compare(actual[name][head],refs[label][head])['nrmse'] for head in range(h))
    boundary={}
    if a.seq_len>32:
        reference=core[32].astype('f8')
        # Resetting State gives only the current token's rank-one update.
        # Report this counterfactual to demonstrate that the test actually
        # exercises a significant incoming-State contribution at the boundary.
        reset=np.einsum('hk,hk->h',q[32],key[32])[:,None]*beta[32,:,None]*v[32]
        boundary=dict(token=32,actual=compare(actual['chunk1.core'][:,0],reference),
                      reset_state_counterfactual=compare(reset,reference))
    passed=all(np.isfinite(v['nrmse']) and v['nrmse']<.03 and v['cosine']>.999 for v in errors.values())
    passed &= all(v.get('max_head_nrmse',0)<.03 for v in errors.values())
    if boundary:passed &= boundary['actual']['nrmse']<.03
    log=(out/'execution.log').read_text();stats={k:int(v) for k,v in re.findall(r'(\w+)=(\d+)',log.splitlines()[-1])}
    passed &= stats.get('host_operations')==0
    with (out/'runtime.pipeline.csv').open() as f:rows=list(csv.DictReader(f))
    end=max(int(r['end'])+(int(r['inner_count'])-1)*int(r['inner_interval'])+(int(r['outer_count'])-1)*int(r['outer_interval']) for r in rows)
    host=sum(int(r['end'])-int(r['start']) for r in rows if r['resource']=='C2C.HostOutput')
    summary=dict(status='passed' if passed else 'failed',seq_len=a.seq_len,chunk_size=32,chunks=a.seq_len//32,real_weights=True,
        input='BF16 fixture; chunk0 package input, chunk1 independent seeded fixture',initial_state='zero',
        state_carried_on_device=True,conv_history_across_chunks=True,icu_compression='raw-macro',
        precision='FP32 State storage/add/decay; BF16 high planes for MXM State products; FP32 accumulations/solve',
        runtime=stats,runtime_trace_cycles=end,diagnostic_host_readback_cycles=host,errors=errors,chunk_boundary=boundary,
        scope='Complete GDN attention; no input RMSNorm, residual or FFN')
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
    run([exe('runtime','ftlpu_icu_program_export'),out/'actual.runtime.ftlpu',out/'runtime_icu_programs','--pre-execution-dir',out/'actual.pre_execution'],out/'export.log')
    assert passed,errors


if __name__=='__main__':main()
