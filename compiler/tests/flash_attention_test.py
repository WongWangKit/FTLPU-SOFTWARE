"""Real raw-macro online-softmax attention with tail tiles and GQA isolation."""
import argparse
import json
from pathlib import Path
import numpy as np
from activation_batch_matmul_test import LogicalGraph, bf16, compile_graph, run
from attention_graph import flash_attention


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('build-dir','target-config','output-dir'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--case',choices=('negative','gqa','causal','query_tiles','causal_offset','single_tile','wide_reduction','flattened_q','odd_reduction','single_key','masked_singleton_tail'),default='negative')
    a=p.parse_args(); rng=np.random.default_rng(5731)
    batch,qt,kt,qh,kh,dim,dv,causal,position = {
        'negative':(1,1,33,2,1,32,32,False,0),
        'gqa':(3,1,65,4,2,32,64,False,0),
        'causal':(1,3,35,4,2,32,32,True,32),
        'query_tiles':(1,17,49,4,2,32,32,True,32),
        # The second KV tile is entirely masked for the first query row,
        # while later rows must still consume it and rescale correctly.
        'causal_offset':(1,3,35,4,2,32,32,True,31),
        # With no online merge the final normalization reads MXM's segmented
        # FP32 output directly (64+32 value channels and a partial query tile).
        'single_tile':(1,2,17,2,1,32,96,True,15),
        # Four reduction tiles reuse the same prepared Q across two KV blocks;
        # the second block has 15 live keys, and PV spans two output tiles.
        'wide_reduction':(1,3,47,8,2,128,128,False,0),
        # GLM-shaped Q arrives as one flattened external row. Its logical
        # reshape creates producer references, not a native [head, D] home.
        'flattened_q':(1,1,33,32,2,128,128,True,32),
        # Three K chunks end the alternating accumulator on the opposite
        # side from D=128; V also has a final 32-channel output tile.
        'odd_reduction':(1,2,37,6,2,96,96,False,0),
        'single_key':(1,2,1,6,3,64,96,False,0),
        'masked_singleton_tail':(1,2,33,4,2,64,96,True,31),
    }[a.case]
    random=lambda shape:bf16(rng.normal(0,.4,shape).astype('f4'))
    q=random((batch,qt,qh,dim)); k=random((batch,kt,kh,dim)); v=random((batch,kt,kh,dv))
    if a.case=='negative':
        # All logits negative; the second KV tile changes the running maximum.
        # A zero-seeded max with underflow would silently produce NaNs/zeros.
        q.fill(-4); k.fill(4); k[:,32:]=bf16(np.full_like(k[:,32:],3.5))
    if a.case=='single_key':
        # The only probability is one for every query. Different head and
        # channel values expose erroneous broadcasts across GQA groups.
        v=bf16(v*np.linspace(.125,16,dv,dtype='f4'))
    if a.case=='masked_singleton_tail':
        # The last key is masked for query 0. For query 1 its score is
        # roughly -40, -1, +3 or +40 across heads, so its contribution
        # ranges from negligible to dominant and shares no row coefficient.
        k[:,:32]=0; k[:,32:]=1
        q=bf16(.05*q+np.array([-5,-.125,.375,5],dtype='f4')[None,None,:,None])
    graph=LogicalGraph()
    query=q
    if a.case=='flattened_q':
        flat=q.reshape(batch*qt,qh*dim)
        ty=graph.type(flat.shape,'bf16')
        source=f'''module {{ func.func @stage(%q: {ty}) -> {ty} {{
          %o = ftlpu.kernel.reshape %q : ({ty}) -> {ty}
          return %o : {ty}
        }} }}'''
        query=graph.run('flattened_query_input',source,[(flat,'bf16')],[(flat.shape,'bf16')])[0].reshape(q.shape)
    output=flash_attention(graph,'attention',query,k,v,causal=causal,query_position=position)
    folder=a.output_dir.resolve()
    exe,env,binary=compile_graph(graph,[output.reshape(batch*qt,qh*dv)],folder,
        a.build_dir.resolve(),a.target_config.resolve(),fp32_spills=True)
    run([exe('runtime','compiled_stage_file_runner'),binary,folder/'inputs.txt',folder/'actual',folder/'runtime.pipeline.csv'],folder/'execution.log',env)
    run([exe('runtime','ftlpu_icu_program_export'),folder/'actual.runtime.ftlpu',folder/'icu_programs',
         '--pre-execution-dir',folder/'actual.pre_execution'],folder/'export.log',env)
    keys=np.repeat(k,qh//kh,axis=2); values=np.repeat(v,qh//kh,axis=2)
    score=np.einsum('bqhd,bkhd->bhqk',q.astype('f8'),keys)/np.sqrt(dim)
    if causal:
        score=np.where(np.arange(kt)[None,None,None,:]<=position+np.arange(qt)[None,None,:,None],score,-np.inf)
    probability=np.exp(score-score.max(-1,keepdims=True)); probability/=probability.sum(-1,keepdims=True)
    expected=np.einsum('bhqk,bkhd->bqhd',probability,values)
    actual=(np.fromfile(folder/'actual.0.bin','<u2').astype('u4')<<16).view('f4').reshape(expected.shape)
    relative=float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))
    assert np.isfinite(actual).all() and relative<.03,(a.case,relative)
    if a.case=='single_key':
        # With one valid finite key the output is the original BF16 V.
        assert np.array_equal(actual,np.broadcast_to(values,actual.shape))
    record=dict(status='passed',case=a.case,relative_l2=relative,shape=dict(batch=batch,q=qt,kv=kt,qh=qh,kh=kh,d=dim,dv=dv),
        precision='BF16 Q/K/V, QK score, scaled score and P; FP32 matmul accumulation and online m/l/U',icu_encoding='raw-macro')
    (folder/'summary.json').write_text(json.dumps(record,indent=2));print(record)


if __name__=='__main__': main()
