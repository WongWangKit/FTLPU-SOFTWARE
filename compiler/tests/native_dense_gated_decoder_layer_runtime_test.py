#!/usr/bin/env python3
"""One dense Kernel function: gated GQA, residual, RMSNorm and native FFN."""
import argparse
from pathlib import Path
import re
import subprocess
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from qwen3_5_full_attention import (bf16,quantize,reference,
    dense_layer_source,dense_layer_inputs)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('opt','compile','runner','target-config','output-dir'):
        parser.add_argument('--'+name,type=Path,required=True)
    a=parser.parse_args();out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    c=dict(hidden_size=256,intermediate_size=384,num_attention_heads=2,
        num_key_value_heads=1,head_dim=256,rms_norm_eps=1e-6,
        rope_parameters=dict(rope_theta=1e7,partial_rotary_factor=.25))
    rng=np.random.default_rng(35084)
    weights={name:bf16(rng.normal(0,.04,shape).astype(np.float32)) for name,shape in (
        ('self_attn.q_proj',(1024,256)),('self_attn.k_proj',(256,256)),
        ('self_attn.v_proj',(256,256)),('self_attn.o_proj',(256,512)),
        ('mlp.gate_proj',(384,256)),('mlp.up_proj',(384,256)),('mlp.down_proj',(256,384)))}
    for name in ('self_attn.q_norm','self_attn.k_norm','input_layernorm','post_attention_layernorm'):
        weights[name]=bf16(rng.normal(0,.05,256).astype(np.float32))
    x=bf16(rng.normal(0,.5,(32,256)).astype(np.float32))
    packed,scales,effective=quantize(weights,c)
    (out/'layer.kernel.mlir').write_text(dense_layer_source(32,c,scales))
    roles={2:'query',3:'key',4:'value',5:'output',8:'attention_gate',10:'gate',11:'up',12:'down'}
    inputs=[]
    for i,(value,dtype) in enumerate(dense_layer_inputs(x,weights,packed)):
        path=out/f'input{i}.bin'
        if dtype=='bf16':(bf16(value).view(np.uint32)>>16).astype('<u2').tofile(path)
        else:value.astype('i1').tofile(path)
        inputs.append(str(path)+(f'\t{scales[roles[i]]:.9e}' if dtype=='i8' else ''))
    (out/'inputs.txt').write_text('\n'.join(inputs)+'\n')
    common=['--target-config',str(a.target_config.resolve()),'--weight-bank','0',
        '--mxm-execution','vector','--ffn-schedule','fused','--projection-rope-overlap','off']
    for src,dst,pipeline in [('layer.kernel.mlir','layer.stream.mlir','ftlpu-kernel-to-stream'),
                           ('layer.stream.mlir','layer.schedule.mlir','ftlpu-stream-to-schedule')]:
        subprocess.run([str(a.opt.resolve()),'--input',str(out/src),'--output',str(out/dst),
            '--pipeline',pipeline,*common],check=True)
    stream=(out/'layer.stream.mlir').read_text()
    assert stream.count('ftlpu.stream.swish_task')==1
    assert stream.count('ftlpu.stream.softmax_task')==1
    assert stream.count('kind = "linear"')==1, 'FFN must not be split into standalone projections'
    assert 'ftlpu.graph_lowering' not in stream and 'ftlpu.connected_graph_schedule' not in stream
    # Initializer lifetime includes asynchronous FFN prefetch, before softmax.
    # No resident causal mask may alias a page's recyclable weight slice.
    weight_slices=set()
    for values in re.findall(r'page_storage_slices = \[([^]]+)\]',stream):
        weight_slices.update(map(int,values.split(',')))
    masks=re.findall(r'\bcausal_mask(?:_mxm1)? = \{([^{}]+)\}',stream)
    assert masks
    for mask in masks:
        values=re.search(r'slices = \[([^]]+)\]',mask)[1]
        assert weight_slices.isdisjoint(map(int,values.split(','))), \
            'FFN weight prefetch must not overwrite a live causal mask'
    for placement in re.findall(r'\b(?:query|key)_norm_weight = \{([^{}]+)\}',stream):
        assert 'bank_locked = true' in placement, \
            'head-norm gamma must retain the whole-function constant allocation'
    subprocess.run([str(a.compile.resolve()),'--input',str(out/'layer.schedule.mlir'),
        '--input-stage','verified-schedule','--output',str(out/'layer.ftlpu'),
        '--target-config',str(a.target_config.resolve())],check=True)
    subprocess.run([str(a.compile.resolve()),'--input',str(out/'layer.kernel.mlir'),
        '--input-stage','kernel','--output',str(out/'direct.ftlpu'),*common],check=True)
    assert (out/'direct.ftlpu').read_bytes()==(out/'layer.ftlpu').read_bytes(), \
        'direct compiler entry must use the same dense passes as the staged pipeline'
    subprocess.run([str(a.runner.resolve()),str(out/'layer.ftlpu'),str(out/'inputs.txt'),
        str(out/'actual'),str(out/'runtime.pipeline.csv')],check=True)
    actual=(np.fromfile(out/'actual.0.bin','<u2').astype(np.uint32)<<16).view(np.float32).reshape(x.shape)
    expected=reference(x,effective,c)
    error=np.linalg.norm(actual-expected)/np.linalg.norm(expected)
    cosine=np.dot(actual.ravel(),expected.ravel())/(np.linalg.norm(actual)*np.linalg.norm(expected))
    print(f'native dense decoder: nrmse={error:.7f}, cosine={cosine:.7f}',flush=True)
    assert np.isfinite(actual).all() and error<.03 and cosine>.999,(error,cosine)


if __name__=='__main__':main()
