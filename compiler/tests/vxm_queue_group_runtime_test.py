#!/usr/bin/env python3
"""The second physical ALU group preserves vector and recurrent arithmetic."""
import argparse
from pathlib import Path
import re
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from qwen3_5_compiled_layer import CompiledStages,VectorGraph,bf16
from qwen3_5_reference import Qwen35DeltaConfig

class SecondGroup(CompiledStages):
    def run(self,name,source,inputs,outputs):
        source=re.sub(r'(@stage\([^{}]+?\)\s*->\s*(?:\([^{}]+\)|tensor<[^>]+>))\s*\{',
            r'\1 attributes {ftlpu.vxm_queue_base = 8 : i64} {',source,count=1,flags=re.S)
        return super().run(name,source,inputs,outputs)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('compile','runner','output-dir'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();s=SecondGroup(a.compile.resolve(),a.runner.resolve(),a.output_dir)
    rng=np.random.default_rng(3610)
    x=bf16(rng.normal(0,.2,(3,64)).astype(np.float32))
    g=VectorGraph(s,'norm_group8',[(x,'bf16'),(np.ones((1,64),np.float32),'f32')])
    y=g.finish(g.norm(0,1,1e-6))
    ref=bf16(x/np.sqrt(np.mean(x*x,axis=1,keepdims=True)+1e-6))
    np.testing.assert_allclose(y,ref,atol=.025,rtol=.02)
    c=Qwen35DeltaConfig(64,128,1,2,64,64)
    q,k,v=[bf16(rng.normal(0,.03,(2,128)).astype(np.float32)) for _ in range(3)]
    decay=np.full((2,2),.97,np.float32);beta=np.full((2,2),.5,np.float32)
    initial=rng.normal(0,.01,(128,64)).astype(np.float32)
    y,state=s.delta(q,k,v,decay,beta,initial,c)
    expected=np.empty_like(v);ref=initial.reshape(2,64,64).copy()
    for token in range(2):
        for head in range(2):
            qi,ki,vi=[z[token].reshape(2,64)[head] for z in (q,k,v)]
            ref[head]*=decay[token,head]
            delta=(vi-np.sum(ref[head]*ki[:,None],axis=0))*beta[token,head]
            ref[head]+=ki[:,None]*delta[None,:]
            expected[token,head*64:(head+1)*64]=bf16(np.sum(ref[head]*qi[:,None],axis=0))
    np.testing.assert_allclose(y,expected,atol=3e-6,rtol=.005)
    np.testing.assert_allclose(state,ref.reshape(128,64),atol=3e-7,rtol=2e-5)
    print('second ALU group norm and multi-head FP32 recurrence passed')

if __name__=='__main__':main()
