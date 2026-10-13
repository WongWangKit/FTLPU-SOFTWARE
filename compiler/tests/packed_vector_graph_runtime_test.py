#!/usr/bin/env python3
"""Real-width packed elementwise graphs execute on ordinary VXM ICU/FU."""
import argparse
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from qwen3_5_compiled_layer import CompiledStages,VectorGraph,bf16

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('compile','runner','output-dir'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    stages=CompiledStages(a.compile.resolve(),a.runner.resolve(),a.output_dir)
    rng=np.random.default_rng(3508)
    gate,up=[bf16(rng.normal(0,.2,(32,3584)).astype(np.float32)) for _ in range(2)]
    g=VectorGraph(stages,'packed_swiglu',[(gate,'bf16'),(up,'bf16')])
    activated=g.add(0,g.sigmoid(0),dtype='bf16')
    actual=g.finish(g.add(activated,1,dtype='bf16'))
    expected=bf16(bf16(gate/(1+np.exp(-gate)))*up)
    error=np.linalg.norm(actual-expected)/np.linalg.norm(expected)
    if not np.isfinite(actual).all() or error>.01:raise AssertionError(f'SwiGLU NRMSE={error}')
    g=VectorGraph(stages,'packed_residual',[(gate,'bf16'),(up,'bf16')])
    actual=g.finish(g.add(0,1,kind='add',dtype='bf16'))
    np.testing.assert_array_equal(actual,bf16(gate+up))
    print(f'packed seq32 width3584 SwiGLU/residual passed: swiglu_nrmse={error}')

if __name__=='__main__':main()
