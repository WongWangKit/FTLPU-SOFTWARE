#!/usr/bin/env python3
"""The common recognizer must find FFN and recurrent patterns across SSA views."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from qwen3_5_logical_graph import LogicalGraph
from qwen3_5_reference import Qwen35DeltaConfig
from qwen3_5_gated_deltanet_reference_test import random_weights
from qwen3_5_compiled_layer import bf16


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('graph-compile','opt','target-config','output-dir'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args(); out=a.output_dir.resolve(); out.mkdir(parents=True,exist_ok=True)
    c=Qwen35DeltaConfig(256,512,1,2,32,32)
    rng=np.random.default_rng(47); weights=random_weights(c,rng)
    packed,scales={},{}
    for name in ('in_proj_qkv','in_proj_z','in_proj_b','in_proj_a','out_proj','gate_proj','up_proj','down_proj'):
        w=getattr(weights,name); scales[name]=float(np.max(np.abs(w))/127)
        packed[name]=np.rint(w/scales[name]).astype(np.int8)
    graph=LogicalGraph()
    finals=graph.layer(bf16(rng.normal(0,.2,(32,256)).astype(np.float32)),weights,packed,scales,c)
    logical=graph.write(out/'logical',finals); kernel=out/'layer.kernel.mlir'
    subprocess.run([str(a.graph_compile.resolve()),'--input',str(logical),'--output',str(kernel),'--emit-kernel'],check=True)
    source=kernel.read_text(encoding='utf-8')
    assert source.count('func.func ')==1 and 'ftlpu.kernel.indexed_view' in source
    assert 'ftlpu.schedule.' not in source and 'page_storage_slices' not in source
    # Deliberately remove model/operator names. Matching must follow typed SSA,
    # not importer labels such as "ffn", "query_l2" or "delta_rule".
    source=re.sub(r'ftlpu.logical_name = "[^"]*"', 'ftlpu.logical_name = "anonymous"',source)
    kernel.write_text(source,encoding='utf-8')
    subprocess.run([str(a.opt.resolve()),'--input',str(kernel),'--output',str(out/'layer.schedule.mlir'),
        '--pipeline','ftlpu-kernel-to-schedule','--target-config',str(a.target_config.resolve()),
        '--mxm-execution','vector','--ffn-schedule','fused'],check=True)
    path=json.loads((out/'lowering_path.json').read_text())
    assert path['ffn_subgraphs']==1 and path['deltanet_subgraphs']==1
    plan=json.loads((out/'graph_plan.json').read_text())
    assert sum(n.get('native_ffn',False) for n in plan['nodes'])==1
    assert sum(n.get('stateful_mixer',False) for n in plan['nodes'])==32
    layouts=[n['direct_output_layouts'][1] for n in plan['nodes'] if n.get('stateful_mixer')]
    assert len({json.dumps(x,sort_keys=True) for x in layouts})==1
    assert plan['native_output_archive_copies']==0
    # A retained snapshot makes an earlier state live past the next scan.
    # The global planner must then allocate a second state rather than update
    # the snapshot in place.
    branch=LogicalGraph()
    q=bf16(rng.normal(0,.1,(1,64)).astype(np.float32))
    k=q.copy(); v=q.copy(); decay=np.full((1,2),.9,np.float32)
    beta=np.full((1,2),.5,np.float32); initial=np.zeros((64,32),np.float32)
    first,state0=branch.delta(q,k,v,decay,beta,initial,c)
    second,state1=branch.delta(q,k,v,decay,beta,state0,c)
    branch_dir=out/'state_snapshot'; branch_dir.mkdir(exist_ok=True)
    logical=branch.write(branch_dir/'logical',[second,state0,state1])
    kernel=branch_dir/'layer.kernel.mlir'
    subprocess.run([str(a.graph_compile.resolve()),'--input',str(logical),'--output',str(kernel),'--emit-kernel'],check=True)
    subprocess.run([str(a.opt.resolve()),'--input',str(kernel),'--output',str(branch_dir/'layer.schedule.mlir'),
        '--pipeline','ftlpu-kernel-to-schedule','--target-config',str(a.target_config.resolve())],check=True)
    branch_plan=json.loads((branch_dir/'graph_plan.json').read_text())
    states=[n['direct_output_layouts'][1] for n in branch_plan['nodes'] if n.get('stateful_mixer')]
    assert len(states)==2 and states[0]!=states[1]
    # Reject invalid view indices in the IR verifier, before any lowering.
    bad=out/'bad.kernel.mlir'
    bad.write_text('''module { func.func @bad(%a: tensor<1x1xbf16>) -> tensor<1x1xbf16> {
      %v = "ftlpu.kernel.indexed_view"(%a) {sources = array<i64: 0>, indices = array<i64: 1>}
        : (tensor<1x1xbf16>) -> tensor<1x1xbf16>
      return %v : tensor<1x1xbf16>
    } }''')
    failed=subprocess.run([str(a.opt.resolve()),'--input',str(bad),'--output',str(out/'bad.out.mlir'),
        '--pipeline','ftlpu-kernel-to-schedule'],capture_output=True,text=True)
    assert failed.returncode!=0 and 'out of bounds' in failed.stderr
    print('shared decoder graph recognition and indexed view validation passed')


if __name__=='__main__': main()
