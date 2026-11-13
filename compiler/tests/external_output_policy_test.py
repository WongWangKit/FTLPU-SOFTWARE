"""Selective external results through the formal compiler and raw-macro CModel."""
import argparse
import csv
import json
from pathlib import Path
import re
import hashlib
import sys

import numpy as np
from activation_batch_matmul_test import LogicalGraph, bf16, compile_graph, run
from attention_graph import flash_attention
from dense_attention_core_test import reference
from glm4_decode_test import attention
from glm4_attention_comparison_test import execute
from model_test_outputs import OutputContract, TestOutput


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('build-dir','target-config','output-dir'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    build,target,root=(x.resolve() for x in (args.build_dir,args.target_config,args.output_dir))
    rng=np.random.default_rng(1111)
    q=bf16(rng.normal(0,.4,(1,2,2,32)).astype('f4'))
    k=bf16(rng.normal(0,.4,(1,17,1,32)).astype('f4'))
    v=bf16(rng.normal(0,.4,k.shape).astype('f4'))
    graph=LogicalGraph()
    context=flash_attention(graph,'attention',q,k,v,causal=True,query_position=15).reshape(2,64)
    # Two logical result roots deliberately alias the same producer. Selecting
    # result 1 must preserve its index and never download unselected result 0.
    outputs=[context,context]
    expected=reference(q,k,v,32**-.5,True,15).reshape(2,64)
    records={}
    for name,indices,attrs in (
        ('all',[0,1],()),
        ('selected',[1],('ftlpu.external_output_indices = [1 : i64]',)),
        ('resident',[],('ftlpu.external_output_indices = []',))):
        folder=root/name
        exe,env,binary=compile_graph(graph,outputs,folder,build,target,fp32_spills=True,module_attrs=attrs)
        run([exe('runtime','compiled_stage_file_runner'),binary,folder/'inputs.txt',folder/'actual',
             folder/'runtime.pipeline.csv'],folder/'execution.log',env)
        plan=json.loads((folder/'graph_plan.json').read_text())
        assert plan['external_output_indices']==indices
        assert [r['index'] for r in plan['resident_output_views']]==[i for i in range(2) if i not in indices]
        with (folder/'runtime.pipeline.csv').open(newline='') as stream:
            downloads=[r for r in csv.DictReader(stream) if r['resource']=='C2C.HostOutput']
        # HostOutput is one aggregate transfer window per invocation. The
        # compiler binding indices and actual output files identify its values.
        assert len(downloads)==int(bool(indices)),(name,downloads)
        schedule=(folder/'decoder_layer.schedule.mlir').read_text()
        bound=[int(i) for i in re.findall(r'ftlpu\.schedule\.binding[^\n]*access = "output"[^\n]*index = (\d+) : i64',schedule)]
        assert bound==indices,(name,bound)
        for index in range(2):
            assert (folder/f'actual.{index}.bin').exists()==(index in indices)
        assert f'outputs={len(indices)} ' in (folder/'execution.log').read_text()
        for index in indices:
            actual=(np.fromfile(folder/f'actual.{index}.bin','<u2').astype('u4')<<16).view('f4').reshape(expected.shape)
            error=float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))
            assert np.isfinite(actual).all() and error<.03,error
        records[name]=dict(status='passed',external_indices=indices,
                           host_output_count=len(downloads),nodes=plan['nodes'])
    assert (root/'all/actual.1.bin').read_bytes()==(root/'selected/actual.1.bin').read_bytes()
    assert records['all']['nodes']==records['selected']['nodes']==records['resident']['nodes']
    for name,value in (('negative','[-1 : i64]'),('out_of_range','[2 : i64]'),('duplicate','[1 : i64, 1 : i64]')):
        try:
            compile_graph(graph,outputs,root/name,build,target,fp32_spills=True,
                          module_attrs=(f'ftlpu.external_output_indices = {value}',))
        except RuntimeError as error:
            assert 'external_output_indices contains a duplicate or invalid result index' in str(error)
        else:
            raise AssertionError('invalid output policy accepted: '+name)
        records[name]=dict(status='rejected_as_expected')
    # Two KV tiles exercise the FP32 running-numerator path, unlike the
    # single-tile prefill case. Cache results must survive output selection.
    q=bf16(rng.normal(0,.4,(1,1,32,128)).astype('f4'))
    k=bf16(rng.normal(0,.4,(1,33,2,128)).astype('f4'))
    v=bf16(rng.normal(0,.4,k.shape).astype('f4'))
    graph,outputs=attention(q.reshape(1,4096), k[:,32].reshape(1,256), v[:,32].reshape(1,256),
        k[:,:32],v[:,:32],dict(num_attention_heads=32),strategy='flash')
    args.build_dir, args.target_config = build, target
    ext='.exe' if sys.platform=='win32' else ''
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    provenance=dict(compiler_sha256=digest(build/'compiler'/('ftlpu-compile'+ext)),
        runner_sha256=digest(build/'runtime'/('compiled_stage_file_runner'+ext)),
        clock_mhz=json.loads(target.read_text())['external_memory']['lpu_clock_mhz'])
    actual,phase=execute(args,1,'flash',(graph,outputs),root/'decode_contract',provenance,
        output_contract=OutputContract(TestOutput('context','diagnostic'),
            TestOutput('key_cache','persistent_state'),TestOutput('value_cache','persistent_state')))
    expected=reference(q,k,v,128**-.5,True,32).reshape(1,4096)
    error=float(np.linalg.norm(actual[0]-expected)/np.linalg.norm(expected))
    assert np.isfinite(actual[0]).all() and error<.03,error
    assert np.array_equal(actual[1],k.reshape(33,256))
    assert np.array_equal(actual[2],v.reshape(33,256))
    records['decode_contract']=dict(status='passed',relative_l2=error,phase=phase)
    for r in records.values():r.pop('nodes',None)
    (root/'summary.json').write_text(json.dumps(dict(status='passed',cases=records),indent=2)+'\n')
    print(records)


if __name__=='__main__':main()
