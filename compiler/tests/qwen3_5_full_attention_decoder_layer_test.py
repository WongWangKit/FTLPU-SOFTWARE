#!/usr/bin/env python3
"""Run a real Qwen3.5 full-attention decoder layer through one ICU program."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from qwen3_5_full_attention import load, quantize, graph_layer, reference


def metrics(actual, expected):
    a,b=actual.astype(np.float64).ravel(),expected.astype(np.float64).ravel()
    if not np.isfinite(a).all(): raise AssertionError('non-finite CModel output')
    return dict(max_abs=float(np.max(np.abs(a-b))),nrmse=float(np.linalg.norm(a-b)/np.linalg.norm(b)),
                cosine=float(a@b/(np.linalg.norm(a)*np.linalg.norm(b))))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('compile','graph-compile','runner','model-dir','output-dir','target-config'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--layer',type=int,default=3)
    p.add_argument('--seq-len',type=int,default=32)
    p.add_argument('--compile-only',action='store_true')
    p.add_argument('--validate-only',action='store_true')
    a=p.parse_args();out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    x,weights,config,provenance=load(a.model_dir,a.layer,a.seq_len)
    packed,scales,quantized=quantize(weights,config)
    binary=out/'decoder_layer.ftlpu';prefix=out/'resident'
    if not a.validate_only:
        (out/'summary.json').unlink(missing_ok=True)
        graph,output=graph_layer(x,weights,packed,scales,config)
        logical=graph.write(out/'logical_graph',[output])
        subprocess.run([str(a.graph_compile.resolve()),'--input',str(logical),'--output',str(out/'decoder_layer.schedule.mlir'),
            '--inputs-dir',str(out),'--target-config',str(a.target_config.resolve())],check=True)
        subprocess.run([str(a.compile.resolve()),'--input',str(out/'decoder_layer.schedule.mlir'),'--input-stage','verified-schedule',
            '--output',str(binary),'--target-config',str(a.target_config.resolve())],check=True)
        if a.compile_only: return
        for name in ('execution_identity.json','resident.0.bin','resident.runtime.ftlpu'):
            (out/name).unlink(missing_ok=True)
        environment=os.environ.copy();environment['FTLPU_SESSION_PROGRESS']='1'
        environment.pop('FTLPU_SESSION_LINK_ONLY',None);environment.pop('FTLPU_SESSION_STOP_CYCLE',None)
        with (out/'execution.log').open('w',encoding='utf-8') as log:
            subprocess.run([str(a.runner.resolve()),str(binary),str(out/'inputs.txt'),str(prefix),str(out/'runtime.pipeline.csv')],
                stdout=log,stderr=subprocess.STDOUT,env=environment,check=True)
        (out/'execution_identity.json').write_text(json.dumps(dict(
            compiler_binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
            runner_sha256=hashlib.sha256(a.runner.read_bytes()).hexdigest(),
            output_sha256=hashlib.sha256(Path(str(prefix)+'.0.bin').read_bytes()).hexdigest()),indent=2)+'\n')
    identity=json.loads((out/'execution_identity.json').read_text())
    assert identity['runner_sha256']==hashlib.sha256(a.runner.read_bytes()).hexdigest()
    assert identity['compiler_binary_sha256']==hashlib.sha256(binary.read_bytes()).hexdigest()
    assert identity['output_sha256']==hashlib.sha256(Path(str(prefix)+'.0.bin').read_bytes()).hexdigest()
    actual=(np.fromfile(str(prefix)+'.0.bin','<u2').astype(np.uint32)<<16).view(np.float32).reshape(x.shape)
    comparisons={name:metrics(actual,reference(x,w,config)) for name,w in [('same_w8a16_weights',quantized),('original_bf16_checkpoint',weights)]}
    error=comparisons['same_w8a16_weights']
    if error['nrmse']>.05 or error['cosine']<.999:
        raise AssertionError(f'same_w8a16_weights: {error}')
    comparisons['quantization_only']=metrics(reference(x,quantized,config),reference(x,weights,config))
    runtime_binary=Path(str(prefix)+'.runtime.ftlpu')
    exporter=a.runner.resolve().with_name('ftlpu_icu_program_export.exe' if sys.platform=='win32' else 'ftlpu_icu_program_export')
    cmd=[str(exporter),str(runtime_binary),str(out/'runtime_icu_programs')]
    pre=Path(str(prefix)+'.pre_execution')
    if pre.exists():cmd+=['--pre-execution-dir',str(pre)]
    subprocess.run(cmd,check=True)
    log=(out/'execution.log').read_text();lines=[line for line in log.splitlines() if line.startswith('cycles=')]
    stats={k:int(v) for k,v in re.findall(r'(\w+)=(\d+)',lines[-1])} if lines else {}
    summary=dict(status='passed',model='Qwen/Qwen3.5-0.8B',layer_index=a.layer,layer_type='full_attention',seq_len=len(x),
        hidden_size=config['hidden_size'],intermediate_size=config['intermediate_size'],query_heads=config['num_attention_heads'],
        kv_heads=config['num_key_value_heads'],head_dim=config['head_dim'],rotary_dim=int(config['head_dim']*.25),
        includes_ffn=True,attention_output_gate=True,qk_norm=True,synthetic_weights=False,program_count=1,runtime_run_calls=1,
        host_intermediate_transfers=0,arithmetic_backend='CModel ICU/FU',hardware_lowering='C++ native dense attention and logical graph planner',
        input_source='official embedding rows 0..seq_len-1, isolated layer stimulus',checkpoint_provenance=provenance,
        compiler_binary_bytes=binary.stat().st_size,runtime_binary_bytes=runtime_binary.stat().st_size,
        execution_stats=stats,numerical_comparisons=comparisons,
        pass_limits=dict(reference='same_w8a16_weights',nrmse=.05,cosine=.999),
        original_bf16_comparison='reported separately; includes measured per-tensor INT8 quantization error')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
