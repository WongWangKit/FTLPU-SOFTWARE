#!/usr/bin/env python3
"""One resident decoder binary, one ICU run, no intermediate host transfers."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from qwen3_5_compiled_layer import bf16,load_imported_layer
from qwen3_5_resident_layer import ResidentGraph
from resident_schedule import ResidentLinker
from qwen3_5_reference import Qwen35DeltaConfig,linear_decoder_layer
from qwen3_5_gated_deltanet_reference_test import random_weights


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--opt',type=Path,required=True)
    p.add_argument('--compile',type=Path,required=True)
    p.add_argument('--runner',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--target-config',type=Path,help='Shared build/compiler hardware target')
    p.add_argument('--pipeline-csv',type=Path,help='Record the actual CModel runtime issue trace')
    p.add_argument('--package',type=Path,help='Use an imported checkpoint package instead of synthetic mini weights')
    p.add_argument('--compile-only',action='store_true',help='Emit and check the ICU image without executing it')
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    (a.output_dir/'summary.json').unlink(missing_ok=True)
    c=Qwen35DeltaConfig(64,128,1,2,32,32)
    rng=np.random.default_rng(3506)
    weights=random_weights(c,rng)
    packed,scales={},{}
    for name in ('in_proj_qkv','in_proj_z','in_proj_b','in_proj_a','out_proj','gate_proj','up_proj','down_proj'):
        w=getattr(weights,name);scale=float(np.max(np.abs(w))/127)
        packed[name]=np.clip(np.rint(w/scale),-127,127).astype(np.int8);scales[name]=scale
        setattr(weights,name,packed[name].astype(np.float32)*np.float32(scale))
    for name in ('conv1d','delta_norm','input_norm','post_attention_norm'):
        setattr(weights,name,bf16(getattr(weights,name)))
    x=bf16(rng.normal(0,.2,(32,c.hidden_size)).astype(np.float32))
    if a.package:
        x,weights,packed,scales,c=load_imported_layer(a.package)
    graph=ResidentGraph(a.opt.resolve(),a.output_dir/'kernels',a.target_config)
    finals=graph.layer(x,weights,packed,scales,c)
    linker=ResidentLinker(graph,finals)
    source=linker.build()
    schedule=a.output_dir/'decoder_layer.schedule.mlir';schedule.write_text(source,encoding='utf-8')
    binary=a.output_dir/'decoder_layer.ftlpu'
    compile_command=[str(a.compile.resolve()),'--input',str(schedule),'--input-stage','verified-schedule','--output',str(binary)]
    if a.target_config:compile_command.extend(['--target-config',str(a.target_config)])
    subprocess.run(compile_command,check=True)
    files=[]
    for i,(value,dtype) in enumerate(linker.inputs):
        path=a.output_dir/f'input{i}.bin'
        if dtype=='bf16':(bf16(value).view(np.uint32)>>16).astype('<u2').tofile(path)
        else:np.asarray(value).astype('<f4' if dtype=='f32' else 'i1').tofile(path)
        files.append(str(path.resolve()))
    listing=a.output_dir/'inputs.txt';listing.write_text('\n'.join(files)+'\n',encoding='utf-8')
    if a.compile_only:
        print(f'compiled {binary} ({binary.stat().st_size} bytes)',flush=True)
        return
    prefix=a.output_dir/'resident'
    command=[str(a.runner.resolve()),str(binary),str(listing),str(prefix)]
    if a.pipeline_csv: command.append(str(a.pipeline_csv.resolve()))
    result=subprocess.run(command,capture_output=True,text=True)
    if result.returncode:raise RuntimeError(result.stderr)
    actual=[]
    for i,value in enumerate(finals):
        path=Path(str(prefix)+f'.{i}.bin')
        data=(np.fromfile(path,'<u2').astype(np.uint32)<<16).view(np.float32) if value.dtype=='bf16' else np.fromfile(path,'<f4')
        actual.append(data.reshape(value.shape))
    golden_stages={}
    golden,state=linear_decoder_layer(x,weights,c,stages=golden_stages)
    expected=(golden,state.recurrent.reshape(finals[1].shape),golden_stages['conv_state'].T)
    errors={}
    for name,value,reference in zip(('output','recurrent_state','conv_state'),actual,expected):
        diff=value.astype(np.float64)-reference.astype(np.float64)
        nrmse=float(np.linalg.norm(diff)/max(np.linalg.norm(reference),1e-30))
        cosine=float(np.dot(value.reshape(-1).astype(np.float64),reference.reshape(-1).astype(np.float64))/max(np.linalg.norm(value)*np.linalg.norm(reference),1e-30))
        errors[name]=dict(max_abs=float(np.max(np.abs(diff))),nrmse=nrmse,cosine=cosine)
        if not np.isfinite(value).all() or nrmse>.03 or cosine<.999:
            raise AssertionError(f'{name}: {errors[name]}')
    summary=dict(status='passed',execution='single_resident_icu_program',arithmetic_backend='CModel ICU/FU',
        layer_type='linear_attention',seq_len=x.shape[0],hidden_size=c.hidden_size,intermediate_size=c.intermediate_size,synthetic_weights=a.package is None,
        host_intermediate_transfers=0,program_count=1,runtime_run_calls=1,ddr_c2c_paging=bool(linker.weight_inputs),
        binary_bytes=binary.stat().st_size,execution_log=result.stdout.strip(),max_abs_errors=errors)
    (a.output_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
