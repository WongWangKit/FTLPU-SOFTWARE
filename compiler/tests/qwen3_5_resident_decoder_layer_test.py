#!/usr/bin/env python3
"""One resident decoder binary, one ICU run, no intermediate host transfers."""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from qwen3_5_compiled_layer import bf16,load_imported_layer
from qwen3_5_logical_graph import LogicalGraph
from qwen3_5_reference import Qwen35DeltaConfig,linear_decoder_layer
from qwen3_5_gated_deltanet_reference_test import random_weights


def validate_compiler_plan(directory):
    plan=json.loads((directory/'graph_plan.json').read_text(encoding='utf-8'))
    path=json.loads((directory/'lowering_path.json').read_text(encoding='utf-8'))
    if plan.get('planning')!='tensor_subgraph_before_schedule' or plan.get('native_output_archive_copies')!=0:
        raise AssertionError('whole graph did not use the shared Tensor memory plan')
    if (path.get('entry')!='kernel' or path.get('graph_matcher')!='match_decoder_graph'
            or path.get('ffn_subgraphs')!=1 or path.get('deltanet_subgraphs')!=1):
        raise AssertionError('decoder did not use shared DeltaNet and FFN recognition')
    states=[node['direct_output_layouts'][1] for node in plan['nodes'] if node.get('stateful_mixer')]
    if not states or len({json.dumps(layout,sort_keys=True) for layout in states})!=1:
        raise AssertionError('a linear recurrent scan should carry one retained FP32 state')
    state=states[0]
    if sum(node.get('native_ffn',False) for node in plan['nodes'])!=1:
        raise AssertionError('FFN was split instead of using native dense lowering')
    schedule=(directory/'decoder_layer.schedule.mlir').read_text(encoding='utf-8')
    def norm_placement(name,kind):
        for line in schedule.splitlines():
            if f'name = "{name}"' in line and f'kind = "{kind}"' in line:
                match=re.search(r'placement = (\{[^{}]+\})',line)
                if match:return match.group(1)
        raise AssertionError(f'missing RMSNorm binding: {name} {kind}')
    for input_index,kind in ((0,'fp16_mxm_distributed_16'),
                             (1,'fp16_vxm_gamma_broadcast')):
        external=norm_placement(f'input_norm.input{input_index}',kind)
        native=norm_placement('input_norm.operand',kind)
        if external!=native:
            raise AssertionError('initial RMSNorm input is staged through a different MEM layout')
    native_start=re.search(r'ftlpu\.schedule\.timeline \{end = \d+ : i64, '
        r'name = "input_norm\.rmsnorm\.transpose_input", start = (\d+) : i64\}',schedule)
    if not native_start:
        raise AssertionError('missing first RMSNorm schedule')
    native_start=int(native_start.group(1))
    for line in schedule.splitlines():
        if 'ftlpu.schedule.mem_transfer ' not in line and 'ftlpu.schedule.vxm ' not in line:
            continue
        issue=re.search(r'\bcycle = (\d+) : i64',line)
        if issue and int(issue.group(1))<native_start:
            raise AssertionError('MEM/VXM layout conversion precedes the first RMSNorm')
    lines=schedule.splitlines()
    def binding_line(name):
        return next(i for i,line in enumerate(lines) if f'name = "{name}"' in line)
    rms_result=binding_line('rmsnorm.result.0')
    qkv_input=binding_line('in_proj_qkv.page0.operand')
    if any('ftlpu.schedule.mem_transfer ' in line or 'ftlpu.schedule.vxm ' in line
           for line in lines[rms_result+1:qkv_input]):
        raise AssertionError('RMSNorm-to-QKV staging fell back to a graph-level layout copy')
    qkv_last=max(i for i,line in enumerate(lines)
        if 'name = "in_proj_qkv.page' in line and '.linear_projection"' in line)
    conv_input=binding_line('conv1d_silu.block0.operand')
    if any('ftlpu.schedule.vxm ' in line for line in lines[qkv_last+1:conv_input]):
        raise AssertionError('QKV-to-Conv byte-plane transfer used an ALU copy')
    for node in plan['nodes']:
        if not node.get('native_ffn'): continue
        if not node['weight_page_workspaces']:
            raise AssertionError('native FFN weight pages were omitted from memory planning')
        for page in node['weight_page_workspaces']:
            same_side=page['hemisphere'] in ('both',state['hemisphere'])
            same_bank=page['bank']==state['bank']
            same_slice=bool(set(page['slices']) & set(state['slices']))
            rows_overlap=(page['base_row'] < state['base_row']+state['rows']
                and state['base_row'] < page['base_row']+page['rows'])
            if same_side and same_bank and same_slice and rows_overlap:
                raise AssertionError('FFN weight page overwrites retained recurrent state')
    return plan,path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--opt',type=Path,required=True)
    p.add_argument('--compile',type=Path,required=True)
    p.add_argument('--runner',type=Path,required=True)
    p.add_argument('--graph-compile',type=Path,help='Native graph planner (defaults to the compiler sibling executable)')
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--target-config',type=Path,help='Shared build/compiler hardware target')
    p.add_argument('--pipeline-csv',type=Path,help='Record the actual CModel runtime issue trace')
    p.add_argument('--package',type=Path,help='Use an imported checkpoint package instead of synthetic mini weights')
    p.add_argument('--compile-only',action='store_true',help='Emit and check the ICU image without executing it')
    p.add_argument('--validate-only',action='store_true',help='Validate outputs of a completed continuous run')
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    (a.output_dir/'summary.json').unlink(missing_ok=True)
    c=Qwen35DeltaConfig(256,512,1,2,32,32)
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
    binary=a.output_dir/'decoder_layer.ftlpu'
    prefix=a.output_dir/'resident'
    if not a.validate_only:
        graph=LogicalGraph()
        finals=graph.layer(x,weights,packed,scales,c)
        logical=graph.write(a.output_dir/'logical_graph',finals)
        kernel=a.output_dir/'decoder_layer.kernel.mlir'
        graph_compiler=a.graph_compile or a.compile.resolve().with_name('ftlpu-graph-compile.exe' if sys.platform=='win32' else 'ftlpu-graph-compile')
        graph_command=[str(graph_compiler.resolve()),'--input',str(logical.resolve()),'--output',str(kernel.resolve()),
            '--emit-kernel']
        if a.target_config:graph_command.extend(['--target-config',str(a.target_config)])
        subprocess.run(graph_command,check=True)
        compile_command=[str(a.compile.resolve()),'--input',str(kernel),'--input-stage','kernel','--output',str(binary),
            '--mxm-execution','vector','--ffn-schedule','fused']
        if a.target_config:compile_command.extend(['--target-config',str(a.target_config)])
        subprocess.run(compile_command,check=True)
        validate_compiler_plan(a.output_dir)
        listing=a.output_dir/'inputs.txt'
        if a.compile_only:
            print(f'compiled {binary} ({binary.stat().st_size} bytes)',flush=True)
            return
        command=[str(a.runner.resolve()),str(binary),str(listing),str(prefix)]
        if a.pipeline_csv: command.append(str(a.pipeline_csv.resolve()))
        for i in range(len(finals)):
            Path(str(prefix)+f'.{i}.bin').unlink(missing_ok=True)
        Path(str(prefix)+'.runtime.ftlpu').unlink(missing_ok=True)
        environment=os.environ.copy()
        environment.setdefault('FTLPU_SESSION_PROGRESS','1')
        for name in ('FTLPU_SESSION_LINK_ONLY','FTLPU_SESSION_STOP_CYCLE'):
            environment.pop(name,None)
        execution_path=a.output_dir/'execution.log'
        with execution_path.open('w',encoding='utf-8') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,text=True,env=environment)
        execution_log=execution_path.read_text(encoding='utf-8').strip()
        if result.returncode:raise RuntimeError(execution_log[-8000:])
        identity=dict(compiler_binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
            output_sha256=[hashlib.sha256(Path(str(prefix)+f'.{i}.bin').read_bytes()).hexdigest() for i in range(len(finals))])
        (a.output_dir/'execution_identity.json').write_text(json.dumps(identity,indent=2)+'\n',encoding='utf-8')
        paged=True
    else:
        from types import SimpleNamespace
        identity=json.loads((a.output_dir/'execution_identity.json').read_text(encoding='utf-8'))
        if identity['compiler_binary_sha256']!=hashlib.sha256(binary.read_bytes()).hexdigest():
            raise AssertionError('completed execution belongs to a different compiler binary')
        for i,digest in enumerate(identity['output_sha256']):
            if digest!=hashlib.sha256(Path(str(prefix)+f'.{i}.bin').read_bytes()).hexdigest():
                raise AssertionError('completed output fingerprint changed')
        finals=[SimpleNamespace(shape=x.shape,dtype='bf16'),
            SimpleNamespace(shape=(c.num_value_heads*c.value_head_dim,c.key_head_dim),dtype='f32'),
            SimpleNamespace(shape=(4,(2*c.num_key_heads*c.key_head_dim+c.num_value_heads*c.value_head_dim)),dtype='bf16')]
        execution_log=(a.output_dir/'execution.log').read_text(encoding='utf-8')
        paged=True
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
    stats_lines=[line for line in execution_log.splitlines() if line.startswith('cycles=')]
    execution_stats={key:int(value) for key,value in re.findall(r'(\w+)=(\d+)',stats_lines[-1])} if stats_lines else {}
    runtime_binary=Path(str(prefix)+'.runtime.ftlpu')
    graph_plan,lowering_path=validate_compiler_plan(a.output_dir)
    summary=dict(status='passed',execution='single_resident_icu_program',arithmetic_backend='CModel ICU/FU',
        hardware_lowering='Native Kernel entry, shared decoder graph recognition and FFN lowering, whole-graph Tensor memory planning',
        lowering_path=lowering_path,
        graph_memory_planning=graph_plan['planning'],native_output_archive_copies=0,
        layer_type='linear_attention',seq_len=x.shape[0],hidden_size=c.hidden_size,intermediate_size=c.intermediate_size,synthetic_weights=a.package is None,
        host_intermediate_transfers=0,program_count=1,runtime_run_calls=1,ddr_c2c_paging=paged,
        reference='FP32 NumPy equations with the same W8A16 quantized weights',
        pass_limits=dict(nrmse=.03,cosine=.999),
        binary_bytes=binary.stat().st_size,
        compiler_binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
        runtime_binary_bytes=runtime_binary.stat().st_size if runtime_binary.exists() else None,
        runtime_binary_sha256=hashlib.sha256(runtime_binary.read_bytes()).hexdigest() if runtime_binary.exists() else None,
        execution_log_path=str((a.output_dir/'execution.log').resolve()),
        execution_stats=execution_stats,
        cycles_note='execution_stats.cycles is the static schedule plus drain; runtime waits and initial transfers are in runtime.pipeline.csv',
        max_abs_errors=errors)
    if a.package:
        summary['checkpoint_provenance']=json.loads((a.package/'metadata.json').read_text(encoding='utf-8')).get('checkpoint_provenance',{})
    exporter=a.runner.resolve().with_name('ftlpu_icu_program_export.exe' if sys.platform=='win32' else 'ftlpu_icu_program_export')
    if runtime_binary.exists() and exporter.exists():
        export_command=[str(exporter),str(runtime_binary),str(a.output_dir/'runtime_icu_programs')]
        pre_execution=Path(str(prefix)+'.pre_execution')
        if pre_execution.exists():export_command.extend(['--pre-execution-dir',str(pre_execution)])
        subprocess.run(export_command,check=True)
        summary['runtime_icu_program_directory']=str((a.output_dir/'runtime_icu_programs').resolve())
    (a.output_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
