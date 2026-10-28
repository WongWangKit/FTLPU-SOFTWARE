"""Real-weight GDN attention, through the shared Kernel graph compiler."""
import argparse
import csv
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from qwen3_5_compiled_layer import load_imported_layer
from qwen3_5_logical_graph import LogicalGraph
from qwen3_5_reference import gated_deltanet,zero_centered_rms_norm,Qwen35DeltaState
from qwen3_5_qkv_conv_prefill_test import compare

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('compile','graph-compile','runner','package','target-config','output-dir'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--compile-only',action='store_true')
    p.add_argument('--validate-only',action='store_true')
    p.add_argument('--run-only',action='store_true',help='Reuse the compiled ICU image and rerun CModel plus validation')
    p.add_argument('--trace',action='store_true',help='Capture the full runtime pipeline (large host memory requirement)')
    p.add_argument('--decode-token',type=int,help='Decode one fixture token with a nonzero recurrent and convolution history')
    p.add_argument('--previous-step',type=Path,help='Use actual state and Conv cache outputs from this prior CModel step')
    p.add_argument('--debug-intermediates',action='store_true')
    a=p.parse_args();out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    x,w,packed,scales,c=load_imported_layer(a.package)
    initial=None
    if a.decode_token is not None:
        x=x[a.decode_token:a.decode_token+1]
        if len(x)!=1: raise ValueError('decode token outside fixture')
        rng=np.random.default_rng(35261027)
        initial=Qwen35DeltaState(rng.normal(0,.02,(c.conv_width,c.conv_kernel_size)).astype('f4'),
            rng.normal(0,.02,(c.num_value_heads,c.key_head_dim,c.value_head_dim)).astype('f4'))
        if a.previous_step:
            initial.recurrent=np.fromfile(a.previous_step/'actual.1.bin','<f4').reshape(initial.recurrent.shape)
            initial.conv=(np.fromfile(a.previous_step/'actual.2.bin','<u2').astype('u4')<<16).view('f4').reshape(c.conv_kernel_size,c.conv_width).T
    graph=LogicalGraph();finals=graph.attention(x,w,packed,scales,c,
        initial_state=None if initial is None else initial.recurrent,
        conv_state=None if initial is None else initial.conv.T)
    if a.debug_intermediates:
        stages=list(graph.attention_stages.items())
        (out/'intermediates.json').write_text(json.dumps([
            dict(index=i+3,name=name,shape=value.shape,dtype=value.dtype) for i,(name,value) in enumerate(stages)],indent=2))
        finals=(*finals,*(value for _,value in stages))
    binary=out/'attention.ftlpu';kernel=out/'attention.kernel.mlir'
    summary=dict(status='pending',scope='input RMSNorm and complete GDN attention; no residual or FFN',
        seq_len=len(x),hidden_size=c.hidden_size,heads=c.num_value_heads,
        decode=a.decode_token is not None, state_source=str(a.previous_step) if a.previous_step else 'synthetic nonzero history' if initial is not None else 'zero prefill state',
        state_precision='BF16 storage, FP32 accumulation; FP32 ingress upper16 truncation',
        gate_precision='BF16 upper-plane reload at spill boundaries, FP32 chain arithmetic; FP32 buffer ABI retained',
        runtime_trace_requested=a.trace,
        checkpoint_provenance=json.loads((a.package/'metadata.json').read_text()).get('checkpoint_provenance',{}))
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    try:
        if not a.validate_only and not a.run_only:
            logical=graph.write(out/'logical_graph',finals)
            for command in ([str(a.graph_compile.resolve()),'--input',str(logical),'--output',str(kernel),'--emit-kernel'],
                [str(a.compile.resolve()),'--input',str(kernel),'--input-stage','kernel','--output',str(binary),'--mxm-execution','vector','--pass-timing']):
                with (out/('import.log' if '--emit-kernel' in command else 'compile.log')).open('w') as log:
                    subprocess.run(command+['--target-config',str(a.target_config.resolve())],stdout=log,stderr=subprocess.STDOUT,check=True)
                if '--emit-kernel' in command:
                    kernel.write_text(kernel.read_text().replace('module {',
                        'module attributes {ftlpu.vector_bf16_spill = true, ftlpu.common_barrier_protocol = true} {', 1))
            if a.compile_only:
                summary['status']='compiled';return
        if not a.validate_only:
            with (out/'execution.log').open('w') as log:
                subprocess.run([str(a.runner.resolve()),str(binary),str(out/'inputs.txt'),str(out/'actual')]
                    + ([str(out/'runtime.pipeline.csv')] if a.trace else []),
                    stdout=log,stderr=subprocess.STDOUT,check=True,env={**os.environ,'FTLPU_SESSION_PROGRESS':'1'})
        normalized=zero_centered_rms_norm(x,w.input_norm,c.rms_norm_eps)
        expected,state=gated_deltanet(normalized,w,c,initial)
        references=(expected,state.recurrent.reshape(finals[1].shape),state.conv.T)
        errors={}
        for i,(name,value,reference) in enumerate(zip(('attention','state','conv_state'),finals,references)):
            file=out/f'actual.{i}.bin'
            actual=(np.fromfile(file,'<u2').astype(np.uint32)<<16).view(np.float32) if value.dtype=='bf16' else np.fromfile(file,'<f4')
            errors[name]=compare(actual.reshape(value.shape),reference)
        summary.update(errors=errors,reference='independent FP32 equations with the same W8A16 weights',
            limits=dict(nrmse=.03,cosine=.999),host_intermediate_transfers=0)
        log=(out/'execution.log').read_text()
        summary['execution_stats']={key:int(value) for key,value in re.findall(r'(\w+)=(\d+)',log.splitlines()[-1])}
        summary['physical_imem_overflows']=[dict(queue_kind=int(kind),queue=int(queue),words=int(words),depth=int(depth))
            for kind,queue,words,depth in re.findall(r'iMEM overflow kind=(\d+) queue=(\d+) slots=(\d+) depth=(\d+)',log)]
        if (out/'runtime.pipeline.csv').exists():
            with (out/'runtime.pipeline.csv').open() as trace:
                cycles=max(int(row['end'])+(int(row['inner_count'])-1)*int(row['inner_interval'])
                    +(int(row['outer_count'])-1)*int(row['outer_interval']) for row in csv.DictReader(trace))
            clock=json.loads(a.target_config.read_text())['external_memory']['lpu_clock_mhz']
            summary.update(runtime_trace_cycles=cycles,trace_latency_ms=cycles/(clock*1000))
        exporter=a.runner.resolve().parent/('ftlpu_icu_program_export'+a.runner.suffix)
        if exporter.exists():
            with (out/'export.log').open('w') as export_log:
                subprocess.run([str(exporter),str(out/'actual.runtime.ftlpu'),str(out/'runtime_icu_programs'),
                    '--pre-execution-dir',str(out/'actual.pre_execution')],stdout=export_log,stderr=subprocess.STDOUT,check=True)
        assert all(e['nrmse']<.03 and e['cosine']>.999 for e in errors.values()),errors
        summary['status']='passed'
    except Exception as e:
        summary.update(status='failed',error=str(e));raise
    finally:
        (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
