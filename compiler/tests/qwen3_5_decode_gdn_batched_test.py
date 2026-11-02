"""One decode step: batch projections, bounded state updates, batch output GEMM.

The three phases are real CModel invocations with DDR intermediates. Python
only serializes tensor rows; every activation operation is compiled from Kernel
IR. Their traces are concatenated without inventing overlap or hiding DMA.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from qwen3_5_compiled_layer import load_imported_layer, bf16
from qwen3_5_logical_graph import LogicalGraph
from qwen3_5_reference import gated_deltanet, zero_centered_rms_norm
from qwen3_5_decode_gdn_attention_test import initial_state, run, errors
from qwen3_5_qkv_conv_prefill_test import compare


def encode(value,dtype):
    return ((bf16(value).view('u4')>>16).astype('<u2') if dtype=='bf16'
            else np.asarray(value,dtype='<f4')).tobytes()


def read(path,shape,dtype='bf16'):
    a=np.fromfile(path,'<u2' if dtype=='bf16' else '<f4')
    return ((a.astype('u4')<<16).view('f4') if dtype=='bf16' else a).reshape(shape)


def concatenate_traces(paths,destination):
    offset=0
    with destination.open('w',newline='') as output:
        writer=None
        for phase,path in paths:
            end=0
            with path.open(newline='') as stream:
                reader=csv.DictReader(stream)
                if writer is None:
                    writer=csv.DictWriter(output,fieldnames=reader.fieldnames);writer.writeheader()
                for row in reader:
                    end=max(end,int(row['end'])+(int(row['inner_count'])-1)*int(row['inner_interval'])
                        +(int(row['outer_count'])-1)*int(row['outer_interval']))
                    row['start']=int(row['start'])+offset;row['end']=int(row['end'])+offset
                    row['detail']=f'phase={phase} '+row['detail'];writer.writerow(row)
            offset+=end
    return offset


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('package','build-dir','target-config','output-dir'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--batches',type=int,nargs='+',default=[1,32,64])
    p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    if any(b<1 for b in a.batches):p.error('batch must be positive')
    build=a.build_dir.resolve();out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    ext='.exe' if sys.platform=='win32' else ''
    exe=lambda area,name:build/area/(name+ext)
    os.environ['FTLPU_FILE_RUNNER_MODEL_SESSION']='1'
    os.environ['FTLPU_SESSION_PROGRESS']='1'
    x,w,packed,scales,c=load_imported_layer(a.package)
    hardware=json.loads(a.target_config.read_text())
    signature=dict(hardware=hardware,encoding='raw-macro',steps=1,
        model_metadata=hashlib.sha256((a.package/'metadata.json').read_bytes()).hexdigest(),
        logical_graph=hashlib.sha256((Path(__file__).resolve().parents[1]/'tools/qwen3_5_logical_graph.py').read_bytes()).hexdigest(),
        compiler=hashlib.sha256(exe('compiler','ftlpu-compile').read_bytes()).hexdigest())
    manifest=out/'run_config.json'
    if a.resume and not manifest.exists():
        raise ValueError('Cannot resume without run_config.json')
    manifest.write_text(json.dumps(signature,indent=2))

    def compile_graph(folder,graph,finals):
        folder.mkdir(parents=True,exist_ok=True)
        binary=folder/'program.ftlpu'
        logical=graph.write(folder/'logical',finals);kernel=folder/'kernel.mlir'
        run([exe('compiler','ftlpu-graph-compile'),'--input',logical,'--output',kernel,
             '--emit-kernel','--target-config',a.target_config.resolve()],folder/'import.log')
        kernel.write_text(kernel.read_text().replace('module {','module attributes {'
            'ftlpu.vector_bf16_spill = true, ftlpu.common_barrier_protocol = true} {',1))
        run([exe('compiler','ftlpu-compile'),'--input',kernel,'--input-stage','kernel',
             '--output',binary,'--target-config',a.target_config.resolve(),
             '--icu-compression','raw-macro','--mxm-execution','vector'],folder/'compile.log')
        return binary

    def execution_key(binary,listings):
        h=hashlib.sha256(binary.read_bytes())
        for listing in listings:
            for path in listing.read_text().splitlines():
                h.update(Path(path).read_bytes())
        return h.hexdigest()

    def execute(folder,binary):
        key=execution_key(binary,[folder/'inputs.txt']);stamp=folder/'execution.sha256'
        if not (a.resume and stamp.exists() and stamp.read_text()==key):
            run([exe('runtime','compiled_stage_file_runner'),binary,folder/'inputs.txt',
                 folder/'actual',folder/'runtime.pipeline.csv'],folder/'execution.log')
            stamp.write_text(key)

    for batch in a.batches:
        directory=out/f'batch{batch}'/'step0';directory.mkdir(parents=True,exist_ok=True)
        record=dict(status='running',batch=batch,steps=1,hardware=hardware,
            icu_compression='raw-macro',real_weights=True,
            batching='batch input/output GEMMs; bounded independent Conv/state updates',
            scope='input RMSNorm + complete GDN attention; no FFN/residual',
            state_policy='DDR state and Conv cache in/out',
            phase_boundary='DDR intermediates, sequential invocations; no simulated overlap',
            latency_limit='CModel trace cycles; host orchestration and iMEM reload wall time excluded')
        try:
            tokens=x[np.arange(batch)%len(x)]
            states=[initial_state(i,c) for i in range(batch)]
            graph=LogicalGraph();finals=graph.decode_projections(tokens,w,packed,scales,c)
            front=directory/'projections';binary=compile_graph(front,graph,finals)
            print(f'batch={batch}: shared input projections',flush=True);execute(front,binary)
            qkv=read(front/'actual.0.bin',(batch,c.conv_width))
            z=read(front/'actual.1.bin',(batch,c.value_width))
            beta=read(front/'actual.2.bin',(batch,c.num_value_heads),'f32')
            decay=read(front/'actual.3.bin',(batch,c.num_value_heads),'f32')

            graph=LogicalGraph();finals=graph.decode_mixer(qkv[:1],beta[:1],decay[:1],
                states[0].conv.T,states[0].recurrent,w,c)
            mixer=directory/'mixer';binary=compile_graph(mixer,graph,finals)
            paths=(mixer/'inputs.txt').read_text().splitlines()
            # Resolve logical input bindings by their exact serialized tensors.
            # No SRAM packing or instruction generation belongs in this harness.
            seeds=[(qkv[:1],'bf16'),(beta[:1],'f32'),(decay[:1],'f32'),
                   (states[0].conv.T,'bf16'),(states[0].recurrent,'f32')]
            indices=[]
            for value,dtype in seeds:
                matches=[i for i,path in enumerate(paths) if Path(path).read_bytes()==encode(value,dtype)]
                assert len(matches)==1,('input ABI changed',value.shape,matches)
                indices.append(matches[0])
            for request in range(batch):
                folder=mixer/f'request{request}';folder.mkdir(exist_ok=True);listing=list(paths)
                values=[qkv[request:request+1],beta[request:request+1],decay[request:request+1],
                        states[request].conv.T,states[request].recurrent]
                for index,value,(_,dtype) in zip(indices,values,seeds):
                    path=folder/f'input{index}.bin';path.write_bytes(encode(value,dtype));listing[index]=str(path)
                (folder/'inputs.txt').write_text('\n'.join(listing)+'\n')
            print(f'batch={batch}: independent Conv/state updates',flush=True)
            key=execution_key(binary,[mixer/f'request{i}'/'inputs.txt' for i in range(batch)])
            stamp=mixer/'execution.sha256'
            if not(a.resume and stamp.exists() and stamp.read_text()==key):
                run([exe('runtime','compiled_batch_file_runner'),binary,mixer,batch],mixer/'execution.log')
                stamp.write_text(key)
            stats=json.loads((mixer/'execution.json').read_text());assert stats['host_operations']==0
            core=np.concatenate([read(mixer/f'request{i}'/'actual.0.bin',(1,c.value_width)) for i in range(batch)])
            graph=LogicalGraph();finals=(graph.decode_output(core,z,w,packed,scales,c),)
            tail=directory/'output';binary=compile_graph(tail,graph,finals)
            print(f'batch={batch}: shared output projection',flush=True);execute(tail,binary)
            actual=read(tail/'actual.0.bin',(batch,c.hidden_size));checked=[]
            for i in range(batch):
                expected=gated_deltanet(zero_centered_rms_norm(tokens[i:i+1],w.input_norm,c.rms_norm_eps),w,c,states[i])
                actual_state=type(states[i])(
                    read(mixer/f'request{i}'/'actual.2.bin',(c.conv_kernel_size,c.conv_width)).T,
                    read(mixer/f'request{i}'/'actual.1.bin',states[i].recurrent.shape,'f32'))
                e=errors((actual[i:i+1],actual_state),expected);checked.append(e)
                assert all(v['nrmse']<.03 and v['cosine']>.999 for v in e.values()),(i,e)
                # Aggregate state error can hide one overwritten head amongst
                # sixteen otherwise correct ones. Check every live state head.
                head_errors=[compare(actual_state.recurrent[h],expected[1].recurrent[h])
                             for h in range(c.num_value_heads)]
                e['state_heads']=head_errors
                assert all(v['nrmse']<.03 and v['cosine']>.999 for v in head_errors),(i,head_errors)
            cycles=concatenate_traces([(name,directory/name/'runtime.pipeline.csv') for name in
                ('projections','mixer','output')],directory/'runtime.pipeline.csv')
            phase_stats={name:{k:int(v) for k,v in re.findall(r'(\w+)=(\d+)',
                (directory/name/'execution.log').read_text().splitlines()[-1])}
                for name in ('projections','output')}
            assert all(s.get('host_operations')==0 for s in phase_stats.values()),phase_stats
            record.update(status='passed',errors=checked,cycles=cycles,
                latency_ms=cycles/(hardware['external_memory']['lpu_clock_mhz']*1000),
                mixer_execution=stats,phase_execution=phase_stats,
                projection_batches=1,output_projection_batches=1)
        except Exception as error:
            record.update(status='failed',error=str(error));raise
        finally:
            (directory/'summary.json').write_text(json.dumps(record,indent=2))
            records=[json.loads(f.read_text()) for f in sorted(out.glob('batch*/step0/summary.json'))]
            (out/'summary.json').write_text(json.dumps(records,indent=2))
        print(json.dumps({k:record[k] for k in ('batch','status','cycles','latency_ms')}),flush=True)


if __name__=='__main__':main()
