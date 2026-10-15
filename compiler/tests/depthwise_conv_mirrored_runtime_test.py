#!/usr/bin/env python3
"""Mirrored Conv1D/SiLU and independent bilateral channel domains on real ICU/FU."""
import argparse
from pathlib import Path
import subprocess
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from qwen3_5_compiled_layer import bf16
from qwen3_5_reference import causal_depthwise_conv1d
from qwen3_5_resident_layer import ResidentGraph
from resident_schedule import ResidentLinker,operation


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('opt','compile','runner','output-dir'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();rng=np.random.default_rng(351616)
    for name,width,mirror in (('mirror_only',64,True),('bilateral',2048,False)):
        out=a.output_dir.resolve()/name;out.mkdir(parents=True,exist_ok=True)
        x=bf16(rng.normal(0,.3,(32,width)).astype(np.float32))
        w=bf16(rng.normal(0,.2,(width,4)).astype(np.float32))
        graph=ResidentGraph(a.opt.resolve(),out/'kernels')
        value=graph.conv(x,w,mirror=mirror)
        linker=ResidentLinker(graph,[value],raw_sr_copies=True)
        source=linker.build();schedule=out/'program.schedule.mlir';schedule.write_text(source,encoding='utf-8')
        native=[operation(line)[1] for line in source.splitlines()
                if 'ftlpu.schedule.vxm ' in line and '%s' in line]
        assert any(v['queue']>=8 and v['opcode']=='multiply' for v in native)
        if width>1024:
            left={v['cycle'] for v in native if v['queue']==0 and v['opcode']=='multiply'}
            right={v['cycle'] for v in native if v['queue']==15 and v['opcode']=='multiply'}
            assert left==right and left,'bilateral MAC domains do not start together'
        binary=out/'program.ftlpu'
        subprocess.run([str(a.compile.resolve()),'--input',str(schedule),'--input-stage','verified-schedule','--output',str(binary)],check=True)
        paths=[]
        for i,(data,dtype) in enumerate(linker.inputs):
            path=out/f'input{i}.bin'
            ((bf16(data).view(np.uint32)>>16).astype('<u2') if dtype=='bf16' else np.asarray(data).astype('<f4')).tofile(path)
            paths.append(str(path))
        listing=out/'inputs.txt';listing.write_text('\n'.join(paths)+'\n',encoding='utf-8')
        prefix=out/'program'
        subprocess.run([str(a.runner.resolve()),str(binary),str(listing),str(prefix),str(out/'runtime.pipeline.csv')],check=True)
        actual=(np.fromfile(out/'program.0.bin','<u2').astype(np.uint32)<<16).view(np.float32).reshape(x.shape)
        expected,_=causal_depthwise_conv1d(x,w,np.zeros_like(w))
        error=float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))
        if not np.isfinite(actual).all() or error>.01:raise AssertionError(f'{name} NRMSE={error}')
        print(f'{name} passed NRMSE={error}',flush=True)


if __name__=='__main__':main()
