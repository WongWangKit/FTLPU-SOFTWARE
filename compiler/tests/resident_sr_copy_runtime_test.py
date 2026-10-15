#!/usr/bin/env python3
"""MEM/SR copies preserve arbitrary BF16/FP32 bytes without VXM ALU instructions."""
import argparse
from pathlib import Path
import subprocess
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from resident_schedule import Schedule,attr,parse_attr,CompiledType


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('opt','compile','runner','output-dir'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    probe=out/'probe.schedule.mlir'
    subprocess.run([str(a.opt.resolve()),'--input',str(Path(__file__).parent/'Inputs/depthwise_conv1d_kernel4.kernel.mlir'),
                    '--pipeline','ftlpu-stablehlo-to-schedule','--output',str(probe)],check=True)
    header=probe.read_text().splitlines()[0]
    target=parse_attr(header[header.index('ftlpu.target = ')+15:-3])
    rng=np.random.default_rng(35516)
    for name,src_hemi,dst_hemi,src_slice,dst_slice,dtype in (
        ('east_forward','east','east',0,32,'bf16'),
        ('westward','west','west',32,0,'bf16'),
        ('cross_same_group','east','west',0,0,'bf16'),
        ('cross_offset_fp32','west','east',8,32,'f32')):
        planes=4 if dtype=='f32' else 2;shape=(32,64);tensor=CompiledType(shape,dtype)
        src=dict(kind='fp32_quad_planar' if planes==4 else 'fp16_pair_planar',slices=list(range(src_slice,src_slice+planes)),
                 base_row=128,bank=1,hemisphere=src_hemi,instruction_count=64,address_stride=1)
        dst=dict(src,slices=list(range(dst_slice,dst_slice+planes)),base_row=512,bank=0,hemisphere=dst_hemi)
        schedule=Schedule(target);schedule.raw_sr_copies=True
        schedule.scalar_domain(src,dst,0,0,(32,2,1),(1,32,0),(1,32,0))
        bindings=[]
        for index,placement in enumerate((src,dst)):
            v=dict(access='input' if index==0 else 'output',index=0,role='activation' if index==0 else 'result',
                   name=name,bytes=shape[0]*shape[1]*planes,placement=placement)
            bindings.append(f'%b{index} = ftlpu.schedule.binding '+('%data ' if index==0 else '')+attr(v)+f' : ({tensor if index==0 else ""}) -> {tensor}')
        source=schedule.module(bindings,[f'%data: {tensor}'])
        if 'ftlpu.schedule.vxm' in source:raise AssertionError('raw SR route emitted a VXM instruction')
        prefix=out/name;ir=prefix.with_suffix('.schedule.mlir');ir.write_text(source,encoding='utf-8')
        binary=prefix.with_suffix('.ftlpu')
        subprocess.run([str(a.compile.resolve()),'--input',str(ir),'--input-stage','verified-schedule','--output',str(binary)],check=True)
        # Include all bit patterns, including NaNs: no decode/re-encode is allowed.
        data=rng.integers(0,256,shape[0]*shape[1]*planes,dtype=np.uint8).tobytes()
        path=prefix.with_suffix('.input.bin');path.write_bytes(data)
        listing=prefix.with_suffix('.inputs.txt');listing.write_text(str(path)+'\n',encoding='utf-8')
        subprocess.run([str(a.runner.resolve()),str(binary),str(listing),str(prefix)],check=True)
        if Path(str(prefix)+'.0.bin').read_bytes()!=data:raise AssertionError(name+' changed payload bytes')
        print(name+' passed bitwise without VXM',flush=True)


if __name__=='__main__':main()
