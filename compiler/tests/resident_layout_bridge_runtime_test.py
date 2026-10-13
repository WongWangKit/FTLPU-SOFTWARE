#!/usr/bin/env python3
"""Hardware layout bridges preserve tensor values inside SRAM."""
import argparse
import subprocess
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from resident_schedule import Schedule, parse_attr, attr, CompiledType
from qwen3_5_compiled_layer import bf16


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("opt", "compile", "runner", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fixture = Path(__file__).parent / "Inputs/depthwise_conv1d_kernel4.kernel.mlir"
    probe_ir=args.output_dir / "target.schedule.mlir"
    subprocess.run([str(args.opt.resolve()),"--input",str(fixture),"--pipeline","ftlpu-stablehlo-to-schedule","--output",str(probe_ir)],check=True)
    header = probe_ir.read_text().splitlines()[0]
    target = parse_attr(header[header.index("ftlpu.target = ") + 15:-3])
    rng = np.random.default_rng(91)
    for name, rows, columns, dtype, packing, hemisphere in (
        ("unpack_cross", 3, 64, "bf16", False, "west"),
        ("unpack_direct", 3, 64, "bf16", False, "east"),
        ("unpack_fp32_bits", 2, 32, "f32", False, "east"),
        ("pack_replicated", 3, 64, "bf16", True, "both"),
    ):
        shape = (rows, columns)
        planes = 4 if dtype == "f32" else 2
        packed = dict(kind="fp32_quad_planar" if planes == 4 else "fp16_pair_planar", slices=list(range(planes)), base_row=0, bank=0, hemisphere=hemisphere if packing else "east", instruction_count=rows*columns//32, address_stride=1)
        scalar = dict(kind="fp32_scalar_broadcast_planar" if planes == 4 else "fp16_scalar_broadcast_planar", slices=list(range(32,32+planes)), base_row=4096, bank=1, hemisphere="west" if packing else hemisphere, instruction_count=rows*columns, address_stride=1)
        source, dest = (scalar, packed) if packing else (packed, scalar)
        schedule = Schedule(target)
        bindings = []
        tensor = CompiledType(shape,dtype)
        for index, placement in enumerate((source,dest)):
            values = dict(access="input" if index == 0 else "output", index=0, role="activation" if index == 0 else "result", name=name, bytes=rows*columns*(4 if dtype == "f32" else 2), placement=placement)
            bindings.append(f"%bind{index} = ftlpu.schedule.binding " + ("%data " if index == 0 else "") + attr(values) + f" : ({tensor if index == 0 else ''}) -> {tensor}")
        arguments = [f"%data: {tensor}"]
        values = rng.normal(0,.4,shape).astype(np.float32)
        values = bf16(values) if dtype == "bf16" else values
        prefix = args.output_dir / name
        input_file = prefix.with_suffix(".input.bin")
        ((values.view(np.uint32)>>16).astype("<u2") if dtype == "bf16" else values.astype("<f4")).tofile(input_file)
        inputs = [input_file.resolve()]
        if packing:
            schedule.pack(source,dest,shape,np.arange(values.size))
            maskp = dict(kind="fp32_quad_planar",slices=[48,49,50,51],base_row=0,bank=1,hemisphere="east",instruction_count=4,address_stride=1)
            bindings.append("%maskbinding = ftlpu.schedule.binding %masks " + attr(dict(access="input",index=1,role="activation",name="masks",bytes=512,placement=maskp)) + " : (tensor<4x32xf32>) -> tensor<4x32xf32>")
            arguments.append("%masks: tensor<4x32xf32>")
            masks=np.zeros((4,32),np.float32)
            for tile in range(4): masks[tile,tile*8:(tile+1)*8]=1
            mask_file=prefix.with_suffix(".mask.bin");masks.astype("<f4").tofile(mask_file);inputs.append(mask_file.resolve())
        else: schedule.unpack_view(source,dest,shape,np.arange(values.size),np.arange(values.size))
        ir=prefix.with_suffix(".schedule.mlir");ir.write_text(schedule.module(bindings,arguments))
        binary=prefix.with_suffix(".ftlpu")
        subprocess.run([str(args.compile.resolve()),"--input",str(ir),"--input-stage","verified-schedule","--output",str(binary)],check=True)
        listing=prefix.with_suffix(".inputs.txt");listing.write_text("\n".join(map(str,inputs))+"\n")
        subprocess.run([str(args.runner.resolve()),str(binary),str(listing),str(prefix)],check=True)
        actual=np.fromfile(str(prefix)+".0.bin","<f4" if dtype=="f32" else "<u2")
        expected=values.view(np.uint32).reshape(-1) if dtype=="f32" else (values.view(np.uint32)>>16).astype(np.uint16).reshape(-1)
        np.testing.assert_array_equal(actual.view(np.uint32) if dtype=="f32" else actual,expected,err_msg=name)
        print(name+" passed bitwise")


if __name__ == "__main__": main()
