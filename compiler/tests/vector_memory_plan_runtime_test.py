#!/usr/bin/env python3
"""Verify planned VXM output sides, SRAM offsets, reductions and both ALU groups."""
import argparse
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from qwen3_5_compiled_layer import CompiledStages, bf16


def placement(shape, hemisphere, first, base, dtype):
    planes = 4 if dtype == 'f32' else 2
    return ('{kind = "' + ('fp32' if planes == 4 else 'fp16') +
        '_scalar_broadcast_planar", slices = [' + ', '.join(str(first+i) for i in range(planes)) +
        f'], bank = 1 : i64, hemisphere = "{hemisphere}", base_row = {base} : i64, '
        f'instruction_count = {int(np.prod(shape))} : i64, address_stride = 1 : i64' + '}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('compile', 'runner', 'output-dir'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    stages = CompiledStages(a.compile.resolve(), a.runner.resolve(), a.output_dir.resolve())
    x = np.arange(128, dtype=np.float32).reshape(2,64) / 128 + .25
    for queue in (0,8):
        for hemisphere in ('east','west'):
            for kind, reduce, dtype in [('multiply',False,'f32'), ('exp',False,'f32'),
                                         ('reciprocal',False,'f32'), ('rsqrt',False,'f32'),
                                         ('multiply',True,'f32'), ('multiply',False,'bf16')]:
                shape = (2,1) if reduce else x.shape
                lhs = placement(x.shape, hemisphere, 0, 256, dtype)
                result = placement(shape, 'east', 40, 512, dtype)
                source = f'''module {{func.func @stage(%x: tensor<2x64x{dtype}>) -> tensor<{shape[0]}x{shape[1]}x{dtype}>
                  attributes {{ftlpu.vxm_queue_base = {queue} : i64}} {{
                  %y = ftlpu.kernel.vector_alu %x, %x {{kind = "{kind}", use_immediate = {str(kind in ('exp','reciprocal','rsqrt')).lower()},
                    immediate = 0.0 : f32, reduce = {str(reduce).lower()}, ftlpu.vector_lhs_placement = {lhs},
                    ftlpu.vector_rhs_placement = {lhs}, ftlpu.vector_result_placement = {result}, ftlpu.vector_packed = false}}
                    : (tensor<2x64x{dtype}>, tensor<2x64x{dtype}>) -> tensor<{shape[0]}x{shape[1]}x{dtype}>
                  return %y : tensor<{shape[0]}x{shape[1]}x{dtype}>
                }} }}'''
                actual = stages.run(f'q{queue}_{hemisphere}_{kind}_{reduce}_{dtype}', source,
                    [(x,dtype)], [(shape,dtype)])[0]
                expected = (np.exp(x) if kind=='exp' else 1/x if kind=='reciprocal' else
                    1/np.sqrt(x) if kind=='rsqrt' else np.sum(x*x,axis=1,keepdims=True) if reduce else x*x)
                if dtype == 'bf16': expected = bf16(expected)
                np.testing.assert_allclose(actual, expected,
                    rtol=.001 if kind in ('exp','reciprocal','rsqrt') else 1e-5, atol=1e-6)
    print('planned East outputs, nonzero bases, BF16/FP32, both sources and ALU groups passed')


if __name__ == '__main__':
    main()
