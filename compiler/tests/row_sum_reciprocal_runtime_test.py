"""FP32 axis-zero reduction feeds reciprocal without a MEM intermediate."""
import argparse
import json
from pathlib import Path

import numpy as np

from activation_batch_matmul_test import LogicalGraph, bf16, compile_graph, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('build-dir', 'target-config', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    build, target, folder = (p.resolve() for p in
                            (args.build_dir, args.target_config, args.output_dir))
    graph = LogicalGraph()
    rng = np.random.default_rng(111132)
    outputs, references, cases = [], [], []
    # Exercise accumulator reset/final emit, back-to-back column words, and
    # full FP32 ingress values that would change under BF16 input truncation.
    for rows, columns, dtype in ((1, 32, 'f32'), (2, 64, 'bf16'),
                                (32, 64, 'f32'), (33, 96, 'f32')):
        x = rng.uniform(.2, 1.2, (rows, columns)).astype('f4')
        if dtype == 'bf16':
            x = bf16(x)
        xt, ot = f'tensor<{rows}x{columns}x{dtype}>', f'tensor<1x{columns}xf32>'
        source = f'''module {{ func.func @stage(%x: {xt}) -> {ot}
          attributes {{ftlpu.vector_bf16_spill = false}} {{
          %sum = ftlpu.kernel.vector_alu %x, %x {{kind = "multiply", use_immediate = true,
            immediate = 1.0 : f32, reduce = true, reduction_axis = 0 : i64,
            reduction_kind = "add"}} : ({xt}, {xt}) -> {ot}
          %r = ftlpu.kernel.vector_alu %sum, %sum {{kind = "reciprocal", use_immediate = true,
            immediate = 0.0 : f32, reduce = false}} : ({ot}, {ot}) -> {ot}
          return %r : {ot} }} }}'''
        name = f'sum_inverse_{rows}_{columns}_{dtype}'
        outputs.append(graph.run(name, source, [(x, dtype)], [((1, columns), 'f32')])[0])
        sum_source = source[:source.index('          %r =')] + f' return %sum : {ot} }} }}'
        summed = graph.run(name+'_sum', sum_source, [(x, dtype)], [((1, columns), 'f32')])[0]
        inverse_source = f'''module {{ func.func @stage(%sum: {ot}) -> {ot}
          attributes {{ftlpu.vector_bf16_spill = true}} {{
          %r = ftlpu.kernel.vector_alu %sum, %sum {{kind = "reciprocal", use_immediate = true,
            immediate = 0.0 : f32, reduce = false}} : ({ot}, {ot}) -> {ot}
          return %r : {ot} }} }}'''
        inverse = graph.run(name+'_inverse', inverse_source, [(summed, 'f32')], [((1, columns), 'f32')])[0]
        outputs.extend((summed, inverse))
        references.append(np.cumsum(x, axis=0, dtype='f4')[-1:])
        cases.append(dict(rows=rows, columns=columns, dtype=dtype))
    exe, env, binary = compile_graph(graph, outputs, folder, build, target, fp32_spills=True)
    run([exe('runtime', 'compiled_stage_file_runner'), binary, folder/'inputs.txt',
         folder/'actual', folder/'runtime.pipeline.csv'], folder/'execution.log', env)
    schedule = (folder/'decoder_layer.schedule.mlir').read_text()
    assert schedule.count('row_sum_reciprocal.fp32_chain') == len(cases)
    for index, (case, summed) in enumerate(zip(cases, references)):
        actual, actual_sum, unfused = (np.fromfile(folder/f'actual.{3*index+j}.bin', '<f4').reshape(summed.shape)
                                      for j in range(3))
        # Reciprocal uses the target's existing LUT, not an IEEE division.
        # Require bit-identical results to the separate FP32 sum/reciprocal
        # path, and verify the sum independently before that LUT is applied.
        np.testing.assert_array_equal(actual_sum, summed)
        np.testing.assert_array_equal(actual, unfused)
        case['identical_to_unfused'] = True
        case['reciprocal_lut_max_absolute_error'] = float(np.max(np.abs(actual - 1/summed)))
    (folder/'summary.json').write_text(json.dumps(dict(status='passed', cases=cases), indent=2)+'\n')
    print('row sum -> reciprocal raw-macro passed', cases)


if __name__ == '__main__':
    main()
