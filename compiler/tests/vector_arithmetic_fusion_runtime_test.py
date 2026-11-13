"""Exercise planar VXM fusion through raw macros, independently of mHC."""
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
    graph, rng = LogicalGraph(), np.random.default_rng(111141)
    outputs, cases = [], []
    attrs = ('ftlpu.vector_bf16_spill=true, ftlpu.vector_independent_ports=true, '
             'ftlpu.tight_serial_schedule')

    def source(inputs, operations):
        types = [graph.type(x.shape, dtype) for x, dtype in inputs]
        names = ['%a' + str(i) for i in range(len(types))]
        body = []
        for kind, lhs, rhs, immediate, shape, dtype, reduce in operations:
            out = graph.type(shape, dtype)
            value = '%v' + str(len(body))
            reduction = ', reduction_axis=0:i64, reduction_kind="add"' if reduce else ''
            body.append(f'{value} = ftlpu.kernel.vector_alu {names[lhs]}, {names[rhs]} '
                        f'{{kind="{kind}", use_immediate={str(immediate is not None).lower()}, '
                        f'immediate={float(immediate or 0):.9e}:f32, reduce={str(reduce).lower()}, '
                        f'ftlpu.vector_require_fp32{reduction}}} : ({types[lhs]}, {types[rhs]}) -> {out}')
            names.append(value)
            types.append(out)
        arguments = ', '.join(f'%a{i}: {types[i]}' for i in range(len(inputs)))
        return (f'module {{ func.func @stage({arguments}) -> {types[-1]} attributes {{{attrs}}} {{ '
                + '\n'.join(body) + f' return {names[-1]} : {types[-1]} }} }}')

    def check_case(name, inputs, ops, expected=None):
        shape, dtype = ops[-1][4:6]
        fused = graph.run(name, source(inputs, ops), inputs, [(shape, dtype)])[0]
        values = list(inputs)
        for i, (kind, lhs, rhs, immediate, shape_i, dtype_i, reduce) in enumerate(ops):
            operands = [values[lhs], values[rhs]]
            # A standalone mixed-width head needs the narrow operand on the
            # first fixed port. Commuting this FP32 add/multiply is exact.
            if immediate is None and kind in ('add', 'multiply', 'max') and operands[0][1] == 'f32' and operands[1][1] == 'bf16':
                operands.reverse()
            stage = [(kind, 0, 1, immediate, shape_i, dtype_i, reduce)]
            value = graph.run(name + '_separate_' + str(i), source(operands, stage),
                              operands, [(shape_i, dtype_i)])[0]
            values.append((value, dtype_i))
        outputs.extend((fused, values[-1][0]))
        cases.append(dict(name=name, shape=shape, dtype=dtype, expected=expected))

    for index, (rows, cols, xd, cd, pd, od, broadcast) in enumerate((
            (1, 32, 'bf16', 'bf16', 'bf16', 'bf16', False),
            (2, 64, 'f32', 'f32', 'f32', 'f32', False),
            (33, 96, 'bf16', 'f32', 'f32', 'bf16', False),
            (64, 64, 'bf16', 'bf16', 'f32', 'f32', True))):
        x = rng.normal(size=(rows, cols)).astype('f4')
        c = rng.normal(size=(1, cols)).astype('f4')
        p = rng.normal(size=(1 if broadcast else rows, cols)).astype('f4')
        x, c, p = (bf16(v) if d == 'bf16' else v for v, d in ((x, xd), (c, cd), (p, pd)))
        expected = (x*c).astype('f4') + p
        check_case('scale_add_' + str(index), [(x, xd), (c, cd), (p, pd)],
                   [('multiply', 0, 1, None, x.shape, 'f32', False),
                    ('add', 3, 2, None, x.shape, od, False)], bf16(expected) if od == 'bf16' else expected)

    for rows, cols in ((1, 32), (24, 64)):
        x, y = (rng.normal(size=(rows, cols)).astype('f4') for _ in range(2))
        c = rng.uniform(.2, 1.2, (1, cols)).astype('f4')
        check_case(f'add_scale_{rows}', [(x, 'f32'), (y, 'f32'), (c, 'f32')],
                   [('add', 0, 1, None, x.shape, 'f32', False),
                    ('multiply', 3, 2, None, x.shape, 'bf16', False)], bf16((x+y).astype('f4')*c))

    for rows, cols, dtype in ((1, 32, 'f32'), (2, 64, 'bf16'), (33, 96, 'f32')):
        x = rng.uniform(-1, 1, (rows, cols)).astype('f4')
        if dtype == 'bf16':
            x = bf16(x)
        check_case(f'square_sum_{rows}', [(x, dtype)],
                   [('multiply', 0, 0, None, x.shape, 'f32', False),
                    ('multiply', 1, 1, 1, (1, cols), 'f32', True)],
                   np.cumsum(x*x, axis=0, dtype='f4')[-1:])

    for n, rows, cols, kind, tail in (
            (3, 1, 32, 'add', []),
            (4, 4, 64, 'max', []),
            (4, 4, 64, 'add', [('add', .001), ('reciprocal', 0)]),
            (4, 1, 64, 'add', [('multiply', .125), ('add', .001), ('rsqrt', 0)])):
        xs = [rng.uniform(.2, 1.2, (rows, cols)).astype('f4') for _ in range(n)]
        ops, lhs = [], 0
        for i in range(1, n):
            ops.append((kind, lhs, i, None, xs[0].shape, 'f32', False))
            lhs = n+len(ops)-1
        for k, value in tail:
            ops.append((k, lhs, lhs, value, xs[0].shape, 'f32', False))
            lhs = n+len(ops)-1
        check_case(f'fold_{n}_{kind}_{len(tail)}', [(x, 'f32') for x in xs], ops)

    exe, env, binary = compile_graph(graph, outputs, folder, build, target)
    run([exe('runtime', 'compiled_stage_file_runner'), binary, folder/'inputs.txt',
         folder/'actual', folder/'runtime.pipeline.csv'], folder/'execution.log', env)
    schedule = (folder/'decoder_layer.schedule.mlir').read_text()
    for name, count in (('scale_add', 4), ('add_scale', 2), ('square_sum', 3), ('fold', 4)):
        assert schedule.count(f'vector.{name}_chain') == count, (name, 'fusion not selected')
    for i, case in enumerate(cases):
        def read(index):
            data = np.fromfile(folder/f'actual.{index}.bin', '<u2' if case['dtype'] == 'bf16' else '<f4')
            if case['dtype'] == 'bf16':
                data = (data.astype('u4') << 16).view('f4')
            return data.reshape(case['shape'])
        actual, separate = read(2*i), read(2*i+1)
        np.testing.assert_array_equal(actual, separate, err_msg=case['name'])
        expected = case.pop('expected')
        if expected is not None:
            np.testing.assert_array_equal(actual, expected, err_msg=case['name'])
        case['bit_identical_to_separate'] = True
    result = dict(status='passed', raw_macro=True, cases=cases)
    (folder/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print('planar arithmetic fusion passed:', len(cases), 'cases')


if __name__ == '__main__':
    main()
