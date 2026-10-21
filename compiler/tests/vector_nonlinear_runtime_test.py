#!/usr/bin/env python3
"""Numerically check C++ sigmoid/softplus expansion on actual ICU/FU execution."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('compile', 'runner', 'target-config', 'output-dir'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    reports = []
    # Full and partial packed words, including a per-head row broadcast.
    for width in (16, 32, 48, 65):
        x = np.resize(np.array([-80, -30, -10, -2, -.1, -1e-6, 0, 1e-6,
                                .1, 1, 2, 5, 10, 20, 40, 80], np.float32), (4, width))
        for kind in ('sigmoid', 'softplus', 'broadcast_softplus'):
            out = a.output_dir.resolve() / f'{kind}_{width}'
            out.mkdir(parents=True, exist_ok=True)
            ty = f'tensor<4x{width}xf32>'
            source = f'''module {{ func.func @stage(%x: {ty}) -> {ty} {{
              %y = ftlpu.kernel.vector_alu %x, %x {{kind = "{kind}",
                use_immediate = true, immediate = 0.0 : f32, reduce = false}} : ({ty}, {ty}) -> {ty}
              return %y : {ty}
            }} }}'''
            bias = np.linspace(-.5, .5, width, dtype=np.float32)[None, :]
            scale = np.linspace(-2, -.25, width, dtype=np.float32)[None, :]
            inputs = [x]
            if kind == 'broadcast_softplus':
                row = f'tensor<1x{width}xf32>'
                source = f'''module {{ func.func @stage(%x: {ty}, %b: {row}, %s: {row}) -> {ty} {{
                  %a = ftlpu.kernel.vector_alu %x, %b {{kind = "add", use_immediate = false,
                    immediate = 0.0 : f32, reduce = false}} : ({ty}, {row}) -> {ty}
                  %p = ftlpu.kernel.vector_alu %a, %a {{kind = "softplus", use_immediate = true,
                    immediate = 0.0 : f32, reduce = false}} : ({ty}, {ty}) -> {ty}
                  %y = ftlpu.kernel.vector_alu %p, %s {{kind = "multiply", use_immediate = false,
                    immediate = 0.0 : f32, reduce = false}} : ({ty}, {row}) -> {ty}
                  return %y : {ty}
                }} }}'''
                inputs += [bias, scale]
            kernel, binary = out / 'test.mlir', out / 'test.ftlpu'
            kernel.write_text(source, encoding='utf-8')
            subprocess.run([str(a.compile.resolve()), '--input', str(kernel), '--input-stage', 'kernel',
                            '--output', str(binary), '--target-config', str(a.target_config.resolve())], check=True)
            paths = []
            for index, value in enumerate(inputs):
                data = out / f'input{index}.bin'
                value.astype('<f4').tofile(data)
                paths.append(str(data))
            listing = out / 'inputs.txt'
            listing.write_text('\n'.join(paths) + '\n', encoding='utf-8')
            env = os.environ.copy()
            env['FTLPU_FILE_RUNNER_MODEL_SESSION'] = '1'
            for key in ('FTLPU_SESSION_LINK_ONLY', 'FTLPU_SESSION_STOP_CYCLE'):
                env.pop(key, None)
            with (out / 'execution.log').open('w', encoding='utf-8') as log:
                subprocess.run([str(a.runner.resolve()), str(binary), str(listing), str(out / 'test'),
                                str(out / 'runtime.pipeline.csv')], env=env, stdout=log,
                               stderr=subprocess.STDOUT, check=True)
            actual = np.fromfile(out / 'test.0.bin', '<f4').reshape(x.shape)
            expected = (1/(1+np.exp(-x.astype(np.float64))) if kind == 'sigmoid'
                        else np.logaddexp(0, x.astype(np.float64)))
            if kind == 'broadcast_softplus':
                expected = np.logaddexp(0, x.astype(np.float64) + bias) * scale
            # The hardware SFU restores LUT results with FP16 rounding and
            # flush-to-zero, even when the enclosing graph stores FP32.
            # Allow that specified error, including exp(-10) flushing to 0.
            np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=1e-4)
            if not np.isfinite(actual).all():
                raise AssertionError('nonfinite nonlinear output')
            if kind == 'sigmoid':
                with (out / 'runtime.pipeline.csv').open(newline='') as stream:
                    trace = list(csv.DictReader(stream))
                alu = [row for row in trace if row['resource'].startswith('VXM.')]
                if ({row['resource']: row['detail'].split()[0] for row in alu} != {
                        'VXM.C0': 'neg', 'VXM.C1': 'exp_lut', 'VXM.C2': 'add',
                        'VXM.C3': 'reciprocal_lut', 'VXM.C4': 'bypass'}
                        or len(alu) != 5 or len({row['start'] for row in alu}) != 1):
                    raise AssertionError('sigmoid must execute as one physical ALU chain')
                for direction in ('Read3D', 'Write3D'):
                    transfers = [row for row in trace
                                 if row['resource'].endswith('.' + direction)]
                    if len(transfers) != 4:
                        raise AssertionError('sigmoid must read/write only its four byte planes')
            reports.append(dict(kind=kind, width=width, max_abs=float(np.max(np.abs(actual-expected)))))
    (a.output_dir / 'summary.json').write_text(json.dumps(reports, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(reports, indent=2))


if __name__ == '__main__':
    main()
