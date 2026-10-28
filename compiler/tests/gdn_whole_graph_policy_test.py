"""Check GDN policy survives whole-function region extraction.

An unsupported chunk size deliberately stops before physical scheduling. This
checks dispatch, not numerical execution of the native chunkwise backend.
"""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--compile', type=Path, required=True)
    p.add_argument('--target-config', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    out = a.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    types = ['tensor<32x32xbf16>'] * 3 + ['tensor<32x1xf32>'] * 2 + ['tensor<32x32xf32>']
    sizes = [2048] * 3 + [128] * 2 + [4096]
    paths = []
    for i, size in enumerate(sizes):
        f = out / f'input{i}.bin'
        f.write_bytes(bytes(size))
        paths.append(json.dumps(f.as_posix()))
    for scope in ('module', 'function'):
        module_policy = ' attributes {ftlpu.gdn_chunk_size = 16 : i64}' if scope == 'module' else ''
        function_policy = ', ftlpu.gdn_chunk_size = 16 : i64' if scope == 'function' else ''
        args = ', '.join(f'%a{i}: {ty}' for i, ty in enumerate(types))
        operands = ', '.join(f'%a{i}' for i in range(6))
        results = f'({types[0]}, {types[-1]})'
        source = f'''module{module_policy} {{
  func.func @decoder({args}) -> {results} attributes {{
    ftlpu.artifact_directory = {json.dumps(out.as_posix())},
    ftlpu.logical_inputs = [{', '.join(paths)}]{function_policy}
  }} {{
    %o, %s = ftlpu.kernel.gated_delta_rule {operands}
      {{heads = 1 : i64, key_dim = 32 : i64, value_dim = 32 : i64}}
      : ({', '.join(types)}) -> {results}
    return %o, %s : {types[0]}, {types[-1]}
  }}
}}
'''
        kernel = out / f'{scope}.mlir'
        kernel.write_text(source)
        run = subprocess.run([str(a.compile.resolve()), '--input', str(kernel),
            '--input-stage', 'kernel', '--output', str(out / f'{scope}.ftlpu'),
            '--target-config', str(a.target_config.resolve())],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        (out / f'{scope}.log').write_text(run.stdout)
        assert run.returncode != 0 and 'chunkwise baseline currently requires chunk32' in run.stdout, run.stdout
    print('passed: module/function GDN policy reaches extracted recurrence')


if __name__ == '__main__':
    main()
