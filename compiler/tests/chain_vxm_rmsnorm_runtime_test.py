"""Run nonzero RMSNorm through the shared, fixed-port chain VXM."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np


def bf16(value):
    bits = np.asarray(value, dtype=np.float32).view(np.uint32)
    return ((bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000).view(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('compile', 'runner', 'target-config', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve().parents[1] / 'examples/rmsnorm/rmsnorm_32x1536.stablehlo.mlir'
    binary = out / 'rmsnorm.ftlpu'
    subprocess.run([str(args.compile.resolve()), '--input', str(source),
                    '--target-config', str(args.target_config.resolve()),
                    '--output', str(binary)], check=True)
    rng = np.random.default_rng(123)
    x = bf16(rng.normal(0, .3, (32, 1536)))
    weight = bf16(rng.uniform(.7, 1.3, 1536))
    inputs = []
    for name, value in [('x', x), ('weight', weight)]:
        path = out / (name + '.bin')
        (value.view(np.uint32) >> 16).astype('<u2').tofile(path)
        inputs.append(str(path))
    listing = out / 'inputs.txt'
    listing.write_text('\n'.join(inputs) + '\n', encoding='utf-8')
    subprocess.run([str(args.runner.resolve()), str(binary), str(listing),
                    str(out / 'actual'), str(out / 'runtime.pipeline.csv')], check=True)
    actual = (np.fromfile(out / 'actual.0.bin', '<u2').astype(np.uint32) << 16).view(np.float32).reshape(x.shape)
    expected = bf16(x / np.sqrt(np.mean(x * x, axis=1, keepdims=True) + 1e-6) * weight)
    error = float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))
    if not np.isfinite(actual).all() or error > .01:
        raise AssertionError(f'chain VXM RMSNorm NRMSE={error}')
    report = dict(status='passed', shape=list(x.shape), nrmse=error,
                  max_abs_error=float(np.max(np.abs(actual - expected))))
    (out / 'summary.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
