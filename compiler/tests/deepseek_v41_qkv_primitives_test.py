"""Run the isolated layer-0 Q/KV primitives on the pinned DeepSeek checkpoint."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / 'compiler/tests'
TOOLS = ROOT / 'compiler/tools'
CONFIG = ROOT / 'results/deepseek_v41_flash_assessment/official_config.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', type=Path, required=True)
    parser.add_argument('--target-config', type=Path, required=True)
    parser.add_argument('--package', type=Path, default=ROOT/'results/deepseek_v41_flash_layer0_qkv_prefill')
    args = parser.parse_args()
    case = args.package.resolve()
    case.mkdir(parents=True, exist_ok=True)
    report = case/'summary.json'
    summary = dict(status='running', scope='isolated Q/KV primitives, layer 0 pure SWA prefill',
                   batch=1, sequence=32, start_pos=0, continuous_device_qkv_executed=False)
    report.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    device = ['--build-dir', args.build_dir.resolve(), '--target-config', args.target_config.resolve()]
    opt = args.build_dir.resolve()/'compiler'/('ftlpu_opt.exe' if os.name == 'nt' else 'ftlpu_opt')
    runs = [
        [TOOLS/'import_deepseek_v41_layer0_qkv.py', '--verify-only', '--output', case/'weights'],
        [TOOLS/'deepseek_v41_qkv_reference.py', '--weights', case/'weights', '--output', case/'reference'],
        [TESTS/'deepseek_v41_qkv_norm_test.py', *device, '--package', case, '--config', CONFIG],
        [TESTS/'rmsnorm_rsqrt_range_test.py', *device, '--output-dir', case/'isolated_norms/variance_range'],
        [TESTS/'deepseek_v41_partial_rope_test.py', *device, '--reference-dir', case/'reference',
         '--q-head', '1', '--output-dir', case/'isolated_rope_real'],
        [TESTS/'deepseek_v41_dot32_test.py', *device, '--package', case],
        [TESTS/'deepseek_v41_grouped_wkv_test.py', *device, '--package', case],
    ]
    cases = ['isolated_norms/q_norm', 'isolated_norms/kv_norm', 'isolated_norms/variance_range',
             'isolated_rope_real/q_head_1', 'isolated_rope_real/kv', 'isolated_dot32',
             'isolated_grouped_wkv/k5120_n32']
    for group in (0, 47):
        runs.append([TESTS/'deepseek_v41_fp8_iw_test.py', *device, '--package', case,
                     '--k-group', str(group)])
        cases.append(f'fp8_iw_wkv/kgroup_{group}')
    for kind, directory in [('absmax', 'isolated_activation_groups'), ('scale', 'isolated_activation_scales'),
                            ('qdq', 'isolated_activation_qdq')]:
        runs.append([TESTS/'deepseek_v41_activation_groups_test.py', *device, '--package', case, '--output', kind])
        cases.extend(f'{directory}/{name}' for name in ('x', 'q_norm', 'kv_rope', 'boundaries'))
    try:
        verified = 'projections.verified.kernel.mlir'
        subprocess.run([opt, '--input', TESTS/'Inputs/deepseek_v41_layer0_projections.kernel.mlir',
                        '--target-config', args.target_config.resolve(), '--output', case/verified], check=True)
        for command in runs:
            subprocess.run([sys.executable, '-B', *command], cwd=ROOT, check=True)
        tests = []
        for name in cases:
            paths = {key: f'{name}/{path}' for key, path in (
                ('metrics', 'summary.json'), ('trace', 'runtime.pipeline.csv'),
                ('icu_programs', 'runtime_icu_programs/index.csv'))}
            if not all((case/path).stat().st_size > 0 for path in paths.values()):
                raise ValueError(f'Missing runtime artifacts for {name}')
            metrics = json.loads((case/paths['metrics']).read_text())
            if metrics['status'] != 'passed':
                raise ValueError(f'Failed numerical case: {name}')
            tests.append(dict(name=name, **paths))
        weights = json.loads((case/'weights/provenance.json').read_text())
        summary.update(status='passed', model=weights['model'], revision=weights['revision'],
                       projection_frontend_ir=verified, mxm_fp8_ingress_executed=True,
                       reference='reference/summary.json', inputs='weights/provenance.json',
                       target=args.target_config.name, compression='raw-macro', tests=tests)
    except Exception as exc:
        summary.update(status='failed', error=str(exc))
        raise
    finally:
        report.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(report)


if __name__ == '__main__':
    main()
