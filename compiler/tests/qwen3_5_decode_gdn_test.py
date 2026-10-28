"""Qwen3.5-size one-token GDN recurrence, independent state per request.

Python supplies fixtures and an independent reference only. The shared C++
lowering emits ICU programs and ModelSession executes all arithmetic and C2C.
Two hemispheres partition heads (batch 1) or independent requests (batch > 1).
Consecutive request pairs prefetch state into alternating banks.
"""
import argparse
import csv
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from qwen3_5_compiled_layer import bf16


def command(args, log):
    env = {k: v for k, v in os.environ.items() if not k.startswith('FTLPU_')}
    with log.open('w') as f:
        result = subprocess.run(list(map(str, args)), stdout=f, stderr=subprocess.STDOUT, env=env)
    if result.returncode:
        raise RuntimeError(log.read_text()[-6000:])


def fixture(request, heads, dim):
    rng = np.random.default_rng(35261027 + request)
    q, k = [rng.normal(size=(heads, dim)).astype('f4') for _ in range(2)]
    q = bf16(q / np.linalg.norm(q, axis=-1, keepdims=True) / np.sqrt(dim))
    k = bf16(k / np.linalg.norm(k, axis=-1, keepdims=True))
    v = bf16(rng.normal(0, .2, (heads, dim)).astype('f4'))
    alpha = rng.uniform(.7, .999, heads).astype('f4')
    beta = rng.uniform(.05, .95, heads).astype('f4')
    state = rng.normal(0, .02, (heads, dim, dim)).astype('f4')
    # Include reset and unchanged-state edge cases; all other heads are random.
    alpha[0], beta[1] = 0, 0
    return q, k, v, alpha, beta, state


def reference(q, k, v, alpha, beta, state):
    decayed = state.astype('f8') * alpha[:, None, None]
    delta = beta[:, None] * (v - np.einsum('hk,hkv->hv', k, decayed))
    updated = decayed + k[:, :, None] * delta[:, None, :]
    return np.einsum('hk,hkv->hv', q, updated), updated


def reload_reference(q, k, v, alpha, beta, state):
    """Declared upper-plane reload precision, separate from FP32 reference."""
    trunc = lambda x: (np.asarray(x, dtype='f4').view('u4') & np.uint32(0xffff0000)).view('f4')
    decayed = trunc(state) * trunc(alpha)[:, None, None]
    remembered = np.einsum('hk,hkv->hv', k, trunc(decayed))
    delta = v - trunc(remembered)
    scaled = trunc(delta) * trunc(beta)[:, None]
    product = k[:, :, None] * trunc(scaled)[:, None, :]
    updated = trunc(decayed) + trunc(product)
    output = np.einsum('hk,hkv->hv', q, trunc(updated))
    return bf16(output), updated


def check_carried_state(directory, binary, runner, heads, dim, run=True):
    """Feed a real CModel state output into the next decode invocation."""
    root = directory / 'carry_step'
    folder = root / 'request0'
    folder.mkdir(parents=True, exist_ok=True)
    values = list(fixture(100000, heads, dim))
    values[-1] = np.fromfile(directory / 'request0/actual.1.bin', '<f4').reshape(heads, dim, dim)
    if run:
        for i, value in enumerate(values):
            data = (value.view('u4') >> 16).astype('<u2') if i < 3 else value.astype('<f4')
            data.tofile(folder / f'input{i}.bin')
        command([runner, binary, root, 1], root / 'execution.log')
    actual = [(np.fromfile(folder / 'actual.0.bin', '<u2').astype('u4') << 16).view('f4').reshape(heads, dim),
        np.fromfile(folder / 'actual.1.bin', '<f4').reshape(heads, dim, dim)]
    errors = [float(np.linalg.norm(x-y)/np.linalg.norm(y)) for x,y in zip(actual,reload_reference(*values))]
    assert errors[0] < .0002 and errors[1] < 1e-5, errors
    result = dict(status='passed', state_source='previous CModel output, no host state update',
        reload_output_nrmse=errors[0], reload_state_nrmse=errors[1])
    (root/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('package', 'build-dir', 'target-config', 'output-dir'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--batches', type=int, nargs='+', default=[1, 32, 64])
    p.add_argument('--check-only', action='store_true')
    a = p.parse_args()
    if any(b < 1 for b in a.batches): p.error('batch must be positive')
    out, build = a.output_dir.resolve(), a.build_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((a.package / 'metadata.json').read_text())
    h, d = meta['linear_num_value_heads'], meta['linear_key_head_dim']
    assert (h, d, meta['linear_value_head_dim']) == (16, 128, 128)
    target = json.loads(a.target_config.read_text())['external_memory']
    clock = target['lpu_clock_mhz']
    ext = '.exe' if sys.platform == 'win32' else ''
    compiler = build / 'compiler' / ('ftlpu-compile' + ext)
    runner = build / 'runtime' / ('compiled_batch_file_runner' + ext)
    exporter = build / 'runtime' / ('ftlpu_icu_program_export' + ext)
    def compile_kernel(heads, stem, bank=0, prefetch=False):
        source, binary = out / (stem + '.kernel.mlir'), out / (stem + '.ftlpu')
        if a.check_only:
            return binary
        template = (Path(__file__).parent / 'Inputs/gated_delta_rule_seq1.kernel.mlir').read_text()
        if prefetch:
            template = template.replace('%state: tensor<64x32xf32>)',
                '%state: tensor<64x32xf32>, %next_state: tensor<64x32xf32>)')
        attributes = ('ftlpu.vector_bf16_spill = true, ftlpu.gdn_partition_heads = true, '
            f'ftlpu.gdn_state_bank = {bank} : i64, ftlpu.gdn_prefetch_next_state = {str(prefetch).lower()}')
        source.write_text(template.replace('module {', f'module attributes {{{attributes}}} {{').replace('1x64', f'1x{heads*d}').replace('1x2x', f'1x{heads}x')
            .replace('64x32', f'{heads*d}x{d}').replace('heads = 2', f'heads = {heads}')
            .replace('dim = 32', f'dim = {d}'))
        command([compiler, '--input', source, '--input-stage', 'kernel', '--output', binary,
            '--target-config', a.target_config.resolve()], out / (stem + '.compile.log'))
        return binary
    binaries = {1: compile_kernel(h, 'decode')}
    if any(b > 1 and b % 2 == 0 for b in a.batches):
        binaries[2] = compile_kernel(2*h, 'decode_pair')
    prefetch_variants = []
    if any(b >= 4 and b % 2 == 0 for b in a.batches):
        prefetch_variants = [compile_kernel(2*h,'decode_pair_bank1',bank=1),
            compile_kernel(2*h,'decode_pair_prefetch_bank0',prefetch=True),
            compile_kernel(2*h,'decode_pair_prefetch_bank1',bank=1,prefetch=True)]
    records = []
    for batch in a.batches:
        directory = out / f'batch{batch}'
        directory.mkdir(exist_ok=True)
        group = 2 if batch > 1 and batch % 2 == 0 else 1
        use_prefetch = group == 2 and batch >= 4
        binary = binaries[group]
        launch_dir = directory if group == 1 else directory / 'paired_requests'
        launch_dir.mkdir(exist_ok=True)
        summary = dict(status='failed', batch=batch, tokens_per_request=1, heads=h,
            key_dim=d, value_dim=d, state_bytes_per_request=h*d*d*4,
            scope='GDN recurrence core only; no projection/conv/norm/gate branches/output projection',
            batching='independent requests, two concurrent requests per invocation' if group == 2 else 'single request',
            requests_per_invocation=group,
            state_prefetch='next pair into alternate bank, compiler MEM_WRITE_SYNC' if use_prefetch else 'none',
            hemisphere_mapping='one request per hemisphere' if group == 2 else 'independent head halves',
            fixture='deterministic synthetic normalized Q/K, V, decay/beta and nonzero FP32 state',
            real_weights=False, weights_note='recurrence core has no learned weights',
            arithmetic='shared direct lowering; BF16 Q/K/V and FP32 accumulation, FP32 interface with upper16 truncation at reloads',
            clock_mhz=clock, ddr_bandwidth_GB_s=target['ddr_peak_bandwidth_mbytes_per_second']/1000)
        print(f'Running GDN decode batch={batch}', flush=True)
        (directory / 'summary.json').write_text(json.dumps({**summary, 'status':'running'}, indent=2)+'\n')
        try:
            for request in range(batch):
                folder = directory / f'request{request}'
                folder.mkdir(exist_ok=True)
                if not a.check_only:
                    for index in (0, 1):
                        (folder/f'actual.{index}.bin').unlink(missing_ok=True)
                    for i, value in enumerate(fixture(request, h, d)):
                        data = (value.view(np.uint32) >> 16).astype('<u2') if i < 3 else value.astype('<f4')
                        data.tofile(folder / f'input{i}.bin')
            if not a.check_only:
                if group == 2:
                    # Concatenation only: recurrence has no cross-head terms.
                    # The compiler partitions these head groups onto E/W.
                    for launch in range(batch // group):
                        packed = launch_dir / f'request{launch}'
                        packed.mkdir(exist_ok=True)
                        for index in range(6):
                            (packed/f'input{index}.bin').write_bytes(b''.join(
                                (directory/f'request{launch*group+j}'/f'input{index}.bin').read_bytes()
                                for j in range(group)))
                command([runner, binary, launch_dir, batch // group]
                    + (prefetch_variants if use_prefetch else []), directory / 'execution.log')
                if group == 2:
                    for launch in range(batch // group):
                        for index in (0,1):
                            data = (launch_dir/f'request{launch}'/f'actual.{index}.bin').read_bytes()
                            size = len(data)//group
                            for j in range(group):
                                (directory/f'request{launch*group+j}'/f'actual.{index}.bin').write_bytes(data[j*size:(j+1)*size])
                    for name in ('runtime.pipeline.csv','actual.runtime.ftlpu','execution.json'):
                        shutil.copy2(launch_dir/name,directory/name)
                    shutil.copytree(launch_dir/'actual.pre_execution',directory/'actual.pre_execution',dirs_exist_ok=True)
                    shutil.copytree(launch_dir/'invocation_programs',directory/'invocation_programs',dirs_exist_ok=True)
            # Each invocation has a distinct linked page/event contract. The
            # final invocation alone omits next-state prefetch by design.
            command([exporter, directory / 'actual.runtime.ftlpu', directory / 'runtime_icu_programs',
                '--pre-execution-dir', directory / 'actual.pre_execution'], directory / 'export.log')
            for image in sorted((directory / 'invocation_programs').glob('*.ftlpu')):
                command([exporter, image, directory / 'runtime_icu_programs' / image.stem],
                    directory / 'invocation_programs' / (image.stem + '.export.log'))
            errors = []
            for request in range(batch):
                folder = directory / f'request{request}'
                values = fixture(request, h, d)
                expected = reference(*values)
                reload_expected = reload_reference(*values)
                actual = [(np.fromfile(folder / 'actual.0.bin', '<u2').astype(np.uint32)<<16).view('f4').reshape(h,d),
                    np.fromfile(folder / 'actual.1.bin', '<f4').reshape(h,d,d)]
                assert all(np.isfinite(x).all() for x in actual)
                error = {name: float(np.linalg.norm(x-y)/np.linalg.norm(y))
                    for name, x, y in zip(('output_nrmse','state_nrmse'), actual, expected)}
                error.update({name: float(np.linalg.norm(x-y)/np.linalg.norm(y))
                    for name,x,y in zip(('reload_output_nrmse','reload_state_nrmse'),actual,reload_expected)})
                errors.append(error)
                assert error['output_nrmse'] < .025 and error['state_nrmse'] < .025, error
                assert error['reload_output_nrmse'] < .0002 and error['reload_state_nrmse'] < 1e-5, error
            stats = json.loads((directory / 'execution.json').read_text())
            assert stats['host_operations'] == 0 and stats['requests'] == batch // group
            assert stats['c2c_ingress_bytes'] + stats.get('prefetched_input_bytes',0) >= batch*h*d*d*4
            assert stats['c2c_egress_bytes'] >= batch*h*d*d*4
            if use_prefetch:
                assert stats['prefetched_input_hits'] == batch//group-1, stats
            with (directory / 'runtime.pipeline.csv').open() as f:
                trace = list(csv.DictReader(f))
            cycles = max(int(r['end'])+(int(r['inner_count'])-1)*int(r['inner_interval'])
                +(int(r['outer_count'])-1)*int(r['outer_interval']) for r in trace)
            summary.update(status='passed', execution_stats=stats, per_request_errors=errors,
                output_nrmse=max(e['output_nrmse'] for e in errors),
                state_nrmse=max(e['state_nrmse'] for e in errors), runtime_trace_cycles=cycles,
                trace_latency_ms=cycles/(clock*1000), requests_per_second=batch*clock*1e6/cycles,
                performance_scope='whole batch including C2C ingress/egress; paired compute and next-state prefetch overlap')
            if batch == 1:
                summary['next_decode_step'] = check_carried_state(directory, binary, runner, h, d, not a.check_only)
        except Exception as error:
            summary.update(status='failed', error=str(error))
        (directory / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        records.append(summary)
        (out / 'summary.json').write_text(json.dumps(records, indent=2)+'\n')
        print(json.dumps({k:v for k,v in summary.items() if k != 'per_request_errors'}), flush=True)
        if summary['status'] != 'passed': raise SystemExit(1)
    consistency = []
    for small, large in zip(sorted(a.batches), sorted(a.batches)[1:]):
        for request in range(small):
            for index in (0, 1):
                path = f'request{request}/actual.{index}.bin'
                assert (out/f'batch{small}'/path).read_bytes() == (out/f'batch{large}'/path).read_bytes(), 'request isolation failure'
        consistency.append(dict(smaller_batch=small, larger_batch=large,
            checked_requests=small, output_and_state_bit_identical=True))
    (out/'batch_consistency.json').write_text(json.dumps(consistency, indent=2)+'\n')
    with (out/'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['batch','status','output_nrmse','state_nrmse',
            'runtime_trace_cycles','trace_latency_ms','requests_per_second'], extrasaction='ignore')
        writer.writeheader(); writer.writerows(records)


if __name__ == '__main__': main()
