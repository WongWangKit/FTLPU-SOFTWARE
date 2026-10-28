"""Real-weight FFN decode: B independent requests, one token per request.

Only semantic graph/fixtures and NumPy validation live here. Shared C++ FFN
lowering owns tiling, layout, paging and instruction generation.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from qwen3_5_compiled_layer import bf16
from qwen3_5_logical_graph import LogicalGraph


def command(args, log, env):
    with log.open('w', encoding='utf-8') as f:
        result = subprocess.run([str(a) for a in args], stdout=f,
                                stderr=subprocess.STDOUT, env=env)
    if result.returncode:
        raise RuntimeError(log.read_text(encoding='utf-8')[-6000:])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('package', 'build-dir', 'target-config', 'output-dir'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--batches', nargs='+', type=int, default=[1, 32, 64])
    p.add_argument('--check-only', action='store_true')
    args = p.parse_args()
    if any(b < 1 for b in args.batches):
        p.error('batch sizes must be positive')
    if any(b != 1 and b % 32 for b in args.batches):
        p.error('current FFN lowering supports batch 1 or a multiple of 32')
    package, build, out = (x.resolve() for x in (args.package, args.build_dir, args.output_dir))
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((package / 'metadata.json').read_text())
    target = json.loads(args.target_config.read_text())
    memory = target['external_memory']
    clock = memory['lpu_clock_mhz']
    provenance = meta['checkpoint_provenance']
    assert provenance['repo'] == 'Qwen/Qwen3.5-0.8B'
    h, n = meta['hidden_size'], meta['intermediate_size']
    assert (h, n) == (1024, 3584)
    roles = ('gate_proj', 'up_proj', 'down_proj')
    packed = {r: np.fromfile(package / f'{r}.i8.bin', 'i1').reshape(
        (n, h) if r == 'down_proj' else (h, n)).T.copy() for r in roles}
    scales = meta['scales']
    weights = {r: packed[r].astype(np.float32).T * np.float32(scales[r]) for r in roles}
    # Independent, deterministic post-norm-like input vectors. Not embeddings
    # masquerading as captured decoder activations. Prefixes are shared across B.
    rng = np.random.default_rng(35261027)
    x_all = rng.normal(size=(max(args.batches), h)).astype(np.float32)
    x_all = bf16(x_all / np.sqrt(np.mean(x_all*x_all, axis=1, keepdims=True) + 1e-6))
    ext = '.exe' if sys.platform == 'win32' else ''
    compiler = build / 'compiler' / ('ftlpu-compile' + ext)
    graph_compiler = build / 'compiler' / ('ftlpu-graph-compile' + ext)
    runner = build / 'runtime' / ('compiled_stage_file_runner' + ext)
    exporter = build / 'runtime' / ('ftlpu_icu_program_export' + ext)
    env = dict(os.environ)
    for key in list(env):
        if key.startswith('FTLPU_'):
            env.pop(key)
    env['FTLPU_FILE_RUNNER_MODEL_SESSION'] = '1'
    env['FTLPU_SESSION_PROGRESS'] = '1'
    records = []
    for batch in args.batches:
        directory = out / f'batch{batch}'
        directory.mkdir(exist_ok=True)
        summary_path = directory / 'summary.json'
        summary_path.unlink(missing_ok=True)
        record = dict(batch=batch, sequence_length_per_request=1,
            model=provenance['repo'], revision=provenance['revision'],
            layer=meta['layer'], hidden_size=h, intermediate_size=n,
            scope='isolated_decode_ffn', activation_source='deterministic_synthetic_rms_normalized',
            weights='official_checkpoint_per_tensor_W8A16', mxm_execution='vector',
            clock_mhz=clock, ddr_bandwidth_GB_s=memory['ddr_peak_bandwidth_mbytes_per_second']/1000,
            pass_limits=dict(nrmse=.03, worst_request_nrmse=.04), status='failed')
        print(f'Running decode FFN batch={batch}', flush=True)
        try:
            x = x_all[:batch]
            if not args.check_only:
                for stale in ('actual.0.bin', 'actual.runtime.ftlpu', 'runtime.pipeline.csv'):
                    (directory / stale).unlink(missing_ok=True)
                graph = LogicalGraph()
                final = graph.ffn(x, packed, scales)
                logical = graph.write(directory / 'logical', [final])
                kernel, binary = directory / 'ffn.kernel.mlir', directory / 'ffn.ftlpu'
                command([graph_compiler, '--input', logical, '--output', kernel,
                    '--emit-kernel', '--target-config', args.target_config.resolve()], directory / 'graph.log', env)
                command([compiler, '--input', kernel, '--input-stage', 'kernel',
                    '--output', binary, '--mxm-execution', 'vector', '--ffn-schedule', 'tail',
                    '--target-config', args.target_config.resolve()], directory / 'compile.log', env)
                command([runner, binary, directory / 'inputs.txt', directory / 'actual',
                    directory / 'runtime.pipeline.csv'], directory / 'execution.log', env)
                if exporter.exists():
                    command([exporter, directory / 'actual.runtime.ftlpu', directory / 'runtime_icu_programs',
                        '--pre-execution-dir', directory / 'actual.pre_execution'], directory / 'export.log', env)
            gate, up = x @ weights['gate_proj'], x @ weights['up_proj']
            expected = (gate * np.exp(-np.logaddexp(0, -gate)) * up) @ weights['down_proj']
            expected.astype('<f4').tofile(directory / 'golden.f32.bin')
            actual = (np.fromfile(directory / 'actual.0.bin', '<u2').astype(np.uint32) << 16).view(np.float32).reshape(batch, h)
            assert np.isfinite(actual).all(), 'non-finite output'
            delta = actual.astype(np.float64) - expected
            nrmse = float(np.linalg.norm(delta) / np.linalg.norm(expected))
            request_errors = np.linalg.norm(delta, axis=1) / np.linalg.norm(expected, axis=1)
            log = (directory / 'execution.log').read_text()
            stats_line = [s for s in log.splitlines() if s.startswith('cycles=')][-1]
            stats = {k: int(v) for k, v in re.findall(r'(\w+)=(\d+)', stats_line)}
            assert stats.get('host_operations') == 0 and stats.get('outputs') == 1, stats
            plan = json.loads((directory / 'graph_plan.json').read_text())
            assert any(node.get('native_ffn') for node in plan['nodes']), 'not native FFN lowering'
            record.update(nrmse=nrmse, worst_request_nrmse=float(request_errors.max()),
                per_request_nrmse=request_errors.tolist(), max_abs_error=float(np.abs(delta).max()),
                execution_stats=stats, native_ffn_regions=sum(bool(node.get('native_ffn')) for node in plan['nodes']))
            # CSV end is one occurrence's exclusive end, not the entire repeat
            # domain. Include both repeat dimensions and initial C2C ingress.
            with (directory / 'runtime.pipeline.csv').open(newline='') as f:
                trace = list(csv.DictReader(f))
            end_cycle = lambda r: int(r['end']) + (int(r['inner_count'])-1)*int(r['inner_interval']) + (int(r['outer_count'])-1)*int(r['outer_interval'])
            trace_cycles = max(end_cycle(r) for r in trace)
            first_mxm = min(int(r['start']) for r in trace if r['resource'].startswith('MXM.'))
            initial_copies = [r for r in trace if r['resource'].startswith('MEM.')
                and r['resource'].endswith(('Read3D', 'Write3D')) and end_cycle(r) <= first_mxm]
            assert not initial_copies, 'FFN ingress retained a pre-compute MEM layout copy'
            # Weight queues join the hardware rendezvous after WRITE_SYNC;
            # they must not replay nominal transfer slack as a NOP.
            for bank in (0, 1):
                path = directory / 'runtime_icu_programs' / f'mem_E_slice20_bank{bank}.icu.csv'
                if not path.exists():
                    continue
                with path.open() as f:
                    instructions = list(csv.DictReader(line for line in f if not line.startswith('#')))
                awaiting_consumer = False
                for instruction in instructions:
                    if instruction['phase'] != 'execution':
                        continue
                    opcode = instruction['opcode']
                    if opcode == 'MEM_WRITE_SYNC':
                        awaiting_consumer = True
                    elif opcode == 'NOP' and awaiting_consumer:
                        raise AssertionError('static NOP after MEM_WRITE_SYNC before consumer SYNC')
                    elif opcode == 'SYNC_PAGE' or opcode.startswith('READ'):
                        awaiting_consumer = False
            compute_end = max(end_cycle(r) for r in trace if r['resource'].startswith(('MEM.', 'VXM.', 'MXM.', 'SXM.')))
            latency_ms = trace_cycles / (clock * 1000)
            record.update(runtime_trace_cycles=trace_cycles, last_fu_end_cycle=compute_end,
                initial_mem_layout_copy_events=len(initial_copies),
                trace_latency_ms=latency_ms, ffn_output_tokens_per_second=batch*1000/latency_ms,
                performance_scope='initial_ingress_through_last_traced_event_including_output_transport; not_full_model_token_latency')
            assert nrmse < .03 and request_errors.max() < .04, record
            record['status'] = 'passed'
        except Exception as error:
            record['error'] = str(error)
        summary_path.write_text(json.dumps(record, indent=2) + '\n')
        records.append(record)
        (out / 'summary.json').write_text(json.dumps(records, indent=2) + '\n')
        print(json.dumps(record), flush=True)
    if any(r['status'] != 'passed' for r in records):
        raise SystemExit(1)
    consistency = []
    for small, large in zip(sorted(set(args.batches)), sorted(set(args.batches))[1:]):
        left = np.fromfile(out / f'batch{small}/actual.0.bin', '<u2').reshape(small, h)
        right = np.fromfile(out / f'batch{large}/actual.0.bin', '<u2').reshape(large, h)[:small]
        mismatches = int(np.count_nonzero(left != right))
        consistency.append(dict(smaller_batch=small, larger_batch=large,
                                mismatched_bf16_values=mismatches))
        assert mismatches == 0, 'batch-dependent output for identical requests'
    (out / 'batch_consistency.json').write_text(json.dumps(consistency, indent=2) + '\n')
    fields = ['batch', 'status', 'nrmse', 'worst_request_nrmse', 'runtime_trace_cycles',
              'trace_latency_ms', 'ffn_output_tokens_per_second', 'clock_mhz', 'ddr_bandwidth_GB_s']
    with (out / 'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)


if __name__ == '__main__':
    main()
