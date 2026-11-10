"""Layer-0 SWA forward partial RoPE, one BF16 head and 32 prefill positions."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from activation_batch_matmul_test import batch_matmul, compile_graph, run
from import_hf_decoder_layer import bf16, bf16_bits, read_bf16
from qwen3_5_logical_graph import LogicalGraph
from tensor_geometry import Value


def partial_rope(graph, data, rotary_dim, theta):
    positions, head_dim = data.shape
    ty = graph.type(data.shape, 'bf16')
    source = f'''module {{ func.func @stage(%x: {ty}) -> {ty} {{
      %o = ftlpu.kernel.reshape %x : ({ty}) -> {ty}
      return %o : {ty}
    }} }}'''
    x = graph.run('input', source, [(data, 'bf16')], [(data.shape, 'bf16')])[0]
    tail = x[:, head_dim - rotary_dim:]
    # Multiplication by i maps adjacent pairs (a,b) to (-b,a). This signed
    # permutation has an exact BF16 result for the finite normal fixtures.
    j = np.zeros((1, rotary_dim, rotary_dim), np.float32)
    pairs = np.arange(0, rotary_dim, 2)
    j[0, pairs, pairs + 1] = 1
    j[0, pairs + 1, pairs] = -1
    quarter_turn = batch_matmul(graph, 'multiply_by_i', tail.reshape(1, positions, rotary_dim), j)
    freq = 1.0 / (np.float32(theta) ** (pairs.astype(np.float32) / rotary_dim))
    phases = np.arange(positions, dtype=np.float32)[:, None] * freq
    cosine = np.repeat(np.cos(phases), 2, axis=1)
    sine = np.repeat(np.sin(phases), 2, axis=1)

    def rotate(alu):
        real_term = alu.add(0, 2)
        imaginary_term = alu.add(1, 3)
        return alu.add(real_term, imaginary_term, kind='add', dtype='bf16')

    rotated = graph.vector_tiles('rotate', [(tail, 'bf16'),
        (quarter_turn.reshape(tail.shape), 'bf16'), (cosine, 'f32'), (sine, 'f32')], rotate)
    return Value.concatenate([x[:, :head_dim - rotary_dim], rotated], axis=1)


def run_case(args, out, x, golden, fixture):
    out.mkdir(parents=True, exist_ok=True)
    summary = out / 'summary.json'
    summary.write_text(json.dumps({'status': 'pending'}))
    graph = LogicalGraph()
    y = partial_rope(graph, x, 64, 10000)
    exe, env, binary = compile_graph(graph, [y], out, args.build_dir.resolve(),
                                     args.target_config.resolve(), fp32_spills=True)
    trace = out / 'runtime.pipeline.csv'
    run([exe('runtime', 'compiled_stage_file_runner'), binary, out / 'inputs.txt',
         out / 'actual', trace], out / 'execution.log', env)
    run([exe('runtime', 'ftlpu_icu_program_export'), out / 'actual.runtime.ftlpu',
         out / 'runtime_icu_programs', '--pre-execution-dir', out / 'actual.pre_execution'],
        out / 'export.log', env)
    actual = read_bf16(out / 'actual.0.bin', x.shape)
    np.testing.assert_array_equal(bf16_bits(golden[:, :448]), bf16_bits(x[:, :448]))
    reference = golden[:, -64:]
    diff = actual[:, -64:].astype('f8') - reference
    stats = {k: int(v) for k, v in re.findall(r'(\w+)=(\d+)', (out / 'execution.log').read_text().splitlines()[-1])}
    with trace.open(newline='') as stream:
        activity = list(csv.DictReader(stream))
    invocation = next(row for row in activity if row['resource'] == 'Session.Invocation')
    host_read = [r for r in activity if r['resource'] == 'C2C.HostOutput']
    host_inputs = [int(r['start']) for r in activity if r['resource'] == 'C2C.HostInput']
    def active_cycles(resource):
        return sum((int(row['end']) - int(row['start'])) *
                   (int(row['inner_count']) * int(row['outer_count']) - int(row['skip_first']))
                   for row in activity if row['resource'] == resource)

    compute = {unit: active_cycles(unit) for unit in ('MXM.E0.Compute', 'MXM.W0.Compute', 'VXM.C0')}
    report = dict(scope='single head forward partial RoPE, no projection or cache',
        input_shape=list(x.shape), positions=[0, 31], rotary_dims=[448, 511], theta=10000, yarn=False,
        **fixture, storage='BF16 input/output, FP32 sine/cosine',
        compute_precision='Exact signed permutation in BF16 MXM. VXM multiplies BF16 by FP32 on fixed ports, spills both products in FP32, adds in FP32 and rounds once to BF16.',
        acceptance='All 16384 BF16 outputs match the CPU reference bit for bit and host_operations is zero.',
        prefix_elements=32*448, tail_elements=32*64,
        prefix_bit_mismatches=int(np.sum(bf16_bits(actual[:, :448]) != bf16_bits(golden[:, :448]))),
        position_zero_bit_mismatches=int(np.sum(bf16_bits(actual[0]) != bf16_bits(x[0]))),
        tail_bit_mismatches=int(np.count_nonzero(bf16_bits(actual[:, -64:]) != bf16_bits(reference))),
        tail_relative_l2=float(np.linalg.norm(diff) / np.linalg.norm(reference.astype('f8'))),
        tail_max_abs=float(np.max(np.abs(diff))),
        invocation_cycles=int(invocation['end']) - int(invocation['start']),
        cold_ingress_cycles=int(invocation['start']) - min([int(invocation['start'])] + host_inputs),
        active_cycles_per_unit=compute, weight_page_wait_cycles=stats['weight_page_runtime_wait_cycles'],
        cold_cycles_through_compute=int(invocation['end']) - min([int(invocation['start'])] + host_inputs),
        diagnostic_readback_cycles=sum(int(r['end']) - int(r['start']) for r in host_read),
        runtime=stats, target=args.target_config.name)
    passed = (np.isfinite(actual).all() and report['prefix_bit_mismatches'] == 0
              and report['position_zero_bit_mismatches'] == 0 and report['tail_bit_mismatches'] == 0
              and stats.get('host_operations') == 0 and all(compute.values()))
    report['status'] = 'passed' if passed else 'failed'
    summary.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))
    if not passed:
        raise RuntimeError('See per-region numerical results in summary.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('build-dir', 'target-config', 'output-dir'):
        parser.add_argument('--' + arg, type=Path, required=True)
    parser.add_argument('--reference-dir', type=Path,
                        help="Directory containing verified DeepSeek CPU projection inputs and RoPE outputs")
    parser.add_argument('--q-head', type=int, default=1, choices=range(64), metavar='0..63')
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if args.reference_dir is None:
        x = bf16(np.random.default_rng(41017).normal(0, 0.7, (32, 512)).astype('f4'))
        z = x[:, -64:].copy().view('complex64')
        frequency = 1.0 / (np.float32(10000) ** (np.arange(0, 64, 2, dtype='f4') / 64))
        phase = np.arange(32, dtype='f4')[:, None] * frequency
        golden = x.copy()
        golden[:, -64:] = bf16((z * np.exp(1j * phase)).view('f4'))
        run_case(args, out, x, golden, dict(fixture_seed=41017))
        return

    summary = json.loads((args.reference_dir / 'summary.json').read_text(encoding='utf-8'))
    if summary['status'] != 'independent_cpu_reference' or summary['batch'] != 1 or summary['start_pos'] != 0:
        raise ValueError('Expected an independent CPU prefill reference at position zero, batch one')

    def load_tensor(name, shape):
        info = summary['tensors'][name]
        path = args.reference_dir / info['file']
        if hashlib.sha256(path.read_bytes()).hexdigest() != info['sha256']:
            raise ValueError(f'CPU reference digest mismatch for {name}')
        value = np.load(path, allow_pickle=False)
        if (value.shape != shape or value.dtype != np.float32 or info['logical_dtype'] != 'bf16'
                or not np.isfinite(value).all() or np.any(value.view(np.uint32) & 65535)):
            raise ValueError(f'Expected finite BF16 {name} in float32 container, shape {shape}')
        return value

    for role, before, after in (('kv', 'kv_norm', 'kv_rope'), ('q', 'q_pre_rope', 'q_rope')):
        shape = (32, 64, 512) if role == 'q' else (32, 512)
        x, golden = load_tensor(before, shape), load_tensor(after, shape)
        name = 'kv'
        fixture = dict(seed=summary['seed'], model=summary['model'], revision=summary['revision'],
                       input_source=before, expected_source=after,
                       input_npy_sha256=summary['tensors'][before]['sha256'],
                       expected_npy_sha256=summary['tensors'][after]['sha256'])
        if role == 'q':
            x, golden = x[:, args.q_head, :], golden[:, args.q_head, :]
            fixture['query_head'] = args.q_head
            name = f'q_head_{args.q_head}'
        run_case(args, out / name, x, golden, fixture)


if __name__ == '__main__':
    main()
