"""Check FP32 beta broadcast bank reuse and continuous native head scaling."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from qwen3_5_logical_graph import LogicalGraph
from qwen3_5_decode_gdn_attention_test import run
from unit_lower_solve_runtime_test import source


def audit_runtime(path, expected_vectors, require_continuous, cycle_range=None):
    rows = list(csv.DictReader(path.open(newline='', encoding='utf-8')))
    result = {}
    # Runtime CSV names the paired instruction queues C0/C1. Their mirrored
    # physical ALUs (C8/C9) execute the same instructions on the other heads.
    for resource, opcode in (('VXM.C0', 'mul '), ('VXM.C1', 'bypass ')):
        spans = []
        for row in rows:
            if row['resource'] != resource or not row['detail'].startswith(opcode):
                continue
            start, end = int(row['start']), int(row['end'])
            for outer in range(int(row['outer_count'])):
                for inner in range(int(row['inner_count'])):
                    if outer == inner == 0 and int(row['skip_first']):
                        continue
                    offset = outer*int(row['outer_interval']) + inner*int(row['inner_interval'])
                    begin, finish = start+offset, end+offset
                    if cycle_range is not None:
                        begin, finish = max(begin, cycle_range[0]), min(finish, cycle_range[1])
                    if begin < finish:
                        spans.append((begin, finish))
        spans.sort()
        active = sum(b-a for a, b in spans)
        assert active == expected_vectors, (resource, active, expected_vectors)
        assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:])), (resource, 'overlapping issues')
        idle = spans[-1][1]-spans[0][0]-active
        if require_continuous:
            assert idle == 0, (resource, 'steady-state bubble', idle)
        result[resource] = dict(first=spans[0][0], end=spans[-1][1], active_cycles=active,
                                steady_state_idle_cycles=idle)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ('build-dir', 'target-config', 'output-dir'):
        p.add_argument('--' + k, type=Path, required=True)
    p.add_argument('--heads', type=int, default=16)
    p.add_argument('--dim', type=int, default=128)
    args = p.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    build = args.build_dir.resolve()
    exe = lambda area, n: build / area / (n + ('.exe' if sys.platform == 'win32' else ''))
    h, d = args.heads, args.dim
    rng = np.random.default_rng(35321103)
    v = rng.normal(size=(32, h*d)).astype('f4')
    v = (v.view('u4') & np.uint32(0xffff0000)).view('f4')
    beta = rng.uniform(.05, .95, (32, h)).astype('f4')
    # Low mantissa bits differ across heads/tokens; wrong plane pairing or
    # stale ping-pong contents cannot pass an exact FP32 product comparison.
    graph = LogicalGraph()
    result = graph.run('head_scale', source(h, d, 'head_scale'),
                       [(v, 'bf16'), (beta, 'f32')], [((h*32, d), 'f32')])[0]
    logical = graph.write(out / 'logical', [result])
    kernel, binary = out / 'kernel.mlir', out / 'program.ftlpu'
    run([exe('compiler', 'ftlpu-graph-compile'), '--input', logical, '--output', kernel,
         '--emit-kernel', '--target-config', args.target_config.resolve()], out / 'import.log')
    kernel.write_text(kernel.read_text().replace('module {', 'module attributes {'
        'ftlpu.sxm_wide = true, ftlpu.common_barrier_protocol = true, '
        'ftlpu.common_transport_overlap = false, ftlpu.vector_bf16_spill = true, '
        'ftlpu.vector_fp32_ingress = true} {', 1))
    run([exe('compiler', 'ftlpu-compile'), '--input', kernel, '--input-stage', 'kernel',
         '--output', binary, '--target-config', args.target_config.resolve(),
         '--icu-compression', 'raw-macro'], out / 'compile.log')
    os.environ['FTLPU_FILE_RUNNER_MODEL_SESSION'] = '1'
    os.environ['FTLPU_FILE_RUNNER_SXM_ARCHITECTURE'] = 'wide'
    run([exe('runtime', 'compiled_stage_file_runner'), binary, out / 'inputs.txt',
         out / 'actual', out / 'runtime.pipeline.csv'], out / 'execution.log')
    actual = np.fromfile(out / 'actual.0.bin', '<f4').reshape(2, h//2, 32, d)
    actual = actual.transpose(1, 0, 2, 3).reshape(h, 32, d).transpose(1, 0, 2)
    expected = v.reshape(32, h, d) * beta[..., None]
    np.testing.assert_array_equal(actual, expected)
    summary = dict(status='passed', heads=h, dim=d, max_abs=0,
                   precision='BF16 V, full FP32 beta and B')
    summary['runtime'] = audit_runtime(out/'runtime.pipeline.csv', 32*(h//2)*(d//32),
                                       h == 16 and d == 128)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(summary)


if __name__ == '__main__':
    main()
