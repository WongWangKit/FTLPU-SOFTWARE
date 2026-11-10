"""Independent BF16 activation GEMMs through real raw-macro SXM/MXM."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from qwen3_5_logical_graph import LogicalGraph
from import_hf_decoder_layer import bf16


def batch_matmul(graph, name, lhs, rhs, transpose=False):
    shape = (lhs.shape[0], lhs.shape[1], rhs.shape[1] if transpose else rhs.shape[2])
    a, b, o = (graph.type(s, 'bf16') for s in (lhs.shape, rhs.shape, shape))
    source = f'''module {{ func.func @stage(%a: {a}, %b: {b}) -> {o} {{
      %o = ftlpu.kernel.batch_matmul %a, %b {{transpose_rhs = {str(transpose).lower()},
        role = "{'qk' if transpose else 'pv'}"}} : ({a}, {b}) -> {o}
      return %o : {o}
    }} }}'''
    return graph.run(name, source, [(lhs, 'bf16'), (rhs, 'bf16')], [(shape, 'bf16')])[0]


def run(command, log, env):
    with log.open('w') as stream:
        result = subprocess.run(list(map(str, command)), stdout=stream,
                                stderr=subprocess.STDOUT, env=env)
    if result.returncode:
        raise RuntimeError(log.read_text(errors='replace')[-8000:])


def compile_graph(graph, outputs, folder, build, target, *, fp32_spills=False, module_attrs=()):
    folder.mkdir(parents=True, exist_ok=True)
    ext = '.exe' if sys.platform == 'win32' else ''
    exe = lambda area, name: build / area / (name + ext)
    env = dict(os.environ)
    for key in list(env):
        if key.startswith('FTLPU_'):
            env.pop(key)
    env.update(FTLPU_FILE_RUNNER_MODEL_SESSION='1', FTLPU_SESSION_PROGRESS='1',
               FTLPU_FILE_RUNNER_SXM_ARCHITECTURE='wide', FTLPU_GRAPH_PROGRESS='1')
    logical = graph.write(folder / 'logical', outputs)
    kernel, binary = folder / 'kernel.mlir', folder / 'program.ftlpu'
    run([exe('compiler', 'ftlpu-graph-compile'), '--input', logical,
         '--output', kernel, '--emit-kernel', '--target-config', target], folder / 'graph.log', env)
    kernel.write_text(kernel.read_text().replace('module {', 'module attributes {'
        'ftlpu.sxm_wide = true, ftlpu.vector_fp32_ingress = true, '
        'ftlpu.vector_bf16_spill = true, ftlpu.vector_bf16_rne_spill = '+str(not fp32_spills).lower()+', '
        'ftlpu.common_barrier_protocol = true' + ''.join(', ' + attr for attr in module_attrs) + '} {', 1))
    run([exe('compiler', 'ftlpu-compile'), '--input', kernel, '--input-stage', 'kernel',
         '--output', binary, '--target-config', target, '--icu-compression', 'raw-macro',
         '--mxm-execution', 'ws'], folder / 'compile.log', env)
    return exe, env, binary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('build-dir', 'target-config', 'output-dir'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    out, build, target = a.output_dir.resolve(), a.build_dir.resolve(), a.target_config.resolve()
    rng = np.random.default_rng(409632)
    # Odd batch count tests the final mirrored pair; M=16 tests trimmed rows.
    q = bf16(rng.normal(0, .15, (3, 16, 128)).astype('f4'))
    k = bf16(rng.normal(0, .15, (3, 64, 128)).astype('f4'))
    v = bf16(rng.normal(0, .15, (3, 64, 128)).astype('f4'))
    graph = LogicalGraph()
    score = batch_matmul(graph, 'qk', q, k, True)
    result = batch_matmul(graph, 'pv', score, v)
    exe, env, binary = compile_graph(graph, [score.reshape(48, 64), result.reshape(48, 128)], out, build, target)
    run([exe('runtime', 'compiled_stage_file_runner'), binary, out / 'inputs.txt',
         out / 'actual', out / 'runtime.pipeline.csv'], out / 'execution.log', env)
    expected_score = bf16(q @ k.transpose(0, 2, 1))
    expected = [expected_score, bf16(expected_score @ v)]
    errors = []
    for i, reference in enumerate(expected):
        actual = (np.fromfile(out / f'actual.{i}.bin', '<u2').astype('u4') << 16).view('f4').reshape(reference.shape)
        error = float(np.linalg.norm(actual - reference) / np.linalg.norm(reference))
        assert np.isfinite(actual).all() and error < .02, (i, error)
        # Perturbation/isolation: reference batches use independent random operands.
        assert all(np.linalg.norm(actual[b] - reference[b]) < np.linalg.norm(actual[b] - reference[(b+1)%3]) for b in range(3))
        errors.append(error)
    (out / 'summary.json').write_text(json.dumps(dict(status='passed', nrmse=errors), indent=2))
    print('activation batch GEMM passed', errors)


if __name__ == '__main__':
    main()
