"""Real-weight batch-1 GLM decode attention: compare packed U with the executed scalar Flash path."""
import argparse
import hashlib
import json
from pathlib import Path, PureWindowsPath
import sys

import numpy as np
from glm4_decode_test import attention, decoder_config, errors, read
from activation_batch_matmul_test import compile_graph, run
from summarize_glm4_flash import phase_timing


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def graph_digest(folder):
    """Hash graph topology, IR and constant contents, ignoring the artifact folder."""
    path = folder/'logical/decoder_layer.graph.json'
    graph = json.loads(path.read_text())
    for const in graph['constants']:
        const['file'] = sha(path.parent/PureWindowsPath(const['file']).name)
    return hashlib.sha256(json.dumps(graph, sort_keys=True).encode()).hexdigest()


def u_plan(folder):
    """Optional nominal plan diagnostics; these are not runtime cycle counts."""
    previous, stages = 0, []
    for node in json.loads((folder/'graph_plan.json').read_text())['nodes']:
        if 'nominal_start' not in node:
            continue
        start, end = node['nominal_start'], node['nominal_end']
        if any(s in node['name'] for s in ('numerator', 'normalize', 'packed_output')):
            stages.append(dict(name=node['name'], nominal_start=start, nominal_end=end,
                               gap_from_previous_node=start-previous))
        previous = end
    return stages


def benchmark(args, previous_root, folder, record, timing_key='packed_flash'):
    batch = record['batch']
    source = previous_root/f'batch{batch}'
    previous = source/'attention'
    build = args.build_dir.resolve()
    target = args.target_config.resolve()
    ext = '.exe' if sys.platform == 'win32' else ''
    compiler = build/'compiler'/('ftlpu-compile'+ext)
    runner = build/'runtime'/('compiled_stage_file_runner'+ext)
    signature = json.loads((previous_root/'signature.json').read_text())
    stamp = json.loads((previous/'completed.json').read_text())
    old_result = json.loads((source/'result.json').read_text())
    assert old_result['status'] == 'passed' and old_result['attention_strategy'] == 'flash'
    assert old_result['batch'] == batch and old_result['past_len'] == 32
    assert old_result['decode_tokens_per_request'] == 1
    assert stamp['binary_sha256'] == sha(previous/'program.ftlpu')
    assert sha(target) == signature['target'], 'target differs from scalar Flash'
    assert sha(runner) == signature['runner'], 'CModel runner differs from scalar Flash'
    config_file = args.checkpoint/'config.json'
    assert sha(config_file) == old_result['checkpoint_provenance']['config_sha256']
    clock = json.loads(target.read_text())['external_memory']['lpu_clock_mhz']
    record.update(target_sha256=sha(target), runner_sha256=sha(runner), clock_mhz=clock,
                  compiler_sha256=sha(compiler), previous_compiler_sha256=stamp['compiler_sha256'],
                  previous_program_sha256=stamp['binary_sha256'])

    config = decoder_config(json.loads(config_file.read_text()))
    config['head_dim'] = 128
    paths = [source/f'projections/actual.{i}.bin' for i in range(3)]
    q, k, v = [read(path, (batch, width)) for path, width in zip(paths, (4096, 256, 256))]
    with np.load(source/'reference.npz', allow_pickle=False) as fixture:
        pk, pv = fixture['past_key'], fixture['past_value']
        expected_k, expected_v = fixture['expected_key'], fixture['expected_value']
        record['fixture_fingerprint'] = str(fixture['fingerprint'])
    assert pk.shape == pv.shape == (batch, 32, 2, 128)
    record['projection_output_sha256'] = [sha(path) for path in paths]
    graph, outputs = attention(q, k, v, pk, pv, config, strategy='flash')
    print(f'Compiling batch{batch} Flash attention', flush=True)
    exe, env, binary = compile_graph(graph, outputs, folder, build, target, fp32_spills=True)
    record['graph_sha256'] = graph_digest(folder)
    assert record['graph_sha256'] == graph_digest(previous), 'graph or constants changed'
    record['binary_sha256'] = sha(binary)
    assert record['compiler_sha256'] == sha(compiler), 'compiler changed during compilation'
    print('Executing raw-macro program', flush=True)
    run([exe('runtime', 'compiled_stage_file_runner'), binary, folder/'inputs.txt',
         folder/'actual', folder/'runtime.pipeline.csv'], folder/'execution.log', env)
    run([exe('runtime', 'ftlpu_icu_program_export'), folder/'actual.runtime.ftlpu',
         folder/'icu_programs', '--pre-execution-dir', folder/'actual.pre_execution'],
        folder/'export.log', env)
    (folder/'completed.json').write_text(json.dumps(dict(binary_sha256=sha(binary),
        compiler_sha256=record['compiler_sha256'], semantic_graph_sha256=record['graph_sha256'])))

    context = read(folder/'actual.0.bin', (batch, 4096))
    key, value = [read(folder/f'actual.{i}.bin', (batch, 33, 2, 128)) for i in (1, 2)]
    # Construct the oracle from the original device Q/K/V and fixture, never
    # from the outputs being validated. No oracle tensors enter the graph.
    source_key = np.concatenate((pk, k.reshape(batch, 1, 2, 128)), axis=1)
    source_value = np.concatenate((pv, v.reshape(batch, 1, 2, 128)), axis=1)
    score = np.einsum('bhqd,bthd->bhqt', q.reshape(batch, 2, 16, 128), source_key)/np.sqrt(128)
    p = np.exp(score-score.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    oracle = np.einsum('bhqt,bthd->bhqd', p, source_value).reshape(batch, 4096)
    record['context_metrics'] = errors(context, oracle)
    record['cache_append_metrics'] = dict(key=errors(key[:, 32], expected_k[:, 32]),
                                          value=errors(value[:, 32], expected_v[:, 32]))
    if batch > 1:
        record['requests'] = [dict(request=i, context=errors(context[i], oracle[i]),
            new_key=errors(key[i, 32], expected_k[i, 32]),
            new_value=errors(value[i, 32], expected_v[i, 32])) for i in range(batch)]
        record['worst_request_context_relative_l2'] = max(
            request['context']['relative_l2'] for request in record['requests'])
    record['cache_matches_fixture_and_current_projection'] = all(
        np.array_equal(actual.view('u4'), expected.view('u4'))
        for actual, expected in ((key, source_key), (value, source_value)))
    differences = {}
    for i, name in enumerate(('context', 'key', 'value')):
        a = np.fromfile(folder/f'actual.{i}.bin', '<u2')
        b = np.fromfile(previous/f'actual.{i}.bin', '<u2')
        assert a.shape == b.shape
        mismatches = np.flatnonzero(a != b)
        differences[name] = dict(mismatching_bf16_words=int(mismatches.size),
                                 first_indices=mismatches[:8].tolist())
    record['bitwise_differences_from_previous_flash'] = differences
    record['timing'] = {label: phase_timing(path, clock) for label, path in (
        ('scalar_flash', previous), (timing_key, folder),
        ('baseline', previous_root/f'baseline_same_build/batch{batch}/attention'))}
    cycles = {label: timing['runtime_cycles'] for label, timing in record['timing'].items()}
    record['cycles_saved_vs_scalar_flash'] = cycles['scalar_flash']-cycles[timing_key]
    record['faster_than_baseline'] = cycles[timing_key] < cycles['baseline']
    if batch == 1:
        record['nominal_u_stages'] = dict(before=u_plan(previous), after=u_plan(folder))
    assert np.isfinite(context).all() and record['context_metrics']['relative_l2'] < .03
    assert all(m['relative_l2'] < .03 for m in record['cache_append_metrics'].values())
    assert record['cache_matches_fixture_and_current_projection']
    assert not any(d['mismatching_bf16_words'] for d in differences.values()), differences
    if batch == 1:
        assert record['cycles_saved_vs_scalar_flash'] > 0, 'no measured execution-time improvement'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path('models/official_dense/glm4_9b'))
    parser.add_argument('--flash-results', type=Path, default=Path('results/glm4_9b_decoder_layer_decode_flash'))
    parser.add_argument('--output-dir', type=Path, default=Path('results/glm4_9b_decoder_layer_decode_flash_packed'))
    parser.add_argument('--build-dir', type=Path, required=True)
    parser.add_argument('--target-config', type=Path, required=True)
    args = parser.parse_args()
    source = args.flash_results.resolve()
    folder = args.output_dir.resolve()/'batch1/attention'
    if folder.is_relative_to(source):
        parser.error('use a separate output directory to preserve comparison artifacts')
    folder.mkdir(parents=True, exist_ok=True)
    result_file = folder.parent/'result.json'
    record = dict(status='running', model='GLM-4-9B W8A16', layer=0, batch=1, past_len=32,
                  scope='one-token attention only; reuses device Q/K/V, no projection/FFN rerun',
                  icu_encoding='raw-macro', sxm='wide', previous_results=str(source))
    result_file.write_text(json.dumps(record, indent=2))
    try:
        benchmark(args, source, folder, record)
        record['status'] = 'passed'
    except Exception as error:
        record.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        result_file.write_text(json.dumps(record, indent=2))
    print(f"{result_file}: {record['status']}; {record['cycles_saved_vs_scalar_flash']} cycles saved", flush=True)


if __name__ == '__main__':
    main()
