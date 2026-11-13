"""Run dense-specialized and Flash GLM attention with the same build and real QKV fixture."""
import argparse
import hashlib
import json
from pathlib import Path, PureWindowsPath
import re
import sys

import numpy as np
from glm4_decode_test import attention, decoder_config, errors, read
from activation_batch_matmul_test import compile_graph, run
from glm4_flash_packed_u_test import graph_digest, sha
from summarize_glm4_flash import phase_timing
from model_test_outputs import OutputContract, TestOutput, execute_with_contract


def save_previous(folder):
    """Keep numerical results and reproduction metadata before replacing a benchmark."""
    source, destination = folder/'result.json', folder/'before_dense.json'
    if not source.exists() or destination.exists():
        return
    previous = json.loads(source.read_text())
    keys = ('status', 'model', 'layer', 'scope', 'batch', 'past_len', 'baseline', 'flash',
            'baseline_vs_flash', 'context_metrics', 'cache_append_metrics', 'timing',
            'compiler_sha256', 'runner_sha256', 'target_sha256', 'clock_mhz',
            'binary_sha256', 'flash_binary_sha256', 'flash_executed_compiler_sha256',
            'fixture_fingerprint', 'projection_output_sha256', 'graph_sha256',
            'icu_encoding', 'sxm', 'comparison_build_verification', 'flash_results')
    snapshot = {key: previous[key] for key in keys if key in previous}
    snapshot['source_result_sha256'] = sha(source)
    snapshot['source_directory'] = str(folder.resolve())
    destination.write_text(json.dumps(snapshot, indent=2)+'\n')


def constants_manifest(folder):
    logical = folder/'logical'
    graph = json.loads((logical/'decoder_layer.graph.json').read_text())
    constants = []
    for item in graph['constants']:
        description = {key: value for key, value in item.items() if key != 'file'}
        description['sha256'] = sha(logical/PureWindowsPath(item['file']).name)
        constants.append(description)
    return constants


def dense_path_evidence(folder):
    """Require an emitter marker in the compiled plan, in addition to the public op."""
    assert 'ftlpu.kernel.dense_attention' in (folder/'kernel.mlir').read_text()
    plan = json.loads((folder/'graph_plan.json').read_text())
    markers = []

    def visit(value, path):
        if isinstance(value, dict):
            if value.get('attention_core_schedule') == 'dense_specialized':
                markers.append(path)
            for key, child in value.items():
                visit(child, f'{path}.{key}')
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f'{path}[{index}]')

    visit(plan, 'graph_plan')
    assert markers, 'dense attention did not record its specialized scheduling path'
    return dict(path='dense_specialized', graph_plan_markers=markers,
                graph_plan_sha256=sha(folder/'graph_plan.json'))


def transport_summary(folder):
    plan = json.loads((folder/'graph_plan.json').read_text())
    transfers = plan.get('linker_mem_transports', [])
    return dict(domains=len(transfers), vectors=sum(t['vectors'] for t in transfers),
                export_domains=sum(t['phase'].startswith('validation_export') for t in transfers),
                export_vectors=sum(t['vectors'] for t in transfers
                                   if t['phase'].startswith('validation_export')))


def execute(args, batch, strategy, graph_outputs, folder, provenance, *,
            module_attrs=(), read_output_indices=None, output_contract=None):
    graph, outputs = graph_outputs
    if output_contract is not None:
        if read_output_indices is not None or any('external_output_indices' in a for a in module_attrs):
            raise ValueError('output contract cannot be combined with a manual output selection')
        return execute_with_contract(output_contract, outputs, folder,
            lambda directory, attrs, indices: execute(args, batch, strategy,
                graph_outputs, directory, provenance,
                module_attrs=tuple(module_attrs)+attrs, read_output_indices=indices))
    print(f'batch={batch}: {strategy} attention compile', flush=True)
    exe, env, binary = compile_graph(graph, outputs, folder, args.build_dir,
                                    args.target_config, fp32_spills=True, module_attrs=module_attrs)
    evidence = dense_path_evidence(folder) if strategy == 'dense' else dict(path='flash_attention')
    assert provenance['compiler_sha256'] == sha(exe('compiler', 'ftlpu-compile'))
    assert provenance['runner_sha256'] == sha(exe('runtime', 'compiled_stage_file_runner'))
    print(f'batch={batch}: {strategy} attention CModel raw-macro', flush=True)
    run([exe('runtime', 'compiled_stage_file_runner'), binary, folder/'inputs.txt',
         folder/'actual', folder/'runtime.pipeline.csv'], folder/'execution.log', env)
    run([exe('runtime', 'ftlpu_icu_program_export'), folder/'actual.runtime.ftlpu',
         folder/'icu_programs', '--pre-execution-dir', folder/'actual.pre_execution'],
        folder/'export.log', env)
    assert re.search(r'^cycles=.*\bhost_operations=0\b', (folder/'execution.log').read_text(), re.M)
    assert provenance['compiler_sha256'] == sha(exe('compiler', 'ftlpu-compile'))
    assert provenance['runner_sha256'] == sha(exe('runtime', 'compiled_stage_file_runner'))
    stamp = dict(binary_sha256=sha(binary), compiler_sha256=provenance['compiler_sha256'],
                 semantic_graph_sha256=graph_digest(folder))
    (folder/'completed.json').write_text(json.dumps(stamp))
    indices = list(range(len(outputs))) if read_output_indices is None else list(read_output_indices)
    values = [read(folder/f'actual.{i}.bin', outputs[i].shape) for i in indices]
    phase = dict(**stamp, lowering=evidence, timing=phase_timing(folder, provenance['clock_mhz']),
                 linker_transport_summary=transport_summary(folder),
                 actual_program_sha256=sha(folder/'actual.runtime.ftlpu'),
                 runtime_pipeline_sha256=sha(folder/'runtime.pipeline.csv'),
                 output_indices=indices,
                 output_sha256=[sha(folder/f'actual.{i}.bin') for i in indices])
    return values, phase


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=Path('models/official_dense/glm4_9b'))
    parser.add_argument('--flash-results', type=Path, default=Path('results/glm4_9b_decoder_layer_decode_flash'))
    parser.add_argument('--target-config', type=Path, default=Path('../FTLPU-CMODEL/config/ftlpu-lpu32.json'))
    parser.add_argument('--output-dir', type=Path,
                        default=Path('results/glm4_9b_decoder_layer_decode_flash_streamed/baseline_current'))
    parser.add_argument('--executed-flash-results', type=Path,
                        default=Path('results/glm4_9b_decoder_layer_decode_flash_streamed'),
                        help='fixed output root in which Flash is freshly compiled and executed')
    parser.add_argument('--batches', type=int, nargs='+', choices=(1, 32), default=[1, 32])
    parser.add_argument('--strategies', nargs='+', choices=('dense','flash'), default=['dense','flash'])
    args = parser.parse_args()
    args.build_dir = args.build_dir.resolve()
    args.target_config = args.target_config.resolve()
    source_root = args.flash_results.resolve()
    dense_root, flash_root = args.output_dir.resolve(), args.executed_flash_results.resolve()
    assert dense_root != flash_root
    assert not dense_root.is_relative_to(source_root) and not flash_root.is_relative_to(source_root)
    extension = '.exe' if sys.platform == 'win32' else ''
    signature = json.loads((source_root/'signature.json').read_text())
    target_hash = sha(args.target_config)
    assert signature['target'] == target_hash, 'different hardware target'
    clock = json.loads(args.target_config.read_text())['external_memory']['lpu_clock_mhz']
    config = decoder_config(json.loads((args.checkpoint/'config.json').read_text()))
    config['head_dim'] = 128
    for batch in args.batches:
        source = source_root/f'batch{batch}'
        original = json.loads((source/'result.json').read_text())
        assert original['status'] == 'passed' and original['attention_strategy'] == 'flash'
        assert original['batch'] == batch and original['past_len'] == 32
        assert original['checkpoint_provenance']['config_sha256'] == sha(args.checkpoint/'config.json')
        projection_paths = [source/f'projections/actual.{i}.bin' for i in range(3)]
        q, k, v = [read(path, (batch, width)) for path, width in zip(projection_paths, (4096, 256, 256))]
        with np.load(source/'reference.npz', allow_pickle=False) as fixture:
            pk, pv = fixture['past_key'], fixture['past_value']
            expected_key, expected_value = fixture['expected_key'], fixture['expected_value']
            fingerprint = str(fixture['fingerprint'])
        assert pk.shape == pv.shape == (batch, 32, 2, 128)
        key = np.concatenate((pk, k.reshape(batch, 1, 2, 128)), axis=1)
        value = np.concatenate((pv, v.reshape(batch, 1, 2, 128)), axis=1)
        score = np.einsum('bhqd,bthd->bhqt', q.reshape(batch, 2, 16, 128), key)/np.sqrt(128)
        probability = np.exp(score-score.max(-1, keepdims=True))
        probability /= probability.sum(-1, keepdims=True)
        expected = np.einsum('bhqt,bthd->bhqd', probability, value).reshape(batch, 4096)
        provenance = dict(model='GLM-4-9B W8A16', layer=0, batch=batch, past_len=32,
            scope='attention core invocation, QK/softmax/PV and KV append', icu_encoding='raw-macro', sxm='wide',
            compiler_sha256=sha(args.build_dir/'compiler'/('ftlpu-compile'+extension)),
            graph_compiler_sha256=sha(args.build_dir/'compiler'/('ftlpu-graph-compile'+extension)),
            runner_sha256=sha(args.build_dir/'runtime'/('compiled_stage_file_runner'+extension)),
            target_sha256=target_hash, clock_mhz=clock, fixture_fingerprint=fingerprint,
            projection_output_sha256=[sha(path) for path in projection_paths],
            past_kv_sha256=[hashlib.sha256(x.tobytes()).hexdigest() for x in (pk, pv)],
            checkpoint_config_sha256=sha(args.checkpoint/'config.json'),
            fixture_root=str(source_root), build_dir=str(args.build_dir), target_config=str(args.target_config),
            precision='BF16 Q/K/V, score/P and context; FP32 MXM accumulation and VXM arithmetic')
        variants = {}
        for strategy, root in (('dense', dense_root), ('flash', flash_root)):
            if strategy not in args.strategies:
                continue
            destination = root/f'batch{batch}'
            destination.mkdir(parents=True, exist_ok=True)
            save_previous(destination)
            storage = (dict(scaled_score='BF16', row_max='BF16', probability='BF16',
                            exponent_and_sum='FP32 ALU', reciprocal_sum='FP32 local scalar')
                       if strategy == 'dense' else
                       dict(scaled_score='BF16', row_max='FP32', probability='BF16',
                            exponent_and_sum='FP32', running_numerator='FP32'))
            record = dict(status='running', attention_strategy=strategy,
                          softmax_precision=storage, **provenance)
            report = destination/'result.json'
            report.write_text(json.dumps(record, indent=2))
            try:
                produced = attention(q, k, v, pk, pv, config, strategy=strategy)
                (actual, actual_key, actual_value), phase = execute(
                    args, batch, strategy, produced, destination/'attention', provenance,
                    output_contract=OutputContract(TestOutput('context', 'diagnostic'),
                        TestOutput('key_cache', 'persistent_state'),
                        TestOutput('value_cache', 'persistent_state')))
                actual_key, actual_value = (x.reshape(batch, 33, 2, 128) for x in (actual_key, actual_value))
                assert np.array_equal(actual_key.view('u4'), key.view('u4')), 'key append differs from inputs'
                assert np.array_equal(actual_value.view('u4'), value.view('u4')), 'value append differs from inputs'
                metrics = errors(actual, expected)
                assert np.isfinite(actual).all() and metrics['relative_l2'] < .03, metrics
                per_request = [errors(a, e) for a, e in zip(actual, expected)]
                record.update(status='passed', context_metrics=metrics, requests=per_request,
                    worst_request_context_relative_l2=max(m['relative_l2'] for m in per_request),
                    cache_matches_fixture_and_current_projection=True, historical_kv_bit_exact=True,
                    cache_append_metrics={label: errors(a[:,32],e[:,32]) for label,a,e in
                        [('key',actual_key,expected_key), ('value',actual_value,expected_value)]},
                    phase=phase, binary_sha256=phase['binary_sha256'], graph_sha256=phase['semantic_graph_sha256'],
                    timing={strategy: phase['timing']})
                if strategy == 'flash':
                    record['timing'] = dict(streamed_flash=phase['timing'])
                    record['actual_runtime_program_sha256'] = dict(streamed_flash=phase['actual_program_sha256'])
                    record['input_sha256'] = {p.name: sha(p) for p in sorted((destination/'attention').glob('input*.bin'))}
                assert all(m['relative_l2'] < .03 for m in record['cache_append_metrics'].values())
                variants[strategy] = (record, actual, destination/'attention')
            except Exception as exc:
                record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
                raise
            finally:
                report.write_text(json.dumps(record, indent=2)+'\n')
        if len(variants) != 2:
            continue
        dense, actual, dense_folder = variants['dense']
        flash, flash_actual, flash_folder = variants['flash']
        if constants_manifest(dense_folder) != constants_manifest(flash_folder):
            dense.update(status='failed', error='different graph input constants')
            (dense_root/f'batch{batch}/result.json').write_text(json.dumps(dense, indent=2)+'\n')
            raise AssertionError(dense['error'])
        before, after = (r['phase']['timing']['runtime_cycles'] for r in (dense, flash))
        dense.update(baseline=dense['context_metrics'], flash=flash['context_metrics'],
            baseline_vs_flash=errors(flash_actual, actual), comparison_build_verification='same_compiler_same_runner_fresh_execution',
            flash_executed_compiler_sha256=flash['compiler_sha256'], flash_binary_sha256=flash['binary_sha256'],
            flash_results=str(flash_root), timing=dict(baseline=dense['phase']['timing'], flash=flash['phase']['timing']),
            flash_cycles_saved_vs_baseline=before-after, flash_speedup_vs_baseline=before/after,
            flash_latency_reduction_percent_vs_baseline=100*(before-after)/before)
        (dense_root/f'batch{batch}/result.json').write_text(json.dumps(dense, indent=2)+'\n')
        print(f'batch={batch}: dense {before} cycles, Flash {after} cycles, speedup {before/after:.4f}', flush=True)


if __name__ == '__main__':
    main()
