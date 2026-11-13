"""Compare native dense and Flash causal prefill using device-produced GLM QKV."""
import argparse
import json
import hashlib
from pathlib import Path
import re
import sys

import numpy as np

from activation_batch_matmul_test import LogicalGraph, bf16
from attention_graph import dense_attention, flash_attention
from dense_attention_core_test import reference
from glm4_attention_comparison_test import constants_manifest, execute
from glm4_decode_test import errors
from glm4_flash_packed_u_test import sha
from glm4_prefill_fixture import load_generated_fixture
from model_test_outputs import OutputContract, TestOutput, verify_compute_schedule


def make_graph(query, key, value, strategy):
    graph = LogicalGraph()
    operator = dense_attention if strategy == 'dense' else flash_attention
    output = operator(graph, 'causal_prefill_attention', query, key, value,
                      causal=True, query_position=0)
    return graph, [output.reshape(query.shape[0] * query.shape[1], -1)]


def verify_resident_compute(folder, diagnostic):
    return verify_compute_schedule(folder, diagnostic)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', type=Path, required=True)
    parser.add_argument('--target-config', type=Path,
                        default=Path('../FTLPU-CMODEL/config/ftlpu-lpu32.json'))
    parser.add_argument('--output-dir', type=Path,
                        default=Path('results/glm4_9b_prefill_attention_comparison'))
    parser.add_argument('--fixture', type=Path,
                        default=Path('results/glm4_9b_prefill_attention_comparison/projections/fixture.npz'))
    parser.add_argument('--synthetic', action='store_true',
                        help='small seq32 causal/GQA regression; never a real-weight acceptance run')
    parser.add_argument('--export-context', action='store_true',
                        help='legacy diagnostic benchmark: export context in the primary invocation')
    args = parser.parse_args()
    args.build_dir = args.build_dir.resolve()
    args.target_config = args.target_config.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    extension = '.exe' if sys.platform == 'win32' else ''
    target = json.loads(args.target_config.read_text())
    provenance = dict(
        compiler_sha256=sha(args.build_dir/'compiler'/('ftlpu-compile'+extension)),
        graph_compiler_sha256=sha(args.build_dir/'compiler'/('ftlpu-graph-compile'+extension)),
        runner_sha256=sha(args.build_dir/'runtime'/('compiled_stage_file_runner'+extension)),
        target_sha256=sha(args.target_config), target_config=str(args.target_config),
        build_dir=str(args.build_dir), clock_mhz=target['external_memory']['lpu_clock_mhz'],
        external_memory=target['external_memory'],
        icu_encoding='raw-macro', sxm='wide', batch=1, sequence_length=32,
        query_position=0, causal=True,
        scope='attention core QK/causal softmax/PV; excludes projection execution time',
        persistent_state='none in the core invocation; Q/K/V supplied by fixture',
        precision='BF16 Q/K/V, score/P and context; FP32 MXM accumulation and VXM arithmetic',
        output_policy='export_context' if args.export_context else 'resident_context_separate_diagnostic',
        limits=dict(context_relative_l2=.03, end_to_end_context_relative_l2=.03))
    if args.synthetic:
        rng = np.random.default_rng(32128)
        query = bf16(rng.normal(0, .5, (1, 32, 4, 128)).astype('f4'))
        key = bf16(rng.normal(0, .5, (1, 32, 2, 128)).astype('f4'))
        value = bf16(rng.normal(0, .5, key.shape).astype('f4')
                     + np.arange(32, dtype='f4')[None, :, None, None] / 4)
        oracle_query, oracle_key, oracle_value = query, key, value
        provenance.update(model='synthetic seq32 causal GQA', fixture_kind='synthetic')
    else:
        args.fixture = args.fixture.resolve()
        fixture_record_path = args.fixture.with_suffix('.json')
        query, key, value, fixture_record = load_generated_fixture(args.fixture.parent)
        assert args.fixture.name == 'fixture.npz'
        # The already executed QKV tensors are immutable inputs to this core
        # benchmark. Keep their original compiler/runtime provenance even when
        # testing a new compiler or CModel; do not attribute projection execution
        # to the current runner. load_generated_fixture verifies artifact hashes.
        assert fixture_record['target_sha256'] == provenance['target_sha256'], 'projection target mismatch'
        provenance['projection_fixture_reused'] = True
        provenance['projection_runner_matches_core'] = (
            fixture_record['runner_sha256'] == provenance['runner_sha256'])
        with np.load(args.fixture, allow_pickle=False) as data:
            oracle_query, oracle_key, oracle_value = (
                data[name] for name in ('reference_q', 'reference_k', 'reference_v'))
        assert query.shape == (1, 32, 32, 128)
        assert key.shape == value.shape == (1, 32, 2, 128)
        provenance.update(model='GLM-4-9B W8A16', layer=0, fixture_kind='actual_raw_macro_projection',
                          fixture_path=str(args.fixture), fixture_sha256=sha(args.fixture),
                          fixture_record_sha256=sha(fixture_record_path),
                          projection_provenance=fixture_record)
    for data in (query, key, value):
        assert np.isfinite(data).all()
        assert np.array_equal(data, bf16(data)), 'QKV must be stored as BF16'
    expected = reference(query, key, value, 128 ** -.5, True, 0)
    end_to_end_expected = reference(oracle_query, oracle_key, oracle_value, 128 ** -.5, True, 0)
    logits = np.einsum('bqhd,bkhd->bhqk', query.astype('f8'),
                      np.repeat(key, query.shape[2]//key.shape[2], axis=2).astype('f8')) / np.sqrt(128)
    assert np.isfinite(logits).all()
    provenance['logit_range_before_causal_mask'] = dict(min=float(logits.min()), max=float(logits.max()))
    np.savez(args.output_dir/'reference.npz', context=expected,
             context_from_reference_projection=end_to_end_expected)
    if args.synthetic:
        unmasked = reference(query, key, value, 128 ** -.5, False, 0)
        assert errors(unmasked, expected)['relative_l2'] > .1, 'causal test is insensitive'
    variants = {}
    for strategy in ('dense', 'flash'):
        folder = args.output_dir/strategy
        folder.mkdir(parents=True, exist_ok=True)
        storage = (dict(scaled_score='BF16', row_max='BF16', probability='BF16 normalized P before PV',
                        exponent_and_sum='FP32 ALU', reciprocal_sum='FP32 local scalar')
                   if strategy == 'dense' else
                   dict(scaled_score='BF16', row_max='FP32',
                        probability='BF16 unnormalized exp(score-max) before PV',
                        exponent_and_sum='FP32', running_numerator='FP32', final_division='FP32 denominator after PV'))
        record = dict(status='running', attention_strategy=strategy, softmax_precision=storage,
                      q_shape=list(query.shape), k_shape=list(key.shape), v_shape=list(value.shape),
                      **provenance)
        report = folder/'result.json'
        report.write_text(json.dumps(record, indent=2)+'\n')
        try:
            contract = OutputContract(TestOutput('context', 'result' if args.export_context else 'diagnostic'))
            (actual,), phase = execute(args, 1, strategy,
                make_graph(query, key, value, strategy), folder, provenance, output_contract=contract)
            if not args.export_context:
                diagnostic = folder/'diagnostic'
                diagnostic_phase = phase['diagnostic_phase']
                diag_plan=json.loads((diagnostic/'graph_plan.json').read_text())
                resident_plan=json.loads((folder/'graph_plan.json').read_text())
                assert resident_plan['external_output_indices']==[]
                assert [v['index'] for v in resident_plan['resident_output_views']]==[0]
                assert resident_plan['nodes']==diag_plan['nodes'], 'output policy changed arithmetic schedule'
                assert resident_plan['linker_mem_transports']==[
                    t for t in diag_plan['linker_mem_transports']
                    if not t['phase'].startswith('validation_export.')]
                assert phase['timing']['output_readback_including_diagnostics_cycles']==0
                assert phase['output_sha256']==[]
                compute_proof=verify_resident_compute(folder,diagnostic)
                # The fixed directory may contain the previous export-mode
                # artifact. Its fresh replacement is under diagnostic/.
                (folder/'actual.0.bin').unlink(missing_ok=True)
                record.update(diagnostic_phase=diagnostic_phase,
                    diagnostic_directory=str(diagnostic),
                    arithmetic_schedule_identical_to_diagnostic=True,
                    compute_schedule_proof=compute_proof,
                    persistent_state_note='K/V already reside in DDR inputs for this core fixture; no new KV is generated here')
            actual = actual.reshape(expected.shape)
            metrics = errors(actual, expected)
            end_metrics = errors(actual, end_to_end_expected)
            record.update(context_metrics=metrics, end_to_end_context_metrics=end_metrics,
                          tokens=[errors(a, e) for a, e in zip(actual[0], expected[0])],
                          phase=phase)
            if strategy == 'flash':
                plan = json.loads((folder/'graph_plan.json').read_text())
                normalized = [n for n in plan['nodes'] if n['name'].endswith('.normalize')]
                schedule = (folder/'decoder_layer.schedule.mlir').read_text()
                native_scales = schedule.count('vector.native_row_scale')
                if any(n.get('attention_core_schedule') == 'flash_tile_pipeline' for n in plan['nodes']):
                    assert len(plan['nodes']) == query.shape[0]
                    assert schedule.count('flash.tile_pipeline') == query.shape[0]
                    assert 'flash.pv.mxm_vxm_direct' in schedule
                    assert 'flash.softmax.p_to_sxm_mxm' in schedule
                    assert not [t for t in plan['linker_mem_transports']
                                if not t['phase'].startswith('validation_export.')]
                    record['lowering_path'] = 'flash_tile_pipeline'
                else:
                    assert native_scales == len(normalized) and native_scales > 0
                    record['lowering_path'] = 'flash_general_tiled'
                assert not any(t['phase'].endswith('.normalize')
                               for t in plan['linker_mem_transports'])
                record['native_row_scale_count'] = native_scales
                record['normalization_layout_copy_bytes'] = 0
            assert np.isfinite(actual).all() and metrics['relative_l2'] < .03, metrics
            assert end_metrics['relative_l2'] < .03, end_metrics
            # At token zero, causal attention must reproduce only V[0],
            # including the distinct GQA groups; no later key may contribute.
            first_value = np.repeat(value[:, 0], query.shape[2]//key.shape[2], axis=1)
            first_metrics = errors(actual[:, 0], first_value)
            assert first_metrics['relative_l2'] < .01, first_metrics
            record.update(status='passed', first_token_causal_metrics=first_metrics,
                          worst_token_context_relative_l2=max(t['relative_l2'] for t in record['tokens']),
                          heads=[errors(actual[:, :, h], expected[:, :, h])
                                 for h in range(query.shape[2])])
            variants[strategy] = (actual, record)
            print(strategy, metrics, phase['timing'], flush=True)
        except Exception as exc:
            record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            report.write_text(json.dumps(record, indent=2)+'\n')
    assert constants_manifest(args.output_dir/'dense') == constants_manifest(args.output_dir/'flash')
    dense, flash = (variants[s][1] for s in ('dense', 'flash'))
    before, after = (r['phase']['timing']['runtime_cycles'] for r in (dense, flash))
    comparison = dict(status='passed', **provenance,
        comparison_build_verification='same_compiler_same_runner_same_inputs_fresh_execution',
        dense=dense, flash=flash, flash_vs_dense=errors(variants['flash'][0], variants['dense'][0]),
        flash_speedup=before/after, flash_cycles_saved=before-after,
        timing_limitations=['iMEM loading/reloading latency is not timed by the runner',
                            'Device execution window includes waits, not just active FU cycles',
                            'Diagnostic context export is separate unless --export-context is selected',
                            'Core K/V are supplied from DDR; this invocation does not generate a new KV cache'])
    if not args.synthetic:
        projection_cycles = fixture_record['phase_timing']['runtime_cycles']
        comparison['projection_plus_core'] = dict(
            projection_cycles=projection_cycles,
            dense_cycles=projection_cycles+before, flash_cycles=projection_cycles+after,
            note='sum of two device invocations with QKV DDR roundtrip; excludes O projection and FFN')
    (args.output_dir/'comparison.json').write_text(json.dumps(comparison, indent=2)+'\n')
    print(f'prefill seq32: dense={before} Flash={after} cycles, speedup={before/after:.4f}', flush=True)


if __name__ == '__main__':
    main()
