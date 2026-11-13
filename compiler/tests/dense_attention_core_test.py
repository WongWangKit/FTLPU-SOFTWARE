"""Synthetic batch-2 decode through the public dense attention core and raw macros."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

from activation_batch_matmul_test import LogicalGraph, bf16, compile_graph
from attention_graph import dense_attention
from glm4_attention_comparison_test import execute
from glm4_decode_test import errors
from glm4_flash_packed_u_test import sha


CASES = ('negative_padding', 'causal_offset', 'scaled', 'unaligned_position')


def reference(query, key, value, scale, causal, position):
    heads_per_key = query.shape[2] // key.shape[2]
    key = np.repeat(key, heads_per_key, axis=2)
    value = np.repeat(value, heads_per_key, axis=2)
    scores = np.einsum('bqhd,bkhd->bhqk', query.astype('f8'), key.astype('f8')) * scale
    if causal:
        allowed = np.arange(key.shape[1]) <= position + np.arange(query.shape[1])[:, None]
        scores = np.where(allowed[None, None], scores, -np.inf)
    probability = np.exp(scores - scores.max(-1, keepdims=True))
    probability /= probability.sum(-1, keepdims=True)
    return np.einsum('bhqk,bkhd->bqhd', probability, value.astype('f8'))


def fixture(case):
    rng = np.random.default_rng(7845)
    query = bf16(rng.normal(0, .4, (2, 1, 4, 128)).astype('f4'))
    key = bf16(rng.normal(0, .4, (2, 33, 2, 128)).astype('f4'))
    value = bf16(rng.normal(0, .4, key.shape).astype('f4')
                 + np.array([.25, 1.25], dtype='f4')[:, None, None, None])
    scale, causal, position = 128 ** -.5, True, 32
    if case == 'negative_padding':
        query[0] = -4
        query[1] = -3.5
        key[:, :32] = 4
        key[:, 32] = 3.5
        key[1, :32] = 4.5
        key[1, 32] = 4
        # All real logits are below zero. Padding to 64 keys must contribute
        # neither a larger maximum nor probability mass to the denominator.
    elif case == 'causal_offset':
        query[:] = 1
        key[:, 1:] = 8
        value[:, 1:] = bf16(value[:, 1:] + 20)
        position = 0
    elif case == 'scaled':
        scale, causal, position = .25, False, 0
        query, key = bf16(query * 3), bf16(key * 2)
    elif case == 'unaligned_position':
        position = 31
    else:
        raise ValueError(case)
    expected = reference(query, key, value, scale, causal, position)
    if case == 'causal_offset':
        unmasked = reference(query, key, value, scale, False, position)
        assert np.linalg.norm(unmasked - expected) / np.linalg.norm(expected) > .1
    if case == 'scaled':
        default = reference(query, key, value, 128 ** -.5, causal, position)
        assert np.linalg.norm(default - expected) / np.linalg.norm(expected) > .03
    return query, key, value, expected, dict(scale=scale, causal=causal, query_position=position)


def make_graph(query, key, value, attributes):
    graph = LogicalGraph()

    def cache_input(name, data):
        physical = data.reshape(66, 256)
        ty = graph.type(physical.shape, 'bf16')
        source = f'''module {{ func.func @input(%x: {ty}) -> {ty} {{
          %o = ftlpu.kernel.reshape %x : ({ty}) -> {ty}
          return %o : {ty}
        }} }}'''
        return graph.run(name, source, [(physical, 'bf16')], [(physical.shape, 'bf16')])[0].reshape(data.shape)

    keys, values = cache_input('key_cache', key), cache_input('value_cache', value)
    context = dense_attention(graph, 'attention', query, keys, values, **attributes)
    return graph, [context.reshape(2, 512), keys.reshape(66, 256), values.reshape(66, 256)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('build-dir', 'target-config', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--case', choices=(*CASES, 'all'), default='all')
    args = parser.parse_args()
    args.build_dir, args.target_config = args.build_dir.resolve(), args.target_config.resolve()
    args.output_dir = args.output_dir.resolve()
    extension = '.exe' if sys.platform == 'win32' else ''
    provenance = dict(compiler_sha256=sha(args.build_dir/'compiler'/('ftlpu-compile'+extension)),
                      runner_sha256=sha(args.build_dir/'runtime'/('compiled_stage_file_runner'+extension)),
                      target_sha256=sha(args.target_config),
                      clock_mhz=json.loads(args.target_config.read_text())['external_memory']['lpu_clock_mhz'])
    for case in CASES if args.case == 'all' else (args.case,):
        folder = args.output_dir/case
        folder.mkdir(parents=True, exist_ok=True)
        query, key, value, expected, attributes = fixture(case)
        record = dict(status='running', case=case, batch=2, query_tokens=1, kv_tokens=33,
                      query_heads=4, kv_heads=2, head_dim=128, icu_encoding='raw-macro', sxm='wide',
                      attributes=attributes, **provenance)
        try:
            if case == 'unaligned_position':
                graph, outputs = make_graph(query, key, value, attributes)
                try:
                    compile_graph(graph, outputs, folder, args.build_dir, args.target_config, fp32_spills=True)
                except RuntimeError as exc:
                    message = str(exc)
                    assert 'aligned' in message and 'position' in message, message
                    record.update(status='passed', expected_compile_rejection=message)
                else:
                    raise AssertionError('unaligned query position must be rejected')
                print(f'{case}: expected compilation rejection passed', flush=True)
                continue
            (actual, actual_key, actual_value), phase = execute(
                args, 2, 'dense', make_graph(query, key, value, attributes), folder, provenance)
            # Wide SXM must consume the normalization stream, not a replay of
            # a materialized P pack. Retain the numerical checks below for
            # cross-head routing, transpose order and causal/padding behavior.
            schedule = (folder/'decoder_layer.schedule.mlir').read_text()
            assert schedule.count('softmax.vxm_sxm_direct"') == query.shape[0] * query.shape[2]
            assert 'name = "attention.probability_pack"' not in schedule
            actual = actual.reshape(expected.shape)
            assert np.array_equal(actual_key.reshape(key.shape).view('u4'), key.view('u4'))
            assert np.array_equal(actual_value.reshape(value.shape).view('u4'), value.view('u4'))
            metrics = errors(actual, expected)
            assert np.isfinite(actual).all() and metrics['relative_l2'] < .03, metrics
            assert all(np.linalg.norm(actual[b]-expected[b]) < np.linalg.norm(actual[b]-expected[1-b])
                       for b in range(2)), 'requests mixed'
            record.update(status='passed', metrics=metrics, cache_bit_exact=True,
                          requests=[errors(a, e) for a, e in zip(actual, expected)], phase=phase)
        except Exception as exc:
            record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            (folder/'summary.json').write_text(json.dumps(record, indent=2)+'\n')
        print(f'{case}: passed, relative_l2={metrics["relative_l2"]:.6g}', flush=True)


if __name__ == '__main__':
    main()
