"""Produce real-weight GLM-4-9B seq32 Q/K/V through the C++ compiler and CModel.

Python imports tensors, describes model equations, and checks an independent
W8A16 oracle. All activation RMSNorm/projection/bias/RoPE work runs as raw-macro
instructions. This fixture ends before attention, O projection, and the FFN.
"""
import argparse
import json
from pathlib import Path
import re
import sys

import numpy as np

from activation_batch_matmul_test import compile_graph, run
from glm4_decode_test import decoder_config, errors, read
from glm4_flash_packed_u_test import graph_digest, sha
from import_hf_decoder_layer import (
    bf16, biased_linear, normalized_rope_linear_at_positions, rms_norm,
    rope_parameters, rope_theta_value,
)
from qwen3_5_logical_graph import LogicalGraph
from summarize_glm4_flash import phase_timing
from tensor_geometry import Value


ROLES = ('query', 'key', 'value')
OUTPUT_SHAPES = ((1, 32, 32, 128), (1, 32, 2, 128), (1, 32, 2, 128))


def load_inputs(checkpoint, imported):
    metadata = json.loads((imported/'metadata.json').read_text())
    config = decoder_config(json.loads((checkpoint/'config.json').read_text()))
    config['head_dim'] = int(metadata['head_dim'])
    assert (metadata['model_type'], metadata['layer'], metadata['seq_len'],
            metadata['hidden_size'], metadata['query_heads'], metadata['kv_heads'],
            metadata['head_dim'], metadata['rotary_dim']) == (
                'chatglm', 0, 32, 4096, 32, 2, 128, 64)
    assert metadata['checkpoint_provenance']['config_sha256'] == sha(checkpoint/'config.json')
    provenance = json.loads((checkpoint/'provenance.json').read_text())
    assert provenance['subset_sha256'] == sha(checkpoint/'model.safetensors')
    assert metadata['checkpoint_provenance']['subset_sha256'] == provenance['subset_sha256']
    widths = dict(query=4096, key=256, value=256)
    weights = {role: np.fromfile(imported/f'{role}.i8.bin', 'i1').reshape(4096, width)
               for role, width in widths.items()}
    biases = {role: read(imported/f'{role}_bias.bf16.bin', (width,))
              for role, width in widths.items()}
    activation = read(imported/'input.bf16.bin', (32, 4096))
    norm = read(imported/'input_layernorm.bf16.bin', (4096,))
    return dict(x=activation, norm=norm, weights=weights, biases=biases,
                config=config, scales=metadata['scales'], metadata=metadata,
                provenance=provenance)


def prefill_rope(graph, name, projected, heads, config):
    """Semantic token-dependent partial RoPE; only immutable tables are host data."""
    tokens, dim = projected.shape[0], int(config['head_dim'])
    inverse, amplitude, rotary = rope_parameters(config, tokens)
    half = rotary // 2
    positions = np.arange(tokens, dtype=np.float32)
    angle = positions[:, None] * inverse[None, :]
    cosine = np.repeat(bf16(np.cos(angle) * amplitude), heads, axis=0)
    sine = np.repeat(bf16(np.sin(angle) * amplitude), heads, axis=0)
    rows = projected.reshape(tokens * heads, dim)

    def equation(g):
        low = g.add(g.add(0, 2), g.add(1, 3), kind='subtract', dtype='bf16')
        high = g.add(g.add(1, 2), g.add(0, 3), kind='add', dtype='bf16')
        return low, high

    low, high = graph.vector_tiles(name, [(rows[:, :half], 'bf16'),
        (rows[:, half:rotary], 'bf16'), (cosine, 'bf16'), (sine, 'bf16')], equation)
    return Value.concatenate((low, high, rows[:, rotary:]), axis=1).reshape(projected.shape)


def projection_graph(fixture):
    graph = LogicalGraph()
    config = fixture['config']
    normalized = graph.norm('input_rmsnorm', fixture['x'], fixture['norm'], config['rms_norm_eps'])
    outputs = []
    for role, heads in (('query', 32), ('key', 2), ('value', 2)):
        projected = graph.project(role+'_projection', normalized,
            fixture['weights'][role].T, fixture['scales'][role])
        projected = graph.vector_tiles(role+'_bias', [(projected, 'bf16'),
            (fixture['biases'][role].reshape(1, -1), 'bf16')],
            lambda g: g.add(0, 1, kind='add', dtype='bf16'))
        if role != 'value':
            projected = prefill_rope(graph, role+'_rope', projected, heads, config)
        outputs.append(projected)
    return graph, outputs


def reference_qkv(fixture):
    """Oracle only: never used as graph inputs or substituted for device outputs."""
    config = fixture['config']
    normalized = rms_norm(fixture['x'], fixture['norm'], config['rms_norm_eps'])
    inverse, amplitude, rotary = rope_parameters(config, 32)
    positions = np.arange(32, dtype=np.float32)
    values = []
    for role, heads in (('query', 32), ('key', 2)):
        values.append(normalized_rope_linear_at_positions(normalized,
            fixture['weights'][role], fixture['scales'][role], fixture['biases'][role],
            heads, rope_theta_value(config), positions, epsilon=config['rms_norm_eps'],
            inverse_frequencies=inverse, attention_factor=amplitude, rotary_dim=rotary))
    values.append(biased_linear(normalized, fixture['weights']['value'],
        fixture['scales']['value'], fixture['biases']['value']))
    return [value.reshape(shape) for value, shape in zip(values, OUTPUT_SHAPES)]


def load_generated_fixture(folder):
    """Return accepted actual BF16 arrays as float32 plus their execution provenance."""
    folder = Path(folder)
    record = json.loads((folder/'fixture.json').read_text())
    assert record['status'] == 'passed'
    assert record['fixture_npz_sha256'] == sha(folder/'fixture.npz')
    for index, digest in enumerate(record['output_sha256']):
        assert digest == sha(folder/f'actual.{index}.bin')
    with np.load(folder/'fixture.npz', allow_pickle=False) as fixture:
        q, k, v = (fixture[key].copy() for key in ('q', 'k', 'v'))
    assert (q.shape, k.shape, v.shape) == OUTPUT_SHAPES
    for value in (q, k, v):
        assert value.dtype == np.float32 and np.array_equal(value, bf16(value))
    return q, k, v, record


def create_fixture(checkpoint, imported, build, target, folder):
    checkpoint, imported, build, target, folder = (
        Path(p).resolve() for p in (checkpoint, imported, build, target, folder))
    folder.mkdir(parents=True, exist_ok=True)
    fixture = load_inputs(checkpoint, imported)
    extension = '.exe' if sys.platform == 'win32' else ''
    compiler = build/'compiler'/('ftlpu-compile'+extension)
    graph_compiler = build/'compiler'/('ftlpu-graph-compile'+extension)
    runner = build/'runtime'/('compiled_stage_file_runner'+extension)
    files = ['metadata.json', 'input.bf16.bin', 'input_layernorm.bf16.bin']
    files += [f'{role}.i8.bin' for role in ROLES]
    files += [f'{role}_bias.bf16.bin' for role in ROLES]
    record = dict(status='running', model='GLM-4-9B W8A16', layer=0,
        batch=1, sequence_length=32, positions=list(range(32)),
        scope='input RMSNorm, QKV projection, QKV bias, Q/K partial RoPE; no attention/O/FFN',
        output_shapes=dict(zip(('q', 'k', 'v'), OUTPUT_SHAPES)),
        precision='W8 symmetric per-tensor; BF16 activations/scales/bias/RoPE tables/output; FP32 MXM accumulation and VXM arithmetic; BF16 RNE vector spill',
        projection_relative_l2_limit=0.03, icu_encoding='raw-macro', sxm='wide',
        checkpoint=str(checkpoint), imported=str(imported), build_dir=str(build), target_config=str(target),
        checkpoint_provenance=fixture['provenance'], checkpoint_config_sha256=sha(checkpoint/'config.json'),
        checkpoint_subset_sha256=fixture['provenance']['subset_sha256'],
        imported_file_sha256={name: sha(imported/name) for name in files},
        compiler_sha256=sha(compiler), graph_compiler_sha256=sha(graph_compiler), runner_sha256=sha(runner),
        target_sha256=sha(target), test_sha256=sha(Path(__file__)),
        clock_mhz=json.loads(target.read_text())['external_memory']['lpu_clock_mhz'],
        host_activation_operations=0)
    result = folder/'fixture.json'
    result.write_text(json.dumps(record, indent=2)+'\n')
    try:
        print('GLM seq32: prepare independent W8A16 QKV oracle', flush=True)
        reference = reference_qkv(fixture)
        graph, outputs = projection_graph(fixture)
        print('GLM seq32: compile RMSNorm/QKV/bias/partial RoPE', flush=True)
        exe, env, binary = compile_graph(graph, outputs, folder, build, target, fp32_spills=False)
        assert (sha(compiler), sha(graph_compiler), sha(runner)) == (
            record['compiler_sha256'], record['graph_compiler_sha256'], record['runner_sha256'])
        print('GLM seq32: execute raw-macro projections', flush=True)
        run([exe('runtime', 'compiled_stage_file_runner'), binary, folder/'inputs.txt',
            folder/'actual', folder/'runtime.pipeline.csv'], folder/'execution.log', env)
        run([exe('runtime', 'ftlpu_icu_program_export'), folder/'actual.runtime.ftlpu',
            folder/'icu_programs', '--pre-execution-dir', folder/'actual.pre_execution'], folder/'export.log', env)
        assert re.search(r'^cycles=.*\bhost_operations=0\b', (folder/'execution.log').read_text(), re.M)
        assert (sha(compiler), sha(graph_compiler), sha(runner)) == (
            record['compiler_sha256'], record['graph_compiler_sha256'], record['runner_sha256'])
        actual = [read(folder/f'actual.{i}.bin', shape) for i, shape in enumerate(OUTPUT_SHAPES)]
        metrics = {role: errors(value, expected) for role, value, expected in zip(ROLES, actual, reference)}
        record['projection_metrics'] = metrics
        assert all(np.isfinite(value).all() for value in actual), 'nonfinite projection output'
        assert all(item['relative_l2'] < record['projection_relative_l2_limit'] for item in metrics.values()), metrics
        np.savez(folder/'fixture.npz', q=actual[0], k=actual[1], v=actual[2],
            reference_q=reference[0], reference_k=reference[1], reference_v=reference[2])
        stamp = dict(binary_sha256=sha(binary), compiler_sha256=record['compiler_sha256'],
                     semantic_graph_sha256=graph_digest(folder))
        (folder/'completed.json').write_text(json.dumps(stamp)+'\n')
        record.update(status='passed', **stamp, phase_timing=phase_timing(folder, record['clock_mhz']),
            output_sha256=[sha(folder/f'actual.{i}.bin') for i in range(3)],
            actual_program_sha256=sha(folder/'actual.runtime.ftlpu'),
            runtime_pipeline_sha256=sha(folder/'runtime.pipeline.csv'),
            fixture_npz_sha256=sha(folder/'fixture.npz'))
        print('GLM seq32 QKV fixture passed:', metrics, flush=True)
    except Exception as error:
        record.update(status='failed', error=str(error))
        raise
    finally:
        result.write_text(json.dumps(record, indent=2)+'\n')
    return load_generated_fixture(folder)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path('models/official_dense/glm4_9b'))
    parser.add_argument('--imported', type=Path, default=Path('tmp/model_tests/glm4_9b/golden'))
    parser.add_argument('--build-dir', type=Path, default=Path('tmp/glm4_dense_build'))
    parser.add_argument('--target-config', type=Path, default=Path('../FTLPU-CMODEL/config/ftlpu-lpu32.json'))
    parser.add_argument('--output-dir', type=Path,
        default=Path('results/glm4_9b_prefill_attention_comparison/projections'))
    args = parser.parse_args()
    create_fixture(args.checkpoint, args.imported, args.build_dir, args.target_config, args.output_dir)


if __name__ == '__main__':
    main()
