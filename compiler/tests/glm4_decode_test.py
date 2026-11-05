"""Official GLM-4-9B layer-0 decode, one independent KV history per request.

Batch projections/FFN and the attention core are actual CModel invocations.
The adapter emits mathematical Kernel graphs and transfers their outputs;
it never supplies a host-computed activation to replace a device operation.
Past KV is initialized from an independent prefill reference (fixture only).
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
from contextlib import contextmanager

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from import_hf_decoder_layer import (bf16, decoder_config, decoder_layer_decode_reference,
    decoder_layer_prefill_kv_reference, rope_parameters, write_bf16)
from qwen3_5_logical_graph import LogicalGraph
from tensor_geometry import Value
from activation_batch_matmul_test import batch_matmul, compile_graph, run
from qwen3_5_decode_gdn_batched_test import concatenate_traces


@contextmanager
def cached_reference_weights():
    """Cache immutable dequantized weights inside the CPU oracle only.

    The expression and BF16 rounding are identical to reference.linear;
    no device input or executed activation uses this cached computation.
    """
    import import_hf_decoder_layer as reference
    original = reference.linear
    cache = {}
    def linear(value, weight, scale, bf16_scale=False):
        if bf16_scale:
            scale = float(bf16(np.asarray([scale], dtype=np.float32))[0])
        key = (id(weight), float(scale))
        if key not in cache:
            cache[key] = bf16(weight.astype(np.float32) * scale)
        return bf16(value @ cache[key])
    reference.linear = linear
    try:
        yield
    finally:
        reference.linear = original
        cache.clear()


def read(path, shape):
    return (np.fromfile(path, '<u2').astype('u4') << 16).view('f4').reshape(shape)


def errors(actual, reference):
    delta = actual.astype('f8') - reference.astype('f8')
    denominator = max(float(np.linalg.norm(reference.astype('f8'))), 1e-30)
    return dict(max_abs=float(np.max(np.abs(delta))), mean_abs=float(np.mean(np.abs(delta))),
                relative_l2=float(np.linalg.norm(delta) / denominator),
                cosine=float(np.sum(actual.astype('f8') * reference) /
                             max(np.linalg.norm(actual.astype('f8')) * denominator, 1e-30)))


def identity(graph, name, value):
    t = graph.type(value.shape, 'bf16')
    source = f'''module {{ func.func @stage(%x: {t}) -> {t} {{
      %o = ftlpu.kernel.reshape %x : ({t}) -> {t}
      return %o : {t}
    }} }}'''
    return graph.run(name, source, [(value, 'bf16')], [(value.shape, 'bf16')])[0]


def rope(graph, name, value, heads, config, position):
    dim = int(config['head_dim'])
    inverse, amplitude, rotary = rope_parameters(config, position + 1)
    half = rotary // 2
    x = value.reshape(-1, dim)
    # Only immutable position coefficients are computed on the host.
    cosine = bf16((np.cos(position * inverse) * amplitude).reshape(1, half))
    sine = bf16((np.sin(position * inverse) * amplitude).reshape(1, half))
    def equation(g):
        low = g.add(g.add(0, 2), g.add(1, 3), kind='subtract', dtype='bf16')
        high = g.add(g.add(1, 2), g.add(0, 3), kind='add', dtype='bf16')
        return low, high
    low, high = graph.vector_tiles(name, [(x[:, :half], 'bf16'),
        (x[:, half:rotary], 'bf16'), (cosine, 'bf16'), (sine, 'bf16')], equation)
    return Value.concatenate([low, high, x[:, rotary:]], axis=1).reshape(value.shape)


def projections(x, packed, scales, biases, norm0, config, position):
    graph = LogicalGraph()
    normalized = graph.norm('input_rmsnorm', x, norm0, config['rms_norm_eps'])
    outputs = []
    for role, heads in [('query', 32), ('key', 2), ('value', 2)]:
        projected = graph.project(role + '_projection', normalized, packed[role].T, scales[role])
        projected = graph.vector_tiles(role + '_bias', [(projected, 'bf16'),
            (biases[role].reshape(1, -1), 'bf16')], lambda g: g.add(0, 1, kind='add', dtype='bf16'))
        if role != 'value':
            projected = rope(graph, role + '_rope', projected, heads, config, position)
        outputs.append(projected)
    return graph, outputs


def attention(q, k, v, past_key, past_value, config, *, diagnostics=False):
    graph = LogicalGraph()
    batch, past, kv, dim = past_key.shape
    heads = int(config['num_attention_heads'])
    padded = ((past + 1 + 31) // 32) * 32
    query = identity(graph, 'query_input', q).reshape(batch * kv, heads // kv, dim)
    keys, values = [], []
    for name, history, current, output in [('key', past_key, k, keys), ('value', past_value, v, values)]:
        old = identity(graph, 'past_' + name, history.reshape(batch * past, kv * dim)).reshape(batch, past, kv, dim)
        new = identity(graph, 'new_' + name, current).reshape(batch, 1, kv, dim)
        zero = identity(graph, 'padding_' + name, np.zeros((batch * (padded-past-1), kv*dim), 'f4')).reshape(batch, padded-past-1, kv, dim)
        output.append(Value.concatenate([old, new, zero], axis=1))
    key, value = keys[0], values[0]
    scores = batch_matmul(graph, 'decode_qk', query, key.transpose(0, 2, 1, 3).reshape(batch*kv, padded, dim), True)
    scores = graph.vector_tiles('score_scale', [(scores.reshape(batch*heads, padded), 'bf16')],
        lambda g: g.add(0, immediate=1 / np.sqrt(dim), dtype='bf16'))
    mask = np.zeros((1, padded), 'f4'); mask[:, past+1:] = -1e9
    scores = graph.vector_tiles('kv_length_mask', [(scores, 'bf16'), (bf16(mask), 'bf16')],
        lambda g: g.add(0, 1, kind='add', dtype='bf16'))
    matrix = graph.type((batch*heads, padded), 'bf16')
    full = graph.type((batch*heads, padded), 'f32')
    row = graph.type((batch*heads, 1), 'f32')
    source = f'''module {{ func.func @softmax(%x: {matrix}) -> ({matrix}, {row}, {full}, {row})
      attributes {{ftlpu.vector_bf16_spill = false}} {{
      %max = ftlpu.kernel.vector_alu %x, %x {{kind = "max", use_immediate = true,
        immediate = 0.0 : f32, reduce = true, reduction_kind = "max"}} : ({matrix}, {matrix}) -> {row}
      %shift = ftlpu.kernel.vector_alu %x, %max {{kind = "subtract", use_immediate = false,
        immediate = 0.0 : f32, reduce = false}} : ({matrix}, {row}) -> {full}
      %exp = ftlpu.kernel.vector_alu %shift, %shift {{kind = "exp", use_immediate = true,
        immediate = 0.0 : f32, reduce = false}} : ({full}, {full}) -> {full}
      %sum = ftlpu.kernel.vector_alu %exp, %exp {{kind = "multiply", use_immediate = true,
        immediate = 1.0 : f32, reduce = true}} : ({full}, {full}) -> {row}
      %inverse = ftlpu.kernel.vector_alu %sum, %sum {{kind = "reciprocal", use_immediate = true,
        immediate = 0.0 : f32, reduce = false}} : ({row}, {row}) -> {row}
      %prob = ftlpu.kernel.vector_alu %exp, %inverse {{kind = "multiply", use_immediate = false,
        immediate = 0.0 : f32, reduce = false}} : ({full}, {row}) -> {matrix}
      return %prob, %max, %exp, %sum : {matrix}, {row}, {full}, {row}
    }} }}'''
    probability, maximum, exponent, denominator = graph.run('softmax', source,
        [(scores, 'bf16')], [((batch*heads, padded), 'bf16'), ((batch*heads, 1), 'f32'),
                            ((batch*heads, padded), 'f32'), ((batch*heads, 1), 'f32')])
    context = batch_matmul(graph, 'decode_pv', probability.reshape(batch*kv, heads//kv, padded),
        value.transpose(0, 2, 1, 3).reshape(batch*kv, padded, dim))
    outputs = [context.reshape(batch, heads*dim),
                   key[:, :past+1].reshape(batch*(past+1), kv*dim),
                   value[:, :past+1].reshape(batch*(past+1), kv*dim)]
    if diagnostics:
        outputs += [scores, maximum, exponent, denominator, probability]
    return graph, outputs


def tail(x, context, norm1, packed, scales, config):
    graph = LogicalGraph()
    output = graph.project('o_projection', context, packed['output'].T, scales['output'])
    residual = graph.vector_tiles('attention_residual', [(x, 'bf16'), (output, 'bf16')],
        lambda g: g.add(0, 1, kind='add', dtype='bf16'))
    normalized = graph.norm('post_attention_rmsnorm', residual, norm1, config['rms_norm_eps'])
    down = graph.ffn(normalized, {r+'_proj': packed[r].T for r in ('gate', 'up', 'down')},
                     {r+'_proj': scales[r] for r in ('gate', 'up', 'down')})
    result = graph.vector_tiles('ffn_residual', [(residual, 'bf16'), (down, 'bf16')],
        lambda g: g.add(0, 1, kind='add', dtype='bf16'))
    return graph, [result]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'imported', 'build-dir', 'target-config', 'output-dir'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--batches', type=int, nargs='+', default=[1, 32, 64])
    p.add_argument('--past-len', type=int, default=32)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--reuse-completed-phase', nargs='*', default=[], choices=('projections', 'attention', 'tail'),
        help='With --resume, keep these completed device binaries across compiler rebuilds; input/program hashes and numerical checks still apply')
    a = p.parse_args()
    if a.past_len != 32 or any(b not in (1, 32, 64) for b in a.batches):
        p.error('This acceptance fixture covers past32 and batch 1/32/64')
    root, imported, build, target = (v.resolve() for v in (a.output_dir, a.imported, a.build_dir, a.target_config))
    root.mkdir(parents=True, exist_ok=True)
    config = decoder_config(json.loads((a.checkpoint/'config.json').read_text()))
    config['head_dim'] = 128
    meta = json.loads((imported/'metadata.json').read_text())
    assert (meta['model_type'], meta['hidden_size'], meta['intermediate_size']) == ('chatglm', 4096, 13696)
    scales = meta['scales']; hidden, intermediate, width = 4096, 13696, 256
    shapes = dict(query=(hidden, hidden), key=(hidden, width), value=(hidden, width),
        output=(hidden, hidden), gate=(hidden, intermediate), up=(hidden, intermediate), down=(intermediate, hidden))
    packed = {role: np.fromfile(imported/f'{role}.i8.bin', 'i1').reshape(shape) for role, shape in shapes.items()}
    norm0 = read(imported/'input_layernorm.bf16.bin', (hidden,))
    norm1 = read(imported/'post_attention_layernorm.bf16.bin', (hidden,))
    biases = {role: read(imported/f'{role}_bias.bf16.bin', (shapes[role][1],)) for role in ('query', 'key', 'value')}
    embeddings = read(imported/'input.bf16.bin', (32, hidden))
    signatures = dict(metadata=hashlib.sha256((imported/'metadata.json').read_bytes()).hexdigest(),
        compiler=hashlib.sha256((build/'compiler/ftlpu-compile.exe').read_bytes()).hexdigest(),
        runner=hashlib.sha256((build/'runtime/compiled_stage_file_runner.exe').read_bytes()).hexdigest(),
        test=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        helper=hashlib.sha256(Path(__file__).with_name('activation_batch_matmul_test.py').read_bytes()).hexdigest(),
        target=hashlib.sha256(target.read_bytes()).hexdigest())
    manifest = root/'signature.json'
    if a.resume and manifest.exists():
        previous = json.loads(manifest.read_text())
        if any(previous.get(key) != signatures[key] for key in ('metadata', 'runner', 'target')):
            raise ValueError('resume hardware/weights signature differs; rerun without --resume')
    else:
        previous = {}
    manifest.write_text(json.dumps(signatures, indent=2))
    records = []
    for batch in a.batches:
        folder = root/f'batch{batch}'; folder.mkdir(exist_ok=True)
        record = dict(status='running', model='GLM-4-9B', layer=0, batch=batch,
            past_len=32, decode_tokens_per_request=1, precision='official W8A16',
            icu_encoding='raw-macro', sxm='wide', scope='full decoder layer; three device phases',
            cache_source='independent quantized prefill reference; different token histories',
            host_activation_operations=0, checkpoint_provenance=meta.get('checkpoint_provenance'))
        (folder/'result.json').write_text(json.dumps(record, indent=2))
        print(f'batch={batch}: prepare independent histories and reference', flush=True)
        rng = np.random.default_rng(4096032)
        token_ids = np.stack([rng.permutation(32) for _ in range(batch)])
        current_ids = np.arange(batch)*7 % 32
        x = embeddings[current_ids].copy()
        cache = folder/'reference.npz'
        fixture_hash = hashlib.sha256(b'glm4-independent-decode-fixture-v1')
        fixture_hash.update(json.dumps(config, sort_keys=True).encode())
        fixture_hash.update(token_ids.tobytes()); fixture_hash.update(current_ids.tobytes())
        fixture_hash.update(embeddings.tobytes()); fixture_hash.update(norm0.tobytes()); fixture_hash.update(norm1.tobytes())
        fixture_hash.update(json.dumps(scales, sort_keys=True).encode())
        for role in sorted(packed): fixture_hash.update(packed[role].tobytes())
        for role in sorted(biases): fixture_hash.update(biases[role].tobytes())
        fingerprint = fixture_hash.hexdigest()
        cached = np.load(cache, allow_pickle=False) if a.resume and cache.exists() else None
        if cached is not None and str(cached['fingerprint']) == fingerprint:
            past_key, past_value, golden, expected_key, expected_value = [cached[key] for key in
                ('past_key', 'past_value', 'golden', 'expected_key', 'expected_value')]
            cached.close()
            print(f'batch={batch}: reused matching reference fixture', flush=True)
        else:
            if cached is not None: cached.close()
            past_key, past_value, golden, expected_key, expected_value = [], [], [], [], []
            with cached_reference_weights():
                for request in range(batch):
                    pk, pv = decoder_layer_prefill_kv_reference(embeddings[token_ids[request]], norm0, packed, scales, config, biases)
                    y, nk, nv = decoder_layer_decode_reference(x[request:request+1], pk, pv, 32, norm0, norm1, packed, scales, config, biases)
                    past_key.append(pk); past_value.append(pv); golden.append(y); expected_key.append(nk); expected_value.append(nv)
            past_key, past_value = np.stack(past_key), np.stack(past_value)
            golden, expected_key, expected_value = np.concatenate(golden), np.stack(expected_key), np.stack(expected_value)
            np.savez(cache, fingerprint=fingerprint, past_key=past_key, past_value=past_value,
                     golden=golden, expected_key=expected_key, expected_value=expected_value)
        (folder/'token_ids.json').write_text(json.dumps(dict(past=token_ids.tolist(), current=current_ids.tolist())))
        write_bf16(folder/'golden.bf16.bin', golden)
        def execute(name, graph, finals, shapes):
            phase = folder/name; stamp = phase/'completed.json'
            # A long run may outlive a compiler rebuild between phases.
            # Attribute each phase to the compiler actually used for it.
            compiler_hash = hashlib.sha256((build/'compiler'/
                ('ftlpu-compile.exe' if sys.platform == 'win32' else 'ftlpu-compile')).read_bytes()).hexdigest()
            helper_hash = hashlib.sha256(Path(__file__).with_name('activation_batch_matmul_test.py').read_bytes()).hexdigest()
            def graph_hash():
                path = phase/'logical/decoder_layer.graph.json'
                if not path.exists():
                    return None
                data = path.read_bytes(); digest = hashlib.sha256(data)
                for constant in json.loads(data)['constants']:
                    digest.update(Path(constant['file']).read_bytes())
                return digest.hexdigest()
            old_graph = graph_hash() if stamp.exists() else None
            graph.write(phase/'logical', finals)
            current_graph = graph_hash()
            reusable = a.resume and stamp.exists() and old_graph == current_graph
            saved = json.loads(stamp.read_text()) if reusable else {}
            reusable = reusable and saved.get('graph_sha256') == current_graph
            valid = reusable and (name in a.reuse_completed_phase or
                (saved.get('compiler_sha256') == compiler_hash and saved.get('helper_sha256') == helper_hash))
            if valid:
                valid = saved['binary_sha256'] == hashlib.sha256((phase/'program.ftlpu').read_bytes()).hexdigest()
                if valid:
                    print(f'batch={batch}: {name} reused completed device execution', flush=True)
            if not valid:
                print(f'batch={batch}: {name} compile', flush=True)
                exe, env, binary = compile_graph(graph, finals, phase, build, target, fp32_spills=name == 'attention')
                # A compiler rebuild may leave an unaffected phase byte-identical.
                # Reuse only when both semantic inputs and encoded program match.
                if reusable and saved['binary_sha256'] == hashlib.sha256(binary.read_bytes()).hexdigest():
                    print(f'batch={batch}: {name} reused byte-identical executed program', flush=True)
                    saved.update(compiler_sha256=compiler_hash, helper_sha256=helper_hash, graph_sha256=current_graph)
                    stamp.write_text(json.dumps(saved))
                    return [read(phase/f'actual.{i}.bin', shape) for i, shape in enumerate(shapes)]
                print(f'batch={batch}: {name} CModel', flush=True)
                run([exe('runtime', 'compiled_stage_file_runner'), binary, phase/'inputs.txt',
                     phase/'actual', phase/'runtime.pipeline.csv'], phase/'execution.log', env)
                run([exe('runtime', 'ftlpu_icu_program_export'), phase/'actual.runtime.ftlpu', phase/'icu_programs',
                     '--pre-execution-dir', phase/'actual.pre_execution'], phase/'export.log', env)
                stamp.write_text(json.dumps(dict(binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                    graph_sha256=current_graph, compiler_sha256=compiler_hash, helper_sha256=helper_hash)))
            return [read(phase/f'actual.{i}.bin', shape) for i, shape in enumerate(shapes)]
        try:
            graph, finals = projections(x, packed, scales, biases, norm0, config, 32)
            q, k, v = execute('projections', graph, finals, [(batch, hidden), (batch, width), (batch, width)])
            record['projection_cache_metrics'] = dict(
                key=errors(k.reshape(batch, 2, 128), expected_key[:, 32]),
                value=errors(v.reshape(batch, 2, 128), expected_value[:, 32]))
            assert np.isfinite(q).all() and all(e['relative_l2'] < .03
                for e in record['projection_cache_metrics'].values()), record['projection_cache_metrics']
            graph, finals = attention(q, k, v, past_key, past_value, config)
            context, actual_key, actual_value = execute('attention', graph, finals,
                [(batch, hidden), (batch, 33, 2, 128), (batch, 33, 2, 128)])
            # Validate attention before launching the expensive FFN. This CPU
            # calculation is an oracle only; execution consumes CModel context.
            scores = np.einsum('bhqd,bthd->bhqt', q.reshape(batch, 2, 16, 128), actual_key) / np.sqrt(128)
            exponential = np.exp(scores - scores.max(-1, keepdims=True))
            probabilities = exponential / exponential.sum(-1, keepdims=True)
            context_reference = np.einsum('bhqt,bthd->bhqd', probabilities, actual_value).reshape(batch, hidden)
            record['attention_metrics'] = errors(context, context_reference)
            assert np.isfinite(context).all() and record['attention_metrics']['relative_l2'] < .03, record['attention_metrics']
            (folder/'result.json').write_text(json.dumps(record, indent=2))
            graph, finals = tail(x, context, norm1, packed, scales, config)
            actual, = execute('tail', graph, finals, [(batch, hidden)])
            metrics = errors(actual, golden)
            per_request = [errors(actual[i], golden[i]) for i in range(batch)]
            cache_errors = dict(key=errors(actual_key, expected_key), value=errors(actual_value, expected_value))
            assert np.array_equal(actual_key[:, :32], past_key), 'historical keys changed'
            assert np.array_equal(actual_value[:, :32], past_value), 'historical values changed'
            assert np.isfinite(actual).all() and metrics['relative_l2'] < .05, metrics
            assert max(r['relative_l2'] for r in per_request) < .08, per_request
            new_cache_errors = dict(key=errors(actual_key[:, 32], expected_key[:, 32]),
                                    value=errors(actual_value[:, 32], expected_value[:, 32]))
            assert all(e['relative_l2'] < .03 for e in new_cache_errors.values()), new_cache_errors
            paths = [(name, folder/name/'runtime.pipeline.csv') for name in ('projections', 'attention', 'tail')]
            cycles = concatenate_traces(paths, folder/'runtime.pipeline.csv')
            record.update(status='passed', metrics=metrics, requests=per_request,
                cache_errors=cache_errors, new_cache_errors=new_cache_errors, runtime_cycles=cycles)
            write_bf16(folder/'actual.bf16.bin', actual)
        except Exception as error:
            record.update(status='failed', error=str(error))
            raise
        finally:
            (folder/'result.json').write_text(json.dumps(record, indent=2))
            records.append({k:v for k,v in record.items() if k != 'checkpoint_provenance'})
            summaries = [json.loads(path.read_text()) for path in sorted(root.glob('batch*/result.json'))]
            temporary = root/f'summary.{os.getpid()}.tmp'
            temporary.write_text(json.dumps(summaries, indent=2))
            temporary.replace(root/'summary.json')
        print(f'batch={batch}: passed {metrics}', flush=True)


if __name__ == '__main__':
    main()
