"""Numerical and request-isolation regression for the device decode core."""
import argparse
import json
from pathlib import Path
import numpy as np
from glm4_decode_test import attention, bf16, compile_graph, errors, read, run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('build-dir', 'target-config', 'output-dir'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    rng = np.random.default_rng(56); batch = 2
    random = lambda shape: bf16(rng.normal(0, .4, shape).astype('f4'))
    q, k, v = random((batch, 4096)), random((batch, 256)), random((batch, 256))
    pk, pv = random((batch, 32, 2, 128)), random((batch, 32, 2, 128))
    graph, outputs = attention(q, k, v, pk, pv, dict(num_attention_heads=32), diagnostics=True)
    folder = a.output_dir.resolve()
    exe, env, binary = compile_graph(graph, outputs, folder, a.build_dir.resolve(), a.target_config.resolve(), fp32_spills=True)
    run([exe('runtime', 'compiled_stage_file_runner'), binary, folder/'inputs.txt',
        folder/'actual', folder/'runtime.pipeline.csv'], folder/'execution.log', env)
    actual = read(folder/'actual.0.bin', (batch, 4096))
    key = np.concatenate([pk, k.reshape(batch, 1, 2, 128)], axis=1)
    value = np.concatenate([pv, v.reshape(batch, 1, 2, 128)], axis=1)
    scores = np.einsum('bhqd,bthd->bhqt', q.reshape(batch, 2, 16, 128), key) / np.sqrt(128)
    exponent = np.exp(scores-scores.max(-1, keepdims=True))
    probability = exponent/exponent.sum(-1, keepdims=True)
    expected = np.einsum('bhqt,bthd->bhqd', probability, value).reshape(batch, 4096)
    metrics = errors(actual, expected)
    np.testing.assert_array_equal(read(folder/'actual.1.bin', key.shape), key)
    np.testing.assert_array_equal(read(folder/'actual.2.bin', value.shape), value)
    assert np.isfinite(actual).all() and metrics['relative_l2'] < .03, metrics
    assert all(np.linalg.norm(actual[i]-expected[i]) < np.linalg.norm(actual[i]-expected[1-i]) for i in range(2))
    (folder/'summary.json').write_text(json.dumps(dict(status='passed', metrics=metrics,
        batch=batch, past_len=32, independent_kv=True, icu_encoding='raw-macro'), indent=2))
    print('decode core passed', metrics)


if __name__ == '__main__':
    main()
