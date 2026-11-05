"""Original GLM-4: fused weights/bias and adjacent-pair partial RoPE."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from dense_decoder_family_importer_test import write_safetensors, values
from import_hf_decoder_layer import (SafeTensorStore, decoder_config, checkpoint_prefixes,
    load_decoder_weights, load_decoder_biases, canonicalize_glm_qk, glm_qk_permutation,
    bf16, rope_at_positions, rope_parameters)
from run_official_dense_acceptance import source_ir


def main():
    raw = dict(model_type='chatglm', architectures=['ChatGLMModel'], hidden_size=16,
        ffn_hidden_size=24, num_attention_heads=2, kv_channels=8, num_layers=40,
        multi_query_attention=True, multi_query_group_num=1, rmsnorm=True,
        layernorm_epsilon=1.5625e-7, add_bias_linear=False, add_qkv_bias=True,
        apply_residual_connection_post_layernorm=False, fp32_residual_connection=False,
        rope_ratio=2, original_rope=True)
    config = decoder_config(raw)
    assert config['intermediate_size'] == 24 and config['num_key_value_heads'] == 1
    assert config['rope_theta'] == 20000 and config['rms_norm_eps'] == 1.5625e-7
    assert config['partial_rotary_factor'] == .5 and config['attention_bias']
    prefix, embedding = checkpoint_prefixes(raw, 0)
    qkv = values((32, 16), 2)
    bias = values((32,), 3)
    gate_up = values((48, 16), 4)
    tensors = {
        embedding: values((64, 16), 1),
        prefix+'.input_layernorm.weight': np.ones(16, 'f4'),
        prefix+'.post_attention_layernorm.weight': np.ones(16, 'f4'),
        prefix+'.self_attention.query_key_value.weight': qkv,
        prefix+'.self_attention.query_key_value.bias': bias,
        prefix+'.self_attention.dense.weight': values((16, 16), 5),
        prefix+'.mlp.dense_h_to_4h.weight': gate_up,
        prefix+'.mlp.dense_4h_to_h.weight': values((16, 24), 6),
    }
    with tempfile.TemporaryDirectory(prefix='glm4_import_') as temp:
        checkpoint = Path(temp)
        (checkpoint/'config.json').write_text(json.dumps(raw))
        write_safetensors(checkpoint/'model.safetensors', tensors)
        store = SafeTensorStore(checkpoint)
        weights, names = load_decoder_weights(store, prefix, config)
        biases = load_decoder_biases(store, prefix, config)
        for role, begin, end in [('query', 0, 16), ('key', 16, 24), ('value', 24, 32)]:
            np.testing.assert_array_equal(weights[role], qkv[begin:end])
            np.testing.assert_array_equal(biases[role], bias[begin:end])
        np.testing.assert_array_equal(weights['gate'], gate_up[:24])
        np.testing.assert_array_equal(weights['up'], gate_up[24:])
        inv, _, rotary = rope_parameters(config, 32)
        assert rotary == 4
        x = tensors[embedding][:32]
        permutation = glm_qk_permutation(8, rotary)
        np.testing.assert_array_equal(permutation, [0, 2, 1, 3, 4, 5, 6, 7])
        for role, heads in [('query', 2), ('key', 1)]:
            original = bf16(x @ weights[role].T + biases[role]).reshape(32, heads, 8)
            angle = np.arange(32, dtype='f4')[:, None] * inv
            c, s = bf16(np.cos(angle))[:, None], bf16(np.sin(angle))[:, None]
            low, high = original[..., :4:2], original[..., 1:4:2]
            rotated = bf16(np.stack((low*c-high*s, high*c+low*s), axis=-1).reshape(32, heads, 4))
            official = np.concatenate((rotated, original[..., 4:]), axis=-1)
            weight = canonicalize_glm_qk(weights[role], 8, rotary)
            permuted_bias = biases[role].reshape(heads, 8)[:, permutation].reshape(-1)
            value = bf16(x @ weight.T + permuted_bias).reshape(32, heads, 8)
            actual = rope_at_positions(value, np.arange(32, dtype='f4'), 20000,
                                      rotary_dim=rotary, inverse_frequencies=inv)
            np.testing.assert_array_equal(actual, official[..., permutation])
            np.testing.assert_array_equal(actual[..., 4:], original[..., 4:])
        subprocess.run([sys.executable, str(ROOT/'tools/import_hf_decoder_layer.py'),
            '--model-dir', str(checkpoint), '--output-dir', str(checkpoint/'imported'),
            '--seq-len', '32'], check=True, stdout=subprocess.DEVNULL)
        metadata = json.loads((checkpoint/'imported/metadata.json').read_text())
        assert metadata['model_type'] == 'chatglm' and metadata['rotary_dim'] == 4
        assert metadata['attention_bias'] and not metadata['ignored_attention_bias']
        assert set(metadata['bias_tensors'].values()) == {prefix+'.self_attention.query_key_value.bias'}
    full = decoder_config(dict(raw, hidden_size=4096, ffn_hidden_size=13696,
        num_attention_heads=32, multi_query_group_num=2, kv_channels=128, rope_ratio=1))
    ir = source_ir(full, 'glm4_9b')
    assert '%query_bias: tensor<4096xbf16>' in ir
    assert '%attention_query_biased' in ir and 'tensor<32x32x64xf32>' in ir
    assert 'tensor<4096x13696xi8>' in ir
    assert 'dense<1.562500e-07>' in ir
    print('GLM-4 fused weights/bias, partial RoPE and dense graph import passed')


if __name__ == '__main__':
    main()
