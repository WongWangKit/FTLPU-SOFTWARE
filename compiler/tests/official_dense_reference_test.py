#!/usr/bin/env python3
"""Check GLM's official rotary layout and exact configured frequency lowering."""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from import_hf_decoder_layer import canonicalize_glm_qk, bf16, rope_at_positions, vxm_lut
from run_official_dense_acceptance import source_ir

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tool', type=Path, required=True)
    p.add_argument('--target-config', type=Path, required=True)
    a = p.parse_args()
    variance = np.array([1e-5, 2e-5, 5e-5], dtype=np.float32)
    reciprocal_root = vxm_lut(variance, 'rsqrt')
    assert np.isfinite(reciprocal_root).all()
    np.testing.assert_allclose(reciprocal_root, 1 / np.sqrt(variance), rtol=.002)
    hidden, heads, dim = 4, 2, 8
    x = np.arange(12, dtype=np.float32).reshape(3, hidden) / 8
    q_weight = np.arange(heads * dim * hidden, dtype=np.float32).reshape(heads * dim, hidden) / 32
    k_weight = np.flip(q_weight, axis=0).copy()
    inverse = np.array([1, .125, .015625, .001953125], dtype=np.float32)
    angle = np.arange(3, dtype=np.float32)[:, None] * inverse[None, :]
    c, s = bf16(np.cos(angle))[:, None, :], bf16(np.sin(angle))[:, None, :]
    def official_rotate(weight):
        value = bf16(x @ weight.T).reshape(3, heads, dim)
        lo, hi = value[..., 0::2], value[..., 1::2]
        return bf16(np.stack((lo*c-hi*s, hi*c+lo*s), axis=-1).reshape(value.shape))
    def canonical_rotate(weight):
        value = bf16(x @ canonicalize_glm_qk(weight, dim).T).reshape(3, heads, dim)
        return rope_at_positions(value, np.arange(3, dtype=np.float32), 10000,
                                 inverse_frequencies=inverse)
    q, k = official_rotate(q_weight), official_rotate(k_weight)
    cq, ck = canonical_rotate(q_weight), canonical_rotate(k_weight)
    original_scores = q.transpose(1, 0, 2) @ k.transpose(1, 2, 0)
    canonical_scores = cq.transpose(1, 0, 2) @ ck.transpose(1, 2, 0)
    np.testing.assert_array_equal(original_scores, canonical_scores)
    config = dict(model_type='minicpm', hidden_size=1024, intermediate_size=4096,
        num_attention_heads=16, num_key_value_heads=2, head_dim=64,
        num_hidden_layers=24, scale_depth=1.4, rms_norm_eps=1e-5,
        rope_theta=10000, max_position_embeddings=32768,
        rope_scaling=dict(rope_type='longrope', original_max_position_embeddings=32768,
            short_factor=[float(1+i) for i in range(32)], long_factor=[float(2+i) for i in range(32)]))
    with tempfile.TemporaryDirectory(prefix='official_dense_') as tmp:
        d = Path(tmp)
        source = d / 'layer.mlir'
        source.write_text(source_ir(config, 'official_mini'), encoding='utf-8')
        before = source
        for stage in ('ftlpu-stablehlo-to-stream', 'ftlpu-stream-to-schedule', 'ftlpu-schedule-to-commands'):
            after = d / (stage + '.mlir')
            subprocess.run([str(a.tool), '--input', str(before), '--output', str(after),
                '--pipeline', stage, '--target-config', str(a.target_config), '--weight-bank', '1',
                '--ffn-schedule', 'fused', '--attention-schedule', 'fused',
                '--projection-rope-overlap', 'off'], check=True)
            before = after
        text = before.read_text()
        bindings = [line for line in text.splitlines() if 'initializer = "rope_table"' in line]
        assert bindings and all('inverse_frequencies = dense<' in line for line in bindings), bindings
    print('official dense RoPE layout and frequency lowering passed')

if __name__ == '__main__':
    main()
