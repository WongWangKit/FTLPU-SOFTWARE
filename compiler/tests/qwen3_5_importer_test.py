#!/usr/bin/env python3
"""Checks Qwen3.5 nested config and Gated DeltaNet checkpoint import."""

from __future__ import annotations

import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
from qwen3_5_compiled_layer import load_imported_layer
from import_hf_decoder_layer import quantize_linear


def write_safetensors(path: Path, tensors: dict[str, np.ndarray]) -> None:
    header: dict[str, object] = {}
    payload = bytearray()
    for name, value in tensors.items():
        array = np.ascontiguousarray(value, dtype="<f4")
        begin = len(payload)
        payload.extend(array.tobytes())
        header[name] = {
            "dtype": "F32",
            "shape": list(array.shape),
            "data_offsets": [begin, len(payload)],
        }
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with path.open("wb") as stream:
        stream.write(struct.pack("<Q", len(encoded)))
        stream.write(encoded)
        stream.write(payload)


def values(shape: tuple[int, ...], seed: int, scale: float = 0.04) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, scale, shape).astype(np.float32)


def main() -> None:
    tool = Path(__file__).parents[1] / "tools" / "import_hf_qwen3_5_layer.py"
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        model = root / "Qwen3.5-mini"
        output = root / "golden"
        model.mkdir()
        text = {
            "model_type": "qwen3_5_text",
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 4,
            "layer_types": [
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ],
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_key_head_dim": 8,
            "linear_value_head_dim": 8,
            "linear_conv_kernel_dim": 4,
            "rms_norm_eps": 1.0e-6,
        }
        config = {
            "model_type": "qwen3_5",
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "text_config": text,
        }
        (model / "config.json").write_text(json.dumps(config), encoding="utf-8")
        prefix = "model.language_model.layers.0"
        conv_width = 64
        tensors = {
            "model.language_model.embed_tokens.weight": values((64, 32), 1),
            f"{prefix}.input_layernorm.weight": values((32,), 2, 0.01),
            f"{prefix}.post_attention_layernorm.weight": values((32,), 3, 0.01),
            f"{prefix}.linear_attn.in_proj_qkv.weight": values((conv_width, 32), 4),
            f"{prefix}.linear_attn.in_proj_z.weight": values((32, 32), 5),
            f"{prefix}.linear_attn.in_proj_b.weight": values((4, 32), 6),
            f"{prefix}.linear_attn.in_proj_a.weight": values((4, 32), 7),
            f"{prefix}.linear_attn.conv1d.weight": values((conv_width, 1, 4), 8, 0.08),
            f"{prefix}.linear_attn.dt_bias": values((4,), 9, 0.03) + 0.1,
            f"{prefix}.linear_attn.A_log": values((4,), 10, 0.03) - 1.0,
            f"{prefix}.linear_attn.norm.weight": values((8,), 11, 0.02) + 1.0,
            f"{prefix}.linear_attn.out_proj.weight": values((32, 32), 12),
            f"{prefix}.mlp.gate_proj.weight": values((64, 32), 13),
            f"{prefix}.mlp.up_proj.weight": values((64, 32), 14),
            f"{prefix}.mlp.down_proj.weight": values((32, 64), 15),
        }
        write_safetensors(model / "model.safetensors", tensors)
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                str(tool),
                "--model-dir",
                str(model),
                "--output-dir",
                str(output),
                "--seq-len",
                "32",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        metadata = json.loads((output / "metadata.json").read_text())
        _, _, packed, _, _ = load_imported_layer(output)
        expected_qkv, _ = quantize_linear(tensors[f"{prefix}.linear_attn.in_proj_qkv.weight"])
        np.testing.assert_array_equal(packed["in_proj_qkv"], expected_qkv.T)
        expected_down, _ = quantize_linear(tensors[f"{prefix}.mlp.down_proj.weight"])
        np.testing.assert_array_equal(packed["down_proj"], expected_down.T)
        expected = {
            "model_type": "qwen3_5",
            "text_model_type": "qwen3_5_text",
            "layer_type": "linear_attention",
            "seq_len": 32,
            "conv_width": 64,
            "conv_state_shape": [64, 4],
            "conv_state_bytes_bf16": 512,
            "recurrent_state_shape": [4, 8, 8],
            "recurrent_state_bytes_fp32": 1024,
            "projection_format": "w8a16_per_tensor",
        }
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise AssertionError(
                    f"Qwen3.5 metadata {key}={metadata.get(key)!r}, expected {value!r}"
                )
        expected_sizes = {
            "in_proj_qkv.i8.bin": 64 * 32,
            "conv1d.bf16.bin": 64 * 4 * 2,
            "dt_bias.f32.bin": 4 * 4,
            "a_log.f32.bin": 4 * 4,
            "golden.bf16.bin": 32 * 32 * 2,
            "golden.conv_state.bf16.bin": 64 * 4 * 2,
            "golden.recurrent_state.f32.bin": 4 * 8 * 8 * 4,
            "quant_scales.f32.bin": 8 * 4,
            "input_norm.bf16.bin": 32 * 2,
            "post_attention_norm.bf16.bin": 32 * 2,
        }
        for name, size in expected_sizes.items():
            observed = (output / name).stat().st_size
            if observed != size:
                raise AssertionError(f"{name} has {observed} bytes, expected {size}")
        golden = np.fromfile(output / "golden.bf16.bin", dtype="<u2")
        if not np.any(golden & np.uint16(0x7FFF)):
            raise AssertionError("Qwen3.5 imported golden is empty")
        if '"layer_type": "linear_attention"' not in completed.stdout:
            raise AssertionError("Qwen3.5 importer did not report its hybrid layer type")
    print("Qwen3.5 Gated DeltaNet checkpoint importer passed")


if __name__ == "__main__":
    main()
