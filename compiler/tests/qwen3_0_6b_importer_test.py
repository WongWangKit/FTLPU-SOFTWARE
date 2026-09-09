#!/usr/bin/env python3
"""Checks dense Qwen3 checkpoint import without downloading model weights."""

from __future__ import annotations

import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


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


def values(shape: tuple[int, ...], seed: int) -> np.ndarray:
    index = np.arange(np.prod(shape), dtype=np.int32).reshape(shape)
    return (
        ((index * (2 * seed + 3) + 5 * seed) % 31) - 15
    ).astype(np.float32) / np.float32(32.0)


def main() -> None:
    tool = Path(__file__).parents[1] / "tools" / "import_hf_decoder_layer.py"
    package_tool = (
        Path(__file__).parents[1] / "tools" / "build_decoder_layer_package.py"
    )
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        model = root / "Qwen3-0.6B"
        output = root / "golden"
        model.mkdir()
        config = {
            "model_type": "qwen3",
            "architectures": ["Qwen3ForCausalLM"],
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "attention_bias": False,
            "rope_theta": 1_000_000.0,
            "rms_norm_eps": 1.0e-6,
        }
        (model / "config.json").write_text(
            json.dumps(config), encoding="utf-8"
        )
        prefix = "model.layers.0"
        tensors = {
            "model.embed_tokens.weight": values((32, 8), 1),
            f"{prefix}.input_layernorm.weight": values((8,), 2) + 1.0,
            f"{prefix}.post_attention_layernorm.weight": values((8,), 3) + 1.0,
            f"{prefix}.self_attn.q_proj.weight": values((16, 8), 4),
            f"{prefix}.self_attn.k_proj.weight": values((8, 8), 5),
            f"{prefix}.self_attn.v_proj.weight": values((8, 8), 6),
            f"{prefix}.self_attn.o_proj.weight": values((8, 16), 7),
            f"{prefix}.self_attn.q_norm.weight": values((8,), 8) + 1.0,
            f"{prefix}.self_attn.k_norm.weight": values((8,), 9) + 1.0,
            f"{prefix}.mlp.gate_proj.weight": values((16, 8), 10),
            f"{prefix}.mlp.up_proj.weight": values((16, 8), 11),
            f"{prefix}.mlp.down_proj.weight": values((8, 16), 12),
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
        expected = {
            "architecture": "Qwen3ForCausalLM",
            "hidden_size": 8,
            "query_heads": 2,
            "kv_heads": 1,
            "head_dim": 8,
            "query_width": 16,
            "kv_width": 8,
            "qk_norm": True,
            "attention_bias": False,
        }
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise AssertionError(
                    f"Qwen3 metadata {key}={metadata.get(key)!r}, expected {value!r}"
                )
        expected_sizes = {
            "query.i8.bin": 8 * 16,
            "key.i8.bin": 8 * 8,
            "value.i8.bin": 8 * 8,
            "output.i8.bin": 16 * 8,
            "query_norm.bf16.bin": 8 * 2,
            "key_norm.bf16.bin": 8 * 2,
            "golden.bf16.bin": 32 * 8 * 2,
        }
        for name, size in expected_sizes.items():
            observed = (output / name).stat().st_size
            if observed != size:
                raise AssertionError(
                    f"{name} has {observed} bytes, expected {size}"
                )
        golden_words = np.fromfile(output / "golden.bf16.bin", dtype="<u2")
        if not np.any(golden_words & np.uint16(0x7FFF)):
            raise AssertionError("Qwen3 imported golden is empty")
        if '"qk_norm": true' not in completed.stdout:
            raise AssertionError("importer did not report Q/K normalization")

        executable = root / "decoder.ftlpu"
        package = root / "qwen3.ftlpum"
        executable.write_bytes(b"synthetic-qwen3-executable")
        subprocess.run(
            [
                sys.executable,
                "-B",
                str(package_tool),
                "--golden-dir",
                str(output),
                "--executable",
                str(executable),
                "--output",
                str(package),
                "--kv-cache-capacity",
                "64",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        if package.stat().st_size <= executable.stat().st_size:
            raise AssertionError("Qwen3 model package was not assembled")

    print("Qwen3 checkpoint importer test passed")


if __name__ == "__main__":
    main()
