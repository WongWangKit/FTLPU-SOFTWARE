#!/usr/bin/env python3
"""Checks Qwen3 MoE routing, expert import, and one-layer golden output."""

from __future__ import annotations

import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


HIDDEN = 8
INTERMEDIATE = 16
MOE_INTERMEDIATE = 8
EXPERTS = 4
TOP_K = 2
SEQ = 32


def values(shape: tuple[int, ...], seed: int) -> np.ndarray:
    index = np.arange(np.prod(shape), dtype=np.int32).reshape(shape)
    return (((index * (seed * 2 + 1) + seed) % 23) - 11).astype(
        np.float32
    ) / np.float32(32.0)


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


def main() -> None:
    importer = Path(__file__).parents[1] / "tools" / "import_hf_decoder_layer.py"
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        model = root / "Qwen3-30B-A3B-mini"
        output = root / "golden"
        model.mkdir()
        config = {
            "model_type": "qwen3_moe",
            "architectures": ["Qwen3MoeForCausalLM"],
            "hidden_size": HIDDEN,
            "intermediate_size": INTERMEDIATE,
            "moe_intermediate_size": MOE_INTERMEDIATE,
            "num_experts": EXPERTS,
            "num_experts_per_tok": TOP_K,
            "norm_topk_prob": True,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "num_hidden_layers": 2,
            "head_dim": 4,
            "hidden_act": "silu",
            "attention_bias": False,
            "rope_theta": 1_000_000.0,
            "rms_norm_eps": 1.0e-6,
            "max_position_embeddings": 128,
            "decoder_sparse_step": 1,
            "mlp_only_layers": [],
        }
        (model / "config.json").write_text(json.dumps(config), encoding="utf-8")
        prefix = "model.layers.0"
        router = np.zeros((EXPERTS, HIDDEN), dtype=np.float32)
        router[0, 0] = 1.0
        router[1, 0] = -1.0
        router[2, 1] = 1.0
        router[3, 1] = -1.0
        gate_up = np.empty(
            (EXPERTS, 2 * MOE_INTERMEDIATE, HIDDEN), dtype=np.float32
        )
        down = np.empty((EXPERTS, HIDDEN, MOE_INTERMEDIATE), dtype=np.float32)
        for expert in range(EXPERTS):
            gate_up[expert] = values((2 * MOE_INTERMEDIATE, HIDDEN), 7 + expert)
            down[expert] = values((HIDDEN, MOE_INTERMEDIATE), 13 + expert)
        tensors = {
            "model.embed_tokens.weight": values((64, HIDDEN), 1),
            f"{prefix}.input_layernorm.weight": np.ones(HIDDEN, dtype=np.float32),
            f"{prefix}.post_attention_layernorm.weight": np.ones(HIDDEN, dtype=np.float32),
            f"{prefix}.self_attn.q_proj.weight": values((8, HIDDEN), 3),
            f"{prefix}.self_attn.k_proj.weight": values((4, HIDDEN), 4),
            f"{prefix}.self_attn.v_proj.weight": values((4, HIDDEN), 5),
            f"{prefix}.self_attn.o_proj.weight": np.zeros((HIDDEN, 8), dtype=np.float32),
            f"{prefix}.self_attn.q_norm.weight": np.ones(4, dtype=np.float32),
            f"{prefix}.self_attn.k_norm.weight": np.ones(4, dtype=np.float32),
            f"{prefix}.mlp.gate.weight": router,
            f"{prefix}.mlp.experts.gate_up_proj": gate_up,
            f"{prefix}.mlp.experts.down_proj": down,
        }
        write_safetensors(model / "model.safetensors", tensors)
        subprocess.run([
            sys.executable, "-B", str(importer),
            "--model-dir", str(model), "--output-dir", str(output),
            "--seq-len", str(SEQ),
        ], check=True)

        metadata = json.loads((output / "metadata.json").read_text())
        expected_moe = {
            "num_experts": EXPERTS,
            "num_experts_per_tok": TOP_K,
            "moe_intermediate_size": MOE_INTERMEDIATE,
            "norm_topk_prob": True,
        }
        for key, expected in expected_moe.items():
            if metadata["moe"][key] != expected:
                raise AssertionError(f"MoE metadata {key} was not preserved")
        expected_sizes = {
            "router.i8.bin": HIDDEN * EXPERTS,
            "expert_gate.i8.bin": EXPERTS * HIDDEN * MOE_INTERMEDIATE,
            "expert_up.i8.bin": EXPERTS * HIDDEN * MOE_INTERMEDIATE,
            "expert_down.i8.bin": EXPERTS * MOE_INTERMEDIATE * HIDDEN,
            "router_indices.i32.bin": SEQ * TOP_K * 4,
            "router_weights.bf16.bin": SEQ * TOP_K * 2,
            "expert_quant_scales.f32.bin": 3 * EXPERTS * 4,
        }
        for name, expected in expected_sizes.items():
            actual = (output / name).stat().st_size
            if actual != expected:
                raise AssertionError(f"{name}: {actual} bytes, expected {expected}")
        indices = np.fromfile(
            output / "router_indices.i32.bin", dtype="<i4"
        ).reshape(SEQ, TOP_K)
        if np.any(indices[:, 0] == indices[:, 1]):
            raise AssertionError("Qwen3 MoE router selected the same expert twice")
        counts = np.fromfile(output / "expert_counts.i32.bin", dtype="<i4")
        if counts.shape != (EXPERTS,) or int(counts.sum()) != SEQ * TOP_K:
            raise AssertionError("Qwen3 MoE dispatch counts lost token assignments")
        weight_bits = np.fromfile(output / "router_weights.bf16.bin", dtype="<u2")
        weights = (weight_bits.astype(np.uint32) << 16).view(np.float32).reshape(
            SEQ, TOP_K
        )
        if not np.allclose(weights.sum(axis=-1), 1.0, atol=0.008):
            raise AssertionError("normalized Qwen3 MoE Top-K weights do not sum to one")
        golden = np.fromfile(output / "golden.bf16.bin", dtype="<u2")
        moe = np.fromfile(output / "golden.moe.bf16.bin", dtype="<u2")
        if not np.any(golden & np.uint16(0x7FFF)):
            raise AssertionError("Qwen3 MoE golden output is empty")
        if not np.any(moe & np.uint16(0x7FFF)):
            raise AssertionError("Qwen3 MoE routed expert output is empty")
    print("Qwen3 MoE importer/reference passed: 4 experts, Top-2, seq32")


if __name__ == "__main__":
    main()
