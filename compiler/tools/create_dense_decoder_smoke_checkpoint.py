#!/usr/bin/env python3
"""Create a deterministic, full-shape HF checkpoint for dense decoder smoke tests."""

from __future__ import annotations

import argparse
import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ModelSpec:
    model_type: str
    architecture: str
    hidden: int
    intermediate: int
    query_heads: int
    kv_heads: int
    head_dim: int
    layers: int
    rope_theta: float
    rms_epsilon: float


SPECS = {
    "llama3_2_1b": ModelSpec(
        "llama", "LlamaForCausalLM", 2048, 8192, 32, 8, 64, 16,
        500_000.0, 1.0e-5,
    ),
    "ministral3_3b": ModelSpec(
        "ministral3", "Ministral3ForCausalLM", 3072, 9216, 32, 8, 128,
        36, 1_000_000.0, 1.0e-5,
    ),
    "phi4_mini": ModelSpec(
        "phi3", "Phi3ForCausalLM", 3072, 8192, 24, 8, 128, 32,
        10_000.0, 1.0e-5,
    ),
    "smollm3_3b": ModelSpec(
        "smollm3", "SmolLM3ForCausalLM", 2048, 11008, 16, 4, 128, 36,
        5_000_000.0, 1.0e-5,
    ),
    "minicpm4_0_5b": ModelSpec(
        "minicpm", "MiniCPMForCausalLM", 1024, 4096, 16, 2, 64, 24,
        10_000.0, 1.0e-5,
    ),
    "glm_edge_1_5b": ModelSpec(
        "glm", "GlmForCausalLM", 2048, 6144, 16, 4, 128, 28,
        10_000.0, 1.0e-5,
    ),
}


def model_config(name: str, spec: ModelSpec) -> tuple[dict[str, object], str, str]:
    text: dict[str, object] = {
        "model_type": spec.model_type,
        "architectures": [spec.architecture],
        "hidden_size": spec.hidden,
        "intermediate_size": spec.intermediate,
        "num_attention_heads": spec.query_heads,
        "num_key_value_heads": spec.kv_heads,
        "num_hidden_layers": spec.layers,
        "head_dim": spec.head_dim,
        "hidden_act": "silu",
        "attention_bias": False,
        "rope_theta": spec.rope_theta,
        "rms_norm_eps": spec.rms_epsilon,
        "max_position_embeddings": 131_072,
    }
    prefix = "model.layers.0"
    embedding = "model.embed_tokens.weight"
    if name == "llama3_2_1b":
        text["rope_scaling"] = {
            "rope_type": "llama3", "factor": 32.0,
            "low_freq_factor": 1.0, "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
        }
    elif name == "ministral3_3b":
        text.pop("rope_theta")
        text["rope_parameters"] = {
            "rope_type": "yarn", "rope_theta": spec.rope_theta,
            "factor": 8.0, "beta_fast": 32.0, "beta_slow": 1.0,
            "mscale": 1.0, "mscale_all_dim": 1.0,
            "original_max_position_embeddings": 32_768,
        }
        prefix = "language_model.model.layers.0"
        embedding = "language_model.model.embed_tokens.weight"
        return {
            "model_type": "mistral3",
            "architectures": ["Mistral3ForConditionalGeneration"],
            "text_config": text,
        }, prefix, embedding
    elif name == "phi4_mini":
        text["partial_rotary_factor"] = 0.75
        text["original_max_position_embeddings"] = 4096
        text["rope_scaling"] = {
            "type": "longrope",
            "short_factor": [1.0] * 48,
            "long_factor": [2.0] * 48,
        }
    elif name == "smollm3_3b":
        text["no_rope_layers"] = [1] * spec.layers
        text["layer_types"] = ["full_attention"] * spec.layers
    elif name == "minicpm4_0_5b":
        # MiniCPM4 identifies itself through its architecture/auto_map.
        text.pop("model_type")
        text["scale_emb"] = 12.0
        text["scale_depth"] = 1.4
    elif name == "glm_edge_1_5b":
        text["partial_rotary_factor"] = 1.0
    return text, prefix, embedding


def tensor_layout(
    name: str, spec: ModelSpec, prefix: str, embedding: str,
    component: str, qk_pattern: str = "zero",
) -> list[tuple[str, str, tuple[int, ...], str]]:
    query = spec.query_heads * spec.head_dim
    kv = spec.kv_heads * spec.head_dim
    tensors: list[tuple[str, str, tuple[int, ...], str]] = [
        (embedding, "F32", (64, spec.hidden), "embedding"),
        (f"{prefix}.input_layernorm.weight", "F32", (spec.hidden,), "ones"),
        (f"{prefix}.post_attention_layernorm.weight", "F32", (spec.hidden,), "ones"),
    ]
    attention_enabled = component in ("attention", "both")
    ffn_enabled = component in ("ffn", "both")
    if spec.model_type == "phi3":
        tensors.append((
            f"{prefix}.self_attn.qkv_proj.weight", "I8",
            (query + 2 * kv, spec.hidden),
            ("fused_qkv_nonzero" if qk_pattern == "nonzero" else "fused_qkv")
                if attention_enabled else "zero",
        ))
    else:
        tensors.extend([
            (f"{prefix}.self_attn.q_proj.weight", "I8", (query, spec.hidden),
             "diag" if attention_enabled and qk_pattern == "nonzero" else "zero"),
            (f"{prefix}.self_attn.k_proj.weight", "I8", (kv, spec.hidden),
             "diag" if attention_enabled and qk_pattern == "nonzero" else "zero"),
            (f"{prefix}.self_attn.v_proj.weight", "I8", (kv, spec.hidden),
             "diag" if attention_enabled else "zero"),
        ])
    tensors.append((
        f"{prefix}.self_attn.o_proj.weight", "I8",
        (spec.hidden, query),
        "diag" if attention_enabled else "zero",
    ))
    if spec.model_type in ("phi3", "glm"):
        tensors.append((
            f"{prefix}.mlp.gate_up_proj.weight", "I8",
            (2 * spec.intermediate, spec.hidden),
            "fused_gate_up" if ffn_enabled else "zero",
        ))
    else:
        tensors.extend([
            (f"{prefix}.mlp.gate_proj.weight", "I8", (spec.intermediate, spec.hidden),
             "diag" if ffn_enabled else "zero"),
            (f"{prefix}.mlp.up_proj.weight", "I8", (spec.intermediate, spec.hidden),
             "diag_alt" if ffn_enabled else "zero"),
        ])
    tensors.append((
        f"{prefix}.mlp.down_proj.weight", "I8",
        (spec.hidden, spec.intermediate),
        "diag" if ffn_enabled else "zero",
    ))
    return tensors


def write_i8(stream, shape: tuple[int, ...], pattern: str, spec: ModelSpec) -> None:
    rows, columns = shape
    block_rows = 256
    query = spec.query_heads * spec.head_dim
    kv = spec.kv_heads * spec.head_dim
    for begin in range(0, rows, block_rows):
        count = min(block_rows, rows - begin)
        block = np.zeros((count, columns), dtype=np.int8)
        absolute = np.arange(begin, begin + count, dtype=np.int64)
        active = np.ones(count, dtype=bool)
        local_pattern = pattern
        if pattern in ("fused_qkv", "fused_qkv_nonzero"):
            active = absolute >= query + kv if pattern == "fused_qkv" else active
            absolute = np.where(absolute < query, absolute,
                np.where(absolute < query + kv, absolute - query,
                         absolute - query - kv))
            local_pattern = "diag"
        elif pattern == "fused_gate_up":
            local_pattern = "diag_alt"
        if local_pattern.startswith("diag"):
            selected = np.nonzero(active)[0]
            columns_index = np.mod(absolute[active], columns)
            values = np.ones(selected.size, dtype=np.int8)
            if local_pattern == "diag_alt":
                values = np.where(absolute[active] & 1, -1, 1).astype(np.int8)
            block[selected, columns_index] = values
        stream.write(block.tobytes())


def write_safetensors(
    path: Path,
    tensors: list[tuple[str, str, tuple[int, ...], str]],
    spec: ModelSpec,
) -> None:
    item_sizes = {"I8": 1, "F32": 4}
    header: dict[str, object] = {}
    offset = 0
    for tensor_name, dtype, shape, _ in tensors:
        size = int(np.prod(shape)) * item_sizes[dtype]
        header[tensor_name] = {
            "dtype": dtype, "shape": list(shape),
            "data_offsets": [offset, offset + size],
        }
        offset += size
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with path.open("wb") as stream:
        stream.write(struct.pack("<Q", len(encoded)))
        stream.write(encoded)
        for _, dtype, shape, pattern in tensors:
            if dtype == "I8":
                write_i8(stream, shape, pattern, spec)
            elif pattern == "ones":
                stream.write(np.ones(shape, dtype="<f4").tobytes())
            elif pattern == "embedding":
                index = np.arange(np.prod(shape), dtype=np.int32).reshape(shape)
                value = (((index * 13 + 7) % 31) - 15).astype(np.float32) / 128.0
                stream.write(np.asarray(value, dtype="<f4").tobytes())
            else:
                raise ValueError(f"unsupported tensor pattern {pattern}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(SPECS), required=True)
    parser.add_argument(
        "--component", choices=("zero", "attention", "ffn", "both"),
        default="both",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--qk-pattern", choices=("zero", "nonzero"), default="zero")
    args = parser.parse_args()
    spec = SPECS[args.model]
    config, prefix, embedding = model_config(args.model, spec)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    tensors = tensor_layout(args.model, spec, prefix, embedding, args.component, args.qk_pattern)
    write_safetensors(args.output_dir / "model.safetensors", tensors, spec)
    manifest = {
        "model": args.model,
        "purpose": "deterministic full-shape compiler/CModel smoke test",
        "component": args.component,
        "qk_pattern": args.qk_pattern,
        "attention_pattern": f"{args.qk_pattern} Q/K with sparse V/O",
        "ffn_pattern": "sparse Gate/Up/Down",
        "minicpm_zero_outputs": False,
        "tensor_count": len(tensors),
    }
    (args.output_dir / "smoke_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(
        f"wrote {args.model} smoke checkpoint: "
        f"{args.output_dir / 'model.safetensors'} "
        f"({(args.output_dir / 'model.safetensors').stat().st_size} bytes)"
    )


if __name__ == "__main__":
    main()
