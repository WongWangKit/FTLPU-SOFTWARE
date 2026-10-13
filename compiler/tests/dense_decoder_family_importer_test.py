#!/usr/bin/env python3
"""Checks every supported dense decoder checkpoint layout without downloads."""

from __future__ import annotations

import json
import math
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


HIDDEN = 16
HEADS = 2
KV_HEADS = 1
HEAD_DIM = 8
INTERMEDIATE = 24
VOCAB = 64


def values(shape: tuple[int, ...], seed: int) -> np.ndarray:
    index = np.arange(np.prod(shape), dtype=np.int32).reshape(shape)
    return (((index * (seed * 2 + 1) + seed) % 29) - 14).astype(
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


def base_config(model_type: str, architecture: str) -> dict[str, object]:
    return {
        "model_type": model_type,
        "architectures": [architecture],
        "hidden_size": HIDDEN,
        "intermediate_size": INTERMEDIATE,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "num_hidden_layers": 4,
        "head_dim": HEAD_DIM,
        "hidden_act": "silu",
        "attention_bias": False,
        "rope_theta": 10_000.0,
        "rms_norm_eps": 1.0e-6,
        "max_position_embeddings": 128,
    }


def family_config(family: str) -> tuple[dict[str, object], str, str]:
    if family == "llama3":
        config = base_config("llama", "LlamaForCausalLM")
        config["rope_scaling"] = {
            "rope_type": "llama3",
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
        }
        return config, "model.layers.0", "model.embed_tokens.weight"
    if family == "ministral3":
        text = base_config("ministral3", "Ministral3ForCausalLM")
        text.pop("rope_theta")
        text["rope_parameters"] = {
            "rope_type": "yarn",
            "rope_theta": 1_000_000.0,
            "factor": 16.0,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
            "original_max_position_embeddings": 16_384,
        }
        return {
            "model_type": "mistral3",
            "architectures": ["Mistral3ForConditionalGeneration"],
            "text_config": text,
        }, "language_model.model.layers.0", "language_model.model.embed_tokens.weight"
    if family == "phi4mini":
        config = base_config("phi3", "Phi3ForCausalLM")
        config["partial_rotary_factor"] = 0.75
        config["rope_scaling"] = {
            "type": "longrope",
            "short_factor": [1.0, 1.0, 1.0],
            "long_factor": [2.0, 2.0, 2.0],
        }
        config["original_max_position_embeddings"] = 64
        config["max_position_embeddings"] = 128
        return config, "model.layers.0", "model.embed_tokens.weight"
    if family == "smollm3":
        config = base_config("smollm3", "SmolLM3ForCausalLM")
        config["no_rope_layers"] = [1, 1, 1, 0]
        config["layer_types"] = ["full_attention"] * 4
        return config, "model.layers.0", "model.embed_tokens.weight"
    if family == "minicpm4":
        config = base_config("minicpm", "MiniCPMForCausalLM")
        # The real MiniCPM4 JSON omits model_type and relies on auto_map.
        config.pop("model_type")
        config["scale_emb"] = 12.0
        config["scale_depth"] = 1.4
        config["rope_scaling"] = {
            "rope_type": "longrope",
            "short_factor": [1.0, 1.1, 1.2, 1.3],
            "long_factor": [2.0, 2.1, 2.2, 2.3],
            "original_max_position_embeddings": 64,
        }
        return config, "model.layers.0", "model.embed_tokens.weight"
    if family == "glmedge":
        config = base_config("glm", "GlmForCausalLM")
        config["partial_rotary_factor"] = 1.0
        return config, "model.layers.0", "model.embed_tokens.weight"
    raise ValueError(family)


def checkpoint_tensors(family: str, prefix: str, embedding: str) -> dict[str, np.ndarray]:
    query_width = HEADS * HEAD_DIM
    kv_width = KV_HEADS * HEAD_DIM
    tensors = {
        embedding: values((VOCAB, HIDDEN), 1),
        f"{prefix}.input_layernorm.weight": values((HIDDEN,), 2) + 1.0,
        f"{prefix}.post_attention_layernorm.weight": values((HIDDEN,), 3) + 1.0,
        f"{prefix}.self_attn.o_proj.weight": values((HIDDEN, query_width), 7),
        f"{prefix}.mlp.down_proj.weight": values((HIDDEN, INTERMEDIATE), 12),
    }
    if family == "phi4mini":
        tensors[f"{prefix}.self_attn.qkv_proj.weight"] = values(
            (query_width + 2 * kv_width, HIDDEN), 4
        )
    else:
        tensors.update({
            f"{prefix}.self_attn.q_proj.weight": values((query_width, HIDDEN), 4),
            f"{prefix}.self_attn.k_proj.weight": values((kv_width, HIDDEN), 5),
            f"{prefix}.self_attn.v_proj.weight": values((kv_width, HIDDEN), 6),
        })
    if family in ("phi4mini", "glmedge"):
        tensors[f"{prefix}.mlp.gate_up_proj.weight"] = values(
            (2 * INTERMEDIATE, HIDDEN), 9
        )
    else:
        tensors.update({
            f"{prefix}.mlp.gate_proj.weight": values((INTERMEDIATE, HIDDEN), 9),
            f"{prefix}.mlp.up_proj.weight": values((INTERMEDIATE, HIDDEN), 10),
        })
    return tensors


def run_family(tool: Path, root: Path, family: str) -> None:
    config, prefix, embedding = family_config(family)
    model = root / family
    output = root / f"{family}.golden"
    model.mkdir()
    (model / "config.json").write_text(json.dumps(config), encoding="utf-8")
    write_safetensors(
        model / "model.safetensors",
        checkpoint_tensors(family, prefix, embedding),
    )
    subprocess.run(
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
    expected_type = "minicpm" if family == "minicpm4" else {
        "llama3": "llama",
        "ministral3": "ministral3",
        "phi4mini": "phi3",
        "smollm3": "smollm3",
        "glmedge": "glm",
    }[family]
    if metadata["model_type"] != expected_type:
        raise AssertionError(f"{family} model type was not recognized")
    if metadata["embedding_tensor"] != embedding:
        raise AssertionError(f"{family} embedding prefix was not recognized")
    if metadata["attention_bias_operands"]:
        raise AssertionError(f"{family} unexpectedly exposes attention bias operands")
    if metadata["head_dim"] != HEAD_DIM or metadata["query_width"] != 16:
        raise AssertionError(f"{family} attention geometry changed")
    if family == "phi4mini":
        if metadata["rotary_dim"] != 6:
            raise AssertionError("Phi-4 mini partial RoPE was ignored")
        sources = metadata["source_tensors"]
        if len({sources[role] for role in ("query", "key", "value")}) != 1:
            raise AssertionError("Phi-4 mini fused QKV was not split")
    if family == "minicpm4":
        expected_scale = 1.4 / math.sqrt(4.0)
        if not math.isclose(metadata["residual_scale"], expected_scale):
            raise AssertionError("MiniCPM4 residual scaling was ignored")
        if metadata["embedding_scale"] != 12.0:
            raise AssertionError("MiniCPM4 embedding scaling was ignored")
    if family == "smollm3" and not metadata["uses_rope"]:
        raise AssertionError("SmolLM3 layer-0 RoPE map was inverted")
    expected_sizes = {
        "query.i8.bin": HIDDEN * HEADS * HEAD_DIM,
        "key.i8.bin": HIDDEN * KV_HEADS * HEAD_DIM,
        "gate.i8.bin": HIDDEN * INTERMEDIATE,
        "golden.bf16.bin": 32 * HIDDEN * 2,
    }
    for name, expected in expected_sizes.items():
        observed = (output / name).stat().st_size
        if observed != expected:
            raise AssertionError(
                f"{family} {name}: {observed} bytes, expected {expected}"
            )
    golden = np.fromfile(output / "golden.bf16.bin", dtype="<u2")
    if not np.any(golden & np.uint16(0x7FFF)):
        raise AssertionError(f"{family} produced an empty golden")


def main() -> None:
    tool = Path(__file__).parents[1] / "tools" / "import_hf_decoder_layer.py"
    families = (
        "llama3",
        "ministral3",
        "phi4mini",
        "smollm3",
        "minicpm4",
        "glmedge",
    )
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        for family in families:
            run_family(tool, root, family)
    print("dense decoder family importer passed: " + ", ".join(families))


if __name__ == "__main__":
    main()
