#!/usr/bin/env python3
"""Imports one dense Qwen3.5 Gated DeltaNet layer and writes LPU fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from import_hf_decoder_layer import (
    SafeTensorStore,
    bf16,
    quantize_linear,
    write_bf16,
)
from qwen3_5_reference import (
    Qwen35DeltaConfig,
    Qwen35DeltaWeights,
    linear_decoder_layer,
)


LINEAR_TENSORS = {
    "in_proj_qkv": "linear_attn.in_proj_qkv.weight",
    "in_proj_z": "linear_attn.in_proj_z.weight",
    "in_proj_b": "linear_attn.in_proj_b.weight",
    "in_proj_a": "linear_attn.in_proj_a.weight",
    "out_proj": "linear_attn.out_proj.weight",
    "gate_proj": "mlp.gate_proj.weight",
    "up_proj": "mlp.up_proj.weight",
    "down_proj": "mlp.down_proj.weight",
}


def text_config(root: dict[str, object]) -> dict[str, object]:
    if root.get("model_type") != "qwen3_5":
        raise ValueError("import_hf_qwen3_5_layer requires model_type='qwen3_5'")
    nested = root.get("text_config")
    if not isinstance(nested, dict):
        raise ValueError("Qwen3.5 config is missing text_config")
    if nested.get("model_type") != "qwen3_5_text":
        raise ValueError("Qwen3.5 text_config has an unexpected model_type")
    return nested


def load_linear_layer(
    store: SafeTensorStore,
    prefix: str,
    config: Qwen35DeltaConfig,
) -> tuple[Qwen35DeltaWeights, dict[str, np.ndarray], dict[str, float], dict[str, str]]:
    source: dict[str, np.ndarray] = {}
    packed: dict[str, np.ndarray] = {}
    scales: dict[str, float] = {}
    names: dict[str, str] = {}
    for role, suffix in LINEAR_TENSORS.items():
        name = f"{prefix}.{suffix}"
        source[role] = store.read(name)
        packed[role], scales[role] = quantize_linear(source[role])
        # The numerical fixture represents the existing W8A16 projection path.
        source[role] = packed[role].T.astype(np.float32) * np.float32(scales[role])
        names[role] = name

    conv_name = f"{prefix}.linear_attn.conv1d.weight"
    conv = store.read(conv_name)
    if conv.ndim == 3 and conv.shape[1] == 1:
        conv = conv[:, 0, :]
    names["conv1d"] = conv_name
    scalar_names = {
        "dt_bias": f"{prefix}.linear_attn.dt_bias",
        "a_log": f"{prefix}.linear_attn.A_log",
        "delta_norm": f"{prefix}.linear_attn.norm.weight",
        "input_norm": f"{prefix}.input_layernorm.weight",
        "post_attention_norm": f"{prefix}.post_attention_layernorm.weight",
    }
    scalars = {role: store.read(name) for role, name in scalar_names.items()}
    names.update(scalar_names)
    weights = Qwen35DeltaWeights(
        in_proj_qkv=source["in_proj_qkv"],
        in_proj_z=source["in_proj_z"],
        in_proj_b=source["in_proj_b"],
        in_proj_a=source["in_proj_a"],
        conv1d=conv,
        dt_bias=scalars["dt_bias"],
        a_log=scalars["a_log"],
        delta_norm=scalars["delta_norm"],
        out_proj=source["out_proj"],
        input_norm=scalars["input_norm"],
        post_attention_norm=scalars["post_attention_norm"],
        gate_proj=source["gate_proj"],
        up_proj=source["up_proj"],
        down_proj=source["down_proj"],
    )
    weights.validate(config)
    return weights, packed, scales, names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=32)
    args = parser.parse_args()

    provenance_path = args.model_dir / "provenance.json"
    provenance = None
    if provenance_path.exists():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        observed = hashlib.sha256((args.model_dir / "model.safetensors").read_bytes()).hexdigest()
        if observed != provenance["subset_sha256"]:
            raise ValueError("checkpoint subset SHA256 does not match recorded provenance")

    root = json.loads((args.model_dir / "config.json").read_text(encoding="utf-8"))
    text = text_config(root)
    layer_types = text.get("layer_types")
    if not isinstance(layer_types, list) or args.layer >= len(layer_types):
        raise ValueError("Qwen3.5 config is missing layer_types for the requested layer")
    layer_type = str(layer_types[args.layer])
    if layer_type != "linear_attention":
        raise ValueError(
            "this importer path targets a Qwen3.5 Gated DeltaNet layer; "
            f"layer {args.layer} is {layer_type!r}"
        )
    if args.seq_len <= 0:
        raise ValueError("seq_len must be positive")

    config = Qwen35DeltaConfig.from_hf(text)
    store = SafeTensorStore(args.model_dir)
    prefix = f"model.language_model.layers.{args.layer}"
    weights, packed, scales, names = load_linear_layer(store, prefix, config)
    embedding_name = "model.language_model.embed_tokens.weight"
    embedding = store.read(embedding_name)
    token_ids = np.arange(args.seq_len, dtype=np.int64) % embedding.shape[0]
    activation = bf16(embedding[token_ids])
    stages: dict[str, np.ndarray] = {}
    golden, state = linear_decoder_layer(
        activation, weights, config, stages=stages
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_bf16(args.output_dir / "input.bf16.bin", activation)
    write_bf16(args.output_dir / "golden.bf16.bin", golden)
    for stage, value in stages.items():
        if stage == "recurrent_state":
            np.asarray(value, dtype="<f4").tofile(
                args.output_dir / "golden.recurrent_state.f32.bin"
            )
        else:
            write_bf16(args.output_dir / f"golden.{stage}.bf16.bin", value)
    for role, value in packed.items():
        value.tofile(args.output_dir / f"{role}.i8.bin")
    for role in ("conv1d", "delta_norm", "input_norm", "post_attention_norm"):
        write_bf16(args.output_dir / f"{role}.bf16.bin", getattr(weights, role))
    # Transformers deliberately evaluates the decay parameters in FP32; BF16
    # A_log can overflow to -inf after exp/softplus composition.
    for role in ("dt_bias", "a_log"):
        np.asarray(getattr(weights, role), dtype="<f4").tofile(
            args.output_dir / f"{role}.f32.bin"
        )
    np.asarray([scales[role] for role in LINEAR_TENSORS], dtype="<f4").tofile(
        args.output_dir / "quant_scales.f32.bin"
    )
    token_ids.astype("<i8").tofile(args.output_dir / "token_ids.i64.bin")

    metadata = {
        "model": args.model_dir.name,
        "architecture": root.get("architectures", ["Qwen3_5ForConditionalGeneration"])[0],
        "model_type": "qwen3_5",
        "text_model_type": text["model_type"],
        "layer": args.layer,
        "layer_type": layer_type,
        "seq_len": args.seq_len,
        "hidden_size": config.hidden_size,
        "intermediate_size": config.intermediate_size,
        "linear_num_key_heads": config.num_key_heads,
        "linear_num_value_heads": config.num_value_heads,
        "linear_key_head_dim": config.key_head_dim,
        "linear_value_head_dim": config.value_head_dim,
        "linear_conv_kernel_dim": config.conv_kernel_size,
        "rms_norm_eps": config.rms_norm_eps,
        "conv_width": config.conv_width,
        "conv_state_shape": list(state.conv.shape),
        "conv_state_bytes_bf16": int(state.conv.size * 2),
        "recurrent_state_shape": list(state.recurrent.shape),
        "recurrent_state_bytes_fp32": int(state.recurrent.nbytes),
        "projection_format": "w8a16_per_tensor",
        "scales": scales,
        "source_tensors": names,
        "embedding_tensor": embedding_name,
        "projection_storage_order": "in_out",
    }
    if provenance is not None:
        metadata["checkpoint_provenance"] = provenance
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
