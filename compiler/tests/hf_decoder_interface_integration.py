#!/usr/bin/env python3
"""Export a deterministic second HF interface, without downloading a checkpoint.

This is an actual Transformers Qwen2 implementation with a local Safetensors
checkpoint and nonzero Q/K/V biases. It is an integration fixture, not evidence
about the quality or all configurations of a pretrained Qwen2 model.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from ftlpu_frontend import export_model
    from ftlpu_frontend.bundle import write_json

    torch.set_num_threads(2)
    torch.manual_seed(20261014)
    architecture = Qwen2Config(vocab_size=256, hidden_size=576,
        intermediate_size=1536, num_hidden_layers=1, num_attention_heads=9,
        num_key_value_heads=3, rope_theta=100000.0, tie_word_embeddings=False)
    architecture._attn_implementation = "eager"
    model = Qwen2ForCausalLM(architecture).eval()
    with torch.no_grad():
        for path in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"):
            bias = model.model.layers[0].get_submodule(path).bias
            bias.copy_(torch.linspace(-0.02, 0.02, bias.numel()))
    checkpoint = root / "checkpoint"
    model.save_pretrained(checkpoint, safe_serialization=True)
    config = {
        "schema_version": 1, "model": str(checkpoint), "revision": None,
        "load_dtype": "float32", "load_options": {
            "attn_implementation": "eager", "local_files_only": True},
        "entry": "ftlpu_frontend.profiles:decoder_entry",
        "entry_options": {"layer": 0, "sequence_length": 128, "cpu_threads": 2},
        "inputs": [{"name": "hidden_states", "shape": [128, 576],
                    "dtype": "bfloat16", "fill": 0}],
        "quantization": "legacy_w8a16_bf16",
        "verification": "legacy_bf16_checkpoint", "exporter": "torch_xla",
    }
    write_json(root / "export_config.json", config)
    write_json(root / "fixture_source.json", {
        "kind": "deterministic_random_hf_qwen2", "pretrained": False,
        "seed": 20261014, "nonzero_qkv_bias": True,
        "entry_wrapper": config["entry"], "extra_model_specific_code": False})
    # Deliberately exercise the normal HF loader from the saved checkpoint.
    # The frontend subsequently reuses loaded state; it does not reread weights.
    export_model(config, root / "bundle", verify_numerics=True)
    print(json.dumps({"fixture": str(root), "status": "exported"}))


if __name__ == "__main__":
    main()
