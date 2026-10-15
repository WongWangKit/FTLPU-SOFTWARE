"""Explicit existing decoder numeric profile around real HF implementation.

Only Linear storage/dequantization and boundary casts are changed here. HF owns
normalization, attention, rotary embedding, masks and activation mathematics.
The CModel's LUT/rounding reference remains the existing importer golden.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from import_hf_decoder_layer import bf16, quantize_linear

PROFILE = "legacy_w8a16_bf16"
LINEARS = {
    "query": "self_attn.q_proj", "key": "self_attn.k_proj",
    "value": "self_attn.v_proj", "output": "self_attn.o_proj",
    "gate": "mlp.gate_proj", "up": "mlp.up_proj", "down": "mlp.down_proj",
}


class ProfileLinear(nn.Module):
    """Use the old per-tensor quantizer and PyTorch's actual Linear primitive."""
    def __init__(self, linear, role):
        super().__init__()
        quantized, self.scale = quantize_linear(linear.weight.detach().float().cpu().numpy())
        self.effective_scale = float(bf16(np.array([self.scale], np.float32))[0])
        self.register_buffer("quantized_weight", torch.from_numpy(quantized.copy()))
        self.register_buffer("bias", None if linear.bias is None else
                             linear.bias.detach().to(torch.bfloat16).clone())
        self.role = role
        self.in_features, self.out_features = linear.in_features, linear.out_features

    def forward(self, value):
        shape = value.shape
        value = value.reshape(-1, self.in_features)
        # int8 [in,out] -> BF16 effective weight, then the existing profile's
        # distinct attention/FFN compute boundaries. Scale is encoded as BF16.
        weight = (self.quantized_weight.float() * self.effective_scale).to(torch.bfloat16)
        compute_dtype = torch.float32 if self.role in ("gate", "up", "down") else torch.bfloat16
        if self.role == "down":
            value = value.to(torch.bfloat16)
        result = F.linear(value.to(compute_dtype), weight.to(compute_dtype).T,
                          None if self.bias is None else self.bias.to(compute_dtype))
        if self.role in ("gate", "up"):
            result = result.to(torch.bfloat16).float()
        elif self.role == "down":
            result = result.to(torch.bfloat16)
        return result.reshape(*shape[:-1], self.out_features)


class DecoderEntry(nn.Module):
    """Static, no-cache decoder boundary; calls the original HF layer and RoPE."""
    def __init__(self, model, options):
        super().__init__()
        allowed = {"sequence_length", "layer", "position_offset", "mask", "decoder_path",
                   "rotary_path", "token_id_offset", "cpu_threads"}
        if set(options) - allowed:
            raise ValueError(f"unsupported decoder profile options: {sorted(set(options) - allowed)}")
        self.sequence = int(options["sequence_length"])
        self.layer_index = int(options.get("layer", 0))
        if self.sequence < 1 or self.sequence % 32:
            raise ValueError("legacy decoder profile requires sequence_length divisible by 32")
        if options.get("position_offset", 0) != 0 or options.get("mask", "causal") != "causal":
            raise ValueError("legacy decoder profile supports positions from zero and pure causal mask only")
        if model.config.rope_scaling is not None:
            raise ValueError("legacy decoder profile does not support RoPE scaling")
        if model.config.hidden_act not in ("silu", "swish"):
            raise ValueError("legacy decoder profile requires the HF SiLU/SwiGLU interface")
        decoder_path = options.get("decoder_path", "model.layers")
        rotary_path = options.get("rotary_path", "model.rotary_emb")
        self.config = copy.deepcopy(model.config)
        self.config._attn_implementation = "eager"
        self.layer = copy.deepcopy(model.get_submodule(decoder_path)[self.layer_index]).eval()
        self.rotary = copy.deepcopy(model.get_submodule(rotary_path)).eval()
        if getattr(self.rotary, "rope_type", "default") != "default":
            raise ValueError("legacy decoder profile requires standard unscaled HF RoPE")
        self.profile = {
            "schema_version": 1, "name": PROFILE,
            "architecture": self.config.to_dict(), "layer": self.layer_index,
            "sequence_length": self.sequence, "roles": {},
            "activation": {"input_name": "hidden_states", "file": "validation/input_0.npy"},
            "mask": "static_causal_no_padding", "position_range": [0, self.sequence - 1],
            "rope": {"theta": float(self.config.rope_theta), "scaling": None,
                     "inv_freq": self.rotary.inv_freq.detach().float().cpu().tolist(),
                     "target": "rotary.inv_freq", "inv_freq_target": "rotary.inv_freq"},
            "immutable_targets": ["rotary.inv_freq"],
            "rotary_path": rotary_path,
            "fp32_reference_file": "validation/fp32_reference.npy",
            "transformations": [],
            "execution": {"torch_cpu_threads": torch.get_num_threads()},
            "numerical_contract": {
                "framework": "real HF with explicit int8 dequantization, BF16 weights and mixed compute casts",
                "backend": "existing legacy W8A16/BF16 decoder golden; LUT and intermediate rounding apply",
                "normalization_is_bitwise_layout_only": True,
                "hf_to_backend_is_bitwise_equivalent": False,
                "boundary_policy": "embedding and final norm remain BF16 host boundaries; LM head is not quantized here",
                "tied_weights": "embedding/LM-head relationship is untouched; only selected decoder linears are quantized",
            },
        }
        for role, path in LINEARS.items():
            original = self.layer.get_submodule(path)
            if not isinstance(original, nn.Linear):
                raise ValueError(f"unsupported projection interface: {path}")
            replacement = ProfileLinear(original, role)
            parent, child = path.rsplit(".", 1)
            setattr(self.layer.get_submodule(parent), child, replacement)
            source = f"{decoder_path}.{self.layer_index}.{path}.weight"
            target = f"layer.{path}.quantized_weight"
            self.profile["roles"][role] = {
                "target": target, "hf_name": source, "weight_layout": "in_out",
                "scale": replacement.scale, "effective_scale": replacement.effective_scale,
            }
            self.profile["transformations"].append({
                "stage": "explicit_quantization", "operation": "symmetric_per_tensor_int8",
                "source": source, "target": target, "range": [-127, 127],
                "source_load_dtype": str(original.weight.dtype).removeprefix("torch."),
                "quantizer_input_dtype": "float32",
                "rounding": "numpy.rint ties-to-even", "storage_dtype": "int8",
                "scale": replacement.scale, "effective_scale": replacement.effective_scale,
                "scale_rounding": "existing BF16 round-to-nearest-even",
                "weight_layout": "in_out", "dequant_weight_dtype": "bfloat16",
                "compute_dtype": "float32" if role in ("gate", "up", "down") else "bfloat16",
                "intermediate_rounding": "BF16 gate/up/down-input boundaries" if role in ("gate", "up", "down") else "BF16 projection result",
            })
            if replacement.bias is not None:
                self.profile["roles"][f"{role}_bias"] = {
                    "target": f"layer.{path}.bias",
                    "hf_name": f"{decoder_path}.{self.layer_index}.{path}.bias",
                }
                if original.bias.dtype != torch.bfloat16:
                    self.profile["transformations"].append({
                        "stage": "explicit_profile", "operation": "dtype_cast",
                        "target": f"layer.{path}.bias",
                        "from": str(original.bias.dtype).removeprefix("torch."), "to": "bfloat16",
                    })
        for role, path in (("input_norm", "input_layernorm"),
                           ("post_attention_norm", "post_attention_layernorm")):
            norm = self.layer.get_submodule(path)
            original_dtype = str(norm.weight.dtype).removeprefix("torch.")
            norm.weight = nn.Parameter(norm.weight.detach().to(torch.bfloat16), requires_grad=False)
            self.profile["roles"][role] = {
                "target": f"layer.{path}.weight",
                "hf_name": f"{decoder_path}.{self.layer_index}.{path}.weight",
            }
            if original_dtype != "bfloat16":
                self.profile["transformations"].append({
                    "stage": "explicit_profile", "operation": "dtype_cast",
                    "target": f"layer.{path}.weight", "from": original_dtype, "to": "bfloat16",
                })

    def forward(self, hidden_states):
        from transformers.masking_utils import create_causal_mask

        value = hidden_states.unsqueeze(0)
        positions = torch.arange(self.sequence, dtype=torch.int64, device=value.device)
        position_ids = positions.unsqueeze(0)
        mask = create_causal_mask(config=self.config, input_embeds=value,
                                 attention_mask=None, cache_position=positions,
                                 past_key_values=None, position_ids=position_ids)
        embeddings = self.rotary(value, position_ids)
        result = self.layer(hidden_states=value, attention_mask=mask,
                            position_ids=position_ids, position_embeddings=embeddings,
                            cache_position=positions, use_cache=False)
        return result.squeeze(0)


def decoder_entry(model, options):
    return DecoderEntry(model, options)


def initialize_decoder_inputs(model, wrapped, config):
    """Use loaded embeddings once; do not reread the checkpoint for golden input."""
    if len(config["inputs"]) != 1 or config["inputs"][0]["name"] != "hidden_states":
        raise ValueError("decoder profile expects one hidden_states input")
    spec = config["inputs"][0]
    expected_shape = [wrapped.sequence, int(model.config.hidden_size)]
    if spec["shape"] != expected_shape or spec["dtype"] != "bfloat16":
        raise ValueError(f"decoder profile requires BF16 hidden_states {expected_shape}")
    offset = int(config["entry_options"].get("token_id_offset", 0))
    token_ids = (torch.arange(wrapped.sequence, dtype=torch.int64) + offset) % model.get_input_embeddings().num_embeddings
    with torch.inference_mode():
        value = model.get_input_embeddings()(token_ids).to(torch.bfloat16)
    wrapped.profile["activation"]["token_ids"] = token_ids.tolist()
    return (value,)
