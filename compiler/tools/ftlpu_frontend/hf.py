"""HF loader and thin entries; model mathematics stay in Transformers."""

from __future__ import annotations

import importlib
from pathlib import Path

import torch

from .weights import SafeTensorStore


class LogitsOnly(torch.nn.Module):
    """Same logits-only positional interface as the original SmolLM2 tool."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask,
                          use_cache=False, return_dict=False)[0]


def submodule_entry(model, options):
    """Select a real HF submodule for an explicitly scoped tensor entry."""
    return model.get_submodule(options["module_path"])


def load_model(config):
    import torch
    import transformers

    if config["quantization"] == "legacy_w8a16_bf16":
        # Keep this CPU-only validation profile reproducible and avoid thread
        # oversubscription while the Windows compiler/CModel is also running.
        torch.set_num_threads(int(config["entry_options"].get("cpu_threads", 2)))

    loader_name = config["loader"]
    if not isinstance(loader_name, str) or not loader_name.startswith("AutoModel"):
        raise ValueError("loader must name a Transformers AutoModel class")
    loader = getattr(transformers, loader_name)
    model = loader.from_pretrained(
        config["model"], revision=config["revision"],
        torch_dtype=getattr(torch, config["load_dtype"]), **config["load_options"])
    model.eval()
    if getattr(model, "is_quantized", False):
        raise ValueError("HF loader produced a quantized model; synchronized quantized export is unsupported")
    return model


def validate_inputs(inputs, config):
    import torch
    inputs = tuple(inputs)
    if len(inputs) != len(config["inputs"]):
        raise ValueError("provided entry inputs differ from configured arity")
    for value, spec in zip(inputs, config["inputs"]):
        if not isinstance(value, torch.Tensor) or list(value.shape) != spec["shape"] or value.dtype != getattr(torch, spec["dtype"]):
            raise ValueError(f"provided entry input differs from configured shape/dtype: {spec['name']}")
        if value.device.type != "cpu":
            raise ValueError("provided capture input must be on CPU")
    return inputs


def prepare_entry(model, config, inputs_override=None):
    import torch

    if config["entry"] == "causal_lm_logits":
        model.config.use_cache = False
        wrapped = LogitsOnly(model).eval()
    else:
        module, name = config["entry"].split(":", 1)
        wrapped = getattr(importlib.import_module(module), name)(model, config["entry_options"])
        if not isinstance(wrapped, torch.nn.Module):
            raise TypeError("wrapper factory must return a torch.nn.Module")
        wrapped.eval()
    if inputs_override is not None:
        inputs = validate_inputs(inputs_override, config)
    elif config["quantization"] == "legacy_w8a16_bf16":
        from .profiles import initialize_decoder_inputs
        if not hasattr(wrapped, "profile"):
            raise ValueError("quantized profile wrapper must expose its actual transformations")
        inputs = initialize_decoder_inputs(model, wrapped, config)
    else:
        inputs = tuple(torch.full(spec["shape"], spec["fill"], dtype=getattr(torch, spec["dtype"]))
                       for spec in config["inputs"])
    return wrapped, inputs


def checkpoint_metadata(model, config):
    # Use the loader's resolved snapshot, including its actual commit. A local
    # non-Safetensors source is explicitly unknown rather than inferred from
    # loaded tensors (which have already undergone dtype conversion).
    from transformers.utils.hub import cached_file

    if config["load_options"].get("use_safetensors") is False:
        return {"status": "unknown", "reason": "HF loading explicitly selects a non-Safetensors checkpoint"}
    root = Path(config["model"])
    if root.is_dir():
        root = root / config["load_options"].get("subfolder", "")
    else:
        resolved = cached_file(config["model"], "config.json",
                               revision=getattr(model.config, "_commit_hash", None) or config["revision"],
                               local_files_only=True,
                               subfolder=config["load_options"].get("subfolder", ""),
                               cache_dir=config["load_options"].get("cache_dir"))
        root = Path(resolved).parent
    variant = config["load_options"].get("variant")
    suffix = f".{variant}" if variant else ""
    if not any((root / filename).exists() for filename in
               (f"model{suffix}.safetensors", f"model.safetensors.index{suffix}.json")):
        return {"status": "unknown", "reason": "resolved checkpoint is not Safetensors"}
    return {"status": "observed", "format": "safetensors",
            "tensors": SafeTensorStore(root, variant=variant).metadata()}
