"""JSON configuration for HF loading and a statically shaped tensor entry."""

from __future__ import annotations

import copy

DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M"
DEFAULT_INPUTS = [
    {"name": "input_ids", "shape": [1, 16], "dtype": "int64", "fill": 0},
    {"name": "attention_mask", "shape": [1, 16], "dtype": "int64", "fill": 1},
]


def effective_config(config: dict) -> dict:
    defaults = {
        "schema_version": 1,
        "model": DEFAULT_MODEL,
        "revision": None,
        "loader": "AutoModelForCausalLM",
        "load_dtype": "float32",
        "load_options": {"attn_implementation": "eager", "local_files_only": False},
        "inputs": DEFAULT_INPUTS,
        "entry": "causal_lm_logits",
        "entry_options": {},
        "exporter": "torch_xla",
        "strict": False,
        "quantization": "none",
        "verification": "strict_torch_close",
    }
    unknown = config.keys() - defaults.keys()
    if unknown:
        raise ValueError(f"unknown configuration fields: {sorted(unknown)}")
    result = copy.deepcopy(defaults)
    result.update(copy.deepcopy(config))
    result["load_options"] = defaults["load_options"] | result["load_options"]
    if result["schema_version"] != 1:
        raise ValueError("unsupported configuration schema_version")
    if not isinstance(result["model"], str) or not result["model"]:
        raise ValueError("model must be a HF model ID or local directory")
    if result["load_dtype"] not in ("float32", "float16", "bfloat16"):
        raise ValueError("load_dtype must be float32, float16 or bfloat16")
    if not isinstance(result["strict"], bool):
        raise ValueError("strict must be boolean")
    if not isinstance(result["entry_options"], dict):
        raise ValueError("entry_options must be an object")
    # A weight-only change cannot express W8A16 graph semantics. Do not accept
    # HF bitsandbytes/etc. loading as a substitute for a graph transformation.
    if result["quantization"] not in ("none", "legacy_w8a16_bf16") or any(
        key in result["load_options"] for key in
        ("quantization_config", "load_in_8bit", "load_in_4bit")
    ):
        raise ValueError("quantized export is not implemented: graph, tensors and mappings "
                         "must be transformed together; use quantization=none")
    if result["quantization"] == "legacy_w8a16_bf16" and result["entry"] != "ftlpu_frontend.profiles:decoder_entry":
        raise ValueError("legacy_w8a16_bf16 requires the explicit decoder_entry profile")
    if result["verification"] not in ("strict_torch_close", "legacy_bf16_checkpoint"):
        raise ValueError("unsupported verification criterion")
    if (result["verification"] == "legacy_bf16_checkpoint"
            and result["quantization"] != "legacy_w8a16_bf16"
            and result["entry"] != "ftlpu_frontend.boundaries:final_norm_entry"):
        raise ValueError("legacy BF16 criterion requires the explicit existing decoder profile")
    if any(key in result["load_options"] for key in
           ("dtype", "torch_dtype", "revision", "device_map", "state_dict")):
        raise ValueError("use revision/load_dtype fields; device_map/state_dict are "
                         "unsupported for this CPU capture path")
    inputs = result["inputs"]
    if not isinstance(inputs, list) or not inputs:
        raise ValueError("inputs must be a nonempty list of tensor specifications")
    names = []
    for spec in inputs:
        if set(spec) != {"name", "shape", "dtype", "fill"}:
            raise ValueError("each input requires name, shape, dtype and fill")
        if not isinstance(spec["name"], str) or not spec["name"].isidentifier():
            raise ValueError("input names must be Python identifiers")
        if not isinstance(spec["shape"], list) or any(
            type(dim) is not int or dim < 1 for dim in spec["shape"]
        ):
            raise ValueError("input dimensions must be positive static integers")
        if spec["dtype"] not in ("int64", "int32", "bool", "float32", "float16", "bfloat16"):
            raise ValueError(f"unsupported input dtype: {spec['dtype']}")
        if type(spec["fill"]) not in (bool, int, float):
            raise ValueError("input fill must be a scalar number")
        names.append(spec["name"])
    if len(set(names)) != len(names):
        raise ValueError("input names must be unique")
    if result["entry"] == "causal_lm_logits":
        if names != ["input_ids", "attention_mask"] or inputs[0]["shape"] != inputs[1]["shape"]:
            raise ValueError("causal_lm_logits expects matching input_ids/attention_mask")
        if len(inputs[0]["shape"]) != 2 or any(s["dtype"] != "int64" for s in inputs):
            raise ValueError("causal_lm_logits expects rank-2 int64 tensors")
        if result["entry_options"]:
            raise ValueError("causal_lm_logits has no entry_options; use a wrapper factory")
    elif not isinstance(result["entry"], str) or ":" not in result["entry"]:
        raise ValueError("entry must be causal_lm_logits or module:wrapper_factory")
    return result
