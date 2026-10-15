"""Thin real-HF final norm and existing host embedding/LM-head boundaries."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np


def final_norm_entry(model, options):
    """Capture the real HF norm; keep Torch optional for Windows validation."""
    import torch
    from torch import nn

    class FinalNormEntry(nn.Module):
        def __init__(self):
            super().__init__()
            if set(options) - {"norm_path"}:
                raise ValueError(f"unsupported final norm options: {sorted(set(options) - {'norm_path'})}")
            path = options.get("norm_path", "model.norm")
            original = model.get_submodule(path)
            if not isinstance(getattr(original, "weight", None), torch.Tensor):
                raise ValueError("final norm must expose a real HF weight tensor")
            self.hidden_size = int(model.config.hidden_size)
            if tuple(original.weight.shape) != (self.hidden_size,):
                raise ValueError("final norm weight shape differs from the HF hidden size")
            self.norm = copy.deepcopy(original).to(torch.bfloat16).eval()
            self.frontend_hf_names = {"norm.weight": f"{path}.weight"}
            self.frontend_transformations = []
            if original.weight.dtype != torch.bfloat16:
                self.frontend_transformations.append({
                    "stage": "explicit_profile", "operation": "dtype_cast",
                    "target": "norm.weight", "hf_name": f"{path}.weight",
                    "from": str(original.weight.dtype).removeprefix("torch."),
                    "to": "bfloat16",
                })
            self.frontend_backend_profile = "legacy_w8a16_bf16"

        def forward(self, hidden_states):
            if hidden_states.ndim != 2 or hidden_states.shape[1] != self.hidden_size:
                raise ValueError("final norm requires rank-2 [sequence, hidden_size] input")
            if hidden_states.dtype != torch.bfloat16:
                raise ValueError("final norm boundary requires BF16 input")
            return self.norm(hidden_states)

    return FinalNormEntry()


def _hf_names(model, value):
    return [name for name, parameter in model.named_parameters(remove_duplicate=False)
            if parameter is value]


def _save_bf16(value, output, filename, hf_names):
    """Save and verify explicit BF16 bits without modifying the loaded tensor."""
    import torch
    converted = value.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
    bits = converted.view(torch.uint16).numpy().astype("<u2", copy=False)
    path = output / filename
    bits.tofile(path)
    saved = np.fromfile(path, dtype="<u2")
    if saved.size != bits.size or not np.array_equal(saved, bits.reshape(-1)):
        raise ValueError(f"BF16 storage bits changed while saving {filename}")
    return {
        "file": filename, "shape": list(value.shape), "dtype": "bfloat16",
        "source_hf_names": hf_names,
        "source_load_dtype": str(value.dtype).removeprefix("torch."),
        "encoding": "raw_bf16_little_endian", "bit_roundtrip_verified": True,
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def save_model_boundaries(model, output, token_ids):
    """Save loaded tied weights once for the existing host execution interfaces.

    This checks Tensor identity, not merely equal values or aliased storage.
    BF16 conversion is explicit and checked at the saved-bit boundary; no
    checkpoint reader or independent normalization mathematics is involved.
    """
    import torch

    embedding_module = model.get_input_embeddings()
    lm_head_module = model.get_output_embeddings()
    if embedding_module is None or lm_head_module is None:
        raise ValueError("model must expose input embeddings and an LM head")
    embedding = embedding_module.weight
    if lm_head_module.weight is not embedding:
        raise ValueError("host boundary requires genuinely tied embedding/LM-head Tensor objects")
    if embedding.ndim != 2 or not embedding.is_floating_point():
        raise ValueError("host embedding weight must be a rank-2 floating tensor")
    vocab, hidden = map(int, embedding.shape)
    if hidden != int(model.config.hidden_size) or vocab != int(model.config.vocab_size):
        raise ValueError("embedding weight shape differs from the HF model configuration")
    norm = model.get_submodule("model.norm").weight
    if tuple(norm.shape) != (hidden,) or not norm.is_floating_point():
        raise ValueError("final norm weight shape/dtype differs from the HF model configuration")

    values = (token_ids.detach().cpu().numpy() if isinstance(token_ids, torch.Tensor)
              else np.asarray(token_ids))
    if values.ndim != 1 or values.size == 0 or values.dtype.kind not in "iu":
        raise ValueError("token_ids must be a nonempty rank-1 integer sequence")
    if np.any(values < 0) or np.any(values >= vocab) or np.any(values > np.iinfo(np.int32).max):
        raise ValueError("token_ids exceed the embedding vocabulary or I32 range")
    tokens = values.astype("<i4")
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("refusing to overwrite a nonempty model boundary directory")
    output.mkdir(parents=True, exist_ok=True)

    embedding_names = _hf_names(model, embedding)
    norm_names = _hf_names(model, norm)
    embedding_entry = _save_bf16(embedding, output, "embed_tokens.bf16.bin", embedding_names)
    norm_entry = _save_bf16(norm, output, "final_norm.bf16.bin", norm_names)
    token_path = output / "token_ids.i32.bin"
    tokens.tofile(token_path)
    if not np.array_equal(np.fromfile(token_path, dtype="<i4"), tokens):
        raise ValueError("I32 token ids changed while saving model boundaries")
    transformations = []
    for entry in (embedding_entry, norm_entry):
        if entry["source_load_dtype"] != "bfloat16":
            transformations.append({
                "stage": "explicit_boundary", "operation": "dtype_cast",
                "source_hf_names": entry["source_hf_names"],
                "from": entry["source_load_dtype"], "to": "bfloat16",
                "output_file": entry["file"], "rounding": "PyTorch BF16 cast",
            })
    if values.dtype != np.dtype("int32"):
        transformations.append({
            "stage": "explicit_boundary", "operation": "checked_integer_cast",
            "target": "token_ids", "from": str(values.dtype), "to": "int32",
        })
    sequence = int(tokens.size)
    index = {
        "schema_version": 1,
        "model_config": json.loads(json.dumps(model.config.to_dict())),
        "model_source": {"name_or_path": getattr(model.config, "_name_or_path", ""),
                         "resolved_commit": getattr(model.config, "_commit_hash", None)},
        "embedding": embedding_entry, "final_norm": norm_entry,
        "token_ids": {"file": token_path.name, "shape": [sequence], "dtype": "int32",
                      "values": tokens.tolist(), "bytes": token_path.stat().st_size,
                      "sha256": hashlib.sha256(token_path.read_bytes()).hexdigest()},
        "tied_weights": {"tensor_identity_verified": True,
                         "source_hf_names": embedding_names,
                         "shared_file": embedding_entry["file"]},
        "inputs": [{"name": "token_ids", "shape": [sequence], "dtype": "int32"}],
        "outputs": [{"name": "final_hidden", "shape": [sequence, hidden], "dtype": "bfloat16"},
                    {"name": "logits", "shape": [sequence, vocab], "dtype": "float32"}],
        "lm_head": {"weight_file": embedding_entry["file"], "last_token_only": False,
                    "output_shape": [sequence, vocab], "output_dtype": "float32",
                    "execution": "existing ModelSession host LM head"},
        "transformations": transformations, "checkpoint_reads": 0,
    }
    (output / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    return index


def _boundary_file(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"missing boundary file or path outside directory: {relative}")
    return path


def validate_model_boundaries(output, final_bundle=None, final_manifest=None):
    """Verify saved host ABI and optional captured norm gamma, without Torch."""
    root = Path(output)
    index = json.loads((root / "index.json").read_text(encoding="utf-8"))
    if index.get("schema_version") != 1:
        raise ValueError("unsupported model boundary schema")
    hidden = index["model_config"]["hidden_size"]
    vocab = index["model_config"]["vocab_size"]
    shape = index["token_ids"]["shape"]
    if (type(hidden) is not int or hidden <= 0 or type(vocab) is not int or vocab <= 0
            or len(shape) != 1 or type(shape[0]) is not int or shape[0] <= 0):
        raise ValueError("invalid model boundary dimensions")
    sequence = shape[0]
    for name, expected_shape, dtype, width in (
            ("embedding", [vocab, hidden], "bfloat16", 2),
            ("final_norm", [hidden], "bfloat16", 2),
            ("token_ids", [sequence], "int32", 4)):
        item = index[name]
        if item["shape"] != expected_shape or item["dtype"] != dtype:
            raise ValueError(f"model boundary shape/dtype differs: {name}")
        path = _boundary_file(root, item["file"])
        size = int(np.prod(expected_shape, dtype=np.int64)) * width
        if item["bytes"] != size or path.stat().st_size != size:
            raise ValueError(f"model boundary byte count differs: {name}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != item["sha256"]:
            raise ValueError(f"model boundary checksum differs: {name}")
    tokens = np.fromfile(_boundary_file(root, index["token_ids"]["file"]), dtype="<i4")
    if (np.any(tokens < 0) or np.any(tokens >= vocab)
            or tokens.tolist() != index["token_ids"]["values"]):
        raise ValueError("saved boundary token ids differ or exceed vocabulary")
    tied = index["tied_weights"]
    if (tied.get("tensor_identity_verified") is not True
            or tied.get("shared_file") != index["embedding"]["file"]
            or index["lm_head"]["weight_file"] != index["embedding"]["file"]
            or tied.get("source_hf_names") != index["embedding"]["source_hf_names"]):
        raise ValueError("saved embedding/LM-head tied weight contract differs")
    if (index["inputs"] != [{"name": "token_ids", "shape": [sequence], "dtype": "int32"}]
            or index["outputs"] != [
                {"name": "final_hidden", "shape": [sequence, hidden], "dtype": "bfloat16"},
                {"name": "logits", "shape": [sequence, vocab], "dtype": "float32"}]
            or index["lm_head"]["last_token_only"] is not False
            or index["lm_head"]["output_shape"] != [sequence, vocab]
            or index["lm_head"]["output_dtype"] != "float32"):
        raise ValueError("saved host boundary input/full-output ABI differs")
    if (final_bundle is None) != (final_manifest is None):
        raise ValueError("captured final norm requires both bundle and manifest")
    if final_manifest is not None:
        functions = final_manifest["functions"]
        if len(functions) != 1:
            raise ValueError("captured final norm must contain one function")
        arguments = functions[0]["arguments"]
        gamma = [arg for arg in arguments if arg["source"].get("type_") == "parameter"]
        activations = [arg for arg in arguments if arg["source"].get("type_") == "input_arg"]
        if len(arguments) != 2 or len(gamma) != 1 or len(activations) != 1:
            raise ValueError("captured final norm requires one actual gamma and activation")
        if (gamma[0]["shape"] != [hidden] or gamma[0]["dtype"] != "bfloat16"
                or activations[0]["shape"] != [sequence, hidden]
                or activations[0]["dtype"] != "bfloat16"):
            raise ValueError("captured final norm ABI differs from host boundaries")
        tensors = [item for item in final_manifest["tensors"]
                   if item["name"] == gamma[0]["source"]["name"]]
        if (len(tensors) != 1 or tensors[0]["shape"] != [hidden]
                or tensors[0]["dtype"] != "bfloat16"
                or set(tensors[0]["hf_names"]) != set(index["final_norm"]["source_hf_names"])):
            raise ValueError("captured gamma logical mapping differs from host final norm")
        array = np.load(_boundary_file(Path(final_bundle), tensors[0]["file"]), allow_pickle=False)
        if (list(array.shape) != [hidden]
                or not (str(array.dtype) == "bfloat16"
                        or array.dtype.kind == "V" and array.dtype.itemsize == 2)):
            raise ValueError("captured gamma is not the declared BF16 tensor")
        saved = np.fromfile(_boundary_file(root, index["final_norm"]["file"]), dtype="<u2")
        if not np.array_equal(array.view("<u2").reshape(-1), saved):
            raise ValueError("captured final norm gamma bits differ from packaged boundary weight")
    return index
