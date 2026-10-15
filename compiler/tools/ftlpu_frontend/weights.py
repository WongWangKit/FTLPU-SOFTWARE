"""Official Safetensors reads plus HF shard routing, without a format parser."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class SafeTensorFile:
    def __init__(self, path: Path) -> None:
        from safetensors import safe_open

        self.path = Path(path)
        # Only inspect metadata here. Do not read checkpoint payloads twice.
        with safe_open(str(self.path), framework="np") as source:
            self.header = {
                name: {"dtype": source.get_slice(name).get_dtype(),
                       "shape": source.get_slice(name).get_shape()}
                for name in source.keys()
            }

    def read(self, name: str) -> np.ndarray:
        from safetensors import safe_open

        dtype = self.header[name]["dtype"]
        if dtype not in ("BF16", "F16", "F32", "I8"):
            raise ValueError(f"unsupported safetensors dtype {dtype}")
        if dtype == "BF16":
            # Register numpy's bfloat16 dtype for safe_open; no PyTorch needed.
            import ml_dtypes  # noqa: F401
        with safe_open(str(self.path), framework="np") as source:
            value = source.get_tensor(name)
        # Preserve the old importer's BF16 bit expansion, including NaN bits.
        if dtype == "BF16":
            words = value.view(np.uint16).astype(np.uint32) << 16
            return words.view(np.float32)
        return value.astype(np.float32)


class SafeTensorStore:
    def __init__(self, model_dir: Path, *, variant: str | None = None) -> None:
        model_dir = Path(model_dir).resolve()
        suffix = f".{variant}" if variant else ""
        index_path = model_dir / f"model.safetensors.index{suffix}.json"
        if index_path.exists():
            self.weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        else:
            path = model_dir / f"model{suffix}.safetensors"
            if not path.exists():
                raise FileNotFoundError(f"{model_dir} has no model.safetensors checkpoint")
            self.weight_map = {name: path.name for name in SafeTensorFile(path).header}
        self.readers = {}
        for filename in set(self.weight_map.values()):
            path = (model_dir / filename).resolve()
            if not path.is_relative_to(model_dir):
                raise ValueError(f"shard path escapes model directory: {filename}")
            self.readers[filename] = SafeTensorFile(path)

    def read(self, name: str) -> np.ndarray:
        filename = self.weight_map.get(name)
        if filename is None:
            raise KeyError(f"checkpoint tensor is missing: {name}")
        return self.readers[filename].read(name)

    def metadata(self) -> dict:
        return {name: self.readers[filename].header[name] | {"shard": filename}
                for name, filename in self.weight_map.items()}

    def contains(self, name: str) -> bool:
        return name in self.weight_map
