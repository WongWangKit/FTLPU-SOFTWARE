#!/usr/bin/env python3
"""Builds an FTLPU ModelPackage for one or more decoder-layer golden tests."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path


I8 = 1
F16 = 3
RAW = 0
SYMMETRIC_PER_TENSOR_I8 = 1


def scalar(stream, code: str, value: object) -> None:
    stream.write(struct.pack("<" + code, value))


def string(stream, value: str) -> None:
    data = value.encode("utf-8")
    scalar(stream, "I", len(data))
    stream.write(data)


def vector(stream, code: str, values: list[object]) -> None:
    scalar(stream, "I", len(values))
    for value in values:
        scalar(stream, code, value)


def tensor(
    stream,
    name: str,
    element_type: int,
    shape: list[int],
    data: bytes,
    encoding: int = RAW,
    scales: list[float] | None = None,
) -> None:
    string(stream, name)
    scalar(stream, "H", element_type)
    scalar(stream, "H", encoding)
    scalar(stream, "i", -1)
    scalar(stream, "I", 0)
    vector(stream, "Q", shape)
    vector(stream, "f", scales or [])
    scalar(stream, "Q", len(data))
    stream.write(data)


def binding_refs(stream, refs: list[tuple[int, str]]) -> None:
    scalar(stream, "I", len(refs))
    for index, name in refs:
        scalar(stream, "I", index)
        string(stream, name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--golden-dir", type=Path, action="append", required=True
    )
    parser.add_argument(
        "--executable", type=Path, action="append", required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    golden_dirs = args.golden_dir
    executables = args.executable
    if len(executables) != len(golden_dirs):
        raise ValueError(
            "one layer-specialized executable is required per golden directory"
        )
    metadata_list = [
        json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        for path in golden_dirs
    ]
    metadata = metadata_list[0]
    for current in metadata_list[1:]:
        for key in (
            "model", "architecture", "hidden_size", "intermediate_size",
            "seq_len", "query_heads", "kv_heads", "head_dim",
        ):
            if current[key] != metadata[key]:
                raise ValueError(f"decoder layer metadata mismatch for {key}")
    hidden = metadata["hidden_size"]
    intermediate = metadata["intermediate_size"]
    seq_len = metadata["seq_len"]
    shapes = {
        "query": [hidden, hidden],
        "key": [hidden, metadata["kv_heads"] * metadata["head_dim"]],
        "value": [hidden, metadata["kv_heads"] * metadata["head_dim"]],
        "output": [hidden, hidden],
        "gate": [hidden, intermediate],
        "up": [hidden, intermediate],
        "down": [intermediate, hidden],
    }
    weight_order = ["query", "key", "value", "output", "gate", "up", "down"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as stream:
        stream.write(b"FTLPUM01")
        scalar(stream, "I", 1)
        string(stream, metadata["model"])
        string(stream, metadata["architecture"])

        # Constants include source/golden tensors so the C++ test is wholly
        # reproducible from one package.
        scalar(stream, "I", 2 + 9 * len(golden_dirs))
        tensor(
            stream, "golden.input", F16, [seq_len, hidden],
            (golden_dirs[0] / "input.f16.bin").read_bytes()
        )
        for golden_dir, layer_metadata in zip(golden_dirs, metadata_list):
            layer = int(layer_metadata["layer"])
            tensor(
                stream, f"layers.{layer}.input_layernorm.weight",
                F16, [hidden],
                (golden_dir / "input_layernorm.f16.bin").read_bytes()
            )
            for role in weight_order[:4]:
                tensor(
                    stream, f"layers.{layer}.{role}.weight",
                    I8, shapes[role],
                    (golden_dir / f"{role}.i8.bin").read_bytes(),
                    SYMMETRIC_PER_TENSOR_I8,
                    [layer_metadata["scales"][role]],
                )
            tensor(
                stream, f"layers.{layer}.post_attention_layernorm.weight",
                F16, [hidden],
                (golden_dir / "post_attention_layernorm.f16.bin").read_bytes()
            )
            for role in weight_order[4:]:
                tensor(
                    stream, f"layers.{layer}.{role}.weight",
                    I8, shapes[role],
                    (golden_dir / f"{role}.i8.bin").read_bytes(),
                    SYMMETRIC_PER_TENSOR_I8,
                    [layer_metadata["scales"][role]],
                )
        tensor(
            stream, "golden.output", F16, [seq_len, hidden],
            (golden_dirs[-1] / "golden.f16.bin").read_bytes()
        )

        scalar(stream, "I", len(golden_dirs) + 1)
        for index in range(len(golden_dirs) + 1):
            flags = (1 if index == 0 else 0) | (
                2 if index == len(golden_dirs) else 0
            )
            name = f"hidden.{index}"
            string(stream, name)
            scalar(stream, "H", F16)
            scalar(stream, "H", flags)
            vector(stream, "Q", [seq_len, hidden])

        scalar(stream, "I", len(executables))
        for executable_path, layer_metadata in zip(
            executables, metadata_list
        ):
            string(
                stream,
                f"decoder_layer_{int(layer_metadata['layer'])}_seq128",
            )
            executable = executable_path.read_bytes()
            scalar(stream, "Q", len(executable))
            stream.write(executable)

        scalar(stream, "I", len(golden_dirs))
        for invocation_index, layer_metadata in enumerate(metadata_list):
            layer = int(layer_metadata["layer"])
            string(stream, f"layers.{layer}")
            scalar(stream, "I", invocation_index)
            binding_refs(stream, [
                (0, f"hidden.{invocation_index}"),
                (1, f"layers.{layer}.input_layernorm.weight"),
                (2, f"layers.{layer}.query.weight"),
                (3, f"layers.{layer}.key.weight"),
                (4, f"layers.{layer}.value.weight"),
                (5, f"layers.{layer}.output.weight"),
                (6, f"layers.{layer}.post_attention_layernorm.weight"),
                (7, f"layers.{layer}.gate.weight"),
                (8, f"layers.{layer}.up.weight"),
                (9, f"layers.{layer}.down.weight"),
            ])
            binding_refs(
                stream, [(0, f"hidden.{invocation_index + 1}")]
            )

    print(f"wrote {args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
