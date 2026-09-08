#!/usr/bin/env python3
"""Builds an FTLPU ModelPackage for one or more decoder-layer golden tests."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path


I8 = 1
BF16 = 5
F32 = 4
I32 = 2
RAW = 0
SYMMETRIC_PER_TENSOR_I8 = 1
KV_KEY = 1
KV_VALUE = 2

# Internal binding indices form a separate namespace from model inputs.
# Keep these synchronized with attention_schedule_emitter.hpp.
ATTENTION_KEY_STATE_BINDING_INDEX = 65536
ATTENTION_VALUE_STATE_BINDING_INDEX = 65537


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
    parser.add_argument("--embedding-table-bf16", type=Path)
    parser.add_argument("--vocab-size", type=int)
    parser.add_argument("--final-norm-bf16", type=Path)
    parser.add_argument("--final-rmsnorm-executable", type=Path)
    parser.add_argument(
        "--kv-cache-capacity",
        type=int,
        default=0,
        help="emit one persistent BF16 K/V cache pair per decoder layer",
    )
    parser.add_argument(
        "--checkpoint-outputs",
        action="store_true",
        help="embed and download every decoder-layer golden output",
    )
    args = parser.parse_args()
    boundary_options = (
        args.embedding_table_bf16,
        args.vocab_size,
        args.final_norm_bf16,
        args.final_rmsnorm_executable,
    )
    include_boundaries = any(value is not None for value in boundary_options)
    if include_boundaries and not all(
        value is not None for value in boundary_options
    ):
        raise ValueError(
            "embedding table, vocab size, final norm, and final RMSNorm "
            "executable must be provided together"
        )

    golden_dirs = args.golden_dir
    executables = args.executable
    if len(executables) not in (1, len(golden_dirs)):
        raise ValueError(
            "provide one reusable executable or one specialized executable per layer"
        )
    layer_executables = (
        executables * len(golden_dirs)
        if len(executables) == 1
        else executables
    )
    unique_executables: list[Path] = []
    executable_indices: list[int] = []
    for executable in layer_executables:
        try:
            executable_index = unique_executables.index(executable)
        except ValueError:
            executable_index = len(unique_executables)
            unique_executables.append(executable)
        executable_indices.append(executable_index)
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
    if args.kv_cache_capacity < 0:
        raise ValueError("KV cache capacity must be non-negative")
    if args.kv_cache_capacity and args.kv_cache_capacity < seq_len:
        raise ValueError(
            "KV cache capacity must be no smaller than seq_len"
        )
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
    bias_operand_flags = [
        bool(current.get("attention_bias_operands", False))
        for current in metadata_list
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as stream:
        stream.write(b"FTLPUM01")
        scalar(stream, "I", 4)
        string(stream, metadata["model"])
        string(stream, metadata["architecture"])

        # Constants include source/golden tensors so the C++ test is wholly
        # reproducible from one package.
        scalar(
            stream, "I",
            2 + sum(9 + 3 * has_bias for has_bias in bias_operand_flags)
            + (len(golden_dirs) if args.checkpoint_outputs else 0)
            + (2 if include_boundaries else 0),
        )
        tensor(
            stream, "golden.input", BF16, [seq_len, hidden],
            (golden_dirs[0] / "input.bf16.bin").read_bytes()
        )
        for golden_dir, layer_metadata, has_bias in zip(
            golden_dirs, metadata_list, bias_operand_flags
        ):
            layer = int(layer_metadata["layer"])
            tensor(
                stream, f"layers.{layer}.input_layernorm.weight",
                BF16, [hidden],
                (golden_dir / "input_layernorm.bf16.bin").read_bytes()
            )
            for role in weight_order[:4]:
                tensor(
                    stream, f"layers.{layer}.{role}.weight",
                    I8, shapes[role],
                    (golden_dir / f"{role}.i8.bin").read_bytes(),
                    SYMMETRIC_PER_TENSOR_I8,
                    [layer_metadata["scales"][role]],
                )
            if has_bias:
                for role, width in (
                    ("query", hidden),
                    ("key", metadata["kv_heads"] * metadata["head_dim"]),
                    ("value", metadata["kv_heads"] * metadata["head_dim"]),
                ):
                    tensor(
                        stream, f"layers.{layer}.{role}.bias",
                        BF16, [width],
                        (golden_dir / f"{role}_bias.bf16.bin").read_bytes(),
                    )
            tensor(
                stream, f"layers.{layer}.post_attention_layernorm.weight",
                BF16, [hidden],
                (golden_dir / "post_attention_layernorm.bf16.bin").read_bytes()
            )
            for role in weight_order[4:]:
                tensor(
                    stream, f"layers.{layer}.{role}.weight",
                    I8, shapes[role],
                    (golden_dir / f"{role}.i8.bin").read_bytes(),
                    SYMMETRIC_PER_TENSOR_I8,
                    [layer_metadata["scales"][role]],
                )
        if args.checkpoint_outputs:
            for index, golden_dir in enumerate(golden_dirs, start=1):
                tensor(
                    stream, f"golden.hidden.{index}", BF16,
                    [seq_len, hidden],
                    (golden_dir / "golden.bf16.bin").read_bytes(),
                )
        tensor(
            stream, "golden.output", BF16, [seq_len, hidden],
            (golden_dirs[-1] / "golden.bf16.bin").read_bytes()
        )
        if include_boundaries:
            tensor(
                stream, "model.embed_tokens.weight", BF16,
                [args.vocab_size, hidden],
                args.embedding_table_bf16.read_bytes(),
            )
            tensor(
                stream, "model.norm.weight", BF16, [hidden],
                args.final_norm_bf16.read_bytes(),
            )

        value_count = len(golden_dirs) + 1 + (
            3 if include_boundaries else 0
        )
        scalar(stream, "I", value_count)
        if include_boundaries:
            string(stream, "token_ids")
            scalar(stream, "H", I32)
            scalar(stream, "H", 1)
            vector(stream, "Q", [seq_len])
        for index in range(len(golden_dirs) + 1):
            flags = (1 if index == 0 else 0) | (
                2 if index == len(golden_dirs)
                    and not include_boundaries else 0
            )
            if args.checkpoint_outputs and index > 0:
                flags |= 2
            if include_boundaries and index == 0:
                flags = 0
            name = f"hidden.{index}"
            string(stream, name)
            scalar(stream, "H", BF16)
            scalar(stream, "H", flags)
            vector(stream, "Q", [seq_len, hidden])
        if include_boundaries:
            string(stream, "final_hidden")
            scalar(stream, "H", BF16)
            scalar(stream, "H", 2)
            vector(stream, "Q", [seq_len, hidden])
            string(stream, "logits")
            scalar(stream, "H", F32)
            scalar(stream, "H", 2)
            vector(stream, "Q", [1, args.vocab_size])

        scalar(
            stream, "I",
            len(unique_executables) + int(include_boundaries),
        )
        for executable_index, executable_path in enumerate(
            unique_executables
        ):
            source_layer = executable_indices.index(executable_index)
            layer_metadata = metadata_list[source_layer]
            string(
                stream,
                f"decoder_layer_variant_{executable_index}_"
                f"from_layer_{int(layer_metadata['layer'])}_seq{seq_len}",
            )
            executable = executable_path.read_bytes()
            scalar(stream, "Q", len(executable))
            stream.write(executable)
        if include_boundaries:
            string(stream, f"final_rmsnorm_seq{seq_len}")
            executable = args.final_rmsnorm_executable.read_bytes()
            scalar(stream, "Q", len(executable))
            stream.write(executable)

        # Version-3 host preprocessing and postprocessing sections.
        scalar(stream, "I", int(include_boundaries))
        if include_boundaries:
            string(stream, "embedding")
            string(stream, "token_ids")
            string(stream, "model.embed_tokens.weight")
            string(stream, "hidden.0")
        scalar(stream, "I", int(include_boundaries))
        if include_boundaries:
            string(stream, "lm_head")
            string(stream, "final_hidden")
            string(stream, "model.embed_tokens.weight")
            string(stream, "logits")
            scalar(stream, "B", 1)

        # Version-4 persistent model state descriptors.
        scalar(
            stream,
            "I",
            2 * len(golden_dirs) if args.kv_cache_capacity else 0,
        )
        if args.kv_cache_capacity:
            state_shape = [
                args.kv_cache_capacity,
                metadata["kv_heads"],
                metadata["head_dim"],
            ]
            for layer_metadata in metadata_list:
                layer = int(layer_metadata["layer"])
                for suffix, kind in (
                    ("key_cache", KV_KEY),
                    ("value_cache", KV_VALUE),
                ):
                    string(stream, f"layers.{layer}.{suffix}")
                    scalar(stream, "H", kind)
                    scalar(stream, "H", BF16)
                    scalar(stream, "I", layer)
                    scalar(stream, "I", args.kv_cache_capacity)
                    vector(stream, "Q", state_shape)

        scalar(stream, "I", len(golden_dirs) + int(include_boundaries))
        for invocation_index, layer_metadata in enumerate(metadata_list):
            layer = int(layer_metadata["layer"])
            string(stream, f"layers.{layer}")
            scalar(
                stream, "I",
                executable_indices[invocation_index],
            )
            input_refs = [
                (0, f"hidden.{invocation_index}"),
                (1, f"layers.{layer}.input_layernorm.weight"),
                (2, f"layers.{layer}.query.weight"),
                (3, f"layers.{layer}.key.weight"),
                (4, f"layers.{layer}.value.weight"),
                (5, f"layers.{layer}.output.weight"),
            ]
            next_binding = 6
            if bias_operand_flags[invocation_index]:
                input_refs.extend([
                    (6, f"layers.{layer}.query.bias"),
                    (7, f"layers.{layer}.key.bias"),
                    (8, f"layers.{layer}.value.bias"),
                ])
                next_binding = 9
            input_refs.extend([
                (next_binding,
                 f"layers.{layer}.post_attention_layernorm.weight"),
                (next_binding + 1, f"layers.{layer}.gate.weight"),
                (next_binding + 2, f"layers.{layer}.up.weight"),
                (next_binding + 3, f"layers.{layer}.down.weight"),
            ])
            binding_refs(stream, input_refs)
            binding_refs(
                stream, [(0, f"hidden.{invocation_index + 1}")]
            )
            state_refs = []
            if args.kv_cache_capacity:
                state_refs = [
                    (
                        ATTENTION_KEY_STATE_BINDING_INDEX,
                        f"layers.{layer}.key_cache",
                    ),
                    (
                        ATTENTION_VALUE_STATE_BINDING_INDEX,
                        f"layers.{layer}.value_cache",
                    ),
                ]
            binding_refs(stream, state_refs)
        if include_boundaries:
            string(stream, "final_norm")
            scalar(stream, "I", len(unique_executables))
            binding_refs(stream, [
                (0, f"hidden.{len(golden_dirs)}"),
                (1, "model.norm.weight"),
            ])
            binding_refs(stream, [(0, "final_hidden")])
            binding_refs(stream, [])

    print(f"wrote {args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
