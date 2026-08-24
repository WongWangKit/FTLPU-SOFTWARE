#!/usr/bin/env python3
"""Checks alternating-bank assembly without requiring an HF checkpoint."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path


def load_tool(path: Path):
    spec = importlib.util.spec_from_file_location("build_hf_decoder_stack", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load build_hf_decoder_stack.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    tool_path = Path(__file__).parents[1] / "tools" / "build_hf_decoder_stack.py"
    tool = load_tool(tool_path)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        model = root / "model"
        output_dir = root / "output"
        model.mkdir()
        output_dir.mkdir()
        (model / "config.json").write_text(
            json.dumps({"num_hidden_layers": 2}), encoding="utf-8"
        )
        metadata = {
            "model": "Qwen2.5-1.5B",
            "architecture": "Qwen2ForCausalLM",
            "hidden_size": 1536,
            "intermediate_size": 8960,
            "seq_len": 128,
            "query_heads": 12,
            "kv_heads": 2,
            "head_dim": 128,
            "layer": 0,
        }
        for layer in range(2):
            layer_dir = output_dir / f"layer{layer}_seq128"
            layer_dir.mkdir()
            current = dict(metadata)
            current["layer"] = layer
            (layer_dir / "metadata.json").write_text(
                json.dumps(current), encoding="utf-8"
            )
            (layer_dir / "input.bf16.bin").touch()
            (layer_dir / "golden.bf16.bin").touch()

        recorded: list[tuple[str, list[str]]] = []

        def fake_run_phase(name: str, command: list[str]) -> None:
            recorded.append((name, command))
            if name == "build model package":
                Path(command[command.index("--output") + 1]).touch()
            elif name == "pack alternating C2C weight pages":
                Path(command[command.index("--output") + 1]).touch()

        tool.run_phase = fake_run_phase
        output = output_dir / "qwen_two_layer.paged.ftlpum"
        old_argv = sys.argv
        try:
            sys.argv = [
                str(tool_path),
                "--model-dir", str(model),
                "--opt", str(root / "ftlpu_opt"),
                "--translate", str(root / "ftlpu_translate"),
                "--stablehlo", str(root / "decoder.stablehlo.mlir"),
                "--target-config", str(root / "target.json"),
                "--pack-model-weights", str(root / "pack_weights"),
                "--c2c-weight-paging",
                "--mxm-execution", "vector",
                "--layer-count", "2",
                "--seq-len", "128",
                "--reuse-golden",
                "--output-dir", str(output_dir),
                "--output", str(output),
            ]
            tool.main()
        finally:
            sys.argv = old_argv

        lower_commands = [
            command for name, command in recorded
            if name.startswith("lower executable variant")
            and name.endswith("to Schedule IR")
        ]
        if len(lower_commands) != 2:
            raise RuntimeError("expected two bank-specific lowerings")
        for bank, command in enumerate(lower_commands):
            position = command.index("--weight-bank")
            if command[position + 1] != str(bank):
                raise RuntimeError("decoder executable bank order is wrong")
            policy = command.index("--mxm-execution")
            if command[policy + 1] != "vector":
                raise RuntimeError("decoder executable did not use Vector MXM")

        package = next(
            command for name, command in recorded
            if name == "build model package"
        )
        executable_args = [
            package[index + 1] for index, value in enumerate(package[:-1])
            if value == "--executable"
        ]
        if len(executable_args) != 2:
            raise RuntimeError("two layers did not receive two executables")
        if (
            "variant0" not in executable_args[0]
            or "variant1" not in executable_args[1]
        ):
            raise RuntimeError("layer executable order does not alternate banks")
        if not output.is_file():
            raise RuntimeError("paged output was not produced")

    print("build_hf_decoder_stack_paging_test passed")


if __name__ == "__main__":
    main()
