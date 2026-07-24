#!/usr/bin/env python3
"""Export SmolLM2-135M's forward pass to a StableHLO program.

The exported graph consumes:
  input_ids:      tensor<batch_size x sequence_length x i64>
  attention_mask: tensor<batch_size x sequence_length x i64>

and returns logits. Generation and the KV cache are deliberately kept outside
the graph; this makes the artifact a simple, statically-shaped compiler input.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM


DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M"


class LogitsOnly(torch.nn.Module):
    """Give torch.export a tensor-only, positional-input interface."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=False,
        )
        return outputs[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=Path("artifacts/smollm2-135m-stablehlo"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument(
        "--dtype",
        choices=("float32", "bfloat16"),
        default="float32",
        help="float32 is the safest choice for a new downstream backend",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="do not access Hugging Face; require the model to be cached",
    )
    return parser.parse_args()


def require_empty_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise SystemExit(
            f"Refusing to overwrite non-empty output directory: {path}\n"
            "Choose another --output or remove the old artifact explicitly."
        )
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.sequence_length < 1:
        raise SystemExit("--batch-size and --sequence-length must be positive")

    require_empty_output(args.output)
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16

    print(f"Loading {args.model} ({args.dtype})...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        attn_implementation="eager",
        local_files_only=args.local_files_only,
    )
    model.eval()
    model.config.use_cache = False
    wrapped = LogitsOnly(model).eval()

    input_ids = torch.zeros(
        (args.batch_size, args.sequence_length), dtype=torch.int64
    )
    attention_mask = torch.ones_like(input_ids)

    print("Capturing the PyTorch graph...")
    with torch.inference_mode():
        exported = torch.export.export(
            wrapped, (input_ids, attention_mask), strict=False
        )

    print("Lowering the captured graph to StableHLO...")
    from torch_xla.stablehlo import exported_program_to_stablehlo

    stablehlo_program = exported_program_to_stablehlo(exported)
    stablehlo_program.save(str(args.output))

    manifest = {
        "model": args.model,
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "parameter_dtype": args.dtype,
        "inputs": {
            "input_ids": [args.batch_size, args.sequence_length],
            "attention_mask": [args.batch_size, args.sequence_length],
        },
        "output": "logits",
        "kv_cache": False,
        "entry_point": "forward",
    }
    (args.output / "export.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    mlir = args.output / "functions" / "forward.mlir"
    bytecode = args.output / "functions" / "forward.bytecode"
    print("Export complete.")
    print(f"  StableHLO text:     {mlir.resolve()}")
    print(f"  StableHLO bytecode: {bytecode.resolve()}")
    print(f"  Manifest:           {(args.output / 'export.json').resolve()}")


if __name__ == "__main__":
    main()
