# SmolLM2-135M to StableHLO

This exporter captures one statically shaped, logits-only forward pass of
`HuggingFaceTB/SmolLM2-135M`. It deliberately excludes tokenization, sampling,
the autoregressive loop, and the KV cache. Those are runtime concerns and
should not be part of the first compiler bring-up graph.

PyTorch/XLA is best supported on Linux. On Windows, run this from WSL2 or a
Linux container:

```bash
python -m venv .venv-stablehlo
source .venv-stablehlo/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torch-xla transformers safetensors

python tools/export_smollm2_stablehlo.py \
  --sequence-length 16 \
  --output artifacts/smollm2-135m-stablehlo
```

The first run downloads the Apache-2.0-licensed model from Hugging Face. To
require an already cached/local model, add `--local-files-only`; `--model` can
also be a local directory.

The main compiler input is:

```text
artifacts/smollm2-135m-stablehlo/functions/forward.mlir
```

The output directory also contains portable bytecode, argument metadata, and
externalized parameter tensors. The graph has fixed input shapes
`1x16xi64` by default. Change `--batch-size` or `--sequence-length` and export a
separate artifact for another shape.

Before feeding the graph to a downstream compiler, verify that the vendored
StableHLO parser accepts it:

```bash
stablehlo-opt \
  artifacts/smollm2-135m-stablehlo/functions/forward.mlir \
  -o /dev/null
```

For initial bring-up, keep the default `float32`. Once the graph compiles
end-to-end, `--dtype bfloat16` gives an artifact closer to the model's native
weight format.
