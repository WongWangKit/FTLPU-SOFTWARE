# Qwen2.5-1.5B decoder layer

This directory contains standard-StableHLO decoder-layer fixtures for sequence
lengths 32 and 128. Both use Qwen2.5-1.5B dimensions: hidden size 1536,
intermediate size 8960, 12 query heads, 2 KV heads, head dimension 128, RoPE
theta 1,000,000, and RMSNorm epsilon 1e-6.

The compiler path is model-independent. The fixtures use standard StableHLO
operations and lower through Kernel, Tensor, Stream, compressed Schedule, and
binary Command representations. The executable uses Block8 MXM, VXM-feedback
RMSNorm, dual-bank paged INT8 weights, and BF16 activations.

Build the sequence-length 32 executable with:

```powershell
python compiler/tests/qwen2_5_1_5b_decoder_layer_pipeline_test.py `
  --opt build-ftlpu-vs2026/compiler/ftlpu_opt.exe `
  --compile build-ftlpu-vs2026/compiler/ftlpu-compile.exe `
  --input compiler/examples/qwen2_5_1_5b_decoder_layer/decoder_layer_seq32.stablehlo.mlir `
  --target-config compiler/examples/targets/cmodel_large_sram.json `
  --weight-bank 1 `
  --output-dir build-ftlpu-vs2026/compiler/ftlpu_lower/qwen2_5_1_5b_seq32
```

The resulting binary has 501,237 compressed queue commands and a maximum
scheduled cycle of 192,382. The CModel golden test passes all 49,152 BF16
outputs with 49,127 nonzero values. Component checkpoint maximum errors are
0.03125 for FFN, 0 for the attention residual, and 0.03125 for the second
RMSNorm.

This is a shape-real decoder-layer test with deterministic sparse INT8 golden
weights. It validates the complete compiler/binary/runtime/CModel path, but it
is not yet an execution of all 28 decoder layers from the original Hugging
Face checkpoint.

## Real-checkpoint FFN

`compiler/tools/import_hf_ffn.py` imports the original layer-0 Gate, Up, and
Down weights plus an embedding-derived BF16 input from the Hugging Face
Qwen2.5-1.5B checkpoint. It applies per-tensor INT8 quantization and produces a
golden that models BF16 MXM dequantization, FP32 partial accumulation, and the
VXM FP16 LUT implementation of SwiGLU.

For `seq_len=32`, the standard-StableHLO input lowers through every compiler IR
to a version-22 paged binary. The runtime uploads 26 real weight pages, fills
the ICU queues, and runs the binary on CModel through cycle 770,646. All 49,152
BF16 outputs pass the numerical comparison: zero threshold mismatches, MAE
`6.54569e-05`, RMSE `0.000266376`, and maximum error `0.00418091`.
