# Qwen2.5-1.5B decoder layer

This directory contains the standard-StableHLO, sequence-length 128 decoder
layer used by the target-independent FTLPU lowering test. Its dimensions match
Qwen2.5-1.5B: hidden size 1536, intermediate size 8960, 12 query heads, 2 KV
heads, head dimension 128, RoPE theta 1,000,000, and RMSNorm epsilon 1e-6.

The compiler path is generic. The model name appears only in this fixture and
its test. Qwen's Q/K/V projection biases are imported and used by the host
golden generator; lowering those bias adds into the LPU schedule is tracked
separately from the standard attention/FFN graph exercised here.

The default CTest
`qwen2_5_1_5b_decoder_layer_stablehlo_to_stream_ir_test` quickly checks the
StableHLO, Kernel, Tensor, and Stream IRs. Build the heavyweight executable
pipeline explicitly:

`qwen2_5_1_5b_paged_weight_layout_test` additionally validates the Tensor and
Stream physical layout for dual-bank paging. Use `--weight-bank 0` and
`--weight-bank 1` to generate alternating-bank executables.

```powershell
cmake --build build-ftlpu-vs2026 --config Release `
  --target qwen2_5_1_5b_decoder_layer_pipeline
```

It emits `decoder_layer.schedule.mlir`, `decoder_layer.command.mlir`, and
`decoder_layer.ftlpu`, while checking the physical 1024-row Vector and 128-row
Block8 MXM accumulator limits. CModel numerical execution is currently blocked
by the latest INT8 IW boundary-register visibility rule, so successful binary
emission is not yet a real-weight Qwen golden result.
Fully expanding the complete layer as textual Command MLIR currently takes
over ten minutes. The paged layout and 45 MiB layer page are covered, but the
deployment compiler should next emit binary directly from compressed Schedule.
