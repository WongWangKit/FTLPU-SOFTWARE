# SmolLM2-135M Quantized LM Head

## Scope

The test uses the real Hugging Face `SmolLM2-135M` checkpoint and its tied
`model.embed_tokens.weight` tensor as the LM-head weight. It covers the full
model dimensions:

- hidden size: 576
- vocabulary size: 49152
- activation batch: 32 rows
- weight storage: symmetric per-tensor INT8
- compute/output format: BF16

The lowering is not model-specific. Standard StableHLO expresses an INT8
weight conversion, scalar dequantization, and `dot_general`. The compiler
recognizes this as a generic W8A16 linear projection.

## Lowering

The pipeline is:

```text
StableHLO dot_general
  -> ftlpu.kernel.matmul (M/N/K and rhs_scale)
  -> ftlpu.tensor.projection_task kind="linear"
  -> ftlpu.stream.projection_task
  -> exact-cycle Schedule IR
  -> Command IR
  -> .ftlpu binary
```

Tensor IR assigns two BF16 activation slices, eight striped INT8 weight
slices across both hemispheres, and four BF16 result slices. Stream IR records
the MEM-to-MXM activation path, MEM-to-VXM-to-MXM weight-dequant path, and
MXM-accumulator-to-MEM result path.

Each `32x576 @ 576x4096` executable performs 18 K partial accumulations.
Weights are read as eight INT8 streams, dequantized and cast to BF16 by VXM,
loaded into one MXM per hemisphere, accumulated in MXM SRAM, then cast and
written as BF16.

## Vocabulary Sharding

The full `576x49152` weight is split into 12 contiguous 4096-column shards.
All shards use the same executable and one global quantization scale. Runtime
reloads the ICU program, binds one weight shard, executes the CModel, and
concatenates the 12 result shards. This keeps Schedule IR compact while still
executing every vocabulary column on the LPU.

## Real-Checkpoint Result

The checked-in test generator reads:

```text
.cache/hf/SmolLM2-135M/model.safetensors
tensor: model.embed_tokens.weight
```

Measured full-vocabulary result:

- 12 CModel executions
- 1,572,864 BF16 logits checked
- 998,388 total cycles, including 64 drain cycles per shard
- cosine similarity: 1.0
- mean absolute error: approximately `1.0e-7`
- maximum absolute error: `0.03125`
- row 31 top-1: token 32 for both CModel and golden
- nonzero logits: 1,572,864

## Run

Build and run the complete real-weight pipeline:

```powershell
cmake --build build-ftlpu-vs2026 --config Release --target smollm2_135m_lm_head_pipeline
```

The generated artifacts are under:

```text
build-ftlpu-vs2026/compiler/ftlpu_lower/smollm2_135m_lm_head/
  data/
  ir/stablehlo/
  ir/kernel/
  ir/tensor/
  ir/stream/
  ir/schedule/
  ir/command/
  smollm2_135m_lm_head_shard.ftlpu
  cmodel_result.txt
```

For a one-shard CModel smoke test, pass `--shard-count 1` to
`compiler/tests/smollm2_lm_head_binary_runtime_test.py`.

## Combined Full-Prefill Test

`smollm2_full_prefill_lm_head_e2e_test` replaces the package's host LM head
at test time and connects the actual CModel output of the 30 decoder layers
and final RMSNorm to the quantized LPU LM head. The final token is staged in
row 31 of one zero-padded 32-row hardware tile. All 12 vocabulary shards then
run through binary loading, runtime binding, ICU queues, and CModel execution.

The test checks three boundaries:

- LPU final RMSNorm against the imported decoder golden
- LPU LM-head logits against an exact BF16-dequantized INT8 golden
- final 49152 logits and top-5 against the original tied BF16 model weight

Run it after generating the full-prefill package and LM-head artifacts:

```powershell
python compiler/tests/smollm2_full_prefill_lm_head_e2e_test.py `
  --runtime-test build-ftlpu-vs2026/runtime/Release/smollm2_full_prefill_lm_head_e2e_test.exe `
  --package build-ftlpu-vs2026/full_prefill_tail/smollm2_135m_seq128_tail_bf16.ftlpum `
  --lm-head-binary build-ftlpu-vs2026/ftlpu_lower/smollm2_135m_lm_head/smollm2_135m_lm_head_shard.ftlpu `
  --weight build-ftlpu-vs2026/ftlpu_lower/smollm2_135m_lm_head/data/lm_head_weight.i8 `
  --metadata build-ftlpu-vs2026/ftlpu_lower/smollm2_135m_lm_head/data/metadata.json `
  --output build-ftlpu-vs2026/full_prefill_tail/lpu_lm_head_e2e_result.txt
```
