# GLM-4-9B batched decode

The test covers decoder layer 0 of the original `zai-org/glm-4-9b`, with
official W8A16 weights and full dimensions (hidden 4096, FFN 13696,
32 query heads, 2 KV heads, head width 128). Each request decodes **one**
new token after an independent 32-token history. Batch is not sequence
length. This is not a complete 40-layer model or a generation benchmark.

## Run

First prepare the official checkpoint and imported fixture using the
[prefill entry](glm4_9b_prefill.md). From the SOFTWARE root:

```powershell
python -B compiler/tests/glm4_decode_test.py `
  --checkpoint models/official_dense/glm4_9b `
  --imported tmp/model_tests/glm4_9b/golden `
  --build-dir build-ftlpu-vs2026-direct `
  --target-config ../FTLPU-CMODEL/config/ftlpu-lpu32.json `
  --output-dir results/glm4_9b_decoder_layer_decode `
  --batches 1 32 64 --past-len 32
```

`--resume` reuses completed executions only when inputs and encoded program
match. A changed compiler triggers recompilation; only a byte-identical
program can reuse an earlier execution. Each batch has its own `result.json`;
`status: passed` is the evidence of numerical acceptance, not the existence
of a compiled binary or pipeline file.

When debugging a later phase, `--resume --reuse-completed-phase projections
attention` explicitly keeps the previously executed binaries for those two
phases across a compiler rebuild. Input/graph, hardware and binary hashes
must still match, and numerical checks are repeated. These phases retain
their original compiler provenance; they are not claimed as new executions.

## Device execution

There are three CModel ModelSession invocations, each using raw-macro ICU:

1. Batched input RMSNorm, shared Q/K/V projections, biases and partial RoPE
   at position 32 (first 64 features, preserving the other 64).
2. Independent KV append, grouped QK, masked softmax and PV. QK/PV use the
   generic batched activation MXM lowerer. The 33 live KV positions are
   padded to 64; positions 33–63 are masked. Softmax uses FP32 fixed-port
   ALU chains and row accumulation; layout transposes use wide SXM.
3. Batched output projection, residual, post-attention RMSNorm, shared
   SwiGLU FFN lowering and final residual.

Intermediate tensors cross DDR between these three invocations. Thus the
combined trace is a concatenation of actual executions, **not** a continuous
single ICU program or an overlapped three-stage schedule. Python constructs
mathematical graphs, fixtures and references; it does not emit FU instructions
or replace device activation operations with host arithmetic.

The compiler tiles activation GEMMs into independent mirrored matrix pairs
and limits RHS column tiles to 64 to keep each MEM ICU's domains serial.
Scalar reductions are tiled by SRAM capacity. Completed tiles provide
compiler-declared drain boundaries for finite VXM/SXM iMEM reloads via the
common barrier protocol; physical iMEM capacity checks remain enabled.
Large external KV tensors are distributed over multiple MEM byte pairs.
BF16/FP32 casts use immediate-one multiplication rather than uploading a
matrix of ones for every tile.
The FFN also tiles independent token rows to keep its live activation and
Gate/Up temporaries within SRAM (32 requests per tile for these dimensions).
Batch 64 remains one logical FFN and one device invocation containing both
tiles. Partial FFN sums use the fixed VXM input ports. RMSNorm source homes
are kept off its simultaneous SXM replay-write queues, so disjoint row
addresses cannot hide a same-ICU read/write conflict.

Historical KV is initialized from the quantized prefill reference, using
different permutations of the 32 official embedding rows for each request.
This initializes a cache fixture; it does not claim that the historical
prefill was executed on the device. Device-produced new K/V is appended in
the attention program. Historical KV must be bit-exact on readback.

## Acceptance and artifacts

- Independent random attention-core test: `compiler/tests/glm4_decode_core_test.py`.
- Odd-batch QK/PV regression: `compiler/tests/activation_batch_matmul_test.py`.
- Real-weight attention is checked before the FFN (relative L2 < 3%).
- Full decoder output is compared with the quantized reference (aggregate
  relative L2 < 5%, every request < 8%); new K/V relative L2 must be < 3%.
- FP32 accumulation does not remove BF16 operand/output rounding or the
  runtime exponential/reciprocal approximation error. The reference uses
  the imported INT8 weights; this is not a native-BF16 checkpoint quality test.

Results are under `results/glm4_9b_decoder_layer_decode/batch{1,32,64}`:

- `result.json`, `actual.bf16.bin`, `golden.bf16.bin`, `token_ids.json`.
- `runtime.pipeline.csv`: actual runtime traces with phase offsets.
- `projections/`, `attention/`, `tail/`: compiled and runtime programs,
  execution logs, individual runtime pipeline CSVs and `icu_programs/`.

The runtime trace includes initial input transfers, dynamic weight paging
and waits. Phase durations include their own DDR input/output overhead;
do not interpret the total as optimized continuous-decoder latency.

## Recorded runs (2026-11-04–05)

All entries use `past_len=32` and one new token per request. Relative L2 is
measured against the W8A16 reference. Runtime cycles come from the combined
actual trace, including each phase's transfers and waits.

| Batch | Status | Output relative L2 | Worst request relative L2 | Runtime cycles |
| --- | --- | ---: | ---: | ---: |
| 1 | Passed | 1.13885% | 1.13885% | 5,956,751 |
| 32 | Passed | 0.94251% | 2.02964% | 8,089,807 |
| 64 | Passed | 1.38834% | 6.78253% | 16,614,302 |

Historical K/V was unchanged bit-for-bit in all three runs. The first 32
outputs of batch 64 are bit-identical to batch 32, using the matching input
and cache fixtures. Batch 64's worst request has 6.78253% relative L2, below
the existing 8% per-request limit but materially larger than the aggregate
error; acceptance does not imply sub-percent accuracy for every request.
These runs retain per-phase compiled binaries and completion stamps,
including phases reused after subsequent compiler fixes, so they are not
an identical-build performance comparison.
