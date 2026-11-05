# Chunkwise output O

The incremental Qwen3.5-0.8B test now appends `O = H D`, where
`H = tril((Q K^T) * R)` and `(I + L) D = beta V`. This is the output
for a zero incoming State. Q already includes the key-dimension scale.
Outgoing State, nonzero incoming-State contributions, gated output norm,
output projection and FFN are not included in this milestone.

The logical graph uses the generic `head_matmul` operator. C++ owns memory
planning, scheduling, DDR/C2C pages and raw macro ICU lowering. H and D retain
their native alternating-head, East-first partition layouts. The common
planner accepts matching identity edges from both operands and places O on
the opposite hemisphere; it does not insert a full H/D layout conversion.

H, D and O are stored as FP32. By default the MXM directly reads H/D byte
planes 2 and 3 as truncated BF16 and performs one product per 32x32 output
tile, with FP32 accumulation and output. There is no separate cast pass,
residual splitting, VXM arithmetic, or component scratch. SXM transposes each
D weight tile directly into MXM's 1vector weight load, without a MEM spill.
Two MXM weight buffers alternate, loading tile n+1 while computing tile n.
After startup, all output tiles and heads compute consecutively at one row
vector per cycle. Pipeline startup and final drain remain.

SXM reads East streams 16/17 and writes East streams 18/19 at the MXM boundary;
the stream commit adds one cycle. Separate output IDs avoid a collision with
the following tile's source stream passing through SXM. MXM reads activations
from 4/5 and emits FP32 output on 24..27. Existing passive SR transport returns
the result to the opposite hemisphere's final O home. No VXM bypass is issued.

MEM emits direct three-dimensional domains over rows, value blocks and heads.
SXM emits one repeated transpose domain per hemisphere. MXM load and compute
each use one domain, with hardware `toggle_dim1` selecting the weight buffer.
There is no interleaving of separate instructions within a physical ICU queue.
Removing scratch also makes input/output hemisphere requirements explicit in
the planner, rather than relying on a scratch allocation to force both sides.

The generic `head_matmul` operation accepts `mxm_precision = "bf16"` or
`"fp32-components"`. Omitting the attribute retains the previous compensated
implementation. Only O selects the direct BF16 mode in this incremental test;
the forward solve and other operators retain their existing precision.
The compensated implementation splits each input into three BF16 components
and accumulates six partial products in FP32, with component/residual scratch.

## Reproduce

```powershell
python compiler/tests/qwen3_5_prefill_qkv_wide_sxm_test.py `
  --package results/qwen3_5_0_8b_decoder_layer_prefill/package `
  --build-dir build-ftlpu-vs2026-direct `
  --target-config ../FTLPU-CMODEL/config/ftlpu-lpu32.json `
  --output-dir results/qwen3_5_0_8b_prefill_chunkwise_o `
  --sxm-architecture wide --with-o --o-mxm-precision bf16 --solve-strategy forward
```

This preserves the FP32 forward substitution used by the preceding H test.
The graph has seq32, 16 heads and value dimension 128; O is `[16,32,128]`.
`actual.12.bin` contains O in native grouped-head order, even heads followed
by odd heads. The test unpacks this ordering for numerical checks.

Validation compares the product with FP64 multiplication of actual device H/D,
then compares it with the real-weight end-to-end reference. Direct BF16 mode
also checks against an explicit truncation reference and requires no runtime
VXM events in O. An independent
sequential State recurrence checks the chunk algebra. Runtime checks require
both MXMs to execute the selected number of products, retain native producer homes,
and introduce no linker MEM transports. Pipeline and per-ICU CSVs are exported
from actual runtime programs, including C2C synchronization.

## Compensated baseline (2026-11-03)

The real-weight raw macro test passed. O executes in `[272115,314233)`,
42,118 cycles. Each MXM performs 6,144 compute cycles, approximately 14.59%
of the complete O stage. Component preparation, transpose and serialized
load/compute scheduling still dominate; this is not a gap-free schedule.

- Worst-head product NRMSE against actual FP32 H/D: `5.47735e-8`.
- Worst-head end-to-end NRMSE: `0.0158163` (1.582%).
- Minimum end-to-end cosine similarity: `0.999879846`.
- Chunk-reference versus independent sequential recurrence NRMSE: `1.08317e-7`.
- Logical execution: 240,425 cycles; 5 weight pages, 6,422,528 prefetch bytes.
- Host model operations: 0; source layout copies: 0.

The standalone generic inverse/matmul regression also passed with NRMSE
`3.50808e-8` and relative residual `3.62478e-8`. Existing H and D tests and
their runtime scheduling checks are run as part of the O test.

Use `--o-mxm-precision fp32-components` to reproduce this baseline.

## Direct BF16 serialized baseline (2026-11-03)

Before SXM-to-MXM streaming, real weights, raw macro execution and numerical
checks passed with a MEM-backed transpose and serialized tiles.

| Measurement | FP32 components | Direct BF16 |
| --- | ---: | ---: |
| O stage cycles | 42,118 | 8,262 |
| MXM compute cycles per hemisphere | 6,144 | 1,024 |
| Products per output tile | 6 | 1 |
| Worst-head error relative to actual FP32 H/D | 5.47735e-8 | 0.65770% |
| Worst-head end-to-end error | 1.58163% | 1.46458% |
| Component scratch | Yes | No |

The O interval is `[272147,280409)`. Its speedup is 5.10x; logical execution
through O is 206,601 cycles. There are zero VXM runtime event rows in this
interval. NRMSE against explicit BF16 truncation followed by FP64 multiplication
is `4.13566e-8`. Source-layout transport count remains zero.

The slightly smaller end-to-end error is specific to this fixture and does not
mean truncation improves precision generally. Compute occupies only 12.39% of
the full stage: weight transpose, load and drain scheduling still leave gaps.
No claim of continuous MXM compute is made for O.

## SXM-to-MXM streaming validation (2026-11-03)

The updated real-weight seq32 raw macro run passed, including all prefix,
forward-solve, H and O numerical checks. The result directory contains the
new runtime pipeline and actual per-ICU programs.

- O runtime interval: `[272147,273274)`, **1,127 cycles** (previously 8,262).
- Both MXMs compute continuously in `[272221,273245)`: **1,024 active cycles,
  zero idle cycles between tiles or heads**, 90.86% of the full stage.
- O stage speedup over the serialized BF16 schedule: **7.33x**.
- Logical execution through O: 199,466 cycles.
- O has zero VXM events and zero transpose MEM spill. Its 8,192 MEM write
  vector issues are exactly the four byte planes of the final FP32 output.
- Numerical errors are unchanged: worst-head end-to-end NRMSE 1.46458%,
  truncation error 0.65770%, hardware-reference NRMSE `4.13566e-8`.

`head_matmul_streaming_runtime_test.py` independently checks random asymmetric
H2/D32 and H4/D128 operands, numerical results, continuous compute and exact
MEM write counts. Both cases passed, as did the previous compensated
inverse/matmul regression. The full real-weight test invokes the same runtime
audit for H16/D128. No CModel hardware changes are needed.
