# Tiled online-softmax attention on LPU

This is an optional compiler lowering of scaled dot-product attention. It
implements the tiled online-softmax algorithm used by FlashAttention with
existing MXM, VXM, SXM and MEM instructions. It does not add a FlashAttention
hardware instruction or a model-level operation to CModel.

## Scope and interface

`ftlpu.kernel.flash_attention` accepts static BF16 tensors:

- Q: `[batch, query_tokens, query_heads, key_width]`.
- K: `[batch, kv_tokens, kv_heads, key_width]`.
- V: `[batch, kv_tokens, kv_heads, value_width]`.
- Output: `[batch, query_tokens, query_heads, value_width]`, BF16.

Query heads must be divisible by KV heads. Key and value widths must be
positive multiples of 32. GQA groups and requests are independent. The
optional causal mask allows key index `j <= query_position + i` for query
index `i`. The verifier rejects incompatible shapes. Physical SRAM/iMEM
checks still determine whether a particular shape can be compiled.

The current implementation uses resident K/V. It does **not** yet stream an
arbitrarily long cache from DDR in KV pages. Its logical-graph entry is
`compiler/tools/attention_graph.py`; tiling, masking, allocation and scheduling
are implemented in C++ `FlashAttentionLowering.inc`, dispatched by the shared
logical-graph planner. GEMMs reuse the public activation batch-matmul lowering.

## Algorithm and precision

The compiler processes 32-key tiles and up to 32 live query rows. For GQA,
query rows combine query tokens and the query heads sharing a KV head. MXM
pairs two independent KV groups across the hemispheres. The final odd group
is mirrored and its duplicate output discarded. The compiler computes
`K Q^T` directly so each score word contains the dot products of one key
with independent query rows. VXM consumes successive keys in time. For
GLM batch-1 decode, each KV head has 16 live Q heads in a 32-lane word;
padded query outputs are discarded. This is not full 32-lane utilization.

For each tile, let `S = scale * Q K^T` with its length/causal mask. Update:

```text
m_new = max(m, row_max(S))
alpha = exp(m - m_new)
P     = exp(S - m_new)
l_new = alpha * l + row_sum(P)
U_new = alpha * U + P V
output = U / l
```

The first tile initializes `m`, `l` and `U` directly. A true row maximum is
seeded with the first value rather than zero; this also handles entirely
negative logits without exponential underflow caused by a zero clamp.
Only valid keys enter max and sum. A per-query causal mask is generated
when necessary and omitted if every live key is visible. For a tile with
one live key, VXM broadcasts each query's BF16 probability over the value
channels and writes the FP32 product. A tile with 2–32 keys uses MXM for
PV. Missing probability rows are zero-padded after the online reductions.

Precision boundaries are explicit:

| Quantity | Storage / calculation |
| --- | --- |
| Q/K/V | BF16 |
| QK GEMM | BF16 operands, FP32 accumulation, BF16 output |
| Scaled and masked scores | BF16, preserving the original decode path's boundaries |
| Row maximum, exponent, denominator, rescale coefficient | FP32 |
| Unnormalized probability entering PV | BF16 conversion |
| PV | BF16 operands, FP32 product for one key or FP32 accumulation for a multi-key tile |
| Running numerator U and its rescaling | FP32 |
| Final normalized context | BF16 |

The baseline normalizes probabilities before its BF16 PV boundary; the
online algorithm rounds unnormalized probabilities and normalizes U at the
end. Thus the two algorithms are not expected to be bit-identical. Existing
attention and whole-layer error limits remain unchanged.

For the numerator update, a VXM word now carries 32 **different value
channels** of one query row. U, PV and their sum stay in packed FP32; only
the FP32 row coefficient is broadcast. A segmented or hemisphere-partitioned
MXM result may need whole-word SR copies to prepare a VXM operand. This
does not unpack the values through SXM. Multiplication and addition remain
separate operations. The final normalization writes the packed BF16 context
directly. Score, exponent and probability stay packed across query lanes.
The C1 accumulator resets once per packed column word, seeds from the
first key and updates each lane in FP32; only the final max or sum is stored.
PV uses a wide transpose into the MXM LHS layout, reusing that prepared
operand across V column tiles. Only the small row coefficients need to be
broadcast for the packed value-channel update. Intermediate stages still
write MEM; this is not a fully fused MXM–VXM streaming kernel. Compiler-declared
drains still protect finite ICU iMEM and live contexts.

## Numerical validation

Run the independent small tests (build both compiler and runtime first):

```powershell
foreach ($case in @('negative', 'gqa', 'causal', 'query_tiles', 'causal_offset', 'single_tile')) {
  python -B compiler/tests/flash_attention_test.py `
    --build-dir build-ftlpu-vs2026-direct `
    --target-config ../FTLPU-CMODEL/config/ftlpu-lpu32.json `
    --output-dir results/flash_attention_streamed/$case --case $case
}
```

They cover all-negative logits with a changing maximum, three KV blocks,
independent GQA requests, odd group counts, tail tiles and per-query causal
masks, including more than 32 grouped query rows and a KV block that is
entirely masked for only some query rows. The oracle is independent
FP64 stable attention on the BF16 inputs.
Acceptance requires finite results and relative L2 below 3%. Actual
raw-macro execution, runtime CSV and per-ICU exports are retained.

Recorded raw-macro results on 2026-11-05:

| Case | Query tokens / KV tokens | Relative L2 |
| --- | ---: | ---: |
| Negative logits, changed maximum | 1 / 33 | 0.04061% |
| Three requests with GQA | 1 / 65 | 0.22746% |
| Causal rows | 3 / 35 | 0.24556% |
| Multiple query tiles | 17 / 49 | 0.21042% |
| Entirely masked block for some rows | 3 / 35, position 31 | 0.22918% |

## GLM-4-9B decode experiment

The real-weight experiment uses official layer 0 imported as W8A16, hidden
width 4096, 32 Q heads, 2 KV heads, head width 128, and FFN width 13696.
Each independent request has 32 historical tokens and decodes one new token.
Inputs are hidden states `[B,4096]` and historical K/V `[B,32,2,128]`;
outputs are decoder hidden states `[B,4096]` and updated K/V `[B,33,2,128]`.
Historical KV is a reference-generated fixture, not a device prefill claim.

```powershell
python -B compiler/tests/glm4_decode_test.py `
  --checkpoint models/official_dense/glm4_9b `
  --imported tmp/model_tests/glm4_9b/golden `
  --build-dir build-ftlpu-vs2026-direct `
  --target-config ../FTLPU-CMODEL/config/ftlpu-lpu32.json `
  --output-dir results/glm4_9b_decoder_layer_decode_flash `
  --batches 1 32 --past-len 32 --attention-strategy flash --resume
```

The default strategy remains `baseline`. This experiment executes three
device invocations: projections, attention, and the output projection/FFN
tail. Intermediate DDR transfers are included in the combined runtime CSV;
this is **not** one continuous decoder ICU program. Every phase exports the
actual executed program and its per-ICU files. Projection artifacts may be
reused only after verifying semantic input and binary hashes; their original
compiler provenance remains in `completed.json` when recorded. The legacy
batch-1 projection stamp has only binary/input hashes, so its original
compiler hash is reported as unknown rather than assigned the current build.

Acceptance is the existing contract: attention relative L2 < 3%, whole-layer
aggregate relative L2 < 5%, every request < 8%, new K/V < 3%, unchanged
historical K/V bit-for-bit. The reference uses the same imported INT8 weights.
This does not measure native-BF16 model quality or full-model generation.

For a controlled attention-only comparison, after Flash attention completes:

```powershell
python -B compiler/tests/glm4_attention_comparison_test.py `
  --checkpoint models/official_dense/glm4_9b `
  --flash-results results/glm4_9b_decoder_layer_decode_flash `
  --build-dir build-ftlpu-vs2026-direct `
  --target-config ../FTLPU-CMODEL/config/ftlpu-lpu32.json `
  --output-dir results/glm4_9b_decoder_layer_decode_flash/baseline_same_build `
  --batches 1 32
```

This reruns baseline with the same compiler and the exact device-produced
projection outputs and cache fixture used by Flash. It rejects a different
Flash compiler hash and checks numerics and identical K/V outputs. Previous
whole-layer baseline artifacts remain unchanged.

If a shared build was rebuilt during a long simulation, use an isolated
compiler/runtime copy and `--verify-flash-recompile`. In this mode the
comparison additionally recompiles the saved Flash kernel and **requires
the resulting program to be byte-identical to the actually executed binary**.
A different binary is rejected. Both original and comparison compiler hashes
and the verification mode are retained in the result JSON.

Batch 64 was cancelled during attention simulation at the user's request;
it is excluded from this Flash acceptance run. Its incomplete artifacts are
marked `cancelled`, not passed. The previous baseline batch-64 result is
unaffected.

The target config used for this experiment specifies 500 MHz, peak DDR
51.2 GB/s and scheduling efficiency 90%. Report runtime cycles including
transfer/wait overhead, separately from static schedule cycles. The aggregate
`C2C.HostOutput` interval includes KV diagnostic readback; the trace does not
provide a separate exact cycle count for that diagnostic subset.

Generate the measured timing breakdown and comparison JSON with:

```powershell
python -B compiler/tools/summarize_glm4_flash.py `
  --flash-results results/glm4_9b_decoder_layer_decode_flash `
  --baseline-results results/glm4_9b_decoder_layer_decode `
  --target-config ../FTLPU-CMODEL/config/ftlpu-lpu32.json
```

The output is `comparison.json` in the Flash results root. Incomplete
whole-layer runs remain explicitly marked as running/failed. The static
schedule includes scheduled gaps and is not a count of active FU cycles.

Further performance work should fuse adjacent online VXM stages, keep tile
layouts directly compatible between MXM and VXM, and amortize per-block
normalization and ICU reload overhead before selecting this path by default.

## Earlier attention comparison with scalar U (2026-11-05)

These are attention-invocation timings at 500 MHz, including KV input and
context/KV readback. They do not include projections or FFN.

| Batch | Baseline cycles | Flash cycles | Baseline ms | Flash ms | Baseline / Flash relative L2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 83,444 | 192,037 | 0.166888 | 0.384074 | 2.44241% / 2.43311% |
| 32 | 2,500,667 | 6,025,610 | 5.001334 | 12.051220 | 2.58086% / 2.57840% |

Both K/V outputs match baseline bit-for-bit. Batch 1 used the same compiler
directly; batch 32 used verified byte-identical Flash recompilation after a
shared compiler rebuild. Original/comparison hashes and raw binaries are
retained under `baseline_same_build`.

At this short KV length, Flash is about 2.30x / 2.41x slower for batch 1 / 32.
This is a functional online-softmax implementation, not a demonstrated
performance improvement. The baseline remains the default.

## Earlier whole-layer acceptance with scalar U (2026-11-05)

Both requested batches passed the original thresholds; batch 64 was cancelled.
Historical K/V is unchanged bit-for-bit. Batch 1's attention output is also
bit-identical to the first request in batch 32. The projection and tail
program binaries match the corresponding original baseline binaries exactly;
the tail was executed again with the new Flash context.

| Batch | Output relative L2 | Worst request relative L2 | Actual total cycles | Total ms |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1.19750% | 1.19750% | 6,057,980 | 12.115960 |
| 32 | 0.95473% | 2.09089% | 11,614,750 | 23.229500 |

The disjoint timeline components sum to each total:

| Batch | Initial ingress cycles | Static schedule cycles | Weight-page wait cycles | Output readback cycles, including diagnostics |
| --- | ---: | ---: | ---: | ---: |
| 1 | 26,377 | 4,016,045 | 2,008,411 | 7,147 |
| 32 | 91,618 | 9,358,208 | 2,012,585 | 152,339 |

Static schedule cycles include scheduled gaps. Output transfer includes the
KV integrity diagnostic, whose separate duration is unavailable in the
aggregate trace. These totals include the DDR boundaries between the three
invocations and do not describe a continuous decoder or a full-model latency.

Each `batchN` directory contains the complete `runtime.pipeline.csv` and
`result.json`. Each of its `projections`, `attention` and `tail` directories
contains `actual.runtime.ftlpu`, `icu_programs/index.csv`, individual
`*.icu.csv` files, input data, kernel IR, compiler logs and execution logs.
The ICU files are exported from the actually executed runtime program,
including pre-execution instruction domains, rather than a static schedule.

## Earlier packed U: independent GLM-4-9B decode attention (2026-11-05)

This run reuses batch 1's actual layer-0 projection outputs, the 32-token
KV history and the same semantic graph, runner and hardware target. It does
not rerun projections, FFN, batch 32 or batch 64. To reproduce, retain the
scalar Flash and baseline artifacts above, rebuild the compiler, then run:

```powershell
python -B compiler/tests/glm4_flash_packed_u_test.py `
  --build-dir build-ftlpu-vs2026-direct `
  --target-config ../FTLPU-CMODEL/config/ftlpu-lpu32.json
```

| Actual attention cycles | Baseline | Scalar U Flash | Packed U Flash |
| --- | ---: | ---: | ---: |
| Initial ingress | 3,244 | 2,712 | 2,712 |
| Static device schedule | 74,839 | 184,471 | 93,262 |
| Weight-page waits | 0 | 0 | 0 |
| Output transfer, including KV diagnostics | 5,361 | 4,854 | 5,238 |
| **Runtime total** | **83,444** | **192,037** | **101,212** |

At 500 MHz this saves 0.181650 ms per attention invocation versus scalar U
(47.3%). The baseline still wins by 17,768 cycles. Its saved kernel was
recompiled with the packed-U compiler and produced the same binary SHA-256
(`add49f2e0e01899a0416de7f83b5ca97fd636d1153eabc9bed70b417f64cf696`).

All BF16 context, K and V words match scalar Flash exactly. Context relative
L2 against the attention oracle remains 2.43311%. In the compiler plan,
the gaps before scaling U and adding PV drop from 34,387 and 33,891 cycles
to 799 each. The terminal `packed_output` stage is gone. In the actual trace,
wide SXM transpose issues drop from 728 to 344 **per hemisphere** over the
whole attention invocation. Neither these plan gaps nor transpose spans
should be added to runtime as separate costs.

Results: `results/glm4_9b_decoder_layer_decode_flash_packed/batch1/result.json`.
The `attention/` subdirectory contains the actual `runtime.pipeline.csv`,
`actual.runtime.ftlpu`, `icu_programs/index.csv` and per-ICU CSV files. The
previous full-layer results remain intact and do not describe packed-U timing.

## Earlier query-lane softmax and shared PV input (2026-11-06)

Scope: GLM-4-9B layer-0 **attention only**, batch 1, one decoded token,
32 historical KV entries, real W8A16 projection outputs, 500 MHz / 51.2
GB/s. The Q/K/V inputs, cache fixture, semantic graph and hardware runner
are identical to the previous runs. No decoder-tail or batch-64 rerun.

```powershell
python -B compiler/tests/glm4_flash_streamed_test.py `
  --build-dir build-ftlpu-vs2026-direct `
  --target-config ../FTLPU-CMODEL/config/ftlpu-lpu32.json
```

| Actual attention cycles | Baseline | Packed U | Query-lane softmax |
| --- | ---: | ---: | ---: |
| Initial ingress | 3,244 | 2,712 | 2,384 |
| Static schedule, including gaps | 74,839 | 93,262 | 41,237 |
| Weight-page wait | 0 | 0 | 0 |
| Output transfer, including KV diagnostics | 5,361 | 5,238 | 4,845 |
| **Total cycles** | **83,444** | **101,212** | **48,466** |
| **Total ms** | **0.166888** | **0.202424** | **0.096932** |

The new version takes 52.1% fewer cycles than packed U and 41.9% fewer
than baseline (1.72x throughput relative to baseline for this invocation).
SXM transpose issues over both hemispheres fall from 688 with packed U
to 70 (34 east, 36 west). Their spans overlap other FUs; the count is not
a latency estimate. Recompiling the baseline kernel with this compiler
reproduced the original executed baseline binary byte-for-byte.

All 4,096 context elements and all returned K/V words match both earlier
Flash implementations bit-for-bit. Context relative L2 remains 2.43311%
against the independent attention oracle. All six small cases passed,
including odd KV groups, one-key tail blocks, causal offsets and multiple
query tiles. This experiment does not establish whole-layer speedup or a
performance crossover for other context lengths. Resident-KV and existing
ICU/MEM limits remain in force.

Artifacts: `results/glm4_9b_decoder_layer_decode_flash_streamed/batch1/result.json`
and `batch1/attention/{runtime.pipeline.csv,actual.runtime.ftlpu,icu_programs/}`
within the same results root. The JSON contains the per-side actual SXM
counts, per-phase timing breakdown and bitwise comparisons.

## Earlier native BF16 operands, cached Q and ALU chains (2026-11-06)

These measurements describe the preceding run; the fixed streamed results
directory now contains the later epilogue run below. Scope is
unchanged: the complete FlashAttention core and KV append for layer 0,
batch 1, one new token after 32 historical tokens. Q/K/V projections, RoPE,
O projection and the decoder tail are outside this invocation. Inputs come
from the previous real W8A16 projection execution; the core uses BF16 Q/K/V
and returns BF16 context `[1,4096]` and K/V caches `[1,33,2,128]`.

The changes stay in the C++ compiler and reuse existing hardware:

- Prepare Q's matrix rows and BF16 RHS tiles once per query block, retaining
  them in MEM for subsequent KV blocks. The cache validates its source,
  shape, orientation and precision mode; it does not retain an ICU context.
- Read BF16 inputs directly in the activation MXM path. Previously it
  expanded them to FP32 in MEM, then read only the original upper BF16 planes.
  Native BF16 homes remove those conversion stages while keeping the same
  BF16 products, FP32 tile accumulations and FP32 inter-tile additions.
- Chain `subtract -> exp` for probabilities and online rescale factors.
  This removes intermediate FP32 stores/loads without moving a BF16 rounding
  boundary. A one-key probability row also serves directly as its FP32 sum.
- Preserve the necessary row assembly before transposing a flattened Q
  projection view. Unsupported lane-changing views now fail compilation
  instead of silently falling through to a whole-word MEM copy.

| Actual attention cycles | Previous query-lane | Current |
| --- | ---: | ---: |
| Initial ingress | 2,384 | 2,384 |
| Static device schedule, including gaps | 41,237 | 28,047 |
| Weight-page waits | 0 | 0 |
| Output transfer, including KV diagnostics | 4,845 | 5,357 |
| **Total** | **48,466** | **35,788** |
| **Total at 500 MHz, ms** | **0.096932** | **0.071576** |

This reduces complete invocation latency by **26.16%** (1.3543x). Against
the original 83,444-cycle baseline, the reduction is **57.11%** (2.3316x).
The total includes input and output transfers at the same 51.2 GB/s target;
diagnostic cache readback is included in the output-transfer window and is
not separately measurable in this trace. SXM transpose issues over both
hemispheres decrease from 70 to 62. Actual ICU instructions decrease from
2,925 to 2,095; iMEM words decrease from 3,517 to 2,459. These counts include
the exported pre-execution domains and are not all resident simultaneously.

Every context and K/V BF16 word matches the previous query-lane, packed-U
and scalar Flash executions. Context relative L2 against the independent
reference remains **2.43311%**. Eight shape/semantic regressions cover
negative logits, GQA, causal offsets, multiple query/KV blocks, value-width
tails, D=128/V=128 with a 15-key tail, and the flattened projection-Q view.
The baseline kernel recompiled with this compiler is byte-identical to its
previous executed binary (`add49f2e0e01899a0416de7f83b5ca97fd636d1153eabc9bed70b417f64cf696`).

Reproduce from the SOFTWARE root, keeping the earlier comparator artifacts:

```powershell
python -B compiler/tests/glm4_flash_streamed_test.py `
  --build-dir tmp/glm4_flash_native_build `
  --target-config ../FTLPU-CMODEL/config/ftlpu-lpu32.json `
  --previous-results tmp/glm4_flash_streamed_before_reuse
```

The build directory is a snapshot of the tested compiler and the unchanged
CModel runner. `--previous-results` is optional and points to the frozen
48,466-cycle run; when supplied, graph, inputs, hardware and program hashes
are verified before the exact output and timing comparison. Results update
`results/glm4_9b_decoder_layer_decode_flash_streamed/batch1/`, including
`result.json`, actual `attention/runtime.pipeline.csv`,
`attention/actual.runtime.ftlpu` and per-queue `attention/icu_programs/`.

Remaining opportunities include reducing the FP32 inter-tile GEMM sum
traffic and fusing more softmax/PV/online-update boundaries. These require
preserving accumulation order, dependencies and ICU queue constraints.
Long-cache DDR paging is still unimplemented. This result establishes no
new whole-layer, batch-32 or batch-64 performance measurement.

## Earlier MXM-to-VXM partial sums and final cast (2026-11-06)

Scope and precision remain unchanged: batch-1 GLM-4-9B layer-0 decode
FlashAttention core plus KV append, one token with 32 historical entries,
500 MHz and 51.2 GB/s. QKV/O projections, RoPE and the decoder tail are
outside this invocation. The CModel runner and hardware configuration are
identical to the preceding measurements.

The native activation GEMM now feeds each subsequent 32-element K partial
product directly from MXM into VXM. VXM adds the previous FP32 partial sum;
it does not accumulate the full K dimension in MXM or change the order of
FP32 additions. Both matrix operands have mirrored homes, so MEM addresses
select the logical head half already colocated with the prior sum. The
new sum crosses to the opposite hemisphere through the existing output
path. Successive K tiles alternate ownership without copying the prior
sum into mirrored storage.

On the final K tile, C0 performs FP32 addition and C1 multiplies by one and
converts to BF16. This preserves the previous final score-cast boundary
while eliminating its separate four-plane store/read. The final online
denominator addition and reciprocal also share one ALU chain. The compiler
recognizes ordinary `head_matmul`, `add` and `multiply` operations; no new
hardware instruction, routing capability or model computation was added
to CModel.

| Actual invocation cycles | Previous Flash | Current Flash | Current baseline |
| --- | ---: | ---: | ---: |
| Initial ingress | 2,384 | 2,384 | 3,244 |
| Static device schedule, including gaps | 28,047 | 22,808 | 49,284 |
| Weight-page waits | 0 | 0 | 0 |
| Output transfer, including KV diagnostics | 5,357 | 5,357 | 5,361 |
| **Total cycles** | **35,788** | **30,549** | **57,889** |
| **Total at 500 MHz, ms** | **0.071576** | **0.061098** | **0.115778** |

Flash latency falls another **14.64%** (1.1715x). Because the public
activation-GEMM path also uses the native lowering, baseline was freshly
compiled and executed with the same compiler, runner, projection outputs
and KV fixture. Current Flash takes **47.23% fewer cycles** than that
baseline (1.8950x). The earlier 83,444-cycle baseline is a historical
comparison, not the current baseline. Output readback includes the KV
integrity diagnostics; their duration cannot be separated from this trace.

Actual exported ICU instructions fall from 2,095 to **1,661**, and iMEM
words from 2,459 to **1,957**; exports include pre-execution domains, not
simultaneous residency. SXM transpose issues remain 62 across both sides.
Every BF16 context/K/V word matches the preceding Flash execution. Context
relative L2 remains **2.43311%**; current baseline has **2.44241%**. All nine
shape/semantic regressions pass and match their previous outputs bitwise,
including the new D=96, V=96 case that exercises an odd number of K tiles
and the opposite final partial-sum hemisphere.

Reproduce the two actual executions from the SOFTWARE root:

```powershell
python -B compiler/tests/glm4_flash_streamed_test.py `
  --build-dir tmp/glm4_flash_epilogue_build `
  --target-config ../FTLPU-CMODEL/config/ftlpu-lpu32.json `
  --previous-results tmp/glm4_flash_streamed_before_epilogue

python -B compiler/tests/glm4_attention_comparison_test.py `
  --checkpoint models/official_dense/glm4_9b `
  --flash-results results/glm4_9b_decoder_layer_decode_flash `
  --executed-flash-results results/glm4_9b_decoder_layer_decode_flash_streamed `
  --build-dir tmp/glm4_flash_epilogue_build `
  --target-config ../FTLPU-CMODEL/config/ftlpu-lpu32.json `
  --output-dir results/glm4_9b_decoder_layer_decode_flash_streamed/baseline_current `
  --batches 1
```

The first command updates the usual `batch1/attention` pipeline, runtime
program and per-ICU CSVs. The second writes a separate same-compiler
comparison to `baseline_current/batch1/result.json`, including both timing
breakdowns, precision checks and input/hardware provenance. It leaves the
original baseline runtime artifacts intact. Regression summaries and
program hashes are in `results/flash_attention_streamed/regression_summary.json`.

Further opportunities remain in probability/PV layouts and online-U
updates. A query-lane U experiment was numerically exact but increased
complete runtime through extra GEMM tiling and transposes, so it is not
selected. Resident KV limits still apply; no long-cache DDR paging,
full-layer or large-batch performance claim is made by this test.

## Singleton PV and Q row transfers (2026-11-07)

The batch-1 GLM-4-9B layer-0 decode fixture consumes BF16 Q/K/V from the
real W8A16 projection, plus 32 historical KV positions. The raw-macro
program returns one 4,096-channel BF16 context and both updated caches,
each shaped `[1,33,2,128]`. It runs on the LPU32 target with wide SXM,
500 MHz and 51.2 GB/s peak DDR bandwidth.

The first block has 32 keys. The second has one. For that second PV,
VXM multiplies each query's BF16 probability by its KV head's BF16
value vector. The four-plane FP32 product feeds the online numerator.
The same lowering applies whenever a tile has one live key. In a
separate schedule change, the passive MEM copy groups consecutive rows
with matching address strides. Preparing Q now takes four two-dimensional
domains covering the same 256 vectors, in the same order, as the previous
64 one-dimensional domains.

| Actual cycles | Previous Flash | Singleton PV only | Both changes | Same-build baseline |
| --- | ---: | ---: | ---: | ---: |
| Initial ingress | 2,384 | 2,384 | 2,384 | 3,244 |
| Static device schedule, including gaps | 22,808 | 21,093 | 18,513 | 49,284 |
| Weight-page wait | 0 | 0 | 0 | 0 |
| Output transfer including KV diagnostics | 5,357 | 4,845 | 4,845 | 5,361 |
| **Invocation total** | **30,549** | **28,322** | **25,742** | **57,889** |

Both changes reduce complete latency by 15.74%. The current Flash
invocation takes 51.484 µs and is 2.249× faster than the same-compiler
baseline. The output-transfer interval includes diagnostic readback of
the complete updated K/V. The trace reports these transfers together.
Against the preceding Flash build, all 4,096 context words and all 16,896
cache words match bit for bit. Against the independent reference, context
relative L2 is 2.43311%. Eleven shape and masking regressions also match
their previous BF16 outputs bit for bit.

From the SOFTWARE root, reproduce Flash and its baseline with the frozen
compiler and runner used in both measurements:

```powershell
python -B compiler/tests/glm4_flash_streamed_test.py `
  --build-dir tmp/glm4_flash_tail_domain_build `
  --target-config ../FTLPU-CMODEL/config/ftlpu-lpu32.json `
  --previous-results tmp/glm4_flash_before_tail

python -B compiler/tests/glm4_attention_comparison_test.py `
  --checkpoint models/official_dense/glm4_9b `
  --flash-results results/glm4_9b_decoder_layer_decode_flash `
  --executed-flash-results results/glm4_9b_decoder_layer_decode_flash_streamed `
  --build-dir tmp/glm4_flash_tail_domain_build `
  --target-config ../FTLPU-CMODEL/config/ftlpu-lpu32.json `
  --output-dir results/glm4_9b_decoder_layer_decode_flash_streamed/baseline_current `
  --batches 1
```

The Flash trace, executed binary and per-ICU CSVs are in
`results/glm4_9b_decoder_layer_decode_flash_streamed/batch1/attention`.
The actual program has 1,326 ICU instructions and occupies 1,646 iMEM
words. The same-directory tree under `baseline_current/batch1` contains
the controlled baseline. Per-case runtime and output checks are in
`results/flash_attention_streamed/regression_summary.json`.

## Batch-32 decode (2026-11-07)

One attention invocation processes 32 independent layer-0 requests, each
with one new token and 32 history tokens. Device projection outputs are
BF16 Q `[32,32,128]` and K/V `[32,2,128]` in logical head order. The
program returns BF16 context `[32,4096]` and two caches `[32,33,2,128]`.
The fixture, raw-macro runner, wide SXM, 500 MHz clock and 51.2 GB/s
peak DDR bandwidth are identical across the two batch-32 Flash runs.

Each request's two KV heads share a pair of hemispheres. Each hemisphere
uses 16 query-head lanes for its own KV head. The program schedules 32
such pairs and applies the singleton PV specialization to all 32 tails.
Q setup transfers the same 8,192 vectors using 128 domains, down from
2,048. The full runtime includes input transfer and the complete updated
caches sent back for validation.

| Batch-32 cycles | Previous Flash | Current Flash | Current ordinary attention |
| --- | ---: | ---: | ---: |
| Initial ingress | 38,998 | 38,998 | 51,094 |
| Device schedule including gaps | 742,430 | 599,252 | 1,575,228 |
| Weight-page wait | 0 | 0 | 0 |
| Output including diagnostic readback | 83,575 | 83,575 | 75,131 |
| Total | 865,003 | 721,825 | 1,701,453 |

The complete Flash batch takes 1.443650 ms, or 22,166 attention tokens/s.
It reduces latency by 16.55% relative to the previous Flash and runs
2.357× faster than ordinary attention built with the current compiler.
Throughput rises 14.12% relative to the current batch-1 run. Per request,
the device schedule uses 18,726.625 cycles at batch 32 and 18,513 at
batch 1. Input and output transfer together drop from 7,229 to 3,830.41
cycles per request. All 131,072 context words
and 540,672 cache words match the previous compiler bit for bit. Overall
context relative L2 against the independent oracle is 2.57840%; the
worst request (31, zero-based) is 3.41154%. Request 0's context and both
full caches also match the batch-1 invocation bit for bit.

The linked VXM0 and VXM1 streams contain 2,208 and 2,592 words against a
2,048-word local depth. The compiler declares 96 drain boundaries. Across
the 97 resulting segments, the two queues peak at 42 and 39 words.
`batch32/attention/compute_imem_segments.json` records the bounds, sizes
and linked-program hash. The data-transfer timeline accounts for inputs and
outputs; it does not measure DDR transport of replacement instruction
images. The fixed directory contains the actual runtime binary, pipeline
and per-ICU program files under `batch32/attention`.

From the SOFTWARE root, run the before/after numerical and timing checks:

```powershell
$streamed = "results/glm4_9b_decoder_layer_decode_flash_streamed"
$previous = "tmp/glm4_flash_before_tail_batch32"
$target = "../FTLPU-CMODEL/config/ftlpu-lpu32.json"
python -B compiler/tests/glm4_flash_streamed_test.py --batch 32 `
  --build-dir tmp/glm4_flash_epilogue_build --target-config $target --output-dir $previous
python -B compiler/tests/glm4_flash_streamed_test.py --batch 32 `
  --build-dir tmp/glm4_flash_tail_domain_build --target-config $target `
  --output-dir $streamed --previous-results $previous
python -B compiler/tests/glm4_attention_comparison_test.py --batches 32 `
  --checkpoint models/official_dense/glm4_9b --build-dir tmp/glm4_flash_tail_domain_build `
  --target-config $target --flash-results results/glm4_9b_decoder_layer_decode_flash `
  --executed-flash-results $streamed --output-dir "$streamed/baseline_current"
```
