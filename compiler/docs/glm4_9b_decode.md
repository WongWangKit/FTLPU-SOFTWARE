# GLM-4-9B batched decode

## Shared planar operand access contracts (2026-11-11)

The [common access-plan refactor](flash_attention.md#common-planar-operand-access-planning-2026-11-11)
replaces special repeated-row/coefficient fields with one pre-schedule
`PlanarReadPlan` shared by the graph planner, VXM and MXM. Joins can resolve
to address views or compact unique logical rows before allocating expanded
buffers. Unsupported storage/ownership, lane changes and resource conflicts
retain explicit conversion. The proof uses tensor references, not weight
values, filenames or a FU trace. Existing concurrency and precision contracts
are unchanged; no CModel hardware capability is added.

Same Flash core + KV append scope and BF16/FP32 precision as below; real
GLM projection-output fixtures, batch1/32, past32, one new token. Scope
excludes projection, RoPE, O projection and FFN. At 500 MHz / 51.2 GB/s:

| Batch | Total cycles before → after | MEM-to-MEM read+write KiB before → after |
| --- | ---: | ---: |
| 1 | 19,191 → 18,695 (-2.58%) | 404.5 → 342.5 (-15.33%) |
| 32 | 527,337 → 511,465 (-3.01%) | 13,456 → 11,472 (-14.74%) |

Device execution is 12,537 / 401,180 cycles, initial C2C ingress 2,384 /
38,998, and output plus diagnostic-cache readback 3,774 / 71,287. Page wait
is zero. The totals are 0.037390 / 1.022930 ms and include this I/O. iMEM
load/reload transfer latency remains unmodeled. Batch1 fits as a whole;
batch32 uses 97 formal compute segments, all of which fit.

For each request's single-key tail, four K32 input tiles now retain two
distinct rows instead of 64 replicated rows; READ3D repeats the addresses.
Batch1 has 7 accepted operand address views and 4 compact joins; batch32 has
224 and 128. Context and both full KV caches remain byte-identical to the
preceding accepted run, and the original reference errors/thresholds are
unchanged. Eleven Flash shape cases, activation batch GEMM, FP32 scalar
broadcast, the planar access unit test and native dense core regression pass.

This is the first shared input-access contract. It does not yet eliminate
multi-producer U assembly or perform global producer-output layout negotiation.
The retained copy decisions are visible in `graph_plan.json`; a failed
optimization reason does not imply that the copy is intrinsically required
by hardware.

```powershell
python -B compiler/tests/glm4_attention_comparison_test.py --build-dir tmp/glm4_access_plan_1111 --batches 1 32 --strategies flash
python -B compiler/tools/summarize_planar_access.py
```

Results stay in `results/glm4_9b_decoder_layer_decode_flash_streamed`;
`planar_access.json` contains the before/after record. For each batch, the
actual trace/program are `attention/runtime.pipeline.csv`,
`attention/actual.runtime.ftlpu` and `attention/icu_programs/index.csv`.
The frozen source, build commands and input hashes are recorded in
`tmp/glm4_access_plan_1111/build_manifest.json`. Concurrent unrelated
workspace edits were preserved and are excluded from this validation build.

## Flash decode address views and serial scheduling (2026-11-11)

Scope: **Flash attention core plus KV append**, batch 1/32, past length 32,
one new token per request. Q/K/V are the existing real W8A16 projection-output
fixtures. This measurement excludes QKV projection, RoPE, O projection and
FFN. Inputs are Q `[B,1,32,128]`, current K/V `[B,1,2,128]`, and independent
past K/V `[B,32,2,128]`; outputs are context `[B,1,32,128]` and full updated
caches `[B,33,2,128]`. BF16 Q/K/V, scores, P and context boundaries and FP32
matmul accumulation / online m, l, U are unchanged.

The C++ graph planner now retains affine views of scalar-broadcast buffers:
the live query-head ranges of probability, alpha and inverse-denominator
read their existing addresses. The single-key tail stores each V row once
per hemisphere and repeats its address in READ3D rather than expanding V
for every query head. Queue exclusion and producer lifetime checks apply
before either alias is selected. Neither optimization changes hardware.

Flash-generated serial kernels now start at the preceding drained cursor,
accounting for their native event origin. The duplicated 96-cycle linker
and 64-cycle launch allowances are removed; terminal-write/native-timeline
drain accounting remains. Typical consecutive small-kernel gaps fall from
191 to 31 cycles. This is not a zero-idle or zero-copy claim: K/V preparation,
cross-hemisphere numerator assembly, and diagnostic exports remain.

Raw-macro results at 500 MHz / 51.2 GB/s:

| Batch | Total cycles before → after | Total latency before → after | Reduction |
| --- | ---: | ---: | ---: |
| 1 | 24,171 → 19,191 | 0.048342 → 0.038382 ms | 20.60% |
| 32 | 702,930 → 527,337 | 1.405860 → 1.054674 ms | 24.98% |

| Batch | Initial C2C ingress | Device execution before → after | Output + diagnostic readback before → after | Page wait |
| --- | ---: | ---: | ---: | ---: |
| 1 | 2,384 | 18,525 → 13,033 | 3,262 → 3,774 | 0 |
| 32 | 38,998 | 592,645 → 417,052 | 71,287 → 71,287 | 0 |

Linker-inserted MEM-to-MEM read+write traffic, including validation exports,
falls from 502,272 to 414,208 bytes for batch1 (-17.53%) and from 16,596,992
to 13,778,944 bytes for batch32 (-16.98%). These counts do not represent all
native MEM activity or external DDR savings. Batch1's changed placement
increases readback cost; that cost is included in the total above.

Context and both full caches are **byte-identical to the previous accepted
run**. Context relative L2 against the original numerical reference remains
2.4331% / 2.5784%, below the unchanged 3% threshold. Eleven synthetic Flash
cases cover negative logits, GQA, causal masking, wholly masked tails,
partial/query tiles, wide and odd reductions and a single valid key.
Batch1's actual program fits iMEM as a whole. Batch32 requires 97 formal
compute segments; all segments and transport programs fit. Instruction
loading/reloading transfer latency is not modeled in the timing above.

Reproduce from `FTLPU-SOFTWARE`:

```powershell
python -B compiler/tests/glm4_attention_comparison_test.py --build-dir tmp/glm4_flash_decode_tight_1111 --batches 1 32 --strategies flash
python -B compiler/tools/summarize_flash_decode_transport.py
```

The fixed results remain under
`results/glm4_9b_decoder_layer_decode_flash_streamed/batch{1,32}/attention/`:
`runtime.pipeline.csv`, `actual.runtime.ftlpu`, `icu_programs/index.csv`,
and `compute_imem_segments.json`. The root `transport_cleanup.json` records
the comparison; each batch's `before_transport_cleanup.json` preserves the
prior result and copy accounting. `tmp/glm4_flash_decode_tight_1111/` contains
the isolated compiler source, build commands and `build_manifest.json`;
the snapshot deliberately excludes concurrent unrelated source edits.

## Stream softmax P directly into wide SXM (2026-11-11)

The standalone native dense core now schedules wide SXM from the final
softmax normalization stream. The route is:

```text
VXM normalization + BF16 cast -> SR -> wide SXM transpose -> PV-layout MEM -> MXM PV
```

The intermediate probability-pack Write3D/Read3D round trip and the separate
post-softmax transpose phase are removed. The transposed P storage remains
because PV reads it for multiple value-channel blocks. SXM still captures
complete 32x32 tiles; its tile fill/drain latency is part of the schedule.

The C++ memory planner starts the transposed P lifetime during softmax and
excludes the score-read ICU queues. SXM uses reverse SR ports 4/5, separate
from the next head's x/max input ports 0/1 and 2/3. Thus SXM draining can
overlap VXM work without preempting a MEM ICU instruction. The existing
BF16 P boundary and FP32 arithmetic are unchanged. The direct path is used
for the wide-SXM, one-MXM-per-hemisphere dense core when its read/write queues
are disjoint; legacy whole-decoder layouts retain their previous path.

Reproduce the same real W8A16 projection-output fixtures (decode one token
per request, past length 32, independent caches):

```powershell
python -B compiler/tests/glm4_attention_comparison_test.py --build-dir tmp/glm4_p_stream_1111 --batches 1 32 --strategies dense
python -B compiler/tests/dense_attention_core_test.py --build-dir tmp/glm4_p_stream_1111 --target-config ../FTLPU-CMODEL/config/ftlpu-lpu32.json --output-dir results/dense_attention_core
python -B compiler/tests/glm4_prefill_attention_test.py --build-dir tmp/glm4_p_stream_1111 --synthetic --output-dir results/glm4_9b_prefill_attention_comparison/synthetic
python -B compiler/tools/summarize_dense_probability_streaming.py
```

The fixed result directories are unchanged. `before_p_streaming.json`
preserves the immediately preceding accepted run in each batch directory.
`tmp/glm4_p_stream_1111/build_manifest.json` records the frozen compiler
source, base build, link inputs and unchanged raw-macro runner. The actual
runtime traces and programs are `attention/runtime.pipeline.csv`,
`attention/actual.runtime.ftlpu`, and `attention/icu_programs/index.csv`.

Accepted raw-macro measurements at 500 MHz / 51.2 GB/s, compared with the
immediately preceding accepted transport-cleanup run:

| Batch | Runtime cycles before / after | Runtime ms before / after | Reduction |
| --- | ---: | ---: | ---: |
| 1 | 53,118 / 52,062 | 0.106236 / 0.104124 | 1.99% |
| 32 | 1,623,921 / 1,590,129 | 3.247842 / 3.180258 | 2.08% |

Batch-32 actual runtime traffic removes 131,072 MEM word reads and 262,144
MEM word writes (393,216 transfers of 32 bytes, 12 MiB of on-chip traffic).
These are native operator transfers, not linker-copy counts or DDR savings.
All 1,024 head windows stream directly; the P-pack binding and the separate
probability-transpose phase are absent. The timing breakdown is:

| Batch | Initial C2C ingress | Device window | Output including diagnostics | Data-page wait |
| --- | ---: | ---: | ---: | ---: |
| 1 | 3,195 | 45,091 | 3,776 | 0 |
| 32 | 39,797 | 1,466,652 | 83,680 | 0 |

Context and both full K/V caches are bit-identical to the prior accepted
run. Aggregate context relative L2 remains 2.4424% / 2.5809%, with the
unchanged 3% acceptance threshold. All four dense-core regressions pass,
now also asserting the streamed route. The synthetic prefill seq32 dense
and Flash regressions pass; dense relative L2 remains 0.1613%.
`baseline_current/probability_streaming.json` contains the trace-bound
comparison. The actual-program iMEM audit passes for batch 1 without
reloads, and for all 33 batch-32 compute segments plus transport queues.
Batch 32 still cannot retain the full aggregate image at once.

The scope is attention core plus KV append, excluding projection, RoPE,
O projection and FFN. Runtime includes C2C ingress and complete cache/context
readback; diagnostic-only readback is not separately measured, and iMEM
loading/reloading transfer latency remains excluded.

## Dense decode transport cleanup (2026-11-11)

The ordinary attention core retains its QK/softmax/PV arithmetic and its
BF16/FP32 boundaries. For a single query token, PV now writes head blocks
directly into a planar byte pair. Complete request results are exposed
through the existing segmented output binding instead of being gathered
into another MEM allocation. Independent KV caches still remain independent.

Single-row retained graph inputs are allocated off their native consumers'
MEM queues, including split-bank segments. This avoids same-ICU copies that previously
needed a round trip through scratch. The necessary input layout conversion
still uses passive SR. Softmax's score mirrors issue consecutive head reads
and drain the writes once before changing direction; each ICU still executes
one instruction at a time. No hardware contract or arithmetic threshold was
changed.

Multi-row input bindings retain the existing C2C striping. An unrestricted
queue-exclusion candidate reduced device cycles but increased batch-32
ingress from 39,797 to 90,393 cycles and made total runtime worse. That
candidate was rejected; its measurements are retained in
`baseline_current/transport_allocation_tradeoff.json`.

Reproduce with the fixed real-projection fixture used by the prior run:

```powershell
python -B compiler/tests/glm4_attention_comparison_test.py --build-dir tmp/glm4_decode_mem_1111 --batches 1 32 --strategies dense
python -B compiler/tests/dense_attention_core_test.py --build-dir tmp/glm4_decode_mem_1111 --target-config ../FTLPU-CMODEL/config/ftlpu-lpu32.json --output-dir results/dense_attention_core
python -B compiler/tests/glm4_prefill_attention_test.py --build-dir tmp/glm4_decode_mem_1111 --synthetic --output-dir results/glm4_9b_prefill_attention_comparison/synthetic
python -B compiler/tools/summarize_dense_decode_transport.py
```

The compiler source/header snapshot and tool hashes are recorded in
`tmp/glm4_decode_mem_1111/transport_build_manifest.json`. The runner, hardware
target and real input fixture match the previous measurement. Results update
`results/glm4_9b_decoder_layer_decode_flash_streamed/baseline_current/batch{1,32}`;
`before_transport_optimization.json` preserves the prior measurements and
output hashes. Flash's real-weight results were not rerun in this cleanup.

These are attention-core plus KV-append measurements, including input and
output transfers; QKV/RoPE/O projection and FFN are outside their timing.
Instruction-image loading/replacement latency is still not timed. The
output readback includes full-cache diagnostics and must not be interpreted
as compute-only latency. The remaining linker copies assemble native Q/K/V
input layouts and the two interleaved, complete KV outputs; the context
output no longer needs a MEM gather. Softmax reduction storage and SXM
transpose buffers are separate from these linker-copy counts.

The accepted raw-macro runs at 500 MHz are:

| Batch | Runtime before / after (cycles) | Runtime before / after (ms) | Latency reduction | Linker-copy vectors before / after |
| --- | ---: | ---: | ---: | ---: |
| 1 | 54,425 / 53,118 | 0.108850 / 0.106236 | 2.40% | 2,032 / 1,744 |
| 32 | 1,659,883 / 1,623,921 | 3.319766 / 3.247842 | 2.17% | 65,024 / 57,856 |

Batch-32 linker-copy domains fall from 2,768 to 2,544. All 224 context
gather domains (7,168 vectors) disappear. The 16 remaining output-copy
domains are complete K/V cache assembly, not context copies. The measured
batch-32 output transfer nevertheless rises slightly (83,578 to 83,680
cycles): segmented context readback does not itself reduce external bytes.

| Batch | Initial ingress before / after | Device window before / after | Output including diagnostics before / after |
| --- | ---: | ---: | ---: |
| 1 | 3,199 / 3,195 | 47,448 / 46,147 | 3,778 / 3,776 |
| 32 | 39,797 / 39,797 | 1,536,508 / 1,500,444 | 83,578 / 83,680 |

There is no weight-page wait in these attention-core fixtures. No separate
diagnostic-only readback interval is inferred. MXM Compute issue counts are
unchanged (8,192 per hemisphere for batch 1; 262,144 for batch 32). Batch-32
Compute issue coverage of the complete runtime increases from 15.79% to
16.14%; this is timeline coverage, not useful MAC-lane utilization.

Context and both complete caches are bit-identical to the previous run.
Context aggregate relative L2 remains 2.4424% / 2.5809% for batch 1 / 32,
within the unchanged 3% criterion. All four dense-core regressions pass
(including the expected unaligned-position rejection), as does the
synthetic prefill comparison for both dense and Flash strategies.
`baseline_current/transport_optimization.json` records the full comparison,
fixture/tool hashes and actual program identities.

The physical iMEM audit is bound to each executed program. Batch 1 fits
without compute reloads. Batch 32 cannot hold the whole aggregate image at
once (six queues overflow); its 33 compute segments all fit at the existing
compiler-declared drained boundaries, and the transport queues fit. This
is not a claim that batch 32 executes from one resident iMEM image. The
reported latency excludes instruction-image load/reload transfer cost.
See each phase's `compute_imem_segments.json` for the capacity audit.

## Ordinary attention uses the dense core (2026-11-10)

The ordinary attention test now emits `ftlpu.kernel.dense_attention`.
The C++ logical-graph planner splits independent requests into
`attention_core` nodes and uses the existing dense attention stage planner,
memory layout, and `emitQk`, `emitSoftmax`, `emitProbabilityTranspose`, and
`emitPv` implementations. It does not emulate projection tasks to match a
whole decoder graph. `strategy='generic'` explicitly selects the former
batch-matmul/vector-ALU baseline for diagnostics; the default and the
`baseline` compatibility alias use the dense core.

This comparison is **attention core plus KV append**, with Q/K/V taken
from the archived real W8A16 layer-0 projection execution. It excludes
projection, RoPE, RMSNorm, O projection and FFN execution time. Each request
has Q `[1,32,128]`, K/V `[33,2,128]`, position 32 and an independent history.
The outputs are context `[batch,4096]` and full K/V `[batch,33,2,128]`.
The numerical check uses an independent oracle derived from the actual
Q/K/V inputs, keeps the existing aggregate relative-L2 limit of 3%, and
requires the complete returned caches to match their inputs bit for bit.

Layout and padding are planned in C++. KV is explicitly zero-padded to
64 physical tokens and padding is excluded from softmax. The dense Q/K/V
ingress uses whole-word address mappings. Wide SXM handles probability
transpose and feeds transposed V directly to MXM one-vector weight loads.
PV writes only live query rows, so decode does not retain 32 padded rows
per head. Independent requests declare drain boundaries for the existing
physical iMEM segmentation mechanism; no FU state crosses a reload.

Inputs, scores, probabilities and context are BF16; MXM accumulates in
FP32. The native dense softmax stores its scaled score and maximum in BF16,
computes exponentials/sums in FP32, and retains the reciprocal sum in an
FP32 local scalar. Flash retains its online maximum/denominator/numerator
in FP32. These algorithm-specific storage boundaries and all build/input
hashes are recorded in each `result.json`.

Both implementations are freshly compiled and executed with the same
compiler, raw-macro runner, wide SXM and target. The fixed output root is
`results/glm4_9b_decoder_layer_decode_flash_streamed`: ordinary results are
under `baseline_current/batch{1,32}/attention`, and Flash results under
`batch{1,32}/attention`. Each contains `runtime.pipeline.csv`,
`actual.runtime.ftlpu` and `icu_programs/index.csv`. `before_dense.json`
preserves compact historical measurements; those older compiler/runner
measurements are not an isolated before/after lowering comparison.

Reproduce with the imported checkpoint and archived projection fixture:

```powershell
python -B compiler/tests/glm4_attention_comparison_test.py --build-dir tmp/glm4_dense_build --batches 1 32
python -B compiler/tests/dense_attention_core_test.py --build-dir tmp/glm4_dense_build --target-config ../FTLPU-CMODEL/config/ftlpu-lpu32.json --output-dir results/dense_attention_core
```

`tmp/glm4_dense_build` freezes the measured tools. A source build uses
`tools/build_vs2026.cmd --build build-ftlpu-vs2026-direct --target ftlpu_compile ftlpu_graph_compile compiled_stage_file_runner ftlpu_icu_program_export -- -j3`.
Pass that build directory to the same tests. The core currently accepts
one query token or a multiple of 32, 32-aligned query positions, head widths
32/64/128, and one MXM per hemisphere with wide SXM. The synthetic regression
covers independent batch-2 requests, negative logits with padding, causal
masking, a non-default scale, cache preservation and rejection of an
unaligned query position.

Reported runtime includes initial data ingress, device execution, data-page
wait and output readback (including full-cache diagnostics). Instruction
image loading/replacement latency is not timed by this runner. Output and
diagnostic readback share one measured interval; an isolated diagnostic
cost is not inferred from it.

The paired run at 500 MHz (2026-11-10) passed for both strategies:

| Batch | Dense cycles / ms | Flash cycles / ms | Flash speedup | Dense context relative L2 | Flash context relative L2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 54,425 / 0.108850 | 24,171 / 0.048342 | 2.252x | 2.4424% | 2.4331% |
| 32 | 1,659,883 / 3.319766 | 702,930 / 1.405860 | 2.361x | 2.5809% | 2.5784% |

For batch 32, the worst request relative L2 is 3.4088% (dense) and
3.4115% (Flash); acceptance uses the unchanged aggregate 3% criterion.
Both complete returned K/V caches match their common input fixture and
device projection outputs bit for bit. The four synthetic dense-core
regressions also pass, including the expected unaligned-position rejection.

Measured phase breakdown, in cycles:

| Strategy / batch | Initial ingress | Device execution | Data-page wait | Output readback, including diagnostics |
| --- | ---: | ---: | ---: | ---: |
| Dense / 1 | 3,199 | 47,448 | 0 | 3,778 |
| Flash / 1 | 2,384 | 18,525 | 0 | 3,262 |
| Dense / 32 | 39,797 | 1,536,508 | 0 | 83,578 |
| Flash / 32 | 38,998 | 592,645 | 0 | 71,287 |

The formal physical-iMEM planner validates the actual runtime programs:
batch-32 dense uses 33 segments and Flash uses 97 segments. Every segment
fits the configured depths, and the transport programs also fit. Whole
batch images do not fit as single resident VXM/SXM images; the results rely
on the existing segmented loader. `attention/compute_imem_segments.json`
records these checks against the program, target and runtime-trace hashes.
The instruction-reload timing limitation above applies to both results.

## Full decoder harness and historical runs

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
2. Independent KV append, grouped QK, masked softmax and PV. The current
   default uses the dense attention core described above; the historical
   November 4–5 runs used generic batched activation MXM lowering. The 33
   live KV positions are padded to 64 and positions 33–63 are masked.
   `--attention-strategy generic` retains the old diagnostic implementation;
   `--attention-strategy flash` selects tiled online softmax.
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

## Optional tiled online-softmax attention

`--attention-strategy flash` selects the public C++ FlashAttention lowering.
The default remains `baseline`. See [flash_attention.md](flash_attention.md)
for precision boundaries, supported shapes, raw-macro tests, and controlled
same-build attention comparison. Its results use the separate fixed root
`results/glm4_9b_decoder_layer_decode_flash`; the original baseline artifacts
are preserved. For batch 1, one decode token and 32 past KV positions,
singleton PV on VXM and batched Q row transfers reduced the complete
attention invocation from 30,549 to **25,742 cycles** at 500 MHz. This
includes ingress and output transfer of context and updated K/V. The
ordinary attention baseline takes 57,889 cycles with the same compiler,
runner and inputs. The current Flash artifacts are in
`results/glm4_9b_decoder_layer_decode_flash_streamed`, with the baseline
under `baseline_current`. Full decoder layer timings are reported separately.

## Decode FlashAttention with QKV and output projections

The projected-attention test takes layer 0 BF16 hidden states `[B,4096]`
and separate BF16 K and V caches `[B,32,2,128]`. Each request decodes one
token at position 32. The graph runs input RMSNorm, imported W8A16 Q/K/V
projections with biases, GLM partial RoPE on Q/K, cache append, FlashAttention
over 33 positions, and the W8A16 output projection. It returns the projected
hidden update `[B,4096]` and both present caches `[B,33,2,128]`. The residual
add and FFN belong to the surrounding decoder layer.

The fused graph keeps Q/K/V and the attention context on chip. Adjacent
weight pages for the same projection can share its staged activation when
the compiler proves the physical input stays live. The staged control runs
three CModel invocations and transfers intermediate outputs through DDR.
Both use the same input tensors, weights, precision, and hardware target.
The initial history comes from the accepted prefill reference fixture.

From the software repository, use an MLIR-enabled Release build and the
existing GLM import and decode fixtures. Both modes use the runner in the
selected build. The result records that runner and the fixture's original
runner separately.

```powershell
$build = "build-ftlpu-vs2026-direct"
cmake --build $build --config Release --target ftlpu_compile ftlpu_graph_compile compiled_stage_file_runner ftlpu_icu_program_export
python -B compiler/tests/glm4_flash_projected_test.py --build-dir $build --batch 1 --mode both
python -B compiler/tests/glm4_flash_projected_test.py --build-dir $build --batch 32 --mode both
```

The fixed output root is `results/glm4_9b_decoder_layer_decode_flash_projected/optimized`.
Each batch has a staged/fused comparison in `result.json`. The fused
directory contains the actual linked program `actual.runtime.ftlpu`, the
runtime trace `runtime.pipeline.csv`, the per-ICU export `icu_programs/`,
and the numerical results. Total cycles include ingress, physical execution
and output readback, including the full diagnostic K/V outputs. Page waits
are already counted in physical execution. The numerical oracle starts
from hidden states and the W8A16 weights. The harness also compares all
bits of O and both caches between the staged and fused executions.

To measure a runtime linker change on an accepted program, use the replay
driver with the original runner and the candidate executable. It freezes
the ordered raw program inputs, verifies the accepted graph, trajectory
and outputs, and executes the candidate on those exact input bytes.

```powershell
python -B compiler/tests/glm4_projected_runner_replay.py --accepted-dir results/glm4_9b_decoder_layer_decode_flash_projected/optimized/batch32/fused --reference-runner tmp/glm4_projected_reuse_build/runtime/compiled_stage_file_runner.exe --candidate-runner $candidateRunner --output-dir tmp/glm4_projected_batch32_fused_replay
```

Use a separate output directory for each `fused` or `staged` replay.
The result records program, input, runner, target and output hashes,
per-phase timing and bitwise differences. The default reference is the
accepted execution. `--rerun-reference` repeats it on the named historical
runner. `--prepare-only` verifies and freezes the inputs before simulation.

### Measured projected attention, 2026-11-07

The table uses the same final runner for all four measurements, with raw
macro execution, OneVector projection weight loads, 500 MHz and 51.2 GB/s.
The fused and staged programs each fetch 18 weight pages (34 MiB).

| Batch | Three invocations, cycles | One invocation, cycles | Fused time | Reduction | O relative L2 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1,230,329 | 1,206,775 | 2.413550 ms | 1.91% | 0.7511% |
| 32 | 1,510,867 | 1,506,017 | 3.012034 ms | 0.32% | 0.5886% |

| Fused cycles | Ingress | Scheduled execution | Weight wait | Readback including cache diagnostics |
| --- | ---: | ---: | ---: | ---: |
| Batch 1 | 21,121 | 1,165,697 | 16,185 | 3,772 |
| Batch 32 | 43,947 | 1,361,587 | 16,907 | 83,576 |

For batch 32 the initial fused graph took 2,030,437 cycles. Reusing the O
input across eight weight pages reduced this to 1,873,945. Proving regular
MEM macro occupancy with exact consecutive-cycle intervals let the pager
overlap transfers on the same compiled image, reducing waits from 384,835
to 16,907 cycles. Compared with the three-invocation control, fusion saves
70,128 boundary I/O cycles and 22,419 weight-wait cycles, while on-chip
scheduling costs 87,697 additional cycles. The Q packing step gathers
partial-RoPE outputs and untouched channels from separate homes, and the
K tile for the appended position also needs additional staging.

O and both present caches match their staged controls bit for bit for both
batches. The original 32 KV positions are bit-exact. The worst per-request
O relative L2 at batch 32 is 1.0413%. Weight, activation and output traffic
are timed. The runner uses non-timed ICU instruction-image loading. Batch
32's VXM C0/C1 totals (2,250 and 2,628 96-bit words) require segmented
loading into 2,048-word queues. The linked-program capacity audit found 97
planned segments and at most 63/75 resident words on those queues. The
reported latency excludes iMEM loading and replacement costs.

The measured demonstrations use `tmp/glm4_projected_reuse_build` for the
compiler/original runner and `tmp/glm4_projected_overlap_build/runtime/compiled_stage_file_runner.exe`
for the final runner. In the fixed output root, `comparison.json` collects
the real paired results. `batch{1,32}/fused_overlap/candidate/` contains the
final runtime CSV, linked binary and per-ICU exports. The staged control's
runtime artifacts are in the corresponding `staged_overlap/candidate/`.

### Projected attention layout run, 2026-11-10

The projected test now enables the existing C++ `ftlpu.passive_sr_multicast`
policy by default. `--passive-sr-multicast off` selects separate copies.
The compiler also applies its existing default elimination of packed writes
overwritten by a later source. A supported inward SR route performs one MEM
read, a local `WRITE_TAP`, and a remote write. Its source and destination
geometry and queue-alias checks determine eligibility.

The following comparison uses the November 7 continuous programs as the
baseline, the same frozen CModel runner and hardware target, and identical
logical input graphs, weights and numerical policies.

| Batch | Previous cycles | Current cycles | Current time at 500 MHz | Reduction |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1,206,775 | 1,202,656 | 2.405312 ms | 0.3413% |
| 32 | 1,506,017 | 1,401,905 | 2.803810 ms | 6.9131% |

| Current fused cycles | Ingress | Scheduled execution | Weight wait | Readback including cache diagnostics |
| --- | ---: | ---: | ---: | ---: |
| Batch 1 | 21,121 | 1,161,588 | 16,175 | 3,772 |
| Batch 32 | 43,947 | 1,257,550 | 16,832 | 83,576 |

Batch 32's layout transport domains decrease from 3,220 to 2,240. The
actual runtime issues 128,896 fewer MEM reads and 79,360 fewer total MEM
writes including taps. Of the remaining domains, 388 use multicast.
Q assembly domains decrease from 800 to 544, and O input domains from
128 to 32. MXM/VXM arithmetic issue counts and active event-cycle sums
match the prior execution. O and both present caches are bit-exact against
the prior results. Their independent numerical errors match the November
7 table.

The fixed output root remains
`results/glm4_9b_decoder_layer_decode_flash_projected/optimized`. Its
`batch{1,32}/fused/` directories now contain the current runtime CSV,
actual linked ICU program and per-ICU CSVs. The previous execution stays in
`fused_overlap/candidate/`, with its source metadata in
`fused_overlap/accepted_result.json` and
`fused_overlap/accepted_graph_plan.json`. `comparison.json` is the verified
before/after layout comparison. `comparison_before_layout.json` retains
the earlier staged/fused comparison.

The measured compiler snapshot and build hashes are retained in
`tmp/glm4_mirrored_build/build_manifest.json`, alongside the compiled source
snapshot and executables. This isolates the comparison from other compiler
work occurring in the shared workspace. Reproduce the measured run with
the existing imported checkpoint and frozen reference fixtures as follows.

```powershell
python -B compiler/tests/glm4_flash_projected_test.py --build-dir tmp/glm4_mirrored_build --batch 1 --mode fused --passive-sr-multicast on
python -B compiler/tests/glm4_flash_projected_test.py --build-dir tmp/glm4_mirrored_build --batch 32 --mode fused --passive-sr-multicast on
python -B compiler/tests/glm4_projected_comparison.py
```

The current capacity audit is `icu_capacity.txt`. Batch 1 fits the resident
image, while batch 32 still uses the existing 97 planned VXM iMEM segments.
The reported timings exclude instruction-image loading and replacement
costs.
