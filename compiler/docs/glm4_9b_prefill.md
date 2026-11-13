# GLM-4-9B seq32 prefill

This entry adapts the original `zai-org/glm-4-9b` checkpoint (`chatglm`),
not GLM-Edge or GLM-4-9B-0414. It exercises decoder layer 0, including
both RMSNorms, Q/K/V projection and bias, partial RoPE, causal grouped-query
attention, output projection, residuals and SwiGLU FFN. It does not run
the complete 40-layer model, tokenizer, LM head or decode/KV export.

The full dimensions are hidden=4096, FFN=13696, Q heads=32, KV heads=2,
head width=128, rotary width=64, sequence length=32, batch=1.
The norm epsilon is 1.5625e-7. Source QKV and gate/up tensors are fused;
the importer splits them and preserves the nonzero QKV bias. Original
adjacent-pair RoPE is canonicalized only within the first 64 features.
The last 64 head features remain unrotated. Physical partial-RoPE packing
applies the same feature order to Q/K weights and biases.

## Reproduce

From the SOFTWARE repository root, with the configured CModel and compiler built:

```powershell
python tools/run_model_tests.py run --model glm4_9b --suite prefill --download
```

The downloader saves only the official first-layer tensors and embedding
rows 0–31, with revision and SHA-256 provenance, under
`models/official_dense/glm4_9b`. These 32 embedding rows are the test input;
they are not a natural-language prompt. The first downloaded revision was
`8cd2b585357ba9e702647ac4e6fa4fafe5cc7bee`.

Outputs are updated in `results/glm4_9b_decoder_layer_prefill`.
Consult `runner.json` and `result.json` for the actual run status. A compiled
binary or generated reference is not evidence that numerical execution passed.

## Execution and precision

The mathematical graph goes through the existing dense whole-layer graph
recognition, attention/FFN lowering and memory planner. Python handles model
import, reference calculation and orchestration; it does not emit FU schedules.
There are no GLM operators added to CModel.

Execution uses W8A16: official BF16 matrix weights are quantized using the
existing symmetric per-tensor INT8 importer, with BF16 activation checkpoints.
This is not native-BF16 matrix multiplication. The report separately measures
hardware vs the quantized reference and hardware vs the original BF16-weight
reference. The latter includes quantization error and should not be confused
with the hardware acceptance comparison.

The program uses raw macro ICU encoding and one-vector projection weight
loading. Common-barrier synchronization is selected with the compiler's
`--common-barrier` option because the whole layer exceeds the configured
shared queue-control record capacity. The configured hardware capacity is
not increased. DDR/C2C transfers and synchronization still execute in CModel.

Regression checks:

```powershell
python compiler/tests/glm4_decoder_importer_test.py
python compiler/tests/glm4_decoder_layout_test.py --tool build-ftlpu-vs2026-direct/compiler/ftlpu_opt.exe --target-config ../FTLPU-CMODEL/config/ftlpu-lpu32.json --output-dir tmp/glm4_layout_test
build-ftlpu-vs2026-direct/runtime/weight_page_builder_test.exe --partial-rope-bias
```

## Shared exponent read for sum and P (2026-11-12, latest core result)

The native B1 seq32 Flash core now reads the FP32 exponent tile once for
both branches. In depth-2 mode C0/C1 rounds E to BF16 and sends it directly
to wide SXM/MXM, while C2/C3 consumes the same fixed FP32 input bundle and
accumulates E in the original key order. Only the terminal sum feeds back
through C3->C2 for Reciprocal. P is unnormalized, so its capture and weight
load can start before the denominator is ready. The existing planar scratch,
BF16 boundaries, FP32 arithmetic, target capacity and CModel are unchanged.

Same real-weight QKV fixture, raw macros, 500 MHz / 51.2 GB/s (90% efficiency):

| Measurement | Separate sum / P reads | Shared sum / P read |
| --- | ---: | ---: |
| Initial DDR/C2C ingress, cycles | 14,355 | 14,355 |
| Device execution window, cycles | 6,440 | 6,006 |
| Total cycles | 20,795 | 20,361 |
| Total time, microseconds | 41.590 | 40.722 |
| FP32 exponent MEM reads, both hemispheres | 256 KiB | 128 KiB |
| Per-MXM issue occupancy | 67.61% | 72.83% |
| Physical ICU instruction bytes | 23,116 | 22,956 |

The total improves 2.09% and the device window improves 6.74%. Same-build
dense remains 27,165 cycles (54.330 microseconds). Flash and dense outputs
are each bit-identical to their previous accepted outputs. Flash relative
L2 remains 2.907921% against actual-QKV attention and 2.926710% against the
independent projection reference. First-token causal error remains zero.
The four D32/D64/D128, causal/noncausal, GQA and batch2 regressions are also
bit-identical to separate-pass results. Tests count actual MEM FU reads to
require exactly one read of each exponent, including across both hemispheres.

Primary and diagnostic ICU programs pass the physical capacity audit with
at most one active FU 3D context per ICU and zero compute reload segments.
The fixed result directory, runtime pipeline and per-ICU programs remain
`results/glm4_9b_prefill_attention_comparison/`. Functional exponent storage
and reduction dependencies remain; this does not claim zero idle cycles.

Reproduce from the SOFTWARE root:

```powershell
python -B tools/run_model_tests.py run --model glm4_9b --suite attention-prefill --build-dir tmp/glm4_flash_sum_p_1112 --target-config ../FTLPU-CMODEL/config/ftlpu-lpu32.json
python -B compiler/tests/flash_tile_runtime_test.py --build-dir tmp/glm4_flash_sum_p_1112 --target-config ../FTLPU-CMODEL/config/ftlpu-lpu32.json --output-dir results/glm4_9b_prefill_attention_comparison/regression/flash_tile --baseline-dir tmp/glm4_flash_sum_p_1112/before_regression
python -B compiler/tools/summarize_attention_schedule.py --baseline tmp/glm4_flash_sum_p_1112/before --results results/glm4_9b_prefill_attention_comparison --imem-check tmp/glm4_flash_sum_p_1112/check.exe
python -B compiler/tools/summarize_flash_sum_p.py --baseline tmp/glm4_flash_sum_p_1112/before --results results/glm4_9b_prefill_attention_comparison
```

The isolated build keeps the accepted compiler sources/objects and unchanged
runner, with only `FlashTileScheduleEmitter.hpp` rebuilt. Build inputs and
hashes are recorded in `tmp/glm4_flash_sum_p_1112/build_manifest.json` and the
fixed result's `scheduling_build.json`. `sum_p_fusion.json` records the actual
read-traffic comparison. Diagnostics and iMEM-loading latency remain outside
primary timing, as before.

## Native Flash tile pipeline (2026-11-11, earlier core result)

The Flash **B1 seq32 attention core** has been rescheduled as one C++ native
tile unit, replacing 310 separately drained graph nodes. Inputs are the same
device-produced layer-0 W8A16 fixture: Q `[1,32,32,128]`, K/V `[1,32,2,128]`.
The result is resident context `[1,32,32,128]`; this invocation generates no
new persistent state. Projection execution, O projection, residual and FFN
are excluded. There is no change to CModel or hardware capacity.

At **500 MHz / 51.2 GB/s**, with the unchanged target and runner:

| Measurement | Previous Flash | Native Flash | Dense, same build |
| --- | ---: | ---: | ---: |
| Initial DDR/C2C ingress, cycles | 22,059 | 14,355 | 16,394 |
| Device execution window, cycles | 54,880 | **6,440** | 10,771 |
| Weight-page wait during this core, cycles | 0 | 0 | 0 |
| Required external output, cycles | 0 | 0 | 0 |
| Total runtime, cycles | 76,939 | **20,795** | 27,165 |
| Total runtime, ms | 0.153878 | **0.041590** | 0.054330 |
| Pure layout-copy domains | 64 | **0** | — |
| Pure layout-copy payload | 128 KiB | **0** | — |
| MXM compute issue occupancy, per hemisphere | 7.63% | **67.61%** | — |
| VXM C0/C1 issue occupancy | 19.80% | **77.80%** | — |

Occupancy is measured within each resource's first-to-last issue interval,
not averaged over ingress or all ALUs. The unchanged 4,096 compute issues
per MXM now span 6,058 cycles, with 16 gaps rather than 95. Each QK K128
contraction and PV D128 output block is internally continuous. Remaining
inter-stage waits include reductions, SXM tile capture/load and VXM chain
reconfiguration. The result is a **72.97% total-time reduction**, an **88.27%
device-window reduction**, and **1.3063x** the same-build dense core speed.
iMEM loading latency remains outside the runner's timed model.

The schedule changes are structural:

- Ordinary planar ingress places Q/K/V directly at consumer addresses.
  KQ^T loads Q vectors as one-vector MXM weight columns without a transpose.
  K32 blocks accumulate continuously in FP32 ACC, with two weight buffers.
- Two independent MEM ICU slice pairs ping-pong the current/next QK tile.
  Only those tiles are retained; scaled score, exponent and denominator
  scratch are also reused. These five scratch allocations total 20,736 bytes
  across both hemispheres, independent of the number of query tiles (plus
  the 3,968-byte immutable causal mask). Q/K/V and output storage are separate.
- VXM retains max in C1, chains mask/subtract/exp, and chains FP32 sum to
  reciprocal. BF16 P streams from VXM through wide SXM directly to MXM.
  During that capture/load interval VXM prepares the next tile's scale/max;
  its max survives the current PV epilogue without a MEM spill.
- V^T P keeps query rows in lanes through the FP32 denominator epilogue.
  Its VXM output streams directly into SXM to restore feature lanes at the
  final planar context address. There is no intermediate context archive.

Scaled-score, P and context BF16 boundaries are retained, with FP32 MXM
accumulation and VXM arithmetic. **Both dense and Flash outputs are bit-identical
to the previous accepted build.** Flash relative L2 remains **2.907921%** against
attention on actual QKV and **2.926710%** against the independent projection
reference. Both existing aggregate 3% thresholds and first-token causal checks
pass. The separate diagnostic run takes 48,640 cycles: 14,355 ingress, 6,440
core, 7,324 diagnostic layout gathering, and 20,521 DDR readback. Diagnostic
gather/readback is not included in the primary runtime or its pipeline.

The actual primary ICU program uses **23,116 physical instruction bytes**
(previously 75,616), fits all queues, has at most one active FU 3D context per
ICU, and requires no compute-iMEM reload. The diagnostic program also fits.
Independent raw-macro regressions cover D32/D64/D128, causal/noncausal GQA,
multiple KV groups and two independent requests; errors are 0.170–0.177%.
The two-KV-tile decode fallback and required KV-only DDR output policy pass.

The native strategy is selected for BF16 Q/K/V, seq32/KV32 at position zero,
equal head widths 32/64/128, even KV-head count, wide SXM, one MXM per hemisphere,
two weight buffers and bounded planar capacity. Independent requests can be
flattened into separate head groups. Other shapes retain the general Flash
lowering, including multi-KV-tile online softmax; this change does not claim
long-sequence or complete decoder-layer deployment.

Reproduce the measured snapshot from the SOFTWARE root:

```powershell
python -B tools/run_model_tests.py run --model glm4_9b --suite attention-prefill --build-dir tmp/glm4_flash_tile_1112 --target-config ../FTLPU-CMODEL/config/ftlpu-lpu32.json
python -B compiler/tests/flash_tile_runtime_test.py --build-dir tmp/glm4_flash_tile_1112 --target-config ../FTLPU-CMODEL/config/ftlpu-lpu32.json --output-dir results/glm4_9b_prefill_attention_comparison/regression/flash_tile
python -B compiler/tools/summarize_attention_schedule.py --baseline tmp/glm4_flash_tile_1112/before --results results/glm4_9b_prefill_attention_comparison --imem-check tmp/glm4_flash_tile_1112/check.exe
```

The fixed result root remains `results/glm4_9b_prefill_attention_comparison`.
`flash/runtime.pipeline.csv` is the actual primary runtime trace;
`flash/icu_programs/index.csv` indexes per-ICU files from `actual.runtime.ftlpu`.
`scheduling_comparison.json` contains same-condition metrics and physical audits;
`scheduling_build.json` records the frozen build inputs and hashes. Compiler SHA:
`077e91ab5ff1df0f9c7024af7532d0834dc645aa55401038656b57984fe7ed2b`.

## Earlier core schedules and VXM reduction chain (2026-11-11 baseline)

This update runs **B1, seq32 attention cores**, using the same device-produced
layer-0 W8A16 fixture: Q `[1,32,32,128]`, K/V `[1,32,2,128]`, context
`[1,32,32,128]`. It covers QK, causal softmax and PV; projection, O projection,
residual, FFN and new KV generation are outside the timed invocation. The
normal context stays resident; a separate diagnostic execution reads it back.
BF16 storage boundaries, FP32 arithmetic and the two aggregate 3% acceptance
limits are unchanged. Both actual context files are **bit-identical** to the
previous accepted resident-output build.

Measured with the unchanged target at **500 MHz / 51.2 GB/s**:

| Measurement (cycles) | Dense before | Dense after | Flash before | Flash after |
| --- | ---: | ---: | ---: | ---: |
| Initial C2C ingress | 16,394 | 16,394 | 22,059 | 22,059 |
| Device execution window | 17,275 | 10,771 | 55,904 | 54,880 |
| In-execution weight-page wait | 0 | 0 | 0 | 0 |
| Required external output | 0 | 0 | 0 | 0 |
| Total runtime | 33,669 | **27,165** | 77,963 | **76,939** |
| Total runtime (ms) | 0.067338 | **0.054330** | 0.155926 | **0.153878** |

The reductions are 19.32% / 1.31% including ingress, and 37.65% / 1.83%
within the device execution window. iMEM loading latency is not modeled.
This short seq32 case continues to favor the ordinary dense core.

Dense scheduling changes:

- Scale/mask and max passes stream consecutive heads, draining once at the
  phase boundary. Softmax drops from 8,382 to 6,088 cycles. The reciprocal
  remains in the existing FP32 local scalar. A one-cycle gap remains when
  C3 changes from two-cycle multiplication to one-cycle addition; issuing
  them consecutively would collide at the Basic ALU output.
- Two producer hemispheres now drain context into separate ordinary planar
  byte pairs. The layout and consumer/export address mapping carry explicit
  `hemisphere_paired` ownership. This uses four slice planes rather than two,
  removing cross-hemisphere contention on the context MEM ICU queues.
- PV drops from 6,507 to 2,297 cycles. On **each** MXM, active compute is still
  2,048 cycles; first-to-last compute span drops from 6,332 to 2,172 cycles
  (**32.34% → 94.29%**). Maximum gap drops from 68 to 4 cycles. QK already
  issues its 2,048 compute cycles continuously on each MXM.

Flash's single-visible-KV-tile path now lowers FP32 row sum → reciprocal as
one native VXM chain: C0 input, C1 accumulation with terminal-only emission,
C2 forwarding, C3 reciprocal. The sum is neither written to MEM nor reread.
All 16 query tiles use this path, saving 1,024 cycles. Multi-tile online
denominator updates retain their previous path. Subtract → exp and the prior
BF16 P → SXM → MXM connection remain fused. BF16 score/P rounding boundaries
are not silently removed to join more operations.

This is **not** a bubble-free Flash implementation: K32 partial contractions,
query-tile boundaries, layout conversion before softmax and stage drains
remain. Further improvement requires scheduling those dependent tiles
together, rather than deleting waits. Dense PV also retains the weight-buffer
reuse/drain spacing described above. The coverage figures are compute issue
coverage inside each stage's first-to-last interval, not whole-chip utilization.

Numerical and hardware checks:

- Dense/Flash context relative L2: **2.91064% / 2.90792%**; comparison against
  the independent projection reference: **2.92964% / 2.92671%**.
- The local sum/reciprocal regression covers 1, 2, 32 and 33 rows, 32–96
  columns, BF16 and FP32 ingress. Sums match independent FP32 accumulation;
  fused reciprocals match the existing unfused hardware-LUT path bit for bit.
- Dense multi-request causal offset, negative/padded score and scale cases
  pass; the Flash two-KV-tile decode regression preserves K/V bit for bit.
- Actual normal/diagnostic ICU programs fit physical iMEM and transport
  capacity, with at most one active FU 3D context per queue and no reload.

Diagnostic costs are separate: dense adds 10,741 cycles of paired-context
gather plus 32,809 cycles of DDR readback (70,715 total diagnostic cycles);
Flash adds 5,660 + 32,807 cycles (115,406 total). The extra dense gather is
only for the host diagnostic boundary and is absent from the resident run.

The fixed result root remains `results/glm4_9b_prefill_attention_comparison`.
`scheduling_comparison.json` contains timing, stage coverage, unchanged-output
hash proofs and physical-program audits. Normal runtime pipelines and actual
per-ICU CSVs remain in `dense/` and `flash/`; diagnostic exports are beneath
their `diagnostic/` directories. Baselines and the frozen compiler sources are
in `tmp/glm4_prefill_chains_1111`, with `build_manifest.json` provenance.

```powershell
python -B tools/run_model_tests.py run --model glm4_9b --suite attention-prefill --build-dir tmp/glm4_prefill_chains_1111 --target-config ../FTLPU-CMODEL/config/ftlpu-lpu32.json
python -B compiler/tests/row_sum_reciprocal_runtime_test.py --build-dir tmp/glm4_prefill_chains_1111 --target-config ../FTLPU-CMODEL/config/ftlpu-lpu32.json --output-dir results/glm4_9b_prefill_attention_comparison/regression/row_sum_reciprocal
python -B compiler/tools/summarize_attention_schedule.py --baseline tmp/glm4_prefill_chains_1111/before --results results/glm4_9b_prefill_attention_comparison --imem-check tmp/glm4_prefill_chains_1111/check.exe
```

## Required external results only (2026-11-11, before tighter scheduling)

The current **attention-core** benchmark keeps context in its native SRAM
producer layout. The normal invocation emits no context export binding,
no context gather, and no C2C HostOutput/TX. K/V are already DDR inputs to
this standalone core; it does not generate a new KV cache. Projection, O
projection, residual and FFN remain outside this benchmark.

`ftlpu.external_output_indices` is an explicit C++ graph boundary policy.
Omitting it preserves the existing behavior of exporting all returned roots.
An empty array retains every returned root in SRAM. If a graph returns
`(context, new_key_cache, new_value_cache)`, selecting `[1 : i64, 2 : i64]`
exports K/V and retains context. The caller must select required persistent
outputs; this policy does not infer model semantics from tensor names.
Invalid, duplicate and out-of-range indices are rejected.

All returned roots remain live through invocation end, including unexported
ones. `graph_plan.json` records native `resident_output_views` and their
producer references. Same-graph consumers use normal compiler aliasing.
This is not a new automatic cross-invocation alias ABI: connecting a later
standalone executable still requires a compatible device-value binding.

Numerical validation is a **separate diagnostic execution**, saved beneath
each variant's `diagnostic/` directory. It exports context for comparison.
Its native compute instructions, input placements, and operator schedule are
identical to the normal execution; only export/gather is added afterward.
Its numerical output is bit-identical to the preceding accepted run. Core
relative L2 remains 2.90792% for Flash and 2.91064% for dense; the independent
projection-reference-to-context errors remain 2.92671% and 2.92964%.

At 500 MHz / 51.2 GB/s, measured with raw-macro instructions:

| Measurement | Flash export baseline | Flash resident | Dense resident |
| --- | ---: | ---: | ---: |
| Total runtime cycles | 116,430 | 77,963 | 33,669 |
| Total runtime ms | 0.232860 | 0.155926 | 0.067338 |
| Initial C2C input | 22,059 | 22,059 | 16,394 |
| Device execution window | 61,564 | 55,904 | 17,275 |
| Context external readback | 32,807 | 0 | 0 |
| Context gather read + write traffic | 0.50 MiB | 0 | 0 |

The Flash difference is 5,660 cycles of export preparation plus 32,807
cycles of diagnostic readback. Native arithmetic is unchanged. Normal
execution still includes the real input C2C cost and ordinary intermediate
SRAM writes needed by the algorithms. Remaining Flash linker layout-copy
traffic is 0.25 MiB for time-major softmax input. No weight-page waits occur
in this core. iMEM image transfer latency remains unmodeled.

Real-weight QKV fixtures and synthetic seq32 pass. The selective-output
regression runs all outputs, only index 1, and no outputs, verifies actual
C2C activity/output files, and rejects invalid selections. Normal, diagnostic
and selective-output programs all pass physical iMEM/transport checks with
one active FU 3D context per ICU queue. No CModel or hardware changes.

The fixed root remains `results/glm4_9b_prefill_attention_comparison`:

- `flash/runtime.pipeline.csv`, `dense/runtime.pipeline.csv`: normal runs.
- Each variant's `icu_programs/index.csv`: actual executed ICU programs.
- Each variant's `diagnostic/`: independent numerical execution and context.
- `resident_output_policy.json`: paired schedule proof, hashes, metrics and audits.
- `*_before_external_output_policy/`: preserved accepted export baselines.

```powershell
python -B compiler/tests/glm4_prefill_attention_test.py --build-dir tmp/glm4_prefill_resident_outputs_1111
python -B compiler/tests/glm4_prefill_attention_test.py --build-dir tmp/glm4_prefill_resident_outputs_1111 --synthetic --output-dir results/glm4_9b_prefill_attention_comparison/synthetic
python -B compiler/tests/external_output_policy_test.py --build-dir tmp/glm4_prefill_resident_outputs_1111 --target-config ../FTLPU-CMODEL/config/ftlpu-lpu32.json --output-dir results/glm4_9b_prefill_attention_comparison/regression/external_output_policy
python -B compiler/tools/summarize_prefill_resident_outputs.py --build-dir tmp/glm4_prefill_resident_outputs_1111
```

Use `--export-context` only when deliberately benchmarking the old diagnostic
boundary. The isolated build and source hashes are in
`tmp/glm4_prefill_resident_outputs_1111/build_manifest.json`.

## Flash planar transport and serial scheduling cleanup (2026-11-11, export baseline)

Scope: the **B1 seq32 attention core**, Q `[1,32,32,128]`, K/V
`[1,32,2,128]`, and context `[1,32,32,128]`. Inputs are the saved real
layer-0 W8A16 projection outputs. This run does not execute O projection,
residuals, FFN, or persistent KV updates. BF16 Q/K/V, scores, probabilities
and context, FP32 accumulation/intermediates, and both aggregate 3% error
thresholds are unchanged. CModel and hardware configuration are unchanged.

The generic packed row-scale planner now reads the FP32 PV pieces and
scalar coefficients at their producer addresses. East/West process different
rows and stream into distinct output queues. The planner reserves the actual
source queues, including separate producer pieces, until output drain.
This removes the mirrored normalization staging and duplicate calculation.
The new path uses ordinary planar byte planes, not distributed16.

Flash's serial kernels start after the preceding native schedule has drained;
the generic 64-cycle launch/96-cycle retirement allowances are removed.
Required native drain and explicit iMEM safe boundaries remain. Prepared K/V
tiles are retained across query blocks within a bounded SRAM lifetime. For
this external-input fixture physical bindings were already deduplicated, so
that last change reduces graph nodes (416 to 326), not execution time.
The prior BF16 P → wide SXM transpose fusion remains active for all 16 tiles.

Measured at **500 MHz / 51.2 GB/s**, compared with the accepted P-streaming run:

| Measurement | Before | After |
| --- | ---: | ---: |
| Total runtime, including C2C | 159,391 cycles / 0.318782 ms | 116,430 cycles / 0.232860 ms |
| Initial input ingress | 22,059 | 22,059 |
| Device execution window | 116,813 | 61,564 |
| Weight-page wait | 0 | 0 |
| Context readback | 20,519 | 32,807 |
| Linker layout-copy traffic, reads + writes | 2.75 MiB | 0.75 MiB |
| VXM C0 busy fraction within device execution | 10.99% | 17.52% |
| Each MXM compute busy fraction within device execution | 3.51% | 6.65% |

Total latency improves **26.95%**, and the device window improves **47.30%**.
Context readback is slower because final SRAM placement changed; it is
included in the total. There are no separate diagnostic outputs. iMEM image
transfer latency is not modeled. Output is **bit-identical** to the accepted
run: core relative L2 **2.90792%**, independent projection-reference-to-context
relative L2 **2.92671%**. These are aggregate thresholds, not per-token bounds.
The current dense core remains faster for seq32 (66,477 cycles); this is not
a claim that Flash is the preferred short-sequence implementation.

Remaining linker copies: 0.25 MiB establishes mirrored time-major input for
softmax, and 0.50 MiB gathers contiguous context for export. Reduction
dependencies, SXM tile collection and explicit drain boundaries still leave
idle intervals; this change does not claim a completely bubble-free graph.

Raw-macro regressions pass for seq32 synthetic GQA, B2/Q33/KV65/D64/V96
(multi-tile and one-key tail), and B1/Q2/KV17/D32/V96 (partial tile fallback).
Physical instruction memory and transport audits pass for all six dense/Flash
and regression programs, with at most one active FU 3D context per ICU queue
and no compute-iMEM reloads. Python only imports/references/analyzes; the
new layouts and READ3D domains are generated in C++.

The fixed result directory is `results/glm4_9b_prefill_attention_comparison`.
`flash/runtime.pipeline.csv` and `flash/icu_programs/index.csv` are from actual
execution; `flash_transport_optimization.json` records comparison, hashes,
numerics, traffic and capacity audits. `flash_before_transport_cleanup` keeps
the accepted baseline. The isolated build's `build_manifest.json` freezes
source, compiler and runtime provenance alongside other ongoing work.

```powershell
python -B compiler/tests/glm4_prefill_attention_test.py --build-dir tmp/glm4_flash_prefill_tight_1111 --export-context
python -B compiler/tests/glm4_prefill_attention_test.py --build-dir tmp/glm4_flash_prefill_tight_1111 --export-context --synthetic --output-dir results/glm4_9b_prefill_attention_comparison/synthetic
python -B compiler/tests/flash_attention_test.py --build-dir tmp/glm4_flash_prefill_tight_1111 --target-config ../FTLPU-CMODEL/config/ftlpu-lpu32.json --output-dir results/glm4_9b_prefill_attention_comparison/regression/prefill_streaming --case prefill_streaming
python -B compiler/tests/flash_attention_test.py --build-dir tmp/glm4_flash_prefill_tight_1111 --target-config ../FTLPU-CMODEL/config/ftlpu-lpu32.json --output-dir results/glm4_9b_prefill_attention_comparison/regression/single_tile --case single_tile
python -B compiler/tools/summarize_flash_prefill_transport.py --build-dir tmp/glm4_flash_prefill_tight_1111
```

## Flash P producer to SXM streaming (2026-11-11, preceding baseline)

The Flash core now streams its BF16 probability producer into wide SXM:
`FP32 exp(score-max) -> BF16 VXM cast -> SR -> SXM transpose -> MEM -> MXM PV`.
The FP32 exponent tensor remains available for denominator reduction. Only
the BF16 write/read before transposition disappears; transposed P remains
in MEM for reuse across V column tiles. Precision and the aggregate 3% limits
are unchanged.

This shared C++ layout fusion recognizes a sole full-tile transpose consumer
from tensor references and cast semantics. It checks the retained home against
native workspace lifetimes, then emits VXM/SXM/MEM domains directly. Padded
tails and multi-consumer producers retain the materialized path. It does not
use model/test names, compress FU traces, or add CModel operations.

The GLM layer-0 fixture is unchanged: real-weight W8A16 device-produced Q/K/V,
B1, seq32, Q heads32, KV heads2, D128. This core ends at context; O projection,
residual, FFN and persistent KV updates are outside its scope.

| Measurement | Before | After |
| --- | ---: | ---: |
| Initial C2C input cycles | 22,059 | 22,059 |
| Device execution window | 120,734 | 116,813 |
| Runtime weight wait | 0 | 0 |
| Context readback cycles | 20,519 | 20,519 |
| Total runtime cycles | 163,312 | 159,391 |
| Total at 500 MHz / 51.2 GB/s | 0.326624 ms | 0.318782 ms |
| MEM Read FU issues, both hemispheres | 225,536 | 221,440 |
| MEM Write FU issues, both hemispheres | 133,888 | 129,792 |

Total latency falls 2.40%; the device window falls 3.25%. Output is bit-identical:
relative L2 is 2.90792% against actual-QKV attention and 2.92671% against the
independent projection reference. There are 16 streamed producer windows;
the first is at runtime cycles 25,440--25,541. The GLM core needs no compute
iMEM reload. There are no separate diagnostic tensor readbacks. Instruction
image loading/reloading transfer time is not modeled.

The seq32 synthetic causal/GQA pair, batch2 Q33/KV65 with two full KV tiles
and a singleton tail, and the Q2/KV17 padded fallback pass. The latter two
have relative L2 0.23457% and 0.22318%. Physical iMEM/transport audits pass,
with at most one active FU 3D context per queue. Dense remains at 66,477 cycles.

From the SOFTWARE root:

```powershell
python -B compiler/tests/glm4_prefill_attention_test.py --build-dir tmp/glm4_flash_p_stream_build
python -B compiler/tests/glm4_prefill_attention_test.py --build-dir tmp/glm4_flash_p_stream_build --synthetic --output-dir results/glm4_9b_prefill_attention_comparison/synthetic
python -B compiler/tests/flash_attention_test.py --build-dir tmp/glm4_flash_p_stream_build --target-config ../FTLPU-CMODEL/config/ftlpu-lpu32.json --output-dir results/glm4_9b_prefill_attention_comparison/regression/prefill_streaming --case prefill_streaming
python -B compiler/tests/flash_attention_test.py --build-dir tmp/glm4_flash_p_stream_build --target-config ../FTLPU-CMODEL/config/ftlpu-lpu32.json --output-dir results/glm4_9b_prefill_attention_comparison/regression/single_tile --case single_tile
python -B compiler/tools/summarize_flash_probability_streaming.py --build-dir tmp/glm4_flash_p_stream_build
```

Results update `results/glm4_9b_prefill_attention_comparison/flash/`:
`runtime.pipeline.csv`, `actual.runtime.ftlpu`, `icu_programs/index.csv` and
`compute_imem_segments.json`. `flash_before_p_streaming/` retains the same-runtime
baseline. `flash_probability_streaming.json` binds comparisons to artifact hashes.
`tmp/glm4_flash_p_stream_build/build_manifest.json` records the frozen base and
two C++ overlays (compiler `ecbc9766e1d5`); CModel is unchanged at `e42e41d`.

## Streaming P from Softmax VXM into SXM (2026-11-11)

The current `results/glm4_9b_prefill_attention_comparison/dense` run uses
`VXM normalize -> SR -> wide SXM transpose -> MEM -> MXM PV`. It removes
the pre-transpose P write/read round trip. The transposed BF16 P remains
in MEM for reuse across the four 32-channel V blocks. This is the standalone
QK/causal Softmax/PV core: B1, seq32, 32 query heads, 2 KV heads, D128, using
the preserved actual QKV fixture from the real-weight W8A16 projection run.
It has no persistent state and does not include O projection, residual or FFN.

`AttentionSoftmaxEmitter` schedules SXM capture from the final VXM multiply's
output cycle and the target SR route latency. It preserves the existing BF16
cast, uses stream 36/37 for SXM output, and requires output MEM queues to be
disjoint from score/exp/mask reads. The next head's VXM work can overlap the
previous SXM drain. Whole-layer and unsupported layouts keep the materialized
fallback. The compiler omits the unused probability-pack binding and the
separate probability-transpose phase when this path is selected.

| Measurement | Before streaming | After streaming |
| --- | ---: | ---: |
| Initial C2C input cycles | 16,394 | 16,394 |
| Device execution window | 18,209 | 17,275 |
| Runtime weight wait | 0 | 0 |
| Context output readback | 32,808 | 32,808 |
| Total runtime cycles | 67,411 | 66,477 |
| Total at 500 MHz | 0.134822 ms | 0.132954 ms |
| MEM Read issues, both hemispheres | 65,408 | 63,360 |
| MEM Write issues, both hemispheres | 22,656 | 18,560 |

Total runtime improves 1.39%, and the device window improves 5.13%. MEM counts
are actual runtime FU issue events, not ICU instruction counts. WriteTap and
WriteSync counts are unchanged. All 32 head windows have direct VXM/SXM
timelines; the actual P transpose spans cycles 22,724--27,041 across both
hemispheres, overlapping Softmax rather than following it as a separate phase.

Output is bit-identical to the prior implementation: core relative L2 remains
2.91064%, and relative L2 against attention on the independent projection
reference remains 2.92964%, below the unchanged aggregate 3% limits. FP32
arithmetic/accumulation and BF16 storage are unchanged. The baseline was
replayed with the same new runtime and CModel `e42e41d` before comparison.
The new module-level SXM matrix reference and RegisteredR1 MEM mode are not used.

The seq32 synthetic causal/GQA case and B2 decode regression with a padded
64-key window pass, including negative logits, causal offsets, explicit scale
and expected rejection of unsupported unaligned positions. Dense/Flash real
and synthetic physical-iMEM audits pass; no compute reload is needed and peak
active FU-3D contexts per queue is one. Flash is unchanged at 163,312 cycles.
There are no separate diagnostic tensor readbacks in the core run; iMEM image
loading/reloading transfer latency is not modeled.

Reproduce with the recorded compiler/runtime snapshot:

```powershell
python -B compiler/tests/glm4_prefill_attention_test.py --build-dir tmp/glm4_prefill_p_stream_build
python -B compiler/tests/glm4_prefill_attention_test.py --build-dir tmp/glm4_prefill_p_stream_build --synthetic --output-dir results/glm4_9b_prefill_attention_comparison/synthetic
python -B compiler/tests/dense_attention_core_test.py --build-dir tmp/glm4_prefill_p_stream_build --target-config ../FTLPU-CMODEL/config/ftlpu-lpu32.json --output-dir results/dense_attention_core
python -B tmp/glm4_prefill_p_stream_build/verify.py
```

Compiler SHA: `4cf1c6934cc14d1b285d2ce92a7c1da75a8393d03885072ae600a91afc9a7bd5`.
`tmp/glm4_prefill_p_stream_build/build_manifest.json` records the frozen base,
source/object overlays, runtime and diagnostic hashes. `probability_streaming.json`
records the same-input/same-runtime comparison; `dense/runtime.pipeline.csv`,
`dense/actual.runtime.ftlpu` and `dense/icu_programs/index.csv` are the actual
executed artifacts. Each phase's `compute_imem_segments.json` binds its formal
capacity report to its executed program and trace hashes.

## Dense prefill transport and scheduling optimization (2026-11-11)

This earlier checkpoint added direct C2C ingress and packed context output;
the current `dense/` result also includes P streaming described above. The exact prior results
and graph plan are preserved as `dense/before_tight.result.json` and
`dense/before_tight.graph_plan.json`; `optimization.json` records the comparison.

The compiler maps each external Q/K/V word into its native MEM address and
emits ordinary planar C2C bindings for contiguous address runs. No activation
arithmetic is moved to the host. This direct upload applies to an isolated
single native core, whose input lifetime starts at invocation ingress;
multi-request graphs retain their existing lifetime-safe staging path.
When the context fits one planar byte pair within the allocator's half-bank
budget, PV writes `(head * head_blocks + block) * tokens + token` directly.
That is the ordinary output layout, so no final layout-gather program is needed.

For wide SXM, a single-query-tile/single-KV-tile native core now prepares the
next V block while the current block executes. The scheduler derives the
load lead from SXM/MXM route latency, tracks both weight buffers' release,
and reserves both hemispheres' mirrored output write windows. The optimization
requires disjoint input/output MEM queues when their banks coincide. It does
not interleave different ICU instructions on one queue or relax FU timing.

| Measurement | Before | After |
| --- | ---: | ---: |
| Linker MEM-to-MEM domains | 65 | 0 |
| Vectors copied by those domains | 12,320 | 0 |
| Initial C2C ingress cycles | 14,166 | 16,394 |
| Device execution window | 41,010 | 18,209 |
| Output readback cycles | 32,809 | 32,808 |
| Total runtime cycles | 87,985 | 67,411 |
| Runtime at 500 MHz | 0.175970 ms | 0.134822 ms |
| PV stage cycles | 14,061 | 6,507 |

Total latency falls 23.38%; the device execution window falls 55.60%.
Direct ingress has more small C2C bindings, increasing its measured overhead;
the table includes that cost. Each MXM still has 4,096 active Compute cycles,
so its active fraction of the device window rises from 9.99% to 22.49%.
These are execution-window fractions, not fractions of total wall time.

The final context is bit-identical to the pre-optimization result (relative
L2 2.91064% against actual-QKV attention, 2.92964% from the independent
projection reference). The remaining softmax score mirror between hemispheres,
reduction passes and probability transpose still execute; zero linker copies
does not mean every MEM read/write in the attention algorithm disappeared.
Both real seq32 variants, the synthetic causal seq32 pair and the existing
batch-2 decode/masking/scale regressions pass. iMEM/transport capacity remains
checked against the actual programs. Instruction loading is still not timed.

Reproduce with the preserved device QKV fixture:

```powershell
python -B compiler/tests/glm4_prefill_attention_test.py --build-dir tmp/glm4_dense_tight_build
```

The build snapshot freezes the base compiler library and the changed C++
translation units; its source and tool hashes are in
`tmp/glm4_dense_tight_build/build_manifest.json`. The projection fixture retains
its original compiler provenance; both new attention variants use the same
new compiler and the same unchanged runner, target and input data. A source
build can be passed through `--build-dir` in the same command.

## Dense versus Flash seq32 attention (2026-11-11)

The measurements in this section are the initial comparison, before the
dense transport optimization above. Current files use the optimized dense
program; the initial measurements are retained in `before_tight.*` artifacts.

The optional Flash prefill test uses the existing C++ FlashAttention
lowering and real raw-macro execution. The ordinary comparison uses the
shared dense attention core, including its QK/softmax/probability transpose/PV
emitters. No model operations were added to CModel for this test.

The fixed result directory is
`results/glm4_9b_prefill_attention_comparison`:

- `projections/`: input RMSNorm, QKV projections and biases, partial RoPE at
  positions 0 through 31, all executed on the device using real W8A16 weights.
  This is a fresh prefill fixture, not decode batch32 at a shared position.
- `dense/` and `flash/`: independently compiled and executed attention cores
  consuming exactly the same device-produced Q/K/V. Each includes
  `runtime.pipeline.csv`, `actual.runtime.ftlpu`, `icu_programs/index.csv`,
  `result.json` and `compute_imem_segments.json`.
- `comparison.json`: numerical, build, input and timing provenance.

The core has Q `[1,32,32,128]`, K/V `[1,32,2,128]`, causal position zero and
context `[1,32,32,128]` (exported as `[32,4096]`). There is no prior cache or
persistent-state update inside this invocation. This test ends at context:
it does not execute O projection, residuals or FFN, and does not replace the
full-decoder benchmark above. The two-stage projection-plus-core measurement
crosses DDR between stages; it is not a fused continuous attention program.

Reproduce from the repository root, with the imported official checkpoint:

```powershell
python -B compiler/tests/glm4_prefill_fixture.py --build-dir tmp/glm4_dense_build
python -B compiler/tests/glm4_prefill_attention_test.py --build-dir tmp/glm4_dense_build
```

The measured tool snapshot is `tmp/glm4_dense_build`, compiler SHA
`7335eca6c0637255ce3de0ac15216fb9d86f0e1633e031f86cc3c73bb84de73b`.
For a new source build, pass its build directory to both commands. The test
also accepts a preserved QKV fixture from a previous compiler/runtime, records
that producer's provenance separately, and rejects a different hardware target.
It does not attribute the reused projection run to the current core runtime.
The fixture loader also validates the NPZ and actual output hashes.

Storage remains BF16 for Q/K/V, scores and context, with FP32 MXM
accumulation. Dense stores normalized probabilities in BF16 before PV.
Flash stores unnormalized `exp(score-max)` in BF16 before PV, then divides
by the FP32 denominator and writes BF16 context. Each implementation is
checked independently against float64 causal attention on its actual QKV,
and against attention on an independent W8A16 projection reference. Both
aggregate relative-L2 limits remain 3%; the test does not require the two
implementations to be bit-identical.

At 500 MHz, nominal DDR 51.2 GB/s and configured scheduling efficiency 90%:

| Path | Runtime cycles | Runtime ms | Core relative L2 | RMSNorm-to-context relative L2 |
| --- | ---: | ---: | ---: | ---: |
| Dense attention core | 87,985 | 0.175970 | 2.91064% | 2.92964% |
| Flash attention core | 163,312 | 0.326624 | 2.90792% | 2.92671% |

Both pass. Token zero reproduces its own V exactly, as required by the
causal mask. Worst token relative-L2 is 4.1205% / 4.0838%, and worst head
relative-L2 is 7.0527% / 7.0430% for dense / Flash; the unchanged acceptance
criterion is aggregate error, not a 3% guarantee on every token or head.
Q/K/V fixture errors are 0.16984% / 0.15008% / 0.17862%. The synthetic seq32
causal GQA regression also passes with 0.1613% / 0.1578% core error.

| Path | Initial ingress | Device execution window | Weight wait within execution | Output readback |
| --- | ---: | ---: | ---: | ---: |
| RMSNorm/QKV/bias/RoPE | 25,676 | 412,469 | 16,959 | 22,897 |
| Dense core | 14,166 | 41,010 | 0 | 32,809 |
| Flash core | 22,059 | 120,734 | 0 | 20,519 |

All entries are measured cycles. Projection uses ten real weight pages
(18,874,368 bytes); its total is 461,042 cycles. The sum of projection and
core invocations is 549,027 cycles (1.098054 ms) for dense and 624,354 cycles
(1.248708 ms) for Flash. Readback is QKV for the projection fixture and
context only for each core; there are no separate core diagnostic outputs.
Instruction loading latency is not timed. Formal physical-iMEM and transport
checks pass for all three programs, with no compute-iMEM reload required.

Flash takes **1.856 times** the ordinary core runtime in this short-sequence
case. There is just one 32-key tile, so no cross-tile online merge is needed.
The current Flash mapping flattens 32 tokens times 16 query heads per KV head
into 512 query rows, scheduled as 16 tiles; the two KV groups occupy the two
hemispheres. The query-tile layout and scheduling work outweigh the benefit
of avoiding a larger score/probability tensor here. This result does not
establish the crossover point at longer sequences, and Flash remains an
explicit option rather than the seq32 default.

## Validated run (2026-11-04)

The layer-0 seq32 CModel run passed, with raw macro instructions, the
unmodified `ftlpu-lpu32.json` capacity and real DDR/C2C paging. The scheduled
execution limit was 3,306,382 cycles; the runtime trace spans 0–5,522,514,
including input/output and transport waits. This is a correctness baseline,
not a fully optimized prefetch schedule or a full-model TTFT measurement.

| Comparison | Max absolute | Mean absolute | Relative L2 | Cosine |
| --- | ---: | ---: | ---: | ---: |
| CModel vs W8A16 reference | 0.0078125 | 0.0000524616 | 0.0119750 | 0.99992944 |
| CModel vs native BF16-weight reference | 0.017578125 | 0.000482237 | 0.0810038 | 0.99675837 |
| W8A16 reference vs native BF16-weight reference | 0.021484375 | 0.000482159 | 0.0811197 | 0.99673646 |

The native reference is independent NumPy checkpoint math, not an official
Transformers execution. Its ChatGLM RMSNorm keeps the normalized values in
FP32 through gamma multiplication; attention scores apply the scale before
BF16 rounding. This comparison includes quantization and arithmetic
differences. The hardware acceptance comparison is the first row.

`result.json` records the metrics, package/runtime hashes and pipeline hash;
`runner.json` records the completed invocation and ICU export. The actual
trace is `runtime.pipeline.csv` (MEM, MXM, VXM, SXM, C2C and waits), with a
standalone `pipeline.html`. `icu_programs/index.csv` indexes the per-queue
programs exported from the runtime-linked `linked.ftlpu`; use the runtime
pipeline for actual issue cycles and dynamic transport behavior.

The model importer, full-size tensor layout, partial-RoPE bias packing,
existing six-family importer regression and 15 model-runner unit tests
passed. The saved numerical output was also rechecked with `--verify-only`.
