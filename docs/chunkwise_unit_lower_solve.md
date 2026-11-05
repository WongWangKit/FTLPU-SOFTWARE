# FP32 chunk32 unit-lower solve

The incremental Qwen3.5 GDN prefill test can stop after solving `(I + L) D = B`, with `B = beta * V` and zero incoming State. This does not yet implement attention output, the outgoing State, or nonzero-State RHS construction.

## Compiler path

`ftlpu.kernel.vector_alu` has two target primitives:

- `head_scale`: BF16 V `[32,H*D]`, FP32 beta `[32,H]` -> FP32 B `[H*32,D]`.
- `unit_lower_solve`: FP32 strictly lower L `[H*32,32]`, FP32 B `[H*32,D]` -> FP32 D `[H*32,D]`.

Both require even H <= 32 and a value width divisible by 32. Native output tensors group even heads before odd heads. The graph planner passes producer addresses to the C++ schedule emitter. Python builds the logical graph, imports weights, checks independent numerical references and exports diagnostics; it does not emit the schedule or evaluate a runtime intermediate.

V stays in its Conv distributed16 home. L and B retain their native planar layouts, with the producer homes allocated on separate MEM ICU queues so coefficient reads can overlap B reads. For forward substitution, wide SXM broadcasts L rows using separate low/high byte-pair tiles for FP32. Two banks of 32 FP32 vectors per hemisphere hold current and prefetched coefficients; no whole-L broadcast table is materialized. For L, the SXM output crosses the passive SR bridge to align with B's head ownership before the coefficient table write.

## Streaming beta times V

For `head_scale`, East SXM transposes the low two FP32 byte planes and West SXM transposes the high two. Each result is tapped into local MEM and continues over passive SR to the opposite hemisphere. This avoids doing the same complete FP32 broadcast on both sides. Both hemispheres reconstruct full FP32 beta; no precision is discarded and no ALU bypass copy is added.

The coefficient table uses slices 24-27, alternating banks by token. Local and bridged writes use separate byte-plane slices; consumption and the next token's production use separate bank ICU queues. The compiler accounts for the SR flight times and bank reuse when choosing the token initiation interval. V reads retain their producer addresses and issue ahead of VXM. Each head's feature blocks use an affine READ3D domain, and coefficient reads/B writes use head/block domains.

For H=16 and D=128, the token interval and work per hemisphere are both 32 cycles. After startup, C0/C8 multiply continuously for all 32 tokens (1024 cycles each); C1/C9 complete the depth-2 result chain. There are no per-head drains or interior iMEM reload points. Smaller workloads may remain SXM-throughput limited; startup and final transport drain are still included in total stage latency.

`compiler/tests/head_scale_streaming_runtime_test.py` checks exact FP32 products with varying low mantissa bits across every token/head, repeated bank reuse, and actual runtime issue coverage on both hemispheres.

The real seq32/H16/D128 test reduced the beta-times-V stage from 13,945 to 1,125 cycles. Runtime C0/C1 (paired with physical C8/C9) are continuously active over `[232068,233092)`, 1,024 cycles without an interior bubble. The complete stage, including beta startup and final B writeback, is `[231983,233108)`. B remains exact against the actual beta/V inputs; the downstream forward-solve relative residual remains `5.38133e-8`. These ranges describe the saved forward-solve run, not a fixed compiler ABI.

## Forward substitution

For row i, each 32-lane word represents 32 value channels. The ALU chain is:

- C0: pass B by multiplying by one, then multiply each `L[i,j] * D[j]`.
- C1: pass the initial B term, then negate the products.
- C2: forward the chain to the accumulator-capable stage.
- C3: FP32 Add/ACC, reset on B, emit only after the last product.

C8-C11 execute the mirrored instruction stream for different heads. The initial row keeps every FP32 bit. Partials never spill to MEM. The fixed VXM return direction places completed D on the opposite fabric; each completed row except the last is returned once via passive MEM/SR to a 32-row feedback workspace. This is a hardware routing requirement, not ALU bypass arithmetic or an expanded partial-sum archive.

The next L row is prefetched into the alternate coefficient bank while the current row computes. Bank reuse waits for the preceding complete read domain to retire. Independent value blocks use initiation interval `max(i+1, 9)`: nine cycles cover the three accumulator configurations' fetch bandwidth, while longer dots run back-to-back. D feedback uses passive streams W28/E28 through W31/E31, separate from the overlapping SXM result/bridge streams W24/E24 and W25/E25. A write-to-read transition preserves the target's complete tile commit window (four cycles on LPU32).

The compiler declares iMEM reload boundaries after groups of rows and their prefetched/feedback streams have drained. For D128 the group is eight rows; larger value widths or smaller iMEM reduce the group size. Local iMEM capacities are unchanged. All execution uses raw macro ICU instructions.

In the real seq32/H16/D128 forward run, the tightened solve occupies runtime
`[233318,269270)`: 35,952 cycles versus 103,722 before this scheduling change
(65.34% fewer). The paired C0 multiply queue performs the same 16,896 useful
cycles; its active fraction over the solve stage rises from 16.29% to 47.00%.
This is not total utilization of all VXM ALUs. Rows i>=8 have no idle cycles
between their value blocks; row dependencies, feedback transport, short-row
coefficient broadcasts and finite-iMEM reload boundaries still leave gaps.
The relative equation residual remains `5.38133e-8`, and worst-head end-to-end
NRMSE remains 1.77934%. The saved summary includes an actual-issue audit.

## Validation

### RHS layout after nonzero-State correction

The multi-chunk path forms `B = beta * V - beta * gamma * (K @ S_in)`.
Its FP32 ALU result can use mirrored `fp32_quad_planar` storage instead of
the native west-first partitioned `head_scale` result. L's coefficient stream
crosses to the opposite hemisphere before the solve. For mirrored B, the
consumer on physical side `side` must therefore read logical head
`(1-side) * (H/2) + pair`; partitioned B already encodes this ownership.
This is an address selection, with no extra transpose or MEM-to-MEM copy.

`unit_lower_solve_runtime_test.py --heads 4 --dim 128 --produced-rhs`
exercises that producer/consumer boundary, checks the FP32 producer layout,
compares every result to an FP64 solve and verifies `D[head,0] == B[head,0]`
exactly. CTest registers it as `unit_lower_solve_produced_rhs_runtime_test`.

The scalar views in this correction also require real lane broadcast:
`gamma[t,h]` and `beta[t,h]*gamma[t,h]` are packed `[32,H]` tensors, not
already-broadcast vectors. The graph linker recognizes constant-index groups
of 32 destination lanes and emits wide-SXM broadcast tiles, using separate
low/high byte-pair passes for FP32. Whole-vector address copies cannot select
the nonzero head lanes.

For a repeated view of a single producer row, such as `gamma[-1,h]` in the
outgoing State decay, the linker retains one compact `[H,32]` FP32 table per
producer/output/row. Wide SXM broadcasts the packed row once (two byte-pair
passes per hemisphere). Both 1024-row State subblocks share this table.
READ3D repeats each head address for 128 key rows, advances by two table rows
for grouped even/odd heads, and reuses those addresses for all four 32-channel
value tiles. The two subblocks begin at table offsets 0 and 1. No expanded
`[H*128,128]` coefficient tensor or intermediate VXM copy is emitted.
At H=16 the retained table is 2048 bytes per hemisphere, versus a 1 MiB
expanded coefficient tensor. Its allocation remains live through the last
consumer. This also preserves the single-row decode compact broadcast path.

`packed_scalar_broadcast_runtime_test` checks FP32 results bit-for-bit,
including grouped State subblocks sharing a table, distinct rows of one
producer requiring distinct tables, and the decode case. It uses raw macro
ICU instructions; repeated addresses are native READ3D domains.

Real-weight seq64 raw-macro acceptance (2026-11-05) retains all six outputs
bit-for-bit versus the corrected expanded-broadcast baseline. Compute cycles
decrease from 1,505,040 to 1,243,184; actual SXM transpose events per hemisphere
decrease from 4,804 to 2,758. The two former 131,072-cycle gamma expansions are
replaced by one 274-cycle compact-table materialization. Runtime weight-page
waits remain 26,954 cycles and host arithmetic operations remain zero. Artifacts
and the comparison report are in
`results/qwen3_5_0_8b_prefill_chunkwise_attention_seq64/`, including
`compact_gamma_comparison.json`, `runtime.pipeline.csv`, and
`runtime_icu_programs/` exported from the actual runtime program.

The standalone test checks the full numerical result against an FP64 triangular solve, its relative equation residual, and exact FP32 preservation of row zero. It includes values whose low mantissa bits would disappear with BF16 reloads. Cases include H2/D32 and H4/D128; the latter exercises compute iMEM reload. It also checks the exact number of actual multiply issues and gap-free value blocks for rows with at least nine terms.

The real-weight `--with-solve` test checks B, the solve against actual L/B, and the final D against the independent model reference. It runs from projection through Conv/SiLU, Q/K L2Norm, beta/g, G, gamma, weighted Gram L and forward substitution.

The merged legacy overlap verifier does not support chain VXM. This test explicitly sets `ftlpu.common_transport_overlap = false`: common barriers and DDR/C2C paging remain active, but old common-protocol transport overlap is disabled. QCTL overlap performance is a separate integration step; the test must not be presented as overlapped paging performance.

Run (from the software repository):

```powershell
python compiler/tests/qwen3_5_prefill_qkv_wide_sxm_test.py --package results/qwen3_5_0_8b_decoder_layer_prefill/package --build-dir build-ftlpu-vs2026-direct --target-config ../FTLPU-CMODEL/config/ftlpu-lpu32.json --output-dir results/qwen3_5_0_8b_prefill_chunkwise_solve --sxm-architecture wide --with-solve --solve-strategy forward
```

## MXM polynomial strategy

`--with-solve --solve-strategy mxm` selects the matrix path; `--solve-strategy forward`
retains the FP32 forward substitution described above. The default solve strategy is
now `mxm`. Use `--solve-strategy forward` with the command above to reproduce its
original result.

The compiler adds `unit_lower_inverse(L, I)` and `head_matmul(T, B)`. Both use
FP32 planar matrices, with an even number of head partitions. The identity operand
is an immutable input constant. Neither Python nor CModel computes an inverse or
any dynamic matrix intermediate on the host.

For a strictly lower chunk32 L, the emitted algorithm is:

```
T = I - L
P = L @ L
repeat four times:
    T = T + P @ T
    P = P @ P  # omit after the fourth update
D = T @ B
```

There are eight square matrix products to form T and one rectangular product to
form D. The identity is exact in exact arithmetic because L^32 = 0. Floating-point
execution is not exact, and a naive single BF16 product is insufficient for this
fixture: large powers of L cancel when forming T.

The MXM path therefore splits each FP32 operand into three BF16 components:
`x0 = trunc_bf16(x)`, `x1 = trunc_bf16(x-x0)`,
`x2 = trunc_bf16(x-x0-x1)`. It accumulates the six products with component-index sum
at most two. Terms x1*y2, x2*y1 and x2*y2 are omitted; this is a compensated
mixed-precision approximation, not a native FP32 multiplier. Small components are
processed before the final x0*y0 term. All partial products accumulate in MXM
FP32 ACC, with a single final output. FP32 VXM performs residual splitting,
`I-L`, and the T updates.

Each 32x32 product uses three wide-SXM transposes and three 1vector IW loads;
each loaded weight component is reused across its matching activation components.
T and P use two FP32 workspaces each. Their ownership alternates between the
hemispheres after each update, avoiding a whole-matrix return copy. Residual
components do require a bounded scratch return through passive SR because VXM's
fixed result fabric is opposite its input fabric. One BF16 transposed weight tile
per hemisphere is staged for MXM IW. There is no partial-sum MEM spill, and no
claim that this strategy removes every scratch transfer.

The final T is already in B's native head partition, and D returns to the native
output partition. Compiler iMEM reload boundaries occur only after complete
matrix products have drained. Local iMEM capacities are unchanged. A change to
native operand head ownership is not silently aliased by the planner.

The test validates T against an FP64 polynomial inverse, D against an independent
FP64 triangular solve, `(I+L)D-B`, and the end-to-end D from model reference
activations. The real test still assumes zero incoming State, so B = beta*V.
It stops at D; attention output and State update are not included.

```
python compiler/tests/qwen3_5_prefill_qkv_wide_sxm_test.py --package results/qwen3_5_0_8b_decoder_layer_prefill/package --build-dir build-ftlpu-vs2026-direct --target-config ../FTLPU-CMODEL/config/ftlpu-lpu32.json --output-dir results/qwen3_5_0_8b_prefill_chunkwise_solve_mxm --sxm-architecture wide --with-solve --solve-strategy mxm
```

The standalone `compiler/tests/unit_lower_mxm_runtime_test.py` also accepts
`--fixture-dir results/qwen3_5_0_8b_prefill_chunkwise_solve --heads 16 --dim 128`
to consume previously verified real L and B as test inputs. This isolates solver
error from upstream model approximation; it is not the integrated prefill test.

### Pipelined RHS integration

Head scaling broadcasts the low and high FP32 byte pairs on different SXMs,
then taps the local result and lets the same SR stream continue to the other
hemisphere. Both coefficient banks are reserved in the graph memory plan.
Beta's producer and external beta inputs are allocated on MEM ICU queues distinct
from the concurrent V reads. The standalone head-scale test checks an exact FP32
product for distinct values across all heads and tokens. These reservations are
required for overlapped broadcasting; different SRAM rows alone do not resolve
instruction-queue conflicts.

### Verified real-weight result (2026-11-03)

Qwen3.5-0.8B, layer 0, seq32, 16 heads, D=128, raw macro execution:

| Stage | Cycles | Runtime interval, end exclusive |
| --- | ---: | --- |
| B = beta*V | 1,125 | 232885–234010 |
| T = inverse(I+L) | 75,517 | 234220–309737 |
| D = T*B | 42,121 | 309952–352073 |

The worst per-head D NRMSE against the solve using actual L/B is 0.07528%;
the relative equation residual is 0.01739%. Worst per-head end-to-end NRMSE
against the model reference is 1.78315%. Both MXMs execute 18,432 active compute
cycles in the solve. The inverse and final product total 117,638 scheduled
operator cycles (117,853 including the inter-operator gap), compared with the
previous FP32 forward solver's 103,722 cycles. This version establishes numerical
feasibility; it does not demonstrate a speedup.

Artifacts: `results/qwen3_5_0_8b_prefill_chunkwise_solve_mxm/summary.json`,
`runtime.pipeline.csv`, `actual.runtime.ftlpu`, and `runtime_icu_programs/`.
Pipeline and ICU exports come from actual CModel execution, including initial
C2C ingress. Host diagnostic readback is excluded from operator cycle counts.

## Block16 forward strategy

`--with-solve --solve-strategy blocked` selects `unit_lower_solve_blocked(L,B)`
in the same logical graph and memory planner. The existing `forward` and `mxm`
strategies remain available. No inverse or matrix powers are formed:

```
solve (I + L00) D0 = B0         # 16 rows, VXM FP32
P = BF16(L10) @ BF16(D0)        # MXM BF16 inputs, FP32 ACC/output
solve (I + L11) D1 = B1 - P     # 16 rows, VXM FP32
```

The diagonal solves reuse the shared forward emitter and its double-buffered
FP32 SXM coefficient broadcast. Each VXM vector covers 32 value channels; the
16-token block size does not introduce a narrower vector operation. Head pairs
execute on both hemispheres. MXM loads weights in `1vector` mode and performs
one product per 32-value-channel tile, without BF16 component decomposition.
Reading the upper two FP32 byte planes implements BF16 truncation.

D0 is transposed directly from its final output address. The future D1 output
rows are explicitly zero-initialized by VXM, providing K=32 padding until the
real D1 overwrites them. A single transposed BF16 weight tile is staged for MXM
IW; no copy of the complete D0 matrix is made. The 16xD cross-block product P
lands on B's owning hemisphere through passive SR. Each D1 row subtracts its P
row in the first VXM operation and immediately accumulates its diagonal-block
terms, so an updated B matrix is never written to MEM. Completed diagonal rows
still need the existing bounded passive feedback return.

Multiply has a two-cycle latency whereas Subtract has one. Consecutive D1 value
tiles leave one ingress cycle between the final Multiply and next Subtract to
avoid a real ALU output collision. The first D1 row's MEM write also accounts
for its shorter chain latency. Compiler drain boundaries retain finite ICU
iMEM capacity; there is no queue interleaving or host solver fallback.

`compiler/tests/unit_lower_blocked_runtime_test.py` checks the hardware result
against both an FP64 triangular solve and an independent mixed-precision CPU
oracle, including the FP32-only first block and both MXM runtime issue counts.
The real-weight test checks each head against actual L/B and the model reference.
The local mixed solve error/residual limit is 1%; the end-to-end per-head limit
remains 3%. Neither iterative refinement nor nonzero incoming State is included.

```
python compiler/tests/qwen3_5_prefill_qkv_wide_sxm_test.py --package results/qwen3_5_0_8b_decoder_layer_prefill/package --build-dir build-ftlpu-vs2026-direct --target-config ../FTLPU-CMODEL/config/ftlpu-lpu32.json --output-dir results/qwen3_5_0_8b_prefill_chunkwise_solve_blocked --sxm-architecture wide --with-solve --solve-strategy blocked
```

### Verified block16 result (2026-11-03)

Real layer-0 weights, seq32, H=16, D=128, wide SXM, raw macro CModel:

| Metric | Result |
| --- | ---: |
| Solver runtime interval (end exclusive) | 233328–269659 |
| Solver cycles | 36,331 |
| Current tightened FP32 forward solver cycles | 35,952 |
| Global D NRMSE using actual L/B | 0.22243% |
| Worst-head D NRMSE using actual L/B | 0.45864% |
| Relative equation residual | 0.26241% |
| Worst-head end-to-end D NRMSE | 1.78643% |
| MXM active compute cycles, per hemisphere | 512 |

The FP32 forward baseline uses byte-identical actual L and B. Block16 is 1.05%
slower than that tightened baseline, despite reducing the diagonal arithmetic.
This result does not establish a speedup; transposing/loading the small padded
MXM products and the coefficient/feedback schedule still dominate. The older
103,722-cycle forward figure above predates the tightened forward schedule and
is not the appropriate baseline for this comparison.

The 2-head/D32 and 4-head/D128 independent fixtures also pass, matching the mixed
CPU arithmetic oracle exactly. The original FP32 forward test passes as a
regression. No CModel arithmetic changes were required. The integrated result has
zero linker MEM transports and zero native output archive copies; internal
coefficient tables, the weight tile, cross-block product and row feedback remain.

Artifacts are under `results/qwen3_5_0_8b_prefill_chunkwise_solve_blocked/`:
`summary.json`, `runtime.pipeline.csv`, `actual.runtime.ftlpu`, and
`runtime_icu_programs/` (232 ICU CSVs plus `index.csv`). The pipeline includes
actual initial C2C transfers. The test still stops at D with zero incoming State;
it does not execute attention output or State update.
