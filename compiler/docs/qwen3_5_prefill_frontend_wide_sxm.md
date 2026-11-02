# Qwen3.5 prefill frontend on wide SXM

The incremental GDN prefill baseline contains only QKV projection followed by
causal depthwise Conv1D (kernel 4) and SiLU. It uses the imported Qwen3.5-0.8B
layer-0 weights, a 32x1024 BF16 input and a 32x6144 output. Convolution history
starts at zero. The input is the raw fixture, without host-side RMSNorm.
RMSNorm, Z/beta/g projections, Q/K normalization, delta recurrence, output
normalization/projection, residuals and FFN are outside this test.

Run from FTLPU-SOFTWARE:

```powershell
python compiler/tests/qwen3_5_prefill_qkv_wide_sxm_test.py --package results/qwen3_5_0_8b_decoder_layer_prefill/package --build-dir build-ftlpu-vs2026-direct --target-config ../FTLPU-CMODEL/config/ftlpu-lpu32.json --output-dir results/qwen3_5_0_8b_qkv_conv_prefill
```

The default `--sxm-architecture wide` selects the new 32x32 streaming transpose
hardware in CModel. `--sxm-architecture local` preserves the old selection.
This graph requires no transpose: projection results are consumed through
native address views. A zero SXM operation count is therefore intentional;
it verifies compatibility but does not validate the new transpose datapath.
The separate CModel `sxm_wide_transpose_test` covers that datapath and timing.

Python constructs semantic graph nodes and independent numerical references.
The shared C++ graph compiler performs layout, paging and raw-macro lowering.
The default test exports only Conv/SiLU and checks the complete path against an
FP32 reference. `--export-qkv` additionally checks QKV and Conv given actual
BF16 QKV, but adds diagnostic intermediate export traffic; leave it disabled
for the normal pipeline view. Actual executed ICU programs are exported from `actual.runtime.ftlpu`
to `runtime_icu_programs`, including pre-execution programs.

The existing results directory is overwritten on each run. `--validate-only`
rechecks saved outputs without recompiling or executing CModel.

## Q/K normalization increment

Add `--with-qk-norm` and use output directory
`results/qwen3_5_0_8b_qkv_conv_qk_norm_prefill` to append per-head L2Norm.
Outputs are Q, K and unchanged V; intermediate QKV/Conv exports are omitted.
Each 128-element head uses `x / sqrt(sum(x*x) + 1e-6)`; Q additionally scales
by `1/sqrt(128)`. There is no mean reduction or gamma multiplication.

The wide configuration sets compiler attribute `ftlpu.sxm_wide` as well as
selecting the CModel architecture. Normalization transposes serialize the
distributed row pairs into a two-byte-stream 32x32 SXM tile and scatter columns
directly into the feedback layout. This replaces local transpose/permute
instructions, retaining the shared VXM feedback reduction and scale path.

The wide Q/K test enables `ftlpu.l2_head_partition`. The graph compiler packs
projection weight columns so that all four 32-feature words of a 128-feature
head land in the same hemisphere (alternating complete heads). It inversely
maps the logical output view and selects the matching Conv coefficient rows.
These are compile-time views/weight packing, not runtime activation copies.
Conv reads projection addresses directly and writes complete distributed16
heads. Two different heads then use the mirrored VXM chains concurrently.

Conv source data and the SXM retention buffer occupy different MEM banks.
Each hemisphere's SXM launches four 32x32 tiles at II=32. MEM `write_tap`
retains the transposed data for normalization without consuming the stream;
VXM squares and accumulates the same arriving vectors without a reduction
MEM reload. No missing-head gather or pre-normalization mirror remains.

The fixed VXM output route reaches the opposite MEM hemisphere. Independent
heads therefore require returning the two-byte sum scalar per token before
computing the inverse norm. The final feature result stays on its native
opposite side through the output transpose. Terminal outputs use the generic
four-word-group partitioned distributed16 binding; the runtime/C2C reader
selects each group's actual hemisphere. No output mirror is generated.
Nonterminal consumers retain the old mirrored contract until their own
physical input plan accepts a partitioned value.

L2Norm still needs retained input: the entire head's sum must be known before
multiplication by its inverse norm. This uses sum, not RMS mean, with epsilon
1e-6, no gamma, and a separate Q scale of 1/sqrt(128).

`projection_conv_direct_view_runtime_test.py --wide-sxm` checks both numerical
results and paired-head streaming markers. Its default local-SXM path keeps
the prior layout. The real-weight test checks Q, K and V independently and
exports the actually executed raw-macro ICU programs, including pre-execution.

## Beta and log-decay increment

`--with-gates` implies `--with-qk-norm` and appends the checkpoint's
`in_proj_b`/`in_proj_a`, sigmoid beta and log-decay g. These projections read
the same raw input fixture as QKV. The test has no host activation evaluation:
only immutable `-exp(A_log)` is folded. Its five outputs are Q, K, V, beta, g.
Weights remain dynamically paged and execution uses raw-macro ICU programs.

The opt-in module policy `ftlpu.vector_log_softplus` enables VXM Log lowering.
Kernel, schedule, command and binary opcode mappings support Log; the runtime
initializes its standard 256-segment mantissa LUT. Log is placed in the same
ALU positions as reciprocal/rsqrt and carries the same special-op latency.
The default polynomial Softplus remains available for other callers.

A naive `log(1+exp(-abs(x)))` loses small tails at the FP16 LUT ingress.
For t=exp(-abs(x)), a=1/16 and u=min(t,a), the new path computes:

    max(x,0) + log(1+max(t,a)) - log(1+a) + u-u*u/2+u*u*u/3

The subtracted baseline matches the standard LUT's FP16 stored coefficient,
and subtraction stays inside its ALU chain. This avoids a fixed small-tail
bias; the cubic's real-arithmetic truncation is below 4e-6. Runtime arithmetic
still includes FP16 LUT quantization and BF16 spill boundaries. Very negative
inputs retain the existing Exp FP16 flush-to-zero limit. No model operation is
implemented inside CModel.

`chain_vxm_gates_runtime_test.py --log-softplus` checks actual Log execution,
whole-gate error and negative-tail error (zero bias isolates the nonlinear
from bias-spill rounding). The real-weight frontend test covers checkpoint
biases and verifies both beta and g independently. Gate intermediates use the
existing packed vector storage; this step does not add GDN recurrence.

Validated seq32 real-weight result: 291,867 runtime trace cycles (0.583734 ms
at 500 MHz), five outputs, five dynamic weight pages, zero host operations.
Beta NRMSE 0.00039625; g NRMSE 0.00742644. Q/K/V errors are unchanged from
the paired-head frontend. Elementwise gate checks additionally use beta
rtol=1%, atol=1e-4 and g rtol=5%, atol=1e-7; the absolute floor covers tiny
log-decay values below the existing Exp FTZ range.

## Distinct token halves for beta/g

The prefill gate test enables `ftlpu.vector_bf16_token_halves`. Each fixed VXM
chain runs 16 vectors per hemisphere with disjoint token halves, rather than
running all 32 tokens twice. VXM's native cross-hemisphere output reverses the
half ownership after a chain; lowering records this ownership for every value.
When a binary join needs differently oriented branches, passive SR fills only
the operand's missing half, which later consumers may reuse. Those joins are
not duplicate VXM computation. The implementation retains one such
alignment point in the Log Softplus graph. Commuting abs(x) to
max(negate(x), x) lets negate/max share one chain using Original, eliminating
the other spill and alignment.

Chain fusion may use `Original` for the original head operand, including the
short polynomial. This reduces spill boundaries without changing other users
of the default BF16 spill path. The narrow a/b projections directly scatter
their first live feature word into token-partitioned MEM; gates alias those
addresses. The passive MXM cross-hemisphere write timing uses the target route
latency without the obsolete extra cycle.

Terminal beta/g outputs are compact native token partitions. Generic runtime
bindings support both first-half-West and first-half-East orientations,
including C2C image packing/unpacking. No output mirror or join is emitted.
For nonterminal vector subgraphs the existing mirrored consumer contract is
preserved until those consumers opt into partitioned layouts.

Checks include the 32-to-16 repeat reduction, native projection/output layouts,
real-weight elementwise tolerances, disabled-policy regression, and BF16/F32
partition-layout roundtrips with odd row counts and nonunit address strides.


## MEM-to-MEM audit (2026-10-29)

The seq32 frontend now binds all five linear projection inputs directly to
one immutable C2C input home. Each emitter reads its actual bank/base/slices;
there is no copy back to the provisional bank-0 activation allocation. This
removes six passive SR transports per projection (30 total).

Terminal Q/K outputs use `fp16_mxm_block4_partition_distributed_16`: consecutive
four-vector column groups alternate West/East, retaining global row addresses.
Terminal V uses `fp16_mxm_compact_block4_partition_distributed_16`: groups
alternate East/West with compact local column addresses. The graph compiler
validates all element references before recognizing this export view. These
are generic external storage descriptions, not model operations in CModel.
Neither output export computes activations on the host.

`graph_plan.json.linker_mem_transports` audits graph-level transport domains;
the frontend regression requires it to be empty. Operator-internal transfers
are audited separately: L2 returns one two-byte vector per hemisphere and
head pair (2 KiB across Q and K), and the remaining gate join aligns 4 KiB of
padded FP32 byte planes. These are hardware routing dependencies in the present
fixed-port schedule, not layout archive buffers.

Conv's BF16 partial-result store/read feeds its separate SiLU chain. It is an
arithmetic spill: the MAC chain's Original is a tap input, not the accumulated
convolution sum needed by SiLU's final multiplication. L2 write-tap retains
input while SXM feeds the square/sum chain directly; normalization must wait
for the whole-head reduction. Its retained-input read and its genuine axis
transposes are therefore not counted as unchanged-data copy domains.

## Tokenwise prefill: stop at old-State decay (2026-10-30)

The two intended recurrence strategies are `tokenwise` (serial updated State
dependency) and `chunkwise` (block matrix products and triangular solve). The
existing chunkwise implementation is retained. The incremental tokenwise test
currently stops **before** the recurrence; it does not select or exercise the
complete tokenwise scan yet.

```powershell
python compiler/tests/qwen3_5_prefill_qkv_wide_sxm_test.py --with-state-decay --package results/qwen3_5_0_8b_decoder_layer_prefill/package --build-dir build-ftlpu-vs2026-direct --target-config ../FTLPU-CMODEL/config/ftlpu-lpu32.json --output-dir results/qwen3_5_0_8b_prefill_tokenwise_state_decay
```

This includes the existing real-weight seq32 QKV, Conv/SiLU, Q/K L2Norm and
beta/g frontend, then computes `eg_0 = exp(g[0])` and
`Sbar_0[h,k,v] = eg_0[h] * S_in[h,k,v]`. The old State is a deterministic
nonzero external fixture with logical shape `[16,128,128]`. Only token 0 is
processed: applying each token's decay to the same initial State would not
implement a recurrence. No KS, delta, State update, output contraction or FFN
is present.

State enters via runtime C2C directly into `[head*key,value]` planar homes.
VXM lanes represent 32 value coordinates. The shared wide-SXM broadcast
repeats the packed `[1,16]` coefficient vector 32 times, transposes it into
16 broadcast rows, and reuses those rows through MEM address repetition.
The two bounded eight-head State ranges share that coefficient table and
are exported separately, avoiding a contiguous State archive. The current
vector implementation mirrors State computation on the two hemispheres;
this is a correctness baseline, not independent head partitioning.

The generic graph planner now aliases a compatible single packed row of a
mirrored planar pointwise producer, so `g[0]` needs no staging copy. The test
requires an empty `linker_mem_transports` audit. The necessary SXM coefficient
transpose and existing operator-internal arithmetic spills remain. Because g
now has a consumer, its producer retains the existing mirrored output contract.

Precision is explicit: the external and result State ABI is FP32, the fixed
VXM input consumes BF16 upper planes, and the multiply/output is FP32. Checks
compare `eg_0`, the decayed State given actual g, and the end-to-end result
against independent FP32 equations; all 16 heads are checked separately.
Execution uses raw macro instructions and exports both `runtime.pipeline.csv`
and actual `runtime_icu_programs`, including pre-execution C2C instructions.

### Token-0 KS through the shared streamed contraction

Use `--with-ks` instead of `--with-state-decay`, with output directory
`results/qwen3_5_0_8b_prefill_tokenwise_ks`, to append the prediction
`p_0[h,v] = sum_k K_0[h,k] * Sbar_0[h,k,v]`. This still stops before the
beta-scaled innovation, outer-product State update, and Q contraction.

This path reuses the same generic C++ `kernel.scaled_matvec` lowering as
decode. External State is packed as `[head,value,key]`, so the key dimension
is in lanes and no transpose of the scaled State is needed. VXM saves its
FP32 multiply result with MEM_WRITE_TAP; its BF16 upper planes continue on
the forward stream into MXM IW. The two MXM weight buffers alternate every
32 cycles. The 16-head, 128x128 State fits one native scheduling domain.
The saved State is never read back for KS.

Q and the two native V head groups remain diagnostic outputs. V is exported
in outputs 2 and 8 to avoid a validation-only archive when its hemispheres
have different base addresses. K currently retains the mirrored L2 consumer
contract and is staged into the compact KS input; these transports are
recorded explicitly in `summary.json`. State and g do not need staging.

Outputs 5/6/7 are eg, scaled State, and KS. Numerical checks include every
head, a strict MXM comparison against actual BF16-truncated scaled State and
actual K, and an end-to-end comparison with the independent FP32 reference.
The shared streaming audit checks both scheduled and actually issued MXM
load intervals, alternating buffers, zero steady-state IW gaps, and no
saved-State reload.

## New chunkwise frontend: G_t prefix (2026-10-30)

`--with-gt` selects the incremental chunkwise branch and is mutually exclusive
with tokenwise `--with-state-decay`/`--with-ks`. Run with output directory
`results/qwen3_5_0_8b_prefill_chunkwise_gt`. It appends only
`G[t,h] = sum(g[0:t+1,h])` after the existing seq32 frontend; gamma, R, the
triangular solve and State recurrence are not yet connected to this new path.
The optional gamma extension below advances this branch by one operation.

The generic semantic operation is `vector_alu kind="cumsum_rows"`: an inclusive
scan along axis 0, BF16/FP32 input, FP32 output, and at most 32 columns.
It is not pointwise and cannot be independently token-tiled or chain-fused.
The fixed-port implementation reads BF16 upper planes from g's FP32 buffer,
uses C0 to receive the operand and C1 Add with its lane-local FP32 accumulator.
A first-token instruction resets and emits; a second instruction repeats
without resetting, emitting every partial sum. All 32 tokens accumulate in
32 consecutive cycles. There is no intermediate accumulator spill, no mean
scaling, no inter-lane prefix operation, and no new CModel opcode.

G has logical shape `[32,16]`, with head in lanes and token in successive MEM
rows. All G values are retained as FP32 for subsequent gamma/R consumers.
This first version retains mirrored full scans on the two hemispheres; it
does not incorrectly reset at the 16-token partition boundary. The g producer
therefore uses its existing mirrored consumer contract. The scan aliases that
input directly, with no linker MEM transport or SXM transpose.

`vector_cumsum_rows_runtime_test.py` checks 1/32/64 rows and two consecutive
scans, with mixed signs and different per-head values. Its expected result
applies the actual BF16 ingress truncation then FP32 cumulative addition.
The real-weight test additionally compares end-to-end G against the FP32
model equations, checks the actual consecutive C1 issue interval, and exports
G in `actual.5.bin`, runtime pipeline CSV and actual per-ICU programs. The
generic scan does not implicitly reset at chunk boundaries: a future
multi-chunk graph must express each chunk's scan boundary explicitly.

### L2 normalization result streams directly into wide SXM

The paired-head wide-SXM prefill path now restores normalized Q/K layout
directly from VXM's forward SR output. It no longer writes normalized vectors
to `feedbackOutput` and reads them back for the transpose. Each hemisphere's
SXM captures its arriving BF16 byte pair (streams 10/11 or 2/3), launches a
32x32 tile every 32 cycles, and scatters the transpose output to final MEM.
`LPUTargetModel` explicitly describes VXMResult -> SxmInput on the forward
fabric, with the transport hops and capture cycle included in its latency.
No CModel changes or new hardware arithmetic are needed.

The reduction's raw-input write-tap and subsequent normalization replay remain:
they retain x until its whole-head squared sum and inverse norm are available.
The separate local-SXM and direct-planar decode paths retain their existing
behavior. `l2norm.vxm_sxm_direct` marks the optimized output path; the seq32
regression rejects normalized-result MEM writes during those intervals.

The real-weight chunkwise-G_t frontend passes with all six output binaries
bit-identical to the preceding run. Q and K each take 3198 nominal cycles,
down from 4311 (25.8% reduction). Across both operators, 512 MEM read domains
and 512 write domains disappear: 256 KiB read plus 256 KiB write traffic.
Runtime trace including C2C is 259279 cycles, down from 261586. The existing
result directory contains the updated runtime CSV, actual ICU programs, and
`l2_vxm_sxm_audit.json`. Native L2/RMSNorm fallback regressions and the stream
route test also pass.

## Chunkwise gamma_t (2026-11-02)

Use `--with-gamma` (implies `--with-gt`) and output directory
`results/qwen3_5_0_8b_prefill_chunkwise_gamma`. This extends the same real-weight
seq32 frontend with `gamma[t,h] = exp(G[t,h])`, not `exp(g[t,h])`. Outputs 0-5
retain Q/K/V/beta/g/G; output 6 is gamma, shape `[32,16]`, FP32 buffer ABI.
R, the triangular solve, and State recurrence remain outside this test.

The Python test adds only a semantic `vector_alu kind="exp"` node. Existing
C++ vector lowering owns placement, instructions and scheduling; no new
CModel operation or host arithmetic is used. Gamma reads G's native SRAM
address (slice 48-51, bank 1, base row 4096 in this run), using its upper
BF16 planes. There is no linker MEM transport or SXM transpose. G remains
stored for future R consumers. This is a separate Exp pass, not a fused
scan/Exp chain: G is read back once. Gamma uses the existing token partition
policy, 16 distinct tokens per hemisphere, and a native partitioned output.

Raw-macro CModel execution completes with seven outputs and zero host
operations. All six preceding outputs are bit-identical to the G-only run.
The actual Exp issue interval is `[228442,228458)` (16 consecutive cycles).
Gamma's nominal stage spans 49 cycles including ingress/drain; the complete
runtime trace, including C2C and output export, is 259689 cycles (0.519378 ms
at 500 MHz). `runtime.pipeline.csv` and `runtime_icu_programs/` are exported
from the actual execution, including pre-execution instructions.

Precision is explicitly limited by the existing hardware: BF16 reload of G,
then FP16 FTZ input/output boundaries in Exp, stored in an FP32 ABI buffer.
The test checks finite, nonnegative, nonincreasing per-head gamma, elementwise
Exp against an independent reference, and local/whole-path errors separately.
Gamma NRMSE relative to `exp(actual G)` is 0.0343%; whole-path NRMSE relative
to the FP32 model equations is 0.1109%. Existing gate/scan errors are amplified
by Exp: heads 0 and 3 have whole-path NRMSE 3.91% and 6.08%, exceeding the
unchanged 3% per-head criterion. The summary therefore records
`passed_with_precision_limits`, not an unconditional numerical pass, and
`end_to_end_head_check_passed=false`. Local Exp checks pass for every head.
116 of 512 outputs are zero due to the decay range and FP16 FTZ behavior;
later R should be derived from G differences, not division by stored gamma.

```powershell
python compiler/tests/qwen3_5_prefill_qkv_wide_sxm_test.py --with-gamma `
  --package results/qwen3_5_0_8b_decoder_layer_prefill/package `
  --build-dir build-ftlpu-vs2026-direct `
  --target-config ../FTLPU-CMODEL/config/ftlpu-lpu32.json `
  --output-dir results/qwen3_5_0_8b_prefill_chunkwise_gamma
```

## Chunkwise causal R (2026-11-02)

`--with-r` implies `--with-gamma` and appends only R, stopping before KKt,
the triangular solve and State recurrence. Run the same command above with
`--with-r` and `--output-dir results/qwen3_5_0_8b_prefill_chunkwise_r`.
Outputs 0-6 remain unchanged; output 7 stores `[i*32+j, head]` as FP32,
equivalent to logical `[32,32,16]`. It contains
`exp(G[i,h]-G[j,h])` below the diagonal, one on the diagonal, zero above it.
It reads retained G, never divides gamma values that may have underflowed.

The importer expresses the generic compiler operation
`vector_alu kind="causal_exp_difference"`, with `[T,H]` BF16/FP32 input and
`[T*T,H]` FP32 output, `1 <= T,H <= 32`. This pairwise operation is excluded
from pointwise token tiling and adjacent pointwise chain fusion. C++ emits
all scheduling and address domains directly; the CModel executes only its
existing hardware instructions. There is no host evaluation of R.

The implementation preserves G's native head-in-lanes layout and mirrored
hemisphere contract. Each row first reads Gi once through C0 into C1's FP32
accumulator (reset, Add, no emit). It then reads Gj for `j <= i`, C0 negates
Gj, C1 adds retained Gi without changing the accumulator, and C5 applies Exp.
C2-C4 and C6-C7 carry pipeline values to the legal Exp column and output
register; they do not perform MEM copies. The upper triangle uses an
immediate-zero chain after the Exp pipeline has drained. This avoids both
unnecessary upper-triangle exponential work and `inf * 0` masking hazards.
The same MEM ICU finishes the Gi load before starting the Gj read domain.
No different ICU instructions execute interleaved in one queue.

There is no expanded Gi/Gj tensor, intermediate difference spill, transpose,
or linker MEM transport. R writes directly to its planned planar output.
Both hemispheres currently compute mirrored R, matching G's existing layout;
head partitioning and a downstream matrix-consumer layout are not claimed
by this step. Later KKt/solve integration must plan that consumer layout
explicitly rather than silently archive and repack R.

`causal_exp_difference_runtime_test.py` checks 1/8/32-token cases, non-full
head widths, per-row accumulator reset, a constant-G head, a rapidly decaying
head with positive masked differences that would overflow Exp, and exact
diagonal/upper-triangle values. References truncate G at BF16 ingress and
allow the existing FP16 FTZ Exp boundary. Real-weight checks separately
report quantized-G, actual-G and end-to-end errors, plus every head; upstream
precision limitations remain visible in the result status.

The real-weight raw-macro run completes with eight outputs and zero host
operations. All seven earlier outputs are bit-identical to the gamma-only
baseline (`previous_outputs_audit.json`). R takes 1803 scheduled cycles;
actual Exp issues span `[228684,230452)`, with row lengths 1 through 32
(528 vector issues per hemisphere), and the last R writes finish at 230473.
The complete diagnostic trace is 269914 cycles / 0.539828 ms at 500 MHz;
this includes export of the additional 64-KiB logical R output, so its
increase over the gamma run must not be interpreted as R compute time.

R NRMSE is 0.0192% against quantized-G Exp, 0.1217% against actual FP32 G,
and 0.1670% end to end. Every R head passes the 3%/0.999 NRMSE/cosine checks;
the worst head NRMSE is 1.211%. Diagonal values are exactly 1 and upper
triangle values exactly 0. Overall status remains `passed_with_precision_limits`
solely because the preceding gamma stage retains its previously reported
head errors. Runtime CSV and 233 actual ICU CSVs are included in the result
directory, including pre-execution instructions.

## Chunkwise weighted Gram A (2026-11-02)

`--with-a` implies `--with-r` and appends `A[h,i,j] = dot(K[i,h,:],
K[j,h,:]) * R[i,j,h]`. Use output directory
`results/qwen3_5_0_8b_prefill_chunkwise_a`. This stops at A: beta weighting,
construction of L, the triangular solve, and State recurrence are not added.
Outputs 0-6 retain the preceding frontend values; output 7 uses the native
transposed R ABI described below. Output 8 contains sixteen
32x32 A matrices, with native head order `[0,2,...,14,1,3,...,15]`; the first
group resides on East and the second on West. `read_head_groups` in the test
only unpacks that ABI for numerical comparison. No FU program reorders or
archives A merely for export. A future logical consumer must describe the
inverse head permutation as a view, not assume natural head order in this
buffer.

The generic compiler semantic `vector_alu kind="weighted_gram"` accepts
K `[32,H*D]` BF16 and weights `[1024,H]` FP32, with even H<=32 and D a positive
multiple of 32. It returns grouped-head rows `[H*32,32]` FP32. This operation
is excluded from pointwise splitting/fusion. All hardware lowering is C++;
the CModel gains no model operation. A standalone packed K is supported;
the real frontend instead aliases the existing distributed16 K allocation.
For that edge, L2 retains its alternating head partition, eliminating its
otherwise required output mirror. Other consumers still retain their own
layout contracts.

For each pair of heads, the two MXMs work on different heads. Four D32 tiles
cover D128. Each native K vector is read once for a column-mode 1vector IW
load and again as an activation. The same MEM ICU completes the 32-vector
weight domain before starting the activation domain, then starts the next
tile's load after the activation domain retires. FP32 MXM ACC retains the
partial sum across D tiles. The last tile emits a BF16 Gram vector through
reverse SR directly to VXM C0. C0 multiplies by the corresponding R vector;
C1 drains the FP32 A result. The full Gram matrix is never spilled to MEM.

R's native `[i,j,head]` vectors span heads, whereas each Gram result vector
spans j. The operator performs one necessary wide-SXM transpose per i, 32
tiles at II32, reading R's BF16 upper planes and retaining a BF16 table whose
rows are `[i,head]` and lanes are j. K needs no copy or transpose. The graph
linker emits no MEM transports; the explicit R transpose remains visible
as `weighted_gram.r_head_transpose`, followed by `weighted_gram.mxm_to_vxm`.
Temporary R storage is part of the compiler memory plan. Input K, R, the
temporary R table, and A use their planned banks/addresses.

`weighted_gram_runtime_test.py` covers H2/D32 and H4/D128 using independent
NumPy Gram products, per-head errors and exact upper-triangle zeros. The
real-weight test also checks end-to-end A against the FP32 model equations,
executes actual MXM instructions, checks 1vector IW domains, rejects Gram
result MEM spills, and exports runtime pipeline / actual ICU CSVs. The
precision boundaries are BF16 K, FP32 MXM accumulation, BF16 Gram stream and
R ingress, and FP32 VXM multiply/output. This is not an all-FP32 path.

Before producer-side R fusion, the real-weight seq32 run passed all A per-head
checks. That version took 3926 scheduled
cycles, including the R layout conversion, with actual MXM compute in
`[231854,234556)`. Tightening the IW/activation read schedule saves 960 cycles
over the initial 4886-cycle implementation; all nine outputs are bit-identical.
The earlier eight outputs also remain bit-identical to the R-only baseline
(`previous_outputs_audit.json`). A NRMSE is 0.2278% against actual K/R and
0.3393% end to end; the worst head is 0.9746%. Upper-triangle zeros are exact.
The full diagnostic trace is 275096 cycles / 0.550192 ms at 500 MHz, including
all diagnostic output transfers. Zero host operations execute. Overall status
remains `passed_with_precision_limits` because of the preceding gamma stage,
not A. The result directory includes runtime pipeline and 233 actual ICU CSVs.

### Producer-side R transpose

The A test now requests `causal_exp_difference` with `transpose_result=true`.
This is a compiler semantic fusion, not a new hardware instruction. It produces
BF16 `[i*H+h,j]` directly: VXM generates 32 vectors for fixed i, a wide SXM
captures their FP32 upper byte planes on the forward SR path, and MEM stores
only the first H transposed vectors. Exp and the fast upper-triangle zero path
are scheduled to join without a missing vector at their boundary. SXM starts
as soon as that row stream arrives; it does not wait for the whole R matrix.

Output 7 in the A test consequently changes from FP32 `[i,j,h]` to BF16
`[i,h,j]` (physical tensor `[32*H,32]`). This retains exactly the truncation
the earlier weighted-Gram consumer performed on its FP32 R buffer. Test-side
unpacking only converts dtype/reorders axes. The R-only test retains its
original FP32 diagnostic ABI. The native weighted-Gram input aliases the new
R allocation and reads rows at stride H; it emits no R transpose or scratch
R table. The graph planner reserves its output on queues independent of K,
since R ingress overlaps the last K activation domain at the Gram epilogue.
The A output allocator also keeps its drain independent of both inputs.
Transposed R still resides in MEM because producer order i/h differs
from the Gram consumer's h/i order. This removes the raw-R write/read pair,
not the final consumer-facing R storage.

The causal-exp regression covers both ABIs and checks that direct transpose
is bit-identical to the old output's BF16 upper planes, including diagonal,
upper-triangle zeros, underflow and masked positive-difference cases. The A
integration audit rejects raw VXM R writes and a second transpose in A.

The validated fused run uses reverse SR ports 4/5 for SXM results, separately
from G ingress on port 0. This matters when final R is below G's slice group:
sharing port 0 would collide before either MEM consumer could accept data.
Each hemisphere issues one repeated SXM domain (32 tiles, interval 56), and
one WRITE3D per output byte plane (H vectors per tile). In the real graph R
resides on slices 8/9 bank1, independently of K on slices 32-47 and A on
slices 48-51; no linker MEM transport is emitted.

After fusion, R takes 1840 scheduled cycles and A takes 2775, compared with
1803+3926 before fusion: combined 5729 -> 4615 cycles, saving 1114 (19.4%).
R Exp issues remain `[228684,230452)` and A MXM compute spans
`[230765,233467)`. The full diagnostic runtime trace is 272983 cycles /
0.545966 ms at 500 MHz. Its reduction also includes the smaller BF16 R host
export, so use the combined stage count for the compute comparison.
Outputs 0-6 and A are bit-identical to the previous run. Transposed R is
bit-identical to the previous R's BF16 upper planes (`r_direct_audit.json`).
A end-to-end NRMSE remains 0.3393%; the existing gamma precision caveat is
unchanged. All arithmetic executes in the CModel with zero host operations.

Validation used a frozen source snapshot in `tmp/r_direct_validation` and
`build-ftlpu-r-frozen`, because shared hardware headers were changing during
the build. The result's `validation_build.json` records source hashes. The
R ABI/route regression and H2/D32, H4/D128 Gram regressions also pass.


### Continuous KK^T using producer-selected bank stripes

For a native paired-head D128 L2 output used only by `weighted_gram`, the
whole-graph planner selects `bank_tile_rows=4`: each D32 tile keeps its global
row address, but uses `bank XOR (feature_tile % 2)`. The producer writes two
WRITE3D domains per slice (even and odd tiles) directly from SXM. It never
copies or duplicates K. Existing unstriped Gram inputs retain their serial
schedule.

The planner reserves the retained K row range in both banks, including
bank0 native scratch lifetimes. R and A are placed on independent MEM queues.
A bank's weight-read domain completes before its activation-read domain;
only different bank ICUs overlap. MXM IW uses one-vector mode and alternates
the two weight buffers, loading tile n+1 during tile n's 32-cycle Compute.
The next head starts without waiting for the previous head's VXM epilogue
and output stores. The accumulator reduction order is unchanged.

The software binding ABI has a bank-striped block4 distributed layout for
diagnostic outputs and external I/O. C2C segments stop at each four-row bank
boundary; output unpacking reads the actual bank without an on-chip gather.
This is physical tensor-layout support in the software runtime, not a GDN
operation in CModel. The layout roundtrip regression covers unaligned base
rows, multiple token blocks, both banks and hemisphere ownership. The real
A integration additionally asserts 32 consecutive Compute intervals of 32
cycles per hemisphere, with no tile/head gaps, and no linker transports.

The overlap also needs independent forward SR streams. VXM C1's FP32
result ports are fixed (0..3 or 8..11); they cannot be reassigned. K
activations therefore use MXM-selectable ports 4/5 and weight loads use
16/17. Reusing activation ports 0/1 causes a real passive-SR collision when
the previous head drains while the next head computes. This is checked by
CModel execution, not hidden by additional waits or relaxed collision rules.

Validated with real Qwen3.5-0.8B layer-0 weights, seq32 and raw macro ICU:

- Both MXM units compute continuously over `[230765,231789)` (1024 active,
  zero idle cycles), versus a 2702-cycle span previously.
- A scheduled stage: 2775 -> 1097 cycles (-60.47%); L2 and R stage durations
  are unchanged. The stage still includes initial load and final drain.
- All nine exported tensors are bit-identical to the pre-optimization run.
  A end-to-end NRMSE is 0.339338%; the prior gamma precision limitation remains.
- Runtime logical cycles: 159576 -> 157898. Full diagnostic trace: 272983 ->
  262537 cycles. The latter also changes with diagnostic C2C export grouping;
  it should not be interpreted as KK^T compute time.
- No host arithmetic, linker MEM transports, raw R spill or Gram spill.

`results/qwen3_5_0_8b_prefill_chunkwise_a/kkt_continuous_audit.json` records
these checks. Runtime pipeline and per-ICU programs are refreshed in the
same result directory. The independent H2/D32 and H4/D128 Gram regression
still exercises the unstriped fallback. The real prefill regression rejects
any Compute gap across either tiles or heads.

### Strict lower L with fused row weighting

`--with-l` extends the same real-weight seq32 graph through
`L[h,i,j] = beta[i,h] * exp(G[i,h]-G[j,h]) * dot(K[i,h],K[j,h])`
for `j<i`, with exact zero diagonal and upper triangle. It stops before
the `(I+L)` triangular solve, RHS construction and State update. L is the
strict lower matrix, not the identity-added solve matrix.

The generic causal-difference kernel accepts `row_scale=true` and
`strict_lower=true`. Its RHS is the existing beta tensor. Each row loads
one beta head-vector into the C7 accumulator, while C1 retains G_i. C5
computes Exp and C7 multiplies by beta_i. The strict mask uses the existing
zero stream; masked differences never enter Exp. The 32-vector stream
feeds wide SXM directly, storing the BF16 coefficient C=beta*R in native
`[i*H+h,j]` layout. No expanded beta table or unweighted A is stored.
The unchanged weighted-Gram emitter consumes C, computes KK^T on MXM,
and streams Gram into VXM for the final multiply. L is FP32
`[16,32,32]`, exported as even heads followed by odd heads.

Example (from the software repository):

```powershell
python compiler/tests/qwen3_5_prefill_qkv_wide_sxm_test.py --with-l --package results/qwen3_5_0_8b_decoder_layer_prefill/package --build-dir build-ftlpu-r-frozen --target-config tmp/r_direct_validation/FTLPU-CMODEL/config/ftlpu-lpu32.json --output-dir results/qwen3_5_0_8b_prefill_chunkwise_l
```

The 2026-11-02 raw-macro CModel run reports:

- Seven independent causal-coefficient cases pass, including scaled strict
  FP32/BF16 outputs, beta=0/1, exact masks and underflow cases.
- Outputs 0-6 are bit-identical to the validated A run. Output 7 now contains
  C rather than unweighted R, and output 8 contains L rather than A.
- Coefficient stage: 1844 scheduled cycles (R was 1840); weighted Gram/L:
  1097 cycles. Both MXM units compute continuously over `[230769,231793)`,
  1024 active cycles and zero inter-tile/head idle cycles.
- No linker MEM transports, Gram spill, A spill or host arithmetic. Actual
  runtime ICU programs are exported for 232 queues (210 active).
- L end-to-end NRMSE is 0.4713%, cosine 0.99999445. Given actual K and C,
  all head NRMSE values are below 0.166%. The fused coefficient boundary
  differs from post-weighting the earlier A by 0.2119% aggregate NRMSE.

**Numerical limitation:** this is `passed_with_precision_limits`, not an
all-head numerical pass. Heads 0/3/11/13 have end-to-end L NRMSE of
22.39%/33.15%/8.91%/3.61%. BF16 reload of cumulative G before subtraction
loses accuracy in strongly decaying heads; removing the large unit diagonal
exposes the error previously hidden by aggregate A/R metrics. With the
quantized-G reference, every L head is within 0.418% NRMSE. The earlier
gamma limitation also remains. Resolve this precision boundary before
claiming validated triangular solve/State accuracy; do not relax the
per-head tolerances to conceal it.

The result directory contains `runtime.pipeline.csv`, actual
`runtime_icu_programs`, `summary.json`, `l_audit.json` and
`validation_build.json`. The full diagnostic trace is 262541 cycles
(0.525082 ms at 500 MHz), including diagnostic C2C outputs; runtime logical
cycles are 157902. Frozen-build provenance is retained as for the A run.

### Full FP32 ingress (2026-11-03)

The L test now enables `ftlpu.vector_fp32_ingress=true` by default. Use
`--no-fp32-ingress` only to reproduce the old BF16 reload experiment.
The original `G_i-G_j` algorithm is retained: four MEM byte planes feed
`stream_f32`, so neither G nor beta is truncated before the chain.
The same policy covers the packed pointwise operations in the g branch,
the g prefix scan and gamma input. Q/K/V and beta outputs remain
bit-identical; g accuracy also improves because its FP32 intermediate
reloads are now complete. Exp LUT FP16 FTZ behavior and BF16 coefficient
storage are unchanged.

Hardware contract: `FTLPU-CMODEL/docs/vxm_fp32_input.md`. The 96-bit ALU
packet uses reserved bits 52/53 to extend operand selectors. Fixed FP32
ports use four byte streams; adjacent depth-2 heads share their buses.
The compiler validates port indices, reserves all four streams and gives
simultaneous binary operands independent MEM ICU queues. No model-specific
arithmetic or host correction was added to CModel.

Final real-weight seq32 raw-macro validation in the same L directory:

| Metric | BF16 reload | Full FP32 reload |
| --- | ---: | ---: |
| g end-to-end NRMSE | 0.74264% | 0.006451% |
| L end-to-end NRMSE | 0.47130% | 0.34165% |
| Worst L head NRMSE | 33.1514% | 0.57320% |
| L failing heads | 0, 3, 11, 13 | None |
| Gamma failing heads | 0, 3 | None |
| Full diagnostic trace cycles | 262541 | 262540 |

Status is now `passed`, with all existing per-head thresholds unchanged.
Both MXM units run continuously over `[230768,231792)`, 1024 active cycles
and zero idle cycles. No linker MEM transports or host operations occur.
Actual ICU exports contain 232 queues, 218 active: the additional active
MEM queues carry the previously omitted lower byte planes.

Hardware FP32-input, Lane, Slice and ICU tests pass (4/4). The compiler
two-input FP32 subtraction test preserves all 128 low-mantissa differences
bit-for-bit through MEM/raw macro execution; BF16 reload would lose them.
Seven FP32 causal-coefficient cases and seven legacy cases also pass.
`fp32_input_audit.json` and the refreshed `l_audit.json` record per-head
before/after comparisons; `validation_build.json` records both repositories'
tested source hashes. Runtime CSV and per-ICU exports describe this FP32 run.
