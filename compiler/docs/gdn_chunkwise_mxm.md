# Chunkwise GDN MXM lowering

The production Kernel graph path uses MXM for KKt, KS, QS, QKt,
attention_D, and state_update. It retains VXM for the triangular solve,
decay/beta arithmetic, and additions. No GDN arithmetic is added to CModel.

## Placement and precision

The supported core has chunk size 32 and key/value dimensions 128. Both
hemispheres retain identical BF16 tensors, as in the previous native core.
Each hemisphere uses its configured MXM. Computation accumulates in FP32 across
K tiles and rounds to BF16 on the final output. Outputs retain their row-packed
MEM placement for the existing VXM consumers.

K is reused directly as the column view of Kt. KS and QS share a cached
transposed incoming State at pair 50/51, rows 5000..5511, bank 0. The
coefficient scratch uses rows below 4096. The cache is regenerated each chunk.
Other right-hand matrices use a temporary column view in pair 4/5, whose
triangular-solve lifetime does not overlap these products. All K-tile column
views are prepared before starting the product, so layout preparation cannot
clobber live cross-K ACC partials.

To form a transpose, existing row-packed vectors are loaded as MXM columns,
then multiplied by hardware-generated identity activations. This emits the
transposed packed vectors directly, using BF16 operands and FP32 accumulation.
It replaces the earlier SXM broadcast plus VXM one-hot gather without any host
intermediate transfer. The identity vectors come from the existing native
constant initializer.

Each output wave contains 32 rows. Cross-K partials use at most 128 ACC rows;
products with a single K tile immediately drain and reuse ACC. Fixed-column
BF16 loads use existing IWColumnDirect16 hardware instructions, encoded as
raw macros. Their spacing includes macro decoder bandwidth. Column induction
that changes the inner column is still rejected by the generic macro converter.

## Remaining cost

MXM identity transposes still consume loading, compute, and memory bandwidth;
they are explicit layout preparation, not free reinterpretations. A future
producer layout could eliminate some of them. Column loads and drain spacing
still account for macro decoder bandwidth, and both hemispheres still duplicate
the same head.
The VXM reference scheduler remains available to the standalone native test;
the production graph emitter selects MXM tasks.

## Validation

`gdn_mxm_chunkwise_runtime_test.py` exercises the real compiler, raw-macro
binary, ModelSession C2C I/O and CModel execution with random inputs and a
nonzero initial State. Seq32 and seq64 cover one and two chunks. It compares
outputs and final State against independent token recurrence and verifies all
six MXM stage markers and runtime MXM events. No intermediate host arithmetic
is used to supply hardware operands.

The real-weight integration uses `qwen3_5_attention_runtime_test.py` with
`--icu-compression raw-macro --trace` and updates
`results/qwen3_5_0_8b_gdn_attention_prefill`.

## Measured real-weight attention (2026-10-28)

Qwen3.5-0.8B, seq32, W8A16, 500 MHz / 51.2 GB/s configured DDR peak:

| Metric | Previous VXM product core | MXM product core |
| --- | ---: | ---: |
| Runtime trace cycles (including C2C) | 9,007,511 | 3,739,943 |
| Simulated latency | 18.015022 ms | 7.479886 ms |
| Linked runtime image bytes | 7,583,913 | 2,456,824 |

Attention NRMSE 0.00758644, State NRMSE 0.0115772, Conv cache NRMSE
0.00373527; all match the prior real-weight run. Host intermediate operations:
zero. This is input RMSNorm plus complete GDN attention, not a decoder layer
with residual/FFN. Standalone nonzero-State seq32 and seq64 tests also pass;
their output/State NRMSEs are 0.00561744/0.00649186 and
0.00587950/0.00585263 respectively.

## Tighter native scheduling

Independent VXM binary rows now form one continuous 3D MEM domain and a
continuous ALU repeat. Row-wise gates use zero inner address stride, preserving
the broadcast without per-row idle gaps. This applies only when the output
MEM pair differs from both input pairs; in-place operations retain serialized
reads and writes on each MEM ICU queue.

MXM column loads are spaced four cycles apart instead of eight. The first
compute wave starts 144 cycles after the first load instead of 320, and
successive 32-row waves start 32 cycles apart instead of 96. Triangular solve
steps retain their data dependencies but reduce the guard from 48 to 16 cycles.
Ingress/egress and epoch guards are also shortened. Transport and decoder
latency remain explicit; these changes do not interleave instructions within
one ICU queue.

Raw-macro CModel tests with nonzero initial State pass for both chunk counts,
with unchanged numerical errors:

| Core test | Before (cycles) | Tight schedule (cycles) |
| --- | ---: | ---: |
| Seq32 | 143,987 | 65,115 |
| Seq64 | 269,267 | 116,707 |

The complete real-weight attention rerun also passes (same configuration and
precision as above). Runtime cycles including C2C fall from 3,739,943 to
2,477,463, or 7.479886 ms to 4.954926 ms (33.76% less time). Logical execution
cycles are 2,283,279; the linked runtime image is 2,418,934 bytes. Numerical
errors are unchanged and host intermediate operations remain zero.

Runtime trace activity, measured as the union of cycles containing at least
one FU operation of that class, increases as follows:

| Class | Before | Tight schedule | Active cycles |
| --- | ---: | ---: | ---: |
| MXM Compute | 5.64% | 8.51% | 210,944 |
| VXM (including bypass) | 2.76% | 4.16% | 103,073 |

This is an issue-activity metric, not arithmetic lane utilization or FLOPS
efficiency. The unchanged active-cycle counts confirm that this improvement
comes from reducing schedule gaps rather than removing mathematical work.
Products and dependent vector stages remain serialized; iMEM segment drains,
layout preparation and recurrence dependencies still leave gaps.

## Shared projection kernel

`ProjectionGemmSchedule.hpp` now owns the closed-form row/K/output-group
MXM domains, alternating weight-buffer selection, final-K stream/clear
protocol, and LOAD/COMPUTE emission. Its callers are the single-token-tile
linear projection path (including GDN), the merged dense Q/K/V path, and
the merged dense O projection path. The binding supplies physical queue,
stream and ACC addresses. Layout-specific MEM transfers, dequantization,
absolute start times, reduction/group intervals, C2C page boundaries and
RoPE remain with the respective callers. Multi-token-tile and non-merged
fallback schedules are unchanged; this does not make every projection path
identical or tighten the GDN projection's existing cadence.

Extraction validation: the Qwen2.5 seq32 serialized full Schedule is byte
identical to a build using the prior three emitters. The existing O projection
regression reports the same Q RoPE scratch-layout assertion before and after
the refactor. The GDN real-weight attention raw-macro binary is byte identical
to the numerically validated tight-schedule binary (SHA256
`db1cd276faf5942279790047150bb375982cc2ee2726a027d2db4682e96286c7`).
No new runtime trace is necessary for this behavior-preserving extraction.

## Continuous seq32 projection

The linear projection caller now schedules an even number of K tiles at a
32-cycle compute cadence on the two-weight-buffer target, instead of imposing
a minimum 64-cycle interval. Weight transport is accounted for in the first
load's lead time; it is not added to every reduction. ACC feedback and buffer
reuse bounds still constrain the interval. Weight streams use E24--E31 (the
target's last eight eastbound streams), independently of the activation
streams, just as in dense attention. Reusing activation stream IDs for weights
caused a passive-SR write collision when loads overlapped computation.

Short-row/decode and odd K-tile counts retain the conservative cadence. In
particular, toggle_dim1 restarts on buffer 0 in each output group; an odd K
count needs a drain before that buffer is reused. Raw-macro RMSNorm-to-linear
tests pass for K=128 and K=96 (NRMSE 0.00326112 and 0.00347387 respectively).

The real-weight seq32 attention rerun passes with identical attention, State
and Conv cache errors. Runtime trace cycles decrease from 2,477,463 to
2,311,858 (4.954926 ms to 4.623716 ms), with zero host intermediate operations.
MXM Compute issue activity increases from 8.51% to 9.12%, and VXM activity
from 4.16% to 4.46%; active-cycle counts remain 210,944 and 103,073.
The trace and actual ICU programs are updated in the existing attention
result directory. The final compiler reproduces the tested binary exactly
(SHA256 `43227ffebe15ee14ccbcb332d502e5a51b877205b65e40c28186fdca4f1ce744`).

## Overlap inside the native GDN core

BF16 matrix products now alternate two weight buffers across output tiles.
The next tile's column loads overlap current computation when activation,
weight and output MEM pairs do not alias. Alias cases retain serialized
execution. The tile interval respects the 128-cycle column instruction issue
span and accumulator/result drain; a 32-row tile starts every 128 cycles
instead of 240. Adjacent identity-transpose tiles use the same pipeline and
only drain the final tile. Cross-K accumulation boundaries remain intact.

Native VXM aliasing-row guards now derive from the maximum MEM group
round-trip plus eight cycles. SXM diagonal/broadcast guards derive from the
MEM/SXM transport distance instead of a fixed 64 cycles. The standalone
VXM-reference implementation retains its original timing.

Raw-macro nonzero-State regression results, with identical errors:

| Test | Before | Overlapped core |
| --- | ---: | ---: |
| Seq32 | 65,115 cycles | 55,677 cycles |
| Seq64 | 116,707 cycles | 99,015 cycles |

The remaining gap between 32-row compute waves is not claimed to be removed:
BF16 column loading still needs 128 issue cycles per tile. More overlap alone
cannot sustain a new 32-row tile every 32 cycles with this loading format.

The complete real-weight seq32 attention run passes with unchanged attention,
State and Conv cache errors and zero host intermediate operations. Logical
execution is 1,966,383 cycles. Runtime trace including C2C is 2,160,865 cycles
(4.321730 ms), down from 2,311,858 (4.623716 ms). MXM Compute issue activity
is 9.76% and VXM activity is 4.77%, with unchanged active-cycle counts.
The existing runtime pipeline CSV and per-ICU runtime program exports have
been regenerated for this run.


## One-vector projection loads and compact Conv1D (2026-10-29)

The LPU32 target selects `mxm_prefill_weight_streams = 1`. The single-token-tile
linear projection lowering honors this policy: eight resident INT8 MEM slices
multiplex onto one weight stream, with slice `s` issuing at `start+s+8*r` and
reading row `base+3-r`. A tile uses 32 consecutive column loads. Compute starts
32 cycles later; even-K seq32 projections retain a 32-cycle tile cadence with
two alternating weight buffers. The existing weight layout is unchanged.

Raw Macro column-load induction now counts the combined five-bit column
(supercell plus inner column). Supercell-mode induction retains its old units.
The CModel Active Context and generic repeat expander implement this carry;
no model-specific operation or expanded FU compression is involved. The Macro
wire format and 256-bit Active Context format are unchanged. RTL implementing
these contexts must likewise advance the combined column counter in column mode.

Fixed-chain Conv1D no longer inserts a 48-cycle drain after every output.
Four causal taps now start every nine cycles instead of 52. The five remaining
cycles cover the three accumulator configurations' nine-word frontend budget;
removing them caused the compiler's real ICU bandwidth check to fail. Block
reload boundaries, input ingress and the continuous SiLU pass are retained.
This improves the steady four-tap MAC issue fraction from 4/52 (7.69%) to
4/9 (44.44%); it is not a claim that all 16 ALUs are fully utilized.

The raw-macro Conv1D+SiLU test passes at NRMSE 0.001397725, and projection tests
with K=128 and K=96 preserve their previous numerical errors (0.003261116 and
0.003473875). Column induction tests cover the 7-to-8 carry, all 32 columns,
buffer alternation, and raw bitstream encode/decode equivalence.

The real-weight seq32 complete-attention rerun passes with unchanged errors:
attention NRMSE 0.007586437, State 0.011577205, Conv cache 0.003735271.
Logical execution falls from 1,966,383 to 1,834,482 cycles. Runtime including
C2C falls from 2,160,865 to 2,028,805 cycles (4.321730 to 4.057610 ms at
500 MHz, a 6.11% reduction). Each paired 1024-channel-per-hemisphere
Conv1D+SiLU region takes 14,336 instead of 58,368 cycles. The runtime CSV
confirms 29 steady four-tap issues at interval nine starting at cycle 263848.
The existing attention result directory contains the refreshed trace,
actual runtime image, numerical summary and scheduling_comparison.json.


## Native projection addresses for Conv1D

The graph planner partitions projection-fed distributed Conv1D by native
32-channel words: the primary chain consumes the West words and its mirrored
partner consumes the East words. Each pair still covers 2048 logical channels.
Channel gather/scatter is represented by tensor reference views, without a
MEM staging operation. An affine view of the producer's hemisphere-pair layout
becomes the consumer's planar placement, including the original bank and row.
Fixed-chain Conv skips its separate input-to-window copy for this direct view.
BF16 Conv coefficients occupy slices 24--31 rather than the projection output
pair on slices 0--1, permitting simultaneous input and coefficient reads.

A Q/K head can now span multiple Conv producers. L2Norm receives per-word
source descriptors and transposes directly into the final feedback tile;
it does not first materialize a contiguous head. Required transposes and
feedback replication are retained. External Conv inputs and unsupported
non-affine views keep their existing materialization path.

`projection_conv_direct_view_runtime_test.py` verifies projection -> bilateral
Conv1D -> head L2Norm through raw Macro, both numerical outputs and absence of
Conv window copies / generic L2 input materialization. The standalone external
Conv1D regression also passes after the coefficient placement change.

The complete real-weight seq32 attention test passes with unchanged numerical
errors. Runtime cycles decrease from 2,028,805 to 2,015,191
(4.057610 to 4.030382 ms). Each Conv1D+SiLU region
takes 13,216 cycles, with only 168--169 logical cycles between adjacent pairs.
The refreshed pipeline and actual runtime ICU images are in the existing
attention result directory.
