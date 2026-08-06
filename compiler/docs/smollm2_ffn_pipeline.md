# Generic W8A16 FFN Pipeline: SmolLM2-135M Instance

![SmolLM2-135M FFN pipeline](smollm2_ffn_pipeline.svg)

## Schedule Policies

The default `--mxm-execution auto` mode selects the 16-stream Block8 path for
this BF16-activation/INT8-weight FFN. Two Block8 policies are validated:

- `--ffn-schedule tail` is the correctness baseline. It completes every
  Gate/Up projection and drains pair-indexed scratch storage before issuing
  the first SwiGLU at cycle 16,574. The program ends at cycle 33,681.
- `--ffn-schedule fused` overlaps each completed Gate/Up pair's SwiGLU with
  later projection work. The stream allocator reserves the target-derived
  projection footprint and allocates a non-overlapping VXM-to-MEM route; the
  emitter contains no fixed high-stream helper. The program ends at cycle
  27,425, saving 6,256 cycles (18.57%) relative to tail.

The runtime samples 33,745 cycles for tail and 27,489 for fused, including 64
drain cycles. MEM issue utilization is 16.89% / 20.74%, MXM compute issue
utilization is 32.08% / 39.38%, and VXM issue utilization is 23.33% / 28.64%
for tail / fused. Both binaries match all 73,728 BF16 reference values, with
72,625 nonzero outputs and maximum absolute magnitude 0.216797.

Tensor lowering now assigns independent physical slices to Gate weights, Up
weights, and the Block8 activation. The planner can therefore pipeline Gate on
MXM0 and Up on MXM1 with a four-cycle weight-load initiation interval instead
of serializing each reduction for eight cycles. Scratch storage reuses weight
slices only at non-overlapping row addresses and its queue lifetime is reserved
explicitly. The current CModel still shares E0..E15 between MXM0 weight traffic
and activation traffic, so `mxm_weight_activation_overlap_enabled=0` keeps the
next load group outside the activation window. Independent placement removes a
further 864 cycles from both policies; a target with independent transport can
enable the capability without changing the planner.

The legacy MXM path remains available, but a fused request currently falls
back to tail because its passive stream-fabric transport lifetimes are not yet
modeled precisely enough to prove a collision-free allocation.

![Fused seq_len=128 FFN pipeline](smollm2_ffn_fused_pipeline.svg)

## Historical Legacy Measurements

The following archived values describe the earlier legacy fused experiment,
not the active Block8 policies above. The monitor includes the runtime's 64
drain cycles, so it samples 92,637 cycles for tail and 86,749 for fused.
`array utilization` is active MXM cell-cycles divided by all sampled
cell-cycles. `active density` removes completely idle cycles from the
denominator.

| MXM | Active cycles (tail/fused) | Tail array util. | Fused array util. | Tail active density | Fused active density |
| --- | ---: | ---: | ---: | ---: | ---: |
| MXM0 | 86,052 / 86,052 | 92.85% | 99.16% | 99.96% | 99.96% |
| MXM1 | 86,052 / 86,052 | 92.85% | 99.16% | 99.96% | 99.96% |
| MXM2 | 79,902 / 79,902 | 86.22% | 92.07% | 99.96% | 99.96% |
| MXM3 | 79,902 / 79,902 | 86.22% | 92.07% | 99.96% | 99.96% |
| Four-MXM mean | - | 89.54% | 95.62% | 99.96% | 99.96% |

The fused policy performs the same MXM cell work in fewer total cycles. This
raises mean full-program MXM utilization by 6.08 percentage points without
changing the nearly saturated density while an MXM is active.

| Resource | Policy | Full-program util. | Active density | Stall rate | Peak |
| --- | --- | ---: | ---: | ---: | ---: |
| VXM ALUs | tail | 15.91% | 74.19% | 0.00% | 512/512 |
| VXM ALUs | fused | 16.99% | 72.67% | 0.00% | 512/512 |

VXM utilization uses 512 lane-ALU execution slots per cycle as capacity.
Fused scheduling leaves the executed work unchanged but spreads it over 414
more non-idle VXM cycles, so full-program utilization rises while active
density falls slightly.

| SR fabric | Policy | Link BW | East BW | West BW | Staged-write util. | Active density | Peak link bytes/cycle |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| East hemisphere | tail | 4.94% | 5.57% | 4.31% | 5.90% | 4.94% | 4,928/24,576 |
| East hemisphere | fused | 5.24% | 5.95% | 4.53% | 6.30% | 5.24% | 6,528/24,576 |
| West hemisphere | tail | 4.57% | 5.11% | 4.03% | 5.47% | 4.91% | 4,928/24,576 |
| West hemisphere | fused | 4.85% | 5.46% | 4.23% | 5.84% | 5.23% | 6,528/24,576 |

SR bandwidth utilization is normalized by each hemisphere's modeled SR-fabric
link capacity, not merely by cycles containing traffic. CModel does not yet expose
capacity-normalized MEM or SXM utilization counters, so those values are not
inferred from the schedule diagram.

Accumulator banks remain on separate rows, while color describes the
operation: purple means `accumulate -> SRAM`, and red means
`accumulate -> stream + clear`.

This is one instantiation of the generic W8A16 FFN lowering implemented by
`ftlpu-stream-to-schedule`. The compiler derives loop counts, physical rows,
slice placement, transport latency, and cycle intervals from the IR shape and
`LPUTargetModel`; it contains no SmolLM2 shape branch.

For this `32x576x1536x576` instance, consecutive K blocks start every 38 cycles
and alternate weight buffers 0/1. Each block issues 32 MXM compute rows, giving
84.2% row-issue occupancy; the remaining six cycles are the modeled MXM
pipeline/control constraint. The last command is issued at cycle 34,270,
compared with 35,743 before projection/SwiGLU overlap and 87,150 before MXM
pipelining.

For `M = 32*T`, the projection loop is `N-tile -> K-tile -> M-tile`. A
`32x32` gate/up weight tile is loaded and dequantized once, then stays in its
MXM weight buffer while it processes all `T` activation tiles. For example,
`seq_len=128` has `T=4`: one weight load is followed by four `M=32` computes
and four independent accumulator address ranges. This is general M tiling,
not a sequence-length-specific path; the `M=32` diagram below is its `T=1`
instance.

All MEM slice numbers in this document are local to one hemisphere. Thus gate
accumulators use local slices 36..39 and up accumulators use local slices
40..43 in both hemispheres. In the CModel's flattened 88-queue view, east
uses queue `local_slice`, while west uses queue `44 + local_slice`.

The compiler now has two complete Gate/Up execution paths. `legacy` keeps the
two-byte-stream vector compute used by the diagrams below. `auto` selects the
Block8 path for a legal BF16-activation/INT8-weight FFN: sixteen distributed
activation streams carry eight rows per issue, and the same E0..E15 values are
multicast to the Gate and Up MXMs in both hemispheres. Four repeated Block8
issues cover one 32-row token tile. Raw weights use eight streams and are
dequantized locally in each MXM before IW.

Gate and Up cannot drain their 8x32 FP32 block accumulators simultaneously
because either result occupies all 32 west streams. The conservative schedule
drains them in turn, casts them into two disjoint short-lived BF16 scratch
layouts, and then runs SwiGLU row by row. Input and hidden tensors use disjoint
sixteen-slice physical layouts so an early hidden write cannot overwrite an
activation that a later output wave still needs. For SmolLM2-135M at
`seq_len=128`, this path reduces the complete FFN from 105,898 legacy cycles to
38,968 cycles while preserving the CModel numerical baseline. The two MXM
weight buffers now ping-pong between adjacent K blocks: the next weight load
uses the idle issue window before the current block's final token tile, so the
next repeated Block8 compute starts immediately after the current one ends.

![All 18 K partial accumulations](smollm2_ffn_partial_accumulation.svg)

The timeline expands one hemisphere and one `M=32, N=32` output tile. It shows
all `P0..P17` writes into the same gate/up accumulator tile and the resulting
states `S0..S17`; the west hemisphere uses the same cadence with distinct
physical MEM/SREG identities.

The expanded SwiGLU section shows the actual VXM micro-schedule. At cycle `t`,
the two VXM paths start `-gate` and `gate * up`. The sigmoid path then executes
`exp`, add-one, and reciprocal in cycles `t+1..t+3`, while pass-through commands
keep `gate * up` aligned. Cycle `t+4` multiplies both paths, producing
`sigmoid(gate) * (gate * up)`, which is exactly `SiLU(gate) * up`. The result is
cast to FP16 at `t+5` and written to local hidden slices 21/22/23/29 after the modeled
transport and MEM-placement latency.

A single 32x32 MXM weight-tile dequantization takes four issue cycles: the VXM
handles eight columns per cycle. Gate/up on east/west form four independent
weight tiles. They are currently staggered into a 16-cycle aggregate issue
window because all four use the same 16 VXM ALU ICU queues; this is four fast
dequantizations, not one 16-cycle dequantization. Down uses the same continuous
four-cycle dequantization and 16-stream IW load, including weights prefetched
under the previous reduction's final M tile. Activation switches between
`E0/E1` and `E16/E17` only during the actual IW conflict windows.

The second panel uses one continuous time axis for projection completion and
the first SwiGLU work. In the tail figure, all Gate/Up projection pairs finish,
the final partial remains in accumulator SRAM, and `W0..W7` then feed SwiGLU.
In the fused figure, the same panel follows one completed accumulator tile
through `stream + clear`, temporary MEM staging, and SwiGLU while later
projection work remains in flight.

The current tail uses `W0..W7` for accumulator input and local hidden slices
21/22/23/29 for output. It deliberately does not claim projection/SwiGLU
overlap until the scheduler reserves all shared VXM, MEM, stream, and transport
resources at precise cycles.
