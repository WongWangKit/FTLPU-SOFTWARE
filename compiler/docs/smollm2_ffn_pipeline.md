# Generic W8A16 FFN Pipeline: SmolLM2-135M Instance

![SmolLM2-135M FFN pipeline](smollm2_ffn_pipeline.svg)

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

An MXM weight load uses 16 streams. One MXM compute consumes two FP16
activation streams; the concurrent gate/up MXM pair therefore consumes E0..E3.
A 32x32 K block streams 32 activation rows, with four spatial K=8 contributions
accumulated per output row, and produces one 32x32 partial tile. The 18 partial
tiles for K=576 accumulate in MEM before the completed gate/up tile enters
SwiGLU. When loading block B+1 overlaps block B compute, the activation route
temporarily moves from E0..E3 to E16..E19. Down output-pair transitions are 95
cycles for this placement because result writes and the next weight read both
use local MEM slice 24 in each hemisphere; that interval is calculated from the conflicting slice and
transport latency.

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
dequantizations, not one 16-cycle dequantization.

The second panel is deliberately a trace of the current lowering, rather than
the desired direct-transport design. Its final K partial (`P17`) is still
accumulated into local SRAM, then read through `W0..W7`. There is currently no
fixed-size SwiGLU overlap batch: the compiler conservatively waits for all
projection pairs to complete, then starts the first row/hemisphere Swish event
at cycle 16,469 and issues the remaining events sequentially. A future
accumulator-to-stream lowering must be shown as a separate schedule, once its
dual-hemisphere VXM resource policy is defined.

The current tail uses `W0..W7` for accumulator input and local hidden slices
21/22/23/29 for output. It deliberately does not claim projection/SwiGLU
overlap until the scheduler reserves all shared VXM, MEM, stream, and transport
resources at precise cycles.
