# Legacy ICU Repeat2D Compiler Design

This document describes the legacy `CommandSequence` compatibility path.
Hardware-facing direct lowering emits FU-specific `READ_3D`, `WRITE_3D`,
`LOAD_3D`, `DEQUANT_3D`, `COMPUTE_3D`, and `RUN_2D` packets from operator
domains and does not run this reconstruction policy.

`Repeat2D` compresses a regular two-dimensional iteration of one functional
instruction. Command IR keeps its existing `repeat_*`, `wave_*`, and
`group_*` fields. The binary emitter maps at most two active dimensions to a
96-bit descriptor, without introducing model-specific lowering.

## Semantics

The functional instruction immediately before the descriptor is coordinate
`(0, 0)` and has already issued. The descriptor emits the remaining points in
outer-major, inner-minor order:

```text
cycle(o, i) = base_cycle + o * outer_interval + i * inner_interval
delta(o, i) = o * outer_stride + i * inner_stride
```

Both counts include the base point. Repeat2D blocks its local queue until the
iteration completes; other ICU queues continue independently. The compiler
therefore uses it only when no same-queue event interleaves its complete time
range. Otherwise it expands the outer dimension.

Typed induction targets are `None`, `MemAddress`, and `MxmWeightColumn`.
MEM induction updates the read address and, for `ReadWrite`, its write address.
MXM induction is valid only for an `IW` weight column.

## Compiler Policy

The legacy binary emitter preserves inner and outer dimensions in queue-local
`CommandSequence` objects. It selects blocking Repeat2D only when the complete
range is already legal. Interleaved outer waves are rejected and must be
replaced by direct operator lowering; they are not split after scheduling.
`throughput.icu_repeat_2d_enabled = 0` forces expansion for hardware without
this capability and for compression-equivalence testing.

Dual-port MEM merging requires identical inner and outer iteration shapes.
Different shapes are expanded before read/write pairing so that neither port
inherits the other port's address induction.

## Validation

- CModel codec, MEM induction, and MXM-column unit tests pass.
- Command IR to binary to runtime to CModel ICU compatibility passes.
- Interleaved same-queue events are rejected at the hardware boundary.
- Repeat2D and fully expanded SmolLM2 sequence-128 Block8 fused FFN binaries
  have identical `max_cycle=21095`, dynamic issue counts, and numeric output.
- The measured binary size falls from 17,650,899 to 12,865,201 bytes, a 27.1%
  reduction.

After the recent CModel C2C-SXM register-column change, both expanded and
Repeat2D FFN binaries expose the same pre-existing fused-FFN numeric mismatch;
it is not a Repeat2D semantic difference.

