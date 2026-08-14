# ICU Iteration Design Review

Generated schedules contain two distinct kinds of repetition:

- one functional instruction repeated over regular inner/outer coordinates;
- an exact multi-instruction window replayed as a group.

They should remain separate ISA mechanisms. `Repeat2D` handles the first case,
while `Loop` handles the second.

## Implemented Decision

`Repeat2D` is a 96-bit, queue-local, blocking iterator of the immediately
preceding functional instruction. Counts include the already-issued base point
and remaining points issue in outer-major, inner-minor order. Its typed
induction target is either a MEM address, an MXM IW weight column, or none.

Blocking semantics avoid an unbounded set of active iterator contexts in ICU.
The binary emitter uses Repeat2D only when no same-queue event interleaves the
complete two-dimensional interval. It expands the outer dimension otherwise.
This legality check is a linear scan over cycle-sorted queue sequences.

The compiler capability `throughput.icu_repeat_2d_enabled` can force expansion
for hardware without Repeat2D and for semantic A/B testing.

## Measured Result

For the SmolLM2 sequence-128 Block8 fused FFN:

| Form | Binary size | max_cycle | Dynamic issue counts |
| --- | ---: | ---: | --- |
| outer dimensions expanded | 17,650,899 bytes | 21,095 | reference |
| blocking Repeat2D | 12,865,201 bytes | 21,095 | identical |

This is a 27.1% binary-size reduction. Both forms also produced identical
numeric output. An earlier 8.76x estimate assumed non-blocking iterators with
many simultaneously live contexts and is not applicable to the implemented
hardware semantics.

Qwen and complete-decoder occupancy should be remeasured after the current
CModel VXM compact-instruction migration stabilizes.

See [icu_repeat_2d.md](icu_repeat_2d.md) for the exact semantics and compiler
fallback policy.

