# ICU Macro Scheduling ISA v1

## Scope

The v1 implementation moves deterministic MEM/MXM command expansion from the
host runtime into each functional-unit ICU. It does not change Schedule IR
cycles or native MEM/MXM instructions. VXM and SXM keep the legacy queue format
for now.

Enable it with:

```text
ftlpu-opt ... --icu-macro-schedule
```

The generated module carries `ftlpu.icu_macro_schedule = true`; binary lowering
then emits the current binary envelope with typed extended ICU descriptors.

## Descriptor

Each macro owns one native functional instruction plus this schedule:

| Field | Meaning |
| --- | --- |
| `start_cycle` | Absolute first issue cycle |
| `inner_count` | Issues in one inner repeat |
| `inner_interval` | Cycles between inner issues |
| `inner_stride` | Native operand induction per inner issue |
| `outer_count` | Number of repeated waves |
| `outer_interval` | Cycles between wave starts |
| `outer_stride` | Native operand induction per wave |
| `induction_target` | MEM address, MXM weight column, MXM accumulator address, or none |

The binary prototype stores a `MACR` tag and the eight fields in extension
words attached to the native payload. This variable-length envelope validates
the semantics before a fixed-width RTL encoding is frozen.

The issue point `(outer, inner)` is:

```text
cycle = start_cycle + outer * outer_interval + inner * inner_interval
operand_delta = outer * outer_stride + inner * inner_stride
```

### MEM_STREAM_ND

MEM queues additionally use `MEM_STREAM_ND`, a MEM-specific descriptor with
one native read/write and up to three affine counters:

| Field | Meaning |
| --- | --- |
| `start_cycle` | Absolute cycle of coordinate `(0, 0, 0)` |
| `rank` | Number of active dimensions, from 1 through 3 |
| `count[d]` | Number of points in dimension `d` |
| `cycle_stride[d]` | Cycle increment for one step in dimension `d` |
| `address_stride[d]` | SRAM row increment for one step in dimension `d` |

Dimension zero is innermost. For coordinate `(i0, i1, i2)`, the MEM ICU
issues the native transfer at:

```text
cycle   = start_cycle + sum(id * cycle_stride[d])
address = base_address + sum(id * address_stride[d])
```

Several descriptors may be active at once. A per-queue next-issue calendar
interleaves them, which preserves irregular phase boundaries while letting one
descriptor cover regular token, block, and page dimensions. Counts and cycle
strides are unsigned 32-bit values in the current binary envelope; address
strides are signed 32-bit values. Dimensions must not overlap in issue time.

### MXM_STREAM_ND

MXM load, compute, and dequant queues use the same one-to-three-dimensional
schedule around one native MXM instruction. `operand_stride[d]` is interpreted
by the typed `induction_target`:

| Queue/native opcode | Induction target |
| --- | --- |
| MXM load / `IW` | Weight column |
| MXM compute / `Compute` or `AccumulatorRead` | Accumulator address |
| MXM dequant | None; only the issue cycles repeat |

All other native fields remain invariant, including weight buffer, activation
and output streams, accumulator destination, clear flag, and output format.
Consequently, a final partial using `accumulator_destination = stream` and
`accumulator_clear = true` remains distinct from an ordinary accumulate
descriptor.

```text
cycle         = start_cycle + sum(i[d] * cycle_stride[d])
operand_delta =               sum(i[d] * operand_stride[d])
```

The compiler sorts affine dimensions by cycle stride and verifies that nested
hardware counters can emit them monotonically. Multiple descriptors can still
interleave through the per-queue next-issue calendar.

### VXM_STREAM_ND

`VXM_STREAM_ND` carries one 96-bit compact VXM packet and a one-to-three-
dimensional absolute-cycle launch domain in one ICU macro instruction. The ICU
latches the packet once, then generates all launch points with its N-D counters.
There is no configuration-slot state or CONFIG/RUN pairing to manage.

The N-D iteration count is the number of ICU launches. The `repeat_count`
inside the compact packet is independently retained and controls the duration
of each Superlane configuration. Version 1 has no operand induction. A scale
relocation directly patches the packet carried by the macro instruction.

### SXM_TILE_PROGRAM

`SXM_TILE_PROGRAM` stores one complete transpose or permute template together
with a one-to-three-dimensional launch domain. Its payload preserves source
and destination stream lists, row/tile selectors, and the full 32-lane map.
Transpose and permute remain separate per-hemisphere queues and can overlap.
Recurring tile maps become separate interleaved programs rather than expanded
per-cycle instructions.

## Compiler Lowering

Binary lowering recognizes repeated queue windows with identical native
instruction shape, fixed cycle interval, and fixed operand stride. It first
forms interleaved two-dimensional schedules, then folds repeated schedules into
a third affine dimension. This is queue-generic and is not an FFN-specific
lowering rule.

Compression is represented before binary lowering as well:

- Schedule `mem_read`/`mem_write` use address waves.
- Schedule `mxm_load` uses an outer group for repeated IW windows.
- Schedule `mxm_compute` and `mxm_accumulate` share a wave whose induction
  variable may advance the accumulator address.
- Command `mem_bundle` stores all physical slice lanes once, while each lane
  retains its own queue, cycle, address, and stream selector.

Schedule verification expands every logical point when checking resource
occupancy. The compact form therefore does not relax cycle accuracy.

For the SmolLM2-135M sequence-32 Vector FFN test:

| Metric | Legacy queue | Macro v1 | Reduction |
| --- | ---: | ---: | ---: |
| Queue commands | 53,511 | 6,214 | 88.4% |
| Binary bytes | 1,620,663 | 412,565 | 74.5% |
| Scheduled `max_cycle` | 46,421 | 46,421 | unchanged |

For the real-weight Qwen2.5-1.5B layer-0 FFN at sequence length 32, using one
Vector MXM per hemisphere and the 128 KiB-per-superlane target:

| Artifact | Expanded | Hierarchical | Reduction |
| --- | ---: | ---: | ---: |
| Schedule IR | 510,631,417 B | 5,199,998 B | 99.0% |
| Command IR | 243,032,926 B | 2,248,327 B | 99.1% |
| Binary | 28,947,545 B | 885,357 B | 96.9% |
| Scheduled `max_cycle` | 770,646 | 770,646 | unchanged |

The compressed binary passed the CModel golden test over 49,152 BF16 outputs
with zero mismatches. It keeps the previous MAE (`6.54569e-05`) and maximum
error (`0.00418091`).

For the complete real-weight Qwen2.5-1.5B layer-0 decoder at sequence length
32, cumulative MEM and MXM N-D compression gives:

| Metric | 2-D macros | MEM N-D | MEM + MXM N-D |
| --- | ---: | ---: | ---: |
| MEM descriptors | 25,458 | 4,652 | 4,652 |
| MXM descriptors | 3,221 | 3,221 | 196 |
| All queue commands | 36,397 | 15,591 | 12,566 |
| Binary bytes | 1,752,208 | 1,053,968 | 945,118 |
| Expanded functional issues | 6,586,470 | 6,586,470 | 6,586,470 |
| Scheduled `max_cycle` | 763,669 | 763,669 | 763,669 |

Relative to the MEM-only N-D binary, `MXM_STREAM_ND` removes 93.9% of MXM
descriptors, 19.4% of all queue commands, and 10.3% of binary bytes. Relative
to the original 2-D macro binary, total command and byte reductions are 65.5%
and 46.1%.

The resulting binary completed in 763,733 modeled cycles and passed all 49,152
BF16 outputs with zero mismatches (MAE `0.004514`, maximum error `0.09375`).

Adding `VXM_STREAM_ND` and `SXM_TILE_PROGRAM` to the same decoder gives:

| Metric | MEM + MXM N-D | All coarse ICU forms | Reduction |
| --- | ---: | ---: | ---: |
| VXM encoded commands | 5,536 | 85 stream descriptors | 98.5% |
| SXM encoded commands | 2,182 | 112 tile programs | 94.9% |
| All queue commands | 12,566 | 5,045 | 59.9% |
| Binary bytes | 945,118 | 347,235 | 63.3% |
| Expanded functional issues | 6,586,470 | 6,586,470 | unchanged |
| Scheduled `max_cycle` | 763,669 | 763,669 | unchanged |

The coarse binary also passed the same 49,152-output CModel golden comparison
with zero mismatches, MAE `0.004514`, and maximum error `0.09375`.

## Local ICU

Each functional-unit ICU has a descriptor FIFO and a next-issue calendar. Multiple
descriptors may be in flight so independent waves can interleave. The CModel
uses a min-heap keyed by next issue cycle; RTL can use a small ordered calendar,
timing wheel, or bounded comparator tree.

The ICU rejects a missed issue cycle, two descriptors issuing to the same queue
in one cycle, an invalid iteration space, or a mixed legacy/macro queue while a
macro is active. The compiler also checks same-queue issue collisions before
writing the binary.

## RTL Encoding Direction

The current extension-word representation is a software validation format, not
the final wire width. A hardware encoding should use a fixed descriptor header,
FU-specific payload words, explicit descriptor length/version, and a bounded
in-flight count exposed through the target model. Long intervals/counts can use
an escape word. Runtime must retain the legacy queue path until the hardware
capability bit selects macro v1.
