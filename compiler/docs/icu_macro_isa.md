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
then emits binary format version 22.

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

The binary v21 prototype stores a `MACR` tag and the eight fields in extension
words attached to the native payload. This variable-length envelope validates
the semantics before a fixed-width RTL encoding is frozen.

The issue point `(outer, inner)` is:

```text
cycle = start_cycle + outer * outer_interval + inner * inner_interval
operand_delta = outer * outer_stride + inner * inner_stride
```

## Compiler Lowering

Binary lowering recognizes repeated queue windows with identical native
instruction shape, fixed cycle interval, and fixed operand stride. It folds
them into interleaved two-dimensional macros. This is queue-generic and is not
an FFN-specific lowering rule.

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

## Local ICU

Each MEM/MXM ICU has a descriptor FIFO and a next-issue calendar. Multiple
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
