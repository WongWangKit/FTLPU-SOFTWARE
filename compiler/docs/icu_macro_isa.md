# FU-Specific ICU Loop ISA and Legacy Macro Compatibility

## Scope

This document distinguishes the current hardware-visible FU-specific loop
instructions from the older `Macro` and `STREAM_ND` serialized forms. A
projection is compiler intent; it is not an ICU instruction or a hardware
program object. Each physical ICU receives only instructions for its own
functional unit.

The current raw local-iMEM formats are:

| Physical queue | FU-local instruction | One logical packet |
| --- | --- | ---: |
| MEM | `READ_3D`, `WRITE_3D`, `WRITE_TAP_3D` | 3 consecutive 96-bit words |
| MEM | `WRITE_READ_2D` | 3 consecutive 96-bit words |
| MEM | `MEM_READ_SYNC` / `MEM_WRITE_SYNC` | 2 consecutive 96-bit words |
| MXM load | `LOAD_3D` | 2 consecutive 128-bit words |
| MXM dequant | `DEQUANT_3D` | 2 consecutive 128-bit words |
| MXM compute | `COMPUTE_3D`, `ACCUMULATOR_READ_3D` | 2 consecutive 128-bit words |
| VXM | `RUN_2D` plus a 96-bit compact config | 3 consecutive 96-bit words |
| SXM | `RUN_2D` plus a 416-bit tile-local config | 6 consecutive 96-bit words |

The queue type identifies the decoder, so these packet families do not carry a
shared MEM/MXM unit selector and do not use a common 320-bit layout. Every
local-iMEM entry is one physical 96-bit or 128-bit word. A complete packet
therefore consumes three MEM-3D/VXM slots, two MEM-write-sync/MXM slots, or six
SXM slots.

`MEM_SLICE_PROGRAM` does not yet have a frozen hardware packet layout.
`VXM_STREAM_ND` and `SXM_TILE_PROGRAM` remain legacy file forms, but the subset
with rank at most two and no operand induction converts directly to the
corresponding `RUN_2D` without expanding per-cycle instructions.

`ScheduleToCommand` now encodes the VXM and SXM domains declared by each operator
directly as physical `command.vxm_run_2d` and `command.sxm_run_2d` packets. A
contiguous VXM run (`repeat_interval=1`) stays in the compact config; a spaced
repeat uses the first ICU counter. A legacy VXM queue that needs runtime scale
relocation remains wholly on the compatibility path so one physical queue
never mixes raw packets with legacy commands.

## Why loop dimensions differ by FU

Each ICU carries only the independent state its functional unit needs. There is
no universal four-dimensional decoder:

| Functional unit | Loop form | Work represented by the dimensions |
| --- | --- | --- |
| MEM | Up to 3-D | Contiguous block, tile, and outer group, with SRAM-address induction |
| MXM | Up to 3-D | K partial, M row/tile, and N tile, with weight-column or accumulator induction |
| VXM | Compact `repeat_count` plus ICU `RUN_2D` | Contiguous elements inside the compact config; outer row/token and tile/head/group launches in the ICU |
| SXM | Tile-local instruction plus ICU `RUN_2D` | The 32-lane map stays inside the tile instruction; the ICU repeats rows and tiles |
| C2C | Burst/vector count plus completion event | Transfer length and vector batches; completion events express producer/consumer dependencies |

MEM and MXM therefore retain three dimensions. VXM does not encode a third
counter that would always be one. SXM does not treat its lane map as a loop
dimension, and C2C uses events for dataflow dependencies. If an operator has a
fourth logical axis, lowering folds it into an existing axis when fields allow,
or emits another coarse instruction at a page, placement, or FU-template
boundary. A universal 4-D form would enlarge every ICU context and its counter
and address-generation paths while extending context occupancy, with no
corresponding need in the current LLM kernels.

## Legacy compiler modes

The standard `ftlpu-stream-to-schedule` pipeline is a direct-lowering
pipeline, not a compression pipeline. It forces
`ftlpu.icu_compression = "none"`, disables `MEM_SLICE_PROGRAM`, and marks the
result with `ftlpu.command_lowering = "direct"`. The
`ftlpu-stablehlo-to-schedule` and `ftlpu-stream-to-uncompressed-schedule`
entry points use the same direct contract.

`ScheduleCompression` is added only by the explicitly selected compatibility
pipelines `ftlpu-stream-to-compressed-schedule` and
`ftlpu-compress-schedule`. The compression option below controls those legacy
pipelines; it is not a post-pass behind the standard schedule pipeline.

Select ICU compression from the command line:

```text
ftlpu-opt ... --icu-compression none|control|macro
```

`macro` is the default legacy serialization mode. `none` materializes
functional instructions while retaining duration-encoded NOP gaps, `control`
enables Repeat and Repeat2D, and `macro` enables the older typed Macro
envelope. The old `--icu-macro-schedule` option remains an alias for
`--icu-compression macro`. These options describe the compatibility pipeline;
they do not select a shared hardware packet.

The generated module carries `ftlpu.icu_compression = "..."`. It also carries
the legacy `ftlpu.icu_macro_schedule` boolean during the command-IR
compatibility window. Binary lowering may retain typed `Macro` and
`STREAM_ND` descriptors in the file envelope. After relocation, the runtime
loader converts the supported MEM/MXM subset in closed form into the same
FU-specific raw packets listed above. It does not materialize per-point native
instructions.

## Legacy descriptor semantics

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

The binary compatibility format stores a `MACR` tag and the eight fields in
extension words attached to the native payload. This variable-length record is
a legacy disk envelope, not a hardware instruction.

The issue point `(outer, inner)` is:

```text
cycle = start_cycle + outer * outer_interval + inner * inner_interval
operand_delta = outer * outer_stride + inner * inner_stride
```

### Legacy MEM_STREAM_ND

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

Counts and cycle strides are unsigned 32-bit values in the legacy binary
envelope; address strides are signed 32-bit values. Dimensions must not overlap
in issue time. During loading, the adapter also checks the narrower current
packet fields and the FU-specific live-context capacity.

### Legacy MEM_SLICE_PROGRAM

`MEM_SLICE_PROGRAM` combines up to sixteen MEM operations for one physical
slice under a shared N-D launch domain. Each body entry retains its native
read/write instruction, relative cycle offset, and three affine address
strides. For body `b` and launch coordinate `i`:

```text
cycle   = start_cycle + cycle_offset[b]
          + sum(i[d] * cycle_stride[d])
address = native[b].address + sum(i[d] * address_stride[b][d])
```

Binary lowering groups only sequences with identical rank, counts, cycle
strides, and binding metadata. It validates all expanded issue cycles before
forming a program. One relocation references the parent command and runtime
applies its delta to every body instruction. The current compiler limits a
body cycle offset to 65,535 cycles so a later fixed-width RTL encoding remains
practical.

### Legacy MXM_STREAM_ND

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
hardware counters can emit them monotonically. Descriptors on one physical
queue have disjoint live intervals because that queue owns one decoded loop
state.

## Current FU-specific raw packet encoding

Every MEM/MXM packet uses a common closed-form 3-D loop. Counter 0 is innermost:

| Logical payload field | Width | Meaning |
| --- | ---: | --- |
| `wait_cycle` | 24 | Queue-local delay before this packet's first FU issue |
| `count[d] - 1` | 16 each | Iteration count for counters 0, 1, and 2 |
| `cycle_stride[d]` | 24 each | Cycle increment for one step of counter `d` |

For coordinate `i`, the ICU compares an instruction-local elapsed counter with:

```text
issue_offset = wait_cycle + sum(i[d] * cycle_stride[d])
```

The ICU has no program-global cycle input. Command IR keeps absolute cycles as
compiler metadata, and binary emission places each queue gap in the next
packet's `wait_cycle` when it fits 24 bits. A longer gap uses a preceding
`NOP` for the excess. Counts range from 1 through 65,536.
Cycle strides are nonzero and the final relative issue offset must fit the
24-bit schedule domain. An inactive dimension uses
`count=1`; its stride is not observed because the counter never advances.

Each physical word repeats an eight-bit FU-local header:

| Physical bits | Meaning |
| ---: | --- |
| `[1:0]` | Global ICU envelope opcode `Extended` (`0b11`) |
| `[3:2]` | Word index within this fixed-size packet |
| `[5:4]` | FU-local operation |
| `[6]` | FU-local 3-D marker, `1` |
| `[7]` | Format version, currently `0` |

Word 0 also carries extended subtype `2` at physical bits `[91:88]`. This
separates FU 3-D packets from Repeat2D. Continuation words repeat the marker,
version, operation, and expected word index, so truncation or reordering is
detected by the local decoder.

The tables below number logical payload bits as `P[n]`. Payload numbering omits
the low-byte header in every word and the word-0 subtype field. The common loop
occupies `P[143:0]`:

| Payload bits | Field |
| ---: | --- |
| `P[23:0]` | `wait_cycle` before the first launch |
| `P[39:24]`, `P[55:40]`, `P[71:56]` | `count[0..2] - 1` |
| `P[95:72]`, `P[119:96]`, `P[143:120]` | `cycle_stride[0..2]` |

### MEM: 3 x 96 bits

The local operation is `0=read`, `1=write`, or `2=write-tap`. The 260 payload
bits are:

| Payload bits | Field |
| ---: | --- |
| `P[143:0]` | Common loop |
| `P[149:144]` | Packed stream selector |
| `P[162:150]` | Bank-local base row |
| `P[178:163]` | `outer_group_size - 1` |
| `P[198:179]` | Signed inner address stride |
| `P[218:199]` | Signed middle address stride |
| `P[238:219]` | Signed outer-within-group address stride |
| `P[258:239]` | Signed outer-group address stride |
| `P[259]` | Reserved, zero |

`outer_group_size` must be a power of two. A value of one gives an ordinary
affine three-counter address. Larger values implement a blocked outer layout
with a shift and mask.

For coordinates `(i0,i1,i2)`, the blocked address generator computes:

```text
address = base + i0*a0 + i1*a1
        + (i2 & (G - 1))*a2 + (i2 >> log2(G))*ag
```

The cycle strides also encode the waits at loop boundaries; the packet does
not need internal NOP instructions. With counts `(C0,C1,C2)` and cycle strides
`(S0,S1,S2)`, the idle cycles after an ordinary inner issue, after the end of
an inner loop, and after the end of a middle loop are respectively
`S0-1`, `S1-(C0-1)*S0-1`, and
`S2-(C1-1)*S1-(C0-1)*S0-1`. These values must be non-negative. Queue NOPs are
only needed before the packet or between domains that cannot share one affine
loop.

### MEM WRITE_READ_2D: 3 x 96 bits

This packet runs two statically scheduled two-dimensional event streams on
one physical MEM bank. Each `(i0,i1)` is written once and read once. Both
events use the same SRAM address domain but have independent timing:

```text
write_cycle  = start_wait + i0*write_cycle_stride0 + i1*write_cycle_stride1
read_cycle   = start_wait + read_start_offset
             + i0*read_cycle_stride0 + i1*read_cycle_stride1
address      = base_address + i0*address_stride0 + i1*address_stride1
write_stream = fixed_write_stream
read_stream  = read_stream_base + i1*read_stream_outer_stride
```

Cycles are relative to packet activation, not to a global clock. `start_wait`
is an in-packet initial delay; longer gaps still use queue NOPs. The read
stream stride permits the outer coordinate to select a different output
stream. The ICU keeps separate two-dimensional write/read cursors and next
issue times but issues at most one MEM FU operation per bank per cycle. The
compiler proves single-port disjointness, read-after-write and no overwrite
before read, and stream availability at scheduled cycles. Hardware neither
checks FIFO occupancy nor changes timing based on data arrival.

The three-word packet uses MEM-local operation `3` and extended subtype `7`.
Its 260 payload bits are:

| Payload bit | Field |
| ---: | --- |
| `P[23:0]` | `start_wait` |
| `P[39:24]`, `P[55:40]` | `count[0..1] - 1` |
| `P[79:56]`, `P[103:80]` | Write cycle strides 0 and 1 |
| `P[127:104]`, `P[151:128]` | Read cycle strides 0 and 1 |
| `P[175:152]` | `read_start_offset` |
| `P[188:176]` | Bank-local base row |
| `P[208:189]`, `P[228:209]` | Signed 20-bit address strides 0 and 1 |
| `P[234:229]`, `P[240:235]` | Write stream and read stream base |
| `P[246:241]` | Signed 6-bit read stream outer stride |
| `P[259:247]` | Reserved zero |

The fixed three-word size requires equal write/read iteration counts, a
shared address formula, and exactly one write and one read per coordinate.
Independent address domains, repeated reads, or multiple write sources are
not implicit in this instruction.

### MEM_READ_SYNC / MEM_WRITE_SYNC: 2 x 96 bits

This command is decoded by the same per-bank MEM ICU as the 3-D commands above.
Word 0 uses extended subtype 6 and word 1 is a native MEM `Read` or `Write`
template. The native opcode distinguishes `MEM_READ_SYNC` from
`MEM_WRITE_SYNC`; both use the same fixed two-word format:

| Word-0 physical bits | Field |
| ---: | --- |
| `[1:0]` | Extended envelope |
| `[17:2]` | Vector count minus one |
| `[33:18]` | C2C synchronization tag |
| `[49:34]` | SR transport delay |
| `[63:50]` | Signed SRAM row stride |
| `[87:64]` | Reserved queue-window cycles minus one |
| `[91:88]` | Extended subtype 6 |
| `[95:92]` | Reserved, zero |

After activation, the command owns the MEM ICU for at least its reserved
window. For `MEM_WRITE_SYNC`, every matching C2C RX token becomes one native
write after the encoded SR delay. `MEM_READ_SYNC` drives MEM-to-DDR page-out:
a C2C TX lane sends one matching token only when its input pipeline has room,
and each token becomes one native read. TX, DMA, and DDR backpressure therefore
propagates to MEM instead of releasing a whole burst at once. If work finishes
early, the ICU still holds the command until the window ends; late work retires
only after the final read or write. Later MEM commands cannot bypass it.

### MXM load: 2 x 128 bits

| Payload bits | Field |
| ---: | --- |
| `P[143:0]` | Common loop |
| `P[144]` | Weight-buffer base |
| `P[146:145]` | Buffer parity mode |
| `P[148:147]` | Weight-column base |
| `P[164:149]`, `P[180:165]`, `P[196:181]` | Signed column strides 0..2 |
| `P[201:197]` | East weight-input stream base |
| `P[202]` | Input mode: INT8 dequant or direct 16-bit |
| `P[235:203]` | Reserved, zero |

### MXM dequant: 2 x 128 bits

| Payload bits | Field |
| ---: | --- |
| `P[143:0]` | Common loop |
| `P[159:144]` | BF16 scale bits |
| `P[235:160]` | Reserved, zero |

### MXM compute: 2 x 128 bits

The MXM compute ICU decodes two FU-local operations from the same fixed-size
packet family: `0=COMPUTE_3D` and `1=ACCUMULATOR_READ_3D`. Because the operation
is repeated in each physical-word header, the two words start with `0x43` and
`0x47` for `COMPUTE_3D`, or `0x53` and `0x57` for
`ACCUMULATOR_READ_3D`.

`COMPUTE_3D` uses this payload:

| Payload bits | Field |
| ---: | --- |
| `P[143:0]` | Common loop |
| `P[144]` | Weight-buffer base |
| `P[146:145]` | Buffer parity mode |
| `P[151:147]` | East activation-stream base |
| `P[156:152]` | West result-stream base |
| `P[169:157]` | Accumulator base row |
| `P[183:170]`, `P[197:184]`, `P[211:198]` | Signed accumulator strides 0..2 |
| `P[224:212]` | Accumulator row stride |
| `P[225]` | Weight/activation data format |
| `P[228:226]` | Regular destination/clear/output-format mode |
| `P[230:229]` | Terminal dimension; `3` disables it |
| `P[233:231]` | Terminal destination/clear/output-format mode |
| `P[235:234]` | Reserved, zero |

The terminal mode lets one closed-form compute loop change destination, clear,
and output format on the last point of a selected dimension without
introducing a projection-level hardware program.

`ACCUMULATOR_READ_3D` uses only the result, accumulator-address, and mode fields
of the same physical compute ICU packet:

| Payload bits | Field |
| ---: | --- |
| `P[143:0]` | Common loop |
| `P[151:144]` | Reserved, zero |
| `P[156:152]` | West result-stream base |
| `P[169:157]` | Accumulator base row |
| `P[183:170]`, `P[197:184]`, `P[211:198]` | Signed accumulator strides 0..2 |
| `P[225:212]` | Reserved, zero |
| `P[228:226]` | Read destination/clear/output-format mode |
| `P[235:229]` | Reserved, zero |

At each loop coordinate, this instruction emits one native accumulator read.
The three signed strides induce its accumulator address; the result stream and
read mode remain fixed. It does not consume weight-buffer, activation-stream,
row-stride, data-format, or terminal-mode fields.

Each complete MEM packet consumes three local 96-bit iMEM slots. Each complete
MXM load, dequant, or compute packet consumes two local 128-bit slots.

### VXM: RUN_2D, 3 x 96 bits

The 96-bit VXM compact config already contains `repeat_count`, the number of
contiguous datapath elements executed after one launch. `RUN_2D` carries only
two outer launch counters:

| Payload bits | Field |
| ---: | --- |
| `P[23:0]` | Reserved, zero |
| `P[39:24]`, `P[55:40]` | `count[0..1] - 1` |
| `P[79:56]`, `P[103:80]` | `cycle_stride[0..1]` |
| `P[135:104]` | Compact control low 32 bits |
| `P[167:136]` | Compact control high 32 bits |
| `P[199:168]` | Compact immediate 32 bits |
| `P[259:200]` | Reserved, zero |

For outer coordinate `(i0,i1)`, the instruction-local launch offset is
`wait_cycle + i0*stride0 + i1*stride1`. Each launch emits the compact config
once, and VXM executes its `repeat_count` contiguous elements internally. The
ICU counters do not duplicate that run, and neither compiler nor runtime
materializes per-cycle VXM instructions.

`VXM_STREAM_ND` is now only a legacy file descriptor. Inputs of rank one or two
without operand induction map directly to `RUN_2D`. Rank three is rejected
because VXM has no third outer launch counter.

### SXM: RUN_2D, 6 x 96 bits

The SXM tile-local payload uses the existing 416-bit encoding. It retains the
transpose/permute opcode, stream lists, row/tile selectors, the 16-lane tile
map, and the 32-lane permute map. `RUN_2D` prefixes the same 104-bit two-counter
launch domain used by VXM: a 24-bit `wait_cycle`, two counts, and two cycle strides.
For permute, `P[521:520]` encodes a per-launch map rotation in multiples of
eight lanes; this folds the four recurring Qwen permutation phases into one
loop. `P[523:522]` is reserved and must be zero.

Because an SXM packet has six words, its FU-local header uses a three-bit word
index in `[4:2]`. The physical queue already identifies SXM, so the header does
not duplicate a local operation; the tile opcode remains in the payload. Each
physical SXM queue has one 512-bit decoded context.

Legacy `SXM_TILE_PROGRAM` descriptors of rank one or two without operand
induction map directly to this `RUN_2D`. Rank three and instruction-field
induction are rejected.

## Direct compiler lowering

Every standard operator emitter constructs one or more piecewise-affine launch
domains directly from tensor shape, physical placement, and its closed-form
timeline. It does not first emit a point for every token, tile, or reduction
step and then recognize repeated points. Attention projections, QK/PV,
softmax/RoPE, RMSNorm, elementwise stages, and FFN projections/SwiGLU all use
this contract. "Closed form for every operator" means direct domain generation;
it does not mean that every FU is forced into a three-dimensional instruction.

MEM Schedule ops carry repeat, wave, and group count/cycle/address strides into
the three MEM counters. MXM issue, load, dequant, compute, and accumulator-read
ops likewise carry up to three launch counters plus the operand induction that
their decoder supports. VXM and SXM preserve their FU-local work in the compact
or tile-local payload and carry at most two outer launch counters. C2C remains a
transport-specific closed form: a contiguous transfer is represented by its
burst/vector count, and producer/consumer readiness is represented by a
completion event and the corresponding ready/release lifetime. It is not
wrapped in a generic arithmetic 3-D loop.

An operator emits another piece only at a physical queue, opcode or FU-template,
binding, page/bank, placement, C2C route, field-range, non-affine, or tail
boundary. Each piece is created once; the emitter does not scan already emitted
fine Schedule ops. `ScheduleToCommand` then maps these FU-specific Schedule
domains to MEM `READ_3D/WRITE_3D/WRITE_TAP_3D`, MXM
`LOAD_3D/DEQUANT_3D/COMPUTE_3D/ACCUMULATOR_READ_3D`, VXM `RUN_2D`, or SXM
`RUN_2D`. It contains no adjacent-command compression or interleaved-domain
recovery for the direct path.

For the verified Qwen2.5-1.5B seq32 decoder layer, the first ungrouped direct
lowering used 41,968 MEM domains, 44,290 total FU domains, and 2,366,141 encoded
bytes. Domain-major grouping of QKV layout blocks, residual phases, and FFN
SwiGLU regions reduces those figures to 17,928 MEM domains, 20,074 total FU
domains, and 1,125,493 bytes. Counter expansion remains exactly 6,645,946 FU
issues. Raw-packet field validation and the physical analyzer both pass, and
every physical queue has a peak of one live decoded context.

The largest MEM reductions are QKV `10,840 -> 8,808`, the two residual adds
`4,896 -> 408` and `3,456 -> 288`, and FFN `18,152 -> 3,800`. These groups are
formed from layout and timeline facts while the operator emitter creates its
domains; they are not recovered by inspecting or compressing fine commands.

The specialized FFN Up path calls `lowerFfnUpToFu3D` with tensor shape, physical
placement, topology route latencies, a closed-form timeline, and the dequant
scale. It returns commands already partitioned by physical MEM, MXM load, MXM
dequant, and MXM compute queue. `materializeFfnUp3DCommands` emits the
corresponding raw FU packet words.

This path has no per-cycle Schedule event list to scan and no fine instruction
stream to compress. Loop counts, cycle strides, address or accumulator
induction, buffer selection, and terminal compute behavior are derived directly
from the shape, placement, and closed-form timeline. Page boundaries, placement
discontinuities, or a change in the FU template produce another coarse
instruction for the affected physical queue.
The direct compile entry marks the module with
`ftlpu.command_lowering = "direct"`. Binary translation rejects any legacy
`CommandSequence` under that contract, so the path cannot silently fall back
to post-schedule compression.

For the Qwen2.5-1.5B Up projection with `M=32`, `K=1536`, and `N=8960`, the
direct-lowering test produces 100 MEM packets and two packets for each MXM
family: 106 coarse instructions encoded as 312 packet words. Queue gaps now
reside in the next packet's `wait_cycle`, so the image needs no extra NOP word
for those gaps. The ICU counters
expand representative queues to the expected 26,880 activation,
26,880 load, 26,880 dequant, and 215,040 compute issues without compiler-side
point materialization.

The full Qwen2.5 seq32 decoder uses the same mechanism across projection page
and output groups. QKV weight READ domains are `720 -> 256`, O projection is
`384 -> 16`, and Gate plus Up is `2,304 -> 192`; Down remains at 384 because
its same-queue streams and buffer reuse are real ordering boundaries. QKV uses
complete `wave_count=48, group_count=2` weight domains instead of the former
`2+46` split. The first overlapping Q-projection MXM domain starts at cycle
8,294 and covers all 48 reductions at a 32-cycle interval while RoPE runs.
Activation bank 0 slices 8/9 hold the primary staging copy and activation bank
0 slices 0/1 hold the pong copy; direct lowering routes only conflicting MEM
read intervals to pong.

An earlier overlapping image contained 19,304 MEM, 292 MXM load, 232 MXM dequant, 226 MXM
compute, 662 VXM, and 136 SXM work domains: 20,852 logical work packets in
total. Together with 20,070 queue NOPs this is 40,922 logical ICU instructions
and 82,284 physical iMEM words across 226 serialized queues. The binary is
1,219,763 bytes, counter expansion is 6,652,186 FU issues, and its
`max_cycle`/measured end are 769,416/769,480. The full decoder-layer CModel
passed with maximum error 0.0625. In the current serial Qwen2.5 seq32 image,
Q/V/K each use one projection domain per active queue. All gaps fit in
`wait_cycle`: 19,678 ICU packets, zero NOPs, 58,866 iMEM words, and a
1,055,197-byte binary. The FU issue count remains 6,646,042.

The VXM/SXM path follows the same rule. `materializeVxmRun2DCommand` and
`materializeSxmRun2DCommand` accept a two-dimensional launch domain already
derived from shape, placement, and a closed-form timeline, then emit
`command.vxm_run_2d` or `command.sxm_run_2d` directly. CommandBinary copies the
three or six physical words unchanged. The end-to-end test checks raw bits,
iMEM slots, FU-specific context occupancy, and every launch cycle.

## Legacy file compatibility

`Macro`, `MEM_STREAM_ND`, `MXM_STREAM_ND`, and `VXM_STREAM_ND` remain accepted serialized input
for existing compiler artifacts. They are transport descriptors, not another
hardware ISA. After relocation, the runtime loader converts each supported
descriptor directly into one typed FU 3-D instruction and then into the same
raw packets documented above. The loader uses a legacy absolute `start_cycle`
only to calculate `gap = start_cycle - queue_cursor`; it encodes that gap as
`wait_cycle` when it fits, then advances the cursor past that packet's last
issue. The adapter copies the two-dimensional Macro domain or the
one-to-three-dimensional STREAM_ND domain into the hardware counters in closed
form; it never enumerates the descriptor's issue points. Already-raw FU 3-D
packets retain their bits and are written in queue order.

The compatibility subset is intentionally narrow:

| Legacy input | Closed-form hardware mapping |
| --- | --- |
| MEM Macro/STREAM_ND read or write | `READ_3D`, `WRITE_3D`, or `WRITE_TAP_3D` with affine address induction |
| MXM load Macro/STREAM_ND | Supercell `IW` to `LOAD_3D` with fixed buffer selection |
| MXM dequant Macro/STREAM_ND | `DEQUANT_3D` with no operand induction |
| MXM compute Macro/STREAM_ND `Compute` | `COMPUTE_3D` with accumulator-address induction |
| MXM compute Macro/STREAM_ND `AccumulatorRead` | `ACCUMULATOR_READ_3D` with accumulator-address induction |
| VXM STREAM_ND rank 1/2 without induction | Preserve compact `repeat_count`; map the outer domain to `RUN_2D` |
| SXM TILE_PROGRAM rank 1/2 without induction | Preserve the tile-local payload; map the outer domain to `RUN_2D` |

Unsupported legacy semantics fail at load time. This includes MEM
gather/scatter, MXM IWColumn, MXM `Decode`, invalid induction
targets, values outside current packet fields, and overlapping live intervals.
`MEM_SLICE_PROGRAM` also fails on the raw MEM path because one parent program
cannot be represented safely as one current packet. Legacy VXM descriptors of
rank three or with operand induction are also rejected. SXM descriptors of rank
three or with instruction-field induction are rejected as well.

## One active context per physical ICU

The decoder latches one complete packet into the queue-local context and runs
it to completion. The fetch IQ may hold following packet words, but the decoder
does not activate a second coarse instruction while the first loop is live.
The context remains occupied from the first through the last launch of its
domain, including cycle gaps between launch points; those gaps cannot be filled
by launches from another coarse instruction on the same queue.
Every physical FU queue therefore provisions the same active depth:

| Physical queue | Decoded contexts |
| --- | ---: |
| MEM (per bank) | 1 |
| MXM load | 1 |
| MXM dequant | 1 |
| MXM compute | 1 |
| VXM | 1 |
| SXM | 1 |

Each SRAM bank has exactly one MEM ICU, one local iMEM, one FIFO, and one PC.
`READ_3D`, `WRITE_3D`, `WRITE_TAP_3D`, and C2C-fed `MEM_WRITE_SYNC` are ordered
in that same instruction stream. A synchronized write at the queue head may
wait for matching C2C tokens and SR data; while it waits, no later MEM command
can pass it. Distinct physical FU queues may overlap, while one queue cannot
interleave two coarse instructions. The compiler's raw CommandBinary path,
runtime capacity analysis, runtime page-program linker, and CModel all enforce
this rule. The compatibility path no longer cuts interleaved waves after
scheduling; any live-interval overlap on one physical ICU is an error and
requires that operator to emit a contiguous domain through direct lowering.

## Historical legacy-envelope benchmarks

**The measurements in this section are retained as historical data for the
variable-length legacy envelope and its software expansion/validation path.
Their binary-byte and descriptor counts are not the current raw local-iMEM
footprint, and their multi-form experiments do not define the current hardware
context depths. Cycle and golden-output results remain useful as compatibility
evidence.**

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

Adding `MEM_SLICE_PROGRAM` groups the remaining 4,652 MEM N-D operations by
physical slice and launch domain:

| Metric | Previous coarse forms | With MEM slice programs | Reduction |
| --- | ---: | ---: | ---: |
| MEM queue commands | 4,652 | 1,628 programs | 65.0% |
| MEM program body entries | - | 4,652 | no semantic change |
| All queue commands | 5,045 | 2,021 | 59.9% |
| Binary bytes | 347,235 | 263,881 | 24.0% |
| Expanded functional issues | 6,586,470 | 6,586,470 | unchanged |
| Scheduled `max_cycle` | 763,669 | 763,669 | unchanged |

The updated binary completed in 763,733 modeled cycles and passed the same
49,152-output Qwen golden comparison with zero mismatches, MAE `0.004514`, and
maximum error `0.09375`.

### Legacy MEM slice experiment and inspector

The `MEM_SLICE_PROGRAM` A/B experiment grouped multiple MEM bodies under one
legacy file descriptor. The format remains behind
`--mem-slice-program on|off` and is disabled by default. It has no current raw
MEM packet and is rejected when a binary is loaded into FU raw-word queues.

`ftlpu_binary_inspect left.ftlpu --compare right.ftlpu` can still expand legacy
artifacts to exact sparse `(queue, cycle, native instruction)` timelines for
semantic comparison. Missing issue points are logical NOPs through
`max_cycle`; a changed issue cycle or native instruction makes the comparison
fail. This is an offline inspector operation, not the compiler lowering route
or the hardware execution mechanism.

## Local ICU execution contract

The raw queue fetches one physical word per local-iMEM entry and recognizes an
FU-loop header at the queue head. Activation is atomic at packet granularity:
all three MEM-3D/VXM words, both MEM-write-sync/MXM words, or all six SXM words
must be available and have consistent headers. The decoder latches the queue's sole active
context, advances two or three counters for the selected FU, and emits at most
one native FU instruction per cycle. It activates the next packet only after
the current loop retires.

Malformed headers, truncated packets, a missed start cycle, invalid loop or
operand fields, and overlapping coarse live intervals are errors. One active
context is the capacity and scheduling rule for every physical FU queue.

## Encoding status

The frozen hardware-facing formats covered here are MEM `READ_3D`,
`WRITE_3D`, and `WRITE_TAP_3D` at 3 x 96 bits, plus MXM `LOAD_3D`,
`DEQUANT_3D`, `COMPUTE_3D`, and `ACCUMULATOR_READ_3D` at 2 x 128 bits, plus VXM
`RUN_2D` at 3 x 96 bits, plus SXM `RUN_2D` at 6 x 96 bits. Legacy
Macro/STREAM_ND records remain only as compatible file input to the closed-form
adapter. `MEM_SLICE_PROGRAM` still requires a hardware packet definition before
it can be treated as a raw ICU instruction.
