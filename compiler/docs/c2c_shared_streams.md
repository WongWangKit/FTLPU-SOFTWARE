# Shared C2C Receive Path

C2C ingress always uses the ordinary SR fabric. Runtime maps the configured
external lanes onto high-numbered west streams (`W24..W31` for eight lanes),
then the target MEM ICU consumes those streams with normal `Write` commands:

```text
host -> DDR4 -> C2C DMA -> C2C RX -> ordinary west SR -> MEM Write -> SRAM
```

For executable weight pages, the compiler reserves the MEM window and emits
one two-word `MEM_WRITE_SYNC` per contiguous segment into the ordinary MEM
iMEM. Its page order, sync tag, vector count, SRAM base/stride, SR ID,
transport latency, and queue duration are fixed in the `.ftlpu` image.
Runtime provides page bytes and DDR addresses, builds the C2C RX/DMA queues,
and verifies that the resulting transport schedule matches those compiler
MEM words; it does not rewrite the executable's MEM iMEM. Each RX vector sends
a point-to-point token to its target `(hemisphere, slice, bank)` MEM ICU.
The CModel preserves notification cycles, so DDR bubbles stall the
corresponding writes.

`MEM_WRITE_SYNC` shares the single MEM ICU and physical MEM port with ordinary
reads and writes. While it waits for its tagged notifications, later MEM
instructions cannot pass it. The payload remains subject to normal SR routing
and collision rules. A page fence becomes ready only after RX completion and
all target MEM writes have been observed.

When the preceding queue interval is idle, the compiler activates
`MEM_WRITE_SYNC` at its start and includes that idle time in the instruction's
24-bit reservation. This removes the leading NOP without moving the following
ordinary MEM instruction. An oversized interval still uses a leading NOP.
Ordinary `SYNC` is a separate control instruction and does not reserve this
MEM write window.
