# Shared C2C Receive Path

C2C ingress always uses the ordinary SR fabric. Runtime maps the configured
external lanes onto high-numbered west streams (`W24..W31` for eight lanes),
then the target MEM ICU consumes those streams with normal `Write` commands:

```text
host -> DDR4 -> C2C DMA -> C2C RX -> ordinary west SR -> MEM Write -> SRAM
```

Each RX vector sends a point-to-point token to its target
`(hemisphere, slice, bank)` MEM ICU C2C command context. The runtime emits one
coarse synchronized-write descriptor per contiguous segment. It contains the
vector count, SRAM base/stride, SR ID, and target-specific transport latency.
The CModel preserves notification cycles, so consecutive vectors stay
pipelined and DDR bubbles stall only their matching writes.

This auxiliary command context is not a dedicated data lane. Its write shares
the physical MEM port with compute, and the payload remains subject to normal
SR routing and collision rules. A page fence becomes ready only after both RX
completion and all target MEM writes have been observed.
