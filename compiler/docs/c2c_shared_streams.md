# Shared C2C-to-SR Ingress

C2C keeps up to eight external transport lanes per direction, each carrying one
32-byte vector per cycle. These lane IDs are not ordinary LPU stream IDs. On
receive, runtime maps the external lanes to an explicitly selected subset of
the ordinary stream-register fabric. The default weight pager maps lane 0..7
to west streams `W24..W31`.

The default receive path is:

```text
DDR4 -> per-lane DMA RX FIFO -> C2C RX -> W24..W31 at sreg13
      -> westward SR propagation -> target MEM Write -> SRAM bank/row
```

`C2cInstruction` therefore carries two independent selectors:

- `stream_index`: external C2C/DMA lane 0..7;
- `fabric_stream_index`: ordinary SR selected for on-chip transport.

The receive descriptor still carries the destination hemisphere, slice, bank,
base row, vector count, and row stride, but C2C no longer bypasses MEM. The
first received vector releases the target MEM `Sync`. Runtime emits the exact
number of route NOPs, one `Write`, and an ICU `Repeat` for the remaining rows.
A page becomes ready only after DMA, RX, the target MEM queue, and the four-tile
SR drain have completed.

This path exposes real conflicts to the scheduler. C2C can overlap compute only
when the selected SRs and target SRAM write ports are free. For feedback
RMSNorm, gamma is layer-specific but immutable during a layer. It is replicated
in two activation-side slices and read on two low-numbered west streams per
hemisphere. The RMSNorm data path stays below `W16`, leaving `W24..W31` for C2C
weight ingress. The separation is a planner decision, not a hidden bypass.

`c2c_dma_ddr4_test` verifies eight simultaneous lanes through ordinary SRs and
MEM writes. `c2c_weight_ping_pong_test` verifies byte-exact next-bank paging
while the current bank is read.
