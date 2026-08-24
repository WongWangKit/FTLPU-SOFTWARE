# Dedicated C2C Streams

The compute stream fabric remains 32 eastward plus 32 westward streams. C2C
uses a separate directional fabric, so C2C traffic never allocates or blocks a
compute `StreamId`.

The target fields are:

- `streams.c2c_streams_per_direction`: active C2C lanes in each direction;
  default 8.
- `streams.c2c_bytes_per_stream_per_cycle`: payload per lane per cycle;
  default 32 bytes.

The default peak is therefore `8 x 32 = 256 bytes/cycle` in each direction.
Both fields are embedded in binary format v20 and participate in the target ABI
fingerprint. Runtime rejects an executable that requests more C2C lanes than
the attached CModel provides.

For weight paging, runtime maps each physical slice to a C2C lane modulo the
active lane count; segments targeting one slice are therefore serialized on
one lane. One receive descriptor carries a contiguous destination row burst. The
DMA ingress path writes `(hemisphere, slice, bank, row)` directly and does not
replay data through `E0..E31/W0..W31`. This permits full-width VXM feedback,
including RMSNorm, to overlap a next-layer write into the other SRAM bank.
