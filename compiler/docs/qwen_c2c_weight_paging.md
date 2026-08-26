# Qwen C2C Weight Double Buffering

![Qwen2.5-1.5B FFN C2C weight-page pipeline](qwen_ffn_c2c_pipeline.svg)

The diagram aligns Command IR weight-page `ready/release/bank` intervals with
the target C2C bandwidth. Solid bars are executable MEM residency and MXM/VXM
compute windows. Hatched bars are C2C prefetch windows derived from eight
external lanes per direction at 32 bytes/cycle per lane, then capped by the
configured DDR source bandwidth. Page 0 is cold-loaded before ICU cycle 0;
later pages are written to the alternate bank while the current bank is being
consumed.

## Goal

Qwen decoder weights no longer need to be resident in on-chip MEM all at once. Runtime first loads the target-packed layer-0 page through C2C into bank 0. While layer `i` reads its current bank, C2C writes layer `i+1` into the other bank. Runtime swaps banks at the layer boundary after the next page reaches SRAM-ready state.

This is also the model-wide external-I/O rule: host data may initialize or read
the external DDR backing store, but it cannot directly mutate LPU MEM. Inputs,
outputs, resident constants, state initialization, and every weight page cross
the C2C path. Device-resident aliases between decoder layers remain internal.

## Physical Contract

- Each MEM slice has two independent single-port SRAM banks.
- A MEM ICU queue identifies `(hemisphere, slice, bank)`.
- SRAM addresses are bank-local rows; each row is 32 bytes. The 128 KiB-per-
  superlane profile has two 64 KiB banks, so each bank exposes rows `0..2047`.
- A bank-0 read may issue in the same cycle as a bank-1 write. Operations in one bank still obey its single-port constraint.
- Compute keeps the original 32 eastward and 32 westward streams. C2C exposes
  a target-configurable external pool, defaulting to 8 lanes per direction. A
  lane transfers one complete 32-byte vector per cycle, for a default external
  bandwidth of `8 x 32 = 256 bytes/cycle` per direction.
- The current `ModelSession` uses dedicated C2C lanes in addition to the 32
  ordinary compute streams. An RX command carries the destination hemisphere,
  slice, bank, and row, and the target MEM receiver commits SRAM. Every byte
  must still arrive through DDR DMA and C2C; the host cannot write LPU MEM
  directly. An optional shared-SR mode maps lanes to `W24..W31` and consumes
  them with ordinary MEM `Write`, but it is not used for current fine-grained
  overlapped paging.
- Page readiness means that the final target-SRAM write committed,
  not merely that DMA placed the final vector in an RX FIFO.

## Software Representation

Binary v24 stores the bank and external-memory target parameters in the target ABI. ModelPackage v5 uses `ModelWeightPage` to record the layer, destination bank, target-packed tensors, and physical segments with DDR offset, hemisphere, slice, row, vector count, and stream.

Only `TargetPackedSramVectors` tensors may be used as C2C page images. `ftlpu-pack-model-weights` offline-reorders quantized row-major weights into target SRAM rows from executable bindings. Runtime only transfers page images.

Weight paging is selected automatically when a physical weight layout exceeds
the target's rows per bank. `ftlpu-opt --weight-bank 0|1` only overrides the
initial bank. Bank and page metadata participate in physical planning starting
in Tensor IR instead of being attached to Command IR afterward.

### Dedicated Slice Roles

The 128 KiB vector deployment target
`cmodel_128kb_superlane_vector.json` separates each 52-slice hemisphere into
activation/workspace slices `0..19` and MXM-local weight slices `20..51`.
Slices `0..15` provide the distributed-16 activation plane required by SXM;
`16..19` are auxiliary activation workspace. MXM accumulators are local to the
functional unit and are sized by `mxm_accumulator_blocks`; no MEM slice is
configured as an accumulator. Both SRAM banks keep the same slice role.
Feedback RMSNorm stores layer gamma in two activation-side slices and reads it
through two low-numbered west streams per hemisphere. Its data path remains
below `W16`, while the pager uses independent dedicated C2C lanes and writes
high-slice weight SRAM ports. Separating both stream and slice/port resources
permits overlap; neither path may bypass the external C2C boundary.

This partition removes transport and port conflicts; it does not increase
capacity. For Qwen2.5-1.5B FFN at sequence length 32, the generic Vector planner
uses four 8-slice weight groups. Gate and Up share seven ping-pong pages, while
Down uses twelve pages. Every page stays within 2048 rows per bank. Block8's
packed page remains a separate valid layout.

## Execution

1. Prefetch page 0 into bank 0 and wait for SRAM commit.
2. Load the layer-0 ICU program.
3. Start the complete layer-1 page C2C DMA into bank 1. Attention and FFN
   remain one executable and one page; there is no phase boundary between them.
4. Advance chip, DMA, and DDR from the same runtime cycle driver.
5. At layer completion, wait only for any unfinished part of page 1.
6. Load layer 1, start its complete Attention -> FFN execution, start page 2
   into bank 0, and continue alternating.

`ModelSessionStats` separates bootstrap and steady-state behavior.
`weight_page_initial_wait_cycles` is the page-0 cold-start cost,
`weight_page_boundary_wait_cycles` is an actual layer-boundary stall, and
`weight_page_hidden_prefetches` counts pages already SRAM-ready when their
layer starts. The legacy `weight_page_wait_cycles` remains their total.

A C2C receive command describes one contiguous SRAM-row burst. Runtime maps
segments by target slice across the configured C2C lanes, so instruction count
scales with segments rather than 32-byte vectors.

## Validation

- `c2c_weight_ping_pong_test` proves same-cycle bank-0 reads and bank-1 C2C writes with byte-exact data.
- `weight_page_planner_test` proves paged weights are excluded from resident allocation and alternate banks.
- `model_session_c2c_weight_pages_test` runs three complete layer invocations,
  proves `bank0 -> bank1 -> bank0` overwrite, and requires both steady-state
  prefetches to finish with zero layer-boundary wait cycles.
- `weight_page_builder_test` compares a bank-1 host upload and offline page image byte for byte and proves bank 0 is untouched.
- `qwen_weight_page_builder_test` packs the real Qwen2.5-1.5B Q/K/V/O, two norm, and gate/up/down shapes into one 47.625 MiB page with 176 contiguous C2C segments; every row stays within one 32768-row bank.
- `qwen2_5_1_5b_paged_weight_layout_test` lowers standard StableHLO through Tensor and Stream IR and checks bank, slice, row ranges, including the head-dimension-128 attention weight formula.
- `build_hf_decoder_stack_paging_test` checks that the generic stack builder
  compiles bank-0/bank-1 variants, assigns consecutive layers to alternating
  variants, and invokes offline C2C page packing.
- `model_session_c2c_io_test` round-trips a 32x1536 BF16 tensor through DDR,
  C2C, and MEM and rejects a host/MEM bypass.
- The real two-layer Qwen seq-len-32 package passes with 6/49,152 tolerance
  outliers, P99 0.25, and MAE 0.06079. It performs one external upload, one
  external download, one device alias, and zero device copies.

Convert a logical package with:

```powershell
build-ftlpu-vs2026/runtime/ftlpu-pack-model-weights.exe `
  --input qwen.logical.ftlpum `
  --output qwen.paged.ftlpum `
  --first-bank 0
```

Build a two-layer Qwen package directly from a local Hugging Face checkpoint:

```powershell
python compiler/tools/build_hf_decoder_stack.py `
  --model-dir .cache/hf/Qwen2.5-1.5B `
  --opt build-ftlpu-vs2026/compiler/ftlpu_opt.exe `
  --translate build-ftlpu-vs2026/compiler/ftlpu-translate.exe `
  --stablehlo compiler/examples/qwen2_5_1_5b_decoder_layer/decoder_layer_seq128.stablehlo.mlir `
  --target-config compiler/examples/targets/cmodel_large_sram.json `
  --pack-model-weights build-ftlpu-vs2026/runtime/ftlpu-pack-model-weights.exe `
  --c2c-weight-paging --layer-count 2 --seq-len 128 `
  --output-dir build-ftlpu-vs2026/qwen_two_layer `
  --output build-ftlpu-vs2026/qwen_two_layer/qwen_two_layer.paged.ftlpum
```

The command compiles bank-0 and bank-1 variants of the same generic decoder
lowering, imports consecutive HF layers, packs alternating C2C pages, and
keeps `hidden.1` device-resident between the two invocations. Run the numerical
package with `hf_two_decoder_layers_model_session_test.exe`.

The two-layer numerical path is now wired and checked whenever a local
Qwen2.5-1.5B checkpoint is supplied; the checkpoint itself is intentionally not
stored in this repository. A remaining issue is compilation performance: fully
expanding and printing the sequence-length-128 layer Command IR takes over ten
minutes. The deployment path should next let the binary emitter consume
compressed Schedule/repeat representation directly, without materializing
giant textual Command MLIR.

Prefill has a long enough compute window to hide much of the next-layer transfer. `M=1` decode has a short compute window and will generally be DDR/C2C-bandwidth bound when tens of MiB are loaded per layer. Double buffering enables overlap but cannot remove that bandwidth lower bound; decode will need higher link bandwidth, projection/wave-granular prefetch, or more resident capacity.

The default eight-lane C2C peak is 256 bytes/cycle per direction, but it is not
the sustained source rate. The default dual-channel DDR4-3200 model peaks at
51.2 GB/s = 102.4 bytes per 500 MHz LPU cycle, and the planner reserves only
90% of it: 92.16 bytes/cycle. Request-level read latency varies from 35 to 50
cycles. The current real two-layer seq-len-32 run therefore records a 4,750
cycle layer-boundary page wait; the overlap is real but not yet complete.
