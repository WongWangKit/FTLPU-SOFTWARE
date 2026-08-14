# Qwen C2C Weight Double Buffering

## Goal

Qwen decoder weights no longer need to be resident in on-chip MEM all at once. Runtime first loads the target-packed layer-0 page through C2C into bank 0. While layer `i` reads its current bank, C2C writes layer `i+1` into the other bank. Runtime swaps banks at the layer boundary after the next page reaches SRAM-ready state.

## Physical Contract

- Each MEM slice has two independent single-port SRAM banks.
- A MEM ICU queue identifies `(hemisphere, slice, bank)`.
- SRAM addresses are bank-local rows in `0..32767`; each row is 32 bytes.
- A bank-0 read may issue in the same cycle as a bank-1 write. Operations in one bank still obey its single-port constraint.
- Page readiness means that the final MEM write committed, not merely that DMA placed the final vector in the RX FIFO.

## Software Representation

Binary v17 stores the bank in `BinaryBinding` and `BinaryMemoryFloor`. ModelPackage v5 uses `ModelWeightPage` to record the layer, destination bank, target-packed tensors, and physical segments with DDR offset, hemisphere, slice, row, vector count, and stream.

Only `TargetPackedSramVectors` tensors may be used as C2C page images. `ftlpu-pack-model-weights` offline-reorders quantized row-major weights into target SRAM rows from executable bindings. Runtime only transfers page images.

Paged executables are compiled with `ftlpu-opt --weight-bank 0|1`. The bank participates in physical planning starting in Tensor IR instead of being attached to Command IR afterward. Qwen2.5-1.5B attention weights use an independent slice plane at slices 32..39, FFN gate/up/down use disjoint slice planes, and RMSNorm gamma is placed downward from the top of the bank.

## Execution

1. Prefetch page 0 into bank 0 and wait for SRAM commit.
2. Load the layer-0 ICU program.
3. Start page 1 C2C DMA into bank 1.
4. Advance chip, DMA, and DDR from the same runtime cycle driver.
5. At layer completion, wait only for any unfinished part of page 1.
6. Load layer 1, start page 2 into bank 0, and continue alternating.

Contiguous row writes use one MEM Write plus ICU Repeat, so instruction count scales with segments rather than 32-byte vectors. The repeat interval is `max(C2C RX tile replay cycles, DDR vector service cycles)`.

## Validation

- `c2c_weight_ping_pong_test` proves same-cycle bank-0 reads and bank-1 C2C writes with byte-exact data.
- `weight_page_planner_test` proves paged weights are excluded from resident allocation and alternate banks.
- `model_session_c2c_weight_pages_test` proves page-0 bootstrap, page-1 overlap during layer 0, and the layer-boundary bank swap.
- `weight_page_builder_test` compares a bank-1 host upload and offline page image byte for byte and proves bank 0 is untouched.
- `qwen_weight_page_builder_test` packs the real Qwen2.5-1.5B Q/K/V/O, two norm, and gate/up/down shapes into one 45 MiB page with 192 contiguous C2C segments; every row stays within one 32768-row bank.
- `qwen2_5_1_5b_paged_weight_layout_test` lowers standard StableHLO through Tensor and Stream IR and checks bank, slice, row ranges, including the head-dimension-128 attention weight formula.

Convert a logical package with:

```powershell
build-ftlpu-vs2026/runtime/ftlpu-pack-model-weights.exe `
  --input qwen.logical.ftlpum `
  --output qwen.paged.ftlpum `
  --first-bank 0
```

A two-layer numerical CModel golden driven by a local Qwen2.5-1.5B checkpoint is still pending. A second confirmed issue is compilation performance: fully expanding and printing the sequence-length-128 layer Command IR takes over ten minutes. The deployment path should next let the binary emitter consume compressed Schedule/repeat representation directly, without materializing giant textual Command MLIR.

Prefill has a long enough compute window to hide much of the next-layer transfer. `M=1` decode has a short compute window and will generally be DDR/C2C-bandwidth bound when tens of MiB are loaded per layer. Double buffering enables overlap but cannot remove that bandwidth lower bound; decode will need higher link bandwidth, projection/wave-granular prefetch, or more resident capacity.
