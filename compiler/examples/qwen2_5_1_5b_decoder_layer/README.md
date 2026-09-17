# Qwen2.5-1.5B decoder layer

This directory contains standard-StableHLO decoder-layer fixtures for sequence
lengths 32 and 128. Both use Qwen2.5-1.5B dimensions: hidden size 1536,
intermediate size 8960, 12 query heads, 2 KV heads, head dimension 128, RoPE
theta 1,000,000, and RMSNorm epsilon 1e-6.

## Current seq32 status

The full-layer pipeline now lowers standard StableHLO through Kernel, Tensor,
and Stream IR into FU-specific closed-form Schedule domains and raw hardware
Command packets. The standard path sets `ftlpu.command_lowering = "direct"`,
`ftlpu.icu_compression = "none"`, and `ftlpu.mem_slice_program = false`.
It never materializes a per-cycle instruction stream and then compresses it.

The recorded results below use `--projection-rope-overlap on`, which pipelines
projection and RoPE across output groups. The default `off` mode changes the
instruction counts, image size, and cycle count. The verified `seq_len=32`,
weight-bank-1, KV-capacity-256 build has these
results:

| Property | Result |
| --- | --- |
| Binary size | 1,161,429 bytes (1.108 MiB) |
| Serialized physical queues | 226 |
| Raw FU domains | 19,876 |
| Logical ICU instructions | 38,880 = 19,876 work packets + 19,004 duration NOPs |
| MEM READ/WRITE_3D / WRITE_READ_2D / MXM load / dequant / compute domains | 17,944 / 384 / 292 / 232 / 226 |
| VXM / SXM domains | 662 / 136 |
| Physical iMEM words | 78,290 |
| Counter-expanded work | 6,652,186 issues |
| Binary `max_cycle` / measured end | 768,137 / 768,201 |
| Raw FU-ICU physical analyzer | `deployable=yes`, every queue peak context 1 |
| Full decoder-layer CModel | Pass, 49,152 BF16 outputs, 49,116 nonzero, max error 0.0625 |
| Dynamic C2C weight pages | Pass, 20 uses, 21 prefetches, 0 runtime page-ready wait cycles |

The first hardware-safe unified-MEM baseline used 1,554,521 bytes for 26,010
raw domains, including 23,776 MEM domains. Selective grouping first reduced
that to 1,458,329 bytes, 24,258 raw domains, and 22,104 MEM domains. Maximal
projection domains first reduced it to 1,246,745 bytes, 21,202 raw domains, and
19,576 MEM domains. With `--projection-rope-overlap on`, activation-region
ping-pong staging lets Q projection continue across RoPE without the old `2+46`
boundary. That mode's earlier image was
1,219,763 bytes with 20,852 raw domains and 19,304 MEM domains. QKV weight reads
change from the original 720 to 256, O projection from 384 to 16, and Gate plus
Up from 2,304 to 192; Down remains at 384 because its same-queue streams and
buffer reuse create real boundaries. Counter expansion is 6,652,186 FU issues.

The current Q projection uses a three-word `WRITE_READ_2D` MEM ICU packet to
pair each raw MXM result write with the subsequent cross-hemisphere copy read.
There are 384 such packets (12 output groups × 2 hemispheres × 16 slices),
generated directly from closed-form domains without per-cycle FU instructions.
The copy stream is fixed by physical source slice; the final Q group switches
stream range to avoid the following V projection.

These domains are constructed directly by the operator emitters. For example,
QKV weight queues use complete `wave_count=48, group_count=2` domains rather
than splitting a reduction into `2+46`. With `on`, the first overlapping Q
projection MXM
domain starts at cycle 8,294 and covers all 48 reductions at a 32-cycle
interval while the previous projection's RoPE runs. Its activation reads use
two copies in activation SRAM: bank 0 slices 8/9 are the primary copy and bank
0 slices 0/1 are the pong copy. Direct lowering routes only the read intervals
that would collide with a same-queue RoPE write to the pong copy, so the MXM
domain remains continuous and each MEM ICU still has one live context.

For the other projections,
each O-projection weight queue uses one `counts=(4,48,24)` READ with cycle
strides `(1,32,1578)`. Gate and Up use the MEM ICU blocked-outer address mode
to express alternating weight buffers without splitting each pair. A residual
block is split during direct lowering only when an operand read and result
write use the same physical MEM queue; its VXM pipeline delays the result until
the read domain has retired.

Each physical `(hemisphere, slice, bank)` MEM ICU has one iMEM, one instruction
FIFO, and one PC. `READ_3D`, `WRITE_3D`, `WRITE_TAP_3D`, and linked external
`MEM_WRITE_SYNC` instructions all execute from that same queue. Separate SRAM
read and write ports therefore do not permit two coarse instructions to be
interleaved. RoPE pair domains and aliased residual domains are ordered directly
by their schedule emitters so every queue needs at most one live FU-domain
context. The verified CModel checkpoint errors are 0.03125 for FFN, 0.0195312
for the attention context, 0 for the residual, and 0.015625 for RMSNorm 2.

### Dynamic C2C single-layer execution

The same seq32 runtime test also executes the layer through
`ModelSession(C2cDmaSystem)`. The executable declares 20 weight-page uses: one
page each for Q, K, V, and O, two Gate pages, two Up pages, and twelve Down
pages. The model package adds one parameter page for the two RMSNorm scales and
the Q/K/V biases. The parameter page is separate from the 20 executable page
uses, so the run performs 21 physical prefetches in total.

Before loading the linked executable, `ModelSession` synchronously transfers
the parameter page and the six executable pages that the planner marks
`pre_execution` (Q/K/V/O and both Up pages). It then links the remaining 14
executable pages (both Gate pages and all twelve Down pages) into static idle
windows. The linker adds C2C DMA/RX packets and `MEM_WRITE_SYNC` packets to the
same physical MEM ICU queues used by ordinary reads and writes. Page fences
reserve each target queue until its first consumer boundary, while page-ready
synchronization absorbs any later transport completion without shifting the
logical static schedule.

The startup transfer programs are transient: they execute before
`runtime.load()` and are not copied into `ModelSession::last_linked_program()`.
No page payload bytes are serialized in a `BinaryProgram`; all payloads remain
in the `ModelPackage`/external DDR backing store. The linked `.ftlpu` therefore
contains the static executable plus the transfer instructions for the 14
overlapped pages, while retaining the original 20 page-use descriptors as
metadata. This distinction matters when counting instructions in an exported
linked image.

The verified dynamic run reports:

| Dynamic C2C property | Result |
| --- | --- |
| Page plan | 20 executable pages + 1 parameter page = 21 prefetches |
| Physical bytes transferred | 47,194,112 = 46,792,704 executable + 401,408 parameter |
| Startup pages / initial wait | 7 pages / 217,183 cycles |
| Runtime-linked pages | 14 executable pages |
| Post-link instruction image | 1,321,249 bytes, 230 serialized queues, 85,242 physical iMEM slots |
| Linked MEM synchronization | 256 `MEM_WRITE_SYNC` instructions |
| Synchronized MEM FU work | 860,160 vector writes = 27,525,120 bytes |
| Executable page-ready wait | 0 cycles |
| Numerical result | Byte-identical to the direct run; 49,152 BF16 values, 49,116 nonzero, maximum error 0.0625 |

## Build and inspect

Both `ftlpu-opt` and `ftlpu-compile` accept `--projection-rope-overlap on|off`;
omitting it defaults to `off`. In `off` mode, the compiler completes every Q
projection and writes all Q results to distinct RoPE staging addresses. Only
after the final write does it read staging and run all Q RoPE work. K projection
and K RoPE follow the same order. The `on` mode preserves the output-group
projection/RoPE pipeline. This switch controls overlap between those stages;
MXM's internal pipeline and dynamic C2C page transfers are separate. In a
multi-step `ftlpu-opt` flow, set the flag on the StableHLO-to-Stream step; later
tools inherit the `ftlpu.projection_rope_overlap` IR attribute unless an
explicit command-line setting overrides it.

For the seq32 serial configuration, Q, K, and V projection reads are lowered
directly to one `READ_3D` per active physical MEM queue for activations and one
for weights. The outer counter covers all output halves (24 for Q, 4 for K/V).
Serial V output packing also waits until its complete projection has finished,
so it does not interrupt either read domain. C2C weight-page synchronization
and result writes remain separate MEM ICU instructions.

For example, generate serial and overlapping seq32 Schedules:

```powershell
build-ftlpu-vs2026-direct/compiler/ftlpu_opt.exe `
  --input compiler/examples/qwen2_5_1_5b_decoder_layer/decoder_layer_seq32.stablehlo.mlir `
  --output build-ftlpu-vs2026-direct/qwen2_5_seq32_serial.schedule.mlir `
  --pipeline ftlpu-stablehlo-to-schedule `
  --target-config ../FTLPU-CMODEL/config/ftlpu-lpu32.json `
  --weight-bank 0 --mxm-execution vector `
  --projection-rope-overlap off

build-ftlpu-vs2026-direct/compiler/ftlpu_opt.exe `
  --input compiler/examples/qwen2_5_1_5b_decoder_layer/decoder_layer_seq32.stablehlo.mlir `
  --output build-ftlpu-vs2026-direct/qwen2_5_seq32_overlap.schedule.mlir `
  --pipeline ftlpu-stablehlo-to-schedule `
  --target-config ../FTLPU-CMODEL/config/ftlpu-lpu32.json `
  --weight-bank 0 --mxm-execution vector `
  --projection-rope-overlap on
```

The full-layer pipeline test below explicitly uses `on` to preserve the
verified overlapping baseline above. To check actual Q projection/RoPE issue
order in both modes, run
`ctest --test-dir build-ftlpu-vs2026-direct -C Release -R qwen2_5_projection_rope_overlap_test --output-on-failure`.

Build the sequence-length-32 raw FU executable with:

```powershell
python compiler/tests/qwen2_5_1_5b_decoder_layer_pipeline_test.py `
  --opt build-ftlpu-vs2026-direct/compiler/ftlpu_opt.exe `
  --compile build-ftlpu-vs2026-direct/compiler/ftlpu-compile.exe `
  --input compiler/examples/qwen2_5_1_5b_decoder_layer/decoder_layer_seq32.stablehlo.mlir `
  --target-config ../FTLPU-CMODEL/config/ftlpu-lpu32.json `
  --weight-bank 1 `
  --kv-cache-capacity 256 `
  --output-dir build-ftlpu-vs2026-direct/compiler/ftlpu_lower/qwen2_5_1_5b_decoder_layer
```

Run the focused direct-domain checks with:

```powershell
python compiler/tests/qwen2_5_1_5b_decode_reference_test.py
ctest --test-dir build-ftlpu-vs2026-direct -C Release `
  -R "ffn_up_3d_lowering_test|standalone_ffn_up_3d_compile_test|ffn_swish_emitter_test|schedule_trace_weight_page_test" `
  --output-on-failure
```

Run the full CModel check with:

```powershell
build-ftlpu-vs2026-direct/runtime/compiled_qwen2_5_1_5b_decoder_layer_seq32_runtime_test.exe `
  build-ftlpu-vs2026-direct/compiler/ftlpu_lower/qwen2_5_1_5b_decoder_layer/decoder_layer.ftlpu
```

Split the complete static prefill program into one file per physical ICU:

```powershell
build-ftlpu-vs2026-direct/runtime/ftlpu_icu_program_export.exe `
  build-ftlpu-vs2026-direct/compiler/ftlpu_lower/qwen2_5_1_5b_decoder_layer/decoder_layer.ftlpu `
  build-ftlpu-vs2026-direct/compiler/ftlpu_lower/qwen2_5_1_5b_decoder_layer/icu_programs
```

`icu_programs/index.csv` records each resource, queue number, physical
location, coarse-instruction count, i-MEM word count, and counter-expanded
work. Each `*.icu.csv` is one physical ICU program. A row is one coarse ICU
instruction and retains its physical `pc_word`, loop domain, decoded fields,
and complete 96/128-bit encoding. An idle physical ICU gets an empty file.
MEM files are one per physical `(hemisphere, slice, bank)` ICU, with
`READ_3D`, `WRITE_3D`, `WRITE_TAP_3D`, and `MEM_WRITE_SYNC` together in PC
order. `_read.icu.csv` or `_write.icu.csv` names identify a stale pre-unification
export; rerunning the exporter removes stale `*.icu.csv` files first.
C2C files are empty for this base image because it contains the static compute
program and page-use metadata, before `ModelSession` has linked dynamic C2C
instructions.

Set `FTLPU_QWEN_C2C_LINKED_BINARY` to export the exact post-link program that
the runtime loads, then split that image by physical ICU:

```powershell
$env:FTLPU_QWEN_C2C_LINKED_BINARY = `
  "build-ftlpu-vs2026-direct/compiler/ftlpu_lower/qwen2_5_1_5b_decoder_layer/decoder_layer.linked.ftlpu"
build-ftlpu-vs2026-direct/runtime/compiled_qwen2_5_1_5b_decoder_layer_seq32_runtime_test.exe `
  build-ftlpu-vs2026-direct/compiler/ftlpu_lower/qwen2_5_1_5b_decoder_layer/decoder_layer.ftlpu
Remove-Item Env:FTLPU_QWEN_C2C_LINKED_BINARY

build-ftlpu-vs2026-direct/runtime/ftlpu_icu_program_export.exe `
  build-ftlpu-vs2026-direct/compiler/ftlpu_lower/qwen2_5_1_5b_decoder_layer/decoder_layer.linked.ftlpu `
  build-ftlpu-vs2026-direct/compiler/ftlpu_lower/qwen2_5_1_5b_decoder_layer/linked_icu_programs
```

The linked export has populated C2C DMA/RX queues and the 256 synchronized MEM
packets for the 14 overlapped pages. It intentionally excludes the seven
startup transfer programs because those pages have already reached SRAM before
the linked executable is loaded. The test writes this diagnostic image after a
successful run and also attempts to preserve it when execution fails after
linking.

Generate the dedicated cycle-accurate MEM CSV:

```powershell
$env:FTLPU_QWEN_MEM_CSV = `
  "build-ftlpu-vs2026-direct/compiler/ftlpu_lower/qwen2_5_1_5b_decoder_layer/decoder_layer.mem.csv"
build-ftlpu-vs2026-direct/runtime/compiled_qwen2_5_1_5b_decoder_layer_seq32_runtime_test.exe `
  build-ftlpu-vs2026-direct/compiler/ftlpu_lower/qwen2_5_1_5b_decoder_layer/decoder_layer.ftlpu
Remove-Item Env:FTLPU_QWEN_MEM_CSV
```

Open `tools/pipeline_viewer/mem.html` and load `decoder_layer.mem.csv`. It records
the MEM ICU state, tile 0..3 pipeline, and committed SRAM transfers. Expect a
large file for the complete seq32 layer.

The binary inspector generates a compact raw-domain CSV for Pipeline Viewer:

```powershell
build-ftlpu-vs2026-direct/runtime/ftlpu_binary_inspect.exe `
  build-ftlpu-vs2026-direct/compiler/ftlpu_lower/qwen2_5_1_5b_decoder_layer/decoder_layer.ftlpu `
  --all-queues `
  --trace build-ftlpu-vs2026-direct/compiler/ftlpu_lower/qwen2_5_1_5b_decoder_layer/decoder_layer.pipeline.csv
```

Open `tools/pipeline_viewer/index.html` and load
`decoder_layer.pipeline.csv`. Each third-counter slice becomes one CSV row;
the first two counters remain `repeat`/`repeat2d` patterns that the viewer
expands only for the visible window. The CSV is an offline view decoded from
the actual raw packets. The CModel result above supplies the execution and
numerical validation.

## Single-token decode reference

`compiler/tests/qwen2_5_1_5b_decode_reference_test.py` uses the real
Qwen2.5-1.5B layer dimensions. It first creates RoPE-transformed K and V caches
for 32 tokens and then performs one decode step at absolute position 32. It
checks the 12:2 GQA mapping, KV append and prefix preservation, decode
attention, residuals, and the 1536/8960 FFN against the last token of a
33-token single-layer mathematical golden.

This test defines the numerical contract needed by compiler decode lowering.
The current `.ftlpu` executable does not yet contain compiled KV-cache
read/write commands, so this reference is not a compiled CModel decode run.

## Historical compatibility numerical baselines

Earlier compatibility-path CModel tests with deterministic sparse INT8 weights
passed all 49,152 BF16 outputs, with 49,127 nonzero results. Their component
checkpoint maximum errors were 0.03125 for FFN, 0 for the attention residual,
and 0.03125 for the second RMSNorm.

The real-checkpoint FFN flow uses `compiler/tools/import_hf_ffn.py` to import
layer-0 Gate, Up, and Down weights plus an embedding-derived BF16 input from the
Hugging Face Qwen2.5-1.5B checkpoint. It applies per-tensor INT8 quantization
and models BF16 MXM dequantization, FP32 partial accumulation, and the VXM FP16
LUT implementation of SwiGLU. Its earlier version-22 compatibility run uploaded
26 weight pages and passed all 49,152 BF16 outputs through cycle 770,646: zero
threshold mismatches, MAE `6.54569e-05`, RMSE `0.000266376`, and maximum error
`0.00418091`.

These results remain numerical baselines for the compatibility schedule. The
current direct binary and its one-context-per-queue validation are reported
above. Neither result executes all 28 decoder layers from the original
checkpoint.
