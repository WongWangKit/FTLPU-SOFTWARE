# GLM-4-9B batched decode

The test covers decoder layer 0 of the original `zai-org/glm-4-9b`, with
official W8A16 weights and full dimensions (hidden 4096, FFN 13696,
32 query heads, 2 KV heads, head width 128). Each request decodes **one**
new token after an independent 32-token history. Batch is not sequence
length. This is not a complete 40-layer model or a generation benchmark.

## Run

First prepare the official checkpoint and imported fixture using the
[prefill entry](glm4_9b_prefill.md). From the SOFTWARE root:

```powershell
python -B compiler/tests/glm4_decode_test.py `
  --checkpoint models/official_dense/glm4_9b `
  --imported tmp/model_tests/glm4_9b/golden `
  --build-dir build-ftlpu-vs2026-direct `
  --target-config ../FTLPU-CMODEL/config/ftlpu-lpu32.json `
  --output-dir results/glm4_9b_decoder_layer_decode `
  --batches 1 32 64 --past-len 32
```

`--resume` reuses completed executions only when inputs and encoded program
match. A changed compiler triggers recompilation; only a byte-identical
program can reuse an earlier execution. Each batch has its own `result.json`;
`status: passed` is the evidence of numerical acceptance, not the existence
of a compiled binary or pipeline file.

When debugging a later phase, `--resume --reuse-completed-phase projections
attention` explicitly keeps the previously executed binaries for those two
phases across a compiler rebuild. Input/graph, hardware and binary hashes
must still match, and numerical checks are repeated. These phases retain
their original compiler provenance; they are not claimed as new executions.

## Device execution

There are three CModel ModelSession invocations, each using raw-macro ICU:

1. Batched input RMSNorm, shared Q/K/V projections, biases and partial RoPE
   at position 32 (first 64 features, preserving the other 64).
2. Independent KV append, grouped QK, masked softmax and PV. QK/PV use the
   generic batched activation MXM lowerer. The 33 live KV positions are
   padded to 64; positions 33–63 are masked. Softmax uses FP32 fixed-port
   ALU chains and row accumulation; layout transposes use wide SXM.
3. Batched output projection, residual, post-attention RMSNorm, shared
   SwiGLU FFN lowering and final residual.

Intermediate tensors cross DDR between these three invocations. Thus the
combined trace is a concatenation of actual executions, **not** a continuous
single ICU program or an overlapped three-stage schedule. Python constructs
mathematical graphs, fixtures and references; it does not emit FU instructions
or replace device activation operations with host arithmetic.

The compiler tiles activation GEMMs into independent mirrored matrix pairs
and limits RHS column tiles to 64 to keep each MEM ICU's domains serial.
Scalar reductions are tiled by SRAM capacity. Completed tiles provide
compiler-declared drain boundaries for finite VXM/SXM iMEM reloads via the
common barrier protocol; physical iMEM capacity checks remain enabled.
Large external KV tensors are distributed over multiple MEM byte pairs.
BF16/FP32 casts use immediate-one multiplication rather than uploading a
matrix of ones for every tile.
The FFN also tiles independent token rows to keep its live activation and
Gate/Up temporaries within SRAM (32 requests per tile for these dimensions).
Batch 64 remains one logical FFN and one device invocation containing both
tiles. Partial FFN sums use the fixed VXM input ports. RMSNorm source homes
are kept off its simultaneous SXM replay-write queues, so disjoint row
addresses cannot hide a same-ICU read/write conflict.

Historical KV is initialized from the quantized prefill reference, using
different permutations of the 32 official embedding rows for each request.
This initializes a cache fixture; it does not claim that the historical
prefill was executed on the device. Device-produced new K/V is appended in
the attention program. Historical KV must be bit-exact on readback.

## Acceptance and artifacts

- Independent random attention-core test: `compiler/tests/glm4_decode_core_test.py`.
- Odd-batch QK/PV regression: `compiler/tests/activation_batch_matmul_test.py`.
- Real-weight attention is checked before the FFN (relative L2 < 3%).
- Full decoder output is compared with the quantized reference (aggregate
  relative L2 < 5%, every request < 8%); new K/V relative L2 must be < 3%.
- FP32 accumulation does not remove BF16 operand/output rounding or the
  runtime exponential/reciprocal approximation error. The reference uses
  the imported INT8 weights; this is not a native-BF16 checkpoint quality test.

Results are under `results/glm4_9b_decoder_layer_decode/batch{1,32,64}`:

- `result.json`, `actual.bf16.bin`, `golden.bf16.bin`, `token_ids.json`.
- `runtime.pipeline.csv`: actual runtime traces with phase offsets.
- `projections/`, `attention/`, `tail/`: compiled and runtime programs,
  execution logs, individual runtime pipeline CSVs and `icu_programs/`.

The runtime trace includes initial input transfers, dynamic weight paging
and waits. Phase durations include their own DDR input/output overhead;
do not interpret the total as optimized continuous-decoder latency.

## Recorded runs (2026-11-04–05)

All entries use `past_len=32` and one new token per request. Relative L2 is
measured against the W8A16 reference. Runtime cycles come from the combined
actual trace, including each phase's transfers and waits.

| Batch | Status | Output relative L2 | Worst request relative L2 | Runtime cycles |
| --- | --- | ---: | ---: | ---: |
| 1 | Passed | 1.13885% | 1.13885% | 5,956,751 |
| 32 | Passed | 0.94251% | 2.02964% | 8,089,807 |
| 64 | Passed | 1.38834% | 6.78253% | 16,614,302 |

Historical K/V was unchanged bit-for-bit in all three runs. The first 32
outputs of batch 64 are bit-identical to batch 32, using the matching input
and cache fixtures. Batch 64's worst request has 6.78253% relative L2, below
the existing 8% per-request limit but materially larger than the aggregate
error; acceptance does not imply sub-percent accuracy for every request.
These runs retain per-phase compiled binaries and completion stamps,
including phases reused after subsequent compiler fixes, so they are not
an identical-build performance comparison.

## Optional tiled online-softmax attention

`--attention-strategy flash` selects the public C++ FlashAttention lowering.
The default remains `baseline`. See [flash_attention.md](flash_attention.md)
for precision boundaries, supported shapes, raw-macro tests, and controlled
same-build attention comparison. Its results use the separate fixed root
`results/glm4_9b_decoder_layer_decode_flash`; the original baseline artifacts
are preserved. For batch 1, one decode token and 32 past KV positions,
singleton PV on VXM and batched Q row transfers reduced the complete
attention invocation from 30,549 to **25,742 cycles** at 500 MHz. This
includes ingress and output transfer of context and updated K/V. The
ordinary attention baseline takes 57,889 cycles with the same compiler,
runner and inputs. The current Flash artifacts are in
`results/glm4_9b_decoder_layer_decode_flash_streamed`, with the baseline
under `baseline_current`. Full decoder layer timings are reported separately.

## Decode FlashAttention with QKV and output projections

The projected-attention test takes layer 0 BF16 hidden states `[B,4096]`
and separate BF16 K and V caches `[B,32,2,128]`. Each request decodes one
token at position 32. The graph runs input RMSNorm, imported W8A16 Q/K/V
projections with biases, GLM partial RoPE on Q/K, cache append, FlashAttention
over 33 positions, and the W8A16 output projection. It returns the projected
hidden update `[B,4096]` and both present caches `[B,33,2,128]`. The residual
add and FFN belong to the surrounding decoder layer.

The fused graph keeps Q/K/V and the attention context on chip. Adjacent
weight pages for the same projection can share its staged activation when
the compiler proves the physical input stays live. The staged control runs
three CModel invocations and transfers intermediate outputs through DDR.
Both use the same input tensors, weights, precision, and hardware target.
The initial history comes from the accepted prefill reference fixture.

From the software repository, use an MLIR-enabled Release build and the
existing GLM import and decode fixtures. Both modes use the runner in the
selected build. The result records that runner and the fixture's original
runner separately.

```powershell
$build = "build-ftlpu-vs2026-direct"
cmake --build $build --config Release --target ftlpu_compile ftlpu_graph_compile compiled_stage_file_runner ftlpu_icu_program_export
python -B compiler/tests/glm4_flash_projected_test.py --build-dir $build --batch 1 --mode both
python -B compiler/tests/glm4_flash_projected_test.py --build-dir $build --batch 32 --mode both
```

The fixed output root is `results/glm4_9b_decoder_layer_decode_flash_projected/optimized`.
Each batch has a staged/fused comparison in `result.json`. The fused
directory contains the actual linked program `actual.runtime.ftlpu`, the
runtime trace `runtime.pipeline.csv`, the per-ICU export `icu_programs/`,
and the numerical results. Total cycles include ingress, physical execution
and output readback, including the full diagnostic K/V outputs. Page waits
are already counted in physical execution. The numerical oracle starts
from hidden states and the W8A16 weights. The harness also compares all
bits of O and both caches between the staged and fused executions.

To measure a runtime linker change on an accepted program, use the replay
driver with the original runner and the candidate executable. It freezes
the ordered raw program inputs, verifies the accepted graph, trajectory
and outputs, and executes the candidate on those exact input bytes.

```powershell
python -B compiler/tests/glm4_projected_runner_replay.py --accepted-dir results/glm4_9b_decoder_layer_decode_flash_projected/optimized/batch32/fused --reference-runner tmp/glm4_projected_reuse_build/runtime/compiled_stage_file_runner.exe --candidate-runner $candidateRunner --output-dir tmp/glm4_projected_batch32_fused_replay
```

Use a separate output directory for each `fused` or `staged` replay.
The result records program, input, runner, target and output hashes,
per-phase timing and bitwise differences. The default reference is the
accepted execution. `--rerun-reference` repeats it on the named historical
runner. `--prepare-only` verifies and freezes the inputs before simulation.

### Measured projected attention, 2026-11-07

The table uses the same final runner for all four measurements, with raw
macro execution, OneVector projection weight loads, 500 MHz and 51.2 GB/s.
The fused and staged programs each fetch 18 weight pages (34 MiB).

| Batch | Three invocations, cycles | One invocation, cycles | Fused time | Reduction | O relative L2 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1,230,329 | 1,206,775 | 2.413550 ms | 1.91% | 0.7511% |
| 32 | 1,510,867 | 1,506,017 | 3.012034 ms | 0.32% | 0.5886% |

| Fused cycles | Ingress | Scheduled execution | Weight wait | Readback including cache diagnostics |
| --- | ---: | ---: | ---: | ---: |
| Batch 1 | 21,121 | 1,165,697 | 16,185 | 3,772 |
| Batch 32 | 43,947 | 1,361,587 | 16,907 | 83,576 |

For batch 32 the initial fused graph took 2,030,437 cycles. Reusing the O
input across eight weight pages reduced this to 1,873,945. Proving regular
MEM macro occupancy with exact consecutive-cycle intervals let the pager
overlap transfers on the same compiled image, reducing waits from 384,835
to 16,907 cycles. Compared with the three-invocation control, fusion saves
70,128 boundary I/O cycles and 22,419 weight-wait cycles, while on-chip
scheduling costs 87,697 additional cycles. The Q packing step gathers
partial-RoPE outputs and untouched channels from separate homes, and the
K tile for the appended position also needs additional staging.

O and both present caches match their staged controls bit for bit for both
batches. The original 32 KV positions are bit-exact. The worst per-request
O relative L2 at batch 32 is 1.0413%. Weight, activation and output traffic
are timed. The runner uses non-timed ICU instruction-image loading. Batch
32's VXM C0/C1 totals (2,250 and 2,628 96-bit words) require segmented
loading into 2,048-word queues. The linked-program capacity audit found 97
planned segments and at most 63/75 resident words on those queues. The
reported latency excludes iMEM loading and replacement costs.

The measured demonstrations use `tmp/glm4_projected_reuse_build` for the
compiler/original runner and `tmp/glm4_projected_overlap_build/runtime/compiled_stage_file_runner.exe`
for the final runner. In the fixed output root, `comparison.json` collects
the real paired results. `batch{1,32}/fused_overlap/candidate/` contains the
final runtime CSV, linked binary and per-ICU exports. The staged control's
runtime artifacts are in the corresponding `staged_overlap/candidate/`.

### Projected attention layout run, 2026-11-10

The projected test now enables the existing C++ `ftlpu.passive_sr_multicast`
policy by default. `--passive-sr-multicast off` selects separate copies.
The compiler also applies its existing default elimination of packed writes
overwritten by a later source. A supported inward SR route performs one MEM
read, a local `WRITE_TAP`, and a remote write. Its source and destination
geometry and queue-alias checks determine eligibility.

The following comparison uses the November 7 continuous programs as the
baseline, the same frozen CModel runner and hardware target, and identical
logical input graphs, weights and numerical policies.

| Batch | Previous cycles | Current cycles | Current time at 500 MHz | Reduction |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1,206,775 | 1,202,656 | 2.405312 ms | 0.3413% |
| 32 | 1,506,017 | 1,401,905 | 2.803810 ms | 6.9131% |

| Current fused cycles | Ingress | Scheduled execution | Weight wait | Readback including cache diagnostics |
| --- | ---: | ---: | ---: | ---: |
| Batch 1 | 21,121 | 1,161,588 | 16,175 | 3,772 |
| Batch 32 | 43,947 | 1,257,550 | 16,832 | 83,576 |

Batch 32's layout transport domains decrease from 3,220 to 2,240. The
actual runtime issues 128,896 fewer MEM reads and 79,360 fewer total MEM
writes including taps. Of the remaining domains, 388 use multicast.
Q assembly domains decrease from 800 to 544, and O input domains from
128 to 32. MXM/VXM arithmetic issue counts and active event-cycle sums
match the prior execution. O and both present caches are bit-exact against
the prior results. Their independent numerical errors match the November
7 table.

The fixed output root remains
`results/glm4_9b_decoder_layer_decode_flash_projected/optimized`. Its
`batch{1,32}/fused/` directories now contain the current runtime CSV,
actual linked ICU program and per-ICU CSVs. The previous execution stays in
`fused_overlap/candidate/`, with its source metadata in
`fused_overlap/accepted_result.json` and
`fused_overlap/accepted_graph_plan.json`. `comparison.json` is the verified
before/after layout comparison. `comparison_before_layout.json` retains
the earlier staged/fused comparison.

The measured compiler snapshot and build hashes are retained in
`tmp/glm4_mirrored_build/build_manifest.json`, alongside the compiled source
snapshot and executables. This isolates the comparison from other compiler
work occurring in the shared workspace. Reproduce the measured run with
the existing imported checkpoint and frozen reference fixtures as follows.

```powershell
python -B compiler/tests/glm4_flash_projected_test.py --build-dir tmp/glm4_mirrored_build --batch 1 --mode fused --passive-sr-multicast on
python -B compiler/tests/glm4_flash_projected_test.py --build-dir tmp/glm4_mirrored_build --batch 32 --mode fused --passive-sr-multicast on
python -B compiler/tests/glm4_projected_comparison.py
```

The current capacity audit is `icu_capacity.txt`. Batch 1 fits the resident
image, while batch 32 still uses the existing 97 planned VXM iMEM segments.
The reported timings exclude instruction-image loading and replacement
costs.
