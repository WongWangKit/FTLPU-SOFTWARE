# Qwen3.5-0.8B GDN decode recurrence

## Full-attention raw Macro decode matrix

`qwen3_5_decode_gdn_attention_test.py` runs the complete real-weight attention
graph for independent batches 1, 32 and 64 over one decode step by default. It includes
input RMSNorm and excludes FFN/residual. This is a **serialized request baseline**
in one ModelSession per step; it does not claim batch-wide projection GEMMs or
the recurrence-only test's two-request hemisphere partition. Checkpoint weight
files are shared. Each request has independent recurrent and convolution state.

The step uploads both histories through C2C and downloads the updated state
and Conv cache to DDR, then ends. No following token is required. Explicit
`--steps N` can still enable a state-continuation regression; it is not part of
the default performance test. Validation checks all three outputs against
a local FP32 reference and an independent FP32 rollout (identical for one step).
Both retain NRMSE < 3% and cosine > 0.999. Request prefixes
are compared byte-for-byte between batch sizes after all steps finish.

MEM/MXM instructions use `--icu-compression raw-macro`; VXM/SXM use their native
instructions. The compiler validates overflowing compute queues using the same
drain-segment planner as runtime. This is not permission to exceed iMEM: every
segment must fit; a missing drain, MEM/MXM overflow or Active Context overflow
remains an error. The runtime leaves MEM/MXM and common synchronization running
while reloading only selected VXM/SXM queues at compiler-declared drains.
Instruction reload hardware latency is not calibrated by this test.

```powershell
python compiler/tests/qwen3_5_decode_gdn_attention_test.py `
  --package results/qwen3_5_0_8b_decoder_layer_prefill/package `
  --build-dir build-ftlpu-vs2026-direct `
  --target-config ../FTLPU-CMODEL/config/ftlpu-lpu32.json `
  --output-dir results/qwen3_5_0_8b_decode_gdn_attention/raw_macro `
  --batches 1 32 64 --steps 1 --resume
```

Each `batchN/stepM` contains a runtime pipeline, actual invocation images,
compute reload segment images, runtime ICU CSVs, per-request outputs and errors.
`summary.json` distinguishes running, failed and passed steps. `--resume` is
only for the same executable, fixture and configuration; use a fresh directory
after changing them. Activity statistics are unions of trace intervals, not
peak arithmetic utilization. Complete matrix validation is confirmed only by
`batch_consistency.json`; a successful first step is not a completed matrix.

The initial raw Macro single-step probe passed with attention/state/Conv-cache
NRMSE 0.8492% / 0.7552% / 0.3392%, matching the previous encoding. Its trace
contains 940,440 cycles at 500 MHz / 51.2 GB/s (1.880880 ms including C2C).

`compiler/tests/qwen3_5_decode_gdn_test.py` tests one state update for each
independent request. The model dimensions come from the imported checkpoint:
16 value heads, key dimension 128, value dimension 128. One FP32 state is
1 MiB per request. This test does not include projections, causal convolution,
Q/K normalization, beta/g projection branches, output normalization/gating,
or the output projection. The recurrence itself has no learned weights.

The fixtures are deterministic synthetic Q/K/V, decay, beta and nonzero
historical states. Q is L2-normalized and scaled by `1/sqrt(128)`; K is
L2-normalized. Heads 0 and 1 exercise zero decay and zero beta respectively.
Different requests have different states and must not attend to or update
one another's state. Batch prefixes are checked byte-for-byte for output
and updated state.
Batch 1 also feeds its actual CModel state output into a second decode step
with different Q/K/V and gates. This separate `carry_step` run checks state
continuation; its latency is not included in the one-token table.

## Execution and precision

The test emits the semantic `ftlpu.kernel.gated_delta_rule` operation with
one token. C++ Tensor and Schedule lowering generates fixed-port VXM
depth-2 chains and MEM domains. No model equations or instruction generation
are implemented in CModel or in Python. The generic
`compiled_batch_file_runner` uses one shared executable in a ModelSession;
each request has independent external input/output bindings. C2C transfers
the historical state in and the updated state out. Initial ingress uses
executable MEM_WRITE_SYNC and actual completion, without inserting a guessed
DDR-duration NOP into the recurrence program.

For batch 1, each hemisphere computes eight distinct heads. For even batches,
each hemisphere computes one independent request (16 heads), with two requests
per invocation. Consecutive pairs alternate state banks. Compiler-authored
MEM_WRITE_SYNC prefetches the next pair while the current pair computes;
ModelSession consumes that exact physical binding in the next invocation.
The final invocation has no next-state prefetch. Every updated state is
written back through C2C, including batch 1. State does not remain resident
between tokens. Current Q/K/V and scalar gates are mirrored; state and result
ownership are partitioned. This is vector recurrence, not an MXM batched GEMM.

Batch latency includes ingress, compute, drains and egress for all requests.
The runtime pipeline covers the entire batch. `invocation_programs` contains
each actual linked ICU image, and `runtime_icu_programs/invocationN` exports
its individual queues, including next-state WRITE_SYNC where applicable.

Current VXM chain ports accept BF16 operands, not arbitrary FP32 SR operands.
The opt-in `ftlpu.vector_bf16_spill` single-token lowering retains the FP32
state interface and FP32 chain arithmetic, but reloads the upper 16 bits of
FP32 buffers, as explicitly truncated BF16. It is **not full FP32 state
arithmetic across memory spills**. The seven stages are:

1. Multiply historical state by decay.
2. Reduce K times the decayed state to obtain the prediction.
3. Subtract prediction from V.
4. Multiply the correction by beta.
5. Form the outer product of K and the correction.
6. Add the outer product to decayed state.
7. Reduce Q times updated state to produce output.

The test reports error against independent FP64 evaluation of the full
recurrence and separately checks a reference with the declared BF16 reload
boundaries. Limits are 2.5% NRMSE against the full-precision equations,
0.02% for reload-reference output and 0.001% for reload-reference state.
These short checks do not establish long-context recurrent stability.

## Run

```powershell
.\tools\build_vs2026.cmd --build build-ftlpu-vs2026-direct --target ftlpu_compile compiled_batch_file_runner -j 4
python compiler/tests/qwen3_5_decode_gdn_test.py `
  --package results/qwen3_5_0_8b_decoder_layer_prefill/package `
  --build-dir build-ftlpu-vs2026-direct `
  --target-config ../FTLPU-CMODEL/config/ftlpu-lpu32.json `
  --output-dir results/qwen3_5_0_8b_decode_gdn
```

Default batches are 1, 32 and 64. Use `--check-only` to revalidate existing
outputs. Each batch directory contains `runtime.pipeline.csv`, exported
`runtime_icu_programs`, execution statistics, and per-request inputs and
outputs. Only a completed, numerically checked batch is marked `passed`.

At 500 MHz / 51.2 GB/s, batch 1 takes 81,320 traced cycles (0.162640 ms).
Full-precision-reference NRMSE is 1.1853% for output and 0.9519% for state;
both outputs exactly match the reload-precision reference for this fixture.
In the batch 4 trace, the first pair computes during cycles 50,604–120,536.
The next pair's state transfers during cycles 50,605–75,688 into bank 1,
and the next invocation consumes the prefetched state without uploading it
again. The completed batches at the same hardware settings are:

| Batch | Entire batch latency | Requests/s | Worst output NRMSE | Worst state NRMSE |
|---|---:|---:|---:|---:|
| 1 | 0.162640 ms | 6148.55 | 1.1853% | 0.9519% |
| 32 | 4.432632 ms | 7219.19 | 1.2322% | 0.9706% |
| 64 | 8.815230 ms | 7260.16 | 1.2322% | 0.9927% |

All requests passed the numerical limits. Identical request prefixes produce
bit-identical output and state across batch sizes. The second decode step
using the first CModel state output also passed. `summary.csv` and
`batch_consistency.json` contain the generated results. Batch 32 and 64
consume 15 and 31 prefetched state pairs respectively. These measurements
cover the recurrence core only, not the complete attention or model.

## Complete attention decode and carried Conv history

`qwen3_5_attention_runtime_test.py --decode-token 0` exercises input RMSNorm,
real checkpoint W8A16 projections, causal Conv1D with a nonzero history,
SiLU, Q/K L2 normalization, beta/decay branches, recurrence, output
normalization/gating and output projection. FFN and residual are excluded.
`--decode-token 1 --previous-step <step0>` feeds the actual CModel state and
Conv cache outputs into the next token. Only tensor serialization occurs on
the host; no runtime activation arithmetic or reference substitution occurs.
The complete graph currently uses mirrored recurrence placement; the
head/request partition and bank prefetch measurements above are core tests.

Results are in `results/qwen3_5_0_8b_decode_gdn_attention/step0` and `step1`.
Both complete attention steps pass the independent FP32-equation reference
at the original limits (NRMSE < 3%, cosine > 0.999):

| Step | Attention NRMSE | State NRMSE | Conv cache NRMSE | Runtime including C2C |
|---|---:|---:|---:|---:|
| 0 | 0.8492% | 0.7552% | 0.3392% | 1.880880 ms |
| 1, actual carried state/cache | 0.9242% | 0.6428% | 0.2853% | 1.880880 ms |

Each directory contains a real runtime pipeline and linked ICU queue CSVs.
The complete graph currently exceeds physical VXM iMEM capacity: queues 0/1
use 2,211/4,413 words versus 2,048 each. These are successful CModel numerical
runs, not a claim that the whole continuous image fits the target iMEM.
`physical_imem_overflows` in the summary records this separately. Reducing
VXM descriptors or compiler-managed executable partitioning is still needed
for deployment of this full graph; increasing a simulator limit does not
resolve it.

The packed-to-scalar graph regression `gdn_decode_graph_runtime_test.py`
checks both hemispheres using nontrivial FP32 gates and a nonzero state.
SXM unpack is local to one hemisphere; mirrored scalar destinations are
populated explicitly through passive SR copies. The test matches the
declared reload-precision reference exactly.

## Batched projections for one decode step

`qwen3_5_decode_gdn_batched_test.py` separates the batch-wide linear layers
from request-local state updates. Input RMSNorm and QKV/Z/beta/decay
projections consume `[batch, hidden]`, and output projection consumes
`[batch, value_width]`. Each weight matrix is applied once to the batch;
compiler weight pages and output tiles still produce multiple MXM intervals.

Conv history and recurrent state remain independent for every request. The
current bounded implementation invokes the Conv/norm/recurrence graph for
one request at a time, then collects its core output for the batch output
GEMM. It is not a sequence of batch tokens in a shared recurrent state.
Every state and updated Conv cache is written back to DDR after this single
step. The Python harness only serializes actual CModel tensor outputs; all
activation arithmetic uses the shared C++ Kernel lowering and raw macros.

There are three execution phases with real DDR intermediate transfers:
projections, request-local mixer, and output projection. The combined
`runtime.pipeline.csv` concatenates their actual traces and labels each row
with its phase. No DMA interval is removed and no cross-phase overlap is
assumed. This is not yet a continuous single-image executable. Reported
cycles exclude host orchestration and uncalibrated iMEM reload overhead.

This test also covers nonzero state ingress before preceding normalization
operators. Its SRAM lifetime starts at initial upload, not at the recurrence
consumer; otherwise an earlier normalization transpose can overwrite the
first state head. State accuracy is checked per head as well as in aggregate.
Partial normalization tiles preserve external input bytes and export only
their live rows; `l2norm_runtime_test.py` covers both RMSNorm and L2Norm with
16 external rows.

Example from the software repository:

```powershell
python compiler/tests/qwen3_5_decode_gdn_batched_test.py `
  --package results/qwen3_5_0_8b_decoder_layer_prefill/package `
  --build-dir build-ftlpu-vs2026-direct `
  --target-config ../FTLPU-CMODEL/config/ftlpu-lpu32.json `
  --output-dir results/qwen3_5_0_8b_decode_gdn_batched `
  --batches 1 32 64
```

Validated on 2026-10-28 with real Qwen3.5-0.8B W8A16 weights, raw macros,
500 MHz and 51.2 GB/s peak DDR (90% scheduling efficiency):

| Batch | Trace cycles including C2C | Latency | Worst attention NRMSE | Worst state NRMSE |
|---|---:|---:|---:|---:|
| 1 | 928,709 | 1.857418 ms | 0.8492% | 0.7552% |
| 32 | 14,817,812 | 29.635624 ms | 1.1761% | 0.7745% |

All 512 state heads in batch 32 pass the 3% / cosine > 0.999 limits;
the maximum per-head NRMSE is 2.4717%. Batch 1 output, state and Conv cache
are bit-identical to the original continuous attention fixture. The eight
normalization regressions also pass. Batch 64 is supported by the test CLI
but has not been validated for this batched-projection path yet.

The old serialized batch 32 baseline took 60.192558 ms. Batched projections
reduce this to 29.635624 ms (2.03x), and projection weight page traffic falls
from 224 pages / 324 MiB to 7 pages / 10.125 MiB. Request-local Conv/state
updates remain serialized. Results and the combined runtime pipeline are in
`results/qwen3_5_0_8b_decode_gdn_batched/batch32/step0`.

## Single-batch MXM old-State retrieval

`qwen3_5_attention_runtime_test.py --decode-token 0 --decode-retrieval mxm
--icu-compression raw-macro --trace` selects the compiler policy
`ftlpu.gdn_decode_retrieval_mxm`. The shared graph compiler preserves this
policy through region extraction. It replaces only the retrieval
`remembered = key @ decayed_state`; decay, the delta update, outer product,
state addition and final `query @ updated_state` remain on VXM.

The current decode layout contains scalar-broadcast K and row-packed State.
The compiler constructs identity vectors using VXM/SXM, uses MXM to pack K
and transpose four State tiles, then computes the four K=32 products into
the same FP32 ACC row. ACC clears after the final output, not after the first
partial product. The preparation uses the existing product scratch while
that scratch is dead, and all intermediate data stays on chip. The resulting
FP32 retrieval vector feeds the existing state-update chain.

This first path supports mirrored 128-wide heads, including the real batch-1
Qwen3.5-0.8B decoder. Request/head partitioning is rejected when this policy
is selected. Native queue reloads happen only at compiler drain boundaries;
the following block is kept after the completed drain. Layout preparation is
explicit in the runtime trace and may cost more than the replaced VXM sum.

`gdn_decode_mxm_retrieval_test.py` compares real raw-macro VXM and MXM
executables with the same nonzero state, Q/K/V and gates. The fixed seed
produces bit-identical output and updated State. The complete real-weight
batch-1 run additionally validates against the independent FP32 model
reference at the existing NRMSE < 3%, cosine > 0.999 limits.

The real-weight batch-1 test passed at 500 MHz / 51.2 GB/s configured DDR:
1,291,160 runtime trace cycles, or 2.582320 ms including C2C. Attention,
State and Conv cache NRMSE are 0.8492%, 0.7552%, and 0.3392% respectively.
This first mapping is slower than the 940,440-cycle / 1.880880-ms VXM
baseline because the existing layouts require repeated identity generation,
packing and transposition. It establishes the actual MXM execution path,
not a performance improvement. Results and linked runtime ICU programs are
under `results/qwen3_5_0_8b_decode_gdn_attention/raw_macro/batch1/mxm_retrieval`.

## Batch-1 wide SXM to L2Norm streaming (2026-10-29)

`qwen3_5_decode_qkv_wide_sxm_test.py --with-qk-norm` runs real-weight QKV,
decode Conv1D/SiLU, per-head Q/K L2 normalization, and Conv cache export.
The native planar lowering prepares only missing source words on each
hemisphere, reusing padding aliases. Both wide SXMs then issue four 32x32
tiles at a 32-cycle interval. East output streams 0/1 and West streams 16/17
feed the mirrored fixed VXM chains directly through the westbound SR fabric.
MEM `write_tap` saves x without consuming the stream. This removes the
post-transpose mirror copy and the first-pass MEM reread. Normalization still
replays x after the complete 128-feature sum and reciprocal square root.

The target models `SxmResult -> VxmInput` latency as the SR hop distance plus
the consume cycle. No model-specific CModel operation or host computation is
used. Runtime square-pass intervals are Q [287507,287635) and K
[288346,288474); their SXM runs start at 287458 and 288297 respectively.
The square pass overlaps subsequent transpose tiles and has no gaps.

At 500 MHz / 51.2 GB/s configured DDR, the complete test changes from
302739 to 302207 runtime trace cycles (0.605478 to 0.604414 ms), including
source preparation and C2C. All component numerical metrics are unchanged;
Q/K L2 NRMSE against actual Conv output is approximately 0.1863% / 0.1864%.
Results are updated in
`results/qwen3_5_0_8b_decode_qkv_conv_qk_norm_wide_sxm/batch1`, including
`runtime.pipeline.csv` and `sxm_vxm_streaming_comparison.json`.

## Complete-head ownership through decode producers (2026-10-29)

The batch-1 QKV/Conv/L2 test now enables `ftlpu.l2_head_partition`. The C++
graph planner derives head width from L2Norm and channel width from Conv.
It permutes immutable projection weight columns before paging and restores
logical channel order with tensor views. For each Q/K/V page, heads 0–7 are
produced on East and 8–15 on West. Conv applies the same channel ownership
to its input history, coefficients, and output views; no Python lowering or
host activation permutation implements this policy.

Each SXM reads complete local heads. Unused rows in its 32x32 tile alias a
local padding head; they do not require the other hemisphere's live heads.
The two VXM chains execute shared configurations on different heads. There
is no `l2norm.prepare_sxm_pair` feature-copy phase. A scalar-vector exchange
remains after reduction because fixed VXM outputs cross to the opposite MEM
fabric. The final normalized features stay at that native output home
(heads 0–7 West, 8–15 East), with explicit logical-row mapping carried through
allocation and consumer/export addressing instead of a copy back.

The real-weight test, per-head numerical checks, nonzero Conv-cache check,
wide-only SXM check, and runtime streaming assertions pass. Numerical
metrics match the preceding mirrored implementation. The complete runtime
trace is 302138 cycles / 0.604276 ms at 500 MHz, down 69 cycles from 302207.
Both ingress and egress now use two SXMs; Q square reduction occupies
[287329,287457), K [288120,288248). Current results remain in the same batch1
directory; `head_partition_comparison.json` records the comparison.

## Decode beta/g gates with ALU Log (2026-10-29)

The same real-weight batch-1 test accepts `--with-gates --log-softplus`.
It appends the W_b/W_a projections and beta=sigmoid(x W_b),
g=-exp(A_log)*softplus(x W_a+dt_bias). C++ lowers the semantic graph;
Python supplies inputs and an independent numerical reference only. The
checkpoint constant -exp(A_log) is folded before execution, while the
activation-dependent softplus executes on VXM.

The opt-in Log softplus uses t=exp(-abs(x)), a threshold of 1/16, and
max(x,0)+log1p(t). Above the threshold it uses the hardware Log LUT;
below it uses a short cubic correction. This avoids rounding 1+t to 1 at
the FP16 Log input. Direct Log alone had 5–8% errors on some small real
gates. The existing EXP FP16 flush-to-zero rule still rounds one expected
g of approximately -3e-9 to zero; per-head validation uses 5% relative plus
1e-7 absolute tolerance, in addition to aggregate error checks.

The combined test passes: beta NRMSE 0.0003769494, g NRMSE 0.0085850841;
Q/K L2 errors remain unchanged. The runtime trace includes `log_lut` at
cycle 297067, uses wide SXM only, and reports zero host operations. Total
trace duration is 311037 cycles (0.622074 ms at 500 MHz / 51.2 GB/s), versus
302138 cycles before gates. Instructions remain raw-macro, with dynamic
C2C synchronization. Results overwrite the same batch1 directory above.

This is still a frontend test: it excludes RMSNorm, recurrent state update,
output projection, and FFN. Gate pointwise operations currently use the
shared mirrored BF16-spill vector chains; the complete-head ownership
policy described above applies to QKV/Conv/L2, not yet to the gate branches.

## State ingress and decay only (2026-10-29)

`--with-state-decay --log-softplus` extends the same frontend with runtime
DDR ingress of a deterministic nonzero FP32 state and `exp(g) * state`.
No correction, outer-product update, or query readout is included. The
state is 16 heads x 128 key dimensions x 128 value dimensions (1 MiB).
For this elementwise-only test its external logical layout is
`[key*value, head] = [16384,16]`: a packed gate vector broadcasts over
matrix positions without any additional SXM operation. This is not the
complete-head-per-hemisphere layout used by QKV/Conv/L2; suitability for
the later matrix operations remains to be established.

The compiler tiles long pointwise row domains while preserving one-row
broadcast operands. Pure VectorAlu arguments bypass the generic tensor
allocator and are assigned by the vector memory planner. Four logical key
ranges are exposed separately to the numerical harness (outputs 9–12),
avoiding a second contiguous full-state archive solely for export.

The completed run passes all per-head checks. State ingress occupies
[3523,16112), [16112,28740), [28740,41299), [41299,53917), with runtime
MEM_WRITE_SYNC activity. These are initialization transfers before compute,
not overlapping demand-paged state loads. Internal staging copies remain.
The state ABI and multiply output are FP32, but this shared fixed VXM path
truncates chain inputs to BF16. NRMSE relative to FP32 exp(actual g)*state
is 0.005467758; relative to reference g it is 0.005413339. This does not
validate accumulated recurrent precision across tokens.

The entire instrumented frontend, input transfers, and validation output
transfers take 551886 runtime trace cycles / 1.103772 ms at 500 MHz.
That is not the standalone decay latency. There are still only the eight
wide SXM trace rows for Q/K normalization and zero host operations. The
Log gate regression also passes after the allocator changes. Results and
runtime pipeline overwrite the same batch1 directory.

## MEM transport audit and direct consumers (2026-10-29)

`graph_plan.json.linker_mem_transports` now records each physical graph-copy
domain (phase, route, source/destination, vector count and nominal interval).
`mem_transport_audit.json` separates these from native operator internal
traffic and validation exports; the domain totals are not total MEM traffic.

Packed VXM external operands and linear-projection activations now read
their lifetime-safe C2C homes directly. This removes the four full state
input copies and the five repeated projection input copies. Exact physical
identity copies are skipped. The paired decode Conv carries native window
word addresses through Tensor/Stream lowering, so its taps directly read
history and QKV homes instead of materializing a concatenated window. Only
288 BF16 vectors (18 KiB) of opposite-side history still transfer before
the two shared-ICU chains start. That transfer has a common ingress phase
to avoid simultaneous reads from the same MEM bank by the other chain.

The real-weight/nonzero-state test and Log gate regression pass. All
component error metrics match the preceding implementation. Model execution
cycles decrease from 310751 to 241920; complete runtime trace including
initialization and validation output transfers decreases from 551886 to
507811 (1.015622 ms at 500 MHz). The test asserts that no graph materialization
domain remains in projection, Conv-window, or state-decay input phases.

Remaining graph copies consist of 256 physical bytes for gate mirroring
and diagnostic output materialization. Native Conv still spills MAC results
for the separate SiLU chain; L2Norm retains x until its full-feature reduction
is available; gate programs retain fixed-chain spill boundaries. These are
not removed by address aliasing and have not been claimed to be theoretically
minimal. In particular the 18 KiB history transfer could be addressed by a
future partitioned C2C history layout, and removing diagnostic export copies
requires a native segmented output ABI rather than deleting numerical checks.

## Matrix state and wide-SXM coefficient broadcast (2026-10-30)

The state-decay test now presents state as `[head*key, value] = [2048,128]`.
Each lane represents one of 32 adjacent value coordinates, rather than a
head. The C++ row tiler produces two 8-head blocks. Within each block the
physical planar address is `base + valueBlock*1024 + localHead*128 + key`;
lane selects `value%32`. The logical state remains 1 MiB FP32, but no longer
wastes half of every vector on padded head lanes. Both hemispheres currently
retain mirrored data; this change does not partition the 16 heads between them.

The semantic multiply uses a repeated coefficient view. C++ recognizes it
and allocates only a 16-row vector table. It reads the packed exp(g) word
32 times with address stride zero and executes a wide 32x32 transpose.
Each useful output row is one head coefficient repeated across all lanes.
Two byte-pair passes preserve the four-plane FP32 representation, on both
hemispheres; no legacy transpose/permute or host activation expansion is used.
The table is generated once and reused by both state blocks. Each coefficient
address repeats for 128 key rows and four value blocks (512 state vectors).
The multiplier still consumes the upper BF16 planes of state and coefficient,
with FP32 arithmetic and output, as before.

The real-weight/nonzero-state test passes per-head checks and confirms four
additional SXM trace rows. Broadcast captures occur at runtime cycles 319727
and 319864 on both East and West. The table is on slices 32–35, bank1,
rows 4192–4207 in this allocation. State NRMSE relative to exp(actual g)*state
is 0.005468359; the gate regression also passes. Model execution takes 233552
cycles, versus 241920 before; complete trace including initialization and
validation exports takes 396159 cycles / 0.792318 ms at 500 MHz. Much of the
complete-trace reduction comes from removing padding in state transfers.
State validation outputs are now 9 and 10, each containing eight full heads.

## Retained state decay streamed into KS (2026-10-30)

`qwen3_5_decode_qkv_wide_sxm_test.py --with-ks --log-softplus` now extends
the same real-weight batch-one frontend through `p = K^T (exp(g) * S)`.
It does not yet calculate the correction, outer-product update, or query
readout. Both the scaled state and prediction are numerically observed.

The compiler primitive `ftlpu.kernel.scaled_matvec` has two results: the
scaled matrix and its grouped vector contraction. Python supplies this
equation, immutable external-state bytes, and references; physical tiling,
coefficient broadcast, placement, and hardware domains are C++ lowering.
There is no new model-level CModel operation.

For this fused test the external state ABI is `[head,value,key]`, flattened
to `[2048,128]`, replacing the previous decay-only `[head,key,value]` ABI.
It represents the same logical matrices with the last two axes exchanged
in storage. C2C loads that orientation directly. All 16 heads now occupy one
full-depth tile. The address is `base + keyBlock*2048 + head*128 + value` and lane is
`key % 32`. Thus a VXM result vector is already one MXM weight column;
no runtime transpose of the state is needed. Both hemispheres still contain
mirrored state; this is not independent head partitioning.

Wide SXM repeats the packed coefficient vector 32 times and transposes its
upper BF16 byte pair into per-head broadcast vectors. Each head tile retains
the full packed coefficient input and selects the corresponding broadcast
rows with `group_offset`; slicing heads 8–15 by changing a packed word's
address would incorrectly select heads 0–7. The remaining lower byte planes
are unnecessary for this multiplier's existing BF16 input contract.

For each 32x32 block, the VXM depth-eight chain multiplies state by its
coefficient. `MEM_WRITE_TAP` saves all four output byte planes without
consuming the stream. The upper two planes continue along the forward SR
fabric into MXM's Direct16 IW port. A single column-induction LOAD_3D covers
all 32 weight columns. MXM contracts the corresponding packed K vector;
ACC retains partial sums across the four key blocks. Physical weight buffers
alternate, admitting a new block every 32 cycles while the previous block
computes. At tile start `t`, VXM emits 32 consecutive vectors; IW starts at
`t + 8 + route_latency` and compute issues immediately after column 31.
The next tile starts at `t+32`. Buffers alternate across head/value boundaries
as well as reduction blocks. KS returns on W24-W27, separate from the W0-W3
state/coefficient inputs, avoiding an SR collision during continuous issue.
The final partial drains in FP32. **No MEM read of the scaled state
is emitted by the contraction.** State is retained for a later update.

The graph planner no longer limits this primitive to half a bank: input and
output have separate slice groups, each with 8192 available rows. Lifetime
and queue allocation checks still apply. This removes the former 642-cycle
head-group boundary. VXM and IW each issue continuously for 8192 cycles.
MXM compute still issues once per 32-column tile for batch one; this is a
weight-streaming limit, not an inserted wait. Filling its compute queue every
cycle would require a different mapping or reuse across multiple vectors.

Validation:

- `scaled_matvec_streaming_test.py` checks signed nonzero matrices, multiple
  reduction blocks and a nonzero coefficient-group offset. `--key-dim 96`
  checks buffer alternation across odd reduction-block counts. Saved state
  is exact relative to the BF16-truncated-input reference; KS NRMSE is
  4.73e-8 for K=64 and 6.70e-8 for K=96.
- Real-weight batch-one test passes all 16 heads. KS NRMSE is 8.01e-8 relative
  to the actual scaled state with the same BF16 truncation as MXM; relative
  to untruncated state times actual exp(g)/K it is 0.007336. This second
  comparison includes the intentional state/weight quantization errors.
- Generated schedule audits zero scaled-state read domains and continuous
  VXM/IW issue. Runtime CSV independently verifies 256 adjacent 32-cycle IW
  intervals on each hemisphere. Runtime uses raw macro instructions, wide
  SXM, and zero host operations.
- Final execution is 233676 cycles versus 242317 for the earlier 64-cycle
  spacing and two head groups (282957 for the original conservative schedule).
  The complete runtime trace is 449874 cycles
  (0.899748 ms at 500 MHz), including input initialization and diagnostic
  output exports; it is not solely compute latency.

Results remain in
`results/qwen3_5_0_8b_decode_qkv_conv_qk_norm_wide_sxm/batch1`.
Output 9 is the complete scaled state in the new orientation; output 10 is
the complete KS result. Exporting whole tensors avoids materializing two
strided half-head views solely for diagnostic readback.
Coefficient transpose issues at cycle 319944. VXM C0 runs continuously in
`[320135,328327)` and both MXM IW ports in `[320160,328352)`; all eight
state byte-plane WriteTap queues also have 8192 active cycles and zero gaps.
Use viewer range 320080–320500 to inspect overlapping VXM, MEM WriteTap and
MXM activity. `summary.json` and `state_decay_ks_comparison.json` record the
measurement scope and numerical results.

## Streamed correction epilogue (2026-10-30)

Run the same test with `--with-correction --log-softplus` to append
`delta = beta * (V - K^T Sbar)`. This does not update state or calculate Q's
readout. The generic `scaled_matvec` primitive accepts optional epilogue
operands `(value, coefficient)` and returns a third result; physical lowering
and placement remain in C++, with no CModel or host arithmetic added.

The existing continuous stream is split into two depth-four chains per
hemisphere. C0-C3 multiply/forward Sbar, retaining its FP32 image with
WriteTap and loading MXM from its BF16 high planes. MXM returns KS in FP32
on W6-W9 (East) and W22-W25 (West). A diagnostic WriteTap retains it, while
its high two planes continue directly into C4's fixed LHS port. V enters
C4's RHS port. C4 computes `KS - V`, C5 multiplies by its locally captured
beta, C6 negates, and C7 writes FP32 delta. The sign reversal accommodates
the fixed input ports without moving KS. No MEM read of either Sbar or KS
is emitted by this fused operation.

Wide SXM broadcasts beta once, using a separate input stream pair from the
decay coefficient. Each head captures beta in C5 once and reuses it for
four value blocks. The shared VXM input Bundle requests a newly decoding
chain's ports one cycle before that chain executes. While C0 is active,
the compiler supplies one preview beat (an extra V read and a coefficient
read on the KS port); C0 discards it and C4 consumes only the next real KS
beat. The final drained tile does not need that preview. These are small
operand reads, not a KS tensor spill or a second state traversal. External
V placement is kept off the state/K read queues.

Validation includes the existing non-epilogue regression, a nonzero group
offset, odd reduction-block counts, and the real-weight batch-one graph.
Correction is exact against the reference using the same BF16-truncated KS
and beta inputs. Against untruncated actual KS/beta, NRMSE is 0.005092.
KS retains its prior 8.01e-8 NRMSE. State and IW streaming still have 8192
consecutive active cycles. The preexisting K layout preparation remains;
V now has its own one-time preparation from the Conv output view. There is
no layout conversion between KS and delta.

Results overwrite the same `batch1` directory. Outputs 9, 10 and 11 are
Sbar, KS and delta respectively. Runtime execution is 234076 cycles, up
400 from the KS-only test. The complete trace is 450764 cycles / 0.901528 ms
at 500 MHz, including initialization and diagnostic exports. VXM C0 runs
in `[320535,328727)`, IW in `[320556,328748)`, and the first correction
instruction issues at 320702; subsequent value blocks issue every 128
cycles. Viewer VXM rows show ICU issue timing, not each ALU's internal
pipeline arrival. All 16 heads pass; `summary.json` records scope and errors.

## Rank-one state update (2026-10-30)

Run with `--with-update --log-softplus` to append
`Snew[h,v,k] = Sbar[h,v,k] + delta[h,v] * K[h,k]`.
This implies the correction/KS/decay/frontend flags. Query readout is still
excluded. The generic `scaled_matvec` optional fourth result is the updated
matrix; existing two- and three-result programs retain their behavior.

Wide SXM repeats each packed 32-element delta vector 32 times and transposes
it into 32 broadcast vectors. Only the BF16 high planes enter this table.
The state remains in `[head,value,key]` order, with `lane=key%32`; neither Sbar
nor Snew is transposed. For each head and key block, C1 first captures the
32 K elements into the 32 lanes' local scalar registers. C0 then admits a
broadcast delta as LHS and the corresponding Sbar vector as its auxiliary
RHS. C1 multiplies delta by local K, C2 adds the carried auxiliary Sbar, and
C3 writes FP32 Snew. All 128 value rows run consecutively before changing K.
This reduces coefficient captures from 2048 per-value loads to 64 K loads.
There is no stored outer-product intermediate and no layout copy of S.

Multiply has a two-cycle result path. Between consecutive 128-vector runs,
the compiler leaves one drain beat followed by one K-capture beat. Runtime
audits verify 64 runs and 8192 useful vector updates, with exactly two issue
cycles between runs. The original decay/IW phase remains continuous for
8192 cycles on both hemispheres. This implementation first broadcasts the
delta table, then updates state; these two phases are not yet overlapped.

Sbar is read once for this update (only its BF16 high planes). The updated
matrix reuses the old external state's MEM home after its last read. The
planner permits that reuse only when no later consumer or retained snapshot
needs the input and no concurrent input/workspace shares its MEM ICU queues.
Otherwise it allocates a separate output. Sbar remains separately available.
The real graph uses slices 16-19 bank 1 for old/new S and slices 12-15 bank 1
for Sbar, on both hemispheres. This state path remains mirrored.

Validation covers a single 32x32 tile, multiple heads with K=96 and nonzero
group offset, the prior correction path, and a subsequent consumer of the
original matrix (which must prevent destructive reuse). The real-weight
batch-one graph passes for all 16 heads. Snew matches the reference using
the same BF16-truncated Sbar/delta inputs exactly. Relative to the same
actual Sbar/delta without this truncation, NRMSE is 0.00221562; the multiply
and following add execute in FP32 but this is not full-FP32 input arithmetic.

Output 12 is the complete updated state; outputs 9/10/11 remain Sbar/KS/delta.
Execution increases from 234076 to 247644 cycles (+13568). The full trace is
562677 cycles / 1.125354 ms at 500 MHz; it includes initialization and all
13 diagnostic exports, including the extra 1 MiB Snew, so its delta is not
the compute-stage delta. Runtime uses raw macro, wide SXM, and zero host ops.

The new SXM broadcast instructions issue over cycles 328856-333896. VXM C1
multiply/C2 add issue over `[334040,342358)`. Viewer range 334000-334600 shows
the first four runs; range 328800-342450 shows the entire appended stage.
Results overwrite the existing `batch1` directory; `summary.json` and
`state_rank_one_update_comparison.json` record numerical and runtime audits.

## Overlapped delta broadcast and state update (2026-10-30)

The same `--with-update --log-softplus` test now overlaps the next head's
delta broadcast with the current head's update. The serialized measurements
in the preceding section are the before-optimization baseline.

The broadcast workspace is two BF16 head buffers on slices 50-51, alternating
bank 1 and bank 0. Each bank holds 128 vectors for the real model, mirrored
on both hemispheres: 32 KiB physical storage in total. SXM writes W4-W5
(W20-W21 on the mirrored side); VXM reads delta on W0-W1 and Sbar on W2-W3.
Separate SR pairs prevent simultaneous producer/consumer stream collisions.
No S layout conversion or outer-product intermediate is introduced.

Head 0 fills its buffer first. While head h runs four 128-vector multiply/add
segments, SXM fills head h+1 in the other bank. The buffer formerly used by
head h-1 is not overwritten until that head's reads have retired. If a small
reduction dimension makes the consumer faster than the broadcaster, only
the next-head boundary waits for readiness. The 128x128 real-model case
has no added head-boundary wait: the existing two issue beats between key
blocks (drain and K capture) are unchanged. No ICU queue interleaves macros.

Real-weight batch1 passes with zero error against the same BF16-input
reference; the untruncated-input comparison remains NRMSE 0.00221562.
The appended stage decreases from 13568 to 8768 cycles (-4800, 35.38%).
Whole-test execution decreases from 247644 to 242844 cycles. Full trace,
including initialization and the same 13 diagnostic exports, is 557877
cycles / 1.115754 ms at 500 MHz. Prior decay/MXM streaming remains intact.

Runtime VXM multiply/add issues span `[329240,337558)`. The 64 East SXM
broadcast instructions issue from 328856 through 336760, overlapping VXM;
West follows the same schedule. Range 328800-330400 shows the startup and
first overlapping heads. The result directory is unchanged, and
`delta_broadcast_overlap_comparison.json` records the before/after data.

Regression checks cover a single 32x32 tile, two heads with K=96/nonzero
group offset, and K=64 with a later consumer of the original matrix. The
latter also exercises a producer-limited head boundary. Schedule audits
check both broadcast banks, address bounds, nonoverlapping MEM ICU domains,
and next-head SXM issue alignment with current-head multiplication.

A separate H=2, V=64, K=32 probe still fails in the preexisting correction
phase at cycle 541 (input Bundle not ready), including with `--with-correction`
and no rank-one update. This optimization does not change that earlier
short-reduction epilogue limitation; the real Qwen K=128 case and single
32x32 case pass.

## Streamed Q readout (2026-10-30)

Use `--with-readout --log-softplus` with the same batch-one test and output
directory. This includes the frontend, gates, retained state decay, KS,
delta, state update, and now `y[h,v] = sum_k Q[h,k] * Snew[h,v,k]`.
Q is the existing normalized and scaled query: no second query scale is
applied. Gated output RMSNorm, z projection, and output projection are not
part of this incremental test yet.

The generic kernel primitive accepts an optional third epilogue operand
(readout vector) and fifth total result (readout). The graph planner handles
its input placement, output queue, and head slices. CModel receives only
normal raw-macro MEM/VXM/MXM/SXM instructions; no model-level operation or
host evaluation is introduced. Previous 2/3/4-result interfaces remain valid.

The readout path traverses update tiles in head, value_block32, key_block32
order. For each tile, C1 captures K and C0-C3 update 32 vectors. FP32 Snew
is saved by MEM_WRITE_TAP, while its high two byte planes continue over SR
to MXM IW. The next tile uses the other weight buffer. MXM contracts with
the appropriate Q vector, retains ACC across key blocks, and emits/clears
the FP32 result only on the final key block. There are no MEM reads of Snew
inside this native stage. The output readback at the end of the test is a
diagnostic export, not a feed into MXM.

Readout output and Snew use different MEM ICU queues, since a result returns
while the next state tile is still being written. The existing next-head
delta broadcast ping-pong remains overlapped. Update segments change from
64 runs of 128 vectors to 256 runs of 32 vectors; each still has the two-beat
drain/capture boundary. This is the cost of completing each value tile's
reduction with the existing MXM accumulator. Snew's physical matrix layout
is unchanged, and the original continuous state-decay/IW phase is preserved.

Single-tile and H=2/K=96 tests pass, including odd key-block parity and
cross-block accumulation. The prior update-only path with retained old
matrix use also passes. Real Qwen3.5 batch1 passes for all 16 heads:
readout NRMSE is 8.3795e-8 against the actual BF16 high-plane Snew and BF16 Q
reference (max absolute error 9.3132e-10). Against untruncated actual Snew,
NRMSE is 0.00340118. MXM accumulates in FP32, but its state input is BF16.

Output 13 is readout `[16,128]`; output 12 remains the complete updated state.
Whole-test execution is 243644 cycles, +800 versus update-only. This includes
the query input preparation, additional tile boundaries, and final drain;
it is not 800 cycles of standalone matrix multiplication. The full trace
is 559039 cycles / 1.118078 ms at 500 MHz, including input initialization
and all 14 diagnostic outputs. There are zero host operations.

Runtime update issues span `[329560,338262)` and readout IW activity spans
`[329581,338283)` on both hemispheres. Range 329500-330200 shows the first
state tiles, their WriteTap fanout, and MXM computations. `summary.json`
and `query_readout_comparison.json` record numerical and schedule audits.

Runtime wide-SXM transpose bars cover the full fixed FU span from an actual
runtime issue: `[issue,issue+70)`, including superlane skew. Input reception
is `[issue,issue+35)` and output is `[issue+35,issue+70)`. These timings come
from the CModel wide-transpose implementation; they are not static compiler
schedule timestamps. Missing scheduled input is an execution error rather
than an SXM stall. The CSV detail labels this `timing=fixed_fu_span` and
`issue=start`. Initiation interval remains 32 cycles, so successive bars may
overlap. Trace compression preserves individual jobs and their issue counts.
Legacy local-block SXM and permute trace conventions are unchanged.

## Decode Z projection and gated output RMSNorm

Run the incremental real-weight batch1 graph with `--with-output-norm
--log-softplus`; this includes the readout stages above and appends Z
projection, per-head RMSNorm, and `normalized_y * SiLU(z)`. It does not yet
include output projection, residual, or input RMSNorm. The existing batch1
result directory is reused. `--validate-only` checks existing artifacts.

The `[16,128]` readout uses an explicit `indexed_view {truncate_bf16}`:
FP32 high byte planes alias BF16 without a MEM copy or ALU cast. The shared
RMSNorm/L2Norm planar-ingress scheduler streams wide SXM output into the
square reduction and retains the input for normalization replay. RMSNorm
still uses **mean** square plus epsilon and checkpoint `delta_norm` gamma
(without a `+1` offset); it is not replaced by L2Norm. Its scratch uses
disjoint bank-0 queues so retained FP32 state remains intact in bank1.

The C++ projection planner partitions complete Z heads, including weight
column ordering, so Z and normalized readout have matching ownership:
heads 0-7 on East and 8-15 on West. The final VXM chain crosses fabrics,
leaving gated outputs on West and East respectively. No runtime activation
reordering is performed by Python. The model supplies only tensors and
equations; native scheduling remains in the compiler.

The compiler recognizes the generic `x * SiLU(z)` arithmetic DAG and emits
one bilateral depth-8 VXM chain: C0 negate z and capture x as auxiliary,
C1 exp, C2 add 1, C3 reciprocal, C4 multiply original z, C5 multiply auxiliary
x, then C6/C7 drain. Each side consumes 32 consecutive vectors. Both inputs
read directly from producer addresses with affine MEM domains; sigmoid and
SiLU do not spill to MEM. Normalization gamma is an immutable broadcast
constant and currently has a small staging copy. Normalization replay is
also retained because its inverse RMS depends on the full head reduction.

Real-weight validation passes for all 16 heads, raw macro and wide SXM only,
with no host operations. NRMSE: Z projection 0.002163, RMSNorm 0.001804,
gated RMSNorm 0.002529 (reference uses the actual BF16 Z and readout inputs).
Outputs 14/15/16 are Z, normalized readout, and gated output respectively.
Runtime execution is 315518 cycles, versus 316516 before native gate fusion.
The full diagnostic trace is 655424 cycles / 1.310848 ms at 500 MHz and
51.2 GB/s nominal DDR bandwidth, including initialization and 17 exports.
RMSNorm ingress transpose starts at runtime cycle 432206; normalization
ALU issues span 432255-432533, output transpose starts at 432583, and the
gate's eight ALUs issue continuously over `[432951,432983)`.

### Direct normalization output to wide SXM

The shared planar L2Norm/RMSNorm emitter now captures the normalized VXM
output directly from the forward SR fabric. It suppresses the normalized
scratch writes and removes the matching MEM-to-SXM reads. Capture begins
at the first normalized vector plus the target VXM-to-SXM transport latency
(15 cycles for lpu32). Four 32-vector tiles launch at a 32-cycle interval
for a 128-feature head, overlapping normalization and output transpose.
The source streams follow the existing opposite-fabric ownership: 10/11
on East and 2/3 on West. SXM writes the same compact output layout as before.

The ingress write-tap and original-input replay remain: the normalization
factor depends on the full head reduction. Gamma staging is unchanged.
The decode regression verifies direct VXM stream capture, absence of the
normalized-result spill/reload, and real-weight numerical results for both
Q/K L2Norm and gated output RMSNorm.

Validation after this change: all 17 output binaries are bit-identical to
the previous run. The graph removes 96 MEM read domains and 96 write
domains (96 KiB total physical byte-plane traffic). Runtime execution drops
from 315518 to 315060 cycles; the diagnostic trace including initialization
and exports drops from 655424 to 654958 cycles. RMSNorm normalization ALU
issues now span `[432095,432223)` while output transpose tiles launch at
432117, 432149, 432181, and 432213. Results remain in the same batch1
directory; `norm_direct_output_comparison.json` records the before/after
counts and output hashes comparison.

### Output projection after gated RMSNorm

`--with-output-projection` implies all preceding stages, including
`--with-output-norm`. It appends `out_proj` to the same logical graph:
the gated `[16,128]` result is viewed as `[1,2048]`, multiplied by the real
checkpoint's INT8 `[2048,1024]` weights and scale, and produces BF16
`[1,1024]`. Output 17 (`actual.17.bin`) is this attention output.

The test only supplies the mathematical matmul and tensor view. The shared
C++ projection planner, memory planner, WS lowering, raw macro encoder and
runtime weight pager execute the new node. There is no host activation
calculation or model-specific CModel operation. Input RMSNorm, residual add
and FFN remain outside this incremental test.

The current planner uses four passive-SR activation transports (8 KiB of
physical byte-plane traffic) to assemble the head-partitioned gated result
for both MXM hemispheres. No new SXM operation is introduced. One additional
2 MiB weight page is fetched through DDR/C2C synchronization.

Run in the existing result directory:

```powershell
python -X utf8 compiler/tests/qwen3_5_decode_qkv_wide_sxm_test.py --with-output-projection --log-softplus --package results/qwen3_5_0_8b_decoder_layer_prefill/package --build-dir build-ftlpu-vs2026-direct --target-config ../FTLPU-CMODEL/config/ftlpu-lpu32.json --output-dir results/qwen3_5_0_8b_decode_qkv_conv_qk_norm_wide_sxm/batch1
```

Validation: all 18 outputs pass; the preceding 17 binaries are bit-identical
to the run without output projection. Projection error against the actual
gated input and imported weights is NRMSE 0.002383, cosine 0.99999791.
Runtime execution is 385402 cycles, with eight epochs and seven weight-page
prefetches. The complete diagnostic trace is 748195 cycles / 1.496390 ms
at 500 MHz and 51.2 GB/s nominal DDR bandwidth, including initialization and
18 output exports; this is not an end-to-end decoder-layer latency.

### Batch32 wide-SXM staged decode

The batched harness supports `--sxm-architecture wide --raw-input`. It runs
one decode step for each independent request, using the same real INT8
checkpoint projections, wide-only SXM, WS MXM and raw macro encoding:

1. Batched QKV, Z and beta/g projections, including `exp(g)`.
2. Independent bounded Conv/SiLU, Q/K L2Norm, state decay/KS, correction,
   rank-one update, Q readout and gated RMSNorm invocations.
3. A single batched `[B,2048] x [2048,1024]` output projection.

State uses the native `[head,value,key]` DDR ABI and is swapped in and out
for every request, together with its Conv cache. Projection weights are
not re-executed separately for every request. The staged harness only
serializes DDR tensor rows between executable phases; C++ performs all
physical layout planning, lowering and activation computation.

This is a staged baseline, not one continuous batch-wide ICU program.
The current 512-record QCTL pool cannot hold the projection graph's full
control plan. The harness explicitly selects ordinary common SYNC with
serialized C2C transfers; it does not insert a static transfer-time budget.
Do not compare its latency to an overlapping-prefetch implementation as
though the synchronization policy were identical. The combined pipeline
concatenates the three measured traces sequentially and labels their
phase; host orchestration wall time is excluded.

External gate inputs require additional persistent SRAM bindings, so the
state planner uses eight heads per tile. Conv ingress retains whole-head
ownership across hemispheres. External Z currently uses generic gated
output staging; the continuous batch1 graph's producer-to-producer SiLU
fusion remains available when both inputs have native word descriptors.
Input RMSNorm, residual add and FFN are outside this test.

The projection/gate phase enables FP32 vector ingress. Truncating the
FP32 gate edges to BF16 before `exp(g)` amplified error in strongly decaying
heads (about 6% decay error in the first trial), which the per-head state
checks caught despite a small aggregate error. FP32 ingress preserves
these edges; the numerical limits are unchanged.

```powershell
python -X utf8 compiler/tests/qwen3_5_decode_gdn_batched_test.py --batches 32 --sxm-architecture wide --raw-input --package results/qwen3_5_0_8b_decoder_layer_prefill/package --build-dir build-ftlpu-vs2026-direct --target-config ../FTLPU-CMODEL/config/ftlpu-lpu32.json --output-dir results/qwen3_5_0_8b_decode_qkv_conv_qk_norm_wide_sxm
```

Results are written to `batch32/step0/` below that directory. `summary.json`
checks every request's attention output, updated state, Conv cache and
individual state heads. `runtime.pipeline.csv` is the complete staged
runtime trace; each phase also retains its actual executed program images.

The batch32 numerical run passes all 32 output, state and Conv-cache checks,
including 512 individual state-head checks. Worst output NRMSE is 0.0128171
(minimum cosine 0.99996597); worst individual state-head NRMSE is 0.0142080.
The original aggregate trace time was invalidated: output DMA accounting
added absolute C2C clocks after `clear_completed_program()` began preserving
the global clock. Accounting now uses elapsed cycles for each download
batch, with a planar-I/O regression checking that reported I/O time does
not exceed elapsed system time. Timing replay records binary output hashes
in `timing_replay.json` so the timing correction is checked independently
of numerical tolerances.

The corrected replay passes with all 101 output binaries bit-identical.
At 500 MHz and nominal 51.2 GB/s DDR, the concatenated measured trace spans
5,112,085 cycles (10.224170 ms): projections `[0,327836)`, the 32 state/mixer
requests `[327836,5035768)`, and output projection `[5035768,5112085)`.
These intervals include C2C initialization and tensor exports. They describe
the serialized staged baseline above, not a complete decoder layer or a
fully overlapped batch32 implementation.
