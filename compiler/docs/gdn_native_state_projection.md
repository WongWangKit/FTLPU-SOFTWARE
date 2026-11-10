# GDN native State products and multi-token projection

This optimization continues the projection/forward-solve overlap policy. It
changes generic C++ BatchMatmul preparation and linear-projection scheduling;
it does not change GDN equations, hardware capabilities, or reference tolerances.

## Delivery scope and precision

The real-weight tests cover Qwen3.5-0.8B layer-0 **GDN attention**, batch 1,
seq32 and seq64, chunk size 32, wide SXM, 500 MHz and 51.2 GB/s DDR. The inputs
are the existing BF16 fixtures and zero initial State. Projection weights are
the existing real-weight int8 package with its dequantization scales. Outputs
include attention, core attention and outgoing State; seq32 additionally exports
intermediates for diagnostics. Seq64 carries State and convolution history
between chunks on device. Input RMSNorm, residual and FFN are outside this scope.

State storage and decay/add arithmetic remain FP32. State operands explicitly
viewed as BF16 still read the high two byte planes; this optimization neither
introduces that approximation nor adds another cast. Matrix products retain
BF16 operands, FP32 accumulation of each 32-element K tile, and the existing
FP32 partial-sum addition order. Forward solve stays FP32. The focused State
test includes a rounding sentinel that distinguishes this order from combining
the whole K dimension into a single accumulator pass.

## Compiler changes

Generic BF16 BatchMatmul now uses the existing native BF16 preparation path.
Compatible views are aliases; otherwise the compiler constructs the required
MEM layout without first running a multiply-by-one conversion to FP32.
An LHS tile cache spans N64 column blocks, avoiding repeated preparation of
the same source tile. The shared native matrix emitter also streams each
later MXM partial product into VXM to add the prior FP32 sum directly, avoiding
a separate partial-product spill/add stage. Necessary operand layout joins
remain; this is not a claim that all MEM copies have disappeared.

Multi-token linear projections use the continuous schedule described in
[linear_projection_multitile.md](linear_projection_multitile.md). The existing
two weight buffers alternate while all M rows consume the current tile. Each
N group has a complete `[row32, token tile, K tile]` ICU Compute instruction;
instructions execute sequentially on each queue. The branch requires target
support for weight/activation overlap. M32 and unsupported shapes retain their
existing paths. No extra SRAM, stream, ICU context, or host computation is used.

## Reproduction

From the SOFTWARE repository root, after building `ftlpu_compile` and the
existing runtime targets:

```powershell
python compiler/tests/activation_state_matmul_test.py --build-dir build-ftlpu-vs2026-direct --target-config ../FTLPU-CMODEL/config/ftlpu-lpu32.json --output-dir results/activation_state_matmul
python compiler/tests/linear_projection_multitile_runtime_test.py --build-dir build-ftlpu-vs2026-direct --target-config ../FTLPU-CMODEL/config/ftlpu-lpu32.json --output-dir results/linear_projection_multitile

python compiler/tests/qwen3_5_prefill_qkv_wide_sxm_test.py --package results/qwen3_5_0_8b_decoder_layer_prefill/package --build-dir build-ftlpu-vs2026-direct --target-config ../FTLPU-CMODEL/config/ftlpu-lpu32.json --output-dir results/qwen3_5_0_8b_prefill_chunkwise_attention --sxm-architecture wide --with-output-projection --o-mxm-precision bf16 --solve-strategy forward --projection-solve-overlap on
python compiler/tests/qwen3_5_prefill_multichunk_test.py --package results/qwen3_5_0_8b_decoder_layer_prefill/package --build-dir build-ftlpu-vs2026-direct --target-config ../FTLPU-CMODEL/config/ftlpu-lpu32.json --output-dir results/qwen3_5_0_8b_prefill_chunkwise_attention_seq64 --seq-len 64 --projection-solve-overlap on

python compiler/tools/summarize_projection_solve_overlap.py --result-dir results/qwen3_5_0_8b_prefill_chunkwise_attention --baseline baseline_native_state_projection.json --output-name native_state_projection_comparison.json --allow-unchanged-time
python compiler/tools/summarize_projection_solve_overlap.py --result-dir results/qwen3_5_0_8b_prefill_chunkwise_attention_seq64 --baseline baseline_native_state_projection.json --output-name native_state_projection_comparison.json
```

Both tests update their fixed directories. `runtime.pipeline.csv` records actual
raw-macro execution. `runtime_icu_programs/*.icu.csv` contains the executed
per-ICU programs and pre-execution uploads; `program.json` records paging
protocol metadata. The comparison uses the
preserved immediately preceding overlap-enabled baseline, not the older
serialized baseline. It checks all exported output hashes, weight traffic,
runtime overlap, and timing from initial ingress through compute completion.
Diagnostic HostOutput is reported separately. These tests require zero host
model operations and retain their existing numerical thresholds.
The unaffected seq32 case uses the explicit no-regression timing policy;
seq64 must improve cold time. Both policies require bit-identical outputs.

## Verified results (2026-11-06)

All 18 seq32 and all 6 seq64 exported tensors are bit-identical to the saved
overlap-enabled baseline, including outgoing State. Both real-weight numerical
tests passed without changing their reference thresholds. Seq64 chunk1
attention NRMSE remains 0.999791%, State NRMSE 0.894269%, and maximum per-head
State NRMSE 1.816759%. Token 32's incoming-State boundary check also passed.

| Timing (cycles) | seq32 before | seq32 after | seq64 before | seq64 after |
| --- | ---: | ---: | ---: | ---: |
| Initial input prefix | 4,682 | 4,682 | 14,529 | 14,529 |
| Logical execution | 244,011 | 244,011 | 1,177,116 | 821,707 |
| Runtime external wait | 45,184 | 45,184 | 45,498 | 45,512 |
| Cold through computation | 293,877 | 293,877 | 1,237,143 | 881,748 |
| Diagnostic HostOutput | 154,001 | 154,001 | 128,240 | 136,432 |
| Full diagnostic trace | 447,878 | 447,878 | 1,365,383 | 1,018,180 |

At 500 MHz, seq64 time through computation, including initial ingress and
runtime external waits, is **2.474286 → 1.763496 ms**, a **28.73% reduction**.
Seq32 remains **0.587754 ms**: its M32 projection and zero initial State do not
exercise the newly optimized paths. Dynamic weight payloads remain 10,551,296
bytes / 6 pages for seq32 and 14,811,136 bytes / 9 pages for seq64. Both runs
have zero host model operations.

The diagnostic readback is explicitly outside the above cold-compute metric:
seq64 readback increased by 8,192 cycles. Including that readback, the complete
trace still decreases from **2.730766 to 2.036360 ms**. The saved baseline did
not retain per-output HostOutput events, so it cannot establish the cause of
the readback difference on its own.

The focused native State tests cover H=2/M32, H=3/M16 and H=4/M32 with K=N128.
Their FP32 State and matrix products match the strict 32-K reference bit for
bit, including the rounding sentinel. The projection suite passes M32/64/96/128
and odd-K fallback cases. A separate disabled-overlap target regression emits
no simultaneous Load/Compute on either MXM and preserves the enabled case's
output bytes. Runtime traces and executed ICU files accompany those tests.

For seq64, each of the three QKV pages now occupies 65,588 logical cycles.
Both MXMs execute 65,536 consecutive Compute cycles per page, with no internal
gap; the remaining 52 cycles are the stage's startup/drain envelope. Paging
barriers between pages are still real and remain in the runtime trace.

The incoming-State QS and KS stages, including layout preparation, now take
89,648 and 89,692 cycles respectively. Each still has 64 matrix tile nodes and
96 layout joins, with 896 explicit transport domains / 21,504 logical vector
copies. MXM arithmetic occupies only 4,096 cycles per side in each stage.
Those tile preparation costs are the next bottleneck; this iteration does
not claim continuous MXM computation for the whole old-State product.

## Direct Q/K tile reads and shared State tiles (2026-11-07)

For a pair of complete 32-token tiles, the compiler can use the L2Norm
producer's distributed16 MEM addresses directly. It checks the physical
storage shape, both head offsets, all lane indices, the producer's lifetime
and MEM queue conflicts. The eight slice pairs feed tokens `j, j+8, j+16,
j+24` into the two MXM activation streams at the issue times implied by each
route's latency. The second output-column block reads the same tile again.
The head on each side follows the placement of the FP32 partial sum.

The incoming Q and K contractions also read the same BF16 high-plane view of
State. The memory planner can keep its prepared RHS tiles live between the
two consumers after verifying producer, output, element indices, shape and
orientation. Seq64 retains 64 tiles of `64 × 64` BF16 elements mirrored on
both hemispheres, a total of 1 MiB. SRAM capacity and all live ranges go
through the existing allocator. If retention prevents allocation, the
compiler plans the two preparations with separate lifetimes.

The focused `distributed_state_matmul_test.py` checks the device-produced
normalized operands and both FP32 products bit for bit. It also checks that
each distributed LHS uses its original home and that the RHS is shared once.
The separate State test permutes the heads in a second view of the same
producer and verifies the two distinct products.

Run the full-attention execution commands above to update the fixed result
directories. Reproduce the focused test and compare both runs with their
immediately preceding baselines using the following commands.

```powershell
python compiler/tests/distributed_state_matmul_test.py --build-dir build-ftlpu-vs2026-direct --target-config ../FTLPU-CMODEL/config/ftlpu-lpu32.json --output-dir results/distributed_state_matmul
python compiler/tools/summarize_projection_solve_overlap.py --result-dir results/qwen3_5_0_8b_prefill_chunkwise_attention --baseline baseline_state_tile_reuse.json --output-name state_tile_reuse_comparison.json --allow-unchanged-time
python compiler/tools/summarize_projection_solve_overlap.py --result-dir results/qwen3_5_0_8b_prefill_chunkwise_attention_seq64 --baseline baseline_state_tile_reuse.json --output-name state_tile_reuse_comparison.json
```

The seq64 report includes per-head preparation counts, checks of the emitted
source-address domains, and runtime measurements.

| Runtime cycles at 500 MHz and 51.2 GB/s | seq32 before / after | seq64 before | seq64 after |
| --- | ---: | ---: | ---: |
| Initial ingress | 4,682 | 14,529 | 14,529 |
| Logical device execution | 244,011 | 821,707 | 729,737 |
| Exposed external wait | 45,184 | 45,512 | 45,512 |
| Cold through compute completion | 293,877 | 881,748 | 789,778 |
| Diagnostic readback | 154,001 | 136,432 | 128,240 |
| Full trace including readback | 447,878 | 1,018,180 | 918,018 |

Cold seq64 attention latency fell by 10.43%, from 1.763496 to 1.579556 ms.
An intermediate run with only RHS retention took 857,619 cycles. Seq64
fetched the same 14,811,136 weight bytes over nine pages. The 18 seq32
and six seq64 output files matched their respective baselines byte for byte.
Chunk 1 attention and State NRMSE versus the independent reference were
0.999791% and 0.894269%, respectively. Both runs used zero host model operations.

| Chunk 1 stage including preparation | Runtime interval [start, end) | Cycles before → after | Explicit vector copies before → after |
| --- | --- | ---: | ---: |
| Q × incoming State | [429910, 486120) | 89,648 → 56,210 | 21,504 → 16,384 |
| K × incoming State | [542468, 569984) | 89,692 → 27,516 | 21,504 → 0 |

The remaining 16,384 vector copies prepare the shared State operand once.
Both stages still execute 64 matrix tiles with 4,096 MXM Compute cycles per
hemisphere. Their stage windows also include the tile loads, SXM transposes,
FP32 partial-sum accumulation and scheduling gaps.
