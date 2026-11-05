# Seq32 chunkwise GDN attention tail

The incremental prefill test supports `--with-output-norm` and
`--with-output-projection`. The latter includes the zero-initial-State chunk
core, outgoing State, Z projection, per-head gated RMSNorm and output projection.
It does not include the decoder input RMSNorm, residual additions or FFN.

```powershell
python compiler/tests/qwen3_5_prefill_qkv_wide_sxm_test.py `
  --package results/qwen3_5_0_8b_decoder_layer_prefill/package `
  --build-dir build-ftlpu-vs2026-direct `
  --target-config ../FTLPU-CMODEL/config/ftlpu-lpu32.json `
  --output-dir results/qwen3_5_0_8b_prefill_chunkwise_attention `
  --sxm-architecture wide --with-output-projection `
  --o-mxm-precision bf16 --solve-strategy forward
```

For each token and value head, the tail computes

```
Z = X W_z
N = O / sqrt(mean(O * O) + epsilon) * delta_norm
Y = N * SiLU(Z)
attention = reshape(Y, [32, value_width]) W_o
```

`delta_norm` is used directly, without the `1 + weight` convention of the
decoder's other RMSNorms. The head dimension is 128. O's FP32 upper byte planes
are read as truncated BF16; norm and gated results are BF16, with FP32 VXM
chain arithmetic. MXM projection uses the common quantized-weight lowering.

The Python test only specifies tensor views and logical operators. C++ splits
retained per-head normalization into 32-row tiles and constructs native source
descriptors for the existing shared RMSNorm emitter. It reads the grouped-head
O layout directly. The normalized VXM output streams into wide SXM before its
final MEM write. The normalization reduction still requires its input replay.

The shared SiLU-multiply emitter accepts piecewise affine operand addresses.
Its MEM domains feed a continuous depth-8 VXM chain, avoiding an intermediate
join of the normalization tiles. Consecutive segments of one operand may share
a MEM queue; the simultaneous other operand and output must use separate queues.

Remaining transport is explicitly recorded in `summary.json`: the broadcast
normalization weight is staged once into the native workspace and reused by
the following 15 tiles; the output projection gathers/mirrors its activation
input. Validation exports are reported separately from execution transports.

The linker only reuses a weight when its immutable binding and execution home
match, and no intervening emitted MEM write touches that home's queues.
Leaving a consecutive native RMSNorm sequence invalidates the cached home.
This preserves the SRAM lifetime and supports different weights without
assuming that every normalization uses the same constant. Reuse does not
remove the actual weight reads performed by each normalization.

For these consecutive tiles the linker also omits its generic 96-cycle gap
and 64-cycle launch allowance. The previous emitter's end (including SR
drain) remains the earliest next launch. Input replay for the reduction and
the final norm output remain present.

The forward-solve test exports 18 tensors: Q, K, V, beta, g, G, gamma, R, L,
B, D, H, O, outgoing State, Z, normalized O, gated O and attention. The tail is
checked both against its actual device inputs and against the independent
sequential GDN reference. All execution is through raw-macro ICU programs;
runtime traces and per-ICU exports live beside the summary.

The gated output's direct diagnostic ABI is
`[head_parity, token, head_pair, value_feature]`; the test unpacks that order
only for host comparison. The compiler's downstream consumers already use
the explicit partition row descriptors.

## Validation (2026-11-03)

The real-weight seq32 run passed on CModel with wide SXM, raw-macro ICU
instructions, 7 dynamic weight pages (10,616,832 physical bytes), 18 output
bindings and zero host operations. Local NRMSE is 0.1768% for RMSNorm,
0.1605% for the gate and 0.2262% for output projection. Final attention NRMSE
against the independent sequential recurrence and tail is **0.8702%**, with
cosine similarity **0.9999783**.

`runtime.pipeline.csv` includes initial input and diagnostic output traffic.
Its final timestamp is not an attention-only latency. The summary reports
per-stage runtime ranges using the relevant page barrier's offset rather than
applying the last epoch's offset to earlier operators.

### Retained norm weight and tighter tile launches

The same real-weight run passes after enabling consecutive-tile weight reuse.
All 18 exported tensors are byte-for-byte identical to the prior run.

| Metric | Before | After |
| --- | ---: | ---: |
| Norm weight materializations | 16 | 1 |
| Norm weight transport domains | 128 | 8 |
| First norm FU issue to last norm write (cycles) | 18,655 | 7,615 |
| Norm including initial weight preparation (cycles) | 19,317 | 8,277 |
| Complete test logical cycles | 309,801 | 298,761 |

The new norm FU runtime range is `[336857, 344472)`; its initial weight
preparation starts at cycle 336195. Input reduction replay and final result
writes are unchanged. The runtime program still has 7 weight pages and zero
host operations. The test asserts the single initial preparation, 15 reused
weights, tightened launches, and the existing end-to-end numerical criteria.


### Bank ping-pong and early C2C (2026-11-04)

The full attention fixture enables `ftlpu.weight_page_ping_pong` and
`ftlpu.common_transport_overlap`. The C++ graph planner reserves two 2 MiB
weight slots across both hemispheres: slices 20--27, rows 4096--8191, in
bank 0 and bank 1. It changes tensor placements before native projection
lowering, so read issue times follow the actual SR distances. The retained
activation allocator excludes those address ranges and prefers independent
queues for small planar values. It can use other rows on the same slice only
when the final MEM-port/SR lease permits the transfer.

The page sequence is QKV0 A, QKV1 B, QKV2 A, beta-branch projection B,
g-branch projection A, Z B, out_proj A. The seven transfers still total
10,616,832 bytes. Large pages are 2 MiB; the two scalar-head projections are
64 KiB each. The forward solve's coefficient ping-pong scratch moves from
slices 24--27 to 28--31, freeing the weight queues during the long solve.

The runtime linker checks the final raw-macro image's MEM ports and SR
routes, including fixed VXM FP16/BF16/FP32 inputs, output byte widths, and
repeated vector trains. A vector train crossing the proposed consumer stop
is rejected. DMA/RX FIFO order follows consumer order: sorting solely by an
early launch opportunity can create a dependency on a future frozen epoch.
Actual MEM commits and transport drain release the consumer; transfer-time
estimates never substitute for completion. Only the hardware drain allowance
remains at the compiler ingress boundary, not a page-sized static NOP window.

Prefetch trace timestamps are converted from the global physical clock to
the trace segment's local clock exactly once. `Paging.Consumer` records the
actual consumer issue cycle. Page-ready time is no longer used as a proxy
for consumer time when calculating stage ranges.

Measured with the real seq32 weights, 500 MHz / 51.2 GB/s, raw macro,
wide SXM, forward solve, BF16 O MXM inputs:

| Metric | Before | After |
| --- | ---: | ---: |
| Test logical cycles | 298,761 | 278,466 |
| Runtime page wait cycles | 114,675 | 22,871 |
| Invocation physical cycles, including diagnostic exports | 413,436 | 301,337 |
| Body MEM transport domains | 12 | 12 |
| Dynamic page bytes | 10,616,832 | 10,616,832 |

The remaining waits are 22,567 cold-start cycles plus 304 cycles for beta.
QKV1/QKV2, g, Z, and out_proj are ready before their consumers. Z transfers
in physical cycles [175337, 197960), with completion observed at 197961;
its consumer starts at 217313. Out_proj transfers in [197961, 220596), with
completion observed at 220597; its consumer starts at 264330. The out_proj
prefetch starts while the GDN core is still executing and completes during Z.

All 18 exported tensors remain byte-identical to the pre-paging baseline.
End-to-end attention NRMSE is 0.0087018333 and cosine similarity is
0.9999783308. Host operations remain zero. Results and actual runtime ICU
program exports remain in the existing chunkwise-attention results directory.
The complete trace includes initial ingress and diagnostic output traffic;
its last timestamp is not an attention-only TTFT.
