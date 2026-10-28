# Qwen3.5-0.8B GDN decode recurrence

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
