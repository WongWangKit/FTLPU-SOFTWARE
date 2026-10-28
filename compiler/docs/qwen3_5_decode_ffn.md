# Qwen3.5 decode FFN validation

The test uses the existing Qwen3.5-0.8B layer-0 checkpoint package (hidden
1024, intermediate 3584), quantized to per-tensor W8A16. Inputs are explicitly
synthetic BF16 RMS-normalized vectors, not captured decoder activations.

Each request contributes one token: `[B,1,1024]` is flattened to `[B,1024]`
because FFN has no cross-token dependency. Batch 1 represents a single-request
edge workload; batch 32 and 64 exercise batched serving on the same LPU and
model. These are not measurements of a larger cloud model or a full decoder.

The graph contains Gate/Up projections, SiLU, elementwise multiplication and
Down projection. Python provides semantic equations, weights and reference
checks only. Shared C++ dense FFN lowering plans SRAM and DDR/C2C weight pages
and emits the ICU programs; CModel executes all arithmetic without host
intermediate operations.

Run from the FTLPU-SOFTWARE root after building `ftlpu_compile`,
`ftlpu_graph_compile`, `compiled_stage_file_runner` and
`ftlpu_icu_program_export`:

```powershell
python compiler/tests/qwen3_5_decode_ffn_test.py --package results/qwen3_5_0_8b_decoder_layer_prefill/package --build-dir build-ftlpu-vs2026-direct --target-config ../FTLPU-CMODEL/config/ftlpu-lpu32.json --output-dir results/qwen3_5_0_8b_decode_ffn
```

Use `--check-only` to check completed outputs and regenerate summaries. The
current supported batch sizes are 1 or multiples of 32. This test selects the
vector MXM path for every batch, including a genuine one-row input for B=1.
Native4 was not used: its graph output-placement check currently rejects this
standalone logical FFN. Arbitrary sub-32 batches require separate lowering work.

Validation compares against FP32 NumPy FFN equations with the same quantized
weights. Fixed limits are aggregate NRMSE < 3% and each request NRMSE < 4%.
Shared request prefixes must also have identical BF16 outputs across batches.

## Results, 2026-10-27

LPU32, 500 MHz, DDR peak 51.2 GB/s, configured scheduling efficiency 90%.

| Batch | NRMSE | Worst request NRMSE | Trace cycles | FFN latency ms | FFN output tokens/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.720% | 0.720% | 218585 | 0.437170 | 2287 |
| 32 | 0.745% | 0.805% | 235707 | 0.471414 | 67881 |
| 64 | 0.739% | 0.805% | 430190 | 0.860380 | 74386 |

Latency runs from runtime cycle zero through the last traced event, including
initial ingress, dynamic waits and output transport. Repeated trace domains
include both inner and outer intervals. It excludes host compilation/loading
time and final idle drain cycles; it is not the runner's static `cycles` field,
full-model token latency or full-model throughput.

All three cases transfer 11,010,048 weight bytes over 10 page prefetches, use
one native FFN region and three execution epochs, and report zero host operations.
The first request is bit-identical between B=1/B=32; all first 32 requests are
bit-identical between B=32/B=64. The small fused/tail chain VXM FFN numerical
regression also passes after the paging fix.

## Event-driven transport correction

Initial C2C ingress now packs the four-slice native MXM activation layout
directly, including both activation port copies and hemispheres. There is no
initial MEM layout conversion. Gate is preloaded; Up is marked for runtime
prefetch into its other bank while Gate executes. Down reuses Gate's bank only
after its weight readers retire.

The logical graph does not reserve an FFN transfer-duration delay. After a
MEM_WRITE_SYNC, a queue with no intervening FU work goes directly to its page
rendezvous instead of retaining nominal idle time. The controller still waits
for page commitment and the participating activation/compute queues, preserving
their producer/consumer phases. Necessary computation delays are retained.

For B=32, the old 56,888-cycle NOP after Down's WRITE_SYNC chain is gone.
Gate compute starts at cycle 43,179 while Up ingress continues. Gate/Up compute
ends at 157,998, SwiGLU runs at 158,022–161,134, and Down compute starts at
164,880 after hidden writes/mirroring and synchronization. The large repeated
transport wait no longer follows this boundary.

Before correction, B=1/32/64 latencies were 0.604876/0.733976/1.221874 ms;
`before_sync_fix.csv` preserves that comparison. The corrected 51.2 GB/s cases
report zero additional runtime page delay, but SYNC waits can still be visible
while another participant computes. A 25.6 GB/s B=1 run also passes with the
same numerical error: 0.574724 ms and 32,132 actual runtime wait cycles. This
confirms that slower transport is handled dynamically. Its diagnostic results
are under `tmp/ffn_sync_slow_ddr/`.

Regression checks cover direct C2C packing versus SRAM upload for 1/32/64
rows, single/grouped page rendezvous without a post-WRITE_SYNC NOP, fused/tail
SwiGLU numerical execution, absence of initial MEM copies, and bit-identical
request prefixes across batches.

Results are under `results/qwen3_5_0_8b_decode_ffn/batch{B}/`:

- `runtime.pipeline.csv`: actual runtime activity, including C2C.
- `runtime_icu_programs/`: exported linked runtime ICU queues plus pre-execution programs.
- `actual.runtime.ftlpu`: runtime-linked binary.
- `graph_plan.json`, `ffn.kernel.mlir`, `ffn.ftlpu`: compiler artifacts.
- `actual.0.bin`, `golden.f32.bin`, `summary.json`: numerical outputs and metrics.

Open the pipeline viewer with
`?csv=/results/qwen3_5_0_8b_decode_ffn/batch32/runtime.pipeline.csv` (replace
32 with 1 or 64). The root `summary.csv` and `batch_consistency.json` summarize
the sweep.
