# SmolLM2-135M Attention Pipeline

![SmolLM2-135M attention pipeline](smollm2_attention_pipeline.svg)

The diagram is generated from the serialized ICU queues executed by
`compiled_smollm2_attention_runtime_test`. It is not a manually
maintained timing sketch. The runtime expands `NOP` and `Repeat` commands into
the same four-column trace format used by FTLPU-CMODEL:

```text
start,end,resource,detail
```

The current `seq_len=128`, `hidden_size=576`, `9` query-head, `3` KV-head
workload covers:

- Q/K/V W8A16 projection and Q/K RoPE;
- one logical MXM per hemisphere for QK work waves;
- recurrent three-pass softmax with causal masking fused into pass 1;
- SXM probability transpose and permutation;
- P x V context calculation;
- one logical MXM per hemisphere for output projection.

The runtime test checks that Command IR contains MEM, MXM, VXM, and SXM queue
commands. It then executes the binary on CModel and compares sampled Q/K/V
values, causal softmax probabilities, P x V context values, and final
O-projection output against an independent CPU Attention reference. The CPU
reference is computed only from the uploaded input and weights; it does not
reuse CModel intermediate values. Every sampled future-token probability must
also be exactly zero.

The causal mask uses only 31 reusable FP32 vectors for the nonzero diagonals of
one 32x32 tile. Fully visible past blocks use an immediate `0`; fully hidden
future blocks use an immediate `-1e9`. Only the current diagonal block reads a
mask vector. Softmax pass 1 executes `scale -> mask add -> recurrent max`, so
passes 2 and 3 consume already-masked scores without another mask read.

## Detail Windows

The renderer uses the same functional-unit lane layout and color convention as
FTLPU-CMODEL's `smollm2_attention_schedule_detail.svg`. It selects nine
readable windows from the full 38,215-cycle trace:

1. Q projection first reduction block;
2. Q RoPE draining its MEM FIFO while the next MXM projection runs;
3. V projection packed 16-stream write;
4. a steady-state two-hemisphere QK wave;
5. the three pipelined softmax passes;
6. softmax pass 3 writing the packed probability layout directly;
7. P x V SXM transport and MXM work;
8. O-projection input staging through local MEM taps and the passive bridge;
9. the first O-projection reduction wave.

Except for an explicit `--window` override, these windows are discovered from
operation signatures in the current runtime trace. They therefore follow
scheduler cycle changes instead of relying on a fixed CModel cycle table.
The checked-in PNG is a rasterized copy of the SVG for viewers without SVG
support.

The scheduler derives phase boundaries from target transport and queue
latencies. Q/K/V and O projection keep each loaded weight tile resident for
four 32-token activation tiles. Two independent 32-column outputs occupy the
two weight buffers and alternate four-cycle Block8 issues. Activation uses
streams 16..31 while eight INT8 streams feed the MXM-local dequantizer and IW.
The next reduction uses wavefront IW: column 0 is replaced only after the
preceding compute has consumed column 0, followed by columns 1..3. This removes
the inter-reduction MXM bubbles without overwriting in-flight weights. QK waves
start every 280 cycles while each wave remains live for
301 cycles, overlapping the next wave's IW load with the preceding
accumulator tail. East and west softmax chains use independent VXM ALU banks
and advance concurrently. PV blocks advance from their actual accumulator and
context-write tails, and softmax pass 3 writes its only physical result directly
to the packed distributed16 probability layout. Command translation still
rejects any overlap on the same ICU queue.

The final reduction does not run a separate accumulator-read tail. Its last
Block8 partial selects `accumulator_destination = stream` and
`accumulator_clear = true`; the MXM converts the completed FP32 accumulation
to BF16 while emitting it. Q and K write the MXM's 16 westbound BF16 byte
streams into a dedicated 16-slice MEM FIFO. The two 32-column halves use a
two-slice cyclic offset, so RoPE can read both BF16 operands through distinct
single-read-port slices at one token per cycle. RoPE drains the FIFO on VXM
through west streams 32..43 and writes the rotated values back through the
otherwise idle east streams 20..23 while the next projection continues on
MXM. Query RoPE selects the hemisphere that owns its shared KV head as its
output hemisphere, so no post-projection VXM pass is required. If two query
heads in one projection pair target the same KV hemisphere, their 128-cycle
RoPE drains are serialized only on that destination; different destinations
remain parallel. V writes directly to its packed distributed16 placement.
O-projection input staging no longer uses a VXM pass: PV context lives in high
MEM slices, dead RoPE FIFO slices are reused for the distributed16 activation,
and each westbound stream first performs a local `write_tap`, then crosses the
passive hemisphere bridge and is consumed by the remote eastbound MEM write.
Both logical MXMs execute O-projection Block8 issues in the same cycles. The
final singleton output group duplicates its weights into both buffers and
alternates them across token blocks; the next output pair prefetches while the
previous pair writes its results. Consequently every MXM compute issue from
cycle 35,616 through 38,204 is exactly four cycles after the previous issue.
The O result is already sharded by the Block8 distributed16 layout, so each
MXM writes its BF16 stream directly to its source-local result slice instead
of performing a redundant VXM fanout. Neither path performs the former FP32
accumulator read and cast. The FIFO overlap and direct GQA placement reduce
the Q/K/V+RoPE boundary from 6,703 to 5,525 cycles.

PV no longer clears its context accumulator with a full read sweep before the
first key block, nor drains it with separate accumulator-read commands after the
last key block. Intermediate key-block partials stay in SRAM. The last partial
selects `accumulator_destination = stream`, `accumulator_clear = true`, and
`accumulator_output_format = bf16`; MXM converts the completed FP32 result and
writes its two BF16 byte streams directly to source-local context MEM. There is
no PV VXM cast or fanout. Value weights use east streams 0..15 while probability
activations use east streams 16..17, allowing the next value tile to prefetch
across key blocks and head waves while the current compute remains active. The
direct MEM write starts at the target-derived `mxm_first_result_latency()` and
the O-projection prefetch uses west streams 2..5, so it can overlap the final PV
west streams 0..1 without a stream-register collision.

## Measured CModel Utilization

These values are measured by runtime while executing the compiled full
Attention binary, including 64 drain cycles. The monitor sampled 38,279 cycles;
`program.max_cycle` is 38,215. Concurrent dual-hemisphere O compute, singleton
weight-buffer ping-pong, overlapped pair prefetch, and source-local result
writeback save 2,819 cycles (6.87%) relative to the previous 41,034-cycle
schedule. The complete optimization series saves 14,466 cycles (27.46%)
relative to the 52,681-cycle baseline.

| Resource | Active queues | Issued queue-cycles | Issue utilization |
| --- | ---: | ---: | ---: |
| MEM | 104/104 | 450,008 | 11.30% |
| MXM load | 2/4 | 4,176 | 2.73% |
| MXM compute | 2/4 | 36,864 | 24.08% |
| MXM local dequant | 2/4 | 3,600 | 2.35% |
| VXM | 14/16 | 46,080 | 7.52% |
| SXM transpose | 2/2 | 864 | 1.13% |
| SXM permute | 2/2 | 1,086 | 1.42% |

Issue utilization is normalized by the physical CModel ICU queue count and the
complete monitored cycle range. The executable declares one logical MXM per
hemisphere, mapped onto two of the CModel's four physical MXMs. Relative to
those two logical compute queues, MXM compute issue utilization is 48.16%.
These values measure scheduled queue occupancy, not active MAC-cell density.

## Regenerate

Build `ftlpu_opt`, `ftlpu_translate`, and
`compiled_smollm2_attention_runtime_test`, then run:

```powershell
python compiler/tests/smollm2_attention_binary_runtime_test.py `
  --opt build-ftlpu-vs2026/compiler/ftlpu_opt.exe `
  --translate build-ftlpu-vs2026/compiler/ftlpu-translate.exe `
  --runtime-test build-ftlpu-vs2026/runtime/compiled_smollm2_attention_runtime_test.exe `
  --input compiler/examples/smollm2_135m_attention/attention_seq128.stablehlo.mlir `
  --output-dir build-ftlpu-vs2026/compiler/ftlpu_lower/smollm2_attention_pipeline
```

The command produces:

- `attention.command.mlir`
- `attention.ftlpu`
- `attention.runtime.csv`
- `attention.pipeline.svg`

To update the checked-in documentation image from an existing trace:

```powershell
python compiler/tools/render_attention_pipeline.py `
  build-ftlpu-vs2026/compiler/ftlpu_lower/smollm2_attention_pipeline/attention.runtime.csv `
  compiler/docs/smollm2_attention_pipeline.svg
```

Use repeatable `--window START:END:TITLE` arguments to replace automatic
semantic window discovery when investigating a specific cycle range.

## Experimental QK-Softmax Fusion

`--attention-schedule fused` has a complete experimental lowering: the final
QK partial uses `accumulator_destination = stream` with
`accumulator_clear = true`, two VXM scratch banks consume alternating QK
waves, and softmax pass 3 writes the packed probability layout directly.

The current ICU repeat encoder can still merge commands across the QKV and
fused regions in a way that violates passive stream-register staging. The
fused legality check therefore falls back to the verified Tail schedule on
the current target. The fallback binary passes the complete CModel numerical
golden test; the experimental schedule remains isolated for continued encoder
work and does not change Tail memory allocation or command generation.
