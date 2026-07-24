# SmolLM2-135M Attention Pipeline

![SmolLM2-135M attention pipeline](smollm2_attention_pipeline.svg)

The diagram is generated from the serialized ICU queues executed by
`compiled_smollm2_attention_softmax_runtime_test`. It is not a manually
maintained timing sketch. The runtime expands `NOP` and `Repeat` commands into
the same four-column trace format used by FTLPU-CMODEL:

```text
start,end,resource,detail
```

The current `seq_len=128`, `hidden_size=576`, `9` query-head, `3` KV-head
workload covers:

- Q/K/V W8A16 projection and Q/K RoPE;
- four-MXM QK work waves;
- recurrent three-pass softmax with causal masking fused into pass 1;
- SXM probability transpose and permutation;
- P x V context calculation;
- four-MXM output projection.

The runtime test checks sampled Q/K/V values, softmax normalization and SXM
layout, P x V context values, and the final O-projection output against CPU
references. It also checks every sampled future-token probability is exactly
zero.

The causal mask uses only 31 reusable FP32 vectors for the nonzero diagonals of
one 32x32 tile. Fully visible past blocks use an immediate `0`; fully hidden
future blocks use an immediate `-1e9`. Only the current diagonal block reads a
mask vector. Softmax pass 1 executes `scale -> mask add -> recurrent max`, so
passes 2 and 3 consume already-masked scores without another mask read.

## Detail Windows

The renderer uses the same functional-unit lane layout and color convention as
FTLPU-CMODEL's `smollm2_attention_schedule_detail.svg`. It selects eight
readable windows from the full 69,304-cycle trace:

1. Q projection first reduction block;
2. Q RoPE and writeback;
3. V projection packed 16-stream write;
4. a steady-state four-MXM QK wave;
5. the three pipelined softmax passes;
6. post-softmax packed probability layout;
7. P x V SXM transport and MXM work;
8. the first O-projection reduction wave.

Except for an explicit `--window` override, these windows are discovered from
operation signatures in the current runtime trace. They therefore follow
scheduler cycle changes instead of relying on a fixed CModel cycle table.
The checked-in PNG is a rasterized copy of the SVG for viewers without SVG
support.

The scheduler derives phase boundaries from target transport and queue
latencies. Q/K/V and O projection keep each loaded weight tile resident for
four 32-token activation tiles. During the fourth tile, the activation stream
temporarily moves to stream registers 16/17 while the next weight tile is
dequantized and loaded into the alternate MXM weight buffer. This removes the
inter-reduction MXM bubbles without changing accumulator lifetime. QK waves
start every 280 cycles while each wave remains live for
301 cycles, overlapping the next wave's IW load with the preceding
accumulator tail. East and west softmax chains use independent VXM ALU banks
and advance concurrently. PV blocks advance from their actual accumulator and
context-write tails, probability packing starts from the actual softmax tail,
and the two O-projection cast paths execute concurrently. Command translation
still rejects any overlap on the same ICU queue.

The Q/K/V final reduction is intentionally different from the steady
reductions. Its issue interval is 64 cycles: 32 cycles of MXM compute followed
by a 32-cycle direct RoPE/cast and packed-MEM writeback window. The activation
planes occupy slices 32..35, the two MXM accumulators occupy slices 36..43,
and the packed query layout plus RoPE table use the remaining queues. Issuing
the next activation tile after only 32 cycles therefore overlaps a live MEM
queue. Removing this final-reduction gap requires a different physical query
layout or another intermediate buffer, not only a cycle-number change.

## Measured CModel Utilization

These values are measured while executing the compiled full Attention binary,
including 64 runtime drain cycles. The monitor sampled 69,368 cycles;
`program.max_cycle` is 69,304.

| MXM | Active cycles | Array utilization | Active density | Peak active cells |
| --- | ---: | ---: | ---: | ---: |
| MXM0 | 48,210 | 68.64% | 98.77% | 16/16 |
| MXM1 | 27,420 | 38.75% | 98.03% | 16/16 |
| MXM2 | 38,516 | 54.99% | 99.03% | 16/16 |
| MXM3 | 16,678 | 23.62% | 98.24% | 16/16 |
| Four-MXM mean | - | 46.50% | 98.52% | 16/16 |

The high active density means a scheduled 32-row MXM block fills the array
well while running. The lower full-program utilization and the large
MXM0..MXM3 imbalance instead show phase gaps and uneven head/wave assignment;
MXM3 is the clearest remaining scheduling opportunity.

| Resource | Full-program utilization | Active density | Stall rate | Peak |
| --- | ---: | ---: | ---: | ---: |
| VXM ALUs | 11.07% | 32.92% | 0.00% | 512/512 |

| SR fabric | Link BW | East BW | West BW | Staged-write util. | Active density | Peak link bytes/cycle |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| East hemisphere | 4.80% | 5.59% | 4.00% | 5.53% | 4.92% | 6,272/24,576 |
| West hemisphere | 3.34% | 3.99% | 2.69% | 3.84% | 4.63% | 6,272/24,576 |

VXM utilization is normalized by 512 lane-ALU execution slots per cycle. SR
bandwidth percentages use each hemisphere's modeled SR-fabric link capacity. The zero VXM
stall rate shows that the current gap is scheduling/phase occupancy rather
than lane-level backpressure. CModel does not yet expose capacity-normalized
MEM or SXM utilization counters, so no percentages are estimated for them.

## Regenerate

Build `ftlpu_opt`, `ftlpu_translate`, and
`compiled_smollm2_attention_softmax_runtime_test`, then run:

```powershell
python compiler/tests/smollm2_attention_softmax_binary_runtime_test.py `
  --opt build-ftlpu-vs2026/compiler/ftlpu_opt.exe `
  --translate build-ftlpu-vs2026/compiler/ftlpu-translate.exe `
  --runtime-test build-ftlpu-vs2026/runtime/compiled_smollm2_attention_softmax_runtime_test.exe `
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
