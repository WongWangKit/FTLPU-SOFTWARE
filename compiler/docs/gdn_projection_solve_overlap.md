# GDN projection / triangular-solve overlap

The first incremental FlashAttention-style scheduling change overlaps an
independent linear projection on MXM with the FP32 unit-lower forward solve on
VXM. In Qwen3.5 GDN this is the Z branch and `(I + L) D = B`. It retains the
existing chunkwise mathematics and intermediate precision; it does not replace
GDN with softmax attention or claim complete tile fusion.

## Scope and contract

The real-weight regressions cover Qwen3.5-0.8B layer-0 GDN attention, batch 1,
sequence lengths 32 and 64, chunk size 32, wide SXM, int8 projection weights,
BF16 activations, and the existing FP32 solve/State storage and arithmetic.
The existing BF16 high-plane operands of State MXM products are unchanged.
The seq64 test carries State and Conv1D history between chunks on device.
Inputs are the existing BF16 fixtures and zero initial State. Outputs include
the attention result, the GDN core, and outgoing State, plus diagnostic
intermediates in seq32. Input RMSNorm, residual and FFN are outside this test.

The target configuration, ICU capacities, FU delays and numerical thresholds
are unchanged. Both configurations execute raw-macro programs in CModel;
Python supplies the graph, inputs and reference only.

## Compiler policy

Set `ftlpu.projection_solve_overlap = true` on the logical kernel module.
The two real-weight drivers expose `--projection-solve-overlap on|off`; the
default remains `off`.

Before graph lifetime analysis and SRAM allocation, the C++ planner finds a
native `unit_lower_solve` and a later independent `linear` projection. All
projection input producers must precede the solve. It moves that projection
immediately before the solve and remaps graph references, including exported
views. No model name, test filename, weight values or cycle patches select
this optimization.

This initial policy selects at most one projection per solve. It requires a
single 32-token projection tile, K divisible by 32, N divisible by 64, int8
weights, local MXM dequantization, nonreplicated hemisphere-local results,
one MXM per hemisphere and OneVector weight input. The resource proof rejects
overlap if native projection emission needs VXM/SXM, distributed-input staging
streams, or a MEM queue also used by the solve. Such candidates execute
serially; unrelated or dependent projections remain serial.

Concurrent operands, outputs and workspaces are planned together. External
token-partition operands are assigned their actual C2C homes before native
lowering, so bank relocation cannot silently change the queue proof afterward.
The Z branch reuses the existing compatible two-plane activation upload when
its queues are available; it does not upload an identical input a second time.
The prohibition applies to the whole `(hemisphere, slice, bank)` ICU queue,
not just overlapping SRAM addresses.

| Concurrent traffic | Existing SR assignment |
| --- | --- |
| Projection activation | East 8/9 |
| Projection OneVector weights | East 31 |
| Projection result | West 8/9 |
| Solve FP32 operands | West 0–7 |
| Solve result | East 2–5 |
| Solve SXM coefficient path | East 16/17, West 24/25 |
| Solve passive feedback | West 12–15 to East 12–15 |

Projection remains a continuous long ICU program. Existing runtime support
reloads only the drained VXM/SXM iMEM queues while MEM/MXM keep running. No
extra hardware context or interleaving of two instructions in one ICU queue
is introduced. Compiler-generated paging barriers remain in effect, including
any external-memory wait exposed when the Z consumer moves earlier.

## Reproduction

Run from `FTLPU-SOFTWARE` with the existing build and package:

```powershell
python compiler/tests/projection_solve_overlap_test.py --build-dir build-ftlpu-vs2026-direct --target-config ../FTLPU-CMODEL/config/ftlpu-lpu32.json --output-dir results/projection_solve_overlap

python compiler/tests/qwen3_5_prefill_qkv_wide_sxm_test.py --package results/qwen3_5_0_8b_decoder_layer_prefill/package --build-dir build-ftlpu-vs2026-direct --target-config ../FTLPU-CMODEL/config/ftlpu-lpu32.json --output-dir results/qwen3_5_0_8b_prefill_chunkwise_attention --sxm-architecture wide --with-output-projection --o-mxm-precision bf16 --solve-strategy forward --projection-solve-overlap on

python compiler/tests/qwen3_5_prefill_multichunk_test.py --package results/qwen3_5_0_8b_decoder_layer_prefill/package --build-dir build-ftlpu-vs2026-direct --target-config ../FTLPU-CMODEL/config/ftlpu-lpu32.json --output-dir results/qwen3_5_0_8b_prefill_chunkwise_attention_seq64 --seq-len 64 --projection-solve-overlap on
```

The focused regression runs two shapes with independent and solve-dependent
projections, both with the policy off/on. It checks FP32 solve accuracy,
unchanged output bytes, real MXM/VXM arithmetic overlap, unchanged weight
traffic, and time through computation including initial C2C input. Dependency
counterexamples must retain the same time and have no arithmetic overlap.

Each real-weight directory is updated in place. `runtime.pipeline.csv` is the
actual execution trace; `runtime_icu_programs/` includes executed programs and
pre-execution uploads, not just the static compiler image. `graph_plan.json`
records selected pairs and resource-proof outcomes. The preserved
`baseline_projection_solve_overlap.json` contains the pre-change metrics and
SHA256 of all outputs.

Generate the same-condition comparison after a successful run:

```powershell
python compiler/tools/summarize_projection_solve_overlap.py --result-dir results/qwen3_5_0_8b_prefill_chunkwise_attention
python compiler/tools/summarize_projection_solve_overlap.py --result-dir results/qwen3_5_0_8b_prefill_chunkwise_attention_seq64
```

The report `projection_solve_overlap_comparison.json` includes output hashes,
actual arithmetic overlap, dynamic weight traffic, initial ingress, page
waits, and diagnostic readback. The old snapshots did not retain HostInput
byte counts; that field is explicitly unknown for the baseline. Cold timing
is recoverable from the saved session end/readback fields and is compared
with the actual current `Session.Invocation.end`.

This step retains the R/L/D materializations and later normalization layouts.
It hides independent MXM work behind the VXM solve but does not make the
entire attention region free of idle cycles. Earlier Z consumption can expose
additional DDR/C2C waits; acceptance uses the complete cold execution time,
not only the logical schedule length.

## Verified results (2026-11-05)

At 500 MHz and 51.2 GB/s, with the same real weights, fixtures and thresholds:

| Timing (cycles) | seq32 off | seq32 on | seq64 off | seq64 on |
| --- | ---: | ---: | ---: | ---: |
| Initial input prefix | 4,697 | 4,682 | 14,529 | 14,529 |
| Logical execution | 277,036 | 244,011 | 1,243,184 | 1,177,116 |
| Runtime external wait | 30,364 | 45,184 | 26,954 | 45,498 |
| Cold through computation | 312,097 | 293,877 | 1,284,667 | 1,237,143 |
| Diagnostic HostOutput | 157,105 | 154,001 | 128,240 | 128,240 |
| Full diagnostic trace | 469,202 | 447,878 | 1,412,907 | 1,365,383 |

Cold time is **0.624194 → 0.587754 ms** for seq32 (**5.84% less**) and
**2.569334 → 2.474286 ms** for seq64 (**3.70% less**). Earlier page consumers
increase external wait, so the complete gain is smaller than the logical
schedule reduction.

All 18 seq32 outputs and all 6 seq64 outputs, including outgoing State, are
byte-identical to the saved serialized baseline. Numerical comparisons with
the original reference and thresholds pass. Actual projection MXM / solve
VXM arithmetic overlap is 16,294 cycles in seq32; seq64 has 16,294 and 14,490
cycles in its two chunks (unions across hemispheres/ALU columns, not summed
lane utilization). Dynamic weight traffic remains 10,551,296 and 14,811,136
bytes respectively; host operations remain zero.

The two-shape focused suite passes all eight off/on executions, including
indirect solve-dependent projection counterexamples. The final compiler
reproduces the exact executed real-weight program images, recorded in
`results/projection_solve_overlap/final_program_equivalence.json`.

## QKV projection and causal convolution on consecutive pages

`ftlpu.projection_conv_overlap` independently enables this C++ policy. Its
default is off. The graph planner matches a paged linear projection and
bilateral depthwise Conv1D pairs by their actual operand references. A pair
becomes ready once its whole channel page has been produced for all tokens.
The next projection works on different channels. Keeping all tokens of a
channel together preserves the four-tap history at the seq64 chunk boundary.

The planner assigns overlapping lifetimes before allocating SRAM. The native
OneVector projection uses E10/11 for activations, E31 for weights, and W10/11
for results. The emitted MEM queues and SRs of both kernels are checked before
the compiler overlaps them. Ordinary Conv accesses must also stay clear of
the dynamic weight-page queues. The native bilateral Conv pair issues a shared
VXM program. Ineligible groups serialize, and an allocation failure retries
with the shorter serial lifetimes.

For the Qwen3.5-0.8B tests, projection results alternate between local slice
pairs 32/33 and 34/35 in bank 1. Conv reads those results in place. Its
simultaneous distributed writes use slices 36–51, and its coefficients arrive
directly in slices 2–9. The dynamic projection weight pages use slices 20–27.

For a packed output that already needs a final gather, the SRAM allocator ranks
legal homes by the longest byte-plane route to C2C TX, then by total route
length. It retains live output/source reservations, excludes linker scratch
and avoids destinations that would need another transfer. Direct output views
keep their producer addresses. If earlier TX-preferred homes leave insufficient
space for a later output, the planner retries the complete set of final
outputs with its original placement order before emitting any final gathers.

For a same-condition comparison, use the two real-weight commands above
with `--projection-solve-overlap on` in both runs. Before changing the
Conv switch, capture each fixed directory's verified baseline. Then repeat
the same execution with `--projection-conv-overlap on` and compare.
The fixed result directories already contain the captured baselines.

The captured runs include the previously enabled projection / solve overlap.
They predate both the projection / convolution change and the TX-aware final
output allocator. Switching only `--projection-conv-overlap` in a current
build isolates the scheduling policy while retaining the current allocator.

```powershell
python compiler/tools/summarize_frontend_overlap.py --result-dir results/qwen3_5_0_8b_prefill_chunkwise_attention_seq64 --capture
# Run the seq64 attention test with --projection-conv-overlap on.
python compiler/tools/summarize_frontend_overlap.py --result-dir results/qwen3_5_0_8b_prefill_chunkwise_attention_seq64 --compare
```

The reports count MXM Compute and non-bypass VXM issues, expanding repeats
and taking the union over hemispheres and ALU columns per cycle. The cold
interval ends at compute completion and includes initial ingress and exposed
DDR waits. The report lists diagnostic readback separately and compares the
actual output hashes, page traffic, and the per-FU arithmetic issue counts.

The smaller regression uses a 64-token linear projection with K=256 and
3,072 output channels, four causal taps, and a downstream head norm. It runs
both configurations, checks projection and convolution against their numeric
references and compares all three device outputs byte for byte. It exports
the full projection and convolution matrices for inspection. Cold completion
in this fixture fell from 96,986 to 79,523 cycles. Its diagnostic readback
grew from 93,225 to 173,097 cycles, so the fixture's full trace grew. The
real-attention acceptance check compares both cold completion and full trace
time for the model's own output set.

```powershell
python compiler/tests/projection_conv_overlap_test.py --build-dir build-ftlpu-vs2026-direct --target-config ../FTLPU-CMODEL/config/ftlpu-lpu32.json --output-dir results/projection_conv_overlap
```

### Real-weight attention results (2026-11-07)

Both runs used batch 1, 500 MHz, 51.2 GB/s and 32-token chunks. The table
compares the frozen baseline against page-level overlap plus final output placement.

| Measured cycles | seq32 before | seq32 after | seq64 before | seq64 after |
| --- | ---: | ---: | ---: | ---: |
| Initial ingress | 4,682 | 4,702 | 14,529 | 14,572 |
| Logical execution | 244,011 | 222,745 | 729,737 | 686,655 |
| Exposed DDR page waits | 45,184 | 44,880 | 45,512 | 44,965 |
| Cold through compute completion | 293,877 | 272,327 | 789,778 | 746,192 |
| Diagnostic HostOutput | 154,001 | 154,001 | 128,240 | 152,816 |
| Full runtime trace | 447,878 | 426,328 | 918,018 | 899,008 |
| MXM Compute and VXM arithmetic simultaneously | 20,866 | 30,694 | 44,092 | 64,160 |

In the overlapping QKV/Conv windows, 9,828 of 9,856 seq32 VXM arithmetic
cycles and 20,068 of 20,096 seq64 VXM arithmetic cycles coincide with MXM
Compute. All 18 seq32 exports and all 6 seq64 exports match their baseline
bytes. Individual reference errors, per-unit arithmetic work, and dynamic
weight traffic match. The second chunk's attention and outgoing State have
NRMSE of 0.99979% and 0.89427% against the reference.

Before TX-aware placement, the second chunk's 1 MiB State was gathered to
West bank 1 slices 8–11, bringing seq64 diagnostic readback to 201,968 cycles.
The final build gathers it to West bank 0 slices 32–35 and saves 49,152
diagnostic cycles. This remains 24,576 cycles longer than the original
baseline with State in slices 44–47. The complete run saves 19,010 cycles.

Each fixed result directory contains `runtime.pipeline.csv`, the executed
per-ICU programs in `runtime_icu_programs/`, the numerical `summary.json`,
and the strict `frontend_overlap_comparison.json` with timestamps per group.

### One SR read for two mirrored MEM destinations

The multichunk driver accepts `--passive-sr-multicast on|off` (default off).
For a lane-preserving, affine packed copy to mirrored SRAM, the compiler
checks that every byte plane passes its local destination on the inward leg
and that the source and destination queues are disjoint. A local WRITE_TAP
preserves the stream for the far hemisphere's WRITE. The domain drains at
the far destination. Other copies keep their original routing.

Run the following from FTLPU-SOFTWARE with a single build for both modes.

```powershell
$root = "results/qwen3_5_0_8b_prefill_chunkwise_attention_seq64"
foreach ($mode in @("off", "on")) {
    python compiler/tests/qwen3_5_prefill_multichunk_test.py --package results/qwen3_5_0_8b_decoder_layer_prefill/package --build-dir build-ftlpu-vs2026-direct --target-config ../FTLPU-CMODEL/config/ftlpu-lpu32.json --output-dir "$root/sr_multicast_$mode" --seq-len 64 --projection-solve-overlap on --projection-conv-overlap on --passive-sr-multicast $mode
    if ($LASTEXITCODE) { exit $LASTEXITCODE }
}
python compiler/tools/compare_sr_multicast.py "$root/sr_multicast_off" "$root/sr_multicast_on"
```

The 2026-11-08 seq64 A/B run measured 3,815,040 → 3,706,496 MEM read
issues and 979,584 total MEM writes in each program. The 92 multicast
domains carried 29,696 vectors. Of these, 16,384 belonged to the final
State update. The cold time fell from 737,890 to 704,672 cycles and the
full trace from 890,706 to 857,488 cycles. The 33,218 saved cycles were
cycles with neither MXM Compute nor VXM arithmetic. Both runs issued
64,160 cycles of simultaneous arithmetic; all six output hashes matched.

### Terminal vector updates alongside output projection

`ftlpu.projection_vector_overlap` enables C++ graph scheduling of terminal
FP32 pointwise results alongside a later independent INT8 linear projection.
The multichunk driver exposes `--projection-vector-overlap on|off`, default
off. The current matcher accepts multiply/add/subtract chains with aligned
rows and a 32-token projection. All delayed values must have only final
exports as consumers. State needed by a later chunk retains its dependency.

The planner moves the terminal updates after the projection's producers,
extends their input lifetimes, and allocates the projection activation/result
outside the vector group's MEM ICU queues. Projection uses E10/11 activation,
W10/11 results, and E31 OneVector weights. Vector blocks prepare and execute
sequentially so their input scratch can be reused. Their emitted preparation
and native instructions are checked against projection and weight-page
resources. Preparation begins at the projection's first emitted instruction,
after the paging rendezvous drain. Resource or capacity failure disables this
policy and retries compilation while retaining the existing Conv policy.

```powershell
$root = "results/qwen3_5_0_8b_prefill_chunkwise_attention_seq64"
foreach ($mode in @("off", "on")) {
    python compiler/tests/qwen3_5_prefill_multichunk_test.py --package results/qwen3_5_0_8b_decoder_layer_prefill/package --build-dir build-ftlpu-vs2026-direct --target-config ../FTLPU-CMODEL/config/ftlpu-lpu32.json --output-dir "$root/projection_vector_$mode" --seq-len 64 --projection-solve-overlap on --projection-conv-overlap on --passive-sr-multicast on --projection-vector-overlap $mode
    if ($LASTEXITCODE) { exit $LASTEXITCODE }
}
python compiler/tools/compare_projection_vector_overlap.py "$root/projection_vector_off" "$root/projection_vector_on"
python compiler/tests/projection_vector_overlap_test.py --build-dir build-ftlpu-vs2026-direct --target-config ../FTLPU-CMODEL/config/ftlpu-lpu32.json --output-dir results/projection_vector_overlap
```

Verified on 2026-11-10 with seq64, batch 1, chunk32, 500 MHz and 51.2 GB/s.
Both runs enabled projection/solve, projection/Conv and passive SR multicast.
All six exported tensors and every reference error matched byte for byte or
value for value. Per-FU arithmetic work, initial input payload and the nine
dynamic weight pages (14,811,136 bytes) matched.

| Actual cycles | Off | On |
| --- | ---: | ---: |
| Initial ingress | 14,572 | 14,572 |
| Logical execution | 645,135 | 613,714 |
| Exposed DDR waits | 44,965 | 44,965 |
| Cold through computation | 704,672 | 673,251 |
| Diagnostic HostOutput | 152,816 | 144,624 |
| Full trace | 857,488 | 817,875 |
| Simultaneous MXM Compute / VXM arithmetic | 64,160 | 79,780 |

The final projection occupies runtime [628915, 661734). State block 0
occupies [637532, 645785), with all 8,192 arithmetic cycles overlapped.
Block 1 occupies [654255, 662509), with 7,428 of 8,192 arithmetic cycles
overlapped. Its last 764 arithmetic cycles remain beyond MXM Compute.
The allocator's revised final placement also saves 8,192 readback cycles.

The focused regression covers two independent FP32 update blocks and a
projection that depends on both blocks. The independent case saves 6,380
cold cycles and gains 16,081 simultaneous cycles. The dependent case keeps
the same 134,181 cold cycles and zero overlap. All outputs and the local
FP32/projection references pass. The final compiler reproduces the executed
images, recorded in each result's `final_program_equivalence.json`.

The policy currently handles terminal exports. Some compatible projection
input aliases can still conflict with vector queues after address reuse and
fall back to serial execution. Broader alias-aware placement remains future
work; the verified seq64 group is enabled without fallback.
