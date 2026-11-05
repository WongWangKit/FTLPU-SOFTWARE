# GDN prefill lowering strategies

`ftlpu.gdn_prefill_strategy = "tokenwise" | "chunkwise"` is a module or
function policy on the semantic `kernel.gated_delta_rule` graph. Function
policy survives whole-function region extraction. Omitting the policy keeps
the existing automatic dispatch; explicitly selecting tokenwise bypasses the
BF16 seq32 native-chunkwise shortcut.

The Python importer supplies one GDN scan. C++ expands tokenwise into the
generic `kernel.scaled_matvec` implementation shared with batch-1 decode:

1. Scale the old State by the multiplicative decay coefficient.
2. Stream the scaled State to MXM for the K contraction, while retaining it
   through MEM_WRITE_TAP.
3. Compute `delta = beta * (V - KS)` in the VXM epilogue.
4. Broadcast delta with wide SXM and perform the rank-one State update.
5. Stream the updated State into MXM for the Q contraction.

The recurrence has a real dependency on the preceding token's State. There
are no host state handoffs. The planner aliases an exact producer State home
at the next step, and allows the updated State to overwrite that home only
after its last reader, while checking other live aliases and MEM ICU queue
conflicts. Composite results used only internally (scaled State, KS, delta)
still receive lifetime-tracked physical allocations.

State remains `[head,value,key]` internally. The semantic GDN ABI continues
to describe `[head*key,value]`; immutable input packing and logical views
translate that interface once, rather than transposing State at every token.
The test exports a native `[head*value,key]` checkpoint view to avoid an
additional transpose for validation. In BF16 storage mode the MXM readout
already has the semantic BF16 type; no extra readout conversion is emitted.

A compiler-declared `imem.drain` follows each completed token. When the
combined raw-macro program exceeds physical iMEM, the runtime may reload at
these drained boundaries without losing SRAM State. This is instruction
paging, not a host-side recurrence or enlarged hardware instruction memory.

## Validation

The test defaults to `--rounding rne`. State, scale, beta, scaled State, KS,
delta, updated State and readout use actual BF16 storage. VXM arithmetic and
MXM accumulation remain FP32; existing FU output converters round to nearest,
ties to even before emitting two byte planes. This changes neither CModel
arithmetic nor its fixed input ports. The gates are emitted with BF16 RNE
outputs, and immutable gate constants/initial State are packed with RNE.
The opt-in module policy `ftlpu.vector_bf16_rne_spill` also changes internal
vector-chain spill results to BF16 before memory planning (including the
expanded Softplus branches). It preserves FP32 values inside each fused
chain and leaves function return types intact. This replaces truncation at
existing boundaries without introducing extra conversion chains.

`--rounding truncate` retains the previous FP32 storage/high-plane-read
baseline. The generic `scaled_matvec` emitter supports both storage types;
existing decode callers keep their original type and behavior. BF16 State
at the semantic GDN interface currently requires explicit tokenwise policy.
The allocator keeps concurrent State/K/V inputs on separate MEM ICU queues.

The current runner uses MXM `ws` and explicit
`ftlpu.common_transport_overlap = false`. The legacy transfer-overlap lease
proof does not support chain VXM and can also exceed its issue-expansion
limit on seq32. This test therefore uses compiler/common SYNC paging, with
real DDR/C2C transfers and MEM synchronization; it does not claim validation
of overlapping transport leases. Recompile old binaries for the current ABI.

Small nonzero-State handoff regression:

```powershell
python compiler/tests/qwen3_5_prefill_tokenwise_test.py --core-only --tokens 2 --build-dir build-ftlpu-vs2026-direct --target-config ../FTLPU-CMODEL/config/ftlpu-lpu32.json --output-dir tmp/tokenwise_scan_core
```

Real Qwen3.5-0.8B seq32 weights, rebuilt prefill frontend and complete GDN
recurrence (before output normalization/gating/projection and FFN):

```powershell
python compiler/tests/qwen3_5_prefill_tokenwise_test.py --tokens 32 --package results/qwen3_5_0_8b_decoder_layer_prefill/package --build-dir build-ftlpu-vs2026-direct --target-config ../FTLPU-CMODEL/config/ftlpu-lpu32.json --output-dir results/qwen3_5_0_8b_prefill_tokenwise_gdn
```

The test uses raw macro instructions, a nonzero initial State, and independent
serial equations. It reports two separate comparisons: hardware correctness
against the actual frontend tensors with explicit BF16 reload boundaries,
and numerical error against the end-to-end FP32 equations. FP32 accuracy is
reported separately because repeated BF16 truncation can accumulate across
tokens. Diagnostic frontend outputs permit these errors to be distinguished
without feeding host-computed intermediates back to the executable.

`runtime.pipeline.csv` and `runtime_icu_programs` describe executed commands,
including synchronized initial input transfers. The shared decode streaming
audit checks the scaled-State and updated-State direct MXM routes. The scan
test also rejects full-State transports between tokens.

### Truncation baseline (2026-10-31)

The real-weight frontend plus 32 recurrent steps completed with 786,819
execution cycles, zero host operations, raw-macro instructions, and wide SXM.
All 32 updated States share the same native home: both hemispheres, bank 1,
slices 16–19, base row 0, 8,192 rows. The runtime streaming audit passed for
all steps. This test begins at QKV projection and ends at the Q/State readout;
it is not a complete decoder layer.

| Reference | Output NRMSE | Final State NRMSE |
| --- | ---: | ---: |
| Actual frontend tensors, explicit BF16 reload truncations | 0.0006203% | 0% |
| End-to-end FP32 equations with real weights | 11.9606% | 19.5340% |

The result is **correctness_passed_accuracy_failed**, not an FP32 accuracy
pass. The runtime matches the chosen arithmetic, including every head's
State, but BF16 truncation accumulates over the recurrence. State storage is
FP32; the current fixed-port reads use its BF16 high planes, so retaining all
four bytes in MEM alone does not preserve FP32 arithmetic between steps.

Using the same actual frontend tensors and separately enabling only one
class of truncations gives the following State errors against the
untruncated recurrence: State reads 18.8449%, gate reads 7.4492%, and
prediction/delta intermediates 0.0566%. These are independent ablations,
not additive contributions. State read precision is the main next issue.

Checkpoint validation currently covers the native `[head,value,key]` output
view. Generic physical export of the transposed FP32 State is not validated
by this test; no physical State transpose is needed for the recurrence.

### BF16 RNE result (2026-11-02)

Real-weight seq32 completed with 777,732 execution cycles, zero host
operations, and the same 32-step recurrent State reuse. State is now stored
on slices 16–17 (both hemispheres, bank 1, base 0, 8,192 rows). All five
scaled-matvec results use two byte planes. RNE happens at existing FU output
boundaries; per-token readout conversion nodes are no longer needed.

| Reference | Output NRMSE | Final State NRMSE |
| --- | ---: | ---: |
| Actual frontend tensors, independent BF16 RNE recurrence | 0.00003497% | 0% |
| End-to-end FP32 equations with real weights | 3.00357% | 4.89306% |

All 16 heads have exact State agreement with the RNE reference. This is
still an FP32 accuracy failure under the 3% acceptance threshold, despite
the large improvement over truncation. The synthetic two-token and seq32
handoff checks match the RNE recurrence exactly for both output and State;
the legacy FP32-storage decode readout regression also passes.

The original truncation metrics are retained in
`results/qwen3_5_0_8b_prefill_tokenwise_gdn/summary.truncate_baseline.json`.
The main directory and its runtime pipeline/ICU export are updated with RNE.

### Current-ABI rerun (2026-11-03)

The full real-weight seq32 frontend plus recurrence was rebuilt with MXM WS
and rerun with the corrected ModelSession output-DMA elapsed clock. The test
explicitly disables the unsupported legacy chain-VXM transport-overlap
proof and retains common SYNC paging. All eight outputs were produced;
execution took 777,732 cycles, with five weight-page transfers (6,422,528
bytes), 69,314 runtime weight-wait cycles and no host operations.

Numerical metrics are unchanged: FP32-reference output NRMSE 3.00357%,
State 4.89306%; every head's State exactly matches the BF16 RNE reference.
The regenerated runtime pipeline and per-ICU programs replace the previous
files in the same results directory. This remains an execution-correctness
pass and an FP32-accuracy failure at the 3% threshold.


### Full FP32 VXM ingress (2026-11-04)

`qwen3_5_prefill_tokenwise_test.py --rounding fp32` selects full FP32
State/gate/intermediate storage and `ftlpu.vector_fp32_ingress = true`.
The test defaults to this mode; `--rounding rne` and `--rounding truncate`
retain the earlier explicit precision modes. This is still raw-macro C++
tokenwise lowering, with common SYNC paging and no host recurrence.

The scaled-matvec primitive reads all four State, eg, beta and delta bytes.
Decay and rank-one update use the C0 FP32 operands on streams 0-3 / 4-7;
the streamed prediction correction uses the independent C4 ports 8-11 /
12-15. The small BF16 V vector is expanded to FP32 before that correction,
so its RHS does not overlap the prediction's four-byte LHS. Beta and K are
retained in local scalar registers. Both State results remain direct
VXM-to-MXM streams with MEM WRITE_TAP retention, and updated State stays at
the same native address for the next token.

The wide SXM hardware transports two byte planes per transpose. FP32 gate
and delta broadcasts therefore use a low-pair pass and a high-pair pass;
FP32 delta scratch occupies slices 48-51 in alternating banks. Updated
State readout returns on W28-W31 to avoid the concurrent broadcast streams.
No four-plane SXM instruction or model-level CModel operation is introduced.
The planner reserves actual operand homes rather than obsolete standalone
scaled-matvec input prototypes.

Precision is mixed, not end-to-end FP32: MXM receives the BF16 high planes
of scaled/updated State, and the final attention output is BF16 RNE. Q/K/V
and projection precision are unchanged. Nonlinear VXM functions retain the
hardware LUT behavior. Full FP32 input eliminates recurrent State reload
truncation but does not change these other boundaries. The independent
numerical reference models them separately from the all-FP32 equations.


The real-weight seq32 run completed in 890,093 execution cycles, with
69,516 weight-wait cycles, five weight pages and zero host operations.
Execution cycles increase 14.45% from the 777,732-cycle BF16 RNE baseline;
FP32 broadcast passes increase the head-boundary scheduling cost. The
persistent State occupies four byte planes (slices 16-19, bank 1, base 0,
8,192 rows, both hemispheres), 1 MiB per hemisphere versus 512 KiB for BF16.

| Reference | Output NRMSE | Final State NRMSE |
| --- | ---: | ---: |
| Actual frontend, independent FP32 recurrence with MXM BF16 high planes | 0.001281% | 0.00008218% |
| End-to-end FP32 equations with real weights | 0.735067% | 0.664640% |
| Previous BF16 RNE run, end-to-end FP32 reference | 3.00357% | 4.89306% |

Both end-to-end errors now meet the unchanged 3% threshold. The previous
RNE summary is preserved as `summary.rne_baseline.json` in the same result
directory. The scope still ends at Q readout and final State, without the
output gate/projection, decoder residuals or FFN.


Validation passed for all 32 runtime streaming steps and all 16 heads.
The audit confirms full four-plane saved-State reads, no updated-State
reload before Q readout, alternating delta broadcast banks, and one shared
State home across all tokens. Synthetic two-token full-FP32, legacy BF16
RNE and retained-input scaled-matvec/readout regressions also pass.
`runtime.pipeline.csv` and `runtime_icu_programs/index.csv` are regenerated
from this execution; the complete runtime trace ends at cycle 1,132,344,
including ingress, weight waits and output transfer outside compute.


### Complete tokenwise GDN attention tail

The real-weight seq32 test now includes the attention tail by default.
`--recurrence-only` retains the earlier Q-readout stopping point;
`--core-only` still runs the small synthetic recurrence regression.

After the 32 FP32 State updates, the graph computes:

```
Z = X W_z                              [32, 2048]
N = O / sqrt(mean(O^2, head_dim) + eps) * delta_norm
U = N * SiLU(Z)                        [32, 16, 128]
attention = reshape(U, [32, 2048]) W_o  [32, 1024]
```

Normalization is per 128-element value head. Its learned `delta_norm`
weight is used directly, without a `1 + weight` offset. O, Z, N and U use
BF16 storage, while the norm/gate VXM chains and MXM accumulators compute
in FP32. Recurrent State and its VXM operands retain the FP32 policy above.
No decoder input RMSNorm, residual additions or FFN are included.

The shared C++ retained-RMSNorm path now recognizes mirrored BF16 planar
producer results as well as FP32 matrix views. It tiles the reduction rows
and selects alternate source hemispheres directly from the retained readout
addresses. There is no full-readout join or layout conversion before RMSNorm.
The shared native SiLU-multiply emitter consumes the resulting partition
row descriptors alongside the Z projection's native descriptors. One initial
normalization-weight preparation is reused for all subsequent tiles. Output
projection uses the existing common projection lowering and DDR/C2C pages;
its required activation gather remains explicit in the graph plan.

Diagnostics keep the prior output indices 0-7 and add Z (8), normalized O
(9), gated O (10), and final attention (11). Gate export uses physical head
parity order, unpacked only for independent host validation. The executable
consumes compiler-provided partition descriptors directly. Tail validation
compares each stage with its actual inputs and the final attention with the
independent FP32 recurrence plus tail. State export stays a native checkpoint
view; Python never supplies a computed recurrence result to the full test.


#### Full attention validation (2026-11-04)

Real-weight seq32 passes both execution correctness and the unchanged 3%
end-to-end numerical threshold. All 32 streaming recurrence audits pass;
final State/readout numerical metrics are unchanged after attaching the tail.

| Comparison | NRMSE |
| --- | ---: |
| Z projection against real-weight reference | 0.209025% |
| RMSNorm given actual readout | 0.176686% |
| SiLU gate given actual normalized O and Z | 0.171133% |
| Output projection given actual gated tensor | 0.251075% |
| Full attention against FP32 recurrence plus tail | **0.590722%** |
| Final State against FP32 recurrence | 0.664640% |

Execution takes 987,164 logical cycles with 114,621 runtime weight-wait
cycles, seven dynamic pages (10,616,832 bytes), twelve output bindings and
zero host operations. The runtime trace ends at 1,327,061 and includes
initial ingress and diagnostic exports; it is not an attention-only TTFT.

The native stage ranges below use actual `Paging.Consumer` events to
translate each stage's logical interval into the runtime trace clock:

| Tail stage | Runtime cycle range [start, end) |
| --- | --- |
| Z projection | [1025457, 1058277) |
| Per-head RMSNorm tiles | [1059129, 1066740) |
| Continuous SiLU gate | [1066931, 1067991) |
| Output projection | [1099220, 1132040) |

There are no readout-to-norm or norm-to-gate layout copies. The eight norm
preparation domains occur only before the first tile and initialize gamma;
the other fifteen tiles reuse it. Projection activation gathers and diagnostic
exports remain separately visible. The recurrence-only FP32 baseline is
preserved as `summary.fp32_recurrence_baseline.json` in the same directory.

`tokenwise_gdn_output_norm_test.py` additionally checks two native 128-wide
normalization tiles, weight reuse and direct ingress (NRMSE 0.183159%). Its
32-wide variant verifies the retained generic single-vector reduction path
(NRMSE 0.186150%); single-vector BF16 rows do not select the multi-vector
native feedback schedule. Both run raw-macro programs on CModel.

### Beta/decay producer-to-recurrence mapping

The C++ connected-graph planner now pairs independent narrow INT8 projections
that share one BF16 input tile. For the seq32 gate branches, East computes B
and West computes A in one common projection schedule. Each side retains its
own checkpoint quantization scale and the existing 1-vector weight-load mode.
The projection writes each output word into a separate mirrored byte pair;
pointwise consumers alias those addresses directly. No Python-side packing
or physical schedule is required. The pairing is limited to the supported
local-dequant MXM strategy; other projections keep their existing path.

The graph planner also permits a private unary pointwise consumer to join its
producer region. Consequently `g = -exp(A_log) * softplus(a + dt_bias)` and
`eg = exp(g)` lower together: the final multiply and exponential share one
VXM chain, and private g does not need a MEM round trip. Externally requested
intermediate results are still preserved. Beta and eg remain packed FP32
rows; the tokenwise scaled-matvec emitter reads a token's original row by
address, including the existing wide-SXM coefficient broadcast. It no longer
copies both packed gate vectors to provisional input allocations every token.

Softplus retains its numerically guarded Log implementation. Let
`t = exp(-abs(x))`, `a = 1/16`, and `p(t) = t - t^2/2 + t^3/3`.
Since p is monotone for `0 <= t <= 1`, the small-tail branch can compute
`min(p(t), p(a))` instead of `p(min(t,a))`. The former fits C0-C7 in one
fixed-port chain using the retained original t:

```
multiply(1/3) -> add(-1/2) -> multiply(t) -> add(1)
              -> multiply(t) -> negate -> max(-p(a)) -> negate
```

This removes the clamped-t spill and the opposite-hemisphere branch-alignment
copy. Other Softplus chain boundaries remain explicit where the fixed input
ports and retained operands require them. Both hemispheres still compute
different token halves; mirroring a result needed by State consumers does
not duplicate its arithmetic.

`chain_vxm_gates_fp32_softplus_runtime_test` checks the negative tail.
`chain_vxm_gates_paired_recurrence_runtime_test` checks distinct projection
scales, fused eg, direct gate addresses and two nonzero-State updates against
an independent reference. The original gate test continues to exercise the
legacy path. All three execute raw-macro ICU programs.

The 2026-11-04 real-weight seq32 rerun passes with the same nonzero initial
State and the full attention tail. The immediately preceding run is the
comparison baseline (966,518 logical cycles), rather than the earlier
987,164-cycle tail-introduction run above.

| Metric | Before | After |
| --- | ---: | ---: |
| B/A projection stage cycles, summed | 2,156 | 1,094 |
| Elementwise beta/g/eg stage cycles, summed | 920 | 638 |
| Projection start to gate-ready logical span | 3,691 | 1,928 |
| Gate-to-recurrence MEM transport domains | 256 | 0 |
| Full attention logical execution cycles | 966,518 | 952,328 |
| Runtime weight-wait cycles, additional | 114,749 | 113,792 |
| Dynamic weight pages | 7 | 6 |

Beta NRMSE remains 0.039625%, eg 0.023616%, final State 0.664640%, and final
attention 0.590722%. The optimized runtime trace ends at 1,287,315, including
ingress and diagnostic exports; its paired-projection/gate interval is
`[254366, 256294)`. The refreshed runtime pipeline, executed ICU programs and
`gate_optimization.json` are in the existing tokenwise GDN result directory.

#### Gate optimization validation (2026-11-04)

The real-weight seq32 full-attention rerun passes. This comparison uses the
same nonzero initial-State regression fixture and FP32 State/ingress as the
saved baseline immediately before the gate changes; it is not a cold-prefill
zero-State measurement.

| Measurement | Before | After |
| --- | ---: | ---: |
| A/B projection schedule cycles, summed | 2,156 | 1,094 |
| Pointwise gate/eg schedule cycles, summed | 920 | 638 |
| A/B start through beta/eg ready, logical cycles | 3,691 | 1,928 |
| Gate-to-recurrence MEM transport domains | 256 | 0 |
| Full attention logical execution cycles | 966,518 | 952,328 |
| Runtime weight-wait cycles, separate from execution | 114,749 | 113,792 |
| Dynamic weight pages | 7 | 6 |

The gate span decreases 47.76%; full attention logical execution decreases
1.47%. Beta, eg, readout, final State and final attention are byte-identical
to this saved baseline. Against the independent reference, beta NRMSE is
0.039625%, eg 0.023616%, final attention 0.590722%, and final State 0.664640%.
The three gate CTests pass, including independent projection scales and
negative-tail coverage.

Results remain in `results/qwen3_5_0_8b_prefill_tokenwise_gdn`. The regenerated
runtime pipeline places the combined projection and gates in
`[254366, 256294)`. `runtime_icu_programs` is exported from
`actual.runtime.ftlpu`, including the pre-execution program; it is not a
static-only export. `gate_optimization.json` records the saved-baseline
comparison. The complete trace ends at 1,287,315, including ingress and
diagnostic exports outside logical execution.
