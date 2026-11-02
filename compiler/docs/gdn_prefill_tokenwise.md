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
