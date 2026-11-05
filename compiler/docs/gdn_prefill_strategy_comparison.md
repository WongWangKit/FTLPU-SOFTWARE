# GDN prefill strategy comparison

`compiler/tests/qwen3_5_prefill_strategy_compare_test.py` compares the current
tokenwise recurrence and MXM chunkwise core under an explicit shared contract.
This is a **single seq32, zero-initial-State core comparison**, not an end-to-end
model, full attention, nonzero-State or multi-chunk acceptance test.

## Immutable inputs

The fixture is frozen from the actual CModel frontend exports of the passed
real-weight `qwen3_5_prefill_qkv_wide_sxm_test.py` test. It contains exactly the
same Q, K, V, beta and g for both strategies. Its manifest records source/output
SHA-256 hashes and canonical tensor dimensions. State is explicitly zero and
both paths reject a modified fixture. The fixture is reused on subsequent runs;
changing the source directory does not silently overwrite it.

Semantic shapes are Q/K/V `[32,16,128]`, beta/g `[32,16]`, and State
`[16,128,128]` in head/key/value order. Q already includes the `1/sqrt(128)`
scale. Q/K L2Norm, causal Conv1D/SiLU and weight projections were executed by the
source frontend; they are excluded from this core comparison. There is no
reconstruction of g through log(exp(g)).

## Precision contract

* Q/K/V: identical BF16 bits; beta/g: identical FP32 bits.
* Initial/final State: FP32. VXM uses complete FP32 ingress and arithmetic.
* MXM uses BF16 operands and FP32 accumulation. Tokenwise reads State high
  planes with truncation; this is explicitly **not RNE**.
* Chunkwise retains FP32 solve workspaces and uses its existing compensated
  three-BF16-component products for the solve. R and output/State contractions
  retain the existing BF16 paths. Internal operation/rounding counts differ.
* Both use the same hardware Exp LUT, including its FP16/FTZ boundaries.
* Both export BF16 RNE readout and FP32 final State. Chunkwise performs the
  readout conversion on hardware and includes its cost, rather than casting on
  the host after timing. Native output layouts are unpacked only for validation.

The common CPU oracle uses FP64 recurrent equations and independently checks
the chunk triangular-solve identity. Acceptance requires NRMSE < 3% and cosine
> 0.999 for readout and State, globally **and per head**. Cross-strategy errors
are reported separately and do not replace comparisons against the oracle.
This measures numerical accuracy at common external boundaries, not identical
internal arithmetic or bitwise equality between algorithms.

## Hardware execution and measurement

Both use production C++ lowering, raw-macro ICU programs, wide SXM, the same
target, compiler and runner, and disabled transport overlap for this resident
core baseline. Python supplies tensor equations, input serialization and
references; it generates no hardware addresses, cycles, or FU/ICU instructions.

Both export only readout and final State. The compute denominator is the
runner's logical execution cycles, including on-chip transfers and conversions
but excluding external blocked time and host ingress/egress. Runtime wall
cycles and external wait counters are reported separately. MXM/VXM utilization
means the union of MXM Compute/non-Bypass VXM issue intervals divided by that
denominator; it is not physical lane occupancy or peak FLOPS utilization.

Zero State is an explicit input of tokenwise and a supported specialization of
chunkwise. Consequently cold input traffic may differ. Both actual runtime
pipelines include their input transfers; this difference must not be hidden in
an end-to-end latency claim.

Example (from the software repository):

```powershell
python compiler/tests/qwen3_5_prefill_strategy_compare_test.py `
  --build-dir build-ftlpu-vs2026-direct `
  --target-config ../FTLPU-CMODEL/config/ftlpu-lpu32.json `
  --source-frontend results/qwen3_5_0_8b_prefill_chunkwise_attention `
  --output-dir results/qwen3_5_0_8b_prefill_strategy_compare
```

`--compile-only`, `--run-only`, `--validate-only`, and
`--strategy tokenwise|chunkwise|both` allow incremental execution. Results are
stored under `fixture/`, `tokenwise/`, and `chunkwise/`; `comparison.json`
appears only after both paths pass with matching fixture/toolchain contracts.
Each strategy exports `runtime.pipeline.csv` and actual `runtime_icu_programs/`.

Remaining acceptance stages are shared nonzero State, continuous multi-chunk
State/Conv-history handoff, then a full-attention comparison with shared frontend
and output tail. They are explicitly outside this first baseline.
