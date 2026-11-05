# Chunkwise output coefficient H

The incremental real-weight Qwen3.5-0.8B test now stops after
`H = tril((Q K^T) * R)`, following the zero-State FP32 forward solve.
Q already includes the model's `1/sqrt(key_dim)` scale. R has a unit diagonal and is not multiplied by beta. It is generated once,
retained across the forward solve, and read by both L and H. L prepares a
separate strict-lower beta-weighted coefficient without modifying shared R.

The logical graph expresses a generic `cross_gram` contraction followed by
pointwise multiplication. The C++ compiler fuses that pair using the existing
weighted-Gram emitter. Q and K retain their L2Norm producer homes, striped by
D32 tiles across banks. Both hemispheres execute distinct alternating heads.
MXM uses 1vector weight loads and sends its BF16 result directly into VXM;
only the final FP32 H is written to MEM. No QK matrix spill, producer layout
copy, host arithmetic, or model-specific CModel operation is introduced.

R is generated from retained FP32 G using the shared causal exponential
lowering. VXM feeds SXM directly to write native `[i,head,j]` BF16 rows.
There is no second causal-exponential node before H. L's coefficient preparation
uses the generic weighted-Gram, row-scale and head-mask graph, lowered entirely
in C++. SXM broadcasts FP32 beta to token-pair vectors; VXM applies a shared
32x32 BF16 static mask and multiplies shared R. The resulting FP32 coefficient
is consumed directly by the MXM-result weighting phase.

The fixed VXM interface cannot take a third MEM stream in an internal chain
stage. Consequently this version retains FP32 beta-mask and coefficient scratch;
it does not claim a single chain for all three independent inputs. The dynamic
Gram matrix itself is never spilled. Both coefficient passes use depth-2 chains
at one vector per cycle, and the MXM compute domain remains continuous. The
2 KiB mask is a logical constant, loaded through normal input C2C, not host-side
model arithmetic. No CModel hardware semantics were changed.

## Reproduce

```powershell
python compiler/tests/qwen3_5_prefill_qkv_wide_sxm_test.py `
  --package results/qwen3_5_0_8b_decoder_layer_prefill/package `
  --build-dir build-ftlpu-vs2026-direct `
  --target-config ../FTLPU-CMODEL/config/ftlpu-lpu32.json `
  --output-dir results/qwen3_5_0_8b_prefill_chunkwise_h `
  --sxm-architecture wide --with-h --solve-strategy forward
```

This executes raw macro ICU programs with real DDR/C2C pages. Python constructs
the logical test graph and independent reference, but does not lower or execute
model arithmetic on behalf of the device.

## Shared-R validation (2026-11-03)

Real weights, seq32, 16 heads, key dimension 128; H shape `[16,32,32]`.
The result directory remains `results/qwen3_5_0_8b_prefill_chunkwise_h`.
`summary.json` records measured cycles, per-head L/D/H errors, R's output index,
and the retained coefficient scratch. Both KKT and QKT must have zero inter-tile
or inter-head MXM compute bubbles in the runtime audit. H must retain its diagonal
and have an all-zero upper triangle; L must also have a zero diagonal.

The asymmetric Q/K standalone test covers H2/D32 and H4/D128. Its `--row-scale`
mode additionally checks strict lower beta weighting against an independent
reference. Original weighted-Gram and cross-Gram regression cases remain valid.
Runtime pipeline and per-ICU programs are exported from actual raw macro execution.
Diagnostic host readback is excluded from operator latency counts. The test stops
at H: `H D`, nonzero-State output contributions, and outgoing State remain later
steps.

Measured after sharing R:

| Item | Before | Shared R |
| --- | ---: | ---: |
| Logical execution cycles | 198,820 | 198,089 |
| Runtime session end, including initial ingress/paging | 272,754 | 272,024 |
| R generation count | 2 | 1 |
| H stage cycles | 1,090 | 1,090 |

H's runtime interval is `[270807,271897)`, with both MXMs continuously active
in `[270846,271870)`. Worst-head H end-to-end NRMSE remains 1.27417%.
Worst-head L end-to-end NRMSE is 0.57591%; D is 1.64113%, and the forward-solve
relative residual is `4.79768e-8`. The removed R stage costs 1,840 cycles, but new
coefficient preparation offsets part of that saving. Net logical saving is 731
cycles; measured session saving is 730 because C2C timing also changes.
