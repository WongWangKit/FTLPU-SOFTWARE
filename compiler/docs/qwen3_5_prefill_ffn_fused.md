# Qwen3.5 prefill FFN fusion experiment

The shared C++ FFN path supports single-token-tile Up/SwiGLU overlap on the
16-ALU, single-MXM-per-hemisphere target. This is not a tiled Gate/Up/Down
fusion: Gate remains resident, each completed Up tile is read by SwiGLU while
later Up tiles compute, and Down begins after hidden materialization/mirroring.
Intermediate tensors still reside in MEM. Multi-token-tile schedules retain
the existing drain-before-SwiGLU behavior.

The fused planar hidden placement uses the result slices when capacity allows,
with Down results assigned an independent bank. Up temporary WRITE3D domains
end at each output pair before a READ3D can consume that pair. Instructions
within one MEM ICU queue remain sequential; no suspended macro interleaving
or additional hardware contexts are introduced.

The chain retains its original Gate operand, so the old second Gate SRAM
write-tap copy is removed. Each completed SwiGLU tile is mirrored while later
Up tiles compute, rather than mirroring the whole hidden tensor at the end.
Both Down hemispheres still require the hidden mirror; this change overlaps
that required traffic and does not claim to eliminate it.

Logical graph lowering propagates common-barrier/encoding policies to native
subgraphs. When linking their schedules it relocates C2C candidate binding IDs
and preserves protocol timeline names, including Down's post-SwiGLU boundary.
The runtime still validates SRAM ownership and drained transport stops.

## Measured seq32 result (2026-10-29)

Qwen3.5-0.8B layer-0 checkpoint weights, W8A16, hidden=1024,
intermediate=3584, deterministic synthetic RMS-normalized BF16 activations.
This is an isolated FFN test, not captured full-decoder activations.
500 MHz, DDR peak 51.2 GB/s, raw-macro execution, no host arithmetic.

| Metric | Fused before tightening | Fused after tightening |
|---|---:|---:|
| Executable logical cycles | 181223 | 174643 |
| Runtime trace cycles including ingress/output | 312620 | 306040 |
| Runtime latency (ms) | 0.625240 | 0.612080 |
| First SwiGLU logical issue | 58693 | 58693 |
| Last Up compute logical issue | 114988 | 114988 |
| Down logical start | 118786 | 115266 |
| MXM MAC issue duty excluding page waits | 94.93% | 98.50% |
| NRMSE versus NumPy reference | 0.00745426 | 0.00745426 |
| Serialized runtime image bytes (includes metadata) | 39982 | 43090 |

Outputs match bit-for-bit. Tightening saves 6580 runtime cycles (2.10%) and
3.63% of executable logical cycles. Each MXM issues useful MAC work for
172032 cycles. The utilization denominator is the invocation span minus
80117 page-ready wait cycles; it excludes initial ingress and final output.
This measures issue duty, not individual arithmetic pipeline stage occupancy.
The runtime image grows because mirror transfers now retire separately per
tile. The schedule IR's nominal timing differs from the linked executable.

The prior same-build tail/fused comparison was 312616/312620 cycles; that tail
result predates the copy-removal change and is not a current-build control.
Do not use earlier mixed-build measurements as comparisons. Down still starts
after the last hidden tile is ready, and intermediate data still resides in
MEM. Keep the default strategy unchanged pending broader fusion work.

Run `compiler/tests/qwen3_5_decode_ffn_test.py` with:

```text
--package results/qwen3_5_0_8b_decoder_layer_prefill/package
--build-dir build-ftlpu-vs2026-direct
--target-config ../FTLPU-CMODEL/config/ftlpu-lpu32.json
--output-dir results/qwen3_5_0_8b_prefill_ffn_fused
--batches 32 --prefill --ffn-schedule fused
```

Use `tail` and the corresponding tail output directory for the control run.
The test checks numerical error and whether SwiGLU actually overlaps Up.
For the fused seq32 LPU32 case it also rejects a whole-hidden mirror tail
or the unused Gate write-tap copy;
`--check-only` rechecks existing outputs without simulation.
