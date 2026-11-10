# Continuous multi-token-tile projection

The generic linear projection schedule now has a continuous local-dequant
path for multiple 32-token tiles. It preserves int8 weight storage, BF16
activation/dequant/output, FP32 accumulation, and the existing two MXM weight
buffers. It changes only scheduling; there is no new hardware mode or host
arithmetic.

## Supported selection

The new path requires 32-row MXM, M greater than 32 and divisible by 32,
M within the target accumulator capacity, N divisible by 64, K divisible by
64, local dequantization, target support for weight/activation overlap,
two weight buffers, and ordinary hemisphere-local
results. M32 retains its existing path. Odd K-tile counts, replicated outputs,
and other target cases retain their existing handling. OneVector weight loading
retains its existing 32-column protocol and stream assignment.

The old multi-tile schedule waited for all M rows to compute before starting
the next K weight read and IW. Each K tile therefore paid another MEM transport
and load lead, despite having two weight buffers. It also issued a separate
accumulator read after each N group.

## Three-dimensional hardware instructions

| Queue | Domain | Behavior |
| --- | --- | --- |
| Weight MEM / MXM IW | column, K tile, N group | Load the next K tile into the alternate buffer during current computation |
| Activation MEM | token, K tile, N group | Read all M tokens consecutively, then advance K |
| MXM Compute | row32, token tile, K tile | One complete sequential ICU instruction per N group |
| Result MEM | token, N group | Store final-K output directly |

Compute's token-tile dimension advances the ACC base by 32. Its K dimension
selects `toggle_dim2`; only the terminal K iteration streams BF16 and clears
ACC. This avoids the incorrect shortcut of a 64-row inner loop, which would
wrap the physical MXM row index onto the first 32 ACC rows. The N-group
instructions execute consecutively and never interleave on one ICU queue.
No transpose, MEM-to-MEM copy, repeated dequantization per token tile, or extra
DDR weight transfer is introduced.

## Reproduction and acceptance

From the repository root, after the shared compiler build:

```powershell
python compiler/tests/linear_projection_multitile_runtime_test.py --build-dir build-ftlpu-vs2026-direct --target-config ../FTLPU-CMODEL/config/ftlpu-lpu32.json --output-dir results/linear_projection_multitile
```

The focused test covers M32, M64, M96 and M128, multiple N groups, and odd
K-tile fallback. It uses distinct data in each token tile to catch ACC-address
aliasing. Acceptance requires projection NRMSE below 2%, no host operations,
and actual runtime Compute spans with no internal gaps in the supported even-K
cases. It exports `runtime.pipeline.csv` and the executed ICU program including
pre-execution uploads. The test scope is a single projection, without persistent
state or claims of a deployed attention/decoder layer.

An optional `--baseline-compiler <pre-change-ftlpu-compile>` checks bit-identical
outputs, unchanged weight traffic, improved complete time through computation
including ingress, and unchanged M32/odd-K fallback timing. Diagnostic host
readback is reported separately. Real Qwen3.5 seq64 attention still requires
its full existing numerical and persistent-State regression; isolated results
are not a substitute for that acceptance.

For an M64, K1024, N2048 page, useful Compute is 65,536 cycles. The old schedule
used 111,504 nominal cycles per page; the new estimate is approximately 65.6k
before measurement. Complete attention latency, including DDR/C2C waits, must
be reported from its actual run rather than inferred from this estimate.
