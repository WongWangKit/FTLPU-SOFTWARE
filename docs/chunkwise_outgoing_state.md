# Chunkwise outgoing State

The incremental real-weight Qwen3.5-0.8B test extends the zero incoming-State
path through O to the outgoing State:

`S_out = K^T diag(R[31,:]) D`.

The compiler uses the equivalent `(diag(R[31,:]) K)^T D`. The existing R is
retained and reused; neither gamma nor causal exponential is recomputed.
The first State dimension is key, the second is value: `[16,128,128]`.
Output storage is FP32 (1 MiB total, split across the two hemispheres).

## Lowering

The logical graph expresses a last-row weighted head transpose, a head
cross-product, and an optional head-matmul sharing D. C++ recognizes this chain and
emits real MEM/SXM/VXM/MXM domains. Python only constructs the logical test and
independent numerical references. There are no model operations in CModel.

For each pair of heads and each V32 block, the dataflow is:

1. Read D's FP32 upper byte planes as truncated BF16. Wide SXM transposes D
   directly to MXM's 1vector weight port on East streams 18/19.
   With output/State fusion, compute 32 O rows from H using that resident D
   tile. Write only final O. Keep the selected MXM buffer resident for the
   State product below; no second D
   transpose or load is issued.
2. Read K directly from its native distributed16, bank-striped L2Norm home.
   Wide SXM transposes K's four K32 tiles consecutively onto VXM's fixed West
   inputs. Each output vector spans the 32 tokens.
3. Read R's retained final-row vector repeatedly and multiply lane-wise in
   VXM. FP32 multiplication is rounded to BF16 at the chain tail; the result
   streams directly to the opposite hemisphere's MXM activation port.
4. MXM reuses the resident D tile for all 128 key rows, accumulates in FP32,
   and sends the final State through passive SR transport to its native home.

There are no coefficient broadcasts, weighted-matrix spills, transpose spills,
or producer-layout copies. Only final O and State are written to MEM. Both sides
compute different alternating heads. State exports use grouped native heads:
even heads followed by odd heads, unpacked by the test.

Wide SXM cannot change output direction while a prior job is live. D's East
phase and K's West phase therefore have explicit drains. Compute is continuous
within each 128-row product, but gaps remain between value blocks and heads.
The compiler protects H/K/R/D operand queues at their
producer homes and aliases all uses of D to one native allocation.

On a target with two overlapping MXM weight buffers, lowering groups two D
tiles at a time. SXM transposes D0 and D1 32 cycles apart into buffers 0 and 1,
then drains and transposes both sets of K tiles consecutively. MXM executes
`O0 -> State0 -> State1 -> O1` as one continuous 320-cycle burst for K128.
The next group's D0 starts loading buffer 0 while O1 consumes buffer 1.
Final MEM writes also overlap preparation of the next group. No group waits
for every output write plus an arbitrary 32-cycle guard before starting D.

The wide SXM direction drain is still enforced: a 70-cycle tile span versus
a 32-cycle initiation interval costs 38 cycles at each of the two direction
changes per group. Ten transposes (two D and eight K) therefore occupy 396
cycles per group. With 320 MXM compute cycles, the steady-state inter-group
gap is 76 cycles. This is the remaining limit of this direct-stream mapping;
the implementation does not claim globally gap-free compute. A lone final
tile and targets without overlapping dual buffers use a one-tile group.

## Reproduce

```powershell
python compiler/tests/qwen3_5_prefill_qkv_wide_sxm_test.py `
  --package results/qwen3_5_0_8b_decoder_layer_prefill/package `
  --build-dir build-ftlpu-vs2026-direct `
  --target-config ../FTLPU-CMODEL/config/ftlpu-lpu32.json `
  --output-dir results/qwen3_5_0_8b_prefill_chunkwise_state `
  --sxm-architecture wide --with-outgoing-state `
  --o-mxm-precision bf16 --solve-strategy forward
```

The output is `actual.13.bin` with the forward solve. The existing prefix,
O, real DDR/C2C paging and raw macro execution remain enabled. Runtime pipeline
and per-ICU programs are exported into the same result directory.
`--no-fuse-output-state` selects the previous separate stages. Fusion is used
only with `--o-mxm-precision bf16`; the FP32-component O path remains separate.

Validation compares hardware arithmetic against an explicit BF16 rounding /
truncation reference, then compares final State against independent sequential
FP64 State recurrence using the real-weight reference frontend. The runtime
audit requires exactly the final State's MEM writes and the expected MXM work.
Standalone asymmetric random tests cover H2/K32/V32 and H4/K128/V128.

Nonzero incoming State, a second chunk, gated output norm, output projection,
and FFN are outside this milestone. For a later chunk, D's right-hand side and
O also need old-State contributions; adding only gamma times old State here
would not be sufficient.

## Validation results (2026-11-03)

The earlier separate-stage real-weight seq32 raw macro test passed. State runs in
`[273492,283866)`, 10,374 cycles. Each MXM computes 4,096 cycles. The stage's
32,768 MEM write-vector issues exactly match its final FP32 State; there are
no intermediate writes or linker layout transports. The single existing R
generation remains shared by L, H and outgoing State.

- Worst-head hardware arithmetic NRMSE: `5.27846e-8`.
- Worst-head error relative to unquantized multiplication of device inputs:
  `0.00453148` (0.453%).
- Worst-head end-to-end State NRMSE versus sequential recurrence:
  `0.00557847` (0.558%); minimum cosine `0.999985777`.
- Chunk formula versus sequential FP64 reference: `1.05860e-7` NRMSE.
- Logical execution through State: 210,058 cycles; 14 diagnostic outputs;
  five real weight pages; zero host model operations.

The H2/K32/V32 and H4/K128/V128 standalone weighted-product tests passed, as
did the direct streaming head-matmul regression. Existing O's continuous MXM
runtime audit also passed in the full prefix run. Pipeline and runtime ICU CSV
artifacts are under `results/qwen3_5_0_8b_prefill_chunkwise_state`.

## Shared-D tile fusion results (2026-11-03)

The default real-weight seq32 run now fuses O and State. The combined runtime
range is `[269670,280017)`, 10,347 cycles. The previous two stages occupied
11,719 cycles including their 218-cycle boundary gap: a reduction of 1,372
cycles (11.7%). Upstream allocation also changes, so the total-program cycle
reduction must not be attributed entirely to the tile fusion.

Per hemisphere, D weight-load cycles fall from 2,048 to 1,024. The 32 D tiles
are transposed once each, while the 128 K transposes remain: 160 SXM tiles
instead of 192. Each MXM executes 5,120 compute cycles (1,024 O + 4,096 State).
The 40,960 MEM write-vector issues exactly match the two final FP32 results;
there are no intermediate spills or linker layout copies.

Both numerical results retain the previous errors: worst-head hardware
reference NRMSE `4.13566e-8` for O and `5.27846e-8` for State. End-to-end
errors are 1.465% and 0.558%, respectively. The full real-weight raw macro
test, fused H2/K32/V32 and H4/K128/V128 tests, and the separate O/State
regressions pass. Tests assert the shared load count and transpose count in
the actual runtime trace, in addition to checking both outputs.

The runtime program and CSVs are updated in the existing State result
directory. This remains a single chunk with zero incoming State.

## Dual-buffer tight schedule (2026-11-03)

The real-weight raw macro run passes with the two-tile schedule. The combined
O/State interval is now `[269670,276103)`: 6,433 cycles instead of 10,347
(3,914 fewer, 37.8%). Each MXM still computes 5,120 cycles, raising utilization
over the whole combined interval from 49.5% to 79.6%.

Each hemisphere has sixteen uninterrupted 320-cycle compute bursts, with
76 cycles between bursts. Idle time between the first and last compute falls
from 5,124 to 1,140 cycles. The trace shows 390 cycles of weight-load/compute
overlap per hemisphere, always on different weight buffers. Tests check both
this overlap and absence of same-buffer load/compute overlap.

D loads remain 1,024 cycles and SXM remains 160 tile transposes per side.
Final writes remain 40,960 vectors; no intermediate MEM traffic was added.
O and State numerical errors are unchanged from the shared-D results above.
The full prefix uses 202,295 logical cycles, five executable weight pages,
and zero host model operations. Fused random tests cover K128/V128 and an
odd number of tiles spanning different heads; the separate State regression
also passes. Existing result files and runtime ICU programs are refreshed.
