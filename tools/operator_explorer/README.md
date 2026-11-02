# LPU operator explorer

Standalone deliverable: `../../../lpu-operator-explorer.html` (workspace root).
No server, CDN, external scripts, lowering, or new hardware simulation required.

## Usage

- Select a Kernel operator or a FU primitive on the left.
- Teaching mode provides a stage-by-stage **illustrative schedule**, 4 staggered
  superlanes, event descriptions, and ideal numerical references.
- Runtime mode plays existing recorded CSV windows, preserving absolute cycles.
  Import another `runtime.pipeline.csv` to inspect its complete recorded range.
- Use previous/next cycle, next event, stage buttons, the slider, or the local
  timeline. Click a functional unit to filter the event table.
- Physical view embeds `ftlpu-lpu32.json`: each hemisphere has 13 MEM groups
  (4 slices/group), 16 SR boundary columns, 32 East streams above and 32 West
  streams below. Each cell represents 8 lanes in the selected superlane.
  Select the fabric, superlane, memory group, and zoom independently.
- VXM/MXM teaching routes now traverse the actual configured MEM/SR endpoint
  columns at one hop per cycle. Changing MEM group changes transport time.
  Other teaching phases have no asserted SR placement. This is not a silicon
  floorplan: internal ALU widths and functional-unit widths are visual only.
- Unknown runtime SR state is hatched, not drawn as empty. Explicit observation
  rows may use `resource=SR.E.W16` and `detail=column=6 tile=0 valid=1` in the
  normal pipeline CSV schema (E fabric, W16 stream). An explicit `valid=0` is
  dark. Missing observations remain unknown; state is not inferred from MEM
  commands. Existing pipeline files generally do not contain these rows.
- Export writes teaching events or original runtime patterns to JSON, with the
  provenance mode explicitly identified.

## Optional new SXM architecture

The SXM selector retains the existing local 8x8 + block-permute illustration and
adds an experimental 32x32 cross-superlane transpose. Click the adjacent button
to enter its transpose demonstration. The selector never rewrites runtime
traces or changes other operators' compiler lowering. This is an HTML-only
architecture model; neither CModel nor compiler gains a new hardware opcode.

The new model receives E16/E17 at column 14, preserves SL skew 0/1/2/3, delays
input SLs by 3/2/1/0, and uses two 32-bank 32x32 BF16 buffers. Column 15 output
E18/E19 is re-skewed by 0/1/2/3. Data capacity is 4096+96+96=4288 B. Three tiles
arrive with initiation interval 32 cycles; the first output starts at cycle 35
and the final SL of the final column of tile 2 exits at cycle 133. Payload
indices, overwrite prevention, FIFO capacity and drain are checked. Control,
ECC and extra SRAM/router pipeline latency are outside this model.

## Coverage and limits

The build reads all 18 definitions in `KernelOps.td`; startup fails if any lacks
a recipe. There are also 10 VXM arithmetic and 9 MEM/SXM/C2C transport examples.
This is coverage of operator **concepts**, not verification that every shape,
layout, or attribute combination lowers successfully.

Six embedded trace windows cover existing RMSNorm, L2Norm, Conv, gate-chain,
projection, and chunkwise GDN test results. Each window names its source. These
are historical test artifacts, not newly executed tests or evidence that the
current repository revision has passed. Gate-chain and chunkwise GDN traces
must not be mistaken for the corresponding simplified teaching recipe.

Teaching mode is intentionally not the compiler scheduler: it uses memory
roundtrips between stages and illustrative routing/reduction/matrix timings.
VXM elementary latency constants follow the inspected CModel sources. Values
are ideal JavaScript floating point, not bit-exact BF16/LUT hardware evaluation.
The GDN reference uses one head, 32-dimensional key/value, alpha=0.9, beta=0.6,
normalized q=k, v=input, and zero initial state; it performs four recurrent
updates. The teaching MXM/VXM stage labels express mathematical work, not a
claim about the exact recurrence lowering implementation.

Runtime mode does not invent SR positions, superlane identity, numerical values,
or completion latency missing from the CSV. Issue intervals are not presented
as internal functional-unit occupancy. It decodes repeat/repeat2d, skip_first,
and address induction using the same interval conventions as pipeline_viewer.

## Rebuild and checks

```powershell
python tools/operator_explorer/build.py
node tools/operator_explorer/test.cjs
```

`build.py` only bundles the HTML/JS and reads selected trace files. It does not
participate in lowering. `test.cjs` checks numerical invariants, skew, all recipes,
CSV parsing, pattern expansion against independent enumeration, and JS UI
bindings with a lightweight DOM stub. It is not browser visual QA.
