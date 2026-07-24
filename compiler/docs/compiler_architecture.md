# FTLPU Compiler Architecture

[English](compiler_architecture.md) | [简体中文](compiler_architecture.zh-CN.md)

This document fixes the compiler split for the first LPU backend work:

```text
ONNX / PyTorch / TensorFlow
  -> StableHLO
  -> FTLPU kernel IR
  -> FTLPU tensor IR
  -> FTLPU stream IR
  -> FTLPU schedule IR
  -> FTLPU command IR
  -> .ftlpu binary
  -> runtime / CModel / hardware
```

StableHLO is the primary frontend/common model IR boundary. IREE is a reference
compiler framework and a comparison tool, not the IR that the LPU backend must
permanently depend on.

## Responsibilities

### StableHLO Boundary

StableHLO should represent frontend model semantics after framework import:

- matmul and batched matmul as `stablehlo.dot_general`;
- convolution as `stablehlo.convolution`;
- elementwise and activation ops as StableHLO arithmetic;
- explicit tensor shapes, element types, and broadcast semantics.

This boundary keeps the LPU compiler independent from ONNX, PyTorch, and
TensorFlow graph quirks.

### FTLPU Kernel IR

The kernel IR is the first FTLPU-owned compiler layer. It should:

- normalize StableHLO ops into a small LPU-oriented kernel set;
- map each kernel to concrete LPU functional units such as MXM and VXM;
- represent reusable computation primitives including `matmul`, `batch_matmul`,
  `rope`, `softmax`, `transpose`, `swish`, and elementwise operations;
- validate static shapes and element types supported by the LPU;
- preserve quantization and layout metadata explicitly.

FFN and attention are graphs of these primitives in the public Kernel IR, not
opaque operations. Fusion is a later optimization decision. The current
Tensor backend still uses an internal `ftlpu-compose-kernel-plans`
compatibility pass while its allocation code is migrated to consume primitive
graphs directly; this compatibility operation is not emitted by the
StableHLO-to-Kernel pipeline.

### FTLPU Tensor IR

The tensor IR owns MEM allocation and tensor placement:

- assign activation, weight, intermediate, and output tensors to MEM ranges;
- choose MEM columns/banks and base addresses;
- describe tile plans that reference the selected kernels;
- keep layouts and element sizes explicit.

The implemented `ftlpu.tensor.matmul` and `ftlpu.tensor.swiglu` operations use
the physical rank-5 MEM address tuple
`[device, hemisphere, slice, bank, word, byte]`. One hemisphere contains 44
slices, each slice contains 2 banks, each bank contains 4096 16-byte words.
Allocation uses role-specific east-hemisphere SRAM row pools. Function inputs
are live at entry, each SSA tensor is kept through its last use, and expired
row ranges are merged and reused with a first-fit policy. Output storage is
allocated before current operands expire,
so a functional unit cannot overwrite an input that it is still consuming.

Matmul placement also carries CModel-facing row geometry. MXM weights use
`mxm_weight_striped` across MEM slices 0 through 15, activations use `vector`
placement on slice 32, and int32 results use four `int32_byte_planar` slices
40 through 43. Each placement records its slice list, base SRAM row,
instruction count, and signed address stride. The current CModel convention
uses a 16-row stride; weight Read commands walk the rows in reverse order.
FFN is represented as a primitive physical task graph:
two `ftlpu.tensor.matmul_task` projections, `ftlpu.tensor.swish_task`,
`ftlpu.tensor.elementwise_task`, and one down-projection matmul task. Each
matmul carries allocation lists for its operands and result. An empty result
list means that the value remains transient in an MXM accumulator or stream.
The elementwise result owns two allocations so the hidden tensor can be
materialized independently in the west and east hemispheres; the down
projection consumes both allocations. Addresses, placements, byte sizes, and
quantization parameters are therefore attached to the primitive operation
that uses them instead of being hidden in a compound `ftlpu.tensor.ffn`.

### FTLPU Stream IR

The stream IR maps MEM-resident tensor tiles onto LPU streams:

- every stream has a source and sink, such as `MEM:A -> MXM0:lhs`;
- every stream records a direction-local contiguous stream range and stream
  register id;
- every stream records start address, byte count, and endpoint functional unit;
- MXM/VXM post-processing streams are explicit instead of implied by kernels.
- streams are long logical vectors when the instruction naturally traverses the
  south-to-north tile chain; the compiler should not expand those 20 physical
  tiles into separate IR ops.

This layer is conceptually similar to IREE Stream, but it is FTLPU-owned and
should model the real LPU data movement directly.

The implemented `ftlpu.stream.route` operation records a direction-local
contiguous `[stream_base, stream_base + stream_count)` range, a physical
MEM-boundary `register_id`, source/destination
endpoints, explicit source/destination functional-unit ids, MEM address, byte
count, and fixed transport latency. MEM endpoints use unit id `-1`; MXM
endpoints identify MXM0 or MXM1. `ftlpu.stream.matmul` also records the selected
MXM unit and weight-buffer id. Each
`ftlpu.tensor.matmul` becomes two eastbound MEM-to-MXM
routes, one `ftlpu.stream.matmul`, and one westbound MXM-to-MEM route. The MXM
weight route owns 16 streams during IW load, activation compute consumes one
stream, and each int32 result owns four consecutive byte streams.

SSA operation order is used internally by the stream allocator to permit safe
stream-range reuse, but no logical lifetime stage is emitted in Stream IR.
Exact issue and arrival cycles are assigned by Stream-to-Schedule.

FFN is also primitive in public Stream IR. Gate and up are separate
`ftlpu.stream.matmul_task` operations fed by one shared activation route and
independent dequantized weight routes. `swish_task` and an
`elementwise_task` with `kind = "multiply"` expose the VXM dataflow and the
two west/east hidden allocations. Two hidden MEM-to-MXM routes then feed
separate down matmul tasks. A final `elementwise_task` with
`kind = "add_quant"` merges their partials and owns the result allocation.
Each matmul task explicitly records its MXM unit, weight buffer, and result
stream range. The current Stream-to-Schedule implementation composes this
graph into its established scheduling descriptor internally; the compound
operation is not emitted by the StableHLO-to-Stream pipeline.

### LPU Target Model

`LPUTargetModel` is the single compiler-side source for MEM geometry, 32
streams per direction, the 64 packed selectors, 12 MEM-boundary register
columns, the additional SXM-to-MXM column, MXM dimensions and throughput,
supported endpoint routes, register mapping, and transport latency. A latency
means producer issue to consumer visibility, including the CModel tick phase;
lowering passes must query the model instead of embedding compensating cycles.

The target is configurable during architecture exploration:

```text
ftlpu_opt --target-config compiler/examples/targets/exploration_40_streams.json ...
```

The JSON file may override fields in the `memory`, `streams`, and `throughput`
sections. Unspecified fields retain the default CModel-compatible values. The
resolved configuration is serialized into the module as the `ftlpu.target`
dictionary, so every intermediate MLIR file carries the parameters needed to
reproduce later lowering. Kernel-to-Tensor, Tensor-to-Stream, and
Stream-to-Schedule recover the model from that attribute. Configuration
validation rejects non-positive dimensions, stream widths that exceed the
directional fabric, invalid MEM slice bases, and incompatible tile geometry.

`exploration_40_streams.json` is a regression configuration with 40 streams
per direction and non-default MXM/VXM latencies. It lowers the generic W8A16
FFN through Schedule IR and must produce a schedule different from the
default target. Non-default Command/Binary execution additionally requires a
runtime and CModel built for the same target ABI; silently executing such a
binary on the default target is not valid.

### FTLPU Schedule IR

The schedule IR is the low-level scheduled target:

- explicit cycle numbers;
- explicit MEM/MXM/VXM queues;
- explicit NOP and repeat opportunities;
- stage-level operations first, such as `mem_read_weight`,
  `mem_read_activation`, `mxm_load`, `mxm_compute`, and `mem_write`;
  queue command expansion belongs in the Command lowering layer.

The implemented Schedule dialect uses `ftlpu.schedule.mem_read`,
`ftlpu.schedule.mxm_load`, `ftlpu.schedule.mxm_compute`, and
`ftlpu.schedule.mem_write`. Every operation carries an ICU issue `cycle` and
`duration`. The scheduler reserves each MEM slice queue, each direction-local
stream, the selected MXM unit's load/compute queues, and its selected weight
buffer. `mxm_load` and `mxm_compute` preserve both ids explicitly. Fixed
transport latency is included between producer and consumer windows, and SSA
consumers cannot read a value before its producer's MEM write completes.

For the current 320x320 int8 GEMM, the CModel-aligned baseline is: weight MEM
reads begin at cycles 5 through 8 according to their MEM boundary, IW runs at
`[18,38)`, activation MEM Read on `E16` runs at `[33,353)`, Compute issue runs
at `[38,358)`, the first MXM result appears at cycle 57, and four int32
byte-plane MEM writes run at `[59,379)`. The output stream remains occupied for
the full 339-cycle MXM result window while the 20 physical tile rows drain.

For the 160x320x640x320 FFN fixture, shared activation startup follows the
CModel path in three segments (`E16`, `E30`, `E0`) for both hidden passes.
The correctness-first schedule computes pass 0 at cycles 58/73/77 and pass 1
at 318/333/337. Twelve stream-register transport cycles separate the first MXM
result from VXM consumption. The two SwiGLU pipelines start at cycles 89 and
349 and write 160 i8 rows to slices 40 and 41 at cycles 110 and 370. Down
weights load into buffer 0 at cycles 538 and 558; both MXMs compute at cycle
590 from activation streams `E0` and `E16`. The six-stage VXM AddQuant starts
at cycle 621 and writes the final slice-42 result at cycle 638. This schedule is
deliberately serial; buffer-1 ping-pong is a separate performance optimization.

### FTLPU Command IR

Command IR is the stable compiler/runtime boundary. The
`ftlpu-schedule-to-command` pass removes the Schedule SSA graph and emits
`ftlpu.command.binding`, `ftlpu.command.mem`, `ftlpu.command.mxm`, and
`ftlpu.command.vxm`
operations. Bindings describe input/output index, shape, element type, byte
size, and physical placement. Results are represented by bindings plus their
physical MEM write commands instead of an SSA tensor return value.

A command maps one-to-one to an ICU queue's first instruction plus Repeat:
absolute first cycle, queue index, opcode, stream selectors, repeat count and
interval, and MEM address stride. A 320x320 GEMM produces 16 weight MEM
commands, one activation MEM command, four result MEM commands, one IW command,
and one Compute command. VXM commands carry ALU opcode, typed stream/ALU/immediate
operands, cast target, output stream, and repeat metadata. Command operations
are explicitly side-effecting so
MLIR canonicalization cannot remove hardware work. Binary emission groups the
flat command stream by queue and derives queue-local NOP counts from absolute
cycles without reconstructing scheduling decisions.

`ftlpu-translate` serializes this layer to `.ftlpu` binary version 2. The
runtime stages bound inputs into SRAM, loads every ICU queue before clocks
start, advances `TspSliceSystem::tick()`, and reconstructs bound outputs. The
320x320 regression compares all 102,400 int32 results against a CPU GEMM.

### Test Artifacts

Compiler tests keep generated IR under a directory named after the test:

```text
build-ftlpu-vs2026/compiler/ftlpu_lower/<test-name>/
```

The complete FFN lowering test preserves every visible boundary as
`ffn.stablehlo.mlir`, `ffn.kernel.mlir`, `ffn.tensor.mlir`,
`ffn.stream.mlir`, `ffn.schedule.mlir`, and `ffn.commands.mlir`. Runtime tests
use a different directory and additionally produce `ffn.ftlpu`, so parallel or
repeated tests cannot overwrite another test's artifacts.

## Role Of IREE

IREE remains useful because it shows mature MLIR compiler engineering patterns:

- frontend import boundaries;
- Flow dispatch formation;
- Stream-style scheduling and resource modeling;
- pass pipeline organization;
- out-of-tree target/backend plugin structure.

The repository may keep tests that use IREE as a reference path:

```text
ONNX -> IREE importer -> IREE Flow IR
```

Those tests are comparison and sanity tests. The main backend path should be:

```text
StableHLO -> FTLPU kernel IR -> FTLPU tensor IR -> FTLPU stream IR
          -> FTLPU schedule IR -> FTLPU command IR
```

## Immediate Milestones

1. Keep ONNX-to-IREE Flow tests as reference coverage.
2. Add StableHLO fixture tests for matmul and elementwise ops.
3. Define textual `ftlpu.kernel`, `ftlpu.tensor`, and `ftlpu.stream` examples.
4. Lower StableHLO matmul to MXM kernel, MEM allocation, and explicit stream form.
5. Lower StableHLO/FTLPU Kernel FFN SwiGLU to the CModel-aligned schedule shape.
6. Lower FTLPU stream/schedule programs to `.ftlpu`.
