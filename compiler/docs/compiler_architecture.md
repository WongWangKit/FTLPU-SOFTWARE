# FTLPU Compiler Architecture

This document fixes the compiler split for the first LPU backend work:

```text
ONNX / PyTorch / TensorFlow
  -> StableHLO
  -> FTLPU kernel IR
  -> FTLPU tensor IR
  -> FTLPU stream IR
  -> FTLPU schedule IR
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
- represent fused kernels such as SwiGLU as unit-level compositions, for
  example `MXM + MXM + VXM + MXM`;
- validate static shapes and element types supported by the LPU;
- preserve quantization and layout metadata explicitly.

### FTLPU Tensor IR

The tensor IR owns MEM allocation and tensor placement:

- assign activation, weight, intermediate, and output tensors to MEM ranges;
- choose MEM columns/banks and base addresses;
- describe tile plans that reference the selected kernels;
- keep layouts and element sizes explicit.

### FTLPU Stream IR

The stream IR maps MEM-resident tensor tiles onto LPU streams:

- every stream has a source and sink, such as `MEM:A -> MXM0:lhs`;
- every stream records stream id and stream register id;
- every stream records start address, byte count, and endpoint functional unit;
- MXM/VXM post-processing streams are explicit instead of implied by kernels.
- streams are long logical vectors when the instruction naturally traverses the
  south-to-north tile chain; the compiler should not expand those 20 physical
  tiles into separate IR ops.

This layer is conceptually similar to IREE Stream, but it is FTLPU-owned and
should model the real LPU data movement directly.

### FTLPU Schedule IR

The schedule IR is the low-level scheduled target:

- explicit cycle numbers;
- explicit MEM/MXM/VXM queues;
- explicit NOP and repeat opportunities;
- direct serialization into `.ftlpu` queue sections.
- stage-level operations first, such as `mem_read_weight`,
  `mem_read_activation`, `mxm_load`, `mxm_compute`, and `mem_write`;
  tile-by-tile expansion belongs in the later queue/binary emission layer.

The runtime consumes this layer through the binary format and loads the CModel
ICU queues before clocks start.

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
StableHLO -> FTLPU kernel IR -> FTLPU tensor IR -> FTLPU stream IR -> FTLPU schedule IR
```

## Immediate Milestones

1. Keep ONNX-to-IREE Flow tests as reference coverage.
2. Add StableHLO fixture tests for matmul and elementwise ops.
3. Define textual `ftlpu.kernel`, `ftlpu.tensor`, and `ftlpu.stream` examples.
4. Lower StableHLO matmul to MXM kernel, MEM allocation, and explicit stream form.
5. Lower StableHLO/FTLPU Kernel FFN SwiGLU to the CModel-aligned schedule shape.
6. Lower FTLPU stream/schedule programs to `.ftlpu`.
