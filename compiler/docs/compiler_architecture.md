# FTLPU Compiler Architecture

This document fixes the compiler split for the first LPU backend work:

```text
ONNX / PyTorch / TensorFlow
  -> StableHLO
  -> FTLPU common tensor IR
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

### FTLPU Common Tensor IR

The common tensor IR is the first FTLPU-owned compiler layer. It should:

- normalize StableHLO ops into a small LPU-oriented operator set;
- validate static shapes and element types supported by the LPU;
- preserve quantization and layout metadata explicitly;
- prepare matmul, elementwise, and post-op fusion candidates.

### FTLPU Stream IR

The stream IR maps tensor work onto LPU resources:

- MEM column reads and writes;
- MXM input, weight, compute, and output streams;
- VXM lanes and post-processing streams;
- SRAM addresses, tile sizes, and scheduling constraints.

This layer is conceptually similar to IREE Stream, but it is FTLPU-owned and
should model the real LPU data movement directly.

### FTLPU Schedule IR

The schedule IR is the low-level scheduled target:

- explicit cycle numbers;
- explicit MEM/MXM/VXM queues;
- explicit NOP and repeat opportunities;
- direct serialization into `.ftlpu` queue sections.

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
StableHLO -> FTLPU common tensor IR -> FTLPU stream IR -> FTLPU schedule IR
```

## Immediate Milestones

1. Keep ONNX-to-IREE Flow tests as reference coverage.
2. Add StableHLO fixture tests for matmul and elementwise ops.
3. Define textual `ftlpu.tensor` or `ftlpu.common` examples.
4. Lower StableHLO matmul to the first FTLPU stream form.
5. Lower StableHLO/FTLPU Tensor FFN SwiGLU to the CModel-aligned schedule shape.
6. Lower FTLPU stream/schedule programs to `.ftlpu`.
