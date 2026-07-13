# IREE Frontend Integration Plan

This repository should reuse IREE's frontend architecture without vendoring the
entire IREE tree. IREE is a large MLIR-based compiler/runtime project, and the
parts useful for FTLPU are the model import boundary and the common MLIR
abstractions before target-specific codegen.

## What To Reuse

The useful IREE-side concepts are:

- Input conversion from frontend MLIR dialects.
- Common tensor program IR before hardware lowering.
- Flow-style workload partitioning.
- Stream-style placement and scheduling ideas.
- The plugin pattern for adding an out-of-tree target backend.

The first FTLPU compiler path should therefore be:

```text
model / frontend MLIR
  -> IREE-compatible common MLIR
  -> FTLPU frontend-normalized IR
  -> FTLPU stream IR
  -> FTLPU ICU timeline IR
  -> .ftlpu binary
  -> CModel runtime
```

## Common IR Boundary

For the first implementation, the common IR boundary is textual MLIR using
standard MLIR plus ML model dialects:

- `builtin`, `func`, `arith`, `tensor`
- `stablehlo` or `tosa` for frontend operator semantics
- `linalg` for structured compute after legalization

The LPU backend should not consume framework-specific graph files directly.
Instead, it should consume this normalized MLIR boundary.

## Why Not Vendor All IREE

Do not copy the whole IREE source tree into `FTLPU-SOFTWARE`:

- IREE is a full compiler and runtime stack with many targets.
- It carries a large LLVM/MLIR dependency surface.
- Its import tools are intentionally separate from the core compiler tree.
- FTLPU only needs the frontend-to-common-IR path now.

The better short-term shape is an adapter that can call an installed IREE
compiler toolchain or accept already-imported MLIR. Later, if we build a real
MLIR C++ compiler binary, we can link against LLVM/MLIR/IREE as external
projects.

## Repository Pieces

The current repository now includes:

```text
tools/ftlpu-iree-import.py
examples/iree_frontend/simple_stablehlo.mlir
third_party/iree_frontend/README.md
third_party/iree_frontend/iree_frontend_manifest.json
```

`tools/ftlpu-iree-import.py` is the first frontend adapter. It can:

- accept already-imported StableHLO/TOSA/Linalg MLIR;
- copy it into the FTLPU common-IR staging format;
- optionally invoke an installed `iree-compile` to lower to an IREE stage;
- generate a command in dry-run mode for environments without IREE installed.

## Milestones

1. Accept StableHLO/TOSA/Linalg MLIR as common IR.
2. Add shape/layout validation for LPU constraints.
3. Lower common IR matmul/elementwise/post-op patterns to `ftlpu.stream`.
4. Schedule `ftlpu.stream` to `ftlpu.icu`.
5. Emit `.ftlpu` binary and run through CModel runtime.
6. Replace command-line adapter with linked MLIR/IREE passes when the backend
   becomes mature enough.

## IREE Source Areas To Study

The IREE areas most relevant to FTLPU are:

- `compiler/src/iree/compiler/InputConversion`
- `compiler/src/iree/compiler/Dialect/Flow`
- `compiler/src/iree/compiler/Dialect/Stream`
- `compiler/src/iree/compiler/Pipelines`
- `compiler/plugins/input`
- `compiler/plugins/target`

The LPU backend should mirror this split: input normalization, common tensor
IR, stream/schedule IR, and target-specific binary emission.
