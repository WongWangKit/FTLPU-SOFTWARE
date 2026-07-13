# IREE Reference Integration Plan

This repository tracks IREE as a git submodule instead of copying selected
source files by hand. IREE is a large MLIR-based compiler/runtime project, and
the parts useful for FTLPU are its compiler architecture patterns and reference
lowering path.

StableHLO is the primary frontend/common model IR boundary for the FTLPU
compiler. IREE is a reference framework and comparison tool, not the required
long-term backend dependency.

## What To Reuse

The useful IREE-side concepts are:

- Input conversion from frontend MLIR dialects.
- Common tensor program IR before hardware lowering.
- Flow-style workload partitioning.
- Stream-style placement and scheduling ideas.
- The plugin pattern for adding an out-of-tree target backend.

The primary FTLPU compiler path should be:

```text
model / frontend graph
  -> StableHLO
  -> FTLPU common tensor IR
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
Instead, it should consume StableHLO or the first FTLPU-owned common tensor IR.

## Why A Submodule

Do not manually copy the whole IREE source tree into `FTLPU-SOFTWARE`:

- IREE is a full compiler and runtime stack with many targets.
- It carries a large LLVM/MLIR dependency surface.
- Its import tools are intentionally separate from the core compiler tree.
- FTLPU only needs the frontend-to-common-IR path now.

The submodule keeps upstream history and makes it easy to update or inspect the
full compiler tree while still letting the default FTLPU build ignore IREE until
LLVM/MLIR and other dependencies are wired into an optional compiler build.

## Repository Pieces

The current repository now includes:

```text
compiler/tools/ftlpu-iree-import.py
compiler/examples/iree_frontend/simple_stablehlo.mlir
third_party/iree
third_party/iree_frontend/README.md
third_party/iree_frontend/iree_frontend_manifest.json
```

`compiler/tools/ftlpu-iree-import.py` is the first reference adapter. It can:

- accept already-imported StableHLO/TOSA/Linalg MLIR;
- invoke IREE's `iree-import-onnx` tool for ONNX protobuf inputs;
- copy StableHLO/TOSA/Linalg MLIR into the FTLPU common-IR staging format;
- optionally invoke an installed `iree-compile` to lower to an IREE stage;
- generate a command in dry-run mode for environments without IREE installed.

For ONNX, install an IREE compiler package with ONNX support, then import:

```powershell
python compiler/tools/ftlpu-iree-import.py `
  --input model.onnx `
  --output build/model.mlir `
  --input-format onnx `
  --mode import-onnx `
  --onnx-opset-version 17
```

To continue directly into an IREE stage:

```powershell
python compiler/tools/ftlpu-iree-import.py `
  --input model.onnx `
  --output build/model.flow.mlir `
  --input-format onnx `
  --mode iree-compile `
  --iree-stage flow `
  --onnx-opset-version 17
```

## Milestones

1. Keep ONNX-to-IREE Flow tests as reference coverage.
2. Accept StableHLO MLIR as the primary frontend/common IR.
3. Add shape/layout validation for LPU constraints.
4. Lower StableHLO matmul/elementwise/post-op patterns to `ftlpu.stream`.
5. Schedule `ftlpu.stream` to `ftlpu.icu`.
6. Emit `.ftlpu` binary and run through CModel runtime.

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
