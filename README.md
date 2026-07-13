# FTLPU-SOFTWARE

Software stack for the FTLPU project.

This repository is intended to host the compiler, low-level IR, binary format,
and runtime code that drives the FTLPU CModel and later hardware targets.

## Current Focus

The first target is the CModel ICU boundary in `FTLPU-CMODEL`:

```text
network model -> MLIR-style lowering -> ftlpu.icu timeline
  -> .ftlpu binary -> runtime -> CModel InstructionControlUnit
```

The initial code provides a reusable path for:

```text
hand-written LPU IR -> .ftlpu binary -> runtime parser
  -> CModel InstructionControlUnit queues -> clock dispatch
```

`IcuProgram` records scheduled MEM, MXM, and VXM events, converts idle cycle
gaps into ICU NOPs, and can be serialized into the first `.ftlpu` binary queue
format.

See [runtime/docs/low_level_ir_runtime_design.md](runtime/docs/low_level_ir_runtime_design.md)
for the proposed low-level IR, binary format, and runtime design.

See [runtime/examples/simple_dispatch.lpuir](runtime/examples/simple_dispatch.lpuir) for the
current hand-written LPU IR syntax used by the runtime test.

## IREE Frontend

The first IREE-facing frontend adapter is intentionally lightweight:

```text
StableHLO / TOSA / Linalg MLIR
  -> compiler/tools/ftlpu-iree-import.py
  -> FTLPU common IR staging file
  -> future FTLPU stream / ICU lowering
```

See [compiler/docs/iree_frontend_integration.md](compiler/docs/iree_frontend_integration.md) and
[compiler/examples/iree_frontend/simple_stablehlo.mlir](compiler/examples/iree_frontend/simple_stablehlo.mlir).

The repository also tracks IREE as a git submodule under
[third_party/iree](third_party/iree). We will use its input conversion, Flow,
Stream, Util, and pipeline code as the frontend/compiler base while developing
the FTLPU/LPU backend.

Example:

```powershell
python compiler/tools/ftlpu-iree-import.py `
  --input compiler/examples/iree_frontend/simple_stablehlo.mlir `
  --output build/simple_stablehlo.common.mlir `
  --input-format stablehlo
```

ONNX inputs go through IREE's ONNX importer first:

```powershell
python compiler/tools/ftlpu-iree-import.py `
  --input model.onnx `
  --output build/model.mlir `
  --input-format onnx `
  --mode import-onnx `
  --onnx-opset-version 17
```

## Build

By default CMake expects `FTLPU-CMODEL` to sit next to this repository:

```powershell
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

If the CModel is elsewhere:

```powershell
cmake -S . -B build -DFTLPU_CMODEL_DIR=E:/path/to/FTLPU-CMODEL
```
