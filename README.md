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

See [docs/low_level_ir_runtime_design.md](docs/low_level_ir_runtime_design.md)
for the proposed low-level IR, binary format, and runtime design.

See [examples/simple_dispatch.lpuir](examples/simple_dispatch.lpuir) for the
current hand-written LPU IR syntax used by the runtime test.

## IREE Frontend

The first IREE-facing frontend adapter is intentionally lightweight:

```text
StableHLO / TOSA / Linalg MLIR
  -> tools/ftlpu-iree-import.py
  -> FTLPU common IR staging file
  -> future FTLPU stream / ICU lowering
```

See [docs/iree_frontend_integration.md](docs/iree_frontend_integration.md) and
[examples/iree_frontend/simple_stablehlo.mlir](examples/iree_frontend/simple_stablehlo.mlir).

Example:

```powershell
python tools/ftlpu-iree-import.py `
  --input examples/iree_frontend/simple_stablehlo.mlir `
  --output build/simple_stablehlo.common.mlir `
  --input-format stablehlo
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
