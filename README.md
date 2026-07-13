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

The initial code provides a reusable `IcuProgram` container that records
scheduled MEM, MXM, and VXM events, converts idle cycle gaps into ICU NOPs, and
loads the resulting queues into the CModel `InstructionControlUnit`.

See [docs/low_level_ir_runtime_design.md](docs/low_level_ir_runtime_design.md)
for the proposed low-level IR, binary format, and runtime design.

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
