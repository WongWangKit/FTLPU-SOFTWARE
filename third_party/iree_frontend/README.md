# IREE Frontend Adapter

This directory documents the IREE frontend pieces that FTLPU intends to reuse.
It intentionally does not vendor the full IREE repository.

IREE is tracked as an external upstream:

```text
https://github.com/iree-org/iree
```

The FTLPU compiler should consume IREE-compatible common MLIR first:

```text
StableHLO / TOSA / Linalg MLIR -> FTLPU common IR -> FTLPU LPU backend
```

Use `tools/ftlpu-iree-import.py` to stage MLIR files or to invoke an installed
`iree-compile` toolchain.

When we later need in-process MLIR passes, add IREE/LLVM as external projects
or pinned submodules instead of copying source files by hand.
