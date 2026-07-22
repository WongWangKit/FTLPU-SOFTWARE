# MLIR Compiler Build

[English](mlir_build.md) | [简体中文](mlir_build.zh-CN.md)

`ftlpu_compiler` uses the C++ MLIR API directly. It does not parse StableHLO
with a text parser or store compiler state in custom C++ structs.

FTLPU owns version-pinned LLVM/MLIR and StableHLO submodules. IREE remains a
reference compiler only and is not in the FTLPU compiler dependency chain.
Initialize the direct submodules once:

```powershell
git submodule update --init --recursive third_party/llvm-project third_party/stablehlo
```

Build MLIR, then configure FTLPU with its package directory:

```powershell
cmake -S third_party/llvm-project/llvm -B build-mlir-vs2026 -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DLLVM_ENABLE_PROJECTS=mlir -DLLVM_TARGETS_TO_BUILD=Native `
  -DLLVM_ENABLE_RTTI=ON -DLLVM_ENABLE_EH=ON `
  -DLLVM_INCLUDE_TESTS=OFF -DLLVM_INCLUDE_BENCHMARKS=OFF
cmake --build build-mlir-vs2026 --target `
  MLIRIR MLIRParser MLIRPass MLIRFuncDialect MLIRSupport `
  MLIRQuantDialect MLIRShapeDialect MLIRSparseTensorDialect `
  MLIRTensorDialect MLIRAffineDialect MLIRMemRefDialect `
  MLIRMemOpInterfaces MLIRMemorySlotInterfaces MLIRArithUtils `
  MLIRComplexDialect MLIRDestinationStyleOpInterface `
  MLIRParallelCombiningOpInterface MLIRRuntimeVerifiableOpInterface `
  MLIRValueBoundsOpInterface -j 8
cmake -S . -B build-ftlpu-vs2026 -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DFTLPU_MLIR_DIR="$PWD/build-mlir-vs2026/lib/cmake/mlir"
cmake --build build-ftlpu-vs2026 --target ftlpu_opt -j 8
ctest --test-dir build-ftlpu-vs2026 --output-on-failure `
  -R stablehlo_matmul_320_to_ftlpu_kernel_mlir_test
```

The pinned LLVM revision fails in `mlir-tblgen` with VS2019 19.29. Use Visual
Studio 2026, or a current VS2022 toolset with MSVC 19.36 or newer. Run these
commands from a VS2026 developer command prompt; Ninja avoids CMake generator
version coupling to the Visual Studio IDE.

The first available pass is `ftlpu-stablehlo-to-kernel`; it parses StableHLO as
MLIR, rewrites `stablehlo.dot_general` to `ftlpu.kernel.matmul`, and preserves
SSA tensor operands and results. Tensor, stream, schedule, and executable
dialects will be added as MLIR passes on this same pass manager.
