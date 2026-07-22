# MLIR 编译器构建

[English](mlir_build.md) | [简体中文](mlir_build.zh-CN.md)

`ftlpu_compiler` 直接使用 C++ MLIR API。它不会用文本解析器解析 StableHLO，
也不会把编译器状态保存在自定义 C++ 结构体中。

FTLPU 自行固定 LLVM/MLIR 和 StableHLO submodule 的版本。IREE 仅作为参考编译器，
不在 FTLPU 编译器依赖链中。首次使用时初始化直接依赖：

```powershell
git submodule update --init --recursive third_party/llvm-project third_party/stablehlo
```

先构建 MLIR，再使用其 package 目录配置 FTLPU：

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

固定的 LLVM revision 使用 VS2019 19.29 构建 `mlir-tblgen` 会失败。请使用
Visual Studio 2026，或带 MSVC 19.36 及以上工具集的新版 VS2022。
应在 VS2026 developer command prompt 中执行这些命令；Ninja 可以避免 CMake
generator 与 Visual Studio IDE 版本耦合。

第一个可用 pass 是 `ftlpu-stablehlo-to-kernel`。它以 MLIR 方式解析 StableHLO，
把 `stablehlo.dot_general` 重写成 `ftlpu.kernel.matmul`，并保留 SSA tensor
操作数和结果。Tensor、Stream、Schedule 和可执行 dialect 都通过同一个 pass
manager 上的 MLIR pass 实现。
