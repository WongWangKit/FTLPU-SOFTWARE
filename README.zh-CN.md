# FTLPU-SOFTWARE

[English](README.md) | [简体中文](README.zh-CN.md)

FTLPU 项目的软件栈。

本仓库用于承载驱动 FTLPU CModel 以及后续硬件目标的编译器、底层 IR、
二进制格式和 runtime。

## 当前重点

首个目标是对接 `FTLPU-CMODEL` 的 CModel ICU 边界：

```text
网络模型 -> MLIR 风格 lowering -> ftlpu.icu 时间线
  -> .ftlpu 二进制 -> runtime -> CModel InstructionControlUnit
```

当前代码提供了以下可复用路径：

```text
手写 LPU IR -> .ftlpu 二进制 -> runtime 解析器
  -> CModel InstructionControlUnit 队列 -> 时钟分发
```

`IcuProgram` 记录已经调度的 MEM、MXM 和 VXM 事件，将空闲 cycle 间隔转换为
ICU NOP，并能序列化为 `.ftlpu` 二进制队列格式。

底层 IR、二进制和 runtime 设计见
[runtime/docs/low_level_ir_runtime_design.zh-CN.md](runtime/docs/low_level_ir_runtime_design.zh-CN.md)。
当前 runtime 测试使用的手写 LPU IR 语法见
[runtime/examples/simple_dispatch.lpuir](runtime/examples/simple_dispatch.lpuir)。

## 编译器方向

编译器主路径为：

```text
ONNX / PyTorch / TensorFlow
  -> StableHLO 前端共同 IR
  -> FTLPU kernel IR
  -> FTLPU tensor IR
  -> FTLPU stream IR
  -> FTLPU schedule IR
  -> FTLPU command IR
  -> .ftlpu 二进制
```

StableHLO 是主要的前端/共同模型 IR 边界。IREE 作为参考编译器框架和对比工具，
用于研究 Flow、Stream、pass pipeline 和后端插件结构。

相关文档见
[compiler/docs/compiler_architecture.zh-CN.md](compiler/docs/compiler_architecture.zh-CN.md)、
[compiler/docs/iree_frontend_integration.zh-CN.md](compiler/docs/iree_frontend_integration.zh-CN.md)，
示例见
[compiler/examples/iree_frontend/simple_stablehlo.mlir](compiler/examples/iree_frontend/simple_stablehlo.mlir)。

仓库还通过 [third_party/iree](third_party/iree) git submodule 跟踪 IREE，
开发 FTLPU/LPU 后端时将其作为可检查的上游参考。

示例：

```powershell
python compiler/tools/ftlpu-iree-import.py `
  --input compiler/examples/iree_frontend/simple_stablehlo.mlir `
  --output build/simple_stablehlo.common.mlir `
  --input-format stablehlo
```

当前 ONNX 对比测试先经过 IREE ONNX importer：

```powershell
python compiler/tools/ftlpu-iree-import.py `
  --input model.onnx `
  --output build/model.mlir `
  --input-format onnx `
  --mode import-onnx `
  --onnx-opset-version 17
```

## 构建

CMake 默认假设 `FTLPU-CMODEL` 与本仓库位于同一父目录：

```powershell
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

如果 CModel 位于其他位置：

```powershell
cmake -S . -B build -DFTLPU_CMODEL_DIR=E:/path/to/FTLPU-CMODEL
```
