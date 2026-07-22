# IREE 参考集成方案

[English](iree_frontend_integration.md) | [简体中文](iree_frontend_integration.zh-CN.md)

本仓库通过 git submodule 跟踪 IREE，而不是手工复制部分源文件。IREE 是大型的
MLIR 编译器/runtime 项目，FTLPU 主要借鉴其编译器架构和参考 lowering 路径。

StableHLO 是 FTLPU 编译器主要的前端/共同模型 IR 边界。IREE 是参考框架和对比
工具，不是长期后端的必要依赖。

## 可复用的内容

IREE 中值得参考的概念包括：

- 从前端 MLIR dialect 转换输入；
- 硬件 lowering 之前的通用 tensor program IR；
- Flow 风格的 workload 划分；
- Stream 风格的放置和调度；
- 用插件添加树外 target backend 的模式。

FTLPU 编译器主路径应为：

```text
模型 / 前端计算图
  -> StableHLO
  -> FTLPU common tensor IR
  -> FTLPU stream IR
  -> FTLPU schedule IR
  -> .ftlpu 二进制
  -> CModel runtime
```

## 共同 IR 边界

第一版共同 IR 使用标准 MLIR 和 ML 模型 dialect 的文本格式：

- `builtin`、`func`、`arith`、`tensor`；
- 用 `stablehlo` 或 `tosa` 表达前端算子语义；
- legalization 后用 `linalg` 表达结构化计算。

LPU 后端不应直接消费框架专用计算图文件，而应消费 StableHLO 或 FTLPU 自有的
第一层 common tensor IR。

## 为什么使用 Submodule

不应把整个 IREE 源码树手工复制到 `FTLPU-SOFTWARE`：

- IREE 是包含大量 target 的完整编译器和 runtime 栈；
- 它带有很大的 LLVM/MLIR 依赖面；
- 其 import 工具有意与核心编译器树分离；
- 当前 FTLPU 只需要前端到共同 IR 的路径。

Submodule 保留上游历史，便于更新和检查完整编译器源码；在 LLVM/MLIR 等依赖接入
可选编译器构建之前，默认 FTLPU 构建仍可忽略 IREE。

## 仓库组成

当前仓库包括：

```text
compiler/tools/ftlpu-iree-import.py
compiler/examples/iree_frontend/simple_stablehlo.mlir
third_party/iree
third_party/iree_frontend/README.md
third_party/iree_frontend/iree_frontend_manifest.json
```

`compiler/tools/ftlpu-iree-import.py` 是第一版参考适配器，可以：

- 接收已经导入的 StableHLO/TOSA/Linalg MLIR；
- 对 ONNX protobuf 输入调用 IREE 的 `iree-import-onnx`；
- 把 StableHLO/TOSA/Linalg MLIR 复制到 FTLPU common-IR 暂存格式；
- 可选调用已安装的 `iree-compile` lowering 到某个 IREE stage；
- 在未安装 IREE 的环境中用 dry-run 生成命令。

对于 ONNX，安装支持 ONNX 的 IREE compiler package 后执行：

```powershell
python compiler/tools/ftlpu-iree-import.py `
  --input model.onnx `
  --output build/model.mlir `
  --input-format onnx `
  --mode import-onnx `
  --onnx-opset-version 17
```

直接继续到 IREE stage：

```powershell
python compiler/tools/ftlpu-iree-import.py `
  --input model.onnx `
  --output build/model.flow.mlir `
  --input-format onnx `
  --mode iree-compile `
  --iree-stage flow `
  --onnx-opset-version 17
```

## 里程碑

1. 保留 ONNX 到 IREE Flow 测试作为参考覆盖。
2. 以 StableHLO MLIR 作为主要前端/共同 IR。
3. 增加 LPU 约束所需的 shape/layout 验证。
4. 将 StableHLO matmul、elementwise 和 post-op 模式 lower 到 `ftlpu.stream`。
5. 将 `ftlpu.stream` 调度为 `ftlpu.schedule`。
6. 生成 `.ftlpu` 并通过 CModel runtime 执行。

## 需要研究的 IREE 源码区域

与 FTLPU 最相关的 IREE 区域包括：

- `compiler/src/iree/compiler/InputConversion`
- `compiler/src/iree/compiler/Dialect/Flow`
- `compiler/src/iree/compiler/Dialect/Stream`
- `compiler/src/iree/compiler/Pipelines`
- `compiler/plugins/input`
- `compiler/plugins/target`

LPU 后端应采用类似分层：输入规范化、共同 tensor IR、stream/schedule IR，
以及 target-specific 二进制生成。
