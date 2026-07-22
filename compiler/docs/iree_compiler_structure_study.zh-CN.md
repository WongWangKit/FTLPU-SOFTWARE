# 面向 FTLPU 的 IREE 编译器结构研究

[English](iree_compiler_structure_study.md) | [简体中文](iree_compiler_structure_study.zh-CN.md)

本文记录 FTLPU 应从 IREE 学习的编译器结构，以及这些经验应如何重塑 FTLPU
编译器。重点不是复制 IREE 的代码规模，而是复制其工程边界：dialect、pass、
pipeline、target backend 和 serialization 是彼此独立的概念。

## IREE 做对了什么

### 工具入口保持轻量

`third_party/iree/tools/iree-compile-main.cc` 只调用
`ireeCompilerRunMain(argc, argv)`，真正的编译器位于稳定的 driver API 后面。

FTLPU 的 `ftlpu_opt` 也应保持轻量：

```text
main()
  -> CompilerSession
  -> CompilerInvocation
  -> 解析输入
  -> 构建所选 pipeline
  -> 运行 pass
  -> 输出 IR 或二进制
```

工具入口不应知道 StableHLO 如何变成 Schedule IR。

### Session 与 Invocation 分离

IREE compiler driver 使用 session 保存 dialect registration、target registry、
plugin activation 和 option 等全局状态；每个 invocation 只拥有一个 compilation
unit 和一次 pipeline 执行。

对于 FTLPU：

- `CompilerSession`：target registry、dialect registry、全局 option；
- `CompilerInvocation`：输入文件、输出文件、选择的 phase 范围、当前 module、
  diagnostics。

当编译器增加多个 compile command、compile-from/to phase，或被测试/runtime 工具
作为 library 调用时，这种分离非常重要。

### Pipeline 是具名的 Phase Builder

IREE 有明确 phase：

```text
Input
ABI
Preprocessing
GlobalOptimization
DispatchCreation
Flow
Stream
ExecutableSources
ExecutableConfigurations
ExecutableTargets
HAL
VM
End
```

每个 phase 都是 pass pipeline builder，并支持提前退出和从后续 phase 进入。
值得学习的是这种形态：

```cpp
void buildFtlpuTransformPipeline(
    TargetRegistry& targets,
    PipelineOptions options,
    OpPassManager& pm,
    FtlpuPipelinePhase compileFrom,
    FtlpuPipelinePhase compileTo);
```

FTLPU 应提供：

```text
Start
StableHLOInput
Kernel
Tensor
Stream
Schedule
ExecutableConfig
ExecutableBinary
End
```

测试可以明确停在 `--compile-to=stream`，或从 `--compile-from=tensor` 恢复，
而不是依赖临时的逗号分隔字符串。

### Dialect 拥有 IR，Transform 拥有 Pass

IREE 的 Stream dialect 不是一个名为 `passes.cpp` 的文件，而是分成：

```text
Dialect/Stream/IR
Dialect/Stream/Analysis
Dialect/Stream/Conversion
Dialect/Stream/Transforms
```

Stream pipeline 本身也有层次：

```text
stream.tensor.* -> stream.async.* -> stream.cmd.*
```

FTLPU 应采用类似边界：

```text
compiler/lib/Dialect/Kernel/IR
compiler/lib/Dialect/Kernel/Transforms
compiler/lib/Dialect/Tensor/IR
compiler/lib/Dialect/Tensor/Analysis
compiler/lib/Dialect/Tensor/Transforms
compiler/lib/Dialect/Stream/IR
compiler/lib/Dialect/Stream/Analysis
compiler/lib/Dialect/Stream/Transforms
compiler/lib/Dialect/Schedule/IR
compiler/lib/Dialect/Schedule/Analysis
compiler/lib/Dialect/Schedule/Transforms
```

旧的 `compiler/src/passes.cpp` 原型已经删除；pipeline glue 位于 `Pipelines/`，
IR 构造位于各层自己的 transform 中。

### Pass 可注册、具名且可测试

IREE 在 `Passes.td` 中定义 pass，每个 pass 使用单独文件实现，并注册 individual
pass 和 named pipeline。

即使尚未完全采用 TableGen，FTLPU 也应遵守相同契约：

- 每个 transformation 对应一个 pass class/function；
- pass name 稳定；
- 有 pass option struct；
- 声明 dependent dialect；
- 每个主要 lowering 边界之后运行 verifier；
- 每个 pass 有 lit/FileCheck 风格测试。

`stablehlo-to-kernel,kernel-to-tensor,tensor-to-stream` 作为原型可以由一个函数中
的字符串分支处理，但正式后端不能停留在这种结构。

### Target Backend 是接口

IREE target plugin 通过 `TargetBackend` 实现：

- 报告支持的类型；
- 注册依赖 dialect；
- 构建 configuration pass pipeline；
- 构建 translation pass pipeline；
- 构建 linking pass pipeline；
- 序列化 executable binary。

FTLPU 需要相同形态：

```cpp
class TargetBackend {
public:
  virtual std::string name() const = 0;
  virtual SupportedTypes supported_types() const = 0;
  virtual void build_configuration_pipeline(PassManager&) = 0;
  virtual void build_translation_pipeline(PassManager&) = 0;
  virtual void build_linking_pipeline(PassManager&) = 0;
  virtual LogicalResult serialize_executable(Module&, BinaryWriter&) = 0;
};
```

第一个 backend：

```text
Target/FtlpuCModel
  -> 配置 MEM/MXM/VXM/SXM/ICU 约束
  -> 将 Schedule IR lower 到 Queue IR
  -> 序列化 .ftlpu
  -> runtime 加载 ICU queue 并启动时钟
```

后续硬件 backend 可以共享大部分高层 pass，只替换 serialization 和 target
constraint。

## 对 FTLPU IR 分层的应用

当前 IR 分层方向合理，但每层需要更严格的所有权。

### Kernel IR

用途：把模型 op 映射到 LPU 功能单元。

```text
ftlpu.kernel.mxm_matmul
ftlpu.kernel.vxm_swiglu
ftlpu.kernel.ffn
```

该层不分配地址或 cycle。

### Tensor IR

用途：拥有 MEM object 和 placement。

Rank-5 地址模型属于该层：

```text
[device, hemisphere, slice, bank, word, byte]
shape [N, 2, 44, 2, 4096]
```

Allocator 和 placement analysis 应放在这里，不能隐藏在 print pass 中。

### Stream IR

用途：描述 FU/MEM 数据移动和 stream lifetime。

每条 stream 需要：

```text
stream_id
direction
producer endpoint
consumer endpoint
produce_cycle
consume_cycle
transport_latency
byte range
stream register mapping
```

该层适合建模 stream endpoint，但最终 issue cycle 仍由 Schedule IR 决定。

### Schedule IR

用途：决定指令 issue cycle 和 queue ordering。

它必须感知以下资源：

- MEM read/write port；
- MXM load 和 compute；
- VXM pipeline；
- SXM pipeline；
- transport latency；
- ICU queue availability；
- stream ID lifetime conflict。

Schedule IR 应接近二进制 queue section，但仍保持可读和可验证。

## 建议的 FTLPU 编译器目录结构

短期目标：

```text
compiler/
  include/ftlpu/compiler/
    API/
      compiler_session.hpp
      compiler_invocation.hpp
    Dialect/
      Kernel/
      Tensor/
      Stream/
      Schedule/
    Target/
      target_backend.hpp
      target_registry.hpp
      ftlpu_cmodel_target.hpp
    Pipelines/
      phases.hpp
      pipelines.hpp
    Support/
      diagnostics.hpp
      source_manager.hpp

  src/
    API/
    Dialect/
      Kernel/IR
      Kernel/Transforms
      Tensor/IR
      Tensor/Analysis
      Tensor/Transforms
      Stream/IR
      Stream/Analysis
      Stream/Transforms
      Schedule/IR
      Schedule/Analysis
      Schedule/Transforms
    Target/
      FtlpuCModel/
    Pipelines/
    Tools/
```

旧的 `target_model.hpp/cpp` 原型已经拆分：

- MEM address model -> `Dialect/Tensor/Analysis/MemoryLayout`；
- stream lifetime allocation -> `Dialect/Stream/Analysis/StreamAllocator`；
- resource/cycle scheduler -> `Dialect/Schedule/Analysis/ResourceScheduler`；
- CModel queue constraint -> `Target/FtlpuCModel`。

## 迁移计划

### 第 1 步：让原型的结构名副其实

保留已有文本 IR，但拆分源文件：

```text
src/Dialect/Tensor/Analysis/MemoryAllocator.cpp
src/Dialect/Stream/Analysis/StreamManager.cpp
src/Dialect/Schedule/Analysis/ResourceScheduler.cpp
src/Pipelines/Pipelines.cpp
```

`passes.cpp` 只保留小型适配层，不再承载实现。

### 第 2 步：引入真正的 Pipeline Phase

增加：

```cpp
enum class PipelinePhase {
  Start,
  StableHLOInput,
  Kernel,
  Tensor,
  Stream,
  Schedule,
  ExecutableBinary,
  End,
};
```

让 `ftlpu_opt` 支持：

```text
--compile-from=
--compile-to=
--pipeline=ftlpu-lpu-pipeline
```

### 第 3 步：增加 Verifier

每个边界之后验证：

- Kernel verifier：支持的 op/type/shape；
- Tensor verifier：地址范围不重叠且位于 MEM 内；
- Stream verifier：endpoint 存在、方向有效且 lifetime 不冲突；
- Schedule verifier：资源不重叠，所有 consume cycle 满足 producer cycle 加
  latency。

这是编译器质量开始清晰可见的位置。

### 第 4 步：增加 Target Backend 接口

将 CModel-specific constraint 移入：

```text
Target/FtlpuCModel/
```

通用 Schedule 层不应硬编码 CModel，而应向选中的 target 查询：

- FU inventory；
- MEM topology；
- transport latency table；
- queue mapping；
- binary serialization rule。

### 第 5 步：实现 MLIR Dialect

编译器现在直接使用 MLIR C++ API。第一个 production boundary 是
`stablehlo.dot_general -> ftlpu.kernel.matmul`，由已注册的 `KernelDialect`
和 `OperationPass<func::FuncOp>` 实现。后续 Tensor、Stream、Schedule 和
Executable 层必须遵循相同模式，不能重新引入仅文本的中间表示。

## 对旧代码的直接评价

旧编译器适合作为验证思路的原型，但结构较弱：

- `passes.cpp` 混合 parsing-level lowering、allocation、scheduling 和 printing；
- IR 仅为文本，pass 无法可靠分析和转换；
- 每层没有 verifier；
- 没有 pass registration 或 pipeline builder；
- target model 不可插拔；
- 测试只检查文本片段，而非结构不变量；
- runtime binary emission 尚未成为真正的 target backend serialization phase。

下一个严肃里程碑应是架构升级，而不是继续向旧文件添加 op。

## 当前重构状态

文本 `Module` 表示、自定义 parser、打印字符串的 pass 和对应 schedule 原型均已删除。
编译器当前基础为：

```text
Dialect/Kernel/IR/KernelDialect.cpp
Dialect/Kernel/Transforms/StableHloToKernel.cpp
Transforms/passes.hpp
tools/ftlpu_opt.cpp (MLIRContext + PassManager)
```

紧接着的工程里程碑是增加 Tensor dialect 和 allocation verifier，再以 MLIR pass
形式增加 Stream 与 Schedule dialect。
