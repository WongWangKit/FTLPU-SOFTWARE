# FTLPU Command 二进制与 Runtime

[English](low_level_ir_runtime_design.md) | [简体中文](low_level_ir_runtime_design.zh-CN.md)

本文定义当前编译器与 runtime 在 CModel ICU 边界上的契约。

## Lowering 边界

```text
StableHLO
  -> ftlpu.kernel
  -> ftlpu.tensor
  -> ftlpu.stream
  -> ftlpu.schedule
  -> ftlpu.command
  -> .ftlpu 二进制
  -> CModelRuntime
  -> CModel ICU 和 TspSliceSystem::tick()
```

Schedule IR 负责资源预留和精确绝对 cycle。Command IR 是可序列化边界：其中包含
runtime binding 和扁平 MEM/MXM/VXM 队列命令。`ftlpu-translate` 按队列分组命令、
按 cycle 排序，并将空闲间隔和规则序列编码成 ICU NOP 与 Repeat 命令。

## Runtime Binding

每个 `ftlpu.command.binding` 和 `BinaryBinding` 记录：

- 输入或输出编号；
- 元素类型和逻辑 shape；
- 字节数；
- 物理 layout 和 MEM slice 列表；
- 基础 SRAM row、指令数量和有符号地址 stride。

已实现的 layout：

| Layout | 用途 | 放置方式 |
| --- | --- | --- |
| `Vector` | int8 activation | 单个 MEM slice，320 字节向量按 row-major 放置 |
| `MxmWeightStriped` | int8 MXM weight | column 条带分布到 16 个 MEM slice |
| `Int32BytePlanar` | int32 result | 四个 MEM slice，每个 slice 保存一个 byte plane |

`CModelRuntime::upload_input()` 验证 binding 并把字节放入声明的 SRAM 位置；
`download_output()` 从物理 byte plane 还原逻辑 tensor。

## 二进制版本 2

所有标量字段均为 little-endian。文件头为：

```text
magic[8] = "FTLPUB01"
u32 version = 2
u64 max_cycle
u32 queue_count
u32 binding_count
BinaryBinding[binding_count]
QueueProgram[queue_count]
```

一个 binding 包含固定元数据，随后是 `rank` 个 64 位维度和 `slice_count` 个
16 位 MEM slice 标识。队列记录包含队列类型、队列编号、命令数和编码后的
`QueueCommand`。每个命令保存 ICU command word、指令类型、word 数量和三个
32 位指令 word。reader 保持版本 1 兼容；版本 1 没有 binding count 和记录。

指令 word 直接复用 CModel codec。MEM 和 MXM 使用一个 word；VXM 使用完整的
三 word 容器编码 ALU opcode、两个带类型操作数、cast target、输出 stream、
scale 和 zero point。ICU 队列 NOP 和 Repeat 命令保留编译器生成的绝对调度。

## 执行契约

Runtime 按以下步骤执行：

1. 读取并验证 `.ftlpu` 元数据和队列。
2. 将所有输入 binding 上传到 CModel SRAM。
3. 填充对应 ICU 队列，包括 NOP 和 Repeat。
4. 通过权威的 `TspSliceSystem::tick()` 循环推进程序并排空 cycle。
5. 从 SRAM 下载输出 binding。

Runtime 不直接调用 MEM 或 MXM datapath，因此未来硬件 runtime 可以继续使用相同
的队列和时序契约。

## 320x320 GEMM 回归

端到端回归把 StableHLO matmul 编译为 Command IR 和二进制，上传两个
320x320 int8 矩阵，在 CModel 上执行，下载 320x320 int32 结果，并将全部
102,400 个元素与 CPU reference 比较。

当前命令时间线：

```text
cycle 5..8   MEM weight read 在 E0..E15 启动
cycle 18     MXM IW 启动，重复 20 次
cycle 33     MEM activation read 在 E16 启动，重复 320 次
cycle 38     MXM Compute 启动，重复 320 次
cycle 59     四个 MEM result write 启动，重复 320 次
```

Transport latency 定义为从 producer 发射到 consumer 可见，并包含 CModel tick
phase。例如 slice 32 到 MXM 边界为 5 cycle，MXM 输出到结果 slice 40..43
为 2 cycle。
