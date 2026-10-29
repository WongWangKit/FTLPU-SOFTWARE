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

## 二进制版本 5

所有标量字段均为 little-endian。文件头为：

```text
magic[8] = "FTLPUB01"
u32 version = 5
u64 target_abi
u16 target_name_size
char target_name[target_name_size]
u64 max_cycle
u32 queue_count
u32 binding_count
BinaryBinding[binding_count]
QueueProgram[queue_count]
```

一个 binding 包含固定元数据，随后是 `rank` 个 64 位维度和 `slice_count` 个
16 位 MEM slice 标识。队列记录包含队列类型、队列编号、命令数和编码后的
`QueueCommand`。每个命令保存 ICU command word、指令类型、word 数量、四个固定
32 位指令 word，以及可选的变长 payload。

编译器会把所有影响命令兼容性的拓扑、吞吐、布局和 latency 参数散列到
`target_abi`，其中包括 `qk_iw_to_compute_latency`。当前 CModel runtime 只接受
`ftlpu-lpu32`；如果 binary 的
target 不匹配或来自没有 target 身份的旧格式，runtime 会在写入任何 ICU queue
之前拒绝执行，避免架构探索配置误跑到默认硬件模型。版本 1 到 4 仍可被 reader
读取用于检查，但不能由当前 runtime 执行。

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

## DMA 到分布式 i-MEM 的逐周期装载

`CModelRuntime::load_timed(program, image_base)` 是完整时序入口，需要
`C2cDmaSystem`。它先准备真实物理字节映像并预置到 DDR，再向系统装载器提交清单，
提交时不会直接写 i-MEM，也不会预取或执行程序。它保留系统的 DDR/DMA 状态和全局
物理周期；下一次 `run_cycles()` 继续推进装载、写入提交和本地前端预取，然后统一启动。
`load()` / `load_direct()` 保留原有自动配置、直接写 i-MEM、同步 prime 的便利路径，
该路径不用于评估程序装载时序。

```cpp
ftlpu::C2cDmaSystem system;
ftlpu::software::runtime::CModelRuntime runtime(system);
runtime.load_timed(program);        // 默认源地址 0x1'0000'0000，调用者可指定
runtime.upload_input(0, input);
runtime.run_cycles(program.max_cycle + 64);
runtime.print_program_load_performance(std::cout);
```

DDR 映像源区间需要由调用者与权重/数据地址分配协调，装载期间不能覆盖同一区间。
`prepare_program_images()` 复用或生成 MEM 96-bit 和 MXM load/compute/dequant 128-bit
Macro 映像；清单源地址按真实 word 长度及 DMA vector 尾部填充前进，目标按物理 ICU
编号解析。尚无明确物理编码的 Native/VXM/SXM 队列在修改 DDR/i-MEM 之前报错。
不将 QueueCommand、扩展描述符或 C++ 语义对象的数量当作物理传输字节数。

`run_cycles()` 的参数仍是 kernel 逻辑周期数。装载、预取以及初始输入/权重等待只推进
物理周期，不消耗 kernel 逻辑周期。输入通过 upload_input/upload_binding/copy_binding
登记；首批权重继续使用现有 page-ready 检查。所有映像已提交、前端已就绪、必要输入和
首批权重均已就绪后，装载器才记录 program_start 并启用计算队列。
独立控制器也可直接调用 `system.program_loader()` 的 submit/busy/
all_images_committed/ready_to_start/start 接口。

`program_load_statistics()` 和 `print_program_load_performance()` 输出受理、首笔 DDR
请求/返回、各队列 source_done 和 image_committed、全程序提交、预取、启动周期，以及
物理字节、DMA 字节、word 和反压等待统计。时间戳是系统物理周期；连续 timed kernel
不会清零。固定估算装载延迟不会再次叠加。五个配置项的默认值和校验规则见 CModel 的
`docs/hardware_configuration.zh-CN.md`。

回归入口 `program_load_runtime_test` 包含真实 DDR 到 21 个 MEM/MXM/dequant 队列的
BF16 矩阵向量数值回归、跨半球数据复制、提前启动门控、首批权重等待及连续程序切换。
两种装载路径的全部输出都与参考值比较；CModel 的 program_loader_test 覆盖带宽、
写间隔、提交/传输流水线、小 FIFO、共享 DDR 和逐字节映像一致性。
