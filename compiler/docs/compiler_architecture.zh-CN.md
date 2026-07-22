# FTLPU 编译器架构

[English](compiler_architecture.md) | [简体中文](compiler_architecture.zh-CN.md)

本文确定第一阶段 LPU 后端的编译器分层：

```text
ONNX / PyTorch / TensorFlow
  -> StableHLO
  -> FTLPU kernel IR
  -> FTLPU tensor IR
  -> FTLPU stream IR
  -> FTLPU schedule IR
  -> FTLPU command IR
  -> .ftlpu 二进制
  -> runtime / CModel / 硬件
```

StableHLO 是主要的前端/共同模型 IR 边界。IREE 是参考编译器框架和对比工具，
不是 LPU 后端必须永久依赖的 IR。

## 各层职责

### StableHLO 边界

StableHLO 表达框架导入后的前端模型语义：

- matmul 和 batched matmul 表达为 `stablehlo.dot_general`；
- convolution 表达为 `stablehlo.convolution`；
- elementwise 和 activation 使用 StableHLO 算术操作；
- 显式记录 tensor shape、元素类型和 broadcast 语义。

该边界让 LPU 编译器不受 ONNX、PyTorch 和 TensorFlow 图格式差异影响。

### FTLPU Kernel IR

Kernel IR 是第一层由 FTLPU 自有的编译器 IR，负责：

- 把 StableHLO op 规范化为小规模、面向 LPU 的 kernel 集合；
- 将每个 kernel 映射到 MXM、VXM 等具体 LPU 功能单元；
- 用功能单元组合表示 SwiGLU 等融合 kernel，例如 `MXM + MXM + VXM + MXM`；
- 验证 LPU 支持的静态 shape 和元素类型；
- 显式保留 quantization 与 layout 元数据。

### FTLPU Tensor IR

Tensor IR 负责 MEM 分配和 tensor 放置：

- 为 activation、weight、intermediate 和 output 分配 MEM 范围；
- 选择 MEM column/bank 和基础地址；
- 描述引用已选 kernel 的 tile plan；
- 显式记录 layout 和元素大小。

已实现的 `ftlpu.tensor.matmul` 和 `ftlpu.tensor.swiglu` 使用物理 rank-5 MEM
地址元组 `[device, hemisphere, slice, bank, word, byte]`。每个 hemisphere
包含 44 个 slice，每个 slice 有 2 个 bank，每个 bank 有 4096 个 16-byte
word。分配器按 tensor role 使用 east hemisphere 的 SRAM row pool。函数输入在
入口处存活，每个 SSA tensor 保留到最后一次使用；失效的 row range 会合并并通过
first-fit 策略复用。当前 operand 失效之前先分配 output，避免功能单元覆盖仍在读取
的输入。

Matmul placement 同时记录 CModel 所需的 row geometry。MXM weight 采用
`mxm_weight_striped`，分布在 MEM slice 0..15；activation 使用 slice 32 上的
`vector` placement；int32 result 使用 slice 40..43 上的四个
`int32_byte_planar` plane。每个 placement 记录 slice 列表、基础 SRAM row、
指令数和有符号地址 stride。当前 CModel 约定 stride 为 16 row，weight Read
命令反向遍历 row。

完整 `ftlpu.tensor.ffn` 使用 320x640 gate/up weight 和 640x320 down weight。
共享 activation 放在 vector slice 32；两个 320-column 的量化 hidden pass
分别存放在 slice 40 和 41；最终 160x320 i8 result 放在 slice 42。
MXM int32 intermediate 始终停留在 stream 上。

### FTLPU Stream IR

Stream IR 将 MEM 中的 tensor tile 映射到 LPU stream：

- 每条 stream 都有 source 和 sink，例如 `MEM:A -> MXM0:lhs`；
- 每条 stream 记录方向内连续的 stream range 和 stream register id；
- 每条 stream 记录起始地址、字节数和端点功能单元；
- MXM/VXM post-processing stream 必须显式表达，不能隐含在 kernel 中；
- 当指令天然沿南到北的 tile chain 传递时，stream 表示为长逻辑向量，不把
  20 个物理 tile 展开成独立 IR op。

该层概念上类似 IREE Stream，但由 FTLPU 自有并直接建模真实 LPU 数据移动。

`ftlpu.stream.route` 记录方向内连续范围
`[stream_base, stream_base + stream_count)`、物理 MEM 边界 `register_id`、
source/destination endpoint、显式功能单元 id、MEM 地址、字节数和固定 transport
latency。MEM endpoint 的 unit id 为 `-1`，MXM endpoint 标识 MXM0 或 MXM1。
`ftlpu.stream.matmul` 还记录选中的 MXM unit 和 weight-buffer id。

每个 `ftlpu.tensor.matmul` 变为两条 eastbound MEM-to-MXM route、一个
`ftlpu.stream.matmul` 和一条 westbound MXM-to-MEM route。MXM weight route
在 IW load 期间占用 16 条 stream；activation compute 使用 1 条 stream；每个
int32 result 使用连续 4 条 byte stream。

Stream allocator 内部使用 SSA op 顺序判断 stream range 能否安全复用，但不会在
Stream IR 中输出逻辑 lifetime stage。精确发射和到达 cycle 由
Stream-to-Schedule 分配。

`ftlpu.stream.ffn` 直接表达完整拓扑。每个 hidden pass 将一对 320-column
gate/up weight 装入 MXM0/MXM1；共享 activation route 同时供给两个 compute；
west stream group `W0..W3` 和 `W4..W7` 输入 SwiGLU VXM pipeline；`E31`
将两个 hidden chunk 写入 slice 40 和 41。Down 阶段加载两个 320-row K partition，
在两条 activation stream 上读取两个 hidden slice，生成两个 int32 partial，
通过 VXM AddQuant 合并后把最终 `E31` result 写入 slice 42。

### LPU Target Model

`LPUTargetModel` 是编译器侧唯一的硬件参数来源，包含：MEM geometry、每个方向
32 条 stream、64 个 packed selector、12 个 MEM 边界 register column、额外的
SXM-to-MXM column、MXM 维度和吞吐、支持的 endpoint route、register mapping
及 transport latency。Latency 表示从 producer 发射到 consumer 可见，并包含
CModel tick phase；lowering pass 必须查询 target model，不能嵌入补偿 cycle。

### FTLPU Schedule IR

Schedule IR 是已调度的底层 target IR，包含：

- 显式 cycle number；
- 显式 MEM/MXM/VXM queue；
- 显式 NOP 和 repeat 机会；
- `mem_read_weight`、`mem_read_activation`、`mxm_load`、`mxm_compute`、
  `mem_write` 等 stage-level op；queue command 展开属于 Command lowering。

Schedule dialect 使用 `ftlpu.schedule.mem_read`、`ftlpu.schedule.mxm_load`、
`ftlpu.schedule.mxm_compute` 和 `ftlpu.schedule.mem_write`。每个 op 带有 ICU
发射 `cycle` 和 `duration`。Scheduler 会预留每个 MEM slice queue、方向内
stream、选定 MXM unit 的 load/compute queue 及选定 weight buffer。
`mxm_load` 和 `mxm_compute` 显式保留两个 id。Producer 与 consumer window
之间计入固定 transport latency，SSA consumer 不能在 producer MEM write
完成之前读取数据。

当前 320x320 int8 GEMM 的 CModel 对齐基线为：weight MEM read 根据 MEM 边界
从 cycle 5..8 启动；IW 在 `[18,38)` 运行；`E16` activation MEM Read 在
`[33,353)` 运行；Compute 发射在 `[38,358)`；第一个 MXM result 在 cycle 57
出现；四个 int32 byte-plane MEM write 在 `[59,379)` 运行。20 个物理 tile row
排空期间，output stream 在完整 339-cycle MXM result window 内保持占用。

对于 160x320x640x320 FFN fixture，共享 activation startup 在两个 hidden pass
中都按 CModel 路径分为三个 segment（`E16`、`E30`、`E0`）。正确性优先的
schedule 在 cycle 58/73/77 计算 pass 0，在 318/333/337 计算 pass 1。
第一个 MXM result 与 VXM 消费之间有 12 个 stream-register transport cycle。
两条 SwiGLU pipeline 分别在 cycle 89 和 349 启动，并在 cycle 110 和 370
向 slice 40/41 写入 160 行 i8 数据。Down weight 在 cycle 538 和 558 装入
buffer 0；两个 MXM 在 cycle 590 从 `E0` 和 `E16` activation stream 开始计算。
六级 VXM AddQuant 在 cycle 621 启动，最终 slice-42 result 在 cycle 638 写回。
当前 schedule 有意采用串行方式；buffer-1 ping-pong 留作独立性能优化。

### FTLPU Command IR

Command IR 是稳定的编译器/runtime 边界。`ftlpu-schedule-to-command` pass 移除
Schedule SSA graph，生成 `ftlpu.command.binding`、`ftlpu.command.mem`、
`ftlpu.command.mxm` 和 `ftlpu.command.vxm`。Binding 描述输入/输出 index、shape、
元素类型、字节数和物理 placement。Result 由 binding 及其物理 MEM write command
表示，不再作为 SSA tensor 返回。

每条 command 一一对应某个 ICU queue 的首条指令加 Repeat：包括绝对首 cycle、
queue index、opcode、stream selector、repeat count/interval 和 MEM address
stride。320x320 GEMM 生成 16 条 weight MEM command、1 条 activation MEM
command、4 条 result MEM command、1 条 IW command 和 1 条 Compute command。
VXM command 携带 ALU opcode、带类型的 stream/ALU/immediate operand、cast
target、output stream 和 repeat 元数据。Command op 被显式标记为有 side effect，
因此 MLIR canonicalization 不会删除硬件工作。

Binary emission 按 queue 对扁平 command stream 分组，并根据绝对 cycle 推导
queue-local NOP，无需重建调度决策。`ftlpu-translate` 将该层序列化为 `.ftlpu`
二进制版本 2。Runtime 在时钟启动前把绑定输入放入 SRAM 并加载全部 ICU queue，
随后推进 `TspSliceSystem::tick()` 并还原绑定输出。320x320 回归将全部
102,400 个 int32 result 与 CPU GEMM 比较。

### 测试产物

编译器测试按照测试名称分别保存生成的 IR：

```text
build-ftlpu-vs2026/compiler/ftlpu_lower/<测试名>/
```

完整 FFN lowering 测试保留每个可见边界：`ffn.stablehlo.mlir`、
`ffn.kernel.mlir`、`ffn.tensor.mlir`、`ffn.stream.mlir`、
`ffn.schedule.mlir` 和 `ffn.commands.mlir`。Runtime 测试使用另一个目录并额外
生成 `ffn.ftlpu`，因此并行或重复运行测试不会覆盖其他测试的产物。

## IREE 的作用

IREE 展示了成熟的 MLIR 编译器工程模式：

- 前端 import 边界；
- Flow dispatch formation；
- Stream 风格调度和资源建模；
- pass pipeline 组织；
- 树外 target/backend 插件结构。

仓库可以保留使用 IREE 的参考路径测试：

```text
ONNX -> IREE importer -> IREE Flow IR
```

这些测试用于对比和 sanity check。主后端路径应为：

```text
StableHLO -> FTLPU kernel IR -> FTLPU tensor IR -> FTLPU stream IR
          -> FTLPU schedule IR -> FTLPU command IR
```

## 近期里程碑

1. 保留 ONNX 到 IREE Flow 测试作为参考覆盖。
2. 增加 matmul 和 elementwise op 的 StableHLO fixture 测试。
3. 定义文本形式的 `ftlpu.kernel`、`ftlpu.tensor` 和 `ftlpu.stream` 示例。
4. 将 StableHLO matmul lower 到 MXM kernel、MEM allocation 和显式 stream。
5. 将 StableHLO/FTLPU Kernel FFN SwiGLU lower 到 CModel 对齐的 schedule。
6. 将 FTLPU stream/schedule program lower 到 `.ftlpu`。
