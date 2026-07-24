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
- 使用可复用的 `matmul`、`batch_matmul`、`rope`、`softmax`、`transpose`、
  `swish` 和 elementwise 等计算原语；
- 验证 LPU 支持的静态 shape 和元素类型；
- 显式保留 quantization 与 layout 元数据。

公开的 Kernel IR 使用这些原语组成 FFN 和 Attention 图，不再把两者表示成
不可拆分的大 op。融合由后续优化 pass 决定。当前 Tensor 后端迁移期间仍通过内部
`ftlpu-compose-kernel-plans` 兼容 pass 接入原有分配代码；该兼容 op 不会出现在
StableHLO-to-Kernel pipeline 的输出中。

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

FFN 在这一层表示为通用的物理 task 图：两个
`ftlpu.tensor.matmul_task` projection、一个 `ftlpu.tensor.swish_task`、
一个 `ftlpu.tensor.elementwise_task` 和一个 down projection matmul task。
每个 matmul 分别记录 operand 和 result 的 allocation 列表。空的 result
allocation 表示数据仍停留在 MXM accumulator 或 stream 中，不写入 MEM。
elementwise 的结果带有东西半球两份 allocation，down projection 同时消费这两份
hidden 存储。地址、placement、字节数和量化参数由实际使用它们的 primitive op
携带，不再隐藏在单个 `ftlpu.tensor.ffn` 中。

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

公开 Stream IR 中的 FFN 也已经拆成 primitive 图。Gate 和 up 分别表示为
`ftlpu.stream.matmul_task`，共享一条 activation route，并使用各自的反量化
weight route。`swish_task` 和 `kind = "multiply"` 的 `elementwise_task`
显式表达 VXM 数据流以及东西半球的两份 hidden allocation。随后，两条 hidden
MEM-to-MXM route 分别送入两个 down matmul task；最后由
`kind = "add_quant"` 的 `elementwise_task` 合并 partial，并持有最终 result
allocation。每个 matmul task 都显式记录 MXM unit、weight buffer 和 result
stream range。当前 Stream-to-Schedule 会在内部把该图整理为已有 scheduler
descriptor，但 StableHLO-to-Stream pipeline 不会输出 compound FFN op。

### 调度计划与 IR 生成

Stream-to-Schedule 已拆分为与 MLIR 无关的计划层和 MLIR 生成层。
`SchedulePlan` 是统一的 task DAG。每个 task 都包含稳定 ID 和名称、功能类型、
模型阶段、最早 cycle、持续时间、资源窗口，以及带固定 transport latency 的
生产者依赖。计划会在 `ResourceScheduler` 分配精确 cycle 前检查重复名称、
非法依赖和依赖环。

Attention 分为 Projection、RoPE、Softmax、PV 和 OutputProjection 五组
planner/emitter。Planner 在不持有 `IRRewriter` 的情况下生成 projection work、
QK/PV work wave 和五阶段 DAG；各阶段 emitter 只读取不可变计划并生成
Schedule IR。RoPE 虽然拥有独立 planner 和 emitter 模块，但仍在每个 Q/K
projection 小块完成后立即融合执行，不会被强制串行化。Attention 物理内存布局
属于 Analysis；Softmax planner 会预约 VXM/MEM 窗口并返回每个 work wave、
每个半球的精确 cycle，Softmax emitter 不再持有资源调度器。

FFN 使用可复用的 WeightLoad、Projection、Swish 和 DownProjection schedule
builder。Gate/Up 和 Down 共用同一套 weight dequant 与 MXM load emitter；
六 cycle Swish ALU 序列可以独立测试。`FfnSwishPlanner` 会避开 weight
dequant 和临时 MEM 流量来安排 VXM/MEM 资源窗口，FFN MLIR emitter 只消费
确定的 cycle，不再调用 `ResourceScheduler`。过时的
compound `ftlpu.stream.ffn` 已删除；公开 Stream IR 只保留 primitive task 和
route op。

### LPU Target Model

`LPUTargetModel` 是编译器侧唯一的硬件参数来源，包含：MEM geometry、每个方向
32 条 stream、64 个 packed selector、12 个 MEM 边界 register column、额外的
SXM-to-MXM column、MXM 维度和吞吐、支持的 endpoint route、register mapping
及 transport latency。Latency 表示从 producer 发射到 consumer 可见，并包含
CModel tick phase；lowering pass 必须查询 target model，不能嵌入补偿 cycle。

架构探索期间可以通过 JSON 覆盖 target 参数：

```text
ftlpu_opt --target-config compiler/examples/targets/exploration_40_streams.json ...
```

JSON 可以覆盖 `memory`、`streams` 和 `throughput` 三组字段；未指定字段继续使用
与默认 CModel 兼容的值。解析后的完整配置会序列化为 module 上的
`ftlpu.target` dictionary，因此每个中间 MLIR 文件都携带可复现后续 lowering
所需的硬件参数。Kernel-to-Tensor、Tensor-to-Stream 和 Stream-to-Schedule
都会从该属性恢复同一份 target model。配置校验会拒绝非正维度、超过方向 stream
容量的功能单元宽度、越界 MEM slice base 和不兼容的 tile geometry。

`exploration_40_streams.json` 是一份回归配置：每个方向使用 40 条 stream，并修改
MXM/VXM latency。通用 W8A16 FFN 必须能 lower 到 Schedule IR，且调度结果应不同于
默认 target。若要继续生成并执行非默认 Command/Binary，runtime 和 CModel 也必须
使用相同 target ABI；不能把探索配置生成的 binary 静默交给默认硬件执行。

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
