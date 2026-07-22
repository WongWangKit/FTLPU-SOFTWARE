# 通用 W8A16 FFN 流水线：SmolLM2-135M 实例

![SmolLM2-135M FFN 流水线](smollm2_ffn_pipeline.svg)

该图是 `ftlpu-stream-to-schedule` 通用 W8A16 FFN lowering 的一个实例。编译器根据
IR shape 和 `LPUTargetModel` 推导循环次数、物理行地址、slice 布局、传输延迟和
cycle 间隔；编译器源码中没有 SmolLM2 尺寸专用分支。

对于 `32x576x1536x576`，相邻 K block 每 38 cycle 启动一次，并交替使用 weight
buffer 0/1。每个 block 连续发射 32 行 MXM compute，行发射占用率为 84.2%；其余
6 cycle 来自已建模的 MXM pipeline/control 约束。最后一条 command 位于 cycle
34,270；projection/SwiGLU 重叠前为 35,743，MXM 流水化前为 87,150。

对于 `M = 32*T`，projection 的循环顺序为 `N-tile -> K-tile -> M-tile`。一个 `32x32` 的 gate/up weight tile 只 load/dequant 一次，随后驻留在 MXM weight buffer 中，依次处理全部 `T` 个 activation tile。以 `seq_len=128` 为例，`T=4`：一次 weight load 后接四次 `M=32` compute，并写入四个互不重叠的 accumulator 地址范围。这是通用的 M tiling，不是 seq_len 专用分支；下图的 `M=32` 只是其 `T=1` 实例。

本文中的 MEM slice 编号均为单个半球内的 local 编号。因此 gate accumulator 在东、西
半球内分别使用 local slice 36..39，up accumulator 分别使用 local slice 40..43。
在 CModel 扁平化的 88 条 MEM queue 视图中，东半球 queue 为 `local_slice`，西半球
queue 为 `44 + local_slice`。

MXM load 使用 16 条 stream。单个 MXM compute 使用两条 FP16 activation stream；并行的
gate/up 两个 MXM 因而共用 E0..E3。一个 32x32 K block 连续流入 32 行 activation，
每个输出行在四个物理 K=8 段内累加，生成一个 32x32 partial tile。K=576 的 18 个
partial tile 先在 MEM 中累加，完整的 gate/up tile 才能进入 SwiGLU。当 B+1 的 load
与 B 的 compute 重叠时，activation 临时从 E0..E3 切换到 E16..E19。对于本例的物理
布局，down 相邻 output pair 的间隔为 95 cycle，因为结果写回和下一组权重读取都会
占用各自半球内的 local MEM slice 24；该间隔由冲突 slice 和 transport latency 动态计算得出。

![完整的 18 个 K partial 累加](smollm2_ffn_partial_accumulation.svg)

该时间线展开一个半球、一个 `M=32, N=32` output tile。图中完整给出写入同一 gate/up
accumulator tile 的 `P0..P17`，以及累加状态 `S0..S17`；西半球采用相同 cadence，
但使用不同的物理 MEM/SREG identity。

图中展开的 SwiGLU 段就是 Swish 的实际 VXM 微调度。在 cycle `t`，两条 VXM 路径
同时开始计算 `-gate` 和 `gate * up`。sigmoid 路径在 `t+1..t+3` 依次执行 `exp`、
加一和倒数；另一条路径通过 pass-through 保持 `gate * up` 的时序对齐。`t+4` 将
两条路径相乘，得到 `sigmoid(gate) * (gate * up)`，也就是
`SiLU(gate) * up`。结果在 `t+5` 转成 FP16，并在计入传输和 MEM 布局延迟后写入
local hidden slices 21/22/23/29。

单个 32x32 MXM weight tile 的 dequantization 只占用 4 个发射周期：VXM 每周期处理
8 列。Gate/Up 与 East/West 一共形成 4 个独立 weight tile。由于它们共享同一组 16 条
VXM ALU ICU queue，当前排程将四组 dequant 错开发射，合计形成 16-cycle 的发射窗口；
这是四次很快的 dequant，而不是一次 dequant 需要 16 cycles。

图中的第二段刻意描述当前 lowering 实际生成的调度，而不是尚未实现的 accumulator 直连 stream 方案。最后一个 K partial（`P17`）仍先写入本半球的 accumulator SRAM，再通过 `W0..W7` 读出。当前没有固定大小的 SwiGLU overlap batch：编译器保守地等待所有 projection pair 完成，再在 cycle 16,469 发射首个“行 × 半球”Swish event，之后顺序发射其余 event。待明确双半球共享 VXM 的资源策略后，再将 accumulator-to-stream 作为一套独立的真实调度实现并绘制。

当前 tail 使用 `W0..W7` 读取 accumulator，并写入 local hidden slices 21/22/23/29。
在 scheduler 能够对共享 VXM、MEM、stream 与 transport 资源做精确 cycle 预约前，
该 lowering 不宣称存在 projection/SwiGLU 重叠。
