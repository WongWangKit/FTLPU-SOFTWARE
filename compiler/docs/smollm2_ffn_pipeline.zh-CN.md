# 通用 W8A16 FFN 流水线：SmolLM2-135M 实例

![SmolLM2-135M FFN 流水线](smollm2_ffn_pipeline.svg)

## 调度策略

编译器保留两套通用调度策略：

- `--ffn-schedule tail` 是默认策略，完整保留已有 FFN 调度。对于
  `seq_len=128`，首个 SwiGLU 在 cycle 55,355 发射，CModel 程序结束于
  cycle 92,573。
- `--ffn-schedule fused` 将已经完成的 Gate/Up tile 与后续 projection
  重叠。首个 SwiGLU 提前到 cycle 2,363，程序结束于 cycle 86,685，减少
  5,888 cycles（6.36%）。

![融合调度的 seq_len=128 FFN 流水线](smollm2_ffn_fused_pipeline.svg)

fused 策略不使用尚未建模完整的 ACC 直连 VXM 旁路。最后一个 K partial
中，东半球 MXM 使用 `W8..W15`，西半球 MXM 使用 `W16..W23`。
`accumulate -> stream + clear` 将完整的 FP32 Gate/Up tile 写入普通 MEM
临时 byte plane：Gate 使用 local slices `1/5/9/13`，Up 使用
`2/6/10/14`。VXM allocator 只有在整条 repeat MEM write queue 释放后才会
读这些临时区，同时避开每个 4-cycle weight-dequant 资源窗口，最后通过
`E30/E31` 写回 FP16 hidden。ACC 到 MEM 的 transport 计算包含 fabric
在周期末 commit 所需的额外一拍。

两套策略由同一套 shape-driven lowering 生成。`seq_len=128` 的 fused
binary 已在 CModel 上运行，并与 CPU reference 的 73,728 个 FP16 数值
逐一一致，其中 72,633 个输出非零。

## CModel 实测利用率

下表来自 CModel performance monitor 对编译后 `seq_len=128` binary 的实际
执行统计。统计包含 runtime 末尾的 64 个 drain cycles，因此 tail 共采样
92,637 cycles，fused 共采样 86,749 cycles。`array utilization` 的分母是
全部采样周期内的 MXM cell 容量；`active density` 会从分母中去掉完全空闲的周期。

| MXM | Active cycles（tail/fused） | Tail array util. | Fused array util. | Tail active density | Fused active density |
| --- | ---: | ---: | ---: | ---: | ---: |
| MXM0 | 86,052 / 86,052 | 92.85% | 99.16% | 99.96% | 99.96% |
| MXM1 | 86,052 / 86,052 | 92.85% | 99.16% | 99.96% | 99.96% |
| MXM2 | 79,902 / 79,902 | 86.22% | 92.07% | 99.96% | 99.96% |
| MXM3 | 79,902 / 79,902 | 86.22% | 92.07% | 99.96% | 99.96% |
| 四个 MXM 平均 | - | 89.54% | 95.62% | 99.96% | 99.96% |

fused 与 tail 完成相同的 MXM cell 工作，但总周期更少，因此全程序 MXM 平均
利用率提高了 6.08 个百分点；MXM 一旦开始工作，内部阵列密度仍接近饱和。

| 资源 | 策略 | 全程序利用率 | Active density | Stall rate | Peak |
| --- | --- | ---: | ---: | ---: | ---: |
| VXM ALU | tail | 15.91% | 74.19% | 0.00% | 512/512 |
| VXM ALU | fused | 16.99% | 72.67% | 0.00% | 512/512 |

VXM 利用率以每周期 512 个 lane-ALU execution slot 为总容量。fused 的实际
执行工作量不变，但非空闲 VXM 周期增加了 414 个，因此全程序利用率提高，
active density 略微下降。

| SR fabric | 策略 | Link BW | East BW | West BW | Staged-write util. | Active density | Peak link bytes/cycle |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 东半球 | tail | 4.94% | 5.57% | 4.31% | 5.90% | 4.94% | 4,928/24,576 |
| 东半球 | fused | 5.24% | 5.95% | 4.53% | 6.30% | 5.24% | 6,528/24,576 |
| 西半球 | tail | 4.57% | 5.11% | 4.03% | 5.47% | 4.91% | 4,928/24,576 |
| 西半球 | fused | 4.85% | 5.46% | 4.23% | 5.84% | 5.23% | 6,528/24,576 |

SR bandwidth utilization 的分母是模型定义的各半球 SR-fabric 链路容量，而不是仅统计有
流量的周期。CModel 目前还没有提供容量归一化的 MEM 和 SXM utilization
counter，因此本文不根据流水图推测这两项百分比。

Accumulator bank 仍拆成独立行，颜色表示具体操作：紫色表示
`accumulate -> SRAM`，红色表示 `accumulate -> stream + clear`。

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
这是四次很快的 dequant，而不是一次 dequant 需要 16 cycles。Down 也使用相同的
连续 4-cycle dequant 和 16-stream IW load，包括与上一个 reduction 最后一个 M tile
重叠的 weight prefetch。Activation 只在实际 IW 冲突窗口内于 `E0/E1` 和
`E16/E17` 之间切换。

第二张图使用一条连续时间轴展示 projection 完成到首个 SwiGLU 的过程。在 tail 图中，
所有 Gate/Up projection pair 完成后，最后一个 partial 保留在 accumulator SRAM，
随后通过 `W0..W7` 送入 SwiGLU。在 fused 图中，同一张图跟踪一个已经完成的
accumulator tile，依次经过 `stream + clear`、临时 MEM staging 和 SwiGLU；与此同时，
后续 projection 仍在执行。

当前 tail 使用 `W0..W7` 读取 accumulator，并写入 local hidden slices 21/22/23/29。
在 scheduler 能够对共享 VXM、MEM、stream 与 transport 资源做精确 cycle 预约前，
该 lowering 不宣称存在 projection/SwiGLU 重叠。
