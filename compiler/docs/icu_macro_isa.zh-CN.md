# ICU 粗粒度调度 ISA v1

## 范围

v1 将确定性的指令展开从 host runtime 下沉到各功能单元的本地 ICU。
它不改变 Schedule IR 的 cycle，也不修改已有功能单元原生指令。

通过命令行选择 ICU 压缩模式：

```text
ftlpu-opt ... --icu-compression none|control|macro
```

默认使用 `macro`。`none` 会物化功能指令，但保留按时长编码的 NOP 间隔；
`control` 启用 Repeat 和 Repeat2D；`macro` 进一步启用带类型的粗粒度 ICU
描述符和物理 Macro 队列编码。旧选项
`--icu-macro-schedule` 继续作为 `--icu-compression macro` 的兼容别名。

生成的 module 带有 `ftlpu.icu_compression = "..."`。在 Command IR 兼容期内，
仍会同时携带旧的 `ftlpu.icu_macro_schedule` 布尔属性。选择 Macro 压缩时，
binary lowering 会在当前文件 envelope 中生成带类型的扩展 ICU 描述符。runtime
完成 relocation 后，把 MEM/MXM `STREAM_ND` 重打包为下面定义的固定硬件包，
再写入目标 ICU；文件 envelope 只承担传输和 relocation，不进入本地 ICU。

## 描述符

每条 macro 包含一条原生功能单元指令和如下调度字段：

| 字段 | 含义 |
| --- | --- |
| `start_cycle` | 第一次发射的绝对 cycle |
| `inner_count` | 单个内层 repeat 的发射次数 |
| `inner_interval` | 内层相邻发射的 cycle 间隔 |
| `inner_stride` | 内层每次发射的操作数字段增量 |
| `outer_count` | wave 重复次数 |
| `outer_interval` | 相邻 wave 起点间隔 |
| `outer_stride` | 每个 wave 的操作数字段增量 |
| `induction_target` | MEM 地址、MXM weight column、MXM accumulator 地址或无 |

binary 原型在原生指令 payload 后附加 `MACR` 标记和 8 个字段。当前
采用可变长 envelope 是为了先验证语义，并不代表最终 RTL 位宽已经冻结。

坐标 `(outer, inner)` 的实际发射为：

```text
cycle = start_cycle + outer * outer_interval + inner * inner_interval
operand_delta = outer * outer_stride + inner * inner_stride
```

### MEM_STREAM_ND

MEM 队列进一步使用专用的 `MEM_STREAM_ND`。一条描述符携带一条原生
read/write 指令和最多三层仿射计数器：

| 字段 | 含义 |
| --- | --- |
| `start_cycle` | 坐标 `(0, 0, 0)` 的绝对发射 cycle |
| `rank` | 有效维数，范围 1 到 3 |
| `count[d]` | 第 `d` 维的迭代次数 |
| `cycle_stride[d]` | 第 `d` 维每前进一步增加的 cycle |
| `address_stride[d]` | 第 `d` 维每前进一步增加的 SRAM row |

第 0 维是最内层。对于坐标 `(i0, i1, i2)`，MEM ICU 的发射位置为：

```text
cycle   = start_cycle + sum(id * cycle_stride[d])
address = base_address + sum(id * address_stride[d])
```

同一 MEM 队列可以同时保持多条活跃描述符，由 next-issue calendar 按下一次
发射 cycle 交错执行。这样既保留不规则的 phase 边界，又能用一条描述符覆盖
规则的 token、block 和 page 三层循环。当前 binary envelope 中 count 和
cycle stride 为无符号 32 位，address stride 为有符号 32 位；各维在发射
时间上不允许重叠。装载到硬件包时，runtime 进一步检查下面的固定字段范围。

### MEM_SLICE_PROGRAM

`MEM_SLICE_PROGRAM` 在一个共享 N 维启动域下，合并同一个物理 slice 上最多
16 项 MEM 操作。每个 body entry 保留自己的原生 read/write 指令、相对 cycle
offset 和三个仿射地址步长。对 body `b` 和启动坐标 `i`：

```text
cycle   = start_cycle + cycle_offset[b]
          + sum(i[d] * cycle_stride[d])
address = native[b].address + sum(i[d] * address_stride[b][d])
```

binary lowering 只会合并 rank、count、cycle stride 和 binding metadata 完全
一致的 sequence，并在形成 program 前校验所有展开后的发射 cycle。一个
relocation 指向父 command，runtime 会将其地址增量应用到每个 body 的原生
MEM 指令。当前编译器把 body cycle offset 限制在 65,535 cycle 以内，为后续
固定宽度 RTL 编码保留实现空间。

### MXM_STREAM_ND

MXM load、compute 和 dequant 队列使用同样的一到三维调度描述符，并携带一条
原生 MXM 指令。`operand_stride[d]` 的含义由强类型 `induction_target` 决定：

| 队列/原生 opcode | 归纳目标 |
| --- | --- |
| MXM load / `IW` | weight column |
| MXM compute / `Compute` 或 `AccumulatorRead` | accumulator address |
| MXM dequant | 无，只重复发射 cycle |

weight buffer、激活流、输出流、accumulator destination、clear 标志和输出
格式等其他原生字段都保持不变。因此，最后一个 partial 所使用的
`accumulator_destination = stream`、`accumulator_clear = true` 会自然保留为
独立描述符，不会错误地与普通累加指令合并。

```text
cycle         = start_cycle + sum(i[d] * cycle_stride[d])
operand_delta =               sum(i[d] * operand_stride[d])
```

编译器按 cycle stride 对仿射维度排序，并验证嵌套硬件计数器能够单调发射。
不同描述符仍可通过每队列的 next-issue calendar 交错执行。

### MEM/MXM 固定硬件 STREAM_ND 包

MEM、MXM load、MXM compute 和 MXM dequant 共用一条固定 320-bit 指令格式。
它由 10 个 32-bit little-endian 传输 word 组成，rank 不改变取指长度：

| bit | 宽度 | 字段 |
| ---: | ---: | --- |
| 3:0 | 4 | opcode，`8` 表示 `STREAM_ND` |
| 5:4 | 2 | version，当前为 `0` |
| 8:6 | 3 | unit：MEM/load/compute/dequant |
| 10:9 | 2 | `rank - 1` |
| 12:11 | 2 | induction target |
| 31:13 | 19 | 保留，必须为 0 |
| 55:32 | 24 | `start_cycle` |
| 119:56 | 64 | 原生 MEM/MXM 指令 |
| 135:120 | 16 | `count[0] - 1` |
| 159:136 | 24 | `cycle_stride[0]` |
| 177:160 | 18 | 有符号 `operand_stride[0]` |
| 235:178 | 58 | 第 1 维，字段顺序同上 |
| 293:236 | 58 | 第 2 维，字段顺序同上 |
| 319:294 | 26 | padding，必须为 0 |

因此 count 上限为 65,536，cycle 和 cycle stride 使用 24 bit，operand stride
范围为 -131,072 到 131,071。未使用维必须编码为 `count=1`、
`cycle_stride=1`、`operand_stride=0`。decoder 还检查版本、保留位、unit 与原生
opcode 的对应关系，以及各维发射 cycle 不重叠。

固定包在本地 i-MEM 中占完整槽：96-bit MEM i-MEM 使用 4 槽，128-bit MXM
i-MEM 使用 3 槽。它仍是一条逻辑描述符，只建立一个三维循环上下文。ICU 在
描述符到达队首时解码和锁存原生指令、三个 count、cycle stride 与 operand
stride；之后计数器/calendar 每 cycle 产生至多一条原生功能指令，不再由 host
逐条展开。

Qwen2.5-1.5B、`seq_len=32` 的 FFN Up 规则主循环测试使用 `M=32`、
`K=1536`（48 个 K tile）、每半球 `N=4480`（140 个 N tile）。一个本地 MEM
weight 队列、MXM load、dequant 和 compute 队列各只装入一条固定包，分别展开
26,880、26,880、26,880 和 215,040 条逐 cycle 指令。完整 FFN binary 也通过
同一固定包装载路径执行，49,152 个 BF16 输出与参考结果逐点一致。

完整 projection 中遇到 SRAM page 边界、weight buffer 切换或末次
accumulator clear/output opcode 时，原生模板发生变化，编译器必须保留独立
描述符；单条 `STREAM_ND` 只合并“原生指令不变且三维坐标仿射”的区域。

### VXM_STREAM_ND

`VXM_STREAM_ND` 将一条 96-bit VXM 紧凑配置包和一到三维绝对 cycle 启动域
放在同一条 ICU 宏指令中。ICU 取到描述符后锁存一次配置，再由 ND counter
产生全部启动点，不需要配置槽、slot 生命周期或 CONFIG/RUN 配对检查。

ND 迭代次数表示 ICU 启动次数。紧凑配置包内部的 `repeat_count` 独立保留，
决定每次启动后 Superlane 配置持续执行多久。v1 不做操作数字段归纳；量化
scale relocation 直接修改宏指令携带的 packet。

### SXM_TILE_PROGRAM

`SXM_TILE_PROGRAM` 将一个完整 transpose 或 permute 模板与一到三维启动域放在
一起。payload 保留源/目标 stream 列表、row/tile 选择器和完整 32-lane map。
每个半球的 transpose 与 permute 仍使用独立队列，可以并行；循环出现的 tile
map 会成为多条可交错 program，而不再逐 cycle 展开。

## 编译器压缩

binary lowering 会识别同一队列中“原生指令形状相同、cycle 间隔固定、
地址或 column/accumulator 步长固定”的重复窗口，先形成可交错的二维调度，
再把重复调度折叠为第三个仿射维度。该规则面向所有队列，不是 FFN 专用
helper。

压缩在 binary lowering 之前就有明确的 IR 表达：

- Schedule `mem_read`/`mem_write` 使用地址 wave；
- Schedule `mxm_load` 使用 outer group 表达重复 IW 窗口；
- Schedule `mxm_compute` 与 `mxm_accumulate` 共享 wave，归纳变量可推进
  accumulator address；
- Command `mem_bundle` 只保存一次物理 slice lane 集合，同时保留每条 lane
  自己的 queue、cycle、address 和 stream selector。

Schedule verifier 会展开所有逻辑发射点再检查资源占用，因此紧凑表示不会放宽
逐 cycle 精度。

SmolLM2-135M、seq_len=32、Vector FFN 的结果如下：

| 指标 | 旧队列 | Macro v1 | 降幅 |
| --- | ---: | ---: | ---: |
| 队列命令数 | 53,511 | 6,214 | 88.4% |
| Binary 字节数 | 1,620,663 | 412,565 | 74.5% |
| `max_cycle` | 46,421 | 46,421 | 不变 |

真实权重 Qwen2.5-1.5B 第 0 层 FFN、`seq_len=32`、每半球一个 Vector MXM、
每 superlane 128 KiB target 的结果如下：

| 产物 | 展开形式 | 层次化形式 | 降幅 |
| --- | ---: | ---: | ---: |
| Schedule IR | 510,631,417 B | 5,199,998 B | 99.0% |
| Command IR | 243,032,926 B | 2,248,327 B | 99.1% |
| Binary | 28,947,545 B | 885,357 B | 96.9% |
| `max_cycle` | 770,646 | 770,646 | 不变 |

压缩 binary 已通过 CModel 的 49,152 个 BF16 输出 golden 对比，mismatch 为
0；MAE 保持 `6.54569e-05`，最大误差保持 `0.00418091`。

对真实权重 Qwen2.5-1.5B 第 0 层完整 decoder、`seq_len=32`，依次加入 MEM
和 MXM N 维压缩后的结果为：

| 指标 | 二维 macro | MEM N-D | MEM + MXM N-D |
| --- | ---: | ---: | ---: |
| MEM 描述符 | 25,458 | 4,652 | 4,652 |
| MXM 描述符 | 3,221 | 3,221 | 196 |
| 全部队列命令 | 36,397 | 15,591 | 12,566 |
| Binary 字节数 | 1,752,208 | 1,053,968 | 945,118 |
| 展开后的功能指令 | 6,586,470 | 6,586,470 | 6,586,470 |
| 调度 `max_cycle` | 763,669 | 763,669 | 763,669 |

与仅使用 MEM N-D 的 binary 相比，`MXM_STREAM_ND` 将 MXM 描述符减少 93.9%，
全部队列命令减少 19.4%，binary 字节数减少 10.3%。相对最初的二维 macro
binary，全部命令和字节数分别减少 65.5% 和 46.1%。

新 binary 在 CModel 中执行 763,733 cycle，49,152 个 BF16 输出 mismatch 为
0，MAE 为 `0.004514`，最大误差为 `0.09375`。

在同一 decoder 上继续加入 `VXM_STREAM_ND` 和 `SXM_TILE_PROGRAM` 后：

| 指标 | MEM + MXM N-D | 全部粗粒度 ICU 形式 | 降幅 |
| --- | ---: | ---: | ---: |
| VXM 编码命令 | 5,536 | 85 stream descriptor | 98.5% |
| SXM 编码命令 | 2,182 | 112 tile program | 94.9% |
| 全部队列命令 | 12,566 | 5,045 | 59.9% |
| Binary 字节数 | 945,118 | 347,235 | 63.3% |
| 展开后的功能指令 | 6,586,470 | 6,586,470 | 不变 |
| 调度 `max_cycle` | 763,669 | 763,669 | 不变 |

粗粒度 binary 也通过同一套 49,152 点 CModel golden 对比：mismatch 为 0，
MAE 为 `0.004514`，最大误差为 `0.09375`。

继续加入 `MEM_SLICE_PROGRAM` 后，剩余 4,652 项 MEM N-D 操作会按物理 slice
和启动域组合：

| 指标 | 之前的全部粗粒度形式 | 加入 MEM slice program | 降幅 |
| --- | ---: | ---: | ---: |
| MEM 队列命令 | 4,652 | 1,628 条 program | 65.0% |
| MEM program body entry | - | 4,652 | 语义不变 |
| 全部队列命令 | 5,045 | 2,021 | 59.9% |
| Binary 字节数 | 347,235 | 263,881 | 24.0% |
| 展开后的功能指令 | 6,586,470 | 6,586,470 | 不变 |
| 调度 `max_cycle` | 763,669 | 763,669 | 不变 |

更新后的 binary 在 CModel 中仍执行 763,733 cycle，并通过相同的 49,152 点
Qwen golden 对比：mismatch 为 0、MAE 为 `0.004514`、最大误差为
`0.09375`。

### MEM slice A/B 策略与 inspector

`MEM_SLICE_PROGRAM` 仍是 `macro` 内部的一种子编码，不增加第四种压缩模式。
`ftlpu-compile`、`ftlpu-opt` 和 `ftlpu-translate` 可通过
`--mem-slice-program on|off` 单独开关它。lowering 不再生成单 body program，
因为其 11-word 共享头比一条 `MEM_STREAM_ND` 更大。

`ftlpu_binary_inspect left.ftlpu --compare right.ftlpu` 会把两个 binary 展开为
精确的 `(queue, cycle, native instruction)` 稀疏时间线。直到 `max_cycle` 为止，
没有功能发射的点都视为逻辑 NOP；NOP 数、发射 cycle 或原生指令任一变化都会
返回非零并输出首个差异。`ftlpu-compile` 和 `ftlpu-translate` 还支持
`--verify-icu-issues`，在写 binary 前将所选编码与 `none` 基线比较。

分组不会降低硬件 context 成本：每个 body entry 仍对应一个活跃 N-D context，
与多条独立 `MEM_STREAM_ND` 相同。因此 runtime 容量 inspector 现在按 body
entry 而不是父 program 统计峰值 context。收益来自描述符存储以及 fetch/decode
带宽。当前结论是：鉴于 decoder 实测降幅明显，保留该编码及独立开关；RTL
应将 body 顺序装载进现有 MEM N-D context calendar，不应要求 16 路同时分配，
也不应增加第二套 issue engine。硬件 capability 不支持时使用 `off` 回退。

## 本地 ICU 结构

每个功能单元 ICU 包含 descriptor FIFO 和 next-issue calendar。多个描述符
可以同时 in-flight，因此不同 wave 可以穿插发射。CModel 使用按下一次发射
cycle 排序的最小堆；RTL 可采用小型有序 calendar、timing wheel 或有限路
比较树。

以下情况会立即报错：错过目标 cycle、同一队列同一 cycle 有两个描述符到期、
非法迭代空间、macro 活跃时混入旧式控制流。编译器在写 binary 前也会检查
同队列 cycle 冲突。

## RTL 定型建议

MEM/MXM `STREAM_ND` 已采用上述固定 320-bit 硬件包。当前文件 envelope 仍用于
磁盘传输和 relocation，装载器输出才是 ICU 接口上的最终 bit pattern。
`MEM_SLICE_PROGRAM`、`VXM_STREAM_ND` 和 `SXM_TILE_PROGRAM` 仍属于软件验证
格式，后续需要各自冻结固定 payload。最大 in-flight 数应纳入 target model；
硬件 capability bit 正式可用之前，runtime 保留旧细粒度队列作为回退路径。
