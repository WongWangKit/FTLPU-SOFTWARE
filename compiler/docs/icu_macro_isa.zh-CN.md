# ICU 粗粒度调度 ISA v1

## 范围

v1 将确定性的 MEM/MXM 指令展开从 host runtime 下沉到各功能单元的本地
ICU。它不改变 Schedule IR 的 cycle，也不修改已有 MEM/MXM 原生指令。
VXM、SXM 暂时保留旧队列格式。

通过以下选项启用：

```text
ftlpu-opt ... --icu-macro-schedule
```

生成的 module 带有 `ftlpu.icu_macro_schedule = true`，binary lowering
会在当前文件 envelope 中生成带类型的扩展 ICU 描述符。

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
时间上不允许重叠。

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

## 本地 ICU 结构

每个功能单元 ICU 包含 descriptor FIFO 和 next-issue calendar。多个描述符
可以同时 in-flight，因此不同 wave 可以穿插发射。CModel 使用按下一次发射
cycle 排序的最小堆；RTL 可采用小型有序 calendar、timing wheel 或有限路
比较树。

以下情况会立即报错：错过目标 cycle、同一队列同一 cycle 有两个描述符到期、
非法迭代空间、macro 活跃时混入旧式控制流。编译器在写 binary 前也会检查
同队列 cycle 冲突。

## RTL 定型建议

当前 extension-word 格式属于软件验证格式。硬件版建议使用固定描述符头、
FU 专用 payload、显式版本和长度，并将最大 in-flight 数纳入 target model；
超长 count/interval 可使用 escape word。硬件 capability bit 正式可用之前，
runtime 必须保留旧细粒度队列作为回退路径。
