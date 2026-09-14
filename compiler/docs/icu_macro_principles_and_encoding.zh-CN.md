---
title: "ICU Macro 原理与压缩编码说明"
subtitle: "二维 Macro v1 · 独立 QueueMode Reference Codec"
date: "2026-09-14"
lang: zh-CN
---

# 文档范围

本文说明当前二维 ICU Macro v1 的工作原理，以及独立 QueueMode 下第一版
reference physical codec 的编码方法，重点包括 Template、Run、Delta、Dictionary、
Compact、Extended 和固定宽 i-MEM 取指。

当前验证范围只包括 MEM/MXM 二维 Macro。三维 `STREAM_ND`、
`MEM_SLICE_PROGRAM` 和 VXM/SXM coarse program 不属于本文性能签核范围。

# 两层压缩

Macro 压缩分为两个相互独立的层次：

1. 编译器语义压缩：把大量重复的 functional issue 合并成二维 Macro。
2. 物理编码压缩：使用 Template、Run、Delta 和 Dictionary 压缩 Macro descriptor
   在 i-MEM 中的存储。

```text
大量动态 functional issue
          │ 二维 Macro 识别
          ▼
较少的语义 Macro descriptor
          │ Template + Delta-RLE
          ▼
紧凑的物理 i-MEM bitstream
```

# 二维 Macro 语义

一个 Macro 自带一条 native functional instruction 和一个二维调度域。

| 字段 | 含义 |
|---|---|
| `start_cycle` | 第一次发射的绝对周期 |
| `inner_count` | 内层循环次数 |
| `inner_interval` | 内层相邻发射的周期差 |
| `inner_stride` | 内层每次 operand 增量 |
| `outer_count` | 外层 wave 数量 |
| `outer_interval` | 相邻 outer wave 的起始周期差 |
| `outer_stride` | 每个 outer wave 的 operand 增量 |
| `induction_target` | 被递增的 operand |

第 `(outer, inner)` 个发射点为：

```text
issue_cycle = start_cycle
            + outer × outer_interval
            + inner × inner_interval

operand = base_operand
        + outer × outer_stride
        + inner × inner_stride
```

`induction_target` 的解释由 queue kind 决定：

| Queue | 被归纳 operand |
|---|---|
| MEM | SRAM address |
| MXM load | weight column |
| MXM compute | accumulator address |
| MXM dequant | 无 operand induction |

## Shape

| Shape code | 名称 | 条件 |
|---:|---|---|
| `00` | Singleton | `inner_count=1, outer_count=1` |
| `01` | Inner1D | `inner_count>1, outer_count=1` |
| `10` | Outer1D | `inner_count=1, outer_count>1` |
| `11` | Full2D | `inner_count>1, outer_count>1` |

当前编译器会识别一维重复、已经带 Repeat 的序列以及相邻仿射列组成的二维矩形。
合并前后必须保持完全相同的 `(queue, cycle, native instruction)` 时间线，同一个
queue 在同一个 cycle 不允许产生两条 issue。

# 独立 QueueMode

每个物理 MEM/MXM ICU queue 静态选择一种模式：

```text
00 = Native + Repeat + Repeat2D
01 = Macro2D packed
10 = Reserved
11 = Reserved
```

这里的 QueueMode 不表示增加两套物理 queue。它是同一块 i-MEM 的静态解释模式。

在 MacroMode 中：

- 所有 functional instruction 都使用 Macro grammar；
- Repeat 进入 inner dimension；
- Repeat2D 进入 inner/outer dimension；
- NOP 不保存，由下一条 Macro 的绝对 `start_cycle` 表示；
- 无法继续合并的 functional instruction 变成 Singleton Macro；
- 同一个 queue image 内不混合 legacy native/Repeat 和 Macro record；
- 因为 mode 和 queue kind 是静态 metadata，所以每条 Macro 不需要 type tag。

# 固定宽 i-MEM 与变长 Macro record

160 bit 是解码后的 active Macro context 预算，不是 i-MEM 中 descriptor 的固定长度。

物理 i-MEM SRAM 保留现有固定宽读口：

| Queue | 固定读取宽度 |
|---|---:|
| MEM | 96 bit/access |
| MXM load/compute/dequant | 128 bit/access |

变长 Macro record 可以跨 word 边界。固定宽 word 先进入 bit reservoir，再由 decoder
按实际字段长度消费：

```text
固定宽 i-MEM → bit reservoir → Macro decoder
                                  ↓
                           160-bit context RAM
                                  ↓
                         next-issue calendar → FU
```

每个 queue 的物理 word 数为：

```text
MEM words = 1 + ceil(payload_valid_bits / 96)
MXM words = 1 + ceil(payload_valid_bits / 128)
```

word 0 是所有 queue 共用的前置控制字，低 28 bit 定义如下，其余 bit 必须为零：

| Bit | 字段 | 含义 |
|---:|---|---|
| 0 | `valid` | image 已完整提交，可以被 ICU 锁存 |
| 1 | `enable` | queue 参与本次 program launch |
| 3:2 | `mode` | `00=Native`、`01=Macro`、`10/11=Reserved` |
| 27:4 | `command_count` | 24-bit 存储 record 数 |

`command_count` 是 decoder 必须恢复的 Macro record 数，不是 run 数，也不是
Macro 展开后的动态 functional issue 数。变长 record 由 run header、template 标志
和 delta 前缀自描述边界；decoder 恢复指定数量的 record 后停止创建 context，等
已有 context 排空后 queue 完成。

queue kind 由每个 ICU 独立的物理 i-MEM 隐含，不进入控制字。第一版硬件也不设置
独立 `codec_version`；`mode=01` 固定表示当前 Macro packed v1。最后一个 word 的
无效尾部补零。硬件通过 `command_count` 和 grammar 决定停止位置，并以 i-MEM
深度作为越界保护，不保存 `valid_bit_length`。

`.ftlpu` 软件 container 仍保留自己的 binary version 和 packed-image version，
以及 `valid_bit_length`，用于拒绝不兼容或截断文件、确定 DMA 字节数和执行
bit-exact round-trip。这些是软件信息，不是硬件控制字字段。自
`.ftlpu` v33 起，MEM、MXM load、MXM compute 和 MXM dequant 的 all-Macro
queue 均使用 packed container image；v28～v32 的数值 3 仍按旧的 MEM-only
packed 格式读取。

当前 24-bit count 是按现有最大 i-MEM 总 bit 数选取的保守统一位宽，而不是按
i-MEM word 地址位宽选取；一个 word 可以包含多个短 delta transition。现有深度
暂时保持 MEM `131072×96 bit`、MXM `65536×128 bit`，等多模型物理编码容量重新
测量后允许调整，codec grammar 不随深度改变。

`valid/enable/mode/count` 是静态 launch 配置。`enable` 在 START 到 DONE 期间不得
动态修改，也不用于实现 Sync；Sync 是 active queue 内部等待 notification token
的执行状态。由于 SRAM 内容通常不随逻辑 reset 清零，系统仍需一个 i-MEM 外部、
可复位的 START/armed 门控，确保 payload 和 word 0 提交完成前 ICU 不会取指。

# Queue bitstream 总体格式

概念上的 Macro queue image 为：

```text
[Dictionary]
[First absolute state]
[Run 0 header][Template 0][record deltas...]
[Run 1 header][Template 1][record deltas...]
...
[tail zero padding to 96/128-bit word]
```

有效 bit 数：

```text
queue_bits = 3 + dictionary_count × 36
           + 32 + initial_operand_bits
           + Σ(run_header_bits)
           + Σ(template_bits)
           + Σ(transition_delta_bits)
```

第一版 reference codec 在整数域内低位优先写入 bitstream。最终 RTL ABI 定型时
仍需把 word 内 bit numbering、word ordering 和 ECC block 边界写入正式规范。

# Template 与 Run

Template 保存相邻 Macro 中不变的部分：

- 归一化后的 native instruction；
- Shape；
- induction target；
- inner/outer count、interval 和 stride。

归一化会移除每条记录变化的 base operand：

- MEM address 清零；
- MXM load weight column 清零；
- MXM compute accumulator address 清零；
- MXM dequant 没有归纳 operand。

相邻记录的归一化 native instruction 和 schedule template 相同，就组成一个 Run。

## Run header

```text
multi_record_run       1 bit
run_length_minus_1    16 bit，仅 multi_record_run=1 时存在
template              variable bits
```

- 单条 Run header 为 1 bit；
- 多条 Run header 为 17 bit；
- 一个 Run 最多覆盖 65,536 条 Macro；
- 整个 queue 除第一条记录外共有 `command_count-1` 个 transition delta。

# MEM Compact Template

MEM Compact Template 的基础字段为：

```text
extended = 0           1 bit
shape                  2 bit
is_write               1 bit
stream                  6 bit
preserve_stream         1 bit
--------------------------------
基础字段               11 bit
```

当前 Compact MEM 条件包括：

- opcode 是 Read 或 Write；
- stream 不超过 63；
- `map_stream=0`；
- induction target 是 `MemAddress`；
- schedule 字段满足下面的 Compact 范围。

## Inner 字段

Inner1D/Full2D 额外保存：

| 字段 | 位宽 | 范围 |
|---|---:|---:|
| `inner_count_minus_1` | 11 | count 1～2,048 |
| `inner_interval_minus_1` | 8 | interval 1～256 |
| `inner_stride` | 10 signed | −512～511 |

Inner 字段合计 29 bit。

## Outer 字段

Outer1D/Full2D 额外保存：

```text
inner_span = (inner_count - 1) × inner_interval
outer_residual = outer_interval - inner_span
```

| 字段 | 位宽 | 范围 |
|---|---:|---:|
| `outer_count_minus_1` | 9 | count 1～512 |
| `outer_residual_minus_1` | 15 | residual 1～32,768 |
| `outer_stride` | 12 signed | −2,048～2,047 |

Outer 字段合计 36 bit。

## MEM Compact Template 长度

| Shape | Template长度 |
|---|---:|
| Singleton | 11 bit |
| Inner1D | 40 bit |
| Outer1D | 47 bit |
| Full2D | 76 bit |

这些长度只表示 Template，不包括 Dictionary、第一条绝对状态、Run header 和
record delta。

# MXM Compact Template

MXM Template 中的归一化 native payload 宽度为：

| Queue | native template |
|---|---:|
| MXM load | 16 bit |
| MXM compute | 49 bit |
| MXM dequant | 16 bit |

Compact MXM Template：

```text
extended = 0           1 bit
normalized native      16/49 bit
shape                  2 bit
induction_target       2 bit
inner fields           optional 29 bit
outer fields           optional 36 bit
```

| Shape | Load/Dequant | Compute |
|---|---:|---:|
| Singleton | 21 bit | 54 bit |
| Inner1D | 50 bit | 83 bit |
| Outer1D | 57 bit | 90 bit |
| Full2D | 86 bit | 119 bit |

# Extended Template

字段超出 Compact 范围时使用 Extended Template：

```text
extended = 1
normalized/full native instruction
inner_count            32 bit
inner_interval         32 bit
inner_stride           32 bit signed
outer_count            32 bit
outer_interval         32 bit
outer_stride           32 bit signed
induction_target        2 bit
```

| Queue | Extended Template长度 |
|---|---:|
| MEM | 227 bit |
| MXM load | 211 bit |
| MXM compute | 244 bit |
| MXM dequant | 211 bit |

Extended 仍是二维 Macro，不表示三维 `STREAM_ND`。它只是在物理编码上使用完整
字段，因此通常跨越多个固定宽 i-MEM word。

# Singleton 与 Extended 的区别

Singleton 和 Extended 不属于同一分类维度：

- Singleton 是语义 Shape，表示 `inner_count=1, outer_count=1`，只发射一次。
- Extended 是物理编码形式，表示字段不能用 Compact Template 表示。

因此可以存在：

| 语义 Shape | Compact | Extended |
|---|---:|---:|
| Singleton | 可以 | 可以 |
| Inner1D | 可以 | 可以 |
| Outer1D | 可以 | 可以 |
| Full2D | 可以 | 可以 |

一个 Compact Singleton 仍可以通过 Run 和 Delta 与其他 Singleton 共享 Template。
一个 Extended Full2D 仍然只有二维语义，只是字段范围更大。

# Delta 编码

相邻 Macro 的变化表示为：

```text
start_delta = current.start_cycle - previous.start_cycle
operand_delta = current.base_operand - previous.base_operand
```

第一条记录保存绝对 start cycle 和绝对 base operand：

| Queue | first start | initial operand | 合计 |
|---|---:|---:|---:|
| MEM | 32 bit | 13-bit address | 45 bit |
| MXM load | 32 bit | 2-bit weight column | 34 bit |
| MXM compute | 32 bit | 13-bit accumulator address | 45 bit |
| MXM dequant | 32 bit | 无 | 32 bit |

# Delta Dictionary

每个 queue 统计最常出现的 Compact delta，选择频率最高的七项作为 queue-local
Dictionary。

```text
dictionary_count       3 bit

每个 Dictionary entry：
start_cycle_delta     22 bit unsigned
operand_delta         14 bit signed
-------------------------------------
                       36 bit
```

完整 Dictionary 最大为：

```text
3 + 7 × 36 = 255 bit
```

## Dictionary 前缀码

| Entry | Prefix | 长度 |
|---:|---|---:|
| 0 | `0` | 1 bit |
| 1 | `10` | 2 bit |
| 2 | `1100` | 4 bit |
| 3 | `1101` | 4 bit |
| 4 | `11100` | 5 bit |
| 5 | `11101` | 5 bit |
| 6 | `11110` | 5 bit |
| Escape | `11111` | 5 bit |

最常见的 delta 因此只需要 1 bit。

## Compact Escape

未命中 Dictionary，但 delta 能放进 Compact 字段时：

```text
escape prefix          5 bit  = 11111
wide                    1 bit  = 0
start_cycle_delta      22 bit
operand_delta          14 bit signed
-------------------------------------
总计                   42 bit
```

范围：

- start delta：0～4,194,303；
- operand delta：−8,192～8,191。

## Wide Escape

Compact 字段仍放不下时：

```text
escape prefix          5 bit  = 11111
wide                    1 bit  = 1
start_cycle_delta      32 bit
operand_delta          32 bit signed
-------------------------------------
总计                   70 bit
```

# 编码示例

假设同一个 MEM queue 中有三条 Macro：

```text
Macro 0: Read(stream=2, base=100), start=10
Macro 1: Read(stream=2, base=112), start=42
Macro 2: Read(stream=2, base=124), start=74
```

三条 Macro 的 Shape 和二维 schedule 相同。归一化后都变成：

```text
Read(stream=2, address=0) + common schedule template
```

相邻变化为：

```text
Macro 0 → Macro 1: delta=(32,12)
Macro 1 → Macro 2: delta=(32,12)
```

Dictionary 选择：

```text
dictionary[0] = (32,12)
```

最终保存：

```text
dictionary_count=1
dictionary[0]=(32,12)
first_start=10
first_address=100
run_length=3
one shared template
delta_ref=0
delta_ref=0
```

两个后续状态变化各只需要一个 1-bit Dictionary reference。

## Singleton 示例

一个 queue 只有一条 Compact MEM Singleton：

```text
dictionary_count         3 bit
first start_cycle       32 bit
first address           13 bit
single-run flag          1 bit
singleton template      11 bit
--------------------------------
有效长度                60 bit
```

物理占用仍是一个 96-bit MEM word，尾部 36 bit padding。因此 Singleton Macro
并不必然比一条 96-bit native instruction 更长。

# 解码与执行

decoder 依次执行：

1. 从固定宽 i-MEM word 向 bit reservoir 补充数据；
2. 读取 Run header 和 Template；
3. 读取第一条绝对状态或后续 Delta；
4. 使用 Dictionary 或 Escape 恢复 cycle/operand；
5. 把 base operand 写回归一化 native instruction；
6. 构造完整二维 Macro；
7. 初始化一个 160-bit active context；
8. 把 context 加入 next-issue calendar。

next-issue calendar 选择最早到期的 context，计算当前 inner/outer 对应的 operand
delta，发射 native instruction，并更新循环计数。Macro 未完成时重新进入 calendar。

推荐第一版硬件吞吐约束：

- i-MEM 每周期最多读取一个固定宽 word；
- decoder 每周期最多生成一个 context；
- context RAM 每周期一个写口；
- functional queue 每周期最多 issue 一条指令；
- descriptor 不完整时允许等待后续 word；
- issue 端与 bitstream fetch/decoder 解耦。

# Qwen seq128 当前测量

第一版 reference codec 已对全部 Macro queue 做逐条 encode/decode round-trip：

- MEM queue：208；
- MXM queue：6；
- Macro record：2,434,228；
- 解码后的 native payload、start cycle 和二维 schedule 逐条一致。

| 资源 | 有效 bit | 固定宽物理 word |
|---|---:|---:|
| MEM | 8,674,274 | 90,460 × 96 bit |
| MXM load | 84,416 | 660 × 128 bit |
| MXM compute | 148,508 | 1,161 × 128 bit |
| MXM dequant | 3,474 | 28 × 128 bit |

汇总：

- 有效数据：1,113,834 bytes；
- 按每个 queue 分别向固定 word 取整：1,115,104 bytes；
- tail padding：1,270 bytes；
- 同一 Command IR 的 Repeat+Repeat2D Macro-capable i-MEM：173.289 MB；
- 独立 MacroMode packed i-MEM：1.115 MB；
- 减少 99.357%，约 155.6 倍。

当前 MEM 中共有 2,429,034 条 Macro 和 34,674 个 Template Run，平均约 70 条
Macro 共享一个 Template，且 Qwen seq128 中没有 Extended Template。MEM 平均
约为 3.57 valid bit/Macro record，说明主要收益来自 Template Run 和高频 Delta
Dictionary，而不仅是单字段位宽缩短。

# 当前边界与后续工作

当前已经完成：

- MEM/MXM bit-exact reference encoder/decoder；
- Compact/Extended Template；
- Run、Delta Dictionary、Compact/Wide Escape；
- 固定 96/128-bit word 容量取整；
- runtime 到 CModel 前的 encode/decode round-trip；
- 2-bit QueueMode：`00=Native`、`01=Macro`、`10/11=Reserved`；
- 前置 i-MEM 控制 word：`valid/enable/mode/24-bit command_count`；
- dictionary count 作为 Macro payload 的起始 3 bit；
- `valid_bit_length` 仅作为软件 container metadata；
- `.ftlpu` v33 的 MEM/MXM packed container writer、reader 和 metadata skip；
- Qwen seq128 容量测量。

尚未完成或尚未冻结：

- 多模型复测后的最终 i-MEM 深度；
- ECC、checkpoint 和 block boundary；
- reservoir 最小深度；
- 固定一 word/cycle 下的 decoder 周期级吞吐；
- context-full backpressure 和启动预取要求；
- RTL 面积、时序和功耗综合。

容量统计已经使用固定 96/128-bit word；后续必须增加 cycle-accurate fetch/reservoir
模型，验证 decoder 能在每条 Macro 的 `start_cycle` 前完成 context 建立。
