# FU 专用 ICU 循环 ISA 与旧 Macro 兼容格式

## 范围

本文区分当前硬件可见的 FU 专用循环指令，以及旧的 `Macro`、`STREAM_ND`
序列化格式。projection 只是编译器意图，不是 ICU 指令，也不是硬件中的 program
对象。每个物理 ICU 只接收其功能单元能够执行的指令。

当前 raw 本地 iMEM 格式如下：

| 物理队列 | FU 本地指令 | 一条逻辑包 |
| --- | --- | ---: |
| MEM | `READ_3D`、`WRITE_3D`、`WRITE_TAP_3D` | 连续 3 个 96-bit word |
| MEM | `WRITE_READ_2D` | 连续 3 个 96-bit word |
| MEM | `MEM_WRITE_SYNC` | 连续 2 个 96-bit word |
| MXM load | `LOAD_3D` | 连续 2 个 128-bit word |
| MXM dequant | `DEQUANT_3D` | 连续 2 个 128-bit word |
| MXM compute | `COMPUTE_3D`、`ACCUMULATOR_READ_3D` | 连续 2 个 128-bit word |
| VXM | `RUN_2D` + 96-bit compact config | 连续 3 个 96-bit word |
| SXM | `RUN_2D` + 416-bit tile-local config | 连续 6 个 96-bit word |

队列类型直接决定 decoder，因此这些包不携带共享的 MEM/MXM unit selector，也
不采用统一 320-bit 格式。每个本地 iMEM entry 就是一个物理 96-bit 或 128-bit
word，所以一条完整 MEM 3D 或 VXM 包占 3 槽，一条完整 MEM 同步写或 MXM 包
占 2 槽，一条完整 SXM 包占 6 槽。

`MEM_SLICE_PROGRAM` 尚未冻结硬件包布局。旧 `VXM_STREAM_ND` 和
`SXM_TILE_PROGRAM` 仍是兼容文件格式，但其 rank 不超过 2、无操作数归纳的
子集会闭式转换为各自的 `RUN_2D`，不会展开为逐 cycle 指令。

当前 `ScheduleToCommand` 会把每个算子声明的 VXM/SXM schedule 域直接编码为
`command.vxm_run_2d` 和 `command.sxm_run_2d` 物理包。VXM 的连续
`repeat_interval=1` 留在 compact config；有间隔的重复由 ICU 第一层 counter
表达。带运行时 scale relocation 的旧 VXM 队列暂时整体保留兼容格式，避免在
同一个物理队列混用 raw 包和旧 command。

## 为什么各 FU 的循环维数不同

维数按功能单元实际需要的独立硬件状态确定，不提供统一四维 decoder：

| 功能单元 | 采用的循环形式 | 维度所表达的工作 |
| --- | --- | --- |
| MEM | 最多 3D | 连续块、tile、外层 group，并同时归纳 SRAM 地址 |
| MXM | 最多 3D | K partial、M row/tile、N tile，并归纳 weight column 或 accumulator 地址 |
| VXM | compact `repeat_count` + ICU `RUN_2D` | compact 内部连续元素；ICU 外层 row/token 与 tile/head/group |
| SXM | tile-local 指令 + ICU `RUN_2D` | 32-lane map 留在 tile 指令内；ICU 只重复 row 和 tile |
| C2C | burst/vector count + completion event | 搬运长度和向量批次；依赖用完成事件表达，不套算术 N-D 循环 |

因此 MEM/MXM 保留三维；VXM 不再编码一个恒为 1 的第三个 counter；SXM 也不把
lane map 当成循环维；C2C 的 producer/consumer 关系由事件控制。若算子还出现
第 4 个逻辑轴，lowering 应在字段范围内折叠到已有轴，或在 page、placement、
FU 模板变化处切成另一条粗指令。增加通用 4D 会扩大每个 ICU 的 context、加宽
计数和地址生成路径，并延长单 context 的占用时间，对当前 LLM 算子没有对应收益。

## 旧编译器模式

标准 `ftlpu-stream-to-schedule` 是 direct-lowering pipeline，不是压缩
pipeline。它固定设置 `ftlpu.icu_compression = "none"`，关闭
`MEM_SLICE_PROGRAM`，并用 `ftlpu.command_lowering = "direct"` 标记结果。
`ftlpu-stablehlo-to-schedule` 和 `ftlpu-stream-to-uncompressed-schedule`
入口采用同一 direct 约束。

只有显式选择兼容 pipeline `ftlpu-stream-to-compressed-schedule` 或
`ftlpu-compress-schedule` 时才会加入 `ScheduleCompression`。下面的压缩选项只
控制这些旧兼容 pipeline；标准 schedule pipeline 后面没有隐藏的压缩步骤。

通过命令行选择 ICU 压缩模式：

```text
ftlpu-opt ... --icu-compression none|control|macro
```

`macro` 是默认的旧序列化模式。`none` 会物化功能指令，但保留按时长编码的
NOP 间隔；`control` 启用 Repeat 和 Repeat2D；`macro` 启用旧的带类型 Macro
envelope。旧选项
`--icu-macro-schedule` 继续作为 `--icu-compression macro` 的兼容别名。
这些选项描述兼容 pipeline，不选择统一硬件包。

生成的 module 带有 `ftlpu.icu_compression = "..."`。在 Command IR 兼容期内，
仍会同时携带旧的 `ftlpu.icu_macro_schedule` 布尔属性。binary lowering 可以
在文件 envelope 中保留带类型的 `Macro` 和 `STREAM_ND` 描述符。完成
relocation 后，runtime loader 将可支持的 MEM/MXM 子集闭式转换为上表同一套
FU 专用 raw 包，不会逐点物化原生指令。

## 旧描述符语义

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

旧 binary 格式在原生指令 payload 后附加 `MACR` 标记和 8 个字段。这个可变长
记录只是磁盘 envelope，不是硬件指令。

坐标 `(outer, inner)` 的实际发射为：

```text
cycle = start_cycle + outer * outer_interval + inner * inner_interval
operand_delta = outer * outer_stride + inner * inner_stride
```

### 旧 MEM_STREAM_ND

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

旧 binary envelope 中 count 和 cycle stride 为无符号 32 位，address stride
为有符号 32 位；各维在发射时间上不允许重叠。装载时，adapter 还会检查当前
硬件包更窄的字段范围，以及单 context 的 live interval 约束。

### 旧 MEM_SLICE_PROGRAM

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

### 旧 MXM_STREAM_ND

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
同一物理队列只有一份 decoded loop 状态，因此各粗指令的 live interval 不得重叠。

## 当前 FU 专用 raw 包编码

MEM/MXM 包都以同一份闭式三维循环开始，counter 0 为最内层：

| 逻辑 payload 字段 | 宽度 | 含义 |
| --- | ---: | --- |
| `wait_cycle` | 24 | 本条包第一次 FU 发射前的队列内等待周期 |
| `count[d] - 1` | 每维 16 | counter 0、1、2 的迭代次数 |
| `cycle_stride[d]` | 每维 24 | counter `d` 每前进一步增加的 cycle |

对坐标 `i`，ICU 只把指令内局部计数器和以下相对 offset 比较：

```text
issue_offset = wait_cycle + sum(i[d] * cycle_stride[d])
```

ICU 不接收程序全局 cycle。Command IR 中的绝对 cycle 只作为编译器排程元数据；
写二进制时，队列空档优先写入下一条包的 `wait_cycle`；若整段空档超过
24-bit 字段范围，则使用前置 `NOP`。
count 范围为 1 到 65,536，cycle stride 必须非零，最后一次相对发射 offset
必须落在 24-bit 调度域内。未使用维采用 `count=1`；由于 counter 不推进，
其 stride 不产生影响。

每个物理 word 都重复一个 8-bit FU 本地 header：

| 物理 bit | 含义 |
| ---: | --- |
| `[1:0]` | 全局 ICU envelope opcode `Extended`（`0b11`） |
| `[3:2]` | 当前 word 在固定长度包内的序号 |
| `[5:4]` | FU 本地 operation |
| `[6]` | FU 本地三维包标志，`1` |
| `[7]` | 格式 version，当前为 `0` |

word 0 的物理 `[91:88]` 还携带 extended subtype `2`，用于和 Repeat2D 区分。
后续 word 重复 marker、version、operation 和期望 word index，因此本地 decoder
可以发现截断或乱序。

下表用 `P[n]` 表示逻辑 payload bit。编号跳过每个 word 的低 8-bit header 和
word 0 的 subtype 字段。公共循环占 `P[143:0]`：

| Payload bit | 字段 |
| ---: | --- |
| `P[23:0]` | 第一次 launch 前的 `wait_cycle` |
| `P[39:24]`、`P[55:40]`、`P[71:56]` | `count[0..2] - 1` |
| `P[95:72]`、`P[119:96]`、`P[143:120]` | `cycle_stride[0..2]` |

### MEM：3 x 96 bit

本地 operation 为 `0=read`、`1=write` 或 `2=write-tap`。260 个 payload bit
如下：

| Payload bit | 字段 |
| ---: | --- |
| `P[143:0]` | 公共循环 |
| `P[149:144]` | 打包后的 stream selector |
| `P[162:150]` | bank 内 base row |
| `P[178:163]` | `outer_group_size - 1` |
| `P[198:179]` | 有符号 inner 地址 stride |
| `P[218:199]` | 有符号 middle 地址 stride |
| `P[238:219]` | 有符号 outer 组内地址 stride |
| `P[258:239]` | 有符号 outer 组间地址 stride |
| `P[259]` | 保留，必须为 0 |

`outer_group_size` 必须是 2 的幂。取 1 时就是普通三维仿射地址；大于 1 时用
shift 和 mask 表达 blocked outer 布局。

对坐标 `(i0,i1,i2)`，blocked 地址生成器计算：

```text
address = base + i0*a0 + i1*a1
        + (i2 & (G - 1))*a2 + (i2 >> log2(G))*ag
```

三个 cycle stride 同时编码循环边界的等待，不需要在一条包内部插入 NOP。
counts 为 `(C0,C1,C2)`、cycle stride 为 `(S0,S1,S2)` 时，普通 inner 发射后、
inner 循环结束后、middle 循环结束后的空闲周期分别为 `S0-1`、
`S1-(C0-1)*S0-1` 和 `S2-(C1-1)*S1-(C0-1)*S0-1`，三者都必须非负。
队列 NOP 只用于包启动前，或两个无法落入同一仿射循环的域之间。

### MEM WRITE_READ_2D：3 x 96 bit

这条指令让同一物理 MEM bank 在一个 ICU context 中执行两条静态二维事件流：
每个 `(i0,i1)` 写一次、读一次。写和读共用 SRAM 地址域，但有独立时间步长：

```text
write_cycle   = start_wait + i0*write_cycle_stride0 + i1*write_cycle_stride1
read_cycle    = start_wait + read_start_offset
              + i0*read_cycle_stride0 + i1*read_cycle_stride1
address       = base_address + i0*address_stride0 + i1*address_stride1
write_stream  = fixed_write_stream
read_stream   = read_stream_base + i1*read_stream_outer_stride
```

cycle 都相对该包激活，不是程序全局时钟。`start_wait` 表示包内部的前置等待；
超过字段范围的空档仍由队列 NOP 表达。stream 步长允许同一二维域的外层
迭代送往不同目的 stream。写和读分别维护二维计数器与下一发射时刻，
ICU 每周期最多给该 bank 发一条 MEM FU 指令。编译器需静态证明写读时间
不冲突、读前已写、读前未覆盖、stream 在相应周期有效；硬件不检查 FIFO
空满，也不根据数据到达时间改变排程。

包使用 MEM 本地 operation `3` 和 extended subtype `7`，总 payload 为
260 bit：

| Payload bit | 字段 |
| ---: | --- |
| `P[23:0]` | `start_wait` |
| `P[39:24]`、`P[55:40]` | `count[0..1] - 1` |
| `P[79:56]`、`P[103:80]` | 写 cycle stride 0、1 |
| `P[127:104]`、`P[151:128]` | 读 cycle stride 0、1 |
| `P[175:152]` | `read_start_offset` |
| `P[188:176]` | bank 内基础 row |
| `P[208:189]`、`P[228:209]` | 有符号 20-bit 地址 stride 0、1 |
| `P[234:229]`、`P[240:235]` | 写 stream、读 stream 基址 |
| `P[246:241]` | 有符号 6-bit 读 stream 外层步长 |
| `P[259:247]` | 保留，必须为零 |

固定三 word 的代价是两侧必须具有相同迭代次数、相同地址公式，且每个坐标
恰好一次写和一次读。独立地址域、多次读取或多种写源不能隐含在本指令中。

### MEM_WRITE_SYNC：2 x 96 bit

这条指令由上述 3D 指令所在的同一个、每 bank 唯一 MEM ICU 解码。word 0 使用
extended subtype 6，word 1 是原生 MEM `Write` 模板：

| word 0 物理 bit | 字段 |
| ---: | --- |
| `[1:0]` | Extended envelope |
| `[17:2]` | vector count 减一 |
| `[33:18]` | C2C synchronization tag |
| `[49:34]` | SR 传播延迟 |
| `[63:50]` | 有符号 SRAM row stride |
| `[87:64]` | 预留队列窗口周期数减一 |
| `[91:88]` | Extended subtype 6 |
| `[95:92]` | 保留，必须为 0 |

激活后，该指令至少占用 MEM ICU 到预留窗口结束。每个匹配的 C2C token 经过编码的
SR 延迟后产生一次原生 write。数据提前写完时，ICU 仍持有该指令直到窗口末尾；
数据迟到时，则完成最后一次 write 才退出。后续 read/write 不能越过它，DDR 提前
完成也不会把后面的静态 MEM 排程提前。

### MXM load：2 x 128 bit

| Payload bit | 字段 |
| ---: | --- |
| `P[143:0]` | 公共循环 |
| `P[144]` | weight-buffer base |
| `P[146:145]` | buffer parity mode |
| `P[148:147]` | weight-column base |
| `P[164:149]`、`P[180:165]`、`P[196:181]` | 有符号 column stride 0..2 |
| `P[201:197]` | east weight-input stream base |
| `P[202]` | 输入模式：INT8 dequant 或 direct 16-bit |
| `P[235:203]` | 保留，必须为 0 |

### MXM dequant：2 x 128 bit

| Payload bit | 字段 |
| ---: | --- |
| `P[143:0]` | 公共循环 |
| `P[159:144]` | BF16 scale bit |
| `P[235:160]` | 保留，必须为 0 |

### MXM compute：2 x 128 bit

MXM compute ICU 从同一固定长度包中解码两种 FU 本地 operation：
`0=COMPUTE_3D` 和 `1=ACCUMULATOR_READ_3D`。operation 会在每个物理 word 的
header 中重复，因此 `COMPUTE_3D` 两个 word 的低字节分别为 `0x43`、`0x47`，
`ACCUMULATOR_READ_3D` 则为 `0x53`、`0x57`。

`COMPUTE_3D` 使用以下 payload：

| Payload bit | 字段 |
| ---: | --- |
| `P[143:0]` | 公共循环 |
| `P[144]` | weight-buffer base |
| `P[146:145]` | buffer parity mode |
| `P[151:147]` | east activation-stream base |
| `P[156:152]` | west result-stream base |
| `P[169:157]` | accumulator base row |
| `P[183:170]`、`P[197:184]`、`P[211:198]` | 有符号 accumulator stride 0..2 |
| `P[224:212]` | accumulator row stride |
| `P[225]` | weight/activation data format |
| `P[228:226]` | 常规 destination/clear/output-format mode |
| `P[230:229]` | terminal dimension，`3` 表示关闭 |
| `P[233:231]` | terminal destination/clear/output-format mode |
| `P[235:234]` | 保留，必须为 0 |

terminal mode 允许一个闭式 compute 循环在指定维最后一点切换 destination、
clear 和 output format，不需要引入 projection 级硬件 program。

`ACCUMULATOR_READ_3D` 只使用同一物理 compute ICU 包中的 result、accumulator
地址和 mode 字段：

| Payload bit | 字段 |
| ---: | --- |
| `P[143:0]` | 公共循环 |
| `P[151:144]` | 保留，必须为 0 |
| `P[156:152]` | west result-stream base |
| `P[169:157]` | accumulator base row |
| `P[183:170]`、`P[197:184]`、`P[211:198]` | 有符号 accumulator stride 0..2 |
| `P[225:212]` | 保留，必须为 0 |
| `P[228:226]` | read destination/clear/output-format mode |
| `P[235:229]` | 保留，必须为 0 |

它在每个 loop 坐标发射一次原生 accumulator read；三个有符号 stride 归纳
accumulator 地址，result stream 和 read mode 保持不变。weight buffer、activation
stream、row stride、data format 和 terminal mode 字段不参与该 operation。

一条完整 MEM 包占 3 个本地 96-bit iMEM 槽；一条完整 MXM load、dequant 或
compute 包占 2 个本地 128-bit 槽。

### VXM：RUN_2D，3 x 96 bit

VXM 的 96-bit compact config 已包含 `repeat_count`，表示一次 launch 后 datapath
连续执行的元素数。`RUN_2D` 只再携带两个外层 launch counter：

| Payload bit | 字段 |
| ---: | --- |
| `P[23:0]` | 保留，必须为 0 |
| `P[39:24]`、`P[55:40]` | `count[0..1] - 1` |
| `P[79:56]`、`P[103:80]` | `cycle_stride[0..1]` |
| `P[135:104]` | compact control low 32 bit |
| `P[167:136]` | compact control high 32 bit |
| `P[199:168]` | compact immediate 32 bit |
| `P[259:200]` | 保留，必须为 0 |

外层坐标 `(i0,i1)` 的指令内 launch offset 为
`wait_cycle + i0*stride0 + i1*stride1`。每次 launch 只向 VXM 发出一次 compact
config，VXM 自己执行其中的 `repeat_count` 个连续元素。这样不会把连续元素重复
编码到 ICU counter，也不会在 compiler/runtime 中展开逐 cycle VXM 指令。

`VXM_STREAM_ND` 现在只是旧文件兼容描述符。rank 1/2 且无操作数归纳的输入会
闭式映射为上述 `RUN_2D`；rank 3 会被拒绝，因为 VXM 没有第三个外层 counter。

### SXM：RUN_2D，6 x 96 bit

SXM 的 tile-local payload 是现有 416-bit 编码，保留 transpose/permute opcode、
stream 列表、row/tile selector、16-lane tile map 和 32-lane permute map。
`RUN_2D` 在它前面放入与 VXM 相同的 104-bit 两维 launch 域：24-bit `wait_cycle`、
两个 count 和两个 cycle stride。对 permute，`P[521:520]` 表示每次 launch
按 8 lane 为单位旋转 map，可把 Qwen 中四个反复出现的 permutation phase 合成
一个循环；`P[523:522]` 保留且必须为 0。

SXM 包有 6 个 word，所以 FU-local header 用 `[4:2]` 三位 word index；队列类型
已经确定 SXM，header 不再重复 local operation，tile opcode 保留在 payload 内。
每条物理 SXM 队列只有一个 512-bit decoded context。

旧 `SXM_TILE_PROGRAM` 的 rank 1/2、无 operand induction 子集会闭式映射为该
`RUN_2D`。rank 3 或带 instruction-field induction 的描述符会被拒绝。

## 编译器直接 lowering

所有标准算子 emitter 都根据 tensor shape、物理 placement 和自身的闭式 timeline，
直接构造一段或多段分段仿射 launch 域。生成过程不会先为每个 token、tile 或
reduction step 建立一个点，再从这些点中识别重复模式。Attention projection、
QK/PV、softmax/RoPE、RMSNorm、elementwise，以及 FFN projection/SwiGLU 都遵守
这套约束。“所有算子闭式生成”指直接生成各自的域，不是强制所有 FU 都采用 3D
指令。

MEM Schedule op 把 repeat、wave、group 的 count、cycle stride 和 address stride
带到三个 MEM counter。MXM issue、load、dequant、compute 和 accumulator-read
同样携带最多三层 launch counter，以及对应 decoder 能支持的操作数归纳。VXM、
SXM 将 FU 本地工作保留在 compact 或 tile-local payload 内，只携带最多两层外部
launch counter。C2C 使用传输专用的闭式形式：连续传输用 burst/vector count
表示，producer/consumer 就绪关系用 completion event 及相应 ready/release
lifetime 表示，不再外套通用算术 3D 循环。

只有遇到物理队列、opcode 或 FU 模板、binding、page/bank、placement、C2C
route、字段范围、非仿射关系或 tail 边界时，算子才生成下一段。每段只创建一次，
emitter 不扫描已经生成的细粒度 Schedule op。`ScheduleToCommand` 随后把这些 FU
专用 Schedule 域映射为 MEM `READ_3D/WRITE_3D/WRITE_TAP_3D`、MXM
`LOAD_3D/DEQUANT_3D/COMPUTE_3D/ACCUMULATOR_READ_3D`、VXM `RUN_2D` 或 SXM
`RUN_2D`；direct 路径中没有相邻指令压缩，也不会事后恢复交错域。

已验证的 Qwen2.5-1.5B seq32 decoder layer 在第一版未归组 direct lowering 中有
41,968 个 MEM 域、44,290 个 FU 域，二进制为 2,366,141 bytes。将 QKV layout
block、residual phase 和 FFN SwiGLU 区域直接按 domain-major 归组后，分别降到
17,928 个 MEM 域、20,074 个 FU 域和 1,125,493 bytes；counter 展开后的工作量
仍然精确保持为 6,645,946 次 FU 发射。raw packet 字段校验和物理分析器均通过，
每个物理队列的 live decoded context 峰值都为 1。

其中主要的 MEM 降幅来自 QKV `10,840 -> 8,808`、两次 residual add
`4,896 -> 408` 和 `3,456 -> 288`，以及 FFN `18,152 -> 3,800`。这些归组在算子
emitter 根据 layout 和 timeline 创建域时直接完成，不通过扫描或压缩细指令恢复。

FFN Up 的专用闭式路径用 tensor shape、物理 placement、拓扑 route latency、闭式
timeline 和 dequant scale 调用 `lowerFfnUpToFu3D`。结果已经按物理 MEM、MXM
load、MXM dequant 和 MXM compute 队列划分；`materializeFfnUp3DCommands`
随后直接生成对应的 raw FU packet word。

这条路径没有逐 cycle Schedule event 列表，也不先生成细指令再识别和压缩。
loop count、cycle stride、地址或 accumulator 归纳、buffer 选择以及 terminal
compute 行为都直接由 shape、placement 和闭式 timeline 得到。遇到 page 边界、
placement 不连续或 FU 模板变化时，只为受影响的物理队列生成另一条粗指令。
direct 编译入口会给 module 标记 `ftlpu.command_lowering = "direct"`；二进制
翻译器只要看到任意旧 `CommandSequence` 就立即报错，禁止静默退回事后压缩。

Qwen2.5-1.5B Up projection（`M=32`、`K=1536`、`N=8960`）的 direct-lowering
测试生成 100 条 MEM 包，以及每类 MXM 各 2 条包：合计 106 条粗指令、312 个
包 word。各物理队列的空档现在编码到下一条包的 `wait_cycle`，不再额外插入
106 条 NOP；因此占用 312 个物理 iMEM word。ICU counter 将代表性队列展开
为预期的 26,880 次 activation、
26,880 次 load、26,880 次 dequant 和 215,040 次 compute 发射，全程不需要
编译器逐点物化。

完整 Qwen2.5 seq32 decoder 也用同一机制跨 projection page 和 output group
生成域。QKV 权重 READ 为 `720 -> 256`，O projection 为 `384 -> 16`，Gate 与
Up 合计为 `2,304 -> 192`；Down 因同队列 stream 和 buffer 复用形成真实顺序边界，
仍为 384。QKV 权重使用完整的 `wave_count=48, group_count=2` 域，不再保留原来的
`2+46` 切分。第一段重叠 Q-projection MXM 域从 cycle 8,294 开始，以 32-cycle
间隔覆盖全部 48 个 reduction，并与 RoPE 同时运行。activation bank 0 的 slice 8/9
保存 primary staging 副本，activation bank 0 的 slice 0/1 保存 pong 副本；direct
lowering 只把发生 MEM 队列冲突的 read 区间路由到 pong。

此前流水重叠版本的已验证镜像包含 19,304 个 MEM、292 个 MXM load、232 个 MXM dequant、226 个
MXM compute、662 个 VXM 和 136 个 SXM 工作域，共 20,852 个逻辑工作包。加上
20,070 条队列 NOP 后，共有 40,922 条逻辑 ICU 指令；226 个序列化物理队列共占
82,284 个物理 iMEM word。binary 大小为 1,219,763 bytes，counter 展开量为
6,652,186 次 FU 发射，`max_cycle`/measured end 为 769,416/769,480。完整 decoder
层 CModel 通过，最大误差为 0.0625。当前串行 Qwen2.5 seq32 单层把 Q/V/K
各 projection 合并为每个活跃队列一条指令，所有队列空档均能放入
`wait_cycle`：共 19,678 条 ICU 包、0 条 NOP、58,866 个 iMEM word，
binary 大小 1,055,197 bytes；展开 FU 发射数保持 6,646,042。

VXM/SXM 路径使用同一原则：`materializeVxmRun2DCommand` 和
`materializeSxmRun2DCommand` 接收已经由 shape、placement 和闭式 timeline
确定的两维 launch 域，直接生成 `command.vxm_run_2d` 或
`command.sxm_run_2d`。CommandBinary 原样写入 3 或 6 个物理 word。端到端测试
检查了 raw bit、iMEM 槽数、单 context 占用和全部 launch cycle。

## 旧文件兼容

现有编译产物中的 `Macro`、`MEM_STREAM_ND`、`MXM_STREAM_ND` 和
`VXM_STREAM_ND` 仍可作为
序列化输入。它们是传输描述符，不是另一套硬件 ISA。完成 relocation 后，
runtime loader 把每条可支持描述符直接闭式转换为一条强类型 FU 三维指令，再
编码成本文定义的同一 raw 包。旧描述符的绝对 `start_cycle` 只在 loader 中用来
计算 `gap = start_cycle - queue_cursor`；字段可容纳时，loader 把空档写入 raw 包
的 `wait_cycle`，再把游标推进到该包最后一次发射的下一 cycle。adapter 将二维 Macro
域或一到三维 STREAM_ND 域直接拷入硬件 counter，绝不枚举逐点发射。已经是
raw FU 三维包的输入则保持 packet bit-exact，按队列顺序写入本地 iMEM。

兼容子集有意保持严格：

| 旧输入 | 闭式硬件映射 |
| --- | --- |
| MEM Macro/STREAM_ND read 或 write | 带仿射地址归纳的 `READ_3D`、`WRITE_3D` 或 `WRITE_TAP_3D` |
| MXM load Macro/STREAM_ND | Supercell `IW` 转 `LOAD_3D`，buffer 选择固定 |
| MXM dequant Macro/STREAM_ND | 无操作数归纳的 `DEQUANT_3D` |
| MXM compute Macro/STREAM_ND `Compute` | 转带 accumulator 地址归纳的 `COMPUTE_3D` |
| MXM compute Macro/STREAM_ND `AccumulatorRead` | 转带 accumulator 地址归纳的 `ACCUMULATOR_READ_3D` |
| VXM STREAM_ND rank 1/2、无归纳 | compact `repeat_count` 保持不变，外层域转 `RUN_2D` |
| SXM TILE_PROGRAM rank 1/2、无归纳 | tile-local payload 保持不变，外层域转 `RUN_2D` |

无法表达的旧语义会在装载时立即报错，包括 MEM gather/scatter、MXM IWColumn、
MXM `Decode`、非法 induction target、超出当前字段范围的值和
重叠 live interval。`MEM_SLICE_PROGRAM` 也会在 raw MEM 路径报错，因为一个
父 program 无法安全表示为一条当前包。VXM rank 3 或带 operand induction 的
旧描述符也会被拒绝。SXM rank 3 或带 instruction-field induction 的旧描述符
同样会被拒绝。

## 每个物理 ICU 只有一个 active context

decoder 把一个完整包锁存到队列本地 context，并执行到循环结束。取指 IQ 可以预取
后续 packet word，但当前粗指令退出前不会激活下一条。context 从该域第一次
launch 到最后一次 launch 始终被占用，包括两个 launch 点之间没有发射的 cycle；
这些空隙不能插入同一队列另一条粗指令的 launch。所有物理 FU 队列的 active depth
都是：

| 物理队列 | Decoded context 数量 |
| --- | ---: |
| MEM（每个 bank） | 1 |
| MXM load | 1 |
| MXM dequant | 1 |
| MXM compute | 1 |
| VXM | 1 |
| SXM | 1 |

每个 SRAM bank 只有一个 MEM ICU、一份本地 iMEM、一个 FIFO 和一个 PC。
`READ_3D`、`WRITE_3D`、`WRITE_TAP_3D` 与由 C2C 供数的 `MEM_WRITE_SYNC`
都在这条唯一指令流中按序执行。同步写到达队首后可以等待匹配的 C2C token 和
SR 数据；等待期间，后面的普通 MEM 指令不能越过它。不同物理 FU 队列可以并行，
但同一队列内不能交错两条粗指令。编译器 raw CommandBinary 路径、runtime 容量
分析、runtime page-program linker 和 CModel 都执行这个约束。兼容路径不会再把
交错 wave 事后切段；只要一个物理 ICU 上存在 live interval 重叠就直接报错，
要求对应算子改用 direct lowering 生成连续域。

## 旧 envelope 历史 benchmark

**本节保留的是可变长旧 envelope 及其软件展开/验证路径的历史测量。表中的
binary 字节数和描述符数量不是当前 raw 本地 iMEM 占用；多种旧粗粒度形式也
不定义当前硬件的 context depth。cycle 和 golden 输出结果仍可作为兼容性证据。**

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

### 旧 MEM slice 实验与 inspector

`MEM_SLICE_PROGRAM` A/B 实验曾将多个 MEM body 合入一条旧文件描述符。该格式
仍由 `--mem-slice-program on|off` 控制，且默认关闭。它没有当前 raw MEM 包，
binary 装入 FU raw-word 队列时会被拒绝。

`ftlpu_binary_inspect left.ftlpu --compare right.ftlpu` 仍可将旧产物展开为精确
`(queue, cycle, native instruction)` 稀疏时间线进行语义比较。直到
`max_cycle` 为止，没有功能发射的点视为逻辑 NOP；发射 cycle 或原生指令变化
都会导致比较失败。这是离线 inspector 操作，不是编译器 lowering 路径，也
不是硬件执行机制。

## 本地 ICU 执行约束

raw 队列的每个本地 iMEM entry 取一个物理 word，并在队首识别 FU loop header。
激活以整包为原子：MEM 3D/VXM 的 3 个 word、MEM 同步写/MXM 的 2 个 word 或
SXM 的 6 个 word 必须全部可用，且
header 一致。decoder 锁存队列唯一的 active context，按 FU 类型推进两个或三个
counter，每 cycle 最多向 FU 发射一条原生指令；当前循环退出后才激活下一条 packet。

header 非法、包截断、错过 start cycle、loop 或操作数字段非法，以及粗指令 live
interval 重叠都会报错。每条物理 FU 队列的容量与调度约束都是一个 active context。

## 编码状态

当前已冻结的硬件侧格式是 MEM `READ_3D`、`WRITE_3D`、`WRITE_TAP_3D`
（3 x 96 bit）和 `MEM_WRITE_SYNC`（2 x 96 bit），以及 MXM `LOAD_3D`、
`DEQUANT_3D`、`COMPUTE_3D`、
`ACCUMULATOR_READ_3D`（2 x 128 bit）、VXM `RUN_2D`（3 x 96 bit），以及 SXM
`RUN_2D`（6 x 96 bit）。旧
Macro/STREAM_ND 记录只作为 closed-form adapter 的兼容文件输入。
`MEM_SLICE_PROGRAM` 仍需定义硬件包，之后才能视为 raw ICU 指令。
