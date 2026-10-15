# ICU 单次、1D、2D 短格式

`.ftlpu` v35 增加 MEM/MXM 短格式；旧 3D 包仍可读取。物理 iMEM 仍是
MEM 96 bit/word、MXM 128 bit/word。每个 ICU 只保持一条活跃指令，短格式
解码为同一个循环执行状态，不引入 macro calendar 或多 context 交错。

## 编译与执行约束

算子 lowering 直接构造循环域。MEM 编码选择前消除 count=1 的维度，只有当
地址步长和周期步长同时连续、合并后的 count 能放入 16 bit 时，才闭式折叠
相邻维度。非仿射 blocked outer 保留 3D；当访问范围内 blocked 地址等价于
普通仿射地址时可选择短格式。全过程不生成 FU 时间线再反向压缩。

MXM 的维度带有 row wave、buffer parity、terminal mode 等语义，因此不重新
编号或折叠维度；短包用两位 axis selector 标记省略了哪些维度。解码恢复原
轴位置，保证 row、K、N 含义不变。

Command IR 的 `mem_3d` / `mxm_*_3d` 仍作为兼容的循环域容器，`words` 属性
保存已选定的实际物理包。runtime 不再按该 IR 名称推断固定三槽或两槽。

## 包长与 opcode

| MEM 物理格式 | 字节 | 本地 operation |
|---|---:|---|
| READ / WRITE / WRITE_TAP | 12 | 0 / 1 / 2 |
| READ_1D / WRITE_1D / WRITE_TAP_1D | 24 | 0 / 1 / 2 |
| READ_2D / WRITE_2D / WRITE_TAP_2D | 24 | 0 / 1 / 2 |
| READ_3D / WRITE_3D / WRITE_TAP_3D | 36 | 0 / 1 / 2 |

单次、1D、2D 分别使用 word 0 物理 `[91:88]` 的 Extended subtype
8、9、10；现有 3D subtype 2 和 WRITE_READ_2D subtype 7 不变。
每个实际存在的 word 保留原 8-bit 本地 header：envelope、word index、
operation、packet marker、version。包长由物理队列类型及 subtype 唯一确定。
不允许把多余的零填充 word 写入 iMEM。

1D 仍保留完整的 24-bit wait、16-bit count、24-bit cycle stride 和
20-bit 地址 stride，所以占两个 word。它没有第三维字段，但不是承诺所有
1D 都能塞进一个 96-bit word。

## MEM payload

设 rank 为 0、1、2，逻辑 payload 跳过 word header 与 subtype：

1. 24-bit `wait_cycle`。
2. rank 个 16-bit `count - 1`。
3. rank 个 24-bit `cycle_stride`。
4. 6-bit stream、13-bit bank 内 base row。
5. rank 个有符号 20-bit address stride。

使用位数为 `43 + 60 × rank`；剩余 payload 位必须为零。单次包不携带
任何循环字段。缺失维度解码为 count=1、cycle_stride=1、address_stride=0。

## MXM payload

1. 24-bit `wait_cycle`。
2. rank 非零时有 2-bit axis selector：1D 表示唯一有效轴；2D 表示唯一
   省略轴。值 3 非法。
3. 按原轴顺序保存 rank 个 16-bit count、rank 个 24-bit cycle stride。
4. 原 3D 包 P[144:] 的 FU 专用有效字段，保留 buffer、accumulator、
   terminal 和 decode 等字段的含义；原 3D 的尾部 padding 不复制。

包长按该硬件配置的字段宽度向 128 bit 对齐。因此 MXM 的部分 1D/2D 包
仍占两个 word；单次和 DEQUANT_1D 等包可缩成一个 word。维度少并不自动
意味着物理槽数少。

## runtime 与检查

所有扫描、relocation、C2C 页依赖分析、静态 trace 和 ICU CSV 导出按包头
确定实际长度。relocation 保持已有格式及包长，不移动后续指令的 PC。
`wait_cycle` 始终位于同一位置，静态 FU 时序空档可直接写入短包。
DDR/C2C 等待仍由 SYNC/MEM_WRITE_SYNC 等完成事件控制。

ICU CSV 显示实际 READ、READ_1D、READ_2D、READ_3D 等 opcode，以及真实
`imem_words`。C++ Packet 类型仍使用最大容量的临时缓冲区；其 `sizeof`
不是短格式的物理存储量，应使用 `fu_loop_packet_word_count()`。

验证入口：CModel `unit.icu.short_packet`，软件 `icu_short_lowering_test`、
`icu_program_test`、`model_session_executable_c2c_weight_test`。测试涵盖闭式
折叠前后 FU 字与周期一致、MXM 原轴语义、混合包长、单 context，以及真实
同步队列装载。未在 RTL 仓库实现对应 decoder。
