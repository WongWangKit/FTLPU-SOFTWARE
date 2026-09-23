# C2C 共享 SR 接收通路

C2C 输入统一走普通 SR fabric。runtime 把外部 lane 映射到高编号 west stream
（8 lane 时为 `W24..W31`），目标 MEM ICU 再通过普通 `Write` 消费：

```text
host -> DDR4 -> C2C DMA -> C2C RX -> 普通 west SR -> MEM Write -> SRAM
```

可执行权重页对每个连续 segment 共需三条 ICU 指令，不生成逐 vector 的 FU 指令，
也不存在 projection program：

1. `C2C_RX_BURST`：1 个 96-bit word，只包含 lane、普通 SR、vector count、
   16-bit sync tag 和目标 MEM 路由；不含 SRAM 地址。
2. `MEM_WRITE_SYNC`：2 个 96-bit word，包含 vector count、sync tag、传播延迟、
   SRAM 起始行与行步长、普通 SR。
3. `C2C_DMA_BURST`：2 个 96-bit word，包含方向、lane、DDR4 地址、vector count、
   字节步长和同一个 sync tag。

所以一个 segment 是 3 条 ICU 指令、5 个物理 iMEM word，大小与 vector count
无关。compiler 根据权重页的顺序和时间窗口，直接把 `MEM_WRITE_SYNC` 写入目标
bank 唯一的 MEM iMEM；runtime 提供权重页数据和 DDR 地址，生成 RX/DMA 队列，
并校验传输计划与 compiler 的 MEM 指令逐字一致，不再改写可执行程序的 MEM iMEM。
RX 每收到一个 vector，就向目标
`(hemisphere, slice, bank)` 的 MEM ICU 发送带 tag 的点对点 token；该同步写位于
队首、有 token 且 SR 数据有效时才推进。CModel 保留每个通知的到达周期，因此 DDR
气泡只会暂停对应的写入，完成条件以最后一个 SRAM 写提交为准。

`MEM_WRITE_SYNC` 没有 sidecar 队列或第二个 PC。它和普通 read/write 3D 指令共享
该 bank 的一个 MEM ICU、一个 FIFO 和一个 active context；等待 C2C 时，后续 MEM
指令不能越过。payload 仍受普通 SR 路由和冲突规则约束。page fence 只有在 RX 完成
且所有目标 MEM write 都已发出后才会 ready。

编译可执行队列时，若 `MEM_WRITE_SYNC` 前面只有空闲周期，compiler 会让该指令
提前成为队首，直接等待对应 tag，并把前面的空闲周期计入其 24-bit 保留时长。
因此不会额外占用一条前置 NOP，后续普通 MEM 指令仍按原定周期发射；保留时长
字段装不下时才保留前置 NOP。普通 `SYNC` 是独立的控制指令，不负责这段 MEM
写入保留窗口。
