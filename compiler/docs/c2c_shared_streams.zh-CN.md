# C2C 共享 SR 接收通路

C2C 输入统一走普通 SR fabric。runtime 把外部 lane 映射到高编号 west stream
（8 lane 时为 `W24..W31`），目标 MEM ICU 再通过普通 `Write` 消费：

```text
host -> DDR4 -> C2C DMA -> C2C RX -> 普通 west SR -> MEM Write -> SRAM
```

lowering 对每个连续 segment 直接生成三条 ICU 指令，不生成逐 vector 的 FU 指令，
也不存在 projection program：

1. `C2C_RX_BURST`：1 个 96-bit word，只包含 lane、普通 SR、vector count、
   16-bit sync tag 和目标 MEM 路由；不含 SRAM 地址。
2. `MEM_WRITE_SYNC`：2 个 96-bit word，包含 vector count、sync tag、传播延迟、
   SRAM 起始行与行步长、普通 SR。
3. `C2C_DMA_BURST`：2 个 96-bit word，包含方向、lane、DDR4 地址、vector count、
   字节步长和同一个 sync tag。

所以一个 segment 是 3 条 ICU 指令、5 个物理 iMEM word，大小与 vector count
无关。loader 在启动前把 `MEM_WRITE_SYNC` 合入目标 bank 已有的唯一 MEM iMEM，
并把 RX 和 DMA 指令放进各自 iMEM。RX 每收到一个 vector，就向目标
`(hemisphere, slice, bank)` 的 MEM ICU 发送带 tag 的点对点 token；该同步写位于
队首、有 token 且 SR 数据有效时才推进。CModel 保留每个通知的到达周期，因此 DDR
气泡只会暂停对应的写入，完成条件以最后一个 SRAM 写提交为准。

`MEM_WRITE_SYNC` 没有 sidecar 队列或第二个 PC。它和普通 read/write 3D 指令共享
该 bank 的一个 MEM ICU、一个 FIFO 和一个 active context；等待 C2C 时，后续 MEM
指令不能越过。payload 仍受普通 SR 路由和冲突规则约束。page fence 只有在 RX 完成
且所有目标 MEM write 都已发出后才会 ready。
