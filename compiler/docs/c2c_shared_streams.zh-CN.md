# C2C 共享 SR 接收通路

C2C 输入统一走普通 SR fabric。runtime 把外部 lane 映射到高编号 west stream
（8 lane 时为 `W24..W31`），目标 MEM ICU 再通过普通 `Write` 消费：

```text
host -> DDR4 -> C2C DMA -> C2C RX -> 普通 west SR -> MEM Write -> SRAM
```

RX 每收到一个 vector，就向目标 `(hemisphere, slice, bank)` MEM ICU 的 C2C
命令上下文发送点对点 token。runtime 为每段连续数据生成一条粗粒度
synchronized-write 描述符，包含 vector 数量、SRAM 起始地址和步长、SR ID、目标
相关的传播延迟。CModel 保留每个通知的到达周期，因此连续 vector 可以流水写入，
DDR 气泡只会暂停与之对应的 write。

这个辅助命令上下文不是专用数据 lane。它与计算共享物理 MEM 端口，payload 仍受
普通 SR 路由和冲突规则约束。page fence 只有在 RX 完成且所有目标 MEM write 都已
发出后才会 ready。
