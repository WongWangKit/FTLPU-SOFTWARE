# C2C 到共享 SR 的接收通路

C2C 对外仍提供每方向最多 8 条 transport lane，每条 lane 每 cycle 搬运一个
32-byte vector。外部 lane ID 不等于片内 LPU stream ID。接收时，runtime 将外部
lane 映射到普通 stream-register fabric 的一组显式 stream；默认权重分页将 lane
0..7 映射为 west streams `W24..W31`。

默认接收路径为：

```text
DDR4 -> 每 lane DMA RX FIFO -> C2C RX -> 在 sreg13 注入 W24..W31
     -> 沿 westward SR 传播 -> 目标 MEM Write -> SRAM bank/row
```

因此 `C2cInstruction` 包含两个相互独立的选择器：

- `stream_index`：外部 C2C/DMA lane 0..7；
- `fabric_stream_index`：片内传输所用的普通 SR 编号。

receive descriptor 仍描述目标 hemisphere、slice、bank、base row、vector count
和 row stride，但 C2C 不再绕过 MEM 直接修改 SRAM。第一个 vector 到达后释放目标
MEM 的 `Sync`；runtime 根据目标 slice 精确插入 route NOP，再发一条 `Write`，其余
连续 row 用 ICU `Repeat` 展开。只有 DMA、RX、目标 MEM queue 和 4-tile SR 尾部都
完成后，权重页才标记 ready。

这样真实冲突会暴露给调度器：只有所选 SR 和目标 SRAM write port 空闲时，C2C
才能与计算重叠。feedback RMSNorm 的 gamma 每层不同，但层内不变；它复制到激活区
两个 slice，每个半球只读两条低编号 west stream。RMSNorm 数据流保持在 `W16`
以下，为 C2C 权重接收保留 `W24..W31`。这是 planner 做出的资源隔离，不是隐藏的
旁路。

`c2c_dma_ddr4_test` 验证 8 条 lane 同时经过普通 SR 和 MEM Write；
`c2c_weight_ping_pong_test` 验证当前 bank 读取期间，下一 bank 的分页数据逐字节正确。
