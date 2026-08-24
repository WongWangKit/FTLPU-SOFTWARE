# 独立 C2C Stream

计算 Stream Fabric 保持每个方向 32 条，即 32 条 eastward 和 32 条
westward。C2C 使用独立的双向数据面，不分配、也不阻塞普通计算
`StreamId`。

硬件参数如下：

- `streams.c2c_streams_per_direction`：每个方向启用的 C2C lane 数，默认 8；
- `streams.c2c_bytes_per_stream_per_cycle`：每条 lane 每 cycle 的数据量，
  默认 32 bytes。

因此默认每个方向的峰值带宽为 `8 x 32 = 256 bytes/cycle`。两个参数都写入
v20 binary，并参与 target ABI fingerprint。若 executable 要求的 C2C lane
数超过 CModel 能力，runtime 会拒绝加载。

权重分页时，runtime 按 `slice % lane_count` 将物理 segment 映射到 C2C lane；
同一 slice 的 segment 因此在同一 lane 上串行。一条 receive descriptor 表示一段
连续目标 SRAM row。DMA ingress 直接写入
`(hemisphere, slice, bank, row)`，不再通过 `E0..E31/W0..W31` 回放。因此即使
RMSNorm feedback 占满 32 条 westward 计算 stream，下一层权重仍可同时写入
另一个 SRAM bank。
