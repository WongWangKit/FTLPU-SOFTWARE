# Stream Fabric Scheduler 设计

`StreamFabricScheduler` 是 LPU Stream Register Fabric 的精确 cycle 占用模型。
它与 `ResourceScheduler` 配合使用：后者管理功能单元和 MEM 端口，前者管理
数据在片上 stream 路径上的传输占用。

## 预留模型

每个占用项由下面四元组唯一确定：

```
(cycle, SR column, E/W 方向, stream id)
```

一个完整 vector beat 每个 cycle 前进一个 SR column。`StreamRouteWindow`
描述源/目的 column、连续 stream 范围、beat 数量和间隔、数据 token，以及
目的端的行为。

- 不同 token 不能在同一 cycle 占用同一个 SR cell。
- East 和 West 两套寄存器互不冲突。
- 同一个 token 可以共享路径，实现硬件 multicast。
- `Tap` 消费一份数据但保留其继续向下游传播。
- `Consume` 表示数据在该端点终止，不能继续穿过这个端点。

`reserve_resources_and_streams` 会寻找功能单元、MEM 端口和全部 stream
路径同时合法的最早 cycle，然后一起提交占用。

## 与 Target 延迟模型的边界

SR 列间传播延迟和端点内部 pipeline 延迟不是同一个量。Fabric scheduler
只建模每 cycle 前进一列；MEM 注入、MXM/VXM/SXM 捕获、结果发射以及 MEM
写回的固定延迟由 `LPUTargetModel::transport_latency` 提供。Emitter 必须按
端点 cycle 放置 route，不能用“路径包含多少列”替代 target latency。

目前最小调度单位是完整 vector beat。Tile/lane 内部错拍仍由 CModel 和
target latency 表负责，等 Schedule IR 明确暴露该层次后再下沉到 scheduler。

## 当前接入范围

- 通用 Matmul 和 legacy SwiGLU 已同时预留 FU 与 stream 资源。
- Block8 FFN hidden 跨半球复制已预留 West/East 路径，并使用当前 CModel
  passive VXM bridge 的真实端点延迟。
- stream 数量、MEM 分组和拓扑均通过 `LPUTargetModel` 参数化，不绑定固定
  硬件配置。

单元测试覆盖冲突后移、multicast、Tap/Consume、双方向独立性、路径延迟和
CModel 拓扑映射。
