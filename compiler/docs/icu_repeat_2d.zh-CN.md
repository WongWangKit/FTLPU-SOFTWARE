# ICU Repeat2D 编译器设计

`Repeat2D` 压缩一条功能指令的二维规则迭代。Command IR 继续使用已有的
`repeat_*`、`wave_*` 和 `group_*` 字段；binary emitter 负责把其中至多两个有效维度
映射成一个 96-bit 描述符，不需要增加模型专用 lowering。

## 语义

紧邻描述符之前的功能指令是坐标 `(0, 0)`，并且已经发射。描述符按 outer-major、
inner-minor 顺序发射其余坐标：

```text
cycle(o, i) = base_cycle + o * outer_interval + i * inner_interval
delta(o, i) = o * outer_stride + i * inner_stride
```

`inner_count` 和 `outer_count` 都包含基点。ICU 在二维迭代结束前阻塞该本地队列，
其他 ICU 队列不受影响。因而 compiler 只在整个二维时间区间内没有同队列交错事件时
使用 Repeat2D；有交错事件时自动展开 outer 维，保留正确的发射顺序。

归纳目标是有类型的：

- `None`：不修改功能指令操作数，两个 stride 必须为 0；
- `MemAddress`：修改 MEM read 地址；`ReadWrite` 同时修改 write 地址；
- `MxmWeightColumn`：只允许用于 MXM `IW` 的 weight column。

## 编译策略

`CommandBinary.cpp` 先把 Command op 转成 queue-local `CommandSequence`，保留 inner 和
outer 维。队列按 cycle 排序后，用线性扫描判断候选二维区间是否与相邻序列重叠：

- 无重叠：生成一条功能指令和一条 Repeat2D 描述符；
- 有重叠：展开 outer 维，每个 outer 点仍可使用已有的一维 Repeat；
- target 设置 `throughput.icu_repeat_2d_enabled = 0`：强制使用展开形式，便于不支持
  Repeat2D 的硬件和压缩前后 A/B 测试。

MEM 双端口合并要求 read/write 的 inner count、interval、stride 以及 outer count、
interval、stride 全部一致。不同二维 shape 会先展开，避免一个端口错误复用另一个端口
的地址归纳。

## 验证结果

- CModel codec、MEM 地址归纳、MXM column 归纳单元测试通过；
- Command IR -> binary -> runtime -> CModel ICU 兼容测试通过；
- 同队列存在交错事件时，自动回退为展开形式；
- SmolLM2 seq_len=128 Block8 fused FFN 的 Repeat2D 与完全展开版本具有相同
  `max_cycle=21095`、相同动态发射计数和相同数值输出；
- 该 FFN binary 从完全展开的 17,650,899 bytes 降为 12,865,201 bytes，减少约 27.1%。

当前 CModel 新增 C2C-SXM 寄存器列后，展开和 Repeat2D 两种 FFN binary 都出现相同的
既有 fused FFN 数值偏差，因此该偏差不属于 Repeat2D 语义差异。

