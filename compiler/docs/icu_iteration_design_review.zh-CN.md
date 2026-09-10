# ICU 迭代指令设计评审

生成调度中有两类不同的重复：

- 单条功能指令在规则的 inner/outer 坐标上重复；
- 一段完全相同的多指令窗口整体回放。

`Repeat2D` 负责第一类。曾提议的多指令 `Loop` ISA 已删除：其额外 ICU body
存储和回放控制成本与实测收益不匹配。多指令重复窗口改为物化；若窗口中的
MEM/MXM 单条指令各自满足仿射条件，则使用独立的绝对 cycle 二维 Macro 描述符。

## 已实现决策

`Repeat2D` 是 96-bit、队列局部、阻塞式的二维迭代器，作用于紧邻它之前的功能指令。
count 包含已经发射的基点，剩余坐标按 outer-major、inner-minor 顺序发射。归纳目标为
MEM 地址、MXM IW weight column 或无归纳。

阻塞语义避免 ICU 保存无上限的并发 iterator context。binary emitter 只在完整二维区间
内没有同队列交错事件时使用 Repeat2D，否则展开 outer 维。合法性判断是对按 cycle 排序
队列的一次线性扫描。

compiler capability `throughput.icu_repeat_2d_enabled` 可为不支持该指令的硬件强制展开，
也可用于压缩前后语义 A/B。

当前不存在 `Loop` command、binary record 或 runtime decoder。支持的模式为
`none`、`repeat`、`macro` 和实验性的 `macro-slice`；旧拼写 `control` 仍作为
`repeat` 的兼容别名。
Macro v1 验证期间关闭 rank-3 `STREAM_ND` 生成。

## 实测结果

SmolLM2 seq_len=128 Block8 fused FFN：

| 形式 | binary 大小 | max_cycle | 动态发射计数 |
| --- | ---: | ---: | --- |
| outer 维完全展开 | 17,650,899 bytes | 21,095 | 基线 |
| blocking Repeat2D | 12,865,201 bytes | 21,095 | 完全一致 |

binary 大小减少约 27.1%，两种形式的数值输出也完全一致。早期 8.76× 估算假设了大量
并发、非阻塞 iterator context，不适用于最终实现的硬件语义。

Qwen 和完整 decoder 的占用率应在当前 CModel VXM compact-instruction 迁移稳定后重测。

精确语义和 compiler fallback 见 [icu_repeat_2d.zh-CN.md](icu_repeat_2d.zh-CN.md)。

