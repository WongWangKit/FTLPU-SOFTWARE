# Macro Frontend 第一轮：i-MEM Fetch/ECC Sweep

## 1. 实验范围

本轮固定逻辑 i-MEM 接口位宽：

| Queue | Word bits |
|---|---:|
| MEM | 96 |
| MXM load/compute，以及当前软件的 dequant scale-control queue | 128 |

其余 frontend 参数固定为当前 baseline：

```text
reservoir words       = 3
decode window         = 64 bit
DDB depth             = 8
DDB run capacity      = 8
admission lookahead   = 8 cycle
context expand width  = 1 / cycle
```

Sweep 参数：

```text
i-MEM response latency = 1, 2, 3, 4 cycle
request initiation II  = 1, 2, 4 cycle
```

`latency` 是 request 到 response 的时间；`II` 是两个 request 的最小发起间隔。
二者在 CModel 中独立建模，因此 latency 大于 1 时仍允许多个 outstanding read。

实验工具为 `macro_frontend_fetch_sweep`。它读取 `.ftlpu` 中的真实 Macro queue，
使用当前 packed-v1 encoder 重新生成物理 word，对 12 个配置逐 queue 解码，并输出
prefill、starvation、descriptor 和 context 统计。每个配置还计算 decoded context
semantic hash，检查 fetch timing 参数没有改变指令、schedule 或 operand。

## 2. 当前可闭环 workload

SmolLM2 seq128 成功完成全部 12 个配置：

| Queue kind | Queue 数 | i-MEM words | Context/record 数 | baseline 最大 prime cycle |
|---|---:|---:|---:|---:|
| MEM | 152 | 17,534 | 133,256 | 85 |
| MXM load | 2 | 279 | 366 | 60 |
| MXM compute | 2 | 466 | 1,090 | 84 |
| dequant scale-control（当前软件模型） | 2 | 138 | 287 | 65 |
| 合计 | 158 | 18,417 | 134,999 | 85 |

12 个配置中，每种 queue kind 的 semantic hash 均只有一个取值，context 数也完全
相同，说明 latency/II sweep 未改变功能语义。

## 3. SmolLM2 seq128 结果

`frontend cycles` 是全部 queue 的 frontend cycle 求和，不是 application wall-clock
cycle；`prime` 是所有 queue 并行 prime 时的最大周期数。

| Latency | II | Frontend cycles | 相对 L1/II1 | Bit wait/starvation | 最大 prime |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 212,497 | 0.0000% | 158 | 85 |
| 1 | 2 | 212,655 | +0.0744% | 316 | 86 |
| 1 | 4 | 213,897 | +0.6588% | 1,558 | 94 |
| 2 | 1 | 212,813 | +0.1487% | 316 | 87 |
| 2 | 2 | 212,813 | +0.1487% | 316 | 87 |
| 2 | 4 | 214,105 | +0.7567% | 1,608 | 96 |
| 3 | 1 | 213,129 | +0.2974% | 474 | 89 |
| 3 | 2 | 213,129 | +0.2974% | 474 | 89 |
| 3 | 4 | 214,317 | +0.8565% | 1,662 | 98 |
| 4 | 1 | 213,461 | +0.4537% | 648 | 91 |
| 4 | 2 | 213,461 | +0.4537% | 648 | 91 |
| 4 | 4 | 214,569 | +0.9751% | 1,756 | 100 |

观察：

1. response latency 从 1 增加到 4、同时保持 II 不超过 2 时，frontend work 增量不超过
   0.454%，最大 prime 从 85 增加到 91 cycle；3-word reservoir 能隐藏大部分响应延迟。
2. 在 latency 为 2～4 时，II=1 与 II=2 的结果相同，说明该 workload 的 parser
   服务速率低于每两周期一个 word，没有从 II=1 获得额外收益。
3. II=4 开始出现持续额外 starvation，但当前 workload 的总 frontend work 增量仍低于
   1%，最大 prime 增加到 94～100 cycle。
4. 这些结论只能说明 SmolLM2 seq128 对 fetch 不敏感，不能替代大模型和极端长
   descriptor workload。

## 4. ECC 候选

当前 CModel 没有 ECC encode/check/fault-injection 模块。因此 ECC 对正常读取的周期影响
必须折算为：

```text
effective_fetch_latency = SRAM response latency + registered ECC decode stages
effective_request_II    = max(SRAM request II, ECC pipeline II)
```

本轮 latency/II 表可以直接用于筛 ECC timing 候选，但不能代表 ECC 面积、功耗或纠错
能力已经验证。

| ECC 方案 | 96-bit MEM 存储位 | 128-bit MXM 存储位 | 开销 | 能力/注意事项 |
|---|---:|---:|---:|---|
| None | 96 | 128 | 0 | 无检测；仅作为面积/性能基线 |
| 每32-bit lane parity | 99 | 132 | 3.125% | 检测奇数位错误，不能纠正 |
| 整word SECDED | 104 | 137 | 8.33% / 7.03% | 位开销低，但校验树跨96/128 bit，需看Fmax和SRAM粒度 |
| 每64/32-bit chunk SECDED | 111 | 144 | 15.63% / 12.5% | 更接近常见64+8组织；MEM为64+32两个chunk |
| 每32-bit lane SECDED | 117 | 156 | 21.875% | 局部纠错和实现简单，但位开销最大 |

注：SECDED 位数按外置 Hamming SECDED 理论最小校验位计算。若 SRAM macro 自带 ECC，
应以 macro datasheet 的实际 width、latency、II、面积和功耗替换本表。

建议保留的 ECC timing 候选：

```text
A. combinational ECC: extra latency 0, II unchanged
B. registered ECC:    extra latency 1, II unchanged
C. low-area ECC:      extra latency 1, II = 2
D. non-pipelined ECC: effective II = 4（低优先级边界点）
```

按 SmolLM2 结果，只要 ECC 后 effective latency 不超过 4 且 II 不超过 2，正常路径的
frontend吞吐损失很小。最终仍必须做 SRAM/PPA 和 fault-injection。

## 5. 尚未关闭的阻塞项

当前 TinyLlama/Qwen seq128 使用 32K-row target。切换到匹配的 target 后，fetch sweep
在进入第一组配置时分别暴露：

- TinyLlama：MEM queue 37 的 row transition 超出 14-bit Active Context字段；
- Qwen2.5-3B：MEM queue 0 存在 overlapping outer iteration，被当前 Context合法性检查拒绝。

这两个错误发生在任何 latency/II 比较之前，因此不能归因于 i-MEM fetch。必须先统一：

1. 32K-row SRAM target 与 Active Context stride/row-transition 位宽；
2. compiler 对 inner/outer 非重叠 schedule 的生成契约；
3. 大模型 `.ftlpu`、target JSON、CModel 和当前 packed-v1 ABI版本。

在这两个 workload 能通过 baseline L1/II1 前，本轮不能最终冻结 latency/II。

## 6. 当前建议

- 保留 `latency=1..4` 作为可接受候选范围；不要先验要求单周期 SRAM。
- 优先保留 `II=1/2`；`II=4` 作为低成本 SRAM/ECC 的边界候选。
- ECC 若需要一个 registered decode stage，可先按 `effective latency +1、II不变`继续设计。
- 暂不冻结具体 ECC码型，等待 SRAM macro 的真实组织、Fmax、面积和功耗数据。
- 修复/统一 TinyLlama和Qwen的 context合法性后，复跑同一12点矩阵；最终选择应以
  “全部 workload 零错误 + PPA最优”为准。
