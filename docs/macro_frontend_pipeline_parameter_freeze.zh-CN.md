# Macro frontend pipeline 参数实验与建议冻结值

> 此文保留实验参数建议。main 的实际默认值与可选入口见
> [物理模型接入说明](macro_frontend_physical_model.zh-CN.md)，建议值不代表默认值已切换。

日期：2026-09-28。

## 1. 结论

建议 RTL/spec 第一版采用以下参数：

| 参数 | 选定值 | 状态 |
|---|---:|---|
| MEM logical i-MEM word | 96 bit | 冻结 |
| MXM load/compute logical i-MEM word | 128 bit | 冻结 |
| dequant scale-control logical i-MEM word | 128 bit | 当前软件模型；若 RTL 合并进 load 则删除 |
| SRAM response latency | 2 cycle | 暂定实现点 |
| ECC decode latency | 1 registered cycle | 暂定实现点 |
| frontend effective fetch latency | **3 cycle** | 冻结接口行为 |
| request initiation interval | **2 cycle** | 冻结 |
| maximum outstanding reads | **2** | 冻结，`ceil(3/2)` |
| reservoir | **3 word** | 冻结 |
| aligned decode window | **64 bit** | 冻结 |
| DDB depth | **4 entry / physical queue** | 冻结 |
| contexts represented by one DDB entry | **4** | 冻结 |
| admission lookahead | **8 cycle** | 冻结 |
| context expand/admit width | **1 context/cycle** | 冻结 |
| Active Context RAM | MEM 32 entry；MXM 40 entry | 保持当前值 |
| Active Context width | 256 bit | 保持当前值 |
| ECC code | whole-word SECDED | 暂定；待 SRAM/PPA 替换确认 |

这组参数的含义是：逻辑 i-MEM word 宽度不因 ECC 改变；SRAM 读请求最早每两周期
发起一次，SRAM 两周期返回，经过一级 registered SECDED 后在请求后的第三周期进入
3-word reservoir。两笔 outstanding read 足以覆盖该 pipeline。

## 2. 搜索方法

没有执行所有参数的全笛卡尔积。实验分四层：

1. **功能门禁**：每个候选必须保持 context count 和 decoded semantic hash 完全一致。
2. **fetch/ECC timing sweep**：固定结构参数，扫描 latency=`1,2,3,4`、II=`1,2,4`。
3. **结构单因素 sweep**：固定 `effective latency=3, II=2`，分别扫描 reservoir、
   decode window、DDB depth 和 DDB run capacity；随后只补 `D4/C4`、`D2/C4`
   两个交互点。
4. **timing-aware admission sweep**：先 prime DDB，再按 context 的绝对
   start/final issue cycle 和有限 Active Context RAM 模拟 lookahead=`2,4,8,16`。

评价指标：

- `frontend_cycles`：各 physical queue frontend cycle 之和，是供给能力/余量代理，
  不是 application wall-clock；满足 deadline 后，更小不会让静态 issue 提前；
- `bit_wait/starvation`、最大 prime cycle；
- `decoder_ddb_stall_cycles`、peak DDB occupancy；
- `DDB depth × entry bits × queue count`，作为 DDB 面积代理；
- admission deadline、最大 admission lead、Active Context 峰值；
- semantic hash/context count 是否改变。

## 3. Workload 与有效性边界

本轮使用：

| Workload | Macro queues | i-MEM words | contexts |
|---|---:|---:|---:|
| SmolLM2-135M seq128 | 158 | 18,417 | 134,999 |
| Qwen seq32 MEM-only CSV | 208 | 12,910 | 32,762 |

Qwen binary 实际还包含 MXM load/compute 和当前软件的 dequant scale-control Macro
queue；实验工具因使用缓存 `packed_macro_imem` presence 作为筛选条件而漏测这些
queue，因此本轮 Qwen 数据只代表 MEM。SmolLM2 数据覆盖了这些 MXM 类型，但不能
替代 Qwen MXM 补跑。

全部有效候选共 136 行结果，按 `(model, queue_kind)` 检查后，semantic hash 和
context count mismatch 均为 0。

Qwen2.5-3B seq128 已用当前 scheduler 重新生成并通过严格 Schedule→Command
验证；translator 还完成了 34,644,016 条 functional issue 的压缩前后等价检查。
但最终 binary 在 `Active Context row transition exceeds 14 bits` 处被拒绝。
TinyLlama seq128 的旧 binary 也受相同约束阻塞。该问题发生在参数 sweep 之前，
需要在扩大/重排 Active Context step 字段和编译器等价拆 Macro 之间完成选择；本轮
没有把这两个非法 binary 当作候选性能失败。

## 4. Fetch latency / II 数据

完整 12 点数据见：

- `out/macro-frontend-pipeline-sweep/fetch-smollm2.csv`
- `out/macro-frontend-pipeline-sweep/fetch-qwen_seq32.csv`

关键点如下；百分比相对 `L1/II1`：

| Workload | L | II | frontend cycles | 增量 | bit wait | max prime |
|---|---:|---:|---:|---:|---:|---:|
| SmolLM2 seq128 | 1 | 1 | 212,497 | 0 | 158 | 85 |
| SmolLM2 seq128 | 2 | 2 | 212,813 | +0.1487% | 316 | 87 |
| SmolLM2 seq128 | **3** | **2** | **213,129** | **+0.2974%** | 474 | 89 |
| SmolLM2 seq128 | 4 | 2 | 213,461 | +0.4537% | 648 | 91 |
| SmolLM2 seq128 | 3 | 4 | 214,317 | +0.8565% | 1,662 | 98 |
| Qwen seq32 | 1 | 1 | 99,100 | 0 | 208 | 89 |
| Qwen seq32 | 2 | 2 | 99,516 | +0.4198% | 416 | 91 |
| Qwen seq32 | **3** | **2** | **99,932** | **+0.8396%** | 624 | 93 |
| Qwen seq32 | 4 | 2 | 100,372 | +1.2836% | 856 | 95 |
| Qwen seq32 | 3 | 4 | 100,804 | +1.7195% | 1,496 | 98 |

结论：

- latency 大于等于 2 时，II=1 与 II=2 在两组 workload 上结果相同；选择 II=2
  可以换取更容易实现的 SRAM/ECC datapath，而没有观察到额外吞吐损失。
- `effective L=3/II=2` 给一级 registered ECC 留出周期，损失仍低于 1%。
- II=4 的 starvation 明显上升，只保留为极低成本 SRAM 的边界方案，不进入第一版冻结。
- L=4 可作为 timing fallback，但不是目标实现点。

## 5. 结构参数数据

完整数据见 `structure-smollm2.csv` 和 `structure-qwen_seq32.csv`。结构实验固定
`L=3/II=2`。

### 5.1 Reservoir

| Workload | words | frontend cycles | bit wait | max prime |
|---|---:|---:|---:|---:|
| SmolLM2 seq128 | 2 | 213,883 | 1,228 | 93 |
| SmolLM2 seq128 | **3** | **213,129** | **474** | **89** |
| SmolLM2 seq128 | 4 | 213,129 | 474 | 89 |
| Qwen seq32 | 2 | 100,380 | 1,072 | 94 |
| Qwen seq32 | **3** | **99,932** | **624** | **93** |
| Qwen seq32 | 4 | 99,932 | 624 | 93 |

3 words 已达到 4 words 的性能；2 words 增加 0.35%～0.45% cycles，并明显增加
bit wait。因此选 3。

### 5.2 Decode window

- 48-bit window 在 SmolLM2 的 MXM compute queue 0 功能失败；
- 32-bit window 在 Qwen MEM queue 37 功能失败；
- 原因均为一次 parser read 超出 aligned window。

因此 64 bit 是当前“Template header 原子读取”微架构的实现下限，不是 packed-v1
grammar 的理论下限。若为最长 54-bit MXM compute header 增加跨周期 assembler，
W32/W48 仍可实现，但需要重新比较 decode latency、状态和面积。

### 5.3 DDB depth 与 run capacity

| Workload | 配置 | frontend cycles | DDB stall | max prime | DDB bits 总量 | peak DDB |
|---|---|---:|---:|---:|---:|---:|
| SmolLM2 | D8/C8 baseline | 213,129 | 0 | 89 | 912,464 | 3 |
| SmolLM2 | **D4/C4** | **213,049** | **0** | **56** | **293,176** | 2 |
| SmolLM2 | D2/C4 | 213,049 | 0 | 36 | 146,588 | 2 |
| Qwen seq32 | D8/C8 baseline | 99,932 | 0 | 93 | 1,203,072 | 3 |
| Qwen seq32 | **D4/C4** | **99,580** | **0** | **47** | **386,880** | 2 |
| Qwen seq32 | D2/C4 | 99,580 | 0 | 30 | 193,440 | 2 |

`D4/C4` 相对 `D8/C8` 将 DDB bit 面积代理减少约 67.9%，没有 stall，prime
更短，frontend cycles 没有退化。D2/C4 在当前数据也通过，但 peak 已达到 2，
没有 future grammar/backpressure 余量，因此不冻结到 2。

选定 C4 后当前软件模型各 queue 的 DDB entry width 为：MEM 465 bit、MXM load
452 bit、MXM compute 483 bit、dequant scale-control 372 bit。D4 对应每 queue 1,860/1,808/
1,932/1,488 bit。

## 6. Admission lookahead

timing-aware 实验使用选定的 `R3/W64/D4/C4/L3/II2`：

| Lookahead | SmolLM2 相对 A8 timeline cycles | Qwen seq32 相对 A8 | deadline | Active RAM full |
|---:|---:|---:|---|---:|
| 2 | +948 | +1,248 | 通过 | 0 |
| 4 | +632 | +832 | 通过 | 0 |
| **8** | **0** | **0** | **通过** | **0** |
| 16 | -1,264 | -1,664 | 通过 | 0 |

A16 只让 frontend 更早完成 admission，不会让静态 issue cycle 提前，同时使 context
更早进入 Active RAM。A8 已满足 deadline，并给 fetch/parser/credit stall 留出
8-cycle 裕量，继续采用 8。观察到的
peak Active Context 为 SmolLM2 32（MXM compute）和 Qwen seq32 MEM 20，因此保留
MEM=32、MXM=40，不在本轮缩减。

context expand width 保持 1：strict resource schedule 已保证每个 physical queue
每周期最多一条 issue，width=2 没有下游吞吐收益，只会复制 admission/update 逻辑。

## 7. ECC 选定

暂定 whole-word SECDED：

| Logical word | 最小 Hamming parity + overall parity | Physical information bits | 位开销 |
|---:|---:|---:|---:|
| 96 | 7 + 1 | 104 | 8.33% |
| 128 | 8 + 1 | 137 | 7.03% |

ECC decode/correct 注册一级，按 effective latency=3 建模；ECC pipeline 自身目标
II=1，系统 II 由 SRAM 的 2-cycle request interval 决定。若 foundry macro 只能提供
112/144 等粒度，可 padding，但 logical word 和 frontend timing contract 不变。

RTL 冻结前仍需完成：

1. SRAM macro + SECDED 综合后的 Fmax/面积/功耗；
2. data/check bit 单错纠正、任意双错检测、uncorrectable 抑制 issue 的 fault injection；
3. reset/加载后的 ECC 初始化、错误计数器和 poison/trap 行为；
4. 若 whole-word correction 路径不收敛，比较 64/32-bit chunk SECDED，但保持
   registered 1-cycle、effective L3/II2 的接口目标。

## 8. 冻结门槛与后续工作

可立即冻结 RTL 端口和 pipeline 的参数为：

```text
MEM/MXM logical word       = 96 / 128 bit
effective fetch latency   = 3 cycle
request II                = 2 cycle
outstanding reads         = 2
reservoir                 = 3 word
decode window             = 64 bit
DDB depth / run capacity  = 4 / 4
admission lookahead       = 8 cycle
context expand width      = 1 / cycle
Active Context            = MEM 32, MXM 40, 256 bit/entry
ECC                       = whole-word SECDED, registered decode
```

尚未关闭但不阻塞第一版 RTL 的事项：

- TinyLlama/Qwen seq128 完成 Active Context step 加宽/重排或 Macro 合法拆分后，
  复跑同一矩阵；
- 对 `D2/C4` 做 synthetic burst/backpressure 压力测试，只作为未来降面积选项；
- SRAM vendor 数据到位后确认 physical width 与 ECC PPA；
- gate-level/RTL 仿真复核 outstanding=2、response ordering 和 ECC error path。
