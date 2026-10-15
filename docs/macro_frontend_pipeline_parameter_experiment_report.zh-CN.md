# Macro Frontend Pipeline 参数实验汇报

日期：2026-09-28。

## 1. 目标与结论摘要

本次实验在冻结 Macro Frontend spec、进入 RTL 设计前，确定 i-MEM fetch、parser
buffer、DDB 和 admission pipeline 的首版参数。实验固定逻辑 i-MEM 接口位宽，使用
真实 packed-v1 Macro queue 做 cycle-accurate frontend 仿真。

| 参数 | 选定值 | 状态 |
|---|---:|---|
| MEM / MXM logical i-MEM word | 96 / 128 bit | 本轮固定 |
| SRAM response / registered ECC | 2 / 1 cycle | 暂定实现点 |
| effective fetch latency | **3 cycle** | 建议冻结接口行为 |
| request II / outstanding reads | **2 / 2** | 建议冻结 |
| reservoir / decode window | **3 words / 64 bit** | 建议冻结 |
| DDB depth / run capacity | **4 / 4** | 建议冻结 |
| admission lookahead / expand width | **8 cycle / 1** | 建议冻结 |
| Active Context depth | MEM 32 / MXM 40 | 保持当前值 |
| ECC code | whole-word SECDED | 暂定，PPA/fault injection 待完成 |

`L3/II2` 表示 SRAM 请求到响应为 2 cycle，再经过一级 registered ECC，于请求后
第 3 cycle 进入 reservoir；最早每 2 cycle 发起一个请求，两笔 outstanding read
足以覆盖该 pipeline。

## 2. Workload 与覆盖范围

| Workload | Queue 覆盖 | Macro queues | i-MEM words | contexts |
|---|---|---:|---:|---:|
| SmolLM2-135M seq128 | MEM、MXM load/compute、当前软件的 dequant scale-control queue | 158 | 18,417 | 134,999 |
| Qwen seq32 CSV | **仅 MEM（实验工具漏测 MXM）** | 208 | 12,910 | 32,762 |

Qwen binary 本身并非没有 MXM。完整 inventory 为 MEM 208、MXM load 2、MXM compute
2，以及当前软件模型中的 dequant scale-control queue 2；其中分别有 32,762、1,448、
1,433、212 条 Macro command。实验工具在 `simulate_queue` 开头用缓存的
`packed_macro_imem` presence 作为筛选条件，而这个 Qwen fixture 只有 MEM queue 满足
该条件，导致 CSV 漏掉 MXM。所以下文的“Qwen seq32”数据准确名称应为“Qwen seq32
MEM-only”，不能用来声称 Qwen MXM 已覆盖；修复筛选条件后需要补跑。

这里的 dequant queue 也不表示存在独立反量化数据通路。当前 local-dequant 数据运算
发生在 MXM 内部，但仓库仍为 scale 更新保留独立的 Schedule/Command/ICU control
queue (`MxmDequant`)。若目标 RTL 已决定将 scale 合并进 MXM load command、取消该
control queue，则 compiler/runtime/resource model 尚未同步，本报告中的该项只能视为
当前软件实现覆盖，不能视为最终硬件 queue 定义。

有效 CSV 共 136 行：fetch 60 行、structure 56 行、admission 20 行。所有有效候选按
`(model, queue_kind)` 检查，decoded context count 和 semantic hash mismatch 均为 0。

Qwen2.5-3B seq128 已通过严格 Schedule→Command 和 34,644,016 条 functional issue
等价检查，但被 32K-row MEM 与 Active Context signed 14-bit step 的不一致阻塞。
该问题不属于 fetch 参数性能失败；需要在 16-bit step/256-bit Context 重排与编译器
等价拆 Macro 之间完成选择，再复跑大模型矩阵。

## 3. 参数、含义、候选值和筛选方法

| 部分 | 参数 | 含义 | 候选/状态 | 筛选方法 |
|---|---|---|---|---|
| i-MEM | word width | 每个逻辑 word 的有效位，不含 ECC/padding | MEM 96、MXM 128；固定 | 沿用 packed-v1 接口 |
| Fetch | effective latency | request 到有效 word 进入 reservoir 的总延迟 | **1/2/3/4** | 比较 work、bit wait、prime |
| Fetch | request II | 相邻 read request 的最小发起间隔 | **1/2/4** | 比较供给能力，最终由 deadline 检查定合法性 |
| Fetch | outstanding | 同时在途的 read 数 | 由 L/II 推导 | 至少覆盖 fetch pipeline |
| Buffer | reservoir | fetch 与 parser 间保存的完整 word 数 | **2/3/4** | 选择性能饱和点 |
| Parser | decode window | parser 一次读取的最大对齐窗口 | **32/48/64 bit** | 功能合法性优先 |
| DDB | depth | 每个 physical queue 的 descriptor entry 数 | **1/2/4/8/16** | stall、full、peak、面积代理 |
| DDB | run capacity | 一个 entry 最多表示的 context 数 | **4/8/16** | entry width、prime、cycle |
| Admission | lookahead | 最早可在 start 前多少周期写 Active Context | **2/4/8/16** | deadline、容量、提前驻留 |
| Admission | expand width | 每周期最多展开/admit 的 context 数 | 1；固定 | 每 queue 每周期最多一个合法 issue |
| Scheduler | Active depth | 同时驻留的活跃 Macro 数 | MEM 32、MXM 40；固定 | 本轮观察 peak，不缩容 |
| ECC | code | i-MEM 错误检测/纠正方式 | None、parity、whole/chunk SECDED | 本轮只做位开销和 timing 抽象 |

全组合至少有 `4×3×3×3×5×3×4 = 6,480` 个配置/工作负载，还未计 ECC
码型和交互项，因此没有执行完整笛卡尔积。

## 4. 搜索流程

1. **功能门禁**：context 数和 semantic hash 必须一致；不得出现 parser 越界、
   truncated stream、operand overflow 或 missed issue deadline。
2. **Fetch sweep**：固定 `R3/W64/D8/A8/C8`，执行
   `latency={1,2,3,4} × II={1,2,4}` 全部 12 点。
3. **结构单因素 sweep**：固定 `L3/II2`，分别扫描 reservoir、window、DDB depth、
   run capacity；再补 `D4/C4`、`D2/C4` 两个交互点。
4. **Timing-aware admission**：固定 `L3/II2/R3/W64/D4/C4`，先 prime，再按绝对
   start/final issue cycle、MEM 32/MXM 40 容量模拟 lookahead 2/4/8/16。

semantic hash 覆盖 record index、native instruction、start cycle、inner/outer count、
interval、stride 和 induction target。ECC CModel 尚无 check/correct/fault injection，
因此仅按以下方式折算 timing：

```text
effective_fetch_latency = SRAM latency + registered ECC stages
effective_request_II    = max(SRAM request II, ECC pipeline II)
```

## 5. 评价指标及具体含义

fetch/structure CSV 中的 cycle 指标按 physical queue 计数，并在同一 queue kind 内
求和。真实硬件上的 queue 并行运行，因此这些指标用于候选间相对比较，不能直接当作
模型端到端 wall-clock。

| 指标 | 具体含义 | 判读方式 |
|---|---|---|
| `frontend_cycles` | 单个 queue 从开始 tick 到完整解码、输出全部 context 的周期数 | CSV 对同类 physical queue 求和；表示 frontend 服务能力/余量，不是 application latency，也不是越小就一定提高模型性能 |
| `prime_cycles_max` / max prime | queue 从空状态到 architectural primed 状态的周期；primed 表示到达 DDB decode-ahead 边界或输入结束 | 对同类 queue 取最大值，因为各 queue 可并行 prime |
| `bit_wait_cycles` / bit wait | parser 本周期需要 bit，但 reservoir 数据不足，必须等待 fetch response | 直接反映 fetch latency/II 未被 buffer 隐藏的程度 |
| `decoder_starvation_cycles` | decoder 因缺 bit 无法前进的周期 | 当前实现与 bit wait 同步增长 |
| `reservoir_empty_cycles` | 周期结束时 reservoir 可用 bit 为 0 | 可能包含 parser 当时不读 bit 的周期，不等同于性能损失 |
| `reservoir_full_cycles` | occupancy 加 pending fetch 已使 reservoir 不能继续 refill 的周期 | full 不一定造成 parser stall |
| `fetched_words` | 实际提交到 frontend 的 i-MEM word 数 | 核对读取量和输入规模 |
| `decoder_active_cycles` | parser 或 delta apply 当周期至少完成一次有效工作 | 衡量 decoder 工作周期 |
| `decoder_ddb_stall_cycles` | descriptor 已可 commit，但 DDB 满，decoder 被迫停止 | DDB 是否过小的直接性能指标，目标为 0 |
| `descriptor_count` | 解码形成的 descriptor 数 | run capacity 变小通常会增加该值 |
| `descriptor_decode_cycles` | descriptor/run decode latency 的累计值 | 比较 grammar/entry 组织成本 |
| `max_descriptor_decode_cycles` | 任一 descriptor/run 的最大 decode latency | 检查最坏 descriptor |
| `ddb_entries_committed` | 写入 DDB 的 entry 数 | 与 descriptor 切分和 run capacity 相关 |
| `peak_ddb_occupancy` | DDB 同时占用 entry 的峰值 | depth 不宜长期贴着该值 |
| `ddb_full_cycles` | occupancy 等于 depth 的周期 | 即使无 commit stall，也表示没有吸收抖动的空间 |
| `ddb_entry_bits` | 对应 queue kind 的单 entry payload 位数 | 不含 queue identity 和 CModel trace sidecar |
| `ddb_storage_bits` | `queue_count × depth × entry_bits` | DDB 数据位面积代理，不等于综合总面积 |
| `timing_window_block_cycles` | DDB 有 context，但当前早于 `start-lookahead`，所以有意不 admission | 大量出现通常是正常时间间隔，不是错误 |
| `active_ram_full_cycles` | context 已进入 admission 窗口，但 Active Context 没空槽 | 目标为 0 |
| `capacity_feasible_wait_cycles` | Active RAM 虽满，但已有 context 会在 deadline 前释放，等待仍合法 | 表示容量压力但尚未错过 deadline |
| `admission_stall_cycles` | eligible context 因 Active Context credit 不足而未 admission | 当前主要由 RAM full 触发 |
| `max_admission_to_start_cycles` | 任一 context 从 admission 到首次 issue 的最大提前量 | 应不超过 lookahead；越大则驻留越早 |
| `peak_active_occupancy` | Active Context 最大同时占用 entry 数 | 校验 MEM 32/MXM 40 容量 |
| `contexts_generated` | frontend 实际展开输出的 context 数 | 必须与参考值一致 |
| `semantic_hash` | 展开后 instruction 与完整 schedule 的语义 hash | 同一 workload/queue kind 的合法候选必须一致 |

指标分为两级。硬门槛是语义一致、每个 context 在最晚 admission deadline 前对
scheduler 可见、无同 queue collision、容量可行；这些条件满足后，issue cycle 由静态
schedule 决定，不会因为 `frontend_cycles` 更小而提前。`frontend_cycles`、bit wait、
reservoir/DDB full 等是供给带宽和抗额外 backpressure/PVT 抖动的余量指标，只用于
同为 deadline-feasible 的候选之间比较。若 admission 无同周期 bypass，则严格的最晚
admission 时刻应为 `start_cycle-1`。

## 6. 重点数据和结论

### 6.1 Fetch latency / II

百分比相对同一 workload 的 `L1/II1`：

| Workload | L | II | frontend cycles | 增量 | bit wait | max prime |
|---|---:|---:|---:|---:|---:|---:|
| SmolLM2 | 1 | 1 | 212,497 | 0 | 158 | 85 |
| SmolLM2 | 2 | 2 | 212,813 | +0.1487% | 316 | 87 |
| SmolLM2 | **3** | **2** | **213,129** | **+0.2974%** | **474** | **89** |
| SmolLM2 | 4 | 2 | 213,461 | +0.4537% | 648 | 91 |
| SmolLM2 | 3 | 4 | 214,317 | +0.8565% | 1,662 | 98 |
| Qwen seq32 | 1 | 1 | 99,100 | 0 | 208 | 89 |
| Qwen seq32 | 2 | 2 | 99,516 | +0.4198% | 416 | 91 |
| Qwen seq32 | **3** | **2** | **99,932** | **+0.8396%** | **624** | **93** |
| Qwen seq32 | 4 | 2 | 100,372 | +1.2836% | 856 | 95 |
| Qwen seq32 | 3 | 4 | 100,804 | +1.7195% | 1,496 | 98 |

- latency 2～4 时，II=1 与 II=2 的供给侧统计相同；在选定配置的 timing-aware
  实验中未发生 deadline/capacity 失败，因此选 II=2，并降低 SRAM/ECC request
  datapath 要求。
- L3/II2 给 registered ECC 留出一级，frontend work 增量仍低于 1%。
- II=4 的 bit starvation 明显上升，只保留为低成本边界点。
- L4 可作为 timing fallback，但不是首选实现点。

### 6.2 Reservoir 与 decode window

| Workload | Reservoir | frontend cycles | bit wait | max prime |
|---|---:|---:|---:|---:|
| SmolLM2 | 2 / **3** / 4 | 213,883 / **213,129** / 213,129 | 1,228 / **474** / 474 | 93 / **89** / 89 |
| Qwen | 2 / **3** / 4 | 100,380 / **99,932** / 99,932 | 1,072 / **624** / 624 | 94 / **93** / 93 |

3 words 已达到 4 words 性能，2 words 增加约 0.35%～0.45% cycles，因此选 3。

SmolLM2 的 48-bit window 在 MXM compute queue 0 失败；Qwen 的 48-bit 成功，但
32-bit 在 MEM queue 37 失败。原因均为 parser read 超出 aligned window，所以
64 bit 是当前“Template header 必须原子 peek”微架构覆盖两套 workload 的实现下限，
不是 packed-v1 grammar 的理论下限。若增加跨周期 header assembler，W32/W48 可以
重新成为候选，但需重新评估 decode latency、状态和面积。

### 6.3 DDB depth / run capacity

| Workload | 配置 | frontend cycles | commit stall | full cycles | peak | storage bits | max prime |
|---|---|---:|---:|---:|---:|---:|---:|
| SmolLM2 | D8/C8 | 213,129 | 0 | 0 | 3 | 912,464 | 89 |
| SmolLM2 | D1/C8 | 213,761 | 632 | 134,999 | 1 | 114,058 | 28 |
| SmolLM2 | D2/C8 | 213,145 | 16 | 692 | 2 | 228,116 | 36 |
| SmolLM2 | **D4/C4** | **213,049** | **0** | **0** | **2** | **293,176** | **56** |
| SmolLM2 | D2/C4 | 213,049 | 0 | 173 | 2 | 146,588 | 36 |
| Qwen | D8/C8 | 99,932 | 0 | 0 | 3 | 1,203,072 | 93 |
| Qwen | D1/C8 | 100,840 | 1,260 | 32,762 | 1 | 150,384 | 29 |
| Qwen | D2/C8 | 99,996 | 64 | 1,404 | 2 | 300,768 | 37 |
| Qwen | **D4/C4** | **99,580** | **0** | **0** | **2** | **386,880** | **47** |
| Qwen | D2/C4 | 99,580 | 0 | 760 | 2 | 193,440 | 30 |

`D4/C4` 相对 D8/C8 将 DDB storage-bit 面积代理降低约 67.9%，无 commit stall、
无 full cycle，且 frontend cycles 未退化。D2/C4 当前虽无 commit stall，但已经频繁
满占用且 peak=depth，没有 backpressure/future grammar 裕量，因此选 D4/C4。

选定 C4 后，当前软件模型的单 entry 宽度为 MEM 465、MXM load 452、MXM compute
483、dequant scale-control 372 bit；若最终取消独立 scale-control queue，最后一项应
从 Macro Frontend 实例和面积汇总中删除。

### 6.4 Admission lookahead

| Lookahead | SmolLM2 timeline | 相对 A8 | Qwen timeline | 相对 A8 | Active full | max lead | peak active（Smol/Qwen） |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 36,250,436 | +948 | 113,016,636 | +1,248 | 0 | 2 | 32 / 20 |
| 4 | 36,250,120 | +632 | 113,016,220 | +832 | 0 | 4 | 32 / 20 |
| **8** | **36,249,488** | **0** | **113,015,388** | **0** | **0** | **8** | **32 / 20** |
| 16 | 36,248,224 | -1,264 | 113,013,724 | -1,664 | 0 | 16 | 32 / 20 |

所有候选均没有 Active RAM full 或 missed deadline。表中 timeline 的减少只表示
frontend 更早完成 admission，不表示静态 issue 或模型结束时间提前。A16 会让 context
更早占用 RAM，而 A8 已满足 deadline，并给 fetch/parser/credit 抖动留出 8-cycle
裕量，因此选 8。观察峰值为 SmolLM2 MXM compute 32、Qwen MEM 20，所以本轮不缩减
MEM 32/MXM 40 的 Active Context depth。

### 6.5 ECC

| 方案 | 96-bit physical bits | 128-bit physical bits | 位开销 | 状态 |
|---|---:|---:|---:|---|
| None | 96 | 128 | 0 | 无保护基线 |
| 32-bit lane parity | 99 | 132 | 3.125% | 仅检测，不能纠正 |
| whole-word SECDED | 104 | 137 | 8.33% / 7.03% | **暂定方案** |
| 64/32-bit chunk SECDED | 111 | 144 | 15.63% / 12.5% | correction path fallback |
| 32-bit lane SECDED | 117 | 156 | 21.875% | 开销最大 |

whole-word SECDED 理论位开销最低，L3/II2 又为 registered correction 留出一级，
所以作为首选。但本次没有验证 foundry SRAM 粒度、综合 Fmax/面积/功耗、单错纠正、
双错检测和 poison/trap 行为，ECC 码型仍是暂定项。

## 7. 最终选定值与后续门槛

建议冻结的 frontend 行为：

```text
logical word             = MEM 96 / MXM 128 bit
effective fetch latency  = 3 cycle
request II               = 2 cycle
outstanding reads        = 2
reservoir                = 3 words
decode window            = 64 bit
DDB depth/run capacity   = 4 / 4
admission lookahead      = 8 cycle
context expand width     = 1 / cycle
```

暂定实现点：SRAM latency=2、ECC latency=1 registered cycle、whole-word SECDED、
Active Context depth=MEM 32/MXM 40。

尚未关闭：

1. 评估 32K MEM Active Context step 扩到 16 bit并重排 256-bit layout；编译器拆
   Macro 作为兜底；
2. TinyLlama/Qwen seq128 合法化后复跑；
3. SRAM+ECC 综合 PPA、physical width/padding 和 fault injection；
4. RTL/gate-level 验证 outstanding=2、response ordering 和 backpressure；
5. D2/C4 synthetic burst/backpressure 压力测试，只作为未来降面积候选。

## 8. 形成的产物、数据和日志

### 8.1 程序和文档

- `runtime/examples/macro_frontend_fetch_sweep.cpp`：fetch、structure、admission 三种模式；
- `docs/macro_frontend_fetch_sweep_round1.zh-CN.md`：第一轮 fetch/ECC timing 记录；
- `docs/macro_frontend_pipeline_parameter_freeze.zh-CN.md`：冻结建议；
- `docs/macro_frontend_pipeline_parameter_experiment_report.zh-CN.md`：本汇报。

运行接口：

```text
macro_frontend_fetch_sweep program.ftlpu
macro_frontend_fetch_sweep --structure program.ftlpu
macro_frontend_fetch_sweep --admission program.ftlpu
```

### 8.2 原始 CSV

目录 `out/macro-frontend-pipeline-sweep/`：

- `fetch-smollm2.csv`：48 行；
- `fetch-qwen_seq32.csv`：12 行；
- `structure-smollm2.csv`：44 行；
- `structure-qwen_seq32.csv`：12 行；
- `admission-smollm2.csv`：16 行；
- `admission-qwen_seq32.csv`：4 行。

### 8.3 日志和构建物

- `structure-smollm2.stderr.txt`：48-bit window 在 MXM compute queue 0 的 expected
  functional rejection；
- `structure-qwen_seq32.stderr.txt`：32-bit window 在 MEM queue 37 的 expected
  functional rejection；
- `admission-smollm2.stderr.txt`、`admission-qwen_seq32.stderr.txt`：0 byte；
- `build-codex-macro-vs-verify/runtime/Release/macro_frontend_fetch_sweep.exe`：已重新构建；
- admission smoke test 已复跑：四个 lookahead 的 context count 均一致，Active RAM
  full 均为 0。

输入 fixture：

- `out/compression-generalization-smollm2-seq128/decoder_layer.ftlpu`；
- `out/macro-rectangle-review/qwen-seq32-macro.ftlpu`。

Qwen seq128 的临时 Schedule/Command/partial binary 超过 2.7 GB，在完成严格转换和
functional issue 等价检查后已清理，没有作为长期产物保留；关键检查结果已记录在
冻结说明和本汇报中。
