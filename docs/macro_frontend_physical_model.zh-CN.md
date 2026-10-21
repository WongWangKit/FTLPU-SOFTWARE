# Macro Frontend 首版物理模型

## main 中的接入方式

本功能按独立能力从实验分支迁移。现有 `macro` 模式继续使用 main 的
compact/FU3D 路径；新增的物理前端通过显式 `raw-macro` 模式选择：

```text
ftlpu-compile --input model.stablehlo.mlir --input-stage stablehlo \
  --output program.ftlpu --target-config target.json \
  --icu-compression raw-macro --verify-icu-issues
ftlpu-translate --input command.mlir --output program.ftlpu \
  --icu-compression raw-macro --verify-icu-issues
macro_frontend_fetch_sweep program.ftlpu
macro_frontend_fetch_sweep --structure program.ftlpu
macro_frontend_fetch_sweep --admission program.ftlpu
```

生产入口已贯通 `ftlpu-compile`、`ftlpu-opt` 和 `ftlpu-translate`。
`ftlpu-opt` 把模式保留在 IR 中，后续入口在没有显式覆盖时继承该模式。
默认 direct lowering 仍使用 FU3D。Command IR 的 `*_3d` 继续表示共同的粗循环域；
最终生成二进制时，`raw-macro` 才把 MEM/MXM 域按 parity、terminal mode、
blocked address 和仿射轴拆成二维 Macro，VXM/SXM 等保持原生 FU 路径。
生产转换不展开整条功能指令流；一个源 packet 拆成多个描述符时，地址和 scale
重定位逐一复制到对应描述符，且不跨不同源 packet 合并。

packed-v1 仍不内嵌同步控制记录。同一 Raw 队列中的 `WAIT_EVENT`、SYNC 和同步 MEM
传输不能直接编码；尚未覆盖的 `WRITE_READ_2D`、native4 decode 继续明确拒绝。
分页程序改由 v37 公共屏障表统一冻结 Native/Macro 计算，通过独立 Native MEM transport
保留实际 DMA/RX 启动、同步写入及 KV page-out 控制。最终链接必须证明候选边界的
排空和 SRAM 生存期，并完整关联实际 MEM commit，之后才允许装载 Raw 分页程序。
无法证明时明确拒绝，禁止静默退回 FU3D。全驻留 StableHLO FFN 已有真实前端数值回归；
完整 decoder 的实际结果及剩余边界见 [公共屏障验证记录](common_barrier_validation.zh-CN.md)。

`--verify-icu-issues` 现在可直接比较 FU3D 与 Macro 的功能指令及绝对发射周期。
此开关不替代物理解码供给门禁；集成回归另用真实 ICU 逐周期回放，检查 cold prime、
Active Context 和 admission deadline。二进制 `MacroPacked` 模式在读取后保留为
必选 raw 队列，后续插入原生控制时，序列化和 loader 会报错，防止绕回旧 FU3D loader。

raw 模式面向 96-bit MEM / 128-bit MXM、至多二维的仿射 Macro；其他 FU
继续使用 main 原有路径。主机重定位后，loader 从更新后的语义指令重新生成镜像，
防止使用失效的编码缓存。新写入容器版本为 v37，main 的 binding layout 编号保持不变；
实验分支的 v33 布局不作为兼容格式引入。软件与更新后的 CModel 需配套构建。
本轮验证对应 CModel main 的 `a99f3bc`；使用工作树时应通过
`-DFTLPU_CMODEL_DIR=/path/to/cmodel-main` 指向包含该能力的 CModel。

冷启动时，issue clock 保持停止，在 parser/DDB 已 prime 后还需完成 cycle-zero
context 的 admission，使 start_cycle=0 的指令在首个执行周期可见；远期 context
仍受 lookahead 和 Active RAM credit 约束。原实验代码仅检查 parser/DDB 已 prime，会漏掉最后一个
尚未写入 Active RAM 的 context，这一边界在迁移时增加了回归并修复。

逻辑压缩等价不代表 frontend 一定来得及供给。回归同时验证了有充分解码间隔的
32-context 场景，以及每周期切换独立 template 时明确拒绝 admission deadline 违例；
模型不会把应发射的指令静默延后。

下面的设计记录和参数扫描保留实验时的口径。当前默认参数仍为 fetch L1/II1、
reservoir=3、window=64、DDB=8×8、lookahead=8；L3/II2、DDB=4×4 是独立扫描
配置和冻结建议，不能视为已经生效的默认配置，也不表示实现了 ECC 数据路径。

HF 整栈工具及未通过数值门禁的完整模型路径没有随本功能迁移。

## 结论

本版在不修改 packed-v1 descriptor ISA 和 compiler 输出的前提下，将 Macro
frontend 冻结为以下逻辑流水：

```text
per-slice fixed-width Macro IMEM
  -> 3-word reservoir (word FIFO + head_bit_offset)
  -> aligned Wdecode window
  -> prefix predecode + one-state-per-cycle parser FSM
  -> finite decoded descriptor buffer (DDB)
  -> <= 1 context/cycle lazy expander
  -> start_cycle-based late admission + Active RAM credit
  -> finite Active Context RAM
  -> abstract issue/completion interface
```

实现采用 `decode early -> compact state waits -> expand when needed -> admit
late`。descriptor decode 完成不再等价于 Active Context RAM allocation。

## 1. 旧实现的问题

旧 raw Macro decoder 已经具有固定宽 fetch 和有限容量的概念，但仍有四个物理建模偏差：

1. reservoir 是 `deque<uint8_t>`，每次 consume 对每个 bit 执行 `pop_front`，等价于逻辑上的全 reservoir 移位；
2. 一个 tick 内用循环连续穿过 dictionary、initial state、run header、template 等多个 parser state，decode latency 接近零延迟；
3. 长 template 必须一次完整出现在 reservoir 中，没有独立的 `Wdecode` 和 multi-cycle field assembly；
4. decoder 直接输出完整 Macro context 到 `active_macros_`，没有 DDB。decode ahead 会提前分配 Active RAM，并把 Active context lifetime 人为延长到 decode 时刻。

旧链路实际是：

```text
IMEM -> bit deque / combinational parser -> full context -> Active RAM
```

## 2. 新 frontend pipeline 与阶段职责

| 阶段 | 本版职责 | 明确不承担的职责 |
|---|---|---|
| Macro IMEM | 每次返回一个 slice-local 固定宽 word；MEM=96 bit，MXM=128 bit | variable-width memory transaction |
| Reservoir | 保存至多 `reservoir_words` 个完整 word；用 head word + head bit offset 表示流位置 | 全宽 barrel shift |
| Prefix predecode | 判断 run 长短格式、template extended/compact、shape、delta dictionary/escape path | 重新定义 descriptor ISA |
| Parser FSM | 每周期执行一个 state；每周期最多消费 `Wdecode` bit；长字段分周期 assemble | 单周期解析完整 descriptor |
| DDB | 保存 decoded template、base timing/operand、compact deltas、run progress、residency state | 保存 N 份 full Context |
| Lazy expander | FIFO head 每周期最多生成一个 context | 多 context/cycle 批量展开 |
| Admission | 检查 timing window 和 Active RAM credit | issue match 与 context update |
| Active Context RAM | 固定深度、allocation/free、valid lifecycle，以及显式 `next_issue_cycle` | descriptor decode 与 DDB storage |
| Issue scheduler | 每个 queue 平坦比较全部 valid context；唯一 due context 发射并更新 | 动态优先级、冲突后的延迟发射 |

各 stage 在同一 tick 并行推进。一个 cycle 内最多发生一次 fetch commit、一次
parser state、一次 DDB commit 和一次 context admission；本周期 admit 的 context
下一周期才对 issue 可见，没有组合 bypass。

## 3. 参数化 baseline

参数由 `IcuMacroV1Decoder` / `DistributedIcuQueue` 模板公开：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `InstructionBits` / `WordBits` | slice-specific | MEM 96，MXM 128；没有全芯片统一 fetch width |
| `FetchLatency` | 1 | 固定宽 IMEM read latency |
| `MacroReservoirWords` | 3 | baseline 容量为 `3F` bit |
| `MacroDecodeWindowBits` | 64 | 测试也覆盖 32-bit window |
| `MacroDdbDepth` | 8 | 首版 sweep 起点，不是最终 RTL 深度 |
| `MacroAdmissionLookahead` | 8 | 首版 sweep 起点，不是最终值 |
| `MacroDdbRunCapacity` | 8 | 每个 DDB entry 最多保存的 compact run segment |
| `MacroContextDepth` | 现有 FU 配置 | 有限 Active Context RAM depth |
| context expansion width | 1 | 本版冻结为 1 context/cycle |

`MacroDdbRunCapacity` 不是 8 份 Context。一个 entry 只保存一份 common
template、一个 absolute base state 和最多 7 个 compact `(start, operand)` delta。

## 4. Reservoir RTL-like structure

```text
words[0] words[1] words[2]
   ^
   +-- head_word (logical FIFO front)
       head_bit_offset
       valid_bits
```

refill 只在 `valid_words + pending_fetch < reservoir_words` 时发起。consume
只增加 `head_bit_offset`；跨过 `F` 后才 pop 一个完整 word。aligned peek 从
`current_word + next_word` 取不超过 `Wdecode` 的 window。因为
`Wdecode <= F`，不需要访问第三个 word，也不需要移动未消费 payload。

## 5. Decoder FSM、latency 与 throughput

主要状态为：

```text
WaitingForControl
DictionaryCount -> DictionaryEntry*
InitialState
RunPrefix -> optional RunLength
Template (one or more Wdecode chunks)
Record / Delta (one or more Wdecode chunks)
CommitDescriptor
RunPrefix ... -> Done
```

dictionary entry 为 36 bit，MEM initial state 为 `32 + SRAM address bits`，wide
delta 最长约 70 bit，extended template 可超过 200 bit。它们均可在
`Wdecode=32` 下 multi-cycle assemble。因而：

- descriptor decode latency 是实际 parser cycle 数，不再是零延迟函数调用；
- parser 每周期最多消费 `Wdecode` bit；
- encoded run 的完成率上限为一条/多个 cycle；
- DDB 满时 `CommitDescriptor` 保持状态且不消费 reservoir；
- reservoir 随后变满并停止 IMEM fetch，构成完整 backpressure chain。

## 6. DDB entry 与 run 分片

packed-v1 run 允许每个 record 使用不同 dictionary/escaped delta，因此不能只用
一个固定 stride 表示任意 65536-record run。为保持 ISA 兼容且避免无限 vector，
本版把长 run 切成固定上限的 compact DDB segments：

```text
first_record_index
decoded common template/schedule
absolute next_start_cycle
absolute next_operand
compact delta[DdbRunCapacity - 1]
run_count
next_context_index
decode_start_cycle / commit_cycle
```

parser 最多持有一个未 commit 的 assembly entry；所有已完成 entry 受
`DDB_DEPTH` 严格限制。默认建议从 depth=8 开始 sweep。判断 depth 足够的条件是：

- workload 中 `ddb_full_cycles` 和 `decoder_ddb_stall_cycles` 很低；
- reservoir starvation 不因 DDB 过浅恶化；
- 增大 depth 不再减少 missed-deadline risk 或 frontend prefill cycles。

## 7. Lazy expansion 与 admission

DDB FIFO head 的 next context 满足下式才 materialize：

```text
admit_cycle = max(0, start_cycle - admission_lookahead)

current_cycle >= admit_cycle
&& active_context_credit > 0
```

成功后只产生一个 context，更新 `next_context_index` 和 next base/delta state；
entry 完全展开后释放。没有 unbounded expanded-context queue。

Active context 在 admission 时 allocation，在该 descriptor 最后一个 issue point
完成时 free。CModel 记录 admission-to-start latency 和 admission-to-completion
lifetime。若 decoder/admission 晚到使 context 无法在 start 前可见，现有 static
schedule error 继续报错，不会静默改变执行周期。

## 8. Active occupancy 对比

新增 micro test 使用 32-context run、DDB depth=2、Active RAM depth=1、lookahead=8：

| 模型 | peak Active contexts | 说明 |
|---|---:|---|
| 旧式 decode 后立即展开 | 32（需要同等容量才能不阻塞） | 所有 context 从 predecode 起长期存活 |
| 新 DDB + lazy admission | 1 | far-future state 留在 compact DDB |

该 micro test 的 peak 降低 `31/32 = 96.875%`，且 observed active lifetime 不超过
9 cycles。这个数字只代表定向测试，不代表完整模型 workload；真实 MEM/MXM
深度、DDB depth 和 lookahead 必须使用编译后的 seq32/seq128 workload sweep。

## 9. Compiler ordering 与 HOL

packed-v1 encoder 已要求所有 logical Macro record 的 `start_cycle` 全局非递减。
因此 FIFO DDB 可以安全地以 next admission time 处理。测试明确拒绝：

```text
A0.start = 100
A1.start = 10000
B0.start = 150
```

当前格式还存在一个限制：同一 run 的后续 delta 位于前一 record 之后，decoder
不能随机跳到未来 descriptor。固定容量 run segmentation 是本版最小硬件补充，
没有引入 associative DDB scheduler。若未来取消 compiler monotonic contract，
应先评估在 encoding 中增加显式 run length/offset/index，而不是直接增加 CAM。

## 10. Issue Scheduler 接口边界

Issue scheduler 已冻结为每个物理 queue 独立的一份固定深度 Active Context RAM。
MXM baseline 深度为 40；每个 entry 单独保存 valid 和 32-bit
`next_issue_cycle`。正常发射路径对全部 entry 执行：

```text
match[i] = valid[i] && (next_issue_cycle[i] == current_cycle)
```

40-bit match vector 全零时等待，one-hot 时发射对应 context，多命中时报告静态
schedule 冲突。编译器已保证合法 binary 在同一个 queue/cycle 最多一个 issue，
因此 scheduler 没有动态 priority policy，也不会在冲突时选择一个 context 并推迟
其他 context。

接口事件为：

```text
context_admitted(context, valid)
context_ready(issue_cycle <= current_cycle)
context_issued
context_state_updated (loop index advance)
context_completed -> Active RAM credit returned
```

每周期先对周期开始时可见的 context 做 match/issue/update/complete，再接收最多一个
admission。新 admission 在下一周期才参与 match，没有组合 bypass。非末次 issue
会在同周期写回 counter 和下一次绝对 issue cycle；末次 issue 释放 slot，该 slot
可以在本周期末被 admission 复用。

首版不使用 priority queue、heap、calendar wheel 或分层 scheduler。RTL 可以把
match-vector reduction 和 payload one-hot mux 综合为树，但逻辑结构仍是 flat match。
正常关键路径只需要 32-bit equality compare；`late` 与 multi-hit 属于错误检测，
不参与动态仲裁。

## 11. 统计与派生指标

frontend 暴露以下原始计数，可直接派生平均值和利用率：

- IMEM：fetch words、frontend cycles；`fetch utilization = fetched_words / cycles`；
- reservoir：occupancy bit samples、empty/full cycles、peak bits、starvation cycles；
- decoder：MEM/MXM descriptor distribution、decode cycle sum/max、payload bits、DDB stall；
- DDB：entry count、occupancy samples/peak、full cycles、residency sum/max；
- expansion/admission：generated contexts、timing-window block、Active-full block、admission count；
- Active RAM：occupancy samples/peak、admission-to-start sum/max、active lifetime sum/max。

平均 decode cycles/descriptor、bits/cycle、DDB occupancy、descriptor residency、Active
occupancy 和 lifetime 均由 sum/count 计算，避免在每个 queue 内提前舍入。

## 12. 对现有 functional implementation 的逐项回答

1. **descriptor decode 是否被当作零延迟？** IMEM fetch 已有 cycle，但旧 parser
   会在一个 tick 内循环越过多个 semantic state，长 template 也原子解析，因此
   field-level decode latency 实际被严重低估；新模型每 state/window 至少一个 cycle。
2. **是否一次产生全部 contexts？** 旧 raw decoder 并非一个 tick 批量产生 N 个，
   但每个 decoded record 都直接成为 full active context；没有 compact waiting stage。
   新模型一个 run 可先成为若干 compact DDB segment，再逐个 lazy expand。
3. **Context 何时进入 Active RAM？** 旧模型在 record decode 完成时；新模型在
   `current_cycle >= start_cycle - lookahead` 且 credit 可用时。
4. **旧模型是否拉长 lifetime？** 是。priming/decode ahead 会提前填 Active RAM。
   新模型允许 DDB 长驻而不提前 materialize context。
5. **peak occupancy 降低多少？** 定向 32-context test 从 eager 所需 32 降至 1，
   即 96.875%；完整 workload 数值尚需 seq32/seq128 sweep，不能用该数字外推。
6. **decoder 是否成为瓶颈？** 首版不能先验断言。现在可用
   `descriptor_decode_cycles / descriptor_count`、bits/cycle、starvation 和 DDB stall
   分离判断；32-bit window correctness 已验证，但吞吐选择尚未冻结。
7. **Reservoir 是否 starvation？** 模型会逐 cycle 记录 starvation；golden 与长
   descriptor 测试均能恢复。真实 workload 的发生率仍需 profiling。
8. **DDB 多深足够？** 默认 8 是 sweep 起点；depth=2 压测可正确反压和恢复，
   但最终深度必须看 full/stall/residency 曲线。
9. **lookahead 应多大？** 默认 8；至少覆盖 admission write-to-visible latency 和
   credit conflict，最终按 missed deadline 与 Active lifetime 的 knee point 选择。
10. **1 context/cycle 是否足够？** 定向 32-context test 足够且语义正确；是否满足
    完整模型峰值只能由 `contexts_generated / cycles` 与 admission stall profile 决定。

## 13. 测试覆盖

| 场景 | 覆盖方式 |
|---|---|
| A 单 word descriptor | MEM/MXM golden raw images |
| B 跨 IMEM word | 96-bit MEM golden vector和 packed run |
| C 大量短 descriptor/context | 32-context packed run |
| D 单个长 descriptor | extended MEM template，32-bit window multi-cycle parse |
| E single context | MEM/MXM/MXM-dequant golden vectors |
| F multi-context | 32-record run，1 context/cycle expansion |
| G 大 start gap | prime 后 Active occupancy 保持 0，直到 admission window |
| H Active RAM backpressure | depth=1，检查 credit stall 和按原周期恢复 |
| I DDB full | depth=2，检查 parser stall/full cycles 和恢复 |
| J HOL | 非单调 `100, 10000, 150` 在 encoder 被拒绝 |

## 14. DDB depth 与 lookahead 建议

首轮 RTL exploration 建议 sweep：

```text
DDB depth:             2, 4, 8, 16
decode window:         32, 64
reservoir words:       2, 3, 4
admission lookahead:   2, 4, 8, 16, 32
Active RAM depth:      workload peak around the knee point
```

默认 DDB=8、lookahead=8 只是安全起点。lookahead 必须至少覆盖 admission write 到
首次 issue 可见的 pipeline latency，并为短期 Active credit conflict 留余量；过大则会
重新拉长 Active lifetime。

## 15. 本版冻结项

- slice-local fixed-width IMEM fetch；
- word-buffer reservoir + head pointer；
- 不使用 full-reservoir barrel shift；
- configurable aligned decode window；
- prefix predecode + cycle-stepped parser FSM；
- descriptor decode ahead；
- finite DDB and explicit backpressure；
- compact run segmentation，不保存 N 份 full Context；
- lazy expansion，最大 1 context/cycle；
- `start_cycle` late admission；
- finite Active Context RAM credit；
- compiler start-time monotonic ordering contract。

## 16. 尚未冻结、RTL 前需继续评估

- 最终 SRAM macro、fetch latency 和各 FU 的 IMEM width；
- 32/64-bit decode window 的面积、时序和实际吞吐；
- DDB depth、run segment capacity、entry packing 和 RAM 实现；
- admission lookahead 与 Active RAM depth 的 workload sweep；
- reservoir 2-word 是否足够，以及 refill latency 敏感性；
- parser/predecode 的具体 pipeline register placement；
- Active Context RAM 的具体寄存器/RAM 映射和物理布线；
- 编译器按 40-entry MXM 容量做精确峰值验证与必要的 Macro 时间分块；
- 多 queue 同时 frontend 活动时的功耗和 clock gating；
- packed-v1 arbitrary-delta run 是否值得在下一版 ISA 中增加 stride mode 或显式 index。

最终 RTL 参数应依据真实 compiler 输出的统计选择，而不是由本 C++ 容器布局反推。
