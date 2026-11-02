# Macro 公共屏障：实现、验证与边界

> 2026-10-27 更新：本文保留 v37 串行阶段的历史记录。计算/分页重叠和 seq128 真实执行已完成当前范围验收，最新结果见 [重叠验收记录](common_transport_overlap_validation.zh-CN.md)，新格式见 [v38 协议](common_barrier_binary_v38.zh-CN.md)。

验证日期：2026-10-21。实现位于两个工程的 main worktree，包含公共屏障协议及后续 seq128 编译修复；没有整体合并实验分支。本文区分已验证结果和仍未完成的执行验证。

- SOFTWARE：`C:/Users/Jiang_Hua/.codex/worktrees/port-seq128-correctness/FTLPU-SOFTWARE`，基础提交 `2410f11`。
- CMODEL：`E:/Fudan_University/internship/timesintelli/LPU/work/FTLPU-CMODEL/out/main-mature-macro`，基础提交 `a99f3bc`。
- 构建：`E:/Fudan_University/internship/timesintelli/LPU/work/FTLPU-SOFTWARE/out/main-common-barrier-ninja`。
- 本文所述日志、二进制和统计：`E:/Fudan_University/internship/timesintelli/LPU/work/FTLPU-SOFTWARE/out/main-common-barrier-e2e`。

## 推送前合并远端 main 后的复验

本次合入 SOFTWARE 远端 `1dd422a` 与 CMODEL 远端 `20020fe`，没有合并实验分支。CMODEL 的协议提交为 `ae19ce6`，合并后 main 为 `bc84188`。SOFTWARE 同时保留本地 Raw CLI、staging、seq128 修复及远端的可配置硬件、直接 lowering 和原生执行更新。

合并时修正了三处衔接：大 SRAM 配置继续使用已有平面 input staging，避免 distributed 直接读与 QKV 写回占用同一 MEM Context；MXM→VXM 的返回相位按实际 VXM 列数和 MXM 首结果延迟推导，避免固定提前两拍；驻留 QKV 保留已支持的 N 半波尾块打包。RoPE 测试同时检查新增输出返回阶段，驻留尾块按地址逐字节对照完整补零镜像，没有删除资源检查、同步或搬运控制。

- SOFTWARE 最终相关回归 **14/14** 通过，55.55 秒，日志 `push-final-regressions.log`。包含真实 Raw CLI、seq128 完整双路径编译、容器/运行时/ModelSession/ping-pong 及相关编译器测试。
- CMODEL 合并后相关回归 **9/9** 通过，2.02 秒，日志为 CMODEL main 的 `out/push-merge-tests.log`。
- `weight_page_builder_test --paged-output-tail` 通过，保留分页 16/32/64 的覆盖，并增加已支持的 32 字节物理行驻留 attention 尾块验证。
- seq128 默认产物 **1,640,659 字节**，Raw Macro **343,391 字节**，均经完整编译、二进制读取和 i-MEM 检查；MEM/MXM 没有静默回退。seq128 真实分页数值执行仍未验证。

合并后重新编译并实际执行 seq32 完整 attention + FFN，再跑两档 DDR。18,432 个 BF16 输出与原生基线逐字节一致，相对 CPU 最大误差 0.03125；attention residual 误差 0，RMS2 最大误差 0.015625。FFN hidden 的硬件参考误差及东西副本误差均为 0。两次 linked 文件完全相同，SHA256 为 `D7695E25FE65DE8171EBEF2C57707BB0C3742246054B9F902CBD6E503ECF0B66`。

| 合并后实测 | DDR 51,200 MB/s | DDR 25,600 MB/s |
| --- | ---: | ---: |
| 主执行逻辑周期（max_cycle + drain 64） | 79,972 | 79,972 |
| HOLD 物理周期 | 29,193 | 55,528 |
| 主执行物理周期（逻辑周期 + HOLD） | 109,165 | 135,500 |
| 启动权重等待（另计） | 1,849 | 3,375 |
| 屏障 / page-use 实例 | 2 / 7 | 2 / 7 |
| MEM 同步写控制 / 实际 FU 写 | 104 / 82,944 | 104 / 82,944 |

最终表为 `(29556,196608,members=0,1)`、`(62092,196610,members=2..6)`；两个有成本排空窗口仍各为 62 个逻辑周期。计算 i-MEM 为 92,992 字节，搬运控制为 9,144 字节，有效 Barrier RAM 为 16 字节，合计 102,152 字节；完整 linked 容器为 152,318 字节。目标实际配置的 Barrier RAM 容量仍为 8,192 字节，不等于本程序使用的 16 字节。

日志 `push-decoder-51200.log`、`push-decoder-25600.log` 和 `push-decoder-linked.inspect.log` 保存本次执行及存储统计。以下首次验收表格和文件哈希保留为历史记录；当前 main 的合并后结果以本节为准。静态逻辑等价工具仍不能展开运行时事件控制；没有通过排除搬运队列绕过该限制。本次针对性回归不等同于全量模型/硬件配置验收。

## 已实现的协议

每任务每芯片一张公共表，容量 1024 项，每项 `{uint32 stop_cycle, uint32 event_id}`。当前、下一项缓存；成员完成状态有 65535 的有限上限。空表有效，损坏、溢出、孤立事件、重复项、过期周期和未完成最终链接均明确报错。`generation` 是运行时状态；实例编号指向一次具体装入，重复装入相同页不能复用旧完成状态。

`P` 每个物理 tick 前进。`L` 仅在 `compute_enable` 时前进；计算发射、Native PC/NOP/循环、Macro admission/deadline 和 Context 坐标均使用 L。DDR、C2C、事件年龄、MEM transport 通知和超时使用 P。`run_cycles(N)` 的 N 是逻辑预算，HOLD 不消耗它。

```text
INIT/PRIME -> RUN(L<C) -> 在任何计算发射前检查 L=C
                              |
                    事件未完成 -> HOLD(L=C)
                                  | DDR/RX/MEM commit 继续
                                  | 所有具体成员完成并锁存
                                  v
                         下一 dispatch 边沿恢复 RUN，发射 C
```

早到事件仅锁存，到 C 才消费。重复完成幂等；旧 generation 忽略，未来 generation/未关联实例拒绝。多个同 C 依赖聚合为 all-of。事件清理和表配置先于执行期 DMA launch；C=0 的 DMA/RX/transport 原生队列显式 launch，避免懒配置清掉第一条通知。

INIT/PRIME 使用真实有限前端，保留 i-MEM 延迟、有限 IQ/reservoir、parser FSM、delta、DDB、Context credit。HOLD 不解析、展开或 admission；已发出的 fetch 请求在发起时预留 credit，HOLD 仅捕获和老化响应，不能覆盖或重复消费。非空 Context 可跨屏障保留，不要求 Context 全部结束。

独立有限 Native MEM transport 控制 RAM 与 Raw Macro i-MEM 分离，物理上仲裁同一 MEM 发射口；同 bank 同拍计算/transport 冲突直接失败。页入的完成条件是全部真实 SRAM 写入已可见，至少经过末 tile 提交，并且实际 SR/FU transient 已排空。DDR 完成或同步包发射完均不等于页就绪。

## 安全点、编译与最终链接

编译器保留独立于 Macro 编号/bit offset 的逻辑周期候选。完整算子边界，以及 Swish 隐藏值和镜像全部写入后的 Down 边界，可插入显式排空窗口。`common_barrier.drain_cost` 记录有成本的重调度；当前 lpu32 一次窗口为 62 个逻辑周期。不能根据偶然空闲间隙推断稳定数据已安全。

链接器检查窗口中不存在计算发射，并在 C 前保守排空全片路径。运行时在第一次到 C 和事件放行后的恢复边沿再次检查 MEM/MXM/SXM/VXM（包括 VXM 内部普通 SR）的在途状态。稳定 SRAM、accumulator、weight、activation 和可保持状态不因 Context 活跃而被清空。

SRAM 生存期验证同时覆盖最终链接、已链接容器装载及启动前权重加载：

- 明确关联到当前权重的重叠读允许消费新页；其他活跃读不能被覆盖。
- 页释放前禁止重叠计算写；旧页释放后，只有已证明无地址空洞、完整覆盖且末 tile 已提交的写域，才能供后续读使用。
- 后续权重换入必须来自另一条已最终关联的实际 packed MEM 段，经过自己的公共屏障，并且在旧页释放后、相应消费前完成。不能用 placement 的 min/max 包络假装真正写满 SRAM。
- 最终装载重新检查相互覆盖页面的释放顺序，及 DMA/RX/MEM 队列依赖图；不允许事件依赖 C 之后被冻结的计算。

首版将页入放在 HOLD 内，按物理完成顺序串行，不与计算重叠。搬运时长不叠加进下一页逻辑下限。同一分页事件的原 Native WAIT/SYNC 映射为公共表；DMA/RX launch、DMA completion SYNC、MEM 同步写和后续 KV 同步读保留。公共路径不再叠加旧的 `phase_offset` 释放补偿。运行时重排页计划会清理派生控制，重新构造关联和屏障。

ModelSession 的公共路径先构建完整最终链接及复用证明，再执行启动前权重搬运；bootstrap 事务隔离于执行事件，最终 runtime load 建立新的 generation/表和硬件镜像后才允许运行。默认非公共路径保持原有流程。Raw 要求仍严格；无法证明的程序报错，不退回 FU3D。

## 容器和 ABI

写入版本从 v36 升为 v37，Macro packed-v1 仍为 1。完整布局和逐字段编码见 [v37 格式说明](common_barrier_binary_v37.zh-CN.md)。

事件元数据之后、计算/transport 队列之前固定存在 BARR 块：`BARR`、LE uint16 version=1、LE uint16 entry_bytes=8、LE uint32 count、LE uint32 reserved=0；随后每项两个 LE uint32。零项也保留 16 字节头，不直接序列化 C++ 结构。

v37 增加事件关联、公共协议标识、transport 队列计数、page-use 的 launch ID、address relocation 的 transport selector；硬件/ABI 增加 barrier encoding version 和容量。stream/span 和两种 metadata-only reader 均处理新块。新 reader 保留旧文件路径；v36 兼容有实际 fixture 验证。旧加载器/不支持目标拒绝新公共程序。

## 实际测试

SOFTWARE 最终相关回归 10/10 通过，日志 `software-final-regressions.log`：公共运行时、二进制屏障、Macro bitstream、默认 FU3D lowering、ModelSession C2C、page-ready sync、MEM trace、模型包、ping-pong 链接和真实 Raw CLI。CLI 包含 StableHLO FFN 的真实数值执行以及分页 decoder 候选编译，不以它代替完整分页执行。最后一次布局元数据调整后单独复跑 CLI，通过（`software-final-cli.log`，34.15 秒）。

CMODEL 相关回归 9/9 通过（`build/Testing/Temporary/LastTest.log`）：公共/旧屏障、Macro schedule、Raw decoder、MEM 3D、VXM RUN2D、SXM RUN2D、双芯片 C2C、DMA/DDR。覆盖多个活跃 Context、单个 Macro 跨屏障、32 个解码阶段、已发未返的 Native/Macro fetch、连续屏障、零等待、早/晚/重复/旧 generation、all-of、重复装入、真实 SR 到 SRAM commit、共享端口冲突及不安全恢复拒绝。

二进制测试覆盖空表/多表四读取路径、截断、magic/version/entry_bytes/reserved、坏事件/实例/launch、重复 C、32 位周期溢出、容量、transport relocation 和旧版兼容。核心实际搬运测试额外覆盖十类有效编码但关联错误的 DMA/RX/MEM 包，以及稳定数据覆盖、释放前写、完整重写、写域空洞、未来换页和错误未来绑定。

`runtime_common_barrier_test` 在 DDR read latency 3/53 下运行同一真实分页程序：L 都为 516，P 分别 579/630，等待 63/114。Macro 的 `(L,address)` 都为 `(0,600),(256,601),(512,602)`；Native 都为 `(32,700),(256,701),(513,702)`。C=0、reload generation 隔离和 SRAM 全字节可见性均通过。ModelSession 还实际验证了一个启动预加载 bank 加一个运行期 refill bank。

## 真实分页 decoder

已用新增的 `compiler/examples/smollm2_135m_decoder_layer/decoder_layer_seq32.stablehlo.mlir` 跑通完整 attention + FFN。它保留 decoder 运算和 KV 通路，权重使用 harness 的非零数值模式；不是删除 SYNC 的码流回放。

先运行带 CPU 检查的原生基线，再让 ModelSession 实际执行 Raw Macro + Native、DDR/C2C、MEM commit、公共等待/恢复及 KV page-out，最终 18,432 个 BF16 值与基线逐字节一致。相对 CPU 参考最大误差 0.03125，attention residual 误差 0，RMS2 最大误差 0.015625，FFN Down checkpoint 最大误差 0.03125。

| 项目 | DDR 51,200 MB/s | DDR 25,600 MB/s |
| --- | ---: | ---: |
| 执行逻辑周期（含末尾 drain 64） | 80,029 | 80,029 |
| RUN/HOLD 物理周期 | 109,305 | 135,565 |
| 其中 HOLD 物理周期 | 29,276 | 55,536 |
| 启动权重等待（另计） | 1,849 | 3,375 |
| 公共屏障 | 2 | 2 |
| 页实例 | 7 | 7 |
| 页入 MEM 同步控制 | 104 | 104 |
| 实际同步 FU 写 | 82,944 | 82,944 |
| KV 页出 MEM 同步读控制 | 56 | 56 |

两档带宽最终 linked 文件均为 166,296 字节，SHA256 完全相同：`C962978985DF6FE185F03592FFB2EAA78BED7CD3971F9FB69162D7602D8B3BE8`。没有将物理等待补偿为逻辑 NOP。基线 max_cycle=79,841；Raw 显式增加两个 62 周期排空窗口，max_cycle=79,965。两个最终表项分别为 `(C=29613,event=196608,members=0,1)`、`(C=62149,event=196610,members=2..6)`。对照 harness 严格归一化这些命名窗口后检查分页消费/释放周期，没有允许任意相位变化。

日志：`decoder-hidden-final-fast.log`（51,200），`decoder-hidden-final-slow.log`（25,600）；镜像 `decoder-final-linked.ftlpu`、`decoder-slow-linked.ftlpu`。L/P 是主执行段相对计数。INIT/bootstrap、i-MEM DDR 装载与真实前端 PRIME 另计；`CModelRuntime::instruction_prefill_cycles()` 单独提供预热统计，表中 P 不冒充包含全部启动/输入/输出搬运的总时延。

复现：构建 `ftlpu_compile`、`compiled_smollm2_common_barrier_decoder_test`，用 seq32 输入分别编译 baseline 和 `--icu-compression raw-macro`，共同使用 `--input-stage stablehlo --weight-bank 1 --kv-cache-capacity 256 --mxm-execution vector --projection-rope-overlap off --ffn-schedule tail` 及 CMODEL main 的 `config/ftlpu-lpu32.json`。测试程序参数为 `baseline.ftlpu raw.ftlpu`。第二档仅设置 `FTLPU_DDR_BANDWIDTH_MBPS=25600`；不设置跳过数值校验的环境变量。

## 存储开销

下表为实际硬件 i-MEM/有效控制项及完整容器，未把元数据当作计算码流，也未计入外部权重数据本身。

| 项目 | 小型真实分页 | 完整 seq32 linked decoder |
| --- | ---: | ---: |
| 计算 i-MEM 字节 | 96 | 102,976 |
| Native 搬运控制字节（含 C2C、MEM） | 1,200 | 9,144 |
| i-MEM 合计 | 1,296 | 112,120 |
| 有效 Barrier RAM 字节 | 8 | 16 |
| i-MEM + 有效 Barrier RAM | 1,304 | 112,136 |
| BARR 文件块（含头） | 24 | 32 |
| 事件关联文件字节 | 12 | 44 |
| page-use launch 元数据字节 | 4 | 28 |
| `.ftlpu` 文件总字节 | 2,689 | 166,296 |

目标 provisioned Barrier RAM 为 8,192 字节，当前/下一项缓存另需 16 字节；上表有效项字节不等于全容量配置成本。事件成员、ready 状态、generation 和 transport 控制存储另外存在，CModel 的 STL 宿主内存开销不冒充 RTL 实现位数。

原生基线计算 i-MEM 257,480 字节，Raw 计算 i-MEM 102,976 字节，减少约 60.0%；基线文件 394,431 字节，最终完整 linked Raw 文件 166,296 字节，减少约 57.8%。这些是本程序实测，不能外推所有模型。使用 inspector 的 `binary common_barrier_storage` 分栏统计；旧 aggregate i-MEM 行不包含独立 transport 区，不能单独充当协议总量。

## 其他发现和仍有的边界

真实验证先发现基线 Key RoPE 临时区覆盖残差输入：Key-local 产品沿用 base=0，在 bank0/slices4..7 覆盖仍需保留的 activation。已把 Key 产品移到输入和 Query IW 之后，保持 allocator 和容量检查。Smol32 实际区间从 `[0,192)` 改为 `[396,588)`；修复后 attention + RMS2 通过。CMODEL main `out/residual_placement_diagnosis.txt` 留有原生指令及地址证据。

同时修正了 harness 的 hidden SRAM 诊断：原先从 RMS2 推导的 distributed16 地址与 FFN 实际四 slice、交替 BF16 pair 的布局不符。编译器现导出 `ffn.hidden` 内部 binding 的真实 pair-planar 布局，harness 同时核对物理 SRAM 和通用 `download_binding`，并将 hidden 数值检查升级为门禁。49,152 个 hidden 值对硬件参考的误差为 0，东西副本误差为 0；对数学参考最大误差为 0.03125。接口测试仍严格要求四输入一输出，内部元数据不计入外部接口数量。

SmolLM2 seq128 完整 decoder 现已通过默认 FU3D 和 Raw Macro 编译。修复包括激活 staging 重复计入 token-block、QKV/PV/FFN 超容量累加器窗口的串行复用、Value 紧凑 SRAM 步长，以及 RoPE/残差/FFN 在单 Context 队列上的交错循环域。没有扩充硬件容量或放宽资源校验。新增 `smollm2_seq128_compile_test` 验证两条完整编译路径、最终二进制读取、i-MEM 容量及 MEM/MXM 不回退；与 attention staging、FFN planner、elementwise 三项测试共 4/4 通过，用时 48.40 秒，日志 `seq128-regression-tests.log`。默认产物为 1,698,875 字节，Raw 为 401,263 字节；seq32 Raw 产物与修复前逐字节相同（SHA256 `F4D7CD2F99F6C63FE882A053A227E36CE2737FECDE94F666F6E9A458BC377E99`）。

seq128 尚未完成真实分页执行与端到端数值比较，不能以编译回归代替它。额外的 `--verify-icu-issues` 静态比较会因运行时事件控制报 `logical inspector cannot map runtime-dependent ICU control`；未删除搬运队列或控制来绕过该限制。Qwen seq32 候选编译通过；完整 Qwen 数值路径没有在本轮修复后重跑，不能宣称所有 decoder/长度已验证。既有 `full_chip_topology_test` 因旧 VXM API 无法构建，本轮没有扩大范围修复它。

首版不支持任意流水线透明抢占、运行期控制表流式 refill、计算/分页重叠、跨 invocation lookahead 或所有别名写域。页面目标队列的生存期证明当前支持仿射 Macro 和简单 Native MEM；复杂 Native MEM 3D/间接 gather/scatter 等无法证明的组合明确拒绝。同一 MEM 队列不能混用绝对时间 Macro 和相对 PC/NOP Native；损坏程序的混合 Read/NOP 已有最终装载拒绝回归。其他 Native SXM/VXM 队列仍正常参与统一暂停。新增安全点或地址形式必须补充可验证的证明，不能删除同步、跳过资源检查或增加理想化前端能力。
