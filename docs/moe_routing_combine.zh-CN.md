# 第四部分：设备端专家输出加权回并

目标是把 assignment／专家分组顺序的 `[T*K,H]` 专家输出，按 token 和 Top-K slot 恢复为 `[T,H]`。每个 token、每个 32 特征块按 slot 从 0 到 K-1 执行 `sum += final_weight * expert_output`。权重来自 PLAN 已保存的最终 FP32 权重，不重新归一化。

专家输出及最终 SRAM 结果采用相同的 BF16 或 FP16 byte-planar 格式。乘法和累加使用现有 VXM FP32 运算，只有最后一次输出才转换为 BF16／FP16。第三部分验收模板同步改为 BF16 输出。没有增加独立浮点乘加器，也没有通过 host 循环计算激活结果。

## 命令与配置

新增可执行 `ROUTE_COMBINE`，opcode=7；5、6 仍是不可执行的 WeightTable／ExpertTemplate 配置记录。Routing 能力升级到 v5，二进制容器仍是 v41。旧能力不能接收包含 Combine 的描述符；旧命令和已有队列编号不变。

96 位命令仍为三个 u32：

| 字 | 内容 |
|---|---|
| 0 | 原指令信封低两位为 00，opcode=7，编码值 0x1c；其他位必须为零 |
| 1 | 有限描述符存储内的编号，范围 0..31 |
| 2 | 完成实例编号，1..0x7fffffff；使用现有 Routing WAIT／NOTIFY 映射 |

`RouteCombineDescriptor` 包含 context_id、source 和 destination 两个 RoutingTensorView。每个视图为 21 个 u32；含类型字段的描述符占 44 个 u32，与已有 Routing 描述符共享 4096 个 u32 容量。builder 为 `RoutingProgramBuilder::combine(descriptor, completion)`。复用直接装载与定时 DMA 装载，全部镜像提交前不能执行。

源必须对应 ExpertTemplate 的完整输出，行数为 T*K；目标行数为 T，两者列数相同且是 32 的倍数。两者元素类型相同，只支持 BF16／FP16；源与目标使用不同物理 bank。布局坐标通过视图计算，支持 token／feature stride、bank、slice 和 hemisphere，不假定相邻逻辑行物理连续。

## 反向索引与就绪约束

PLAN 写 assignment 时增加一个串行阶段，写 `inverse[token*K+slot]=assignment_row`。上限 128*16 个 u32，8 KiB 有限元数据 RAM；每次写入占一个周期，并计入 metadata_accesses。Combine 通过已有单请求／单响应元数据端口读取 T、K、inverse 和 assignment.weight_bits，没有无周期扫描。

只有内置 ExpertMacroExecutor 的全部专家任务完成、真实输出写入并排空，设备才登记 `{generation,plan_epoch,context,output_view}` 就绪记录。Combine 必须匹配这条记录，并 pin 同一个计划；SELECT／PLAN／新的 DISPATCH 会使旧记录失效。回并结束消费该记录，不能重复消费或切换到旧计划。

## 数据通路与状态机

```text
读取 T/K → 等待排空并取得 bank/VXM 控制权
 → 读 inverse[token,slot] 和 FP32 weight
 → 两个字节平面分别通过真实 MEM Gather 读取
 → E31 collector 收齐完整 32 元素向量
 → SR bridge 向 VXM 固定输入组提供 BF16/FP16 数据
 → VXM Multiply(weight) → VXM Add(本地 accumulator)
 → 非末 slot：保留 FP32 accumulator，继续下一个贡献
 → 末 slot：转换成 BF16/FP16，经 SR collector 收集
 → SR bridge / 普通 MEM Write 写目标行
 → 等待实际提交 → 下一特征块／token
 → 全部排空 → 释放计划及资源 → 完成事件
```

第一项通过 accumulator_reset 初始化各 lane 的累加寄存器，无需预先清零整个目标 SRAM。新 token 或特征块重新初始化；同一个块的 K 个贡献按 slot 顺序串行累加，使用分开的乘法和加法，不采用 FMA，也不将中间部分和反复写成 BF16。没有新增并行 Scatter、原子加法或完整重排激活缓冲。

RoutingCombineController 仅控制地址、权重和状态。算术仍实际经过共享 VXM 的 C0 Multiply、C1 Add，配置中的 FP32 immediate 位来自设备元数据。控制器在已排空边界取得 VXM 命令端口所有权，通过新增的小型命令来源选择／owner 检查复用原 issue_south 入口；不由 CPU 改 iMEM，也不增加第三部分 macro 的编码字段。每次仅一组配置在途，配置解码、ALU 延迟和 tile 传播仍由现有 VXM 建模。

SR bridge 的有效数据缓冲为两个字节平面共 64 字节，输入与最终输出分时复用。复用 VXM 固定输入组 0/8、输出块 0/4 的现有 hemisphere mux，执行前保存、执行后恢复；镜像通路按现有 VXM 结构供数。结果直接从现有 VXM SR 输出收集，写回使用原 E31 bridge 和普通 MEM Write。

MEM Gather 使用已有有限地址 FIFO，本控制器最多保持一个 Gather 向量在途；读取向量检查 transfer tag。写回对每个字节平面等待所有 tile 真正提交并检查 commit 数，再释放对应完成记录。bank 在已排空边界交接，源/目标共享原 SRAM 端口，不增加第二读端口；本版逐平面、逐贡献执行。VXM 使用期间不允许其他生产者竞争其命令端口；普通生产者冲突会报错，不隐式暂停或重排其静态程序。

## 完成、统计和范围

全部目标向量的普通 MEM Write 提交，且 MEM/VXM/SR 等瞬态资源排空后才发布 ROUTE_COMBINE 完成，可作为 COMBINE_READY 事件。busy、计划租约和待完成通知纳入 kernel done、idle、程序切换判定。保留 cycles、bank_wait、stream_wait、contributions、gathered_vectors、committed_segments 统计。

精简验证包括：原 RCU 用例内的 BF16/FP16、K=2、H=64 跨特征块和跨 hemisphere 回并；bank 等待、旧 generation、缺少专家输出就绪、重复消费、布局重叠和尾部哨兵检查。软件用原直接装载单行、定时装载 32/5/5 行链路追加回并，并增加小规模 K=2 的真实双专家模板链。

首版保持串行执行并使用现有全局瞬态排空检查，吞吐以功能验证为目标。完整 Gate/Up/SiLU/Down FFN lowering、共享专家分支、任意混合精度布局及全模型适配不在本轮范围。

## 验证记录

2026-11-04，Release、ftlpu-lpu32 目标：

| 用例 | 物理周期 | 结果 |
|---|---:|---|
| CModel BF16 / FP16，T=2、K=2、H=64，含 bank 等待 | 各 261（仅 Combine） | 最终 16 位位模式逐项精确一致 |
| 完整单行 K=1，直接装载 | 3061 | 专家 BF16 输出及最终回并逐项精确一致 |
| 完整 T=42、K=1，专家任务 32/5/5，定时装载 | 13553 | 精确一致，恢复原 token 顺序 |
| 完整 T=3、K=2，专家任务 2/1/2/1，定时装载 | 9035 | 使用 FP32 0.1/0.9 权重，最终 BF16 精确一致 |

完整链路周期包含路由、程序装载（定时用例）、权重 DMA、专家模板与回并，不代表完整 FFN 或模型性能。参考按实际专家 BF16 输出、FP32 分开的乘法/加法、最后一次格式转换计算；不拿未量化的高精度专家输出替代参考输入。尾部哨兵不变，回并期间 WAIT 不提前放行且不能切换程序。

CModel 的 Routing、MEM array、VXM slice、macro schedule、raw macro decoder、ProgramLoader、BF16 MXM 共 7 项相关回归通过。
软件侧 routing_runtime_test、binary_queue_control_test、macro_bitstream_test、program_load_runtime_test 共 4 项通过。
