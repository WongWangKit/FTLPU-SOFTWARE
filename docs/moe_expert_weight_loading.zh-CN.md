# 设备端专家权重间接加载

第二部分在 ExpertTaskController 后接单个 ExpertWeightLoader。任务的专家编号由 RCU 产生；加载器在设备时钟内查表、提交真实 DDR DMA/C2C RX/MEM Write，全部权重写入后发布带完整任务身份的 WEIGHT_READY。CPU 只配置、装载和初始化数据，不逐任务选择地址或确认完成。

## 配置和编码

不增加可执行 ISA 指令。Routing 描述符增加 WeightTable 配置记录（类型 5，不能编码成指令），Routing 能力为 v3，容器继续为 v41。SELECT/PLAN/ADDR/DISPATCH 的编码不变；v1/v2 程序没有权重表时保持原行为。

每个程序只有一张表、一个工作集。配置记录包含 workset_id、experts、segments、word_count 和小端 u32 数据。每专家两字：first_segment、segment_count；每段八字：DDR 地址低/高 32 位、vector_count、hemisphere、slice、bank、base_row、fabric_stream。搬运单位为原生 32 B 物理向量，DDR 和 SRAM 都连续步进；不隐式转换权重布局。

全局最多 1024 个专家、256 个物理段；与其余 Routing 描述符共同受现有 4096 个 u32 容量约束。每个专家必须引用非空合法段集合，同一专家的目标段不能重叠；不同专家允许复用同一缓冲。

表位于既有有限 Routing 描述符 RAM，而非数据 SRAM。通过现有 ProgramLoader 描述符映像提交，不增加数据 SRAM 读端口或新的装载通路。运行时只有一个表读请求和一个返回槽，逐周期读取一个 u32；条目和段扫描均计入 table_reads。不会一次性遍历主机描述符数组来选择和搬运专家。

```cpp
RoutingProgramBuilder b(program);
// 配置 SELECT、PLAN、地址模板及 DISPATCH。
b.weight_table(workset_id, entries, segments); // 配置记录，不发射指令
```

## 执行与同步

```text
INPUT_READY
 → 查专家表项
 → 扫描并保留该专家所需 bank
 → 逐段读取、配置 RX、提交 DMA
 → RX 通知逐向量触发普通 MEM Write
 → 最后一段最后一个 SRAM tile 提交、流水排空
 → WEIGHT_READY
 → 消费者完成后 TASK_DONE
```

每段复用真实 C2cDmaEngine 和 C2cRxSlice。内部接收适配器最多允许一个向量在途，使用 RX 通知、入口寄存器和 West SR 传播距离决定 MEM Write 时刻；写入四个 tile 并排空后才发下一个接收信用。MEM 提交计数和向量标签检查防止提前通知或错误数据消费。

WEIGHT_READY 是设备消费者端口的任务级通知，不是可被下一块重复使用的公共 QCTL 锁存事件，也不表示 TASK_DONE。只有消费者明确完成后，任务控制器才推进。

单驻留标签包含程序 generation、plan_epoch、专家身份，以及当前不可变工作集配置。同专家后续块复用权重，不再次 DMA。搬运期间权重 bank 由加载器独占；就绪后切换为只读驻留保护，允许普通 MEM Read 供专家计算消费，拒绝其他写入和 RCU 覆盖。保护保留到该专家最后一个块的消费者完成、读流水排空，随后释放并使标签无效；切换专家重新加载。加载状态、bank 使用权、DMA/RX 和 MEM 流水都参与程序排空判断。

## 首版约束

- 单任务、单驻留权重工作集；一次工作集必须放得下 SRAM，不生成自动权重 tile 遍历。
- 权重 bank 必须与激活源和输入缓冲的 bank 分离，避免跨块保留权重时与 Gather 互相等待。
- 使用 lane 0 和配置的 West SR（默认 W30），程序须保留对应通路；普通 C2C TX/RX/DMA 队列不能与本功能混用。独立 MEM bank 仍可运行。
- 不预取、不多专家缓存、不新增量化/解包/转置。段表描述的是既有物理权重布局。
- 测试消费者等待输入和权重均准备完成后模拟后续完成，尚未接入真实专家 macro。

地址溢出、向量数为零、段表越界、同专家目标重叠、与激活 bank 冲突、能力不匹配均拒绝。内部 DMA 完成只由适配器消费，不向普通 ICU 泄漏完成令牌。

## 精简验证

新增三组权重用例，共享已有 [3,3,1] 专家任务和工具：驻留复用、真实两段 DDR/RX/MEM 搬运及最后提交、旧任务身份/非法表/忙时程序切换。预期三次 WEIGHT_READY、四次段 DMA、六个物理向量写入、一次驻留命中。只运行受影响的定向回归。

2026-11-04，Windows / MSVC / Release 验证通过：routing_runtime_test（包含新增三组、第一部分三组和原有直接/定时装载用例）、subsystem.routing.control、unit.mem.array、subsystem.c2c.dma_ddr4、binary_queue_control_test、program_load_runtime_test。新增权重用例使用直接装载，最终数据逐字节一致；均记录四次段加载、六个已提交向量、一次驻留命中、68 次描述符字读取。第二组还通过真实 MEM Read 验证就绪后的读访问，并确认覆盖写入被拒绝。未重跑全部算子数值回归，未接入实际专家计算。
