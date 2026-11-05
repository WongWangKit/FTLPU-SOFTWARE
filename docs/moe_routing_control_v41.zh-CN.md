# MoE 设备端路由控制单元：v41 实现

本轮实现设备端 `SELECT → PLAN → ADDR/Gather`。运行中的专家选择、计数、稳定分组、地址生成和激活搬运均在 CModel 设备时钟内进行；runtime 不读取结果后修补指令。MXM/VXM 仍负责投影、偏置、非线性和归一化。

## 接口与容量

共享语义位于 CModel `include/ftlpu/routing/instruction.hpp`，逐周期实现位于同目录 `control_unit.hpp`。软件入口为 `runtime/include/ftlpu/software/runtime/routing_program.hpp`。

| 资源 | 固定上限 |
| --- | --- |
| 全片 RCU / 活动作业 | 1 / 1 |
| E / K / T | 1024 / 16 / 128 |
| 输入块行数 / 地址 FIFO | 32 / 8 |
| Native Routing iMEM / IQ | 256 个 96-bit 字 / 4 项 |
| 描述符 | 32 项，序列化存储上限 4096 个 32-bit 字 |
| 在途搬运 | 一个字节平面向量，完整向量缓冲 32 B |

内部计划存储固定属于当前 context：selected ID、原始分数、count、offset、cursor、active list 和 assignment。按上限计算，32-bit 元数据约 56 KiB，另有 K 个候选寄存器、地址 FIFO、描述符 RAM 和控制状态。C++ vector/deque 只是有限存储的容器；输入容量在作业开始前检查。调试 trace 保留最近 256 条命令，不属于硬件数据通路。

`context_id` 是单活动上下文的身份检查，不表示并发上下文。新 SELECT 替换旧计划。超过上限由调用方分块；不隐式丢 token。

## ISA

追加 `QueueKind::Routing = 10`、`InstructionKind::Routing = 8`，已有编号保持不变。Routing 只支持 Native 模式，不使用 MEM/MXM macro 编码。

| 96-bit 字段 | 语义 |
| --- | --- |
| word0[1:0] | 标准 Instruction 信封，`00` |
| word0[3:2] | 1=`ROUTE_SELECT`，2=`ROUTE_PLAN`，3=`ROUTE_ADDR` |
| word0[31:4] | 保留，必须为零 |
| word1 | 描述符编号，0～31，其余高位必须为零 |
| word2 | 完成实例，1～0x7fffffff；最高位保留给完成来源命名空间 |

同一程序中的完成实例唯一。每个描述符有显式子操作码，必须与引用指令一致。NOP、WAIT_EVENT、NOTIFY 等继续使用 ICU 标准控制信封。

描述符按小端 u32 显式序列化，以 `RCU1` 标识和项数开头，不序列化 C++ padding。TensorView 固定 21 个 u32：元素类型、rows、columns、token_stride、feature_stride，以及四个 `{hemisphere,slice,bank,base_row}` 字节平面记录。FP16/BF16 只使用前两平面。

## SRAM 视图

逻辑元素 `(token, feature)` 的字节平面 p：

```text
row = plane[p].base_row + token * token_stride
      + floor(feature / 32) * feature_stride
lane = feature % 32
```

地址中的 hemisphere、slice、bank、字节平面和 lane 都参与物理访问。`token_stride=1, feature_stride=T` 对应按特征块分区的 byte-planar 视图。每个字节平面使用不同物理 bank（同 slice 的不同 bank 也可）；重叠的 token/feature 行映射、越界、未知格式均拒绝。

SELECT 输入 F32，资格掩码为同形状 U32，非零表示有资格。`raw_scores` 缺省或等于 `choice_scores` 时复用选择分数。selected ID / selected score 可选择导出到 SRAM；导出采用真实控制读改写，保留同一物理段中未写的 lane。

PLAN 的 `weights` 缺省表示原始选中分数已是最终权重；提供时必须为 `[T,K]` F32。该阶段只传递位模式，不隐式归一化，也不转换到 FP16。

ADDR 只接受 F16/BF16 整向量 byte-planar 输入，特征数是 32 的倍数。源/目标区间不得重叠。当前拷贝保留 byte-planar 布局；接入 MXM 所需的 SXM 转换由已有下游程序显式安排。

## 逐周期行为

Top-K 用一个 FP32 比较器和 K 个候选寄存器进行迭代比较插入；每个有效候选比较 K 次，每周期一次比较交换，不能解释为每周期接收一个专家。排序键是选择分数降序、专家 ID 升序。候选携带原始分数位模式。合法正负无穷正常排序；读到 NaN 或有效候选少于 K 时作业报错，禁止产生 ready。无资格专家跳过，不读取其分数。

PLAN 使用串行元数据访问：清 count → 第一次 token/slot 遍历计数并记录首次出现顺序 → count 前缀扫描 → 第二次遍历写 assignment。每个 assignment 的 token、slot、weight 分别写入，cursor 读写也计入时钟。满足 `sum(count)=offset[E]=T*K`，专家内部保持 token-major、slot-major 顺序。

ADDR 从设备内部 active list 读取专家 ID，再读取 count、offset 和 assignment token，生成完整的源/目标行地址。`active_index` 必须有效；`row_offset >= count[e]` 返回空块。只搬有效行，目标尾部保持原内容。RCU 不自动遍历下一专家或下一块。

## 端口交接与真实 Gather

RCU 收集命令所需物理 bank，在全部 bank 的既有 MEM 流水排空后统一取得所有权，避免部分占有。控制适配器每 hemisphere 只有一个有限请求/返回槽；一次访问一个 8 B 物理 tile 段，延迟由 `routing.control_latency_cycles` 指定。它使用原 bank 端口，不增加第二个 SRAM 读口。

持有期间，关联 Native/transport MEM 队列继续有限预取，功能命令不消费；其他 bank 的队列继续运行。调用方仍需通过设备 WAIT/NOTIFY 或静态已排空阶段保证分数/权重生产先于 SELECT/PLAN，不能用端口互斥替代数据依赖。

地址 FIFO 和实际 MEM Gather 一一对齐，只有获得 SR 完整 burst 信用后才消耗下一地址。真实路径为：

```text
assignment → 8 项地址 FIFO → MEM.Gather → 源 SR
           → 32 B 有限 SR bridge → 目标 SR → 普通 MEM.Write → SRAM commit
```

这条基线包含一个新增的 store-and-forward SR bridge，支持跨 hemisphere，按现有 8 B tile 段逐周期收集和发送；它不是免费跨片连线，也不是 CPU 数组拷贝。一个完整向量提交后才处理下一地址，属于面积优先实现。收集、发送、bank 等待、SR 背压和写入提交都有时序。

Gather 使用涉及边界的 East stream 31，程序必须为此保留该流的整个搬运 burst。目标所有 tile 段有空间后才开始发送，再按普通 MEM Write 的连续节拍提交；发送期间若其他生产者非法占用已保留的 burst，明确报错。当前不支持每 lane 任意地址、Scatter 或 Gather 内隐式转置。

## 事件、排空和装载

完成来源追加 `QueueEventSource::Routing = 3`，复用 QCTL 事件实例列表。内部以实例最高位区分 Routing 与 PageCommit，不改变原 PageCommit 通知行为。

| 完成实例对应操作 | 语义 |
| --- | --- |
| SELECT_READY | 选中结果和可选 SRAM 导出均已提交 |
| PLAN_READY | count、offset、assignment、active list 均可读 |
| INPUT_READY | 最后一个目标 SRAM tile 已真实提交，地址 FIFO 和搬运流水已空 |

事件是当前 generation 的完成锁存，可供多个队列 WAIT。重复通知幂等，旧 generation 忽略，未来 generation 报错。RCU 命令、描述符读取、控制访存、地址 FIFO、真实 MEM 流水均参与程序排空、kernel done 和程序替换判断。异常作业保持未完成，需显式设备复位。

二进制版本升级为 v41。在 QCTL/QLNK 后、队列数据前增加 `{routing_isa_version, word_count, descriptor_words}`；非 Routing 程序能力为 0，旧目标 ABI 的计算结果保持不变。v40 及更旧读取路径不读取该段，Routing 能力默认为 0。

直接装载与定时装载共享 ProgramImage 准备过程。定时路径把 Routing iMEM、描述符和事件配置通过原 DDR/DMA ProgramLoader 写入；全部映像 seal/commit 后才允许取指。描述符执行读取按一个 u32/周期建模，统计独立于 DMA 装载周期。

## Builder 示例

```cpp
BinaryProgram p;
RoutingProgramBuilder route(p);
route.select(select_descriptor, 1);
route.plan(plan_descriptor, 2);
route.address(address_descriptor, 3); // active_index 为设备表序号
route.completion_event(77, {3});
route.wait_for(QueueKind::Mem, destination_mem_queue, 77);
// 继续追加消费该输入块的设备命令。
// runtime.load(p) 或 runtime.load_timed(p)，运行期间无需读取路由结果。
```

设备默认关闭 Routing。可在运行前设置 `SystemHardwareConfiguration::routing.enabled=true`，或使用 CModel `config/ftlpu-lpu32-routing.json` 验证目标。Builder 更新程序的 Routing 能力与 ABI，不会修改原有目标配置。

统计由 `system.chip().routing().statistics()` 提供，包含执行周期、描述符读取等待、比较次数、元数据访问、控制 SRAM 请求、bank 等待、地址 FIFO 信用等待、SR 等待、地址 FIFO 峰值及真实写入段数。`trace()` 提供最近命令的开始/提交时钟；ICU 日志和软件 schedule/execution trace 也包含 RCU 命令。trace 时钟从设备程序开始、允许 RCU 执行时累计，DMA 装载周期另见 ProgramLoader 报告。

本轮不包含 RTL、专家权重 DMA 参数化、专家 macro 动态调度、输出加权回并、完整模型 lowering 或全链路 FP32 改造。验收通过只代表路由控制和专家输入块组织能力完成。
