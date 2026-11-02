# DMA 完整程序装载与安全切换

`CModelRuntime::load_timed()` 接收 runtime 已解析的完整程序，生成实际物理字节并预置 DDR，然后返回。物理时间由后续 tick 推进；装载时间不是额外叠加的估算值。

## 存储与顺序

装载顺序固定为程序清单中的计算/C2C i-MEM、独立 MEM transport i-MEM、QCTL 或 BARR、事件配置。每个存储完全提交后再读取下一个存储。Native 使用现有硬件编码，包括功能字、控制字和多 word packet；C2C DMA packet 占两个 96-bit word。Macro 使用现有 packed-v1 编码。

QCTL 执行 RAM 每条记录只有三个小端 u32：`cycle/opcode/operand`，共 96 bit。Native 绑定、`release_group`、`phase_offset` 等继续由 runtime 链接，不计入控制 RAM。BARR 每条记录为小端 `stop_cycle/event_id`，共 64 bit。事件绑定沿用二进制文件的现有小端记录，以 32-bit word 提交到有限配置存储。事件控制器使用已提交的字节初始化。任务 generation 必须递增；从新任务受理开始，旧 generation 通知即被隔离，包括新控制存储尚未提交的阶段。

空控制分区设置零有效长度，不产生 DDR 请求。普通切换清除上一任务的指令和控制存储有效范围，保留 SRAM、DDR/DMA 和全局物理时间。超容量定时装载提前拒绝，不能通过直接写入式分段重装绕过 DMA。

## 启动与完成

全部存储提交之后，各 Native IQ、Macro 前端和 QCTL 缓存按有限端口和读延迟并行预取。装载和预取期间，新任务的计算、C2C 和 transport 指令均不执行；共享 DDR 和外部独立数据 DMA 继续推进。启动还受必要输入和 runtime 首批权重依赖门控。

- `busy()` 只表示装载状态，不能证明 kernel 正在执行或已结束。
- `kernel_done()` 在队列、控制状态、功能流水、SR、C2C、DMA FIFO 和 DDR 请求及未消费响应全部排空的物理边沿锁存。
- `can_load_program()` 在完成锁存之外重新检查当前资源是否排空；完成之后新增的独立 DMA 流量仍会阻止覆盖。
- `run_until_complete(max_physical_cycles)` 返回是否完成；返回 false 时保留原任务，可检查状态或继续等待。
- `run_cycles(N)` 仍按逻辑执行预算推进，不能作为完成证明。

runtime 和底层 `submit()` 均在修改任务、统计或 DDR 映像前拒绝不安全的覆盖。直接装载保留便利预取行为，并遵守同一保护。

## 统计与验证

装载报告包含受理、首笔 DDR 请求/返回、各存储最后提交、全程序提交、前端就绪、启动和 kernel 完成时间戳。`statistics().stores` 按 `ProgramStorage` 分别记录有效字节、实际 DMA 字节、物理写入 word 数及最后提交周期。

最小验证入口为 CModel 的 `program_loader_test` 和软件 runtime 的 `program_load_runtime_test`，不运行完整大模型回归。

本阶段没有增加 CALLP、返回栈、专家选择、DDR 描述符链、双份 i-MEM 或抢占，也没有改变 QCTL/BARR 同步语义。
