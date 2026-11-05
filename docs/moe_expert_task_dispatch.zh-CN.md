# 专家任务生成与输入搬运

在现有 RCU 上增加单个 ExpertTaskController。CPU 一次下发 ROUTE_DISPATCH；设备通过有时序的元数据接口遍历 active list，将各专家的输入按最多 32 行分块，并调用真实 Gather。首版只有一个任务在途、一套输入缓冲和一个设备消费者端口。

## 指令与软件接口

Routing ISA 能力从 1 扩展到 2；二进制容器继续使用 v41 的 Routing 段。原 SELECT/PLAN/ADDR 编码及 v1 程序保持兼容。旧实现不接受能力 2。新增指令仍为 96 位：word0=0x10（子操作码 4 左移两位），word1=描述符编号，word2=整次 DISPATCH 的公开完成实例。

RouteTaskDispatchDescriptor 为三个 u32：context_id、address_template_descriptor_id、block_rows。序列化仍采用现有显式小端描述符记录。模板必须引用同 context 的 ADDR 描述符，块容量不能超过模板容量和 32 行上限。

```cpp
RoutingProgramBuilder b(program);
b.select(select_descriptor, 1);
b.plan(plan_descriptor, 2);
auto id = b.address_template(address_descriptor); // 注册模板，不执行
b.dispatch({1, id, 32}, 3);
b.completion_event(77, {3});
```

任务控制器在内部副本绑定 active_index、row_offset 和 max_rows，不改共享描述符或指令存储。后续公开 Routing 指令等待 DISPATCH 完成，内部地址请求不经过这个被阻塞的指令队列。

## 元数据与计划生命周期

active_count、active[index]、count[expert]、offset[expert] 通过一个请求槽和一个返回槽读取。响应不会在提交请求的调用内返回；设备时钟推进后产生，未取走前反压后续请求。读取与 RCU 原操作串行，共用元数据端口，每次读取计入统计。

PLAN 完成后发布 plan_epoch。DISPATCH 检查 context 和 program generation 并持有计划，全部任务完成后释放。持有期间不允许 SELECT/PLAN 覆盖、普通程序复位或程序替换。硬复位清除执行状态。

## 任务和消费者

ExpertTaskContext 包含 generation、plan_epoch、task_id、route_context_id、active_index、expert_id、expert_token_count、row_offset、valid_rows、assignment_base、more。任务编号在当前程序内递增。

消费者实现 ExpertTaskConsumer，运行前连接到 system.chip().expert_tasks()。这是设备模块接口；运行时不得用 CPU 读取路由结果和逐任务确认来代替。

1. accept_task：返回 false 时任务上下文保持不变；返回 true 后启动 Gather。
2. input_ready：最后一个目标 SRAM 段提交、Gather 排空后通知一次。
3. poll_done：消费者在完成使用后返回该任务完整 identity；此前不能推进任务或复用缓冲。

旧 generation、旧计划和已退休任务的完成被忽略；未来或阶段错误的完成报错。INPUT_READY 不等于 TASK_DONE。未连接消费者时启动失败。

全部活跃专家按首次出现顺序处理；同专家 row_offset 按 valid_rows 递增。每次内部 Gather 完成使用任务标签返回控制器，不产生可被下一块误用的公开锁存完成事件。公开 completion 只在整个 DISPATCH 结束时产生一次。

## 数据通路与边界

复用 MEM Gather、原 SRAM、E31、SR bridge 和普通 MEM Write，不增加激活复制通路。仍仅支持既有 FP16/BF16 byte-planar 整向量布局，不填充尾部，源目标禁止重叠，调用方保留 E31 burst。计划被持有不等于长期占有数据 SRAM bank；bank 所有权仅由实际 RCU 命令按既有机制取得和释放。

本轮消费者为测试专用设备端桩，验证延迟接收、延迟完成和旧标签。没有实现专家权重加载、macro 参数化执行或输出回并，不能据此运行完整 MoE。后续专家执行模块应接入同一消费者接口。

## 精简验证

在 routing_runtime_test 中新增三组用例，共享配置和工具：任务遍历/接收背压、真实 BF16 Gather/尾块/提交时机、旧任务标签/计划保护/非法能力。活跃专家 [3,1]、计数 [37,5]、块容量 32，产生 (3,0,32)、(3,32,5)、(1,0,5) 三个任务，复用同一目标缓冲。

仅运行受影响的现有 RCU、二进制/QCTL 和 ProgramLoader 回归，不新增布局组合或性能基准矩阵。

2026-11-04，Windows / MSVC / Release 验证通过：routing_runtime_test（包含新增三组及原有直接/定时装载用例）、subsystem.routing.control、binary_queue_control_test、program_load_runtime_test。未重跑全部算子数值回归。
