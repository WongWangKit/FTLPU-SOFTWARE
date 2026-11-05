# MoE RCU v41 验证记录

环境：Windows / MSVC 19.44 / Release；CModel 基线 `06405e4`，软件基线 `8e33bb2`。硬件默认目标 `ftlpu-lpu32` 保持 Routing 关闭，测试显式启用。另配置并验证 `ftlpu-lpu32-routing` 目标。

## 新增测试

`FTLPU-CMODEL/tests/subsystem/routing/control_unit_test.cpp`：

- K=1、K=E=16、E=1024、T=128；同分按专家 ID 排序、合法无穷、NaN 和资格不足拒绝。
- choice/raw 分离、U32 资格掩码、选择结果 SRAM 导出、最终权重逐位传递。
- count、offset、active 首次出现顺序、专家内部稳定 token/slot 顺序逐项比对独立参考。
- BF16/FP16，64 特征跨字节平面、跨 bank/hemisphere，同 bank 的源/目标分时访问。
- 块首、块中、尾块、越界空块；有效行逐字节比较，尾部哨兵保持不变。
- 等待既有 bank 流水排空后统一交接；地址 FIFO 达到 8 项，源/目标 SR 背压，最后物理写入后完成；忙时复位请求拒绝。

`FTLPU-SOFTWARE/runtime/tests/runtime/routing_runtime_test.cpp`：

- builder → v41 四种读取入口 → Native Routing → 直接/定时 DMA 装载 → 真实 Gather。
- 定时装载未全部提交前不比较分数；描述符映像必须真实提交。
- 完成实例释放 MEM WAIT；旧 generation 无法释放新程序，重复通知幂等，未来 generation 拒绝。
- RCU 或 Gather 未完成时不能切换程序；独立 MEM 队列在 RCU 忙时仍能发射。
- VXM 原生 FP32 Bypass 输出 0.75，经普通 MEM 写入分数 SRAM，RCU 选择并 Gather；CPU 只预先配置和初始化激活，不在运行中修补路由。
- 与现有 `plan_qwen3_moe_routes` 的选择 ID、CSR、活跃顺序、assignment 和最终权重位模式逐项一致。CPU 参考仅用于测试准备与结果断言。
- 描述符 32 项、iMEM 256 字的边界往返，溢出及非法能力/描述符/实例/队列/格式拒绝。
- v40 无 Routing 能力读取、v41 损坏描述符在四个入口均拒绝。
- 原生 SYNC/NOTIFY 在 MEM 与 Routing 间双向传播，性能报告和 ICU 总发射统计纳入 RCU。
- schedule/execution CSV 包含 RCU 命令；保持期间不重复记为发射。

## 周期与资源实测

以下是固定合成用例的 CModel 物理周期，不是完整 MoE 层性能，也不是 RTL 面积/频率结果。

| 用例 | 周期 | 比较 | 元数据访问 | 控制 SRAM 请求 | bank 等待 | 地址信用等待 | SR 等待 | FIFO 峰值 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 独立 RCU，控制延迟 1 | 2506 | 96 | 270 | 816 | 0 | 0 | 0 | 7 |
| 同一用例，控制延迟 3 | 4138 | 96 | 270 | 816 | 0 | 0 | 0 | 7 |
| bank 排空 + 双端 SR 阻塞 | 387 | 4 | 61 | 16 | 4 | 130 | 70 | 8 |
| runtime 直接装载 | 720 | 24 | 96 | 144 | 0 | 0 | 0 | 3 |
| runtime 定时装载 | 1152 | 24 | 96 | 144 | 0 | 0 | 0 | 3 |

控制延迟从 1 改为 3，增加 `4138−2506=1632=816×2` 个周期，说明 816 次控制请求的延迟真实进入模型。runtime 两种模式均读取 180 个描述符字、提交 16 个目标 SRAM tile 段；阻塞用例提交 64 段。定时用例多出的周期包括指令、描述符和事件映像的原 DMA/ProgramLoader 装载。

给定 FP32 分数的选择与独立参考精确一致；最终权重逐位一致。本次 VXM Bypass 的 0.75 是精确可表示数，输出误差为零。未在该测试中测量投影量化、指数/归一化近似误差，不能据此宣称真实模型路由数值误差为零。

## 现有回归

通过的 CModel 回归覆盖 instruction codec、MEM array、QCTL、macro schedule、raw macro decoder、ProgramLoader、hardware configuration、MXM BF16、VXM ALU 和 16-ALU FP16；新增 Routing 测试通过。Routing 验证目标的配置/profile 和路由测试也通过。

通过的软件回归覆盖 `binary_queue_control_test`、`binary_barrier_test`、`program_load_runtime_test`、`macro_bitstream_test`、`qwen3_moe_router_test` 和 `imem_capacity_test`，以及新增的 Routing runtime 测试。

两项既有数值用例仍失败，并在从 **未修改 CModel HEAD `06405e4`** 导出的独立源码/build 中复现相同错误：

| 用例 | 修改后 / 原始基线的相同错误 |
| --- | --- |
| `kernel.projection.w8a16` | MXM accumulator address 1536 超过当前配置的 128-row SRAM |
| `kernel.rmsnorm` | VXM input Buffer 在 Bundle 活跃时被重新配置 |

因此本次不能报告“全部既有数值回归通过”。这两项不属于 RCU 改动引入的问题，未在本轮修改其算子或容量配置。

## 复现

在 CModel 根目录，构建并执行：

```powershell
cmake --build build-program-load --config Release --target routing_control_unit_test
ctest --test-dir build-program-load -C Release --output-on-failure -R '^subsystem.routing.control$'
```

在软件根目录：

```powershell
cmake --build build-program-load --config Release --target routing_runtime_test binary_queue_control_test binary_barrier_test program_load_runtime_test macro_bitstream_test qwen3_moe_router_test imem_capacity_test
ctest --test-dir build-program-load -C Release --output-on-failure -R '^(routing_runtime_test|binary_queue_control_test|binary_barrier_test|program_load_runtime_test|macro_bitstream_test|qwen3_moe_router_test|imem_capacity_test)$'
```

运行跟踪由 runtime 测试在构建目录的 `runtime/test-output` 中生成。完整边界与硬件增量见同目录 `moe_routing_control_v41.zh-CN.md`。
