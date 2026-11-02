# 共享 QCTL 接入 main：2026-11-02

在 SOFTWARE `383ca6b`、CModel `63de099` 的 main 工作区，三方集成 `codex/qctl-shared-pool` 的实现。保留 main 已有的 ICU JSON 参数化、完整 DMA 程序装载及其他未提交修改。没有提交或推送；原共享池 worktree 未修改。集成前文件快照保存在 SOFTWARE `out/shared-pool-main/before/`。

## 当前配置与集成

- v40 容器，512 条全片共享池，每队列最多 16 条控制、4 条缓存；共享单读端口、延迟 2 拍、请求间隔 1 拍。Native 绑定不占控制池。
- 保留单端口轮转仲裁、有限响应信用和最终链接/装载的供给证明；不能证明及时补充的程序仍拒绝，没有隐藏空拍或退回独立 RAM。
- 保留 ICU JSON 中的 i-MEM、IQ、Context、解码参数；除切换到共享池及 4 条控制缓存外，其余默认值不变。
- 定时 DMA 的 QCTL 镜像在提交后配置共享池分区，不再调用独立控制 RAM 后端。任务 reset 初始化新池并清除旧句柄/响应；装载阶段不推进执行取数，所有镜像与事件配置提交后，由统一 PRIME 边沿推进共享端口。
- 普通 RUN 和程序 PRIME 分别推进共享池，防止一次物理拍重复推进。原生控制、transport 包退休、SRAM 提交事件及释放相位不改。
- 原有 DMA 控制镜像 staging/readback 用于装载与核对；执行前端只持共享池句柄及有限缓存，没有用 staging 数据绕过共享端口。

## 实际验证

CModel：`unit.icu.queue_control`、`subsystem.program.loader`、`subsystem.icu.configuration`，3/3 通过。

SOFTWARE：`binary_queue_control_test`、`binary_barrier_test`、`program_load_runtime_test`，3/3 通过。涵盖四种读取路径、旧容器、共享池容量与供给拒绝、控制逐拍对照，以及 DMA 装载/等待/恢复/任务切换。新增检查确认定时装载的控制进入共享池，PRIME 确实产生共享 RAM 读取；保留提交前无发射、控制镜像逐字节回读、空表和旧 generation 检查。

重新编译并运行 seq32 Native 与 Macro＋共享池，DDR 25,600 MB/s，完整 attention＋FFN 和真实分页执行，没有跳过数值检查：

| 指标 | Native | Macro＋共享池 |
|---|---:|---:|
| 主执行物理周期 P | 110,845 | 110,845 |
| PRIME | 0 | 296 |
| 初始页面准备 | 3,385 | 3,385 |
| 后置搬运尾部 | 34,692 | 34,692 |
| 计算∩DMA | 17,707 | 17,707 |
| 计算∩RX | 26,676 | 26,676 |
| 计算∩SRAM commit | 26,668 | 26,668 |
| BF16 比较数 | 18,432 | 18,432 |
| CPU 参考最大绝对误差 | 0.03125 | 0.03125 |
| FFN hidden 硬件 mismatch | 0 | 0 |

1,779 组轨迹、各 960,840 个资源活动拍逐拍一致；最终镜像的计算、搬运、操作数、relocation、事件与释放相位等价检查，以及两个最终文件独立加载均通过。共享池预热读取 294 条、运行期读取 56 条，缓存峰值 4、响应峰值 2。每次实际完成 7 个运行期 page use、8 次预取和 331,776 个 tile 提交。

与旧 worktree 的 110,978 拍属于不同 main 基点结果，不把该差异作为共享池带来的加速。本次只在同一 main 内比较 Native 与 Macro；包含全部上述阶段后，分别为 148,922 与 149,218 拍，差值为共享池 PRIME 的 296 拍。

## 边界及证据

本次未重跑 seq128、51,200 MB/s 或延迟扰动矩阵。seq128 的共享池验收仍是原 worktree 的历史证据。真实 seq32 decoder 使用当前直接装载路径；完整定时 DMA 装载与共享池的集成由专门运行时测试验证，本次没有将整个 decoder 改为定时程序装载测试。

记录载荷估算仍为 12.23～16.31 KiB（220 队列口径），不含目录/仲裁/有效位等，不是 RTL 综合结果。保守供给证明、全片 512 条上限和单队列 16 条上限继续适用。

本次小型日志、轨迹/镜像摘要及复现脚本位于 [evidence/queue-control-shared-pool-main](evidence/queue-control-shared-pool-main/)。真实 CSV、二进制及构建日志保存在 `out/shared-pool-main/`。原 worktree 的历史报告保留在 [queue_control_shared_pool.zh-CN.md](queue_control_shared_pool.zh-CN.md)。
