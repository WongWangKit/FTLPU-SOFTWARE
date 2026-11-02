# QCTL 合入当前 main 的验证记录

日期：2026-10-30。将 SOFTWARE 原基点 `7ba85e7`、CModel 原基点 `bc84188` 上的 v38/v39 未提交工作，三方合入 SOFTWARE `ea81307`、CModel `06f9ee7`。保留 main 已有的 Qwen3.5 链式 VXM、调度修复及定时 DMA 程序装载功能；没有整体合并实验分支。原 worktree 及其修改保留。

历史验收与性能矩阵见 [旧基点验证记录](queue_control_v39_validation.zh-CN.md)。其中的旧二进制与 CSV 对应旧 VXM ABI，不能直接作为新 main 的执行证据。本记录单列集成后的重新编译、执行和限制。

## 合入时处理的接口差异

- 链式 VXM 的在途状态由 `VxmSlice::idle()` 检查，包括输入捕获、队列和 ALU 流水；不再访问已移除的 SR VXM datapath。普通 SR 的计算/搬运来源检查保留。
- Native 分页的 Gate/Up 共同 readiness epoch、显式 `common_barrier_protocol` 选项、VXM 16 ALU / 8 队列约束保持。默认 Raw Macro 使用 QCTL；显式旧协议请求仍进入旧路径。
- 定时 DMA 程序装载目前只有 packed MEM/MXM i-MEM 编码，没有 QCTL RAM 的装载/提交接口。因此 `load_timed(QCTL)` 在修改设备状态前显式拒绝。QCTL 通过 `load()` / `load_direct()` 装载并完成有限 PRIME；这是主 decoder 验证使用的路径。
- 内核切换清除上一任务的 QCTL 激活门控、计算 Native PC 绑定及完成事件状态，保留独立搬运队列。新增回归检查搬运状态保留和旧 generation 隔离。
- v38 `CommonOverlapCalendar` 的旧资源证明依赖显式 VXM SR 操作数。新链式 VXM 的捕获输入及反馈生存期不能用该证明推断，故对此组合显式拒绝；v39 QCTL 使用 Native 相位计划，不经过该旧证明。旧容器读取、旧协议 MEM/MXM 定向执行继续验证，但不声称旧 VXM ABI 可在新拓扑上直接执行。

## 本次验证

详细日志、最终文件和真实流水 CSV 保存在 `out/main-qctl-integration`；小型结果与复现脚本归档在 [evidence/queue-control-main-integration](evidence/queue-control-main-integration/)。CModel 对应提交 `c0988ab`。

- 两个工程构建通过；CModel 5 项定向回归、SOFTWARE 9 项定向回归通过。内核切换修复后重跑队列控制和程序装载 2 项，通过；链式 VXM 旧证明拒绝用例的独立复验通过。
- seq32/seq128 × Native/Raw Macro 四个程序在新 main 上重新编译。完整 attention＋FFN、实际 DMA/RX/MEM 写入和 SRAM 提交均执行通过；每次完成 7 个运行期 page use、8 次预取、331,776 次 tile 提交。没有跳过数值比较。
- 本次在 25,600 MB/s 下重跑两组 Native/Macro；计算与搬运逐拍轨迹均一致，最终执行镜像等价性检查及四个最终文件独立加载均通过。不是仅比较总周期。

| 序列 | Native 主执行 P | Macro 主执行 P | Macro PRIME | 计算∩DMA | 计算∩RX | 计算∩SRAM commit | 相同轨迹组数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| seq32 | 110,978 | 110,978 | 80 | 17,733 | 26,905 | 26,872 | 1,794 |
| seq128 | 417,796 | 417,796 | 92 | 24,169 | 35,858 | 35,826 | 1,784 |

主执行 P 不含 PRIME、初始页面准备及后置搬运；Native PRIME 统计为 0，初始页面准备两种编码均为 3,375 拍。seq32 比较 18,432 个 BF16 值，seq128 比较 73,728 个；本次 CPU 参考最大绝对误差均为 0.03125，FFN hidden 的硬件 Native mismatch 和 replica mismatch 均为 0。

两档带宽及延迟扰动的六组完整矩阵仍是旧基点的历史验收，本次未重复该矩阵。新 main 更换了 VXM 和计算调度，因此本表与历史表不能按同调度比较加速比；本次等价结论限定于上述新 main 的两个完整程序。

## 交付范围

本次提交包含公共屏障、提交反馈、计算/分页重叠、QCTL 队列控制、容器和 ABI、最终链接、回归及历史证据。任意流水线抢占、定时 DMA 装载 QCTL RAM、任意手写控制表的完整静态安全证明不在已支持范围内。没有推送。
