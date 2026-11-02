# QCTL 每队列 16 条 RAM＋8 条缓存复验

日期：2026-11-02。基于 SOFTWARE `6f3473e`、CModel `c0988ab` 的本地 main 修改；本次不提交、不推送。沿用既有调度、分页计划、控制语义和数值门限，仅缩小目标控制 RAM 并分离两类容量校验。

## 配置和校验

- lpu32 每队列伪控制 RAM 从 256 条改为 **16 条**；缓存仍为 **8 条**，读延迟 2 拍、每拍单读请求。16 条是队列程序总容量，缓存不额外增加程序容量。
- Native 绑定用于校验真实控制字、源位置及释放关联，不装入伪控制 RAM。每分区的容器数量上限仍为 256 条；伪控制分区另受目标 RAM 的 16 条上限约束。
- 同一分区禁止混合 Native 绑定和伪控制；保留源位置唯一、指令匹配、事件关联及装载时的目标容量检查。超过 16 条的伪控制明确拒绝，未增加运行中控制 RAM 换页。
- `ftlpu_binary_inspect --storage-only` 分别输出伪控制峰值和 Native 绑定峰值，避免用 Native 记录数估算执行 RAM 深度。
- packed-v1、v39 布局和控制状态机不变。容量字段参与目标 ABI；旧的 256 条容量文件需要针对新目标重新编译/链接，不能直接截断或跳过校验。

## 最小验证结果

两个工程使用原构建目录增量重建。只运行 CModel `icu_queue_control_test`、SOFTWARE `binary_queue_control_test`，以及 seq32 在 **25,600 MB/s** 下的一组 Native/Macro 完整 decoder 对照，没有重跑 seq128 或带宽矩阵。

- 四条容器读取路径通过：stream、span 和两种 metadata-only。64 条 Native 绑定在 16 条 RAM 配置下可读；16 条伪控制可读，完整的第 17 条被拒绝。Native 错误源指令、重复源位置、混合绑定类型和超过容器记录上限仍拒绝。
- CModel 有限前端验证通过：16/17 条边界、缓存不足拒绝、真实读延迟、WAIT 同拍放行、SYNC/NOTIFY 占拍、活跃 Context 跨等待、独立队列继续执行、共享 MEM 交接、在途数据接收及提交事件测试。
- Native 和 Macro 分别从新配置重新编译，执行真实 attention＋FFN、DMA/RX/MEM 同步写和 SRAM 提交。两次均完成 7 个运行期 page use、8 次预取、331,776 个 tile 提交；没有跳过数值校验。

| 指标 | Native | Macro＋QCTL |
|---|---:|---:|
| 主执行物理周期 P | 110,978 | 110,978 |
| PRIME | 0 | 80 |
| 初始页面等待 | 3,375 | 3,375 |
| 后置搬运尾部 | 34,825 | 34,825 |
| 计算∩DMA 周期 | 17,733 | 17,733 |
| 计算∩RX 周期 | 26,905 | 26,905 |
| 计算∩SRAM commit 周期 | 26,872 | 26,872 |
| BF16 输出数 | 18,432 | 18,432 |
| CPU 参考最大绝对误差 | 0.03125 | 0.03125 |
| FFN hidden 硬件对照 mismatch | 0 | 0 |

P 不含 PRIME、初始页面准备和后置搬运。1,794 组计算与搬运轨迹、各 961,457 个资源活动拍逐拍相同；比较仅统一 Read/Read3D 等编码名称并忽略前端 PC/source 诊断字段，保留物理周期及其余事件细节。另将同一 Native 同步计划降低后的最终执行镜像与 Macro 镜像对比，确认计算、搬运、操作数、relocation、事件和释放相位等价。两个最终文件的独立加载校验也通过。

这些结果与 [main 集成时的 seq32 结果](queue_control_main_integration.zh-CN.md) 相同：缩小 RAM 未增加主执行或 PRIME 周期。结论限于本次程序及既有有限前端条件，不外推到任意控制密度的程序。

## 实际占用和存储估算

Macro 最终文件包含 350 条伪控制和 148 条 Native 绑定。每队列伪控制峰值 **8 条**，占 16 条 RAM 的一半；Native 绑定峰值 **63 条**（另一个 DMA 队列 54 条），不占伪控制 RAM。最终文件为 485,258 B，其中计算 i-MEM 85,876 B、transport i-MEM 2,496 B、QCTL 块 13,868 B、QLNK 340,763 B。文件大小与物理 RAM 预留容量是不同指标；Native 校验元数据继续保留在文件中。

按此前全配置 220 个 MEM/MXM 控制分区、每条执行记录 9 B 紧凑或 12 B 对齐估算：

| 存储 | 容量估算 |
|---|---:|
| 16 条 RAM × 220 队列 | 30.94～41.25 KiB |
| 8 条缓存 × 220 队列 | 15.47～20.63 KiB |
| RAM＋缓存记录载荷 | **46.41～61.88 KiB** |

RAM 记录载荷较此前 256 条方案的 495～660 KiB 减少 93.75%；缓存深度未变。估算不包含游标、有效位、在途请求状态、事件/释放计划、ECC 和物理存储宏粒度，也不是 RTL 综合面积。QCTL 的有限容量模型和目标声明已修改，本次未进行 RTL 实现。

## 证据与复现

小型结果、复现脚本和文件 SHA-256 见 [evidence/queue-control-16x8](evidence/queue-control-16x8/)。真实流水 CSV、编译产物、最终文件和构建日志保留在 `out/qctl-16x8/`，没有覆盖 `out/main-qctl-integration/` 的历史结果。

先用更新后的 CModel 配置重新配置并增量构建 `out/main-qctl-integration/build` 中的 `ftlpu_compile`、`ftlpu_binary_inspect`、`binary_queue_control_test`、`compiled_smollm2_common_barrier_decoder_test`，以及 CModel `build` 中的 `icu_queue_control_test`。之后执行：

```text
python docs/evidence/queue-control-16x8/validate.py test
python docs/evidence/queue-control-16x8/validate.py compile
python docs/evidence/queue-control-16x8/validate.py run
python docs/evidence/queue-control-16x8/compare.py 32
```

运行脚本清理继承的 `FTLPU_*` 开关再设置本次参数，保证未启用 trace-only、数值跳过或替代调度。已有边界仍保留：QCTL 使用直接装载和有限 PRIME；尚无定时 DMA 装载 QCTL RAM、运行中控制 RAM 换页或任意流水线抢占。本次未复验 seq128。
