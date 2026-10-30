# v39 队列伪控制协议：验证记录

> 本页保留 2026-10-29 原基点的验收证据。2026-10-30 合入更新后的 main 及其复验结果见 [main 集成记录](queue_control_main_integration.zh-CN.md)。下文的“未提交”和工作区状态描述均为当时记录。

日期：2026-10-29。SOFTWARE 原 main 基点 `7ba85e7`，CModel 原 main 基点 `bc84188`，包含原有未提交修改及本次 v39 修改；本任务未提交、未推送。实现说明见 [queue_control_v39.zh-CN.md](queue_control_v39.zh-CN.md)。本记录对应真实 CModel 执行，不是删除同步后的压缩回放。

工作区状态变化：18:01–18:03 的外部 Git 操作将上述两个 worktree 改到 `codex/preserved-main-worktree-20261029`，入口 main 分别前进到 `ea81307` / `06f9ee7`。本记录的结果仍对应原基点及其未提交修改；尚未验证与新 main 的 Qwen3.5／DMA 程序装载改动集成后的行为。本任务没有自行切换分支或覆盖入口 main。

## 结论与比较口径

seq32、seq128 完整 attention＋FFN，在 51,200、25,600 MB/s 和 51,200 MB/s 额外读延迟 17 拍（seed=20261029）三种条件下均通过。每组 Native/Raw Macro 使用同一 Native 计算调度、布局、搬运包及释放相位，并使用同一种**实际 SRAM 提交**完成事件。

六组比较的主执行物理周期相等；计算、DMA、RX、SRAM commit 的 CSV 活动时间相等。额外把最终 Native 镜像按同一同步计划转换为 Macro，去除仅供宿主重新链接的 QLNK 后，对整个执行镜像做字节比较，六组均一致。这覆盖指令操作数、地址/scale relocation、同步包、事件及释放相位，不仅比较 CSV 中的指令名称。

这证明已验收程序恢复了 Native 同步与流水重叠效果；不证明任意程序在有限前端条件下都能逐拍等价。PRIME 单列，不以插入隐藏计算空拍补偿前端供给。

## 真实 decoder 结果

下表 `P` 从主执行第一拍计起，不含 PRIME、初始页面准备及后置 KV/输出搬运；Native 和 Macro 的 `P` 相同。重叠是计算数据通路存在活动时，DMA/RX/实际 SRAM commit 也存在活动的物理拍数，三个计数不能相加。

| 序列 | DDR MB/s / 额外延迟 | Native P | Macro P | 计算∩DMA | 计算∩RX | 计算∩commit |
|---|---|---:|---:|---:|---:|---:|
| seq32 | 51,200 / 0 | 94,192 | 94,192 | 23,257 | 28,111 | 28,117 |
| seq32 | 25,600 / 0 | 112,928 | 112,928 | 19,408 | 29,351 | 29,317 |
| seq32 | 51,200 / +17 | 101,131 | 101,131 | 27,351 | 35,656 | 35,643 |
| seq128 | 51,200 / 0 | 409,011 | 409,011 | 23,872 | 29,095 | 29,094 |
| seq128 | 25,600 / 0 | 427,435 | 427,435 | 24,365 | 36,139 | 36,140 |
| seq128 | 51,200 / +17 | 415,863 | 415,863 | 27,893 | 36,396 | 36,382 |

- 原生前端的额外 PRIME 统计为 0；Macro seq32 为 80 拍，seq128 为 104 拍。原生本来就有的有限 IQ 启动行为保留，0 不表示硬件没有启动成本。
- 初始页面准备等待，两种序列/编码在三种条件下均分别为 1,849、3,375、2,401 拍。
- seq32 比较 18,432 个 BF16 值，CPU 参考最大绝对误差 0.03125；seq128 比较 73,728 个，最大绝对误差 0.0625。FFN hidden 的硬件 Native 对照 mismatch=0、replica mismatch=0，沿用既有误差门限。
- 每次执行真实完成 7 个运行期 page use、8 次预取、2,801,664 字节，104 个 MEM_WRITE_SYNC 包、82,944 次同步 FU 写、331,776 次 tile 提交。56 个 MEM_READ_SYNC/KV 控制保留在单独 page-out 计划中，不作为普通页面 WAIT 删除。
- 日志旧字段 `runtime_wait_cycles` 已改名 `completion_tail_cycles`（仅 QCTL 路径）。它表示超出初始运行预算的尾部完成拍数，**不是全片 HOLD 或逐队列等待之和**。依次为 seq32 14,280 / 33,016 / 21,219，seq128 14,943 / 33,367 / 21,795。

每个序列的三档延迟使用完全相同的最终 Macro 文件（SHA-256）：

| 序列 | 最终文件 SHA-256 |
|---|---|
| seq32 | `a1295ea3f79e72e61f652f915c50a8cf05cf1dbf35516c24733d0ccf6463022d` |
| seq128 | `bc1141f4c4ebd4e5a7f127b4f34ef7c2b4ec888bcef9b2d6cf1616640669e2cf` |

保存文件重载验收：使用 51,200 MB/s 时保存的最终 Native 和 Macro 文件，在 25,600 MB/s 下装载执行；装载前对本次重新核验的分页计划与文件做字节一致性检查，执行对象使用文件读取结果。四次执行均通过，P 与上表相同；两组重新加载的物理活动轨迹也完全一致。结果见 `loaded-*-seq*-25600.log` 和 `loaded-*-trace-check.log`；这项检查不依赖离线队列回放。

## 逐队列进度

低带宽场景 Native/Macro 每个队列的等待计数一致。以下是同类队列等待拍数范围；完整每队列坐标、发射数、结束物理拍和等待拍记录在 `*-queues.csv`。Raw 的坐标是剥离控制后的 `Lq`，Native 保持既有坐标，不能直接把两种坐标列当作相同语义。

| 队列种类 | 数量 | seq32 等待范围 | seq128 等待范围 |
|---|---:|---:|---:|
| MEM | 128 | 0–33,066 | 0–84,619 |
| MXM Load / Compute / Dequant | 各 2 | 0–33,066 | 0–32,844 |
| VXM | 12 | 33,066 | 32,844 |
| SXM Transpose / Permute | 各 2 | 0 | 0 |
| DMA | 2 | 92,665–94,981 | 240,838–243,110 |
| RX | 2 | 88,632–92,749 | 236,771–240,899 |

DMA/RX 等待包含其原生启动/同步等待，不能当作计算停顿。不同队列可以同时等待，因此新协议不再使用 `P=L+HOLD` 作为全片恒等式。高带宽首次六组验证中未单独导出逐队列 CSV；低带宽及延迟扰动两组有完整导出，不能由主执行 P 反推缺失的队列计数。

## 存储及硬件成本

| 项目（字节） | Native seq32 | Macro seq32 | Native seq128 | Macro seq128 |
|---|---:|---:|---:|---:|
| 主队列 i-MEM（Native 含其同步包） | 252,848 | 98,968 | 1,136,096 | 257,264 |
| 独立 transport i-MEM | 0 | 2,496 | 0 | 2,496 |
| QCTL 文件块 | 13,820 | 14,252 | 13,820 | 14,252 |
| QLNK 宿主重新链接配方 | 363,830 | 363,830 | 1,621,786 | 1,621,786 |
| 完整最终文件 | 757,203 | 527,418 | 3,275,539 | 1,987,881 |

Native 列也携带 QCTL 绑定及 QLNK，以便同一释放/提交协议作受控对照。Macro 主队列＋transport i-MEM 相比此 Native 分别减少约 59.9% / 77.1%。整个文件仍有大量宿主配方、元数据和 relocation；不能宣称获得同样的文件压缩率。

两种序列均有 512 条控制记录，其中 Macro 的 354 条为伪控制，其余为 Native 绑定。最大分区 63 条；目标每队列深度 256、缓存 8、读延迟 2、单读端口。QCTL 24 字节线格式包含宿主源位置/相位关联；CModel 控制 RAM 使用执行字段，未据此推算综合后的 RTL 面积。BARR 仍有 16 字节空头。

## 基线变化

锁定的旧 Native 实现按“写发射数＋估计/退休状态”判页就绪时，P 为 seq32 94,184 / 112,920，seq128 408,627 / 427,051。改成真实末 tile 提交后，同一 Native 计划增加 8 / 384 拍；Macro 在相同真实完成事件下与它逐拍一致。这项变化是完成语义修正，不能隐去并声称与旧的提前就绪事件完全相同。

历史串行版本 seq32 为 109,165 / 135,500 拍；新路径相比分别少 14,973 / 22,572 拍（约 13.7% / 16.7%）。历史版本包含不同的全局屏障调度空窗，这只是总体历史对比，不能代替上表固定同一 Native 调度的对照。

## 定向验证

SOFTWARE 8 项、CModel 3 项相关回归通过。旧 BARR/v37/v38 读写和执行仍保留其校验，默认非分页 lowering、packed bitstream、Native 分页及 ping-pong 均在定向回归范围内。

| 验证点 | 证据 |
|---|---|
| Native/伪 WAIT 早到、晚到、同拍连续消费；SYNC 占拍；NOTIFY 逐拍 | CModel `queue_control_test::compare`，比较完整 MEM 编码和物理发射拍 |
| 多活跃 Context、单 Macro 跨等待、错误标签及其他队列继续 | `overlapping_contexts`、`compare` |
| 有限缓存、单端口读延迟、请求已发响应未到时暂停 | `finite_frontend`、`response_during_wait`；已有 Macro decoder hold 测试继续覆盖 reservoir 响应 |
| 消费者在等待后仍接收先前在途数据 | `issued_data_keeps_flowing(false)`，实际逐 tile 写入 SRAM |
| MEM 包退休但末 tile 未提交；页面不能提前 ready | `issued_data_keeps_flowing(true)`；实际提交计数独立于控制退休 |
| generation、all-of、重复完成、同页重装、未知实例 | `completion_generation`；合法普通 NOTIFY 不经过页去重 |
| 空表、四条读取路径、损坏块、容量/周期、未知操作、绑定错误、孤立事件 | SOFTWARE `binary_queue_control_test` |
| 装载后第一拍前到达的事件 | `early_event_after_load`；提前完成 Native 初始化，避免首拍清掉通知 |
| 候选镜像与最终页事件要求 | `link_source_roundtrip`；QCTL/QLNK 全部读取后再验证关联 |
| 自身未来搬运依赖、RX/MEM 反向页顺序 | `transport_dependency_order`；基于实例依赖图拒绝，不引入公共 Lq |
| 共享端口交接 | `transport_order`，Native/pseudo 完整指令逐拍一致 |
| SR/端口冲突、活跃存储覆盖及错误搬运链 | 既有 `runtime_common_barrier_test`、`c2c_weight_ping_pong_test`；QCTL 保留 Native 运行时竞争检查及最终包关联检查 |

上述资源拒绝测试中，旧全局安全点的案例继续测旧协议；它们不构成 QCTL 任意手写控制表的静态安全证明。QCTL 针对新依赖图增加了专用拒绝测试。

## 复现和证据

原始产物目录：`E:/Fudan_University/internship/timesintelli/LPU/work/FTLPU-SOFTWARE/out/queue-control-v39`。小型日志、汇总 JSON、逐队列 CSV 和复现脚本归档在 [evidence/queue-control-v39](evidence/queue-control-v39/)；大 CSV 和二进制留在原始产物目录，可用工程 csv_viewer 查看。

编译输入为 `compiler/examples/smollm2_135m_decoder_layer/decoder_layer_seq32.stablehlo.mlir` 和 seq128 对应文件。共同选项：

```text
ftlpu-compile --input <mlir> --output <ftlpu> --input-stage stablehlo
  --target-config <CM main>/config/ftlpu-lpu32.json --weight-bank 1
  --kv-cache-capacity 256 --mxm-execution vector
  --projection-rope-overlap off --ffn-schedule tail
  --icu-compression none|raw-macro
```

真实执行使用 `compiled_smollm2_common_barrier_decoder_test.exe` / `compiled_smollm2_common_barrier_seq128_decoder_test.exe`。Native 传 compiled-none 并设置 `FTLPU_QUEUE_CONTROL_FINAL_ENCODING=native`；Macro 传 compiled-none 与 compiled-raw-macro 两个参数。`FTLPU_DDR_BANDWIDTH_MBPS` 选带宽，扰动组额外设置 `FTLPU_DDR_EXTRA_READ_LATENCY=17`。没有跳过数值检查。

`compare-final.py` 展开 CSV 重复项、只消除 Native/Raw 来源命名差异，保留物理周期；`binary_queue_control_test --compare-final <native> <macro>` 比较完整执行镜像。`--validate-final <file>` 独立装载并验证最终 QCTL、搬运依赖、SRAM 目标和有限预热。

## 尚未支持与后续工作

- 不实现任意流水线抢占。WAIT 只阻止所在队列后续命令，已发出的流水继续。一般手写程序仍须遵守 Native 的数据/资源依赖。
- 控制缓存只有 8 条；超出真实供给能力的同拍链或过密 Macro 解码显式报错，不自动添加空拍。
- 当前验证覆盖 MEM/MXM coarse-domain packed-v1，其他计算队列保留 Native。native4 和未支持的粗粒度组合继续拒绝。
- QLNK 只保存紧凑 Native 域，仍占较大文件空间；后续可压缩配方/剥离不再需要运行时重排的配方，但必须保留可检查的控制映射。
- v39 metadata-only 两条路径暂时完整解析再丢弃执行队列，峰值内存不等于纯元数据扫描。
- 分页 launch 使用原有 ModelSession 释放计划；直接 CModelRuntime 使用者仍须投递声明的 External 事件。仅装载 QCTL 不会凭空启动外部 DMA。
- 没有综合 RTL 面积/频率数据，没有所有模型、任意 SRAM 别名或任意手工 QCTL 的形式化安全证明。真实对照结论限定于本文列出的程序、调度和前端条件。
