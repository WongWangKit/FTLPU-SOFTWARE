# `.ftlpu` v38：计算与分页重叠协议

v38 在 [v37 公共屏障格式](common_barrier_binary_v37.zh-CN.md) 上增加独立搬运启动周期与目标能力。Macro packed-v1 和 BARR encoding 1 均保持不变。v37 继续按原串行语义执行：运行期搬运在 C 启动。新 writer 写 v38；未知该版本的旧 reader 明确拒绝。

## 字段与布局

硬件配置在 v37 的 `common_barrier_encoding_version`、`common_barrier_capacity` 后追加 `common_transport_overlap_version:uint32`。0 表示没有该能力，1 表示支持本文协议；该字段同序加入版本化 target ABI hash。共享 schema-v2 JSON 从 `icu` 对象读取，MLIR target 属性仍使用 `icu_memory`。未声明重叠能力的配置按 0 处理。

`metadata_flags` 的 bit 0 仍表示公共屏障，bit 1 新增表示计算/分页重叠。未知位拒绝。bit 1 要求 bit 0 及目标 overlap version 1；不支持的 CModel 目标拒绝装载。

每条 weight-page-use 在原 `transport_launch_event_id:uint32` 后追加明确 little-endian 的 `transport_launch_cycle:uint32`，即 E。未关联 launch 的候选/预执行实例必须为 0；未启用重叠的程序也必须为 0。最终重叠程序要求 E≤C≤ready_cycle。同一 launch 关联的所有实例必须具有相同 E、相同屏障，并能追溯到实际 DMA WAIT、RX 路由、MEM WRITE_SYNC 包。周期、实例、容量和孤立关联均检查，禁止截断或回绕。

顺序保持为：文件头/ABI/硬件配置及数量，binding/timeline/memory-floor/page-use/stream-release 元数据，事件成员关联，BARR，各计算/C2C 队列及 Native MEM transport 控制区，scale/address relocation。新增 E 位于 page-use 元数据中。stream、span、stream metadata-only、span metadata-only 四条路径均处理这些字段及屏障块。

BARR 始终存在，包括空表：

| 偏移 | 字段 | 小端编码 |
| --- | --- | --- |
| 0 | magic | ASCII `BARR`，4 字节 |
| 4 | encoding_version | uint16，1 |
| 6 | entry_bytes | uint16，8 |
| 8 | entry_count | uint32 |
| 12 | reserved | uint32，0 |
| 16 + 8i | stop_cycle | uint32 |
| 20 + 8i | event_id | uint32 |

目标配置仍为 1024 项、8192 字节 Barrier RAM。C 严格递增；真正相同 C 的依赖合并为 all-of。每任务的完成成员/实际搬运关联有 65535 的有限上限。generation 属于运行时状态，不在每条 BARR 项中重复。

## 执行与边沿

1. INIT：解析、分页最终链接、重定位、生成完整镜像。清理旧任务搬运和事件状态，分配新 generation；配置实例、MEM 提交计数、整张 Barrier RAM 及各 i-MEM，然后允许本任务搬运启动。
2. PRIME：沿用实际有限 i-MEM、取指延迟、reservoir、parser、DDB 和 Context credit 完成前端准备。
3. RUN：L 只随公共计算使能前进。达到 E 且前一搬运已经真实提交并交还资源，才通知相应 DMA/RX launch；计算继续运行。
4. C 的 dispatch 边沿：先检查计算在途状态和屏障，后允许任何 Native/Raw Macro 计算发射。已预解码/入 Context RAM 的指令也受此优先级控制。早到 ready 只锁存，到 C 才消费；此时无需额外 HOLD 拍。
5. 事件未完成则 HOLD，L=C。Native PC/NOP/循环和 Macro Context/坐标/地址/解码缓冲保持；已发取指响应按原有预留 credit 捕获。P、DDR、RX、MEM transport、事件及物理超时继续运行。
6. 最后所需 SRAM tile 提交且搬运尾部交还资源后锁存完成。若完成在物理拍 T 末观察到，计算最早在后续 dispatch 边沿恢复 C。屏障消费后游标前移，并更新当前/下一项缓存。

同页重装对应不同 page-use/launch 实例；计数归属为 generation + launch + MEM bank/slice，完成通知归属 generation + page-use。重复通知幂等，旧 generation 不影响新任务。首版单芯片任务串行切换，换任务前清理旧 DDR/C2C/控制状态，不支持跨任务搬运乱序重叠。

## 资源证明与实际完成

最终链接按计算码流的真实发射域建立静态资源日历。分析展开只用于宿主证明，受 800 万项工作上限约束；不会替换硬件 Macro 前端。选中的 MEM bank 发射口及 West SR 路径在 [E,C) 完整预留，包含 tile 和被动传播排空。首版对两半球按相同 SR 编号保守预留。

选择顺序是最早合法 E、同 E 最多 lane、固定高 SR 到低 SR。允许少于原有 lane 数；不增加端口、SR 或 DMA FIFO。找不到提前窗口时可使用经证明的 E=C，并在 `c2c.serial_fallback.*.launch:*` timeline 记录原因。SRAM 实际写域、旧值生存期、后继页面替换顺序也必须成立。最终镜像加载器根据实际 DMA/RX/MEM 包重新检查这些条件，不信任预留元数据本身。

C 必须已有完整计算排空窗口；计算 Context 可以非空，稳定 SRAM/ACC/权重状态保持。允许带明确来源的搬运继续在途。共享 MEM 同拍竞争仍报错，不通过临时冻结计算掩盖冲突。

MEM 发射后附加来源、generation 和 launch 控制侧带，不改变原生 FU 指令字。C2C 端口显式标记搬运 SR 数据，来源不依赖调试名称或 trace 开关。计数在实际 `write_segment` 后增加，完成必须覆盖所有目标 bank/tile；缺失数据、未注册实例或过量提交明确失败。页完成不依赖无关计算全片 idle，也不依赖 C 后被冻结的计算。

## 编译调度与兼容边界

Gate、Up 和各 Down 页使用实际消费阶段的候选点。Up 的 Gate 写回尾部必须先完成，当前目标为此收取 124 个 L 的调度成本；其中最后 62 拍才是要求全空的排空窗口。`common_barrier.schedule_cost` 与 `common_barrier.drain_cost` 分开记录。Down 页间成本在有限 ACC 槽串行化之后插入，避免 seq128 的第二槽尾部侵入 C 前窗口。

所有 E/C 都是公共 L 坐标，与 Macro 下标/bit offset 无关。运行时分页调整重新链接关联、预留和表；公共路径不再叠加队列 phase_offset。计算端同一分页依赖映射到公共屏障，实际 DMA/RX、MEM 同步写、DMA 完成 SYNC 和 KV readback 控制保留。Raw Macro 失败明确报错，不静默转 FU3D。默认 Native/FU3D 和非分页 Raw 不增加这些分页调度成本。

首版保守预留整个 bank 发射口和整条 SR 编号，可能拒绝更细粒度可行重叠；不支持任意流水线位置抢占、乱序页调度、额外 staging 缓冲或多任务同时执行。未证明的暂停点与资源安排拒绝，不能删除同步或放宽排空规则放行。
