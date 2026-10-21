# `.ftlpu` v37 公共屏障容器

v37 沿用 `FTLPUB01` magic，Macro packed-v1 编码保持 1。旧加载器由于容器版本未知而拒绝 v37；新 reader 继续读取 v1–v36。v29–v36 的硬件字段不含屏障几何，reader 只有在旧 command ABI hash 匹配时才升级为含公共屏障能力的新 ABI。读取成功仍不替代装载器的目标、资源与安全点校验。

`ExecutableHardwareConfig` 在 v36 硬件字段后追加两个 uint32：`common_barrier_encoding_version = 1`、`common_barrier_capacity = 1024`。两项按此顺序追加到现有 command ABI hash；容量不可大于目标物理容量。每任务每芯片有一张最多 1024 项的有限表，事件完成成员总数不得超过 65535，硬件 RAM 占 8192 字节。描述符 Context、packed Macro 和 Native 队列共享此表。

## 记录顺序

1. 原有文件头、target ABI/name、硬件配置、max_cycle 和八个数量字段。
2. 新增三个 uint32：`barrier_event_count`、`metadata_flags`、`transport_queue_count`。flags bit 0 表示采用公共屏障协议，其余位必须为零。原有 queue_count 是计算与 transport 队列记录的总数。
3. 原有 binding、timeline、memory floor、weight page use、stream release metadata。v37 每个 weight page use 在原有 release_cycle 后追加 little-endian uint32 `transport_launch_event_id`，候选/pre-execution 为 0，最终 runtime 关联实例必须非零。
4. `barrier_event_count` 个事件关联记录。
5. 必须存在的 BarrierBlock，包括零项情况。
6. queue_count 条队列记录，末尾 transport_queue_count 条属于独立 Native MEM transport 控制区；其余为计算及 C2C 队列。
7. 原有 scale relocation；address relocation 在原有 write_port 字段后追加 uint16 flags，bit 0 选择 transport 控制区，其余位保留为零。旧版默认为计算区。

独立 transport 队列沿用现有 Native 控制/原生 MEM 序列化方式；它禁止 Raw Macro 或 Macro schedule。其 CompactTagged 容器方式仍可编码现有 MEM 同步控制字，不表示 Macro 执行模式。文件布局不表示运行顺序。运行时必须先解析、最终分页链接、重定位并生成完整硬件镜像，清理 generation 相关事件，再分别装载 Barrier RAM 和 i-MEM，然后允许 DMA 启动。

## 事件关联

每条事件记录显式 little-endian 编码：

| 字段 | 编码 |
| --- | --- |
| event_id | uint32，非零 |
| dependency_count | uint32，非零 |
| page_use_indices | dependency_count 个 uint32 |

page_use_indices 指向本文件 weight_page_uses 的具体记录下标。每条记录代表一次搬运实例，同一 binding/page 多次装入必须各有一条 use 记录；不能以 page_index 缓存永久完成状态。use 的 transport_launch_event_id 指向实际 DMA/RX 队列等待的 launch tag，控制器到关联的 C 时释放，加载器可独立执行最终 linked 文件。一个事件聚合其全部依赖，只有所需页面实例全部完成 SRAM 可见 commit 才能置 ready。DDR 读取完成不够。一个 launch tag 可以代表同一公共屏障内的一组 use，但不得跨公共屏障重用。一个 use 实例只关联一个事件，event_id 全表唯一且与屏障一一对应，禁止孤立 ID 和重复消费。generation 属于任务运行时状态，换任务后旧 generation 的通知不得改变新表状态。

## BarrierBlock

固定 16 字节块头，逐字段 little-endian 写入，禁止直接写 C++ struct：

| 偏移 | 字段 | 编码 |
| --- | --- | --- |
| 0 | magic | 4 字节 ASCII `BARR` |
| 4 | encoding_version | uint16，1 |
| 6 | entry_bytes | uint16，8 |
| 8 | entry_count | uint32，0–1024 |
| 12 | reserved | uint32，0 |

随后每项 8 字节：`stop_cycle:uint32`、`event_id:uint32`。空表仍保留整个块头。表项 stop_cycle 严格递增，相同 C 的多页面必须合并为一个 all-of 事件。周期超过 uint32 必须在链接算术转换前通过 `make_binary_barrier` 拒绝；公共协议程序的 max_cycle 也不得超过 uint32。stop_cycle 不得超过 max_cycle，且不得晚于其依赖页面的 ready_cycle。不得以 Macro 下标或码流 bit offset 表达同步坐标。

stop_cycle=C 在逻辑周期 C 的任何计算发射之前检查。所有逻辑周期小于 C 的计算已执行；事件未 ready 则 L 保持 C，计算域保持状态。物理时间、事件与 transport 推进。事件早到只锁存，仍必须到 C 消费；重复或旧 generation 的通知不得重复放行。表必须在允许计算推进前完整装载。表项损坏、容量不足、逻辑时间已越过 C、表未准备好均为显式错误。

## Reader 与验证

stream、span、stream metadata-only、span metadata-only 均解析并校验 BarrierBlock、关联记录、模式标识及 transport 区域计数。metadata-only 保留 weight_page_uses 以验证事件关联，仍不物化队列内容。完整装载器必须再次验证 Native transport 内容、资源冲突及安全点证明。

`binary_barrier_test` 检查四条路径空表/多表往返、v36 兼容、all-of 和同页重装实例、transport 区与 relocation selector、magic/version/entry_bytes/reserved 损坏、截断、未知事件/无效依赖、重复 C/ID/实例、容量和周期溢出。
