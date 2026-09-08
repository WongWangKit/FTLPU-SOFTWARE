# Qwen KV Cache

## 当前范围

第一阶段把 BF16 KV cache 作为 decoder-layer executable 的持久状态常驻在
LPU MEM 中，在保持现有 seq32 prefill 调度不变的前提下，先建立后续 decode
需要的状态 ABI。本阶段尚未实现逐 token 追加、对历史 token 做 attention、
DDR 换入换出和 KV 量化。

cache 容量由与模型无关的编译选项指定：

```powershell
ftlpu-opt input.mlir --target-config ftlpu-lpu32.json `
  --kv-cache-capacity 256 -o tensor.mlir
```

容量必须是 MXM tile 宽度的整数倍，且不能小于本次编译的 sequence length。
设为 0 时保留原来的临时 attention buffer 行为。

## ABI

每个 decoder executable 暴露两个固定编号的 internal binding：

| Binding | 编号 | Role | 类型与逻辑 shape |
| --- | ---: | --- | --- |
| K cache | 65536 | `state.kv.key` | `bf16[capacity, kv_heads, head_dim]` |
| V cache | 65537 | `state.kv.value` | `bf16[capacity, kv_heads, head_dim]` |

Schedule MEM transfer 同时携带 `address_binding` 和
`address_binding_access = "internal"`。Schedule 到 Command 的 lowering 会保留
这两个属性，binary relocation 因而解析到 `ModelState`，而不是普通模型输入。
`ModelPackage` 为每层记录一对具名 K/V state，`ModelInvocation` 再引用这些名称。

## 物理布局

K 使用 `fp16_head_planar`。由于不同 query-head group 会在两个半球消费 K，
物理地址保留全局 KV-head stride，runtime 把 K 复制到所有启用的半球。V 使用
`fp16_value_x16`，每个 KV head 只存到所属半球，并直接采用 PV 需要的布局。

对于 capacity=256、2 个 KV head、head dimension=128、双半球、每个 SRAM
vector 32 字节的 Qwen2.5-1.5B：

| 数量 | K | V |
| --- | ---: | ---: |
| 每层逻辑数据 | 128 KiB | 128 KiB |
| 每个所选 slice 的行数 | 1024 | 256 |
| 使用的 slice 数 | 4 | 16 |
| 实际写入的物理数据 | 256 KiB | 128 KiB |
| allocator 保守预留 | 256 KiB | 256 KiB |

V 的预留空间大于实际写入量，是因为当前一个 binding 在两个半球都保留了完整
的全局 head 地址区间。后续可拆成按半球分段的 binding，回收这部分空洞。

当前固定 slice placement 是单层里程碑，并不表示整模型 KV 都能常驻。在每个
SRAM bank 只有 8192 行时，K 每层会在同一组 4 个 slice 上占 1024 行，因此
capacity=256 时最多容纳 8 层；V 每层占 256 行，可容纳 32 层。Qwen 的 28 层
要么需要按层把 K 分散到更多 activation slice，要么需要使用 DDR-paged KV。
当前全局 lifetime planner 会正确拒绝超出物理容量的布局。

## Runtime 生命周期

`ModelSession::load` 为每个 state 分配整个 session 生命周期内有效的空间，并
通过 C2C 清零。运行 decoder invocation 时，internal K/V MEM 指令会 relocation
到对应物理分配。`read_state(name)` 通过 C2C 下载，并把物理布局还原为逻辑
`[token, head, dimension]` 顺序；`reset_states()` 通过同一路径清零所有持久状态。
host API 不会绕过 C2C 直接写 MEM。

多层模型使用 `layers.N.key_cache` 和 `layers.N.value_cache` 命名。它们与权重
乒乓页相互独立；物理 placement 能容纳时，不会在 invocation 边界被换出。

## 已验证链路

Qwen2.5-1.5B 第 0 层 seq32 测试以 capacity=256 编译，生成逻辑
ModelPackage，转换为 v5 paged package，从磁盘重新加载后运行 CModel。测试会
对比 Hugging Face fixture 中全部 8,192 个 K 和 8,192 个 V，检查未使用容量
仍为零，验证 49,152 个 decoder 输出，并验证 state reset。

当前误差为：

| 输出 | 超出容差数量 | 最大绝对误差 |
| --- | ---: | ---: |
| K cache | 0 | 0.00390625 |
| V cache | 0 | 0 |
| Decoder 输出 | 0 | 0.015625 |

## 下一步 Decode 工作

下一步需要在 compiler/runtime 契约中加入动态 token position（或 cache
length）、K/V append 区间、QK/PV 有效前缀边界，以及针对
`past_len + current_len` 的 causal mask。数值闭环后，再加入片外 page table
和可选的 per-head/per-page KV 量化；这些扩展不需要改变当前逻辑 state ABI。
