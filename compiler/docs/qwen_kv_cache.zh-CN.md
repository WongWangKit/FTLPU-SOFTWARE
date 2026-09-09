# Qwen KV Cache

## 当前范围

当前实现为每个 decoder layer 建立片外 session backing 中的逻辑 BF16 KV
cache，LPU MEM 只保留 executable 当前使用的窗口。在保持 seq32 prefill 调度
不变的同时，建立后续 decode 所需的持久状态 ABI。本阶段尚未实现逐 token
追加、对历史前缀做 attention、移动 page offset 和 KV 量化。

cache 容量由与模型无关的编译选项指定：

```powershell
ftlpu-opt input.mlir --target-config ftlpu-lpu32.json `
  --kv-cache-capacity 256 -o tensor.mlir
```

capacity 是逻辑上限，不能小于本次编译的 sequence length。物理 resident
window 是 sequence length 向上对齐到 `mxm_rows` 的结果，`page_tokens` 由目标
拓扑推导。设为 0 时保留原来的临时 attention buffer 行为。

## ABI

每个 decoder executable 暴露两个固定编号的 internal binding：

| Binding | 编号 | Role | executable SRAM shape |
| --- | ---: | --- | --- |
| K cache | 65536 | `state.kv.key` | `bf16[resident_tokens, kv_heads, head_dim]` |
| V cache | 65537 | `state.kv.value` | `bf16[resident_tokens, kv_heads, head_dim]` |

Schedule MEM transfer 同时携带 `address_binding` 和
`address_binding_access = "internal"`。Schedule 到 Command 的 lowering 会保留
这两个属性，binary relocation 因而解析到 `ModelState`，而不是普通模型输入。
`ModelPackage` v6 为每层记录逻辑 `[capacity, kv_heads, head_dim]` shape、
`page_tokens` 和 `resident_tokens`，`ModelInvocation` 再引用对应的 K/V state 名称。

## 物理布局

K 使用 `fp16_head_planar`。由于不同 query-head group 会在两个半球消费 K，
物理地址保留全局 KV-head stride，runtime 把 K 复制到所有启用的半球。V 使用
`fp16_value_x16`，每个 KV head 只存到所属半球，并直接采用 PV 需要的布局。
两个 persistent window 都放在对应 slice group 的 SRAM 高端，并一直保留到
executable 结束，避免后续 FFN 在 runtime page-out 前覆盖 KV 数据。

对于 seq_len=32、capacity=256、2 个 KV head、head dimension=128、双半球、
每个 SRAM vector 32 字节的 Qwen2.5-1.5B：

| 数量 | K | V |
| --- | ---: | ---: |
| 每层逻辑数据 | 128 KiB | 128 KiB |
| resident window 逻辑数据 | 16 KiB | 16 KiB |
| 每个所选 slice 的行数 | 128 | 32 |
| 使用的 slice 数 | 4 | 16 |
| C2C 实际传输数据 | 32 KiB | 16 KiB |
| SRAM interval 预留 | 32 KiB | 32 KiB |

V 的预留空间大于实际写入量，是因为当前一个 binding 在两个半球都保留了完整
的全局 head 地址区间。后续可拆成按半球分段的 binding，回收这部分空洞。

Qwen 的 28 层仍有各自独立的逻辑状态，capacity=256 时合计 7 MiB。各层按顺序
执行并复用 compiler 声明的 K/V staging 地址；bank0/bank1 executable variant
最多各需要一对 K/V slot，而不是 28 对。如果 target ABI、layout、shape、
slice、bank 或 binding index 不同，planner 仍会保留不同的 slot。

## Runtime 生命周期

`ModelSession::load` 在片外 session backing 中为每个 state 分配并清零完整逻辑
空间。每次 decoder invocation 开始前，runtime 通过 DDR 和 C2C 把该层 K/V
resident window 搬到共享 SRAM slot；执行后再把两个 window 搬回并更新逻辑
backing。`read_state(name)` 返回逻辑 `[token, head, dimension]` image，
`reset_states()` 清零所有逻辑状态。host API 不会绕过 C2C 直接写 MEM。

多层模型使用 `layers.N.key_cache` 和 `layers.N.value_cache` 命名。它们与权重
乒乓页相互独立。runtime stats 和 pipeline CSV 分别用 `state_page_in/out` 与
`C2C.StatePageIn/Out` 展示 KV 流量。

## 已验证链路

Qwen2.5-1.5B 第 0 层 seq32 测试以 capacity=256 编译，生成逻辑
ModelPackage，序列化为 v6 package 后运行 CModel。测试会
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
length）、resident-window page offset、K/V append 区间、QK/PV 有效前缀边界，
以及针对 `past_len + current_len` 的 causal mask。当前逻辑 state 和 page-size
ABI 已为移动窗口留出空间；decode 数值闭环后可继续加入 per-head/per-page KV
量化。
