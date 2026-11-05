# DeepSeek V4.1 Flash 适配计划

状态：2026-11-04 完成官方架构核对与本地能力盘点，尚无 LPU 可执行模型测试。

以 `deepseek-ai/DeepSeek-V4.1-Flash-1017` 为第一版固定检查点，revision
`1bae0e1088993b9b82de08f59de5be5a34e42170`。已下载配置与少量官方参考源码，
未下载模型权重、未执行外部源码。来源、SHA256 与逐层清单保存在
`results/deepseek_v41_flash_assessment/`。

## 架构边界

- 40 层主干，CED 分为 20 层 causal encoder 和 20 层 decoder。
- hidden=5120，64 个 query head，head_dim=512，RoPE 64 维；Q rank=1280，
  O 分 8 组、每组 rank=1024；SWA window=128。
- 主干配置对应 2 层纯 SWA、4 层 CSA2 Full、4 层 Reindex、30 层 Reuse。
  Full 的 KV owner 为层 2/8/14/20；索引 owner 还包括 24/28/32/36。
- 每层 MoE 有 384 个 routed expert、1 个 shared expert，每 token 选 6 个。
  router 用 sqrt(softplus)，选专家时加 correction bias，组合权重使用未加 bias 的分数。
- mHC 维护 4 条 residual stream；Engram 位于层 1/14。三层 DSpark 是额外草稿分支。

来源：[固定版 config](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/1bae0e1088993b9b82de08f59de5be5a34e42170/config.json)、
[官方参考实现](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/1bae0e1088993b9b82de08f59de5be5a34e42170/inference/model.py)、
[技术报告](https://arxiv.org/abs/2609.19969)。逐层 owner 清单由配置生成，不能替代执行验证。

## 实施阶段

| 阶段 | 实现内容 | 验收界限 |
| --- | --- | --- |
| 1. 导入与参考 | 固定配置/权重索引；按 tensor 取单层权重；识别原始 FP8/FP4 scale；建立独立数学参考和精度清单 | 形状、权重命名、层类型与官方一致；不把未支持模型误匹配为 dense decoder |
| 2. SWA attention | 第 0 层，batch1、seq32、真实尺寸；低秩 Q、共享 latent KV、归一化、RoPE、带 sink 的 causal/window softmax、inverse RoPE、分组 O 投影 | 每个中间值与参考对齐；raw-macro CModel 执行；随后测试 seq160 和跨窗口 decode |
| 3. mHC 与 MoE | 四路 residual、混合系数与 Sinkhorn；router、动态 Top-6、dispatch、带 clamp 的 SwiGLU、shared expert、combine | 输入改变后，同一编译产物能改变 expert 选择；权重/索引/输出均对齐。固定 expert 单测不能算完整 MoE |
| 4. CSA2 | learned compressor、indexer、Top-K、稀疏 KV 访问；Full/Reindex/Reuse 显式跨层状态依赖；decoder candidate pool | 独立验证压缩尾组、causal mask、索引、共享 KV 生命周期和三种模式；避免复制整份共享 cache |
| 5. Engram 与完整代表层 | token n-gram hash、表查询、门控融合；接入各类 attention、mHC、MoE | 覆盖普通层、带 Engram 层、压缩 owner 层、索引 owner 层、复用层的真实数值与流水图 |
| 6. CED 与模型执行 | encoder/decoder 接口、decoder global KV 来源、SWA replay、prefill→多步 decode、embedding/head | 缓存有效长度/所有权/续跑一致，分块 prefill 与参考对齐，最终 logits/生成可验证 |
| 7. 原生精度与性能 | FP8/FP4 权重/cache、DDR/C2C 动态页、布局融合、预取与利用率；再扩视觉与 DSpark | 区分格式转换功能基线与原生格式；功能与性能分别验收。视觉和草稿加速作为独立覆盖项 |

内存规划和 DDR/C2C 从阶段 2 开始参与；阶段 7 才集中做性能收敛。
首版范围是纯文本，视觉适配另列；Engram 属于文本主干，必须纳入文本模型完成条件。

## 第一个实际算子测试

先取第 0 层 attention 的输入 X，作为子图输入，按下面顺序增加算子：

```
X [32,5120]
  ├─ Wq_a -> RMSNorm [32,1280] -> Wq_b -> Q [32,64,512] -> partial RoPE
  └─ Wkv -> RMSNorm -> latent KV [32,512] -> partial RoPE
          -> causal SWA + attention sink
          -> inverse RoPE -> grouped Wo_a -> Wo_b -> Y [32,5120]
```

第一个可交付点只到 Q/KV 两条分支，接着补完整 SWA attention。
这是 attention 子图测试；外层 mHC/FFN 还未包含时，报告不能写成完整 block。
第一轮可以把上游输入作为固定 fixture；随后必须由真实 mHC 生产该输入。

seq32 小于窗口 128，只能验证基础数值；必须加跨窗口用例。
类似地，CSA2 的 Top-K=512，seq32 不会触发真正的候选裁剪。
需单独用超过 512 条有效 cache 的合成索引测试，以及候选块数超过上限的测试，
验证同一程序在不同输入下产生不同索引。动态索引需保持整数语义，不能放入
精度不足的 BF16 数据通路后当作地址使用。

## 当前硬件与编译器缺口

基础 projection、RMSNorm、RoPE、Softplus、FFN 和静态 C2C 分页有可复用实现。
新增 DeepSeek 图识别/状态语义，共享通用算子 lowering 和全图内存规划。

动态访问需先明确硬件合同：`FTLPU-CMODEL/include/ftlpu/mem/mem_array.hpp`
目前拒绝 Gather/Scatter；软件的 `common_transport_validation.hpp` 也拒绝间接地址。
`IndexedView` 是编译期索引；已有 `moe_router.cpp` 主要为 CPU router/dispatch/combine
和页规划，不能据此宣称 LPU 已实现动态 MoE。

通用 Top-K/索引输出、地址描述符、按选择结果调度的 expert 页，是 MoE 与 CSA2 的
共同前置能力。静态 ICU 指令可以保留在固定形状计算块内；运行时数据决定块间页与索引。
这些边界应先做硬件可实现性验证。若采用 Host 控制原型，必须在报告中显式标注，
不能把 Host 计算结果作为设备原生算子验收结果。CModel 只新增获确认的通用硬件行为。

精度也有独立缺口：MXM 当前 weight input mode 为 Int8DequantBf16/Direct16，
未见原生 FP8/FP4 通路。官方 checkpoint 的反量化高精度参考、LPU BF16 功能基线、
可能的 W8A16 再量化版本和最终原生格式必须分开比较，不静默替换。
大矩阵的 Direct16 分页支持也需在阶段 1/2 验证。

所有最终设备测试继续使用 raw-macro ICU，导出 runtime pipeline 和实际执行的
每 ICU program。Python 负责导入和参考数值；C++ 编译器负责布局、分块、排程和指令。
