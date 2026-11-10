# DeepSeek V4.1 Flash 适配计划

截至 2026-11-07，已建立第 0 层纯 SWA 的 Q/KV 数值参考，并在 LPU32
上执行了两个 RMSNorm、两组单头 partial RoPE 和一个真实权重 K32 点积。
三次全尺寸投影和窗口缓存写入仍待接成连续设备程序。

以 `deepseek-ai/DeepSeek-V4.1-Flash-1017` 为第一版固定检查点，revision
`1bae0e1088993b9b82de08f59de5be5a34e42170`。配置、参考源码摘要和逐层清单见
`results/deepseek_v41_flash_assessment/`。本轮按 safetensors 字节范围提取的 8 个
权重及 scale 张量共 51,171,584 字节，来源与 SHA256 保存在下文测试目录的
`weights/provenance.json`。

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
  └─ 每 token 每 32 维动态量化，供 Wq_a 与 Wkv 共享
      ├─ Wq_a -> RMSNorm [32,1280] -> 再次动态量化 -> Wq_b -> Q
      │      -> 64 个 head 各自做 partial RoPE [32,64,512]
      └─ Wkv -> RMSNorm -> partial RoPE [32,512] -> 每 32 维 FP8 量化/反量化
             -> 写 BF16 SWA 环形缓存 [128,512]，本例 32 行有效
  Q 和有效 KV -> causal SWA + attention sink -> inverse RoPE
              -> grouped Wo_a -> Wo_b -> Y [32,5120]
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

首版 Q/KV 已选择官方 A8W8 量化，权重在 MXM IW 展开到 BF16，激活在 VXM
完成动态量化/反量化后走现有 BF16 端口，点积与跨块累加使用 FP32。
X 与 Q-RMSNorm 输出分别按每个 token 的连续 32 维动态量化。
W 保留 checkpoint 的 E4M3FN 码和每个 [N32,K32] 权重块的 UE8M0 比例。
在同一 token 内，Wq_a 和 Wkv 可共享 X 的码及 scale。窗口 KV 写回要再执行一次官方
动态量化和反量化。`block_fp8_quantize` 的两个结果和 `block_fp8_matmul` 的四个输入
在 Kernel IR 中有独立的浮点类型和形状。真实形状例子见
`compiler/tests/Inputs/deepseek_v41_layer0_projections.kernel.mlir`。

已按确认方案增加 WS 权重 IW 的 `Fp8DequantBf16`：读取原始 E4M3FN
字节和对应 N32/K32 块的比例，在写入权重缓冲前展开为 BF16。激活继续使用
现有 BF16 端口；动态 group32 FP8 量化和反量化由设备 VXM 执行，不能省略为
未量化的 BF16 输入，也不能在 Python 中代算。两个半球使用各自权重块的比例。
每 K32 重新开始 FP32 部分和，再按 K32 顺序加到独立 FP32 总和，末尾才转 BF16。
真实 Wkv 的 K 分组 0 和 47 已验证动态激活 QDQ 与 FP8 IW 的连续执行。
已解码输入的测试时延仅计量下游计算。使用当前后端编译整个投影 IR 时，
通用入口尝试在东半球分配常驻输入，因容量不足而终止。单是 Wq_b 就有
40 MiB 的 FP8 payload，另有 40 KiB 比例。实现完整子图时，编译器需要把
权重与对应的比例一起分页到 DDR/C2C 路径；还需独立核对真实页等待和 iMEM 开销。

所有最终设备测试继续使用 raw-macro ICU，导出 runtime pipeline 和实际执行的
每 ICU program。Python 负责导入和参考数值；C++ 编译器负责布局、分块、排程和指令。

## 第 0 层 Q/KV 的复现与验收

在 `FTLPU-SOFTWARE` 根目录运行。下列命令使用本工作区的 Release 构建、LPU32 和 wide SXM。
输入 X 是固定 seed 1017 的 BF16 张量，窗口起始位置为 0。原始投影权重是 E4M3FN，
每个 32×32 权重块有一个 E8M0 scale。参考对每个 K32 点积应用激活和权重的
两组 scale，按 K 顺序用 FP32 累加，投影末尾转回 BF16。CPU 参考独立于设备，
可单独检查完整 Q 和有效 KV 缓存。这里的周期都来自各自的独立设备调用。

```powershell
$case = 'results/deepseek_v41_flash_layer0_qkv_prefill'
$build = 'build-ftlpu-vs2026-direct'
$target = '../FTLPU-CMODEL/config/ftlpu-lpu32.json'
cmake --build $build --target ftlpu_opt ftlpu_compile ftlpu_graph_compile compiled_stage_file_runner ftlpu_icu_program_export
python compiler/tools/import_deepseek_v41_layer0_qkv.py
python compiler/tests/deepseek_v41_qkv_primitives_test.py --build-dir $build --target-config $target --package $case
```

| 独立设备用例 | 固定目录 | 验收条件 |
| --- | --- | --- |
| Q 低秩 RMSNorm，32×1280 | `isolated_norms/q_norm` | 原始 BF16 gamma、ε=1e-20，NRMSE ≤ 1%，cosine ≥ 0.9999 |
| 共享 KV RMSNorm，32×512 | `isolated_norms/kv_norm` | 同上 |
| Q head 1 与共享 KV 的 partial RoPE，各 32×512 | `isolated_rope_real/q_head_1`、`isolated_rope_real/kv` | 每个位置前 448 维保持原值，后 64 维与黄金值逐 BF16 位匹配 |
| Wkv，32×32×32 的单块点积 | `isolated_dot32` | 已量化输入无损展开为 BF16，Direct16 MXM 输出 FP32，每元素绝对误差 ≤ `1e-6 × Σ(abs(x_k w_k))` |

测试入口会离线校验权重、重新生成黄金值，依次执行算子和混合幅度回归。
测试根目录的 `summary.json` 索引每次运行的数值结果、实际流水 CSV 与各 ICU 的指令清单。
各子目录包含 `summary.json`、`runtime.pipeline.csv` 和 `runtime_icu_programs/index.csv`。
两路 RMSNorm 与两路 RoPE 吃独立参考提供的中间张量。单块 MXM 测试验证未缩放
的 FP32 点积。所有用例均检查 `host_operations=0`。对比修复前的两个 RMSNorm
结果见 `isolated_norms/before_range_fix.json`。下一步从设备侧的每组 32 维激活量化开始，
再将每个点积的两组比例与 FP32 跨块累加接入 Wkv，验收同一输入运行至缓存写回。

## 官方量化与 BF16 计算的验证

| 测试及输出目录 | 输入和观测 |
| --- | --- |
| `scale_equivalence_cpu` | 独立 CPU 实验用三次真实投影比较前置和后置 scale，并对比 K32 和贯穿全 K 的累加。报告记录 BF16 可表示性及分块误差。 |
| `isolated_grouped_wkv/k5120_n32` | LPU32，seq32，真实 Wkv 前 32 个输出、完整 K=5120。fixture 提供已解码的 BF16 X 和 W。设备计算 160 个 K32 部分和，以 159 次 FP32 加法有序折叠，最后转 BF16。分别检查 FP32 和 BF16 输出。 |
| `isolated_activation_scales` | C++ 将 `block_fp8_scale` 映射到现有 VXM。同一程序分别读入 X、归一化 Q、RoPE 后 KV 的第一个 K32 及边界输入，逐位检查每组的比例。输出是 FP32 表示的二次幂。 |
| `isolated_activation_qdq` | C++ 将 `block_fp8_qdq` 映射到现有 VXM。程序从 BF16 输入导出动态比例，按 E4M3FN 做 RNE，再乘回比例。观察量是 BF16；原始 E4M3FN 码和 UE8M0 字节仍须在 MXM 端口另行验收。 |

```powershell
python compiler/tools/deepseek_v41_scale_equivalence.py
python compiler/tests/deepseek_v41_grouped_wkv_test.py --package $case --target-config $target --build-dir $build
python compiler/tests/deepseek_v41_activation_groups_test.py --package $case --target-config $target --build-dir $build --output scale
python compiler/tests/deepseek_v41_activation_groups_test.py --package $case --target-config $target --build-dir $build --output qdq
```

## 连续 FP8 投影（2026-11-10）

`block_fp8_projection` 由 C++ 图规划器分成动态激活 QDQ、N64/K32 MXM、
FP32 有序累加和最终 BF16 RNE。MXM 部分和直接按 FP32 四字节平面写出；
累加算子显式要求完整 FP32 VXM 读取。输入按紧凑布局一次读入，每个量化分组
用于全部 N64 输出块后即可释放。Python 仅提供原始权重、输入和独立参考。

以下通过项均为 batch1、seq32、真实第 0 层 Wkv 的子矩阵，没有持久状态，
不包含 RMSNorm、RoPE、SWA 或 FFN。输出要求逐 BF16 位匹配，`host_operations=0`。

| 目录（相对测试根目录） | 验证范围 | 数值结果 | 含首次读入的周期 | 诊断回读周期 |
| --- | --- | --- | ---: | ---: |
| `fp8_iw_wkv/kgroup_0/projection_only/fp32` | 单 K32 FP32 部分和；已 QDQ 的 BF16 激活 fixture | 2048 个 FP32 值逐位一致 | 438 | 551 |
| `fp8_wkv_projection/k64_n64` | 两个 K32，设备 QDQ 与有序累加 | 2048 个 BF16 值逐位一致 | 179065 | 551 |
| `fp8_wkv_projection/k64_n128` | 两个 K32、两个 N64；紧凑输入、分组即时消费 | 4096 个 BF16 值逐位一致 | 182678 | 681 |

完整 `K5120/N512` 尚未取得运行数值验收，不能用这些子矩阵结果替代。
两个 K64 用例的形状和排程不同，表中周期不能作为同条件优化前后比较。
每个目录都保存 `runtime.pipeline.csv`、`actual.runtime.ftlpu` 和
`runtime_icu_programs/index.csv`；pipeline viewer 用 `?csv=/results/.../runtime.pipeline.csv` 打开。

```powershell
python compiler/tests/deepseek_v41_fp8_projection_test.py --package $case --target-config $target --build-dir $build --k 64 --n 128
# 完整形状：
python compiler/tests/deepseek_v41_fp8_projection_test.py --package $case --target-config $target --build-dir $build
```
