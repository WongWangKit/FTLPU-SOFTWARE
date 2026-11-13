# DeepSeek V4.1 Flash 适配计划

2026-11-11 精度方案已调整为 **官方 FP8 权重/scale 在 MXM IW 反量化为 BF16，
激活直接使用 BF16，FP32 累加**；KV Cache 也直接保存 BF16，不再执行 FP8 QDQ。
整段脚本默认 `--precision-policy bf16-activation`，可用 `official-fp8` 显式选择旧方案。
以下已有数值记录属于旧的官方激活 QDQ 路径，不能作为新方案的设备验证结果。
用户已停止整段仿真；后续按下文输入投影、归一化、Q_B/RoPE、attention 核、输出投影
的顺序逐步验证。当前没有重新启动整段运行。

截至 2026-11-11，在旧 FP8 激活方案下已建立第 0 层纯 SWA 的 Q/KV 数值参考，完整真实 Wkv
（包含设备动态量化）、两个 RMSNorm、partial RoPE、64-head 带 sink 的公共
attention 核及分组输出投影原语均已通过 LPU32 raw-macro 数值测试。
从输入经 Q/KV、attention 到输出投影和有效 KV 缓存的完整串接尚未通过验收，
不能据局部结果宣称完整 SWA 或模型已部署。

以 `deepseek-ai/DeepSeek-V4.1-Flash-1017` 为第一版固定检查点，revision
`1bae0e1088993b9b82de08f59de5be5a34e42170`。配置、参考源码摘要和逐层清单见
`results/deepseek_v41_flash_assessment/`。Q/KV 阶段按 safetensors 字节范围提取的
8 个权重及 scale 张量共 51,171,584 字节；完整 SWA 另导入 `wo_a`、`wo_b`、
对应 scale 及 64 个 sink，共 75,571,456 字节。两部分合计 126,743,040 字节，
来源与 SHA256 分别保存在 Q/KV、SWA 测试目录的 `weights/provenance.json`。

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

## 连续 FP8 投影的拆分基线（2026-11-10）

原 `block_fp8_projection` 由 C++ 图规划器分成动态激活 QDQ、N64/K32 MXM、
FP32 有序累加和最终 BF16 RNE。MXM 部分和直接按 FP32 四字节平面写出；
累加算子显式要求完整 FP32 VXM 读取。输入按紧凑布局一次读入，每个量化分组
用于全部 N64 输出块后即可释放。Python 仅提供原始权重、输入和独立参考。

以下是收回公共 projection 路径之前的基线结果，均为 batch1、seq32、真实第 0 层 Wkv 的子矩阵，没有持久状态，
不包含 RMSNorm、RoPE、SWA 或 FFN。输出要求逐 BF16 位匹配，`host_operations=0`。

| 目录（相对测试根目录） | 验证范围 | 数值结果 | 含首次读入的周期 | 诊断回读周期 |
| --- | --- | --- | ---: | ---: |
| `fp8_iw_wkv/kgroup_0/projection_only/fp32` | 单 K32 FP32 部分和；已 QDQ 的 BF16 激活 fixture | 2048 个 FP32 值逐位一致 | 438 | 551 |
| `fp8_wkv_projection/k64_n64` | 两个 K32，设备 QDQ 与有序累加 | 2048 个 BF16 值逐位一致 | 179065 | 551 |
| `fp8_wkv_projection/k64_n128` | 两个 K32、两个 N64；紧凑输入、分组即时消费 | 4096 个 BF16 值逐位一致 | 182678 | 681 |

旧拆分版完整 `K5120/N512` 仿真已取消，未取得数值验收；其曾因 1280 个逐页
barrier 超过 LPU32 的 1024 项容量而被拒绝，后续成对等待的版本也不能作为整尺寸
通过证据。新路径改为公共 projection，完整 K 保留在 MXM ACC 中，按页容量拆 N；
完整形状的 projection 原语已按 `K5120/N384` 和 `K5120/N128` 两个投影通过
FP32 数值验收；它的激活已由 fixture 做 QDQ，不能替代包含设备动态激活 QDQ 的
完整 Wkv 验收。后者先前遇到的 raw-macro iMEM 容量错误已修复，现已通过编译
和容量校验并进入真实执行，尚未完成数值验收。
旧 `K64/N128` 成对分页复测曾通过：4096 值逐位一致，执行 epoch 由 5 降到 3，
含输入周期仍为 182678；该中间结果在 `before_paired_pages.summary.json`。
公共投影改造前的最终基线另存于 `before_shared_projection.summary.json`。

运行前的 common-drain 指令域和 SRAM 队列描述现在只解码一次；每一页仍执行
原有安全检查。对成对分页改动之前的同一 `K64/N128` 程序，缓存优化前后的
已链接 ICU 镜像逐字节一致（SHA-256：
`1301c7eebaeafeb754873e04dfeb1bfa4a417e696c9d14ff412fa53b14b7bcfb`）。
`c2c_weight_ping_pong_test` 和 `runtime_common_barrier_test` 已通过。
`model_session_c2c_weight_pages_test` 的跨层完全隐藏预取断言仍失败，未修改的
分页代码也复现同一失败；不能将本轮回归报告为全部通过。

两个 K64 用例的形状和排程不同，表中周期不能作为同条件优化前后比较。
每个目录都保存 `runtime.pipeline.csv`、`actual.runtime.ftlpu` 和
`runtime_icu_programs/index.csv`；pipeline viewer 用 `?csv=/results/.../runtime.pipeline.csv` 打开。

```powershell
python compiler/tests/deepseek_v41_fp8_projection_test.py --package $case --target-config $target --build-dir $build --k 64 --n 128
# 完整形状：
python compiler/tests/deepseek_v41_fp8_projection_test.py --package $case --target-config $target --build-dir $build
```

## 复用公共 projection 与 MXM 两级累加

先前将每个 K32 部分和写到 MEM，再用 VXM 做跨 K 累加，并不是保留官方分组
精度的硬件必需条件。检查 CModel 的 WS 计算语义后，原先认为普通 projection
会把所有 K 项直接放进一个 FP32 点积累加器的判断需要纠正：MXM 本身已有两级累加。

1. 阵列内部的临时部分和按一块权重独立计算。LPU32 的四个 superlane 各处理
   八项，完成一个 K32 点积；新计算 wave 或行地址回绕会重置这一级。
2. 完成的 K32 部分和再通过 `MxmAccumulator::accumulate` 加到持久 FP32 ACC。
   `Sram` 目标指的是 MXM 内部 ACC SRAM，不是写回 MEM slice；普通 K 块不导出或清零。
3. 只在最后一个 K 块把完整结果以 BF16 RNE 或测试用 FP32 输出到 SR，并清零对应 ACC。

对应实现位于 `FTLPU-CMODEL/include/ftlpu/mxm/mxm.hpp` 的
`reset_buffer_state`、`compute_column_block`、`emit_column_output`，以及
`FTLPU-CMODEL/src/mxm/accumulator.cpp` 的 `accumulate`。
公共 `ProjectionGemmSchedule` 的 K 循环可以复用这一行为；不需要为了 FP8
分组语义另外插入每 K32 的 MEM 部分和与 VXM 加法节点，也不新增硬件能力。
权重的每 N32/K32 scale 由公共 linear projection 产生相应的 Dequant 指令，
与每次 IW 一一匹配；同一队列的不同 ICU 指令依次执行。

跨 K 累加的顺序保持从小到大。把官方二次幂 scale 前置到 BF16 输入的等价性，
仍受已验证的数值范围约束：反量化值须能由 BF16 精确表示，不能发生上溢、下溢或
FTZ。新路径已通过下列小形状复测、完整形状的 projection 原语测试，以及包含设备
动态激活 QDQ 的完整 Wkv；后者 16384 个 BF16 输出与独立参考逐位一致。

完整设备 QDQ＋Wkv 已生成 163 个图节点，其中 2 个是公共 `linear` projection，
配置分别为 `M32/K5120/N384` 与 `M32/K5120/N128`；第二个投影复用第一个投影
的输入布局，输出归档拷贝为 0。静态 raw-macro 程序为 14118191 字节。
此前 SXM iMEM 超限的原因是多个 producer 的输入 materialization 合并到一个
指令段；现在每个串行 producer 完成 materialization 后按已声明边界 drain，
不增加张量拷贝，也不放宽 iMEM 或其他硬件限制。全量 Wkv 已完成 raw-macro 执行，
实际 `runtime.pipeline.csv`、`actual.runtime.ftlpu` 和 `runtime_icu_programs` 均已导出。
调用 13785660 周期，首次输入 7928 周期，诊断回读 4136 周期，页等待 0，
`host_operations=0`。设备动态量化当前占据主要成本；该 Wkv 结果仍不代表完整 SWA 通过。

| 公共路径用例（相对测试根目录） | 数值验收 | 含首次读入的周期 | 诊断回读周期 | 范围 |
| --- | --- | ---: | ---: | --- |
| `fp8_wkv_projection/k64_n64/projection_only/fp32` | 2048 个 FP32 值逐位一致，转 BF16 后与官方参考也一致 | 1260 | 555 | fixture 已 QDQ，仅验证原始 FP8 权重 IW 与跨 K32 的 MXM FP32 ACC |
| `fp8_wkv_projection/k64_n128/projection_only/fp32` | 4096 个 FP32 值逐位一致，转 BF16 后与官方参考也一致 | 1439 | 1066 | fixture 已 QDQ，覆盖跨 N64 输出组的 FP32 布局 |
| `fp8_wkv_projection/k5120_n512/projection_only/fp32` | 16384 个 FP32 值逐位一致，转 BF16 后与官方参考也一致 | 79259 | 4136 | 完整真实 Wkv 权重；fixture 已 QDQ，两个完整 K5120 的 N 分页 |
| `fp8_wkv_projection/k64_n128` | 4096 个 BF16 值逐位一致 | 173066 | 1065 | 设备动态激活 QDQ、FP8 IW、完整 K64 投影与最终 BF16 输出；最终 ingress 排程 |
| `fp8_wkv_projection/k5120_n512` | 16384 个 BF16 值逐位一致 | 13793588 | 4136 | 完整设备动态激活 QDQ、真实 Wkv FP8 权重、两个完整 K5120 的 N 页 |

正式 `K64/N128` 与上表拆分基线使用相同形状、权重、输入及目标配置。含首次
读入的时延由 **182678 降至 173066 周期（减少 5.26%）**；诊断回读由 681
变为 1065 周期，单独报告，不能混入计算改进。图从原来的 11 个节点降为 4 个，
其中只有一个原生 `linear` projection，配置为 `M32/K64/N128`，不再生成逐 K32
部分和累加或独立最终舍入图节点。权重页从 4 页合为 1 页，总 payload 仍为 8192
字节；本例实际权重等待均为 0，`host_operations=0`。
最新结果使用 `tmp/shared_projection_ingress_snapshot`：输入 materialization 在每个
producer 完成后遵守其声明的 iMEM drain 边界，避免把多个 producer 的 SXM 指令
堆入一个 iMEM 段。相比首次公共投影版的 172682 周期增加 384 周期，但没有新增
张量拷贝，4096 个 BF16 输出仍逐位一致，图结构与上述分页数量不变。

可分别查看 [正式 K64/N128 runtime pipeline](http://127.0.0.1:8765/tools/pipeline_viewer/index.html?csv=/results/deepseek_v41_flash_layer0_qkv_prefill/fp8_wkv_projection/k64_n128/runtime.pipeline.csv)、
[K64/N64 FP32 原语 runtime pipeline](http://127.0.0.1:8765/tools/pipeline_viewer/index.html?csv=/results/deepseek_v41_flash_layer0_qkv_prefill/fp8_wkv_projection/k64_n64/projection_only/fp32/runtime.pipeline.csv)、
[K64/N128 FP32 原语 runtime pipeline](http://127.0.0.1:8765/tools/pipeline_viewer/index.html?csv=/results/deepseek_v41_flash_layer0_qkv_prefill/fp8_wkv_projection/k64_n128/projection_only/fp32/runtime.pipeline.csv)
和 [完整 Wkv projection 原语 runtime pipeline](http://127.0.0.1:8765/tools/pipeline_viewer/index.html?csv=/results/deepseek_v41_flash_layer0_qkv_prefill/fp8_wkv_projection/k5120_n512/projection_only/fp32/runtime.pipeline.csv)。
各目录仍保存实际 `actual.runtime.ftlpu`、`runtime_icu_programs/index.csv` 与数值摘要。

`K64/N128/projection_only/fp32` 的首次运行发现结果布局把中间两个 N32 块交换：
实际块顺序为 `[0,64,32,96]`，各块内容与对应参考逐 FP32 位一致，属于输出布局
映射错误。修正为原生 FP32 半球交错布局后，用原参考数据与原阈值复测已通过，
上表为修正后的结果；没有通过调整参考输出顺序绕过问题。

完整 Wkv projection 原语保持两个公共 `linear` 图节点，分别为
`M32/K5120/N384` 与 `M32/K5120/N128`，共 2 页、2621440 字节 FP8 权重；
无逐 K32 图节点，无中间结果归档拷贝或计算阶段 MEM→MEM 搬运。
实际调用为 63642 周期，其中逻辑执行 42560 周期、运行时页等待 21082 周期；
首次输入另计 15617 周期，诊断回读另计 4136 周期，`host_operations=0`。

从实际 runtime CSV 展开重复模式并合并区间，两侧 MXM Compute 各执行 40960 周期，
区间均为 `[36869,67589)` 与 `[67951,78191)`。排除首次输入、页等待和诊断回读后，
MXM 利用率为 `40960/42560 = 96.24%`；若保留调用内页等待则为 64.36%，
再包含首次输入则为 51.68%。这些数值只描述该 projection 原语，不能作为包含
设备激活 QDQ 的完整 Wkv 利用率；原语没有执行 VXM 量化。

公共 INT8 projection 回归也已运行，固定目录为
`results/deepseek_v41_shared_projection_int8_regression`。`M64/N128/K128` 连续计算
与关闭权重/激活 overlap 两种配置均通过，NRMSE 为 0.0029245157，输出逐位一致；
含首次读入分别为 2592、3023 周期，诊断回读均为 2084 周期，`host_operations=0`。
这验证的是公共 linear projection，不等于完整 dense attention 或 decoder 回归。
分页复用用例保存在 `results/deepseek_v41_shared_projection_page_reuse`，使用最终
快照 `tmp/shared_projection_final_snapshot` 复测已通过。两组不同输入各做
`M1/K128/N256`，各 256 个输出与独立参考完全一致；每组四个 N 页的输入复用
标记均为 `[false,true,true,true]`。运行调用为 5726 周期，含首次读入 6008 周期，
诊断回读 198 周期，8 次权重预取，0 运行时权重等待和 0 host operations。
该目录的 `summary.json` 保存命令、程序摘要与各组数值结果，实际 ICU 指令已导出。

测试脚本新增结构验收：从 `graph_plan.json` 的 `native_projections` 读取真实
`kind/m/n/k/weight_dtype/rhs_scale_block_count`，确认每个公共 linear projection
保持完整 K，各页的 N 合计覆盖输出，scale 表覆盖每个 N32/K32 块。大权重可按
真实页容量拆 N，但不允许恢复每 K32/N64 一个图节点、再由 VXM 归并的旧方式。
`stages.json` 保存各阶段与统计，并检查旧的部分和累加、独立最终舍入节点消失。
图级搬运数量、字节平面向量数与程序大小也记录在 `summary.json.structure`。

另有明确标注的 `--projection-only` 原语测试：Python 只为该原语提供已经完成 QDQ
的 BF16 输入，原始 FP8 权重与 scale 仍交给设备；这不能替代正式的设备激活量化测试。
可同时加 `--fp32-output`，在最终 BF16 舍入之前，逐 FP32 位对比独立的
“K32 临时部分和 → 有序 FP32 总和”参考，避免 BF16 舍入掩盖跨 K 累加错误。
该参考还必须在转 BF16 后与原官方分组参考逐位一致。原语结果单独放在
`fp8_wkv_projection/k{K}_n{N}/projection_only[/fp32]`。

```powershell
# 快速核对公共 projection 的完整 K 累加；不计入设备激活 QDQ 验收。
python compiler/tests/deepseek_v41_fp8_projection_test.py --package $case --target-config $target --build-dir $build --k 64 --n 128 --projection-only --fp32-output
# 正式路径：仍由设备进行动态激活 QDQ。
python compiler/tests/deepseek_v41_fp8_projection_test.py --package $case --target-config $target --build-dir $build --k 64 --n 128
```

## 第 0 层 causal encoder SWA：真实权重整段测试

固定目录为 `results/deepseek_v41_flash_layer0_swa_prefill`。模型仍使用上文固定的
Flash-1017 revision；本轮范围是 batch1、seq32、起始位置 0、空初始缓存的 **attention**，
不是完整 decoder 层或完整模型。输入是 BF16 `[32,5120]` 的可复现层输入 fixture，
不包含 tokenizer、embedding 或 attention 外的层归一化、mHC、残差、MoE。
输出是 BF16 `[32,5120]` 和 `[32,512]` 有效 window KV；后者按官方缓存数值
执行 post-RoPE group32 FP8 QDQ，以反量化后的 BF16 保存。未覆盖 decode ring-buffer
续接和超过 128-token window 的淘汰；当前 lowering 对越窗形状明确拒绝。
目标为 `ftlpu-lpu32.json`：500 MHz，DDR 峰值 51.2 GB/s，调度效率 90%，
保留配置内的读写延迟与抖动；本轮未修改这些硬件或外存参数。

整段逻辑图为：

1. `Wq_a(5120→1280) → RMSNorm → Wq_b(1280→32768) → 64-head partial RoPE`。
2. `Wkv(5120→512) → RMSNorm → partial RoPE → group32 FP8 QDQ`，同一 latent 用作 K 和 V。
3. 原生公共 dense attention：QK、causal mask、带 per-head sink 的 softmax、PV。
   D512 按独立的两头一组调度，保留完整 D 归约；sink 只加入分母，不携带 V。
4. 对 context 最后 64 维做 inverse adjacent-pair RoPE。
5. `wo_a` 八组独立 `4096→1024`，再 `wo_b(8192→5120)`。

所有 projection 复用 `ProjectionGraphLowering.inc` 的完整 K 原生 linear 路径、
DDR/C2C N 分页与 MXM FP32 ACC。分组 `wo_a` 在 C++ 中拆独立组，使用原始 FP8
权重/scale，经 IW 反量化得到官方 BF16 权重值；激活保持官方 group32 QDQ。
M32 动态量化采用每个 lane 对应一个 token 的映射：K32 分组转为 channel-major，
使用现有 VXM axis-zero max 得到 32 个独立 scale，后续 FP32 算术保留 token lane，
输出恢复逻辑 token-major 视图。量化公式及舍入位置不变。其他受支持形状保留标量
路径，并按真实 bank 行容量分批打包、释放广播中间结果，避免大 K 超出 MEM。
没有扩容 MEM/iMEM，也没有增加 CModel 模型算子。
RoPE 常量表仅初始化一次，两路同时输入 VXM 时由公共规划器排除 MEM ICU 冲突。

验收阈值在运行前固定：整段输出及 KV relative L2 < 3%、cosine > 0.999；
attention 核 relative L2 < 1%；RoPE relative L2 < 0.1% 且未旋转区域逐位不变。
独立输出投影/量化原语要求 BF16 逐位一致。当前局部结果如下，**不替代整段验收**：

| 测试 | 数值结果 | 调用周期 | 首次输入 | 诊断回读 |
| --- | --- | ---: | ---: | ---: |
| `rope_regression`，真实 Q 两头，正向与反向 | 两个输出均逐位一致 | 26052 | 4914 | 10315 |
| `core_real_weights`，64 头、官方 post-RoPE KV 精度及真实 sink | relative L2 0.0026390746 | 332900 | 103828 | 98347 |
| `qdq_core_regression`，两头真实 fixture，设备 KV QDQ → 公共 attention | relative L2 0.0024627956 | 91731 | 4901 | 7204 |
| `grouped_projection_regression`，两组真实 `wo_a` 切片、10 个独立 K32 量化组 | 投影及 QDQ 均逐 BF16 位一致 | 74604 | 1630 | 3658 |
| `projection_transport_regression`，MXM 交错半球输出接 VXM 归约及 QDQ | 投影与再次量化逐 BF16 位一致，归约误差 0 | 48199 | 383 | 2299 |
| `norm_transport_regression`，distributed16 RMSNorm 直接接 QDQ、经 partial RoPE 接 QDQ | 两条 QDQ 对其实际输入均逐 BF16 位一致；端到端 relative L2 分别 0.0037598604、0.0036735749 | 174922 | 3919 | 13382 |
| `dense_regression`，原有无 sink 公共核 | padding、causal offset、scale 数值通过；非对齐位置按预期拒绝 | 各子目录记录 | 单列 | 单列 |

整段状态以根目录 `summary.json` 为准；`preparing`、`compiled_only`、`executing`
均不能声称数值通过。仅在实际 CModel 运行完成后导出根目录 `runtime.pipeline.csv`、
`actual.runtime.ftlpu` 和 `runtime_icu_programs/index.csv`。局部用例各自保存实际流水图。

上述量化映射的同条件原语比较保存为 `grouped_projection_regression/baseline.summary.json`
与 `summary.json`：调用周期从 1202396 降至 74604（减少 93.80%），首次输入相同；
恢复 feature-lane 输出布局后的诊断回读由 2698 变为 3658 周期，单独列出。
两种映射的投影与量化结果均逐 BF16 位一致。此比较包含两组
小投影及独立 QDQ，不能作为完整 SWA 的加速比。整份程序的 VXM/SXM 指令超出
单次 iMEM 容量时，使用已有编译器 drain 边界分段装载，每段仍接受真实容量检查。

计时边界：本测试使用现有 ModelSession 的直接指令装载入口。runtime CSV 记录
真实执行的 FU 行为、首次数据读入和调用内 DDR/C2C 权重等待；诊断回读另列。
分段 iMEM 镜像写入当前没有建模为计时的 DDR/C2C 指令 DMA，不能把本测试的
`cold_cycles_through_compute` 称为包含指令装载和主机编译的部署总 TTFT。
`instruction_prime_cycles` 也只是 ICU 前端预热统计，不是完整镜像的搬运时间。
所有分段边界均检查当前 ICU 队列和 VXM 已 drain，不放宽 iMEM 容量或指令交错约束。

ICU 导出使用 `--compute-imem-segments`，通过与 runtime 相同的 C++ 分段规划器
重建每个 drain 的镜像；VXM/SXM 行的 `phase=execution_segment`、`load_id` 和
`pc_word` 标识装载段及段内 PC，`raw_words` 保留重定位后的 wait 编码。
CSV 的周期列仍是全局逻辑周期，动态 C2C 等待以 runtime pipeline 为准。
`compute_imem_segments.csv` 保存段边界及容量校验；该导出是运行时链接镜像的
确定性重建，不是额外采集的硬件取指 trace。小型 QDQ→core 用例共 17 段，
5 个被分段的队列在还原全局周期后与原队列的所有非 NOP 指令一致，
总 FU issues 均为 41196。

串接回归还发现并修复了公共 attention 的初始化常量生命周期问题：causal mask
在运行开始前上传，却原先使用节点内的临时地址，可能被前序 VXM 算子覆盖。
现在由整图规划器分配跨整段生命周期的只读地址，重算原生读取排程；兼容的
串行 attention 节点复用同一份 mask。接口回归中去掉 causal mask 会使参考变化
81.77%，修复后实际误差为 0.246%，原有无 sink 的公共核回归也再次通过。

首次完整执行暴露了另一处连接错误：distributed16 → token-lane QDQ 是矩阵转置，
原普通搬运分支却只选取源向量的 lane 0 来发整向量复制，造成非旋转前缀错误。
完整输出/KV relative L2 分别为 1.1422/1.1676，该次失败摘要保留为
`failed_before_distributed_transpose.summary.json`，不能作为数值通过的结果。
短测试进一步确认 RMSNorm 和 RoPE 本身正确，错误首次出现在 QDQ 的输入搬运。
公共 wide-SXM 搬运现在直接从 distributed16 的八个 byte-plane pair 各读取
4 个向量，按间隔 8 周期交错到同一个 32×32 转置块；没有额外 planar MEM 中转。
普通 distributed 搬运也增加 lane 不变性检查，防止以后把转置静默当成复制。
修复后的上述短测试及 projection/QDQ→attention 回归均通过。第二次完整串接已由用户
中止，最后报告的逻辑进度为 900000/12014089；没有完整数值结果。根目录
`runtime.pipeline.csv` 仍是第一次失败运行的旧产物，不能作为修复通过的证据。

```powershell
$build = 'tmp/deepseek_swa_token_lanes_snapshot'
$target = '../FTLPU-CMODEL/config/ftlpu-lpu32.json'
$qkv = 'results/deepseek_v41_flash_layer0_qkv_prefill'
$swa = 'results/deepseek_v41_flash_layer0_swa_prefill'
# 历史 FP8 激活方案的复现命令；当前不启动整段运行。
python compiler/tests/deepseek_v41_swa_test.py --build-dir $build --target-config $target --package $qkv --output-dir $swa --stage full --precision-policy official-fp8
# --prepare-only 仅编译；--reuse-prepared 校验 source/target/executable/program 摘要后复用已编译程序。
python compiler/tests/deepseek_v41_native_rope_test.py --build-dir $build --target-config $target --reference-dir "$qkv/reference" --output-dir "$swa/rope_regression"
python compiler/tests/deepseek_v41_swa_core_test.py --build-dir $build --target-config $target --output-dir "$swa/core_real_weights" --heads 64 --reference-dir "$qkv/reference" --sink-file "$swa/weights/attn_sink.bin" --precision-policy official-fp8
python compiler/tests/deepseek_v41_swa_core_test.py --build-dir $build --target-config $target --output-dir "$swa/qdq_core_regression" --heads 2 --reference-dir "$qkv/reference" --sink-file "$swa/weights/attn_sink.bin" --device-kv-qdq
python compiler/tests/deepseek_v41_swa_projection_test.py --build-dir $build --target-config $target --package $swa --output-dir "$swa/grouped_projection_regression"
python compiler/tests/deepseek_v41_norm_transport_test.py --build-dir $build --target-config $target --package $qkv --output-dir "$swa/norm_transport_regression"
```

## BF16 激活方案：第 1 步只接两个输入投影

固定目录：`results/deepseek_v41_flash_layer0_swa_prefill/input_projections`。
输入为同一个 BF16 attention 输入 `X[32,5120]`；使用完整真实 `Wq_a[1280,5120]`
和 `Wkv[512,5120]` 及官方 E8M0 block32 scale，输出分别为 BF16
`Q_A[32,1280]` 与 `KV_raw[32,512]`。这一阶段没有持久状态，也不包含 RMSNorm、
RoPE、Q_B、attention、mHC 或 FFN。

公共 `block_fp8_projection` 新增明确的 `quantize_activation=false` 语义，
直接调用原来的公共 projection lowering，跳过激活 QDQ；省略此属性时仍保留旧行为。
权重仍由 DDR/C2C 按物理容量分页进入设备、在 MXM IW 反量化，Python 不生成
反量化后的设备权重或物理排程。独立 CPU 参考按照 BF16 权重/激活、K32 FP32
部分和、有序 FP32 总和和最终 BF16 RNE 生成；设备验收要求两个输出逐 BF16 位一致。
同时检查完整 K5120、N 页覆盖，以及图中不存在激活 QDQ 节点。

第 1 步脚本默认仅编译，必须显式添加 `--run` 才启动 CModel。`--emit-only`
只生成逻辑图和 CPU 参考。没有运行时不能声称有新的 runtime pipeline 或数值通过。
当前已通过编译及结构检查：6 个原生 projection 节点，Q_A/KV 分别 4/2 个完整 K5120
权重页，激活 QDQ 节点为 0，输出归档拷贝为 0，结果状态为 `compiled_only`。
小型分组投影编译检查确认 `false` 不生成 QDQ、省略属性保留旧 QDQ；错误属性类型
被拒绝。编译器及脚本检查通过，但本次没有执行 CModel。
CPU 新参考与官方 FP8 激活参考的相对 L2 差异分别为 Q_A 2.6790%、KV_raw 2.6882%；
这描述精度策略差异，不是设备计算误差，也不代表模型质量评估。

```powershell
$build = 'tmp/deepseek_swa_bf16_snapshot'
$target = '../FTLPU-CMODEL/config/ftlpu-lpu32.json'
$qkv = 'results/deepseek_v41_flash_layer0_qkv_prefill'
$step1 = 'results/deepseek_v41_flash_layer0_swa_prefill/input_projections'
# 只编译和检查图结构，不启动仿真。
python compiler/tests/deepseek_v41_swa_input_projection_test.py --build-dir $build --target-config $target --package $qkv --output-dir $step1
# 后续允许设备测试时，在同一命令末尾加 --run。
```

后续依次连接：Q_A/KV 两路 RMSNorm → Q_B 与 Q/KV partial RoPE → 带 causal mask
和 sink 的公共 SWA 核 → inverse RoPE、分组 O_A 与 O_B。每一步保留独立输出验收和
实际 runtime pipeline，再向后串接；不自动启动下一步。

## mHC 后的第 1 步：attention RMSNorm

2026-11-11 按用户分工，mHC 由另一个对话实现；本分支从 `hc_pre` 输出接口开始。
注意这里是投影之前的 `layers.0.attn_norm`，不是 Q_A/KV 投影之后的两个 RMSNorm。
固定目录为 `results/deepseek_v41_flash_layer0_attention_rmsnorm`。

输入/输出均为 BF16 `[32,5120]`（batch1、seq32），无持久状态。当前输入是独立
`hc_pre` 输出 fixture，来自现有 `reference/x.npy`，本测试没有执行 mHC，也未证明
与另一个对话的 mHC 已串接。测试通过 `--input-npy` 接收这个明确的逻辑接口。
gamma 是固定官方检查点的原始 BF16 `layers.0.attn_norm.weight[5120]`，epsilon=1e-20。
参考使用独立 FP64 RMSNorm 公式并在输出做 BF16 RNE；设备使用原有公共 C++ RMSNorm
lowering，BF16 输入/权重/输出、FP32 运算，无 FP8 激活 QDQ，无 CModel 模型算子。

已通过 raw-macro CModel 测试：relative L2 **0.0011902051**，cosine **0.9999992956**，
最大绝对误差 0.00048828125；阈值预先固定为 relative L2 < 0.01、cosine > 0.9999。
图中仅 1 个原生 RMSNorm 节点，图间 MEM 搬运 0、输出归档拷贝 0。
C2C 输入直接使用编译器选定的普通 `fp16_pair_planar` 布局；输出和第二遍使用的
转置 x 缓存也全部使用 planar，不再使用 distributed16。实际运行 `host_operations=0`。

两遍仍有完整 hidden 维归约的依赖，但各遍内部直连：

1. `MEM → wide SXM → VXM 平方/求和`。在 SXM 到 VXM 的 SR 路径上用 MEM write-tap
   保留转置后的 x，供第二遍使用；第一遍不从这个缓存重新读 x。
2. 缓存 x 与 gamma 进入 VXM，乘归一化因子后直接 `VXM → wide SXM → 最终 MEM`。
   没有归一化结果的中间写回及重读。epsilon=1e-20 的 FP32 feedback / 分段 LUT
   处理与旧版相同，不改变数值策略。

新公共路径支持 wide SXM、32-token tile、hidden 为 32 的倍数，且一个 planar
重读缓存及其标量槽可放入 MEM bank。当前 LPU32 为 hidden ≤ 8160；尚未扩展为
任意 seq 的分块调度。旧路径保留兼容，可显式用 `ftlpu.norm_distributed16=true`
选择；本工作区后续新实现默认不选择 distributed16（见根目录 AGENTS.md）。

| 时段 | 周期 | 500 MHz 时延 |
| --- | ---: | ---: |
| 首次输入及 gamma 读入 | 31137 | 62.274 μs |
| 算子调用（权重页等待为 0） | 10782 | 21.564 μs |
| 首次读入至计算完成 | 41919 | 83.838 μs |
| 诊断输出回读，单列 | 71717 | 143.434 μs |

同输入、gamma、epsilon、硬件配置及 runner 比较：算子调用由 20994 降到 10782
周期，减少 48.64%；包含首次读入由 43706 降到 41919，减少 4.09%。普通 planar
集中于少量 slice，当前 C2C 输入和诊断输出的传输时间变长，因此不能声称整段
冷启动加速一倍；若把诊断回读也计入，总周期为 113636，旧版为 52749。
这是独立测试的数据传输开销，后续同图消费者应直接读取保留的 planar 输出。
旧版证据保存在同目录 `baseline_before_planar_streaming/`。

目标仍为 `ftlpu-lpu32.json` 的 500 MHz / DDR 峰值 51.2 GB/s。
时延口径沿用上文 ModelSession：包含数据读入，未计入指令镜像 DMA 或主机编译。
实际运行镜像为 `actual.runtime.ftlpu`（7628 字节），ICU CSV 位于
`runtime_icu_programs/index.csv`，无需 compute iMEM 分段。runtime CSV 为 924319 字节，
已有 viewer 的流式解析器检查通过（6652 patterns、21 resources）。packed MEM macro
的 CSV 按真实域显示读/写/write-tap，其可跨物理取指字边界，因此不虚构逐指令
PC；同目录 `*.icu.csv.imem.bin` 保留 relocated raw loader 使用的完整物理镜像。
MEM 导出共 20 条 packed 计算域（另含首次读入的同步指令），已检查不再出现
无法解释的 `EXTENDED` 行。该修复仅解码和导出，不改变设备执行。

补充回归通过：BF16 hidden=32/96（epsilon=1e-20，含零行及小/大方差行）、
BF16 hidden=512（epsilon=1e-5）和 FP16 hidden=96（epsilon=1e-5）。最差逐行
相对 L2 为 0.3651%，零行输出全零。FP16 通过原生 typed Kernel API 测试，
并非扩展逻辑模型导入器的 dtype 支持。结果保存在 `regression/summary.json`。

[查看实际流水图](http://127.0.0.1:8765/tools/pipeline_viewer/index.html?csv=/results/deepseek_v41_flash_layer0_attention_rmsnorm/runtime.pipeline.csv&v=planar_streaming_20261111&start=31137&end=41919)

```powershell
python compiler/tools/import_deepseek_v41_attention_norm.py --cached-index tmp/deepseek41_official_index.json --output results/deepseek_v41_flash_layer0_attention_rmsnorm/weights
python compiler/tests/deepseek_v41_attention_norm_test.py --build-dir tmp/deepseek_swa_planar_norm_snapshot --target-config ../FTLPU-CMODEL/config/ftlpu-lpu32.json --output-dir results/deepseek_v41_flash_layer0_attention_rmsnorm --input-npy results/deepseek_v41_flash_layer0_qkv_prefill/reference/x.npy --model-config results/deepseek_v41_flash_assessment/official_config.json
python compiler/tests/rmsnorm_planar_streaming_test.py --build-dir tmp/deepseek_swa_planar_norm_snapshot --target-config ../FTLPU-CMODEL/config/ftlpu-lpu32.json --output-dir results/deepseek_v41_flash_layer0_attention_rmsnorm/regression
```

下一步在同一 C++ 逻辑图中连接 `attention RMSNorm → Q_A / KV`，继续使用 BF16 激活
和原始 FP8 权重；不把 CPU 参考或诊断回读当成设备算子之间的数据通路。

## mHC 后的第 2 步：RMSNorm 连续接 Q_A / KV

2026-11-11 已完成同图 raw-macro CModel 数值测试。固定目录为
`results/deepseek_v41_flash_layer0_attention_input_prefill`，输入是上一节相同的
BF16 `hc_pre` 接口 fixture `[32,5120]`（batch1、seq32），没有执行 mHC。
输出为 Q_A `[32,1280]`、KV `[32,512]`；另返回 RMSNorm `[32,5120]` 供局部验收。
无持久状态。本步不包含后续 Q_A/KV RMSNorm、Q_B、RoPE、SWA 或 FFN。

复用公共 RMSNorm、block-FP8 projection lowering 和图内存规划。Python 仅构造
逻辑图、导入真实权重并计算独立参考；设备算子之间没有 host 往返，实际运行
`host_operations=0`。所有输入、中间激活和输出均为 BF16，激活不做 FP8 QDQ。
权重保留原始 E4M3FN 和每 32×32 块 E8M0 scale，在 MXM IW 入口反量化为 BF16；
每 K32 的乘积部分和及后续累加使用 FP32，最终输出做 BF16 RNE。

编译器按完整 K=5120、分 N 列分页：Q_A 的 N 为 384/384/384/128，KV 为 384/128。
合计 7 个图节点（1 RMSNorm + 6 页投影），6 次实际权重预取，权重传输 9,175,040
字节，使用两个 bank 交替驻留。六页都直接读取 RMSNorm 的 `fp16_pair_planar`
输出（两半球、slice8/9、bank1、base0），没有重新布局或重新上传归一化结果。
本步输出、中间缓存均不使用 distributed16。

各页投影结果原本已在 `fp16_hemisphere_pair_planar` 中连续。新增公共输出视图
识别会逐元素证明地址、lane、半球、bank 和 slice 一致，再直接绑定完整逻辑输出，
避免为拼接分页结果生成 MEM→MEM 拷贝。证明失败时仍走原有合法导出路径，
不按测试名称或权重内容选分支。当前图间搬运、输出归档拷贝均为 **0**。

验收阈值事先固定：relative L2 < 1%、cosine > 0.9999；以设备实际 RMSNorm 输出
重算两路投影时，必须逐 BF16 位一致。结果：

| 输出 | 相对 L2 | cosine | 投影局部 BF16 不一致元素 |
| --- | ---: | ---: | ---: |
| attention RMSNorm | 0.11902% | 0.9999992956 | 不适用 |
| Q_A | 0.20628% | 0.9999978789 | 0 |
| KV | 0.21214% | 0.9999977577 | 0 |

相对端到端参考的误差包含 RMSNorm 设备实现的舍入差异；投影本身与约定的
BF16 激活、分 K32 FP32 累加参考完全一致。去掉输出拼接前后，全部三个输出
也逐字节一致。未改变数值阈值、硬件容量或 FU 延迟。

| 时段 | 周期 | 500 MHz 时延 |
| --- | ---: | ---: |
| 初始输入与 gamma 读入 | 31,137 | 62.274 μs |
| 计算调用，扣除权重页等待 | 156,203 | 312.406 μs |
| 运行时权重页等待 | 20,537 | 41.074 μs |
| 首次读入至计算结束 | 207,877 | 415.754 μs |
| 诊断回读，含中间 RMSNorm，单列 | 86,136 | 172.272 μs |

仍为 500 MHz / DDR 峰值 51.2 GB/s、调度效率 90%。计时不包含主机编译和
指令镜像 DMA。含诊断回读的总 trace 为 294,013 周期。
同输入、权重、硬件和 runner 对比，输出拼接优化前计算调用为 158,758 周期，
优化后减少 2,555 周期（1.61%）；冷启动从 210,432 减到 207,877 周期（1.21%）。
对比证据保存在同目录 `baseline_before_native_export/`。

实际执行镜像 `actual.runtime.ftlpu` 为 31,549 字节；ICU 导出在
`runtime_icu_programs/index.csv`，包含首次装载与实际运行镜像的指令。
packed MXM Load/Compute/Dequant 域现已正确解码，附完整 relocated 物理 iMEM
镜像和验证结果，不再误显示成单周期 `EXTENDED`。CSV 中域的周期为逻辑排程周期，
动态等待后的实际 FU 周期以 runtime pipeline 为准；packed 域可跨物理取指字，
故不虚构每域的独立 PC。没有 compute iMEM 分段。

runtime CSV 为 5,521,828 字节、44,096 patterns、36 个资源；已通过现有 viewer
流式解析器检查，viewer 与 CSV 的本地 HTTP 地址均返回 200。

[打开本步实际流水图](http://127.0.0.1:8765/tools/pipeline_viewer/index.html?csv=/results/deepseek_v41_flash_layer0_attention_input_prefill/runtime.pipeline.csv&v=norm_qa_kv_20261111)

```powershell
python compiler/tests/deepseek_v41_attention_input_test.py --build-dir tmp/deepseek_swa_attention_input_snapshot --target-config ../FTLPU-CMODEL/config/ftlpu-lpu32.json --package results/deepseek_v41_flash_layer0_qkv_prefill --norm-package results/deepseek_v41_flash_layer0_attention_rmsnorm --model-config results/deepseek_v41_flash_assessment/official_config.json --output-dir results/deepseek_v41_flash_layer0_attention_input_prefill
```

编译与 runner 的精确 SHA256 记录在 `summary.json`。本次保留与前一版相同的 runner
做比较；后续常规构建可以用正式 build 目录替换该快照路径。
下一步连接 **Q_A 的 RMSNorm（1280 维）和 KV 的 RMSNorm（512 维）**，然后再接
Q_B 与 partial RoPE；本次未自动执行这些后续步骤。

## mHC 后的第 3 步：接上 Q_A / KV 各自的 RMSNorm

2026-11-11 已跑通连续图 `attention RMSNorm → Q_A / KV projection → Q_A / KV RMSNorm`。
固定结果目录为 `results/deepseek_v41_flash_layer0_attention_input_norm_prefill`。
仍使用相同 hc_pre 接口 fixture `[32,5120]`（batch1、seq32）；最终输出为 BF16
`q_norm[32,1280]` 和 `kv_norm[32,512]`，无持久状态。为定位误差，还诊断回读
输入 RMSNorm、Q_A 和原始 KV，共返回 5 个张量；设备算子之间没有主机往返。
不包含 mHC、Q_B、partial RoPE、SWA、输出投影或 FFN。

新增两路直接调用公共 RMSNorm lowering。使用同一官方 revision 的原始 BF16
`layers.0.attn.q_norm.weight[1280]`、`layers.0.attn.kv_norm.weight[512]`，epsilon=1e-20。
投影仍用官方 FP8 权重按块反量化、BF16 激活；归一化使用 BF16 输入/gamma/输出、
FP32 运算，输出 BF16 RNE。全图不使用 distributed16，也没有激活 FP8 量化。
Python 只构图、导入数据和计算参考，布局及排程全部由 C++ 生成。

投影原生布局按 N32 在两半球交替存列，现有公共 RMSNorm 的两条镜像链各自需要
完整 hidden 维；仅改地址不能把另一半球的列变为本地数据。本次保留这一归约
策略，并优化其连接方式：

- 连续投影页经逐元素地址/lane/半球证明后作为一个完整视图处理，不逐页归档。
- 结果分配到比消费端更靠外的合法 MEM 列；一次 inward SR 读取，先 `write_tap`
  到本地 RMSNorm 输入，再继续到另一半球，去掉原先同列绕行和重复读取。
- 两路 gamma 由 C2C 直接写到生命周期及队列冲突检查通过的最终 broadcast 位置，
  不再先分段存放、执行前重新搬运。RMSNorm 两遍内部仍保持 SXM→VXM 和 VXM→SXM 流水。

因此图间传输从 **52 段减至 4 段**（Q/KV 各两条半球分发），读取向量从
**8,960 减至 1,792**，减少 80%；输出归档拷贝为 0。这里并非零 MEM→MEM：剩余
4 段用于当前镜像归约所需的数据分发。进一步消除它们需要改成分片归约并合并
半球部分和，不能仅把布局名称改成相同；本次没有改变归约算法或硬件契约。

数值阈值保持 relative L2 < 1%、cosine > 0.9999；既验收从 hc_pre fixture 开始的
端到端输出，也用实际投影输出重算局部 RMSNorm 参考：

| 输出 | 端到端相对 L2 | 局部 RMSNorm 相对 L2 | 端到端 cosine |
| --- | ---: | ---: | ---: |
| Q_A RMSNorm | 0.26009% | 0.13231% | 0.9999966183 |
| KV RMSNorm | 0.25290% | 0.09623% | 0.9999968056 |

Q_A/KV 投影对设备实际输入的局部 BF16 位差仍为 0。优化前后全部 5 个输出逐字节
一致，没有通过放宽阈值或更改精度接受结果。实际运行 `host_operations=0`。

同逻辑图、权重、输入、target 及 runner 对比（500 MHz / DDR 峰值 51.2 GB/s，
调度效率 90%；不包含主机编译与指令镜像 DMA）：

| 时段 | 优化前周期 | 优化后周期 | 优化后时延 |
| --- | ---: | ---: | ---: |
| 首次输入及全部 gamma 读入 | 32,630 | 36,640 | 73.280 μs |
| 计算调用，扣除权重页等待 | 172,076 | 162,800 | 325.600 μs |
| 权重页等待 | 20,592 | 20,567 | 41.134 μs |
| 首次读入至计算完成 | 225,298 | 220,007 | 440.014 μs |
| 诊断回读，单列 | 113,092 | 112,191 | 224.382 μs |

gamma 直接 broadcast 增加了初始 C2C 装载量，不能只看计算段改善。计入该开销后，
冷启动仍减少 5,291 周期（2.35%）；计算调用减少 9,276 周期（5.39%）。
含诊断回读的完整 trace 为 332,198 周期。基线证据在同目录
`baseline_before_native_layout/`，逐字节比较和时延对比在 `comparison.json`。

本图有 9 个原生计算节点，运行时 7 个执行 epoch、6 次权重预取，权重传输仍为
9,175,040 字节。`actual.runtime.ftlpu` 为 42,392 字节；实际 ICU program 的索引为
`runtime_icu_programs/index.csv`，包含首次装载、同步和 packed MEM/MXM 物理镜像。
没有 compute iMEM 分段。runtime CSV 为 5,702,520 字节、45,385 patterns、36 个资源；
已通过 viewer 流式解析器检查，viewer 和 CSV HTTP 地址均返回 200。

[打开本步实际流水图](http://127.0.0.1:8765/tools/pipeline_viewer/index.html?csv=/results/deepseek_v41_flash_layer0_attention_input_norm_prefill/runtime.pipeline.csv&v=qa_kv_rmsnorm_20261111)

```powershell
python compiler/tests/deepseek_v41_attention_input_test.py --build-dir tmp/deepseek_swa_input_norm_snapshot --target-config ../FTLPU-CMODEL/config/ftlpu-lpu32.json --package results/deepseek_v41_flash_layer0_qkv_prefill --norm-package results/deepseek_v41_flash_layer0_attention_rmsnorm --model-config results/deepseek_v41_flash_assessment/official_config.json --output-dir results/deepseek_v41_flash_layer0_attention_input_norm_prefill --with-qkv-norm
```

输入、权重、可执行文件和 target 的 SHA256 在 `summary.json`，下一步是 Q_B 升维
投影和 Q/KV 的 partial RoPE。本步没有执行这些后续算子。

## mHC 后的第 4 步：Q_B 升维与 Q/KV partial RoPE（融合前基线）

2026-11-11 新增固定测试目录
`results/deepseek_v41_flash_layer0_qkv_rope_prefill`，从同一个 BF16
hc_pre 接口 fixture `[32,5120]` 连续执行到 Q/KV RoPE。最终验证状态和实际运行
数据以本目录 `summary.json` 为准。以下是融合前已通过的基线，保存在
`baseline_before_rotary_fusion/`；后续优化结果见下一节。本步仍不包含执行 mHC、SWA、cache 写回、
attention 输出投影或 FFN，也不代表完整 attention/decoder 已完成。

逻辑链为：

1. 输入 RMSNorm，Q_A `[32,1280]` / KV `[32,512]` 投影。
2. Q_A 和 KV 各自的真实 gamma RMSNorm。
3. Q_B `[1280 → 32768]`，输出重解释为 `[32,64,512]`。
4. Q 与单头 KV 的 partial RoPE，输出分别为 `[32,64,512]` 和 `[32,1,512]`。

保留官方 revision `1bae0e1088993b9b82de08f59de5be5a34e42170` 的 E4M3FN
权重与 E8M0 block32 scale；MXM 权重入口反量化为 BF16，激活保持 BF16，
乘积/累加使用既有 FP32 路径，输出 BF16 RNE。Q_B 复用公共投影 lowering，
按 N 分成 21 页，每页仍包含完整 K=1280，并直接读取 Q_A RMSNorm 的原地址。
没有额外激活 QDQ，没有 distributed16 输入、输出或中间缓存。

官方纯 SWA 层的 `compress_ratios[0]=0`，推理实现对这种层关闭 YaRN；因此这里
使用 theta=10000、位置 0–31，只旋转每个 512 维 head 的最后 64 维相邻偶/奇对，
前 448 维保持原值。RoPE 沿用公共 `lowerInterleavedRotary`：设备计算相邻对的
带符号置换，VXM 计算两个 FP32 乘积后相加，最终只在输出处转为 BF16。
RoPE 的逐 head 中间布局转换仍存在，尚未改成 SXM/VXM 连续链。

本轮新增通用的原地址输出视图：编译器逐段证明 planar 地址、lane、半球、
bank 和非重叠关系，生成 C2C 输出描述符，直接回读已有前缀和旋转尾部。
不再为了诊断输出构造整块 MEM 归档。主机只按描述符拼接回读字节，不计算
任何模型算子。描述符的软件解析上限由 4096 调整为 65536；该限制不是硬件
iMEM 或队列容量，实际 C2C 仍以现有有界批次执行。旧解析器不能读取超过
4096 段的新结果，编译器、runner 和导出器应使用同一次构建。

测试返回 8 个张量，其中前 6 个是诊断检查点，最后 2 个是本阶段结果。
验收保持相对 L2 < 1%、cosine > 0.9999；三个投影还逐位核对基于设备实际
输入的局部参考。RoPE 单独要求全张量相对 L2 < 0.1%、旋转尾部 < 1%，
未旋转前缀和 position0 均要求 BF16 位差为零。无持久状态。

```powershell
python compiler/tests/deepseek_v41_attention_input_test.py --build-dir tmp/deepseek_swa_qkv_rope_snapshot --target-config ../FTLPU-CMODEL/config/ftlpu-lpu32.json --package results/deepseek_v41_flash_layer0_qkv_prefill --norm-package results/deepseek_v41_flash_layer0_attention_rmsnorm --model-config results/deepseek_v41_flash_assessment/official_config.json --output-dir results/deepseek_v41_flash_layer0_qkv_rope_prefill --with-qb-rope
```

`--validate-existing` 可以在严格核对逻辑图、程序、输入、权重、target 和工具链
SHA256 后重做验收，不再次执行硬件仿真。`layout_comparison.json` 是编译结果
的搬运统计比较，旧基线仅编译、没有运行，不能当作实测时延改善。

[打开本步实际流水图](http://127.0.0.1:8765/tools/pipeline_viewer/index.html?csv=/results/deepseek_v41_flash_layer0_qkv_rope_prefill/runtime.pipeline.csv&v=qb_rope_20261111)

实际执行的 ICU 文件在 `runtime_icu_programs/index.csv`，包含首次读入和运行时
同步；计算、外存等待、诊断回读的时间分开记录在 `summary.json`。

本次 8 个输出均通过既定阈值，三个投影相对设备实际输入的局部 BF16 位差均为 0。
Q RoPE 的端到端相对 L2 为 **0.304388%**（cosine=0.9999953683），KV RoPE 为
**0.253824%**（cosine=0.9999967825）。Q RoPE 本身相对实际 Q_B 输出的局部全量
相对 L2 为 3.74846e-6，旋转尾部为 9.61036e-6；KV RoPE 的局部位差为 0。
两路未旋转前缀和 position0 均逐位保持输入；前五个上游检查点与上一阶段
`attention_input_norm_prefill` 结果逐字节一致。`host_operations=0`。

500 MHz、DDR 峰值 51.2 GB/s、效率 90% 下的实测时序如下。数据不包含主机
编译和指令镜像 DMA，不可当作完整 attention 或完整模型 TTFT。

| 时段 | 周期 | 时延 |
| --- | ---: | ---: |
| 初始输入/常量读入 | 38,042 | 0.076084 ms |
| 调用内计算与静态排程，扣除权重页等待 | 1,198,327 | 2.396654 ms |
| 运行时权重页等待 | 25,166 | 0.050332 ms |
| 初次读入至本阶段计算完成 | 1,261,535 | **2.523070 ms** |
| 八个输出的诊断回读 | 578,664 | 1.157328 ms |
| 包括诊断回读的完整 trace | 1,840,199 | 3.680398 ms |

有 28 个执行 epoch、27 次权重页预取，共 51,118,080 字节；没有 host 算子。
runtime CSV 为 27,997,157 字节，viewer 流式解析器识别 221,997 patterns、40 个
资源，viewer 与 CSV 的 HTTP 检查均为 200。计算/初始装载 ICU 导出包含 232 个
队列文件，124 个活跃，40 条 MEM_WRITE_SYNC、448 条 SYNC；packed MEM/MXM
物理镜像验证通过。末尾 HostOutput 诊断回读在 runtime CSV 中，其逐批临时
ICU 镜像不包含在计算 ICU 文件里。

本次移除了原编译结果中 3,298 段用于输出归档的 MEM→MEM。当前仍有 1,300 段
图间传输：Q/KV RMSNorm 的半球分发 4 段、Q RoPE 内部 1,280 段、KV RoPE 内部
16 段。后两者来自现有逐 head 置换/乘加路径的布局与存储转换，不能声称已经
完成紧密融合。旧编译基线没有运行，因此只比较搬运结构，不报告实测加速比。

回归证据在 `validation/`：两头真实 Q 的正/反向 RoPE 局部输出逐位一致；
原地址回读的 stride/半球映射、5,000 段描述符的四种解析入口、描述符超限拒绝
测试通过；旧输入阶段的编译入口通过。`pipeline_parser_validation.json`、
`http_validation.json` 和 `runtime_icu_programs/macro_export_validation.json`
分别保留流水解析、链接和实际指令镜像检查结果。

## 相邻对 RoPE 的原生 planar 融合

交付范围仍是同一个 seq32 第 0 层输入接口到 Q/KV partial RoPE，权重、输入、
精度和阈值保持不变。`RotaryGraphLowering.inc` 识别相邻对 RoPE，
`RotaryPlanarSchedule.inc` 使用公共地址分配器和真实 MEM/SXM/VXM ICU 指令生成排程。
没有修改 CModel 硬件能力，也没有 Python 布局或周期生成。

每个 head 的 64 维尾部不再通过矩阵乘完成偶/奇换位，也不再生成两个 FP32
乘积张量和独立加法节点。当前映射为：

1. 从 Q_B/KV RMSNorm 的物理原地址读取旋转区域，通过 wide SXM 转成
   “每个 32-lane 向量对应 32 个 token 的同一维”普通 planar 缓存。
2. C0/C8 计算 `x[d]*cos[d]`，C1/C9 清零并累加；下一拍计算
   `x[d^1]*signed_sin[d]`，在同一个 FP32 累加器中相加，再以 BF16 RNE 输出。
   偶维的 sine 为负，奇维为正；逆 RoPE 反转 sine 符号。
3. wide SXM 恢复 token×feature 的输出方向，直接写最终尾部地址。未旋转的
   前缀和后缀保持原地址视图，不再构造完整输出归档。

两个乘积不落 MEM。当前仍保留两次转置的缓存：前一份支持相邻维的重复寻址，
后一份把 VXM 的非连续输出转为 wide SXM 要求的 32 拍连续输入。
这些缓存没有被计入“已消除”的搬运；完整比较同时统计它们的实际向量数。

相邻输出的间隔为 6 拍。两条不同的累加器配置各占 3 个 raw-macro word，
ICU 取指率为 1 word/cycle；4 拍方案被公共 frontend 带宽检查拒绝。
没有交错运行同一 ICU 队列中的不同宏指令，没有更改取指带宽。
Q 的 64 个 head 分成 8 组，KV 再占 1 组。每组至多 1,536 个 VXM iMEM words，
在编译器声明的 drain 边界复用现有分段装载机制；这不扩大 2,048-word 容量。

原生策略支持 BF16、wide SXM、T 为 32 的倍数、head 宽度和旋转起点为 32 的
倍数，以及旋转宽度 64/128/192/256。其他旋转宽度保留原公共兼容路径。
使用 FP32 sin/cos、FP32 乘积和累加，最终输出 BF16；不引入 distributed16。
`projection_rope_overlap` 的原有配置未被暗中打开。

回归在固定结果目录的 `validation/rotary_real_heads`、`rotary_multi_tile`、
`rotary_full_width`：真实 Q 的正/反向旋转，以及 seq64、3 heads、不同前后缀、
128/256 维旋转均通过原有 0.1% 相对 L2 阈值，未旋转部分逐位一致。
多 tile 测试的 `runtime_icu_programs/compute_imem_segments.csv` 保留了每段容量证明。

复现连续测试：

```powershell
python compiler/tests/deepseek_v41_attention_input_test.py --build-dir tmp/deepseek_rope_fusion_snapshot --target-config ../FTLPU-CMODEL/config/ftlpu-lpu32.json --package results/deepseek_v41_flash_layer0_qkv_prefill --norm-package results/deepseek_v41_flash_layer0_attention_rmsnorm --model-config results/deepseek_v41_flash_assessment/official_config.json --output-dir results/deepseek_v41_flash_layer0_qkv_rope_prefill --with-qb-rope
python compiler/tools/compare_rotary_fusion.py --before results/deepseek_v41_flash_layer0_qkv_rope_prefill/baseline_before_rotary_fusion --after results/deepseek_v41_flash_layer0_qkv_rope_prefill
```

比较工具只分析已完成的结果，不参与 lowering。它核对输入、权重、精度、阈值、
target、runner、exporter 的一致性，并检查前六个上游输出逐字节不变。

连续 raw-macro 数值重跑结果为 **passed**，八个输出全部与融合前逐字节一致。
Q/KV 端到端相对 L2 仍分别为 0.304388% / 0.253824%，三个投影的局部 BF16 位差
仍为零，RoPE 未旋转前缀和 position0 位差仍为零，`host_operations=0`。

同条件实测（500 MHz，DDR 51.2 GB/s，效率 90%）：

| 时段 | 融合前周期 | 融合后周期 | 融合后时延 |
| --- | ---: | ---: | ---: |
| 初始输入/常量读入 | 38,042 | 37,097 | 0.074194 ms |
| 调用内计算及静态排程，扣除权重页等待 | 1,198,327 | 844,018 | 1.688036 ms |
| 运行时权重页等待 | 25,166 | 25,045 | 0.050090 ms |
| 初次读入至本阶段计算完成 | 1,261,535 | **906,160** | **1.812320 ms** |
| 八个输出的诊断回读 | 578,664 | 568,014 | 1.136028 ms |
| 包括诊断回读的完整 trace | 1,840,199 | 1,474,174 | 2.948348 ms |

包含外存的本阶段时延从 2.523070 ms 降至 1.812320 ms，减少 **28.17%**。
两个测试均不计入指令镜像 DMA，不能将此时间宣称为完整 attention/模型 TTFT。
编译器节点边界的 Q_B 结束至 RoPE 结束区间从 372,686 缩短为 18,371 周期；
这个边界统计与上表实际 runtime 计时分开保存。

图间传输域从 1,300 降到 4，剩余的是两次 RMSNorm 的半球分发。
RoPE 原有 1,296 个图间转换域（74,752 个向量读写对）被移除；新实现两次
wide transpose 共保留 **8,320 个向量读写对**，并非零搬运。
图节点从 749 降到 41；两个乘积中间张量的 MEM 写入为零。

实际 ICU 导出为 232 个队列文件、104 个活跃队列、68,257 条 ICU 指令、
24,276 个 iMEM words，包含 32 条 MEM_WRITE_SYNC 和 444 条 SYNC。
28 个权重执行 epoch、27 次权重预取、51,118,080 字节权重流量保持不变；
`compute_imem_segments.csv` 另列计算 ICU 的装载分段，所有容量检查通过。
分段统计中的最大使用量为 1,536 words，未扩大硬件 iMEM。
诊断回读仍只在 runtime CSV 中，未把其临时 ICU 批次冒充计算 program。

新 CSV 为 26,140,588 字节、208,387 个 patterns、36 个资源，已通过现有 viewer
解析器检查。[完整 runtime 流水图](http://127.0.0.1:8765/tools/pipeline_viewer/index.html?csv=/results/deepseek_v41_flash_layer0_qkv_rope_prefill/runtime.pipeline.csv&v=rope_fused_20261111)
与 [RoPE 局部](http://127.0.0.1:8765/tools/pipeline_viewer/index.html?csv=/results/deepseek_v41_flash_layer0_qkv_rope_prefill/runtime.pipeline.csv&v=rope_fused_20261111&start=887000&end=906200)
使用同一份实际 trace。`fusion_comparison.json` 保存同条件证明和逐项比较。

## RoPE 的 SXM/VXM tile 流水重叠

在上述乘加融合基础上，通过 `ftlpu.rope_tile_overlap=true` 开启内部 tile 排程。
它默认关闭；测试入口以 `--rope-tile-overlap` 显式开启。关闭时继续使用整组
输入转置、整组 VXM、整组输出转置的串行路径。`projection_rope_overlap`
仍独立控制投影与 RoPE 的边界，没有被这个选项隐式打开。

`RotaryTilePipeline.inc` 为每侧保留 X_A/X_B、Y_A/Y_B 四个普通 planar tile，
每个为 32×32 BF16。缓存放到不同 MEM ICU 队列，SXM 完成 tile i+1 的输入转置
和 tile i−1 的输出转置时，VXM 可以继续计算 tile i。两次 SXM 任务在同一个
SXM 上依次发出，遵守 32 拍的最小启动间隔，且保持相同的流方向。

SR 分配也隔离：VXM 的激活占本地 0–1、FP32 系数占 4–7，SXM 返回数据使用
本地 2–3（East 的 encoded 34/35、West 的 50/51）。系数初始 C2C 装载直接落到
读队列独占的地址；公共内存规划在其消费者区间内禁止其他保留张量复用这些
队列，避免“地址不同但 ICU 相同”导致的冲突。没有额外复制系数。

当前目标下，每个计算 tile 为 32 个输出向量、每向量 6 拍，即 192 拍。
组内相邻 tile 的 VXM 起点保持 192 拍间隔，SXM 不再插入额外的 tile 间停顿。
每组首 tile 仍需填充，末 tile 仍需排空；单头 KV 每侧只有一个 tile，无法借助
后继 tile 隐藏这些时间。现有 iMEM drain 边界和 6 拍累加器配置间隔均保留。

这是带 MEM 双缓冲的流水重叠，不是无缓存的 SXM→VXM 直连。宽转置需要连续
的 32 拍输入，而相邻维复用和当前 VXM 输出节拍仍需要缓存解耦。转置流量保持
8,320 个向量读写对；最大八头组的 X/Y 缓存从 128 KiB 降至合计 16 KiB
（两半球总量，不包括系数和最终结果）。

验证包含真实 Q 的多 head 正/反向 RoPE、seq64 的 128 维旋转、256 维旋转和
关闭重叠选项的串行回归。`graph_plan.json` 给出每个 tile 的 SXM/VXM 起止、
buffer slot 和物理队列；测试同时检查 192 拍节拍、转置重叠、缓存队列隔离，
并在 raw-macro CModel 上验证数值与实际 SR 路由。

固定目录继续为 `results/deepseek_v41_flash_layer0_qkv_rope_prefill`；上一步结果
保存在 `baseline_before_sxm_overlap/`，其中也包含上一步的 `fusion_comparison.json`。

```powershell
python compiler/tests/deepseek_v41_attention_input_test.py --build-dir tmp/deepseek_rope_overlap_snapshot --target-config ../FTLPU-CMODEL/config/ftlpu-lpu32.json --package results/deepseek_v41_flash_layer0_qkv_prefill --norm-package results/deepseek_v41_flash_layer0_attention_rmsnorm --model-config results/deepseek_v41_flash_assessment/official_config.json --output-dir results/deepseek_v41_flash_layer0_qkv_rope_prefill --with-qb-rope --rope-tile-overlap
python compiler/tools/compare_rotary_fusion.py --before results/deepseek_v41_flash_layer0_qkv_rope_prefill/baseline_before_sxm_overlap --after results/deepseek_v41_flash_layer0_qkv_rope_prefill --output results/deepseek_v41_flash_layer0_qkv_rope_prefill/sxm_overlap_comparison.json
```

连续 raw-macro 实测通过，八个检查点与上述串行版本全部逐字节一致。
本次输入仍是 BF16 的 hc_pre fixture `[32,5120]`，输出为 Q `[32,32768]`
和 KV `[32,512]`，没有持久状态。范围止于 Q/KV partial RoPE，未执行 mHC、
SWA core、cache 写回、输出投影或 FFN。原始 E4M3FN 权重和 E8M0 block32 scale
保持不变，在 MXM IW 入口反量化为 BF16；激活不做 FP8 动态量化。

同条件比较（500 MHz、DDR 51.2 GB/s、效率 90%）：

| 时段 | 串行 SXM 周期 | tile 重叠周期 | 重叠版本时延 |
| --- | ---: | ---: | ---: |
| 初始输入/常量读入 | 37,097 | 37,097 | 0.074194 ms |
| 调用内计算及静态排程，扣除权重页等待 | 844,018 | 840,254 | 1.680508 ms |
| 运行时权重页等待 | 25,045 | 25,045 | 0.050090 ms |
| 初次读入至本阶段计算完成 | 906,160 | **902,396** | **1.804792 ms** |
| 八个检查点的诊断回读 | 568,014 | 552,410 | 1.104820 ms |
| 包括诊断回读的完整 trace | 1,474,174 | 1,454,806 | 2.909612 ms |

九组 RoPE 的编译器节点区间合计从 18,133 降至 **14,369 周期**，减少 20.76%；
当前测试范围的包含外存时延从 1.812320 降至 1.804792 ms，减少 0.415%。
诊断回读时间单列，不把回读节省计作算子加速。两版均未建模指令镜像 DMA 时间。
Q/KV 端到端相对 L2 分别为 0.304388% / 0.253824%，阈值不变；
三个投影的局部 BF16 位差、未旋转前缀位差和 position0 位差仍为零。

九组启用重叠，检查到 56 个前一 tile 输出转置与当前 VXM 计算重叠的实例。
实际 ICU 导出包含 232 个队列文件、116 个活跃队列、68,481 条 ICU 指令、
24,624 个 iMEM words，其中 MEM_WRITE_SYNC 32 条、SYNC 444 条。
所有计算段 iMEM 容量检查通过，最大段 1,536 words；`host_operations=0`。
图间搬运域仍为 4，RoPE 图间复制和输出归档为零；两次 wide transpose 的
8,320 个向量读写对仍存在，没有把缓存重叠宣称为零搬运。

本次 runtime CSV 为 26,168,804 字节，208,615 个 patterns、36 个资源，已通过
现有 viewer 解析器检查；localhost viewer 与 CSV 均返回 HTTP 200，文件长度一致。
[SXM/VXM 重叠局部流水图](http://127.0.0.1:8765/tools/pipeline_viewer/index.html?csv=/results/deepseek_v41_flash_layer0_qkv_rope_prefill/runtime.pipeline.csv&v=rope_sxm_overlap_20261112&start=887850&end=889750)
和 [完整 runtime 流水图](http://127.0.0.1:8765/tools/pipeline_viewer/index.html?csv=/results/deepseek_v41_flash_layer0_qkv_rope_prefill/runtime.pipeline.csv&v=rope_sxm_overlap_20261112)
指向同一份本次 trace。实际计算 ICU program 为固定结果目录下的
`runtime_icu_programs/index.csv`，诊断回读临时 ICU 批次仍仅显示在 runtime CSV 中。
同条件输入/权重/精度/硬件证明与逐字节比较保存在 `sxm_overlap_comparison.json`。

## 第 0 层 seq32：首次连接 planar SWA core

本节记录首次连接的基线；产物已保存在固定结果目录的
`baseline_before_swa_tight/`。最新优化结果见下一节。

固定结果目录为 `results/deepseek_v41_flash_layer0_swa_core_prefill`。
输入为同一 BF16 `hc_pre[32,5120]` fixture，经 attention RMSNorm、真实权重
Q_A/KV、两路 RMSNorm、Q_B 和 partial RoPE，连接公共 `dense_attention`。
core 输入为 Q `[1,32,64,512]`、共享 K=V `[1,32,1,512]`，输出为
BF16 context `[32,32768]`。causal position=0、window=128，使用官方第 0 层
64 个 FP32 attention sink。没有初始历史 KV，也没有持久 cache 写回。
本步止于 core context；inverse RoPE、分组 O_A/O_B、mHC 和 FFN 尚未串入。

公共 D512 lowering 按两个独立 Q head 分组，分配给两个半球。共享 KV 由逻辑
view 引用；每侧计算不同的 Q head。它调用已有 `attention_core` 的
`flash_tile` planar 内存规划和 QK/softmax/PV emitter，未添加模型专用 CModel。
配置仍为 wide SXM、raw-macro；投影沿用官方 FP8 权重入口反量化与 BF16 激活，
core 内也不做激活/KV 的 FP8 QDQ。所有新 scratch 都是普通 planar。

计算路径为 MXM KQᵀ → VXM scale/causal/max/exp → SXM → MXM VᵀP →
VXM FP32 除以分母 → SXM 恢复 feature lanes → MEM context。
QK 结果及 scale 后分数为 BF16；max、指数、分母和 PV 累加使用 FP32。
指数在进入 PV 权重口时转换为 BF16，分母仍累加原始 FP32 指数，最终 context
转换为 BF16。相同 exponent MEM read 同时服务 P→SXM 和分母归约。
sink 进入稳定最大值和分母，不进入 value 产品；其 per-head FP32 行按实际
跨半球 SR 数据所有权读取。没有把 sink 当作普通 value token。

RoPE→V 的 feature-lane/token-lane 转换由公共 wide-SXM 搬运实现；跨半球结果
通过已有 VXM 输入边界的被动 SR 转发直接写目标 MEM，不发 VXM bypass，也不
增加一份整块中转 tensor。32 个 head group 之间的 KV 重新准备仍保留，尚未
完成跨组驻留复用。测试请求回读的 context 另有 `validation_export.8` 打包，
需要与算子计算时间分开统计，不能宣称整段零搬运。

验收保持端到端 relative L2 <1%、cosine >0.9999，另外用实际 Q/KV 检查
core 局部 relative L2 <1%；上游投影仍要求局部 BF16 逐位一致，RoPE 前缀和
position0 仍要求位差为零。独立 core 的两个不同 sink 合成用例、真实 Q/KV
接口用例、RoPE→core 连接用例，以及原公共无 sink 的 GQA128/MHA32 均已通过。
连续 raw-macro 运行也已通过，固定目录 `summary.json` 的状态为 `passed`。

```powershell
python compiler/tests/deepseek_v41_attention_input_test.py --build-dir tmp/deepseek_swa_core_snapshot --target-config ../FTLPU-CMODEL/config/ftlpu-lpu32.json --package results/deepseek_v41_flash_layer0_qkv_prefill --norm-package results/deepseek_v41_flash_layer0_attention_rmsnorm --model-config results/deepseek_v41_flash_assessment/official_config.json --output-dir results/deepseek_v41_flash_layer0_swa_core_prefill --with-swa-core --swa-package results/deepseek_v41_flash_layer0_swa_prefill --rope-tile-overlap
python compiler/tests/deepseek_v41_swa_core_test.py --build-dir tmp/deepseek_swa_core_snapshot --target-config ../FTLPU-CMODEL/config/ftlpu-lpu32.json --output-dir results/deepseek_v41_flash_layer0_swa_core_prefill/validation/core_real --heads 2 --input-run results/deepseek_v41_flash_layer0_qkv_rope_prefill --sink-file results/deepseek_v41_flash_layer0_swa_prefill/weights/attn_sink.bin
python compiler/tests/deepseek_v41_swa_core_test.py --build-dir tmp/deepseek_swa_core_snapshot --target-config ../FTLPU-CMODEL/config/ftlpu-lpu32.json --output-dir results/deepseek_v41_flash_layer0_swa_core_prefill/validation/rope_core --heads 2 --device-rope
```

最终 context 端到端 relative L2 为 **0.368593%**、cosine 为
**0.9999932312**；用实际 Q/KV 单独核对 core，relative L2 为 **0.240369%**、
cosine 为 **0.9999971115**。此前八个检查点与上一步 Q/KV/RoPE 连续测试
逐字节一致。输入、原始权重、gamma、精度、硬件配置、runner/exporter 的一致性
由 `core_comparison.json` 校验。新增了 SWA，因此下面的总时间不作为同范围加速比。

实测配置仍为 500 MHz、DDR 51.2 GB/s、效率 90%：

| 时段 | 周期 | 时延 |
| --- | ---: | ---: |
| 初始输入/常量读入 | 39,379 | 0.078758 ms |
| 调用内计算、静态排程和诊断打包，扣除权重等待 | 1,176,577 | 2.353154 ms |
| 运行时权重页等待 | 25,087 | 0.050174 ms |
| 初始读入至设备调用结束，含 context 诊断打包 | **1,241,043** | **2.482086 ms** |
| 九个检查点的诊断回读 | 650,794 | 1.301588 ms |
| 包含诊断回读的完整 trace | 1,891,837 | 3.783674 ms |

32 组 core 原生节点区间合计 **45,542 周期**。从 RoPE 结束到最后一组 core
结束共 **291,347 周期**，包含输入准备和图间搬运，不能把 45,542 周期当作
整段 SWA 实际耗时。随后 `validation_export.8` 的 256 个被动搬运域打包
32,768 个向量，跨度 **44,720 周期**；这部分仍在设备调用内，表中已明确包含。
权重页等待均在 SWA 之前。上述运行时周期仍未计入指令镜像 DMA。

实际 ICU 导出包含 232 个队列文件、158 个活跃队列、91,137 条 ICU 指令、
36,318 个 iMEM words，其中 MEM_WRITE_SYNC 288 条、SYNC 508 条。
69 个计算 iMEM 段全部通过容量检查，最大段 1,536 slots；`host_operations=0`。
实际指令位于 `runtime_icu_programs/index.csv` 及其各队列文件；诊断回读的
临时 ICU 批次仅在 runtime CSV 中呈现。

本次 runtime CSV 为 29,223,560 字节，230,365 个 patterns、40 个资源。
现有 viewer 的流式 CSV 解析器检查通过；本机 viewer 和 CSV 均返回 HTTP 200，
CSV Content-Length 与磁盘文件相同。这是解析及服务检查，未代替浏览器渲染检查。

- [完整 runtime 流水图](http://127.0.0.1:8765/tools/pipeline_viewer/index.html?csv=/results/deepseek_v41_flash_layer0_swa_core_prefill/baseline_before_swa_tight/runtime.pipeline.csv&v=swa_core_20261112)
- [SWA 段流水图，含输入准备](http://127.0.0.1:8765/tools/pipeline_viewer/index.html?csv=/results/deepseek_v41_flash_layer0_swa_core_prefill/baseline_before_swa_tight/runtime.pipeline.csv&v=swa_core_20261112&start=904500&end=1196000)

复核同条件输入和上游结果，并汇总本步周期：

```powershell
python compiler/tools/summarize_deepseek_swa_core.py --result results/deepseek_v41_flash_layer0_swa_core_prefill --frontend results/deepseek_v41_flash_layer0_qkv_rope_prefill
```

后续优先复用跨 head group 的 KV 常驻布局，并让下游 inverse RoPE/输出投影
直接读取分组 context，避免把诊断打包当作正式算子接口。当前尚未验证超出 seq32、
position0、无历史 KV 的本模型 SWA 连续路径。

## 第 0 层 seq32：SWA 分组复用与搬运排程

本轮仍使用上述输入、权重、BF16/FP32 精度和输出范围。固定目录不变；首次连接
版本的产物保存到 `baseline_before_swa_tight/`，包括原始 runtime CSV、实际 ICU
program、检查点、编译产物和运行记录。以下优化由 C++ 公共 lowering 完成。

- D512 core 按 native Q/context 的 SRAM 容量选择 head 分组。当前两半球每个
  planar byte pair 各有 8,192 rows，因此一次可覆盖 32 个 Q head，64 heads
  分成两组。每侧 16 个 head 共享已准备好的 K/V，沿用公共 core 的一 tile
  前瞻：后续 QK 与当前 softmax 重叠，后续 scale/max 利用 P 的 SXM/LOAD 时间。
- 宽 head 组使用 head-major 的普通 planar 地址顺序，每个 tile 对应同一 head
  的 32 个 token；causal mask 和 sink lane 的归属同步调整。数学运算、K32
  累加顺序和各精度边界不变。原有 D32/D64/D128 tile 布局兼容路径保留。
- 公共被动 MEM 搬运在源/目标物理队列分离、目标 word 不重复时，比较逻辑顺序、
  源地址顺序和目标地址顺序对应的仿射域数量，选择域更少的遍历。整块可用 3D
  域表示时直接生成该域；分页边界产生的多个区域分别生成连续域。输入是逻辑
  word 地址关系，不从 FU 指令反向压缩。重叠队列先走原快照路径；重复目标
  地址保留原顺序。

这同时消除了“两头一组”之间重复的 K/V 准备，并减少了 Q 分页源地址造成的
小搬运域和逐域启动/排空间隔。两个大组之间尚未复用 K/V 的物理驻留，Q 的
consumer layout 准备和最终诊断 context 打包也仍存在。

复现本轮连续执行和同条件比较：

```powershell
python compiler/tests/deepseek_v41_attention_input_test.py --build-dir tmp/deepseek_swa_tight_snapshot --target-config ../FTLPU-CMODEL/config/ftlpu-lpu32.json --package results/deepseek_v41_flash_layer0_qkv_prefill --norm-package results/deepseek_v41_flash_layer0_attention_rmsnorm --model-config results/deepseek_v41_flash_assessment/official_config.json --output-dir results/deepseek_v41_flash_layer0_swa_core_prefill --with-swa-core --swa-package results/deepseek_v41_flash_layer0_swa_prefill --rope-tile-overlap
python compiler/tools/compare_deepseek_swa_schedule.py --before results/deepseek_v41_flash_layer0_swa_core_prefill/baseline_before_swa_tight --after results/deepseek_v41_flash_layer0_swa_core_prefill
```

验收继续要求所有九个输出通过原数值阈值，并检查与 baseline 的 BF16 文件
逐字节一致。独立验证包括 32-head core、8-head RoPE→core，以及公共无 sink
GQA128/MHA32。独立 32-head RoPE fixture 的单片输入需 16,384 rows，超过现有
8,192-row bank，编译器正确拒绝；该接口用例改用 8 heads。真实 64-head 连续
路径从已分页的 projection/RoPE 输出接入，不使用这份超容量 fixture。

最终连续运行通过：九个 BF16 检查点（含 context）均与首次连接版逐字节一致。
端到端 relative L2 仍为 0.368593%，core 局部为 0.240369%，阈值未改变。
同条件比较保存在 `scheduling_comparison.json`：

| 项目 | 优化前 | 优化后 |
| --- | ---: | ---: |
| native core 分组数 | 32 | 2 |
| SWA 含输入准备 | 291,347 cycles | **93,949 cycles** |
| core 原生节点区间合计 | 45,542 cycles | **36,150 cycles** |
| SWA 输入被动搬运向量 | 80,000 | **35,840** |
| SWA 输入被动搬运域 | 576 | 300 |
| context 诊断打包 | 44,720 cycles / 256 域 | 33,376 cycles / 16 域 |
| MXM 发射占用率，含 SWA 输入准备 | 11.25% | **34.88%** |
| MXM 发射占用率，仅 core 原生区间 | 71.95% | **90.64%** |
| VXM 各 ALU 平均发射占用率，含 SWA 输入准备 | 2.13% | 6.60% |

占用率来自实际 runtime CSV 的 issue 区间并集；该 SWA 区间内没有外存等待。
VXM.C0–C7 是镜像 ALU 对，平均值覆盖所有 16 个物理 ALU，不把只用两列的
归一化链算成整个 VXM 满载，也不把该指标当作 FLOP 效率。

500 MHz / DDR 51.2 GB/s、效率 90% 下的完整时间：

| 时段 | 优化前周期 | 优化后周期 | 优化后时延 |
| --- | ---: | ---: | ---: |
| 初始输入/常量读入 | 39,379 | 37,426 | 0.074852 ms |
| 调用内计算、静态排程和诊断打包，扣除权重等待 | 1,176,577 | 967,835 | 1.935670 ms |
| 运行时权重页等待 | 25,087 | 24,950 | 0.049900 ms |
| 初始读入至设备调用结束，含诊断打包 | 1,241,043 | **1,030,211** | **2.060422 ms** |
| 诊断回读 | 650,794 | 650,794 | 1.301588 ms |
| 完整 trace | 1,891,837 | 1,681,005 | 3.362010 ms |

SWA 段缩短 **67.75%**，整个测试设备调用含初始外存开销缩短 **16.99%**。
上游 RoPE 仍结束于逻辑周期 840,131；新 core 的两个区间为
`[868830,886905)`、`[916005,934080)`。实际 SWA 含准备的窗口为约
`[902507,996456)`。两个组之前的 Q 准备仍会形成可见空闲；下一步可让消费者
直接使用 projection/RoPE 的地址视图，或在不冲突的队列上与前组计算重叠。
本轮没有为填满图上的空白引入重复计算。

实际 ICU 导出：232 个队列文件、166 个活跃队列、83,007 条指令、30,712 个
iMEM words，MEM_WRITE_SYNC 48 条、SYNC 448 条；39 个计算段容量检查全部通过，
最大段 1,536 slots。
`host_operations=0`。runtime CSV 为 28,104,564 字节、222,914 个 patterns、
40 个资源；viewer 解析器通过，优化前后 CSV 与 viewer 均已确认 HTTP 200。

- [优化后 SWA 流水图](http://127.0.0.1:8765/tools/pipeline_viewer/index.html?csv=/results/deepseek_v41_flash_layer0_swa_core_prefill/runtime.pipeline.csv&v=swa_tight_20261112&start=902400&end=996600)
- [优化前 SWA 流水图](http://127.0.0.1:8765/tools/pipeline_viewer/index.html?csv=/results/deepseek_v41_flash_layer0_swa_core_prefill/baseline_before_swa_tight/runtime.pipeline.csv&v=swa_baseline_20261112&start=904500&end=1196000)
- [优化后完整 runtime](http://127.0.0.1:8765/tools/pipeline_viewer/index.html?csv=/results/deepseek_v41_flash_layer0_swa_core_prefill/runtime.pipeline.csv&v=swa_tight_20261112)
