# Qwen3.5 Full Attention decoder 层

`qwen3_5_full_attention_decoder_layer_test.py` 使用官方
`Qwen/Qwen3.5-0.8B` 的第 3 层（从 0 开始计数，首个 full-attention
层），默认 seq32。输入为官方 embedding 的前 32 行，作为隔离整层
测试的输入，不表示已经执行了前面三个 decoder 层。

完整数据流为 input RMSNorm → Q/K/V 和 attention gate projection →
Q/K head RMSNorm → partial RoPE → causal GQA attention →
context × sigmoid(gate) → O projection → residual → post RMSNorm →
SwiGLU FFN → residual。hidden=1024，FFN=3584，Q heads=8，KV heads=2，
head dim=256，rotary dim=64。

## 权重和数值口径

下载器保存官方 revision、下载 tensor 范围和 SHA256。测试验证权重
subset 与 config 的 SHA256，并拒绝非 full-attention 层。当前硬件
运行 W8A16：真实 checkpoint BF16 矩阵采用 per-tensor INT8 量化，
Q 和 attention gate 在量化后按 head 拆分，保留共同的 scale。
四处 RMSNorm 均使用 Qwen3.5 的 `1 + checkpoint_weight`。

验收基准是相同 INT8 权重反量化后的独立 FP32 方程，要求 NRMSE
不超过 5%、cosine 至少 0.999。另行报告相对原始 BF16 checkpoint 的
误差，以及单独的 INT8 量化误差，避免混淆量化损失与执行误差。

## 编译和执行

Python 只提供 checkpoint 导入、整层语义 Kernel IR 和独立参考方程。
整层进入与 Qwen2.5 相同的 `KernelToTensor`：`match_attention_graph`
识别完整 attention 子图，`match_ffn_graph` 识别 Gate/Up/Swish/Multiply/Down，
由 `FunctionMemoryPlanner` 和原生 attention/FFN 内存规划分配物理存储。
随后统一经过 `TensorToStream`、`StreamToSchedule`、调度验证与直接 ICU
指令 lowering。此测试不调用 LogicalGraph 编译器，不逐算子编译后拼接。
四处 RMSNorm 的 gamma 统一由整层常量规划分配；attention 的 Q/K norm
直接采用该分配并锁定 bank，避免后续权重 bank 分配把 gamma 移入
RoPE 表所在位置。
分页内存规划将持久 causal mask 放在 activation 区，避免 FFN 提前预取
覆盖 softmax 尚未使用的常量；串行 projection 的首个权重 MEM 读取也必须
在前一阶段跨半球 SR 搬运完成后，才能安全进入 page-ready SYNC。

FFN 使用原生 fused FFN lowering。Stream IR 中只有一个额外的独立
linear projection，即 attention gate；Gate/Up/Down 属于完整 FFN 子图。
Qwen3.5 必需差异为 Q/K head RMSNorm、partial RoPE、attention output
gate 和权重导入时的 `1 + gamma`。通用 `sigmoid_multiply` 在 VXM 上
展开为 negate/exp/add/reciprocal/multiply，不向 CModel 添加模型算子。
测试显式关闭 projection/RoPE overlap，采用 vector MXM 和 tail attention。

整个 decoder 层生成一个连续 ICU program，权重由真实 DDR/C2C
页传输与编译器生成的同步指令供给。中间张量由编译器管理 SRAM，
无 host 中间结果计算或传输。整层 QKV 输入直接采用共享 RMSNorm 的
`fp16_mxm_distributed_16` 布局；C++ attention lowering 将它转换到高地址
planar staging，串行 Q/K/V 复用该缓冲，避免 Q Norm/RoPE 写入 query-IW
时破坏后续 projection 的输入。O projection、residual、post RMSNorm
和 FFN 直接按共享布局衔接，不经过逐节点归档程序。最终输出由 C2C
从编译器规划的分布式存储导出为连续 BF16 张量。

`native_dense_gated_decoder_layer_runtime_test` 使用与真实权重测试相同的
整层入口，同时断言 FFN 被整体识别、额外 linear projection 只有一处，
并与独立数值参考比较。当前复制 gate 布局支持 seq32 单 token tile；
其他 token tile 数会明确报错，不静默套用该布局。

独立 `gated_attention_runtime_test` 使用 256 维 head、partial RoPE
和 GQA，直接比较 attention 输出，防止 residual 掩盖 attention 错误。
`gated_attention_decoder_layer_runtime_test` 另外覆盖完整 residual/FFN，
防止同名布局但 byte plane 数量不同的边被错误地直接 alias。

首次下载对应层的官方权重：

```powershell
python compiler/tools/download_official_dense_layer.py `
  --model Qwen/Qwen3.5-0.8B --layer 3 --embedding-rows 32 `
  --revision 2fc06364715b967f1860aea9cf38778875588b17 `
  --output-dir tmp/models/Qwen3.5-0.8B-layer3
```

```powershell
python compiler/tests/qwen3_5_full_attention_decoder_layer_test.py `
  --compile build-ftlpu-vs2026/compiler/ftlpu-compile.exe `
  --runner build-ftlpu-vs2026/runtime/compiled_stage_file_runner.exe `
  --model-dir tmp/models/Qwen3.5-0.8B-layer3 `
  --output-dir results/qwen3_5_0_8b_full_attention_decoder_layer_prefill `
  --target-config ../FTLPU-CMODEL/config/ftlpu-lpu32.json
```

输出目录包含 `summary.json`、`lowering_path.json`、`execution_identity.json`、编译器
Schedule、真实执行的 `resident.runtime.ftlpu`、`runtime.pipeline.csv`
以及按 ICU 分文件的 `runtime_icu_programs/`。只有完成 CModel 执行
并通过数值验收才写出 `summary.json` 的 `status: passed`。

Pipeline viewer：
`http://127.0.0.1:8765/tools/pipeline_viewer/index.html?csv=/results/qwen3_5_0_8b_full_attention_decoder_layer_prefill/runtime.pipeline.csv`。

实测数值和周期以本次输出的 `summary.json` 为准；旧 LogicalGraph 路径的
性能数据不作为此整层 dense 路径的结果。

## 2026-10-15 共享 dense 路径实测

官方第 3 层 seq32 完整 decoder 层通过：相同 W8A16 权重参考下
NRMSE **0.863038%**，cosine **0.99996605**。相对原始 BF16 checkpoint
的 NRMSE 为 8.973086%，单独 INT8 量化误差为 8.966343%。

一个连续程序、一次 runtime run，包含完整 FFN，`host_operations=0`。
8 个执行 epoch、15 次权重页预取，物理权重页共 18,350,080 bytes。
编译程序 1,269,925 bytes，实际 runtime program 1,274,009 bytes。
真实 pipeline 含最初输入和最终输出，最后事件结束于 cycle 452,454；
静态编译周期与 runtime 等待、边界传输周期分别统计。

新增整层 gated dense 数值回归、Qwen2.5 两种权重 bank 的整层 IR
回归、共享 RMSNorm 数值回归、weight page builder 共 5 项通过。
新增回归还验证整层直接编译与分阶段编译的程序逐字节一致，检查
mask 与权重页隔离，以及 Q/K norm gamma 的 bank 锁定。
