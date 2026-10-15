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

Python 只提供 checkpoint 导入、逻辑 Kernel 图和独立参考方程。
Q/K/V、QK norm、RoPE、QK、softmax、PV、output gate、O 走 C++
原生 dense-attention lowering；其余算子复用 C++ logical graph planner
与共享算子 lowering。通用 `sigmoid_multiply` 在 VXM 上展开为
negate/exp/add/reciprocal/multiply，不向 CModel 添加模型算子。

整个 decoder 层生成一个连续 ICU program，权重由真实 DDR/C2C
页传输与编译器生成的同步指令供给。中间张量由编译器管理 SRAM，
无 host 中间结果计算或传输。整层 QKV 输入直接采用共享 RMSNorm 的
`fp16_mxm_distributed_16` 布局，保留到全部 Q/K/V projection 完成；
每次 projection 刷新其短期 planar staging，避免 Q Norm/RoPE 写入
query-IW 时破坏 K/V 尚需的输入。传统 planar QKV 输入的四个 plane 表示同一个
K32 BF16 向量的两份副本，供两个 MXM activation 端口读取；O 输出
的四个 plane 则分布不同 N32 输出块。两种布局使用不同 kind，不能
将输入副本误当成按特征切分的两组数据，也不能当成 FP32 的四个
字节。外部输出在 DDR 边界转换为单个连续 BF16 pair。

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
  --graph-compile build-ftlpu-vs2026/compiler/ftlpu-graph-compile.exe `
  --runner build-ftlpu-vs2026/runtime/compiled_stage_file_runner.exe `
  --model-dir tmp/models/Qwen3.5-0.8B-layer3 `
  --output-dir results/qwen3_5_0_8b_full_attention_decoder_layer_prefill `
  --target-config ../FTLPU-CMODEL/config/ftlpu-lpu32.json
```

输出目录包含 `summary.json`、`execution_identity.json`、编译器
Schedule、真实执行的 `resident.runtime.ftlpu`、`runtime.pipeline.csv`
以及按 ICU 分文件的 `runtime_icu_programs/`。只有完成 CModel 执行
并通过数值验收才写出 `summary.json` 的 `status: passed`。

Pipeline viewer：
`http://127.0.0.1:8765/tools/pipeline_viewer/index.html?csv=/results/qwen3_5_0_8b_full_attention_decoder_layer_prefill/runtime.pipeline.csv`。

## 2026-10-14 实测

官方第 3 层 seq32 整层数值通过。相同 W8A16 权重参考下 NRMSE
0.864829%、cosine 0.99996585；相对原始 BF16 checkpoint 的 NRMSE
8.975423%，其中单独量化造成的 NRMSE 为 8.966343%。

编译程序 1,161,820 bytes，实际 runtime program 1,163,792 bytes。
9 个执行 epoch、11 次权重页预取，共传输 18,350,080 bytes 的物理
权重页。`host_operations=0`，一个连续程序、一次 runtime run。
实际流水 CSV 包含 MEM、MXM、VXM、SXM、SR、C2C 和 ICU 同步事件，
也包含最初的数据传入；最后事件结束于 cycle 957,717。

回归：256 hidden 的独立 gated attention NRMSE 0.63468%；含完整
residual/FFN 的 decoder 回归 NRMSE 0.73259%。权重页打包、packed
vector graph、vector memory plan 三项回归全部通过。
