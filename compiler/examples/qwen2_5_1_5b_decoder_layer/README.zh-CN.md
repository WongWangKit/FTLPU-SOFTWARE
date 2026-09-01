# Qwen2.5-1.5B decoder 层

本目录提供 `seq_len=32` 和 `seq_len=128` 两套标准 StableHLO decoder 层输入。
两者均采用 Qwen2.5-1.5B 的真实尺寸：hidden size 1536、intermediate size
8960、12 个 Q head、2 个 KV head、head dimension 128、RoPE theta 1,000,000，
以及 RMSNorm epsilon 1e-6。

编译流程与具体模型无关。输入只使用标准 StableHLO 算子，并依次 lower 到
Kernel IR、Tensor IR、Stream IR、压缩 Schedule IR 和二进制 Command。可执行文件
使用 Block8 MXM、VXM-feedback RMSNorm、双 bank INT8 权重分页和 BF16 激活。

使用以下命令生成 `seq_len=32` 可执行文件：

```powershell
python compiler/tests/qwen2_5_1_5b_decoder_layer_pipeline_test.py `
  --opt build-ftlpu-vs2026/compiler/ftlpu_opt.exe `
  --compile build-ftlpu-vs2026/compiler/ftlpu-compile.exe `
  --input compiler/examples/qwen2_5_1_5b_decoder_layer/decoder_layer_seq32.stablehlo.mlir `
  --target-config ../FTLPU-CMODEL/config/ftlpu-lpu32.json `
  --weight-bank 1 `
  --output-dir build-ftlpu-vs2026/compiler/ftlpu_lower/qwen2_5_1_5b_seq32
```

生成的 binary 包含 501,237 条压缩队列命令，最大调度周期为 192,382。CModel
数值测试已通过全部 49,152 个 BF16 输出，其中 49,127 个非零。分阶段最大误差为：
FFN 0.03125、attention residual 0、第二次 RMSNorm 0.03125。

这是一套尺寸真实、使用确定性稀疏 INT8 golden 权重的 decoder 单层测试，已经验证
完整 compiler/binary/runtime/CModel 链路；它还不是 Hugging Face 原始 checkpoint
全部 28 层的执行。

## 真实 checkpoint FFN

`compiler/tools/import_hf_ffn.py` 从 Hugging Face Qwen2.5-1.5B checkpoint
导入第 0 层真实 Gate、Up、Down 权重和 embedding 产生的 BF16 输入。导入过程执行
per-tensor INT8 量化，并生成符合硬件语义的 golden：BF16 MXM 反量化、FP32 partial
累加，以及 VXM FP16 LUT 实现的 SwiGLU。

在 `seq_len=32` 下，标准 StableHLO 输入已经逐层 lower 到 v22 分页 binary。
runtime 上传 26 个真实权重页、填充 ICU 队列，并在 CModel 上执行到 cycle 770,646。
49,152 个 BF16 输出全部通过数值比较：超阈值 mismatch 为 0，MAE 为
`6.54569e-05`，RMSE 为 `0.000266376`，最大误差为 `0.00418091`。
