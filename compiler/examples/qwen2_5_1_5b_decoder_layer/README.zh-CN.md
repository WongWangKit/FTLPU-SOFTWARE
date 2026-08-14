# Qwen2.5-1.5B decoder 层

本目录存放 `seq_len=128` 的标准 StableHLO decoder 整层测试。模型尺寸与
Qwen2.5-1.5B 一致：hidden size 1536、intermediate size 8960、12 个 Q head、
2 个 KV head、head dimension 128、RoPE theta 1,000,000，以及 RMSNorm epsilon
1e-6。

编译流程是通用的，模型名只出现在测试输入和测试名称中。HF 权重导入器已经读取并
用于 golden 计算 Qwen 的 Q/K/V projection bias；LPU 端 bias add 的正式调度接入与
这里验证的标准 attention/FFN 图分开跟踪。

默认 CTest `qwen2_5_1_5b_decoder_layer_stablehlo_to_stream_ir_test` 快速验证
StableHLO、Kernel、Tensor 和 Stream IR。完整可执行链路是重型构建目标：

`qwen2_5_1_5b_paged_weight_layout_test` 额外验证双 bank 分页部署的 Tensor/Stream
物理布局。使用 `--weight-bank 0` 和 `--weight-bank 1` 可生成交替 bank 的 executable。

```powershell
cmake --build build-ftlpu-vs2026 --config Release `
  --target qwen2_5_1_5b_decoder_layer_pipeline
```

该目标生成 `decoder_layer.schedule.mlir`、`decoder_layer.command.mlir` 和
`decoder_layer.ftlpu`，并检查 MXM accumulator 地址符合 Vector 1024 行、
Block8 128 行的真实指令编码限制。当前 CModel 数值测试仍受最新 INT8 IW 边界
寄存器可见性规则阻塞，因此二进制生成成功不能视为真实 Qwen 权重 golden 已通过。
此外，当前完全展开的整层 Command MLIR 文本生成超过十分钟；分页布局和 45 MiB
权重 page 已通过测试，但实际部署应继续优化为从压缩 Schedule 直接生成 binary。
