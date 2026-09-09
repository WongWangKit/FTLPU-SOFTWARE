# Qwen3-0.6B 单层 prefill

这里提供 Qwen3-0.6B 的 seq=32 完整 decoder layer prefill。模型参数为
hidden size 1024、intermediate size 3072、16 个 Q head、8 个 KV head、
head dimension 128、RoPE theta 1,000,000 和 RMSNorm epsilon 1e-6。

Qwen3 的 Q 投影宽度是 `16 × 128 = 2048`，K/V 投影宽度是 1024，O 投影
因此使用 `2048 × 1024` 权重。Q/K 在 RoPE 前分别执行逐 head RMSNorm，参数来自
`self_attn.q_norm.weight` 和 `self_attn.k_norm.weight`。Attention 投影不含 bias。

## 生成固定 StableHLO

```powershell
python compiler/tools/generate_qwen3_0_6b_decoder_layer.py `
  --output compiler/examples/qwen3_0_6b_decoder_layer/decoder_layer_seq32.stablehlo.mlir
```

生成器保留真实层尺寸，同时使用适合编译回归的固定参数占位符。真实 checkpoint
可通过 `import_hf_decoder_layer.py` 量化并打包；metadata 会记录
`head_dim=128`、`query_width=2048`、`kv_width=1024` 和 `qk_norm=true`。

## 编译和 C2C CModel 数值验证

配置 Visual Studio 构建目录后，可用一个目标完成 StableHLO、Stream、压缩
Schedule、二进制和 `ModelSession(C2cDmaSystem)` 数值闭环：

```powershell
cmake --build build-ftlpu-vs2026 --config Release `
  --target qwen3_0_6b_decoder_layer_seq32_prefill_cmodel_test
```

测试使用确定性输入和稀疏 INT8 权重，覆盖两次层 RMSNorm、Q/K 逐 head
RMSNorm、RoPE、16:8 GQA causal attention、2048→1024 O 投影、两个残差以及
1024/3072 SwiGLU FFN，并比较全部 `32 × 1024` 个 BF16 输出。当前硬件配置下
生成的二进制约 845 KB，整层调度结束于约 28.7 万 cycle；允许的最大绝对误差为
0.125。外部输入、输出、RMSNorm 参数和 Attention/FFN 分页权重都经过
`DDR4 -> C2C DMA -> C2C RX -> shared SR -> MEM`，不会由 host 直接写入 SRAM。
同一 `(hemisphere, slice, bank)` 即使使用不重叠的 row，也共享单端口 MEM 队列；
runtime 会把页写入排到该端口之前的计算释放之后。Down projection 按 output wave
切成 8 个可独立消费的 tile，物理位置在 bank 1/0 之间交替并复用同一 row slot；
计算 tile `i` 时，C2C 把 tile `i+1` 写进另一个 bank，tile `i-1` 释放后其 slot
即可装入 tile `i+1`。只有 tile 到达 consumer cycle 仍未 SRAM-ready 时才暂停计算 ICU。
当前默认 DDR/C2C 配置下，完整 C2C CModel 的 executable 内等待为 0 cycle。

## Pipeline Viewer

设置 `FTLPU_QWEN_PIPELINE_CSV` 后运行数值测试，可以输出 C2C CModel 实际执行
trace：

```powershell
$env:FTLPU_QWEN_PIPELINE_CSV = `
  "../qwen3_0_6b_seq32_prefill.runtime.c2c.csv"
build-ftlpu-vs2026/runtime/compiled_qwen3_0_6b_decoder_layer_seq32_runtime_test.exe `
  build-ftlpu-vs2026/compiler/ftlpu_lower/qwen3_0_6b_decoder_layer_prefill/decoder_layer.ftlpu
```

用浏览器打开 `tools/pipeline_viewer/index.html`，单击“打开 CSV”或把
`qwen3_0_6b_seq32_prefill.runtime.c2c.csv` 拖入页面。实际 trace 记录真实 physical
cycle，并包含 `C2C.*.DMA/RX/Prefetch`、`SR.*.C2C.Shared`、
`MEM.*.C2CWrite`、`C2C.HostInput/HostOutput` 和 `ICU.PageReadyWait`；使用
`ftlpu_binary_inspect --trace` 生成的 `decoder_layer.schedule.csv` 可用于检查离线
调度计划。
