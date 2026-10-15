# HF 模型前端验证记录

更新（2026-10-14）：已在独立 `FTLPU-XLA` WSL2 环境完成真实 SmolLM2 forward
与 MLP 子图的 XLA 转换、保存、数值比较和保存产物复用；Linux 回归 38 项
通过。当前结果及左 padding/后端限制见
[Linux 环境与 bundle 复用记录](../tools/README.xla-linux.zh-CN.md)。
下文保留 2026-10-13 在 Windows 环境中的历史检查结果。

日期：2026-10-13。范围：配置/HF 加载、torch.export 捕获、原生 XLA bundle
索引与校验、官方 Safetensors reader 迁移和现有 compiler 的选定路径。

## 环境

- Windows 11，Python 3.13.9。
- 仓库 `.cache/frontend-python` 中安装验证依赖：torch 2.14.1+cpu、
  Transformers 4.57.1、Safetensors 0.8.0、NumPy 2.3.5、ml_dtypes 0.6.0。
  没有修改用户全局 Python 环境。Anaconda NumPy/MKL 与 torch 最初发生 OpenMP
  冲突，使用仓库目录内官方 NumPy wheel 解决，没有开启重复 OpenMP runtime
  的忽略开关。
- 没有 torch-xla；WSL 查询返回 `E_ACCESSDENIED`。未配置可运行的 Linux XLA
  环境。本次没有声称 StableHLO 自动导出成功。
- 使用已有 `build-codex-macro-vs-verify/compiler/Release/ftlpu_opt.exe`，未重新
  构建用户正在修改的 compiler/runtime。
- 根目录和相关 tools/compiler/docs 路径没有发现适用的 AGENTS.md。实施前
  检查了 README、compiler 架构文档、现有 exporter、importer 和工作区 diff。

## 已完成

运行：

```powershell
$env:PYTHONPATH = (Resolve-Path .cache/frontend-python).Path
python -m pytest -q compiler/tests/frontend_bundle_test.py `
  compiler/tests/frontend_weights_test.py compiler/tests/frontend_capture_test.py `
  compiler/tests/import_hf_decoder_layer_math_test.py
```

结果：**35 passed，1 skipped**。其中 XLA 集成测试显式跳过。

覆盖内容：

- 根据 `.meta.input_locations` 关联实际函数参数位置，包含与捕获顺序不同的
  参数重排、未引用状态、常量及未使用输入。
- 文件缺失、参数 shape/dtype 错误、映射遗漏/错误、manifest 篡改、常量缺失、
  额外状态文件、文本 IR 输入/输出签名不一致和 bytecode 摘要不一致均报错。
  这些测试使用协议 fixture，bytecode 是非空测试占位符，不是实际 XLA 产物。
- BF16 全部 65,536 种位模式逐位兼容，包含 signed zero、Inf、NaN payload；
  F16/F32/I8、HF shard index 路由和旧 importer re-export 接口兼容。
- 原有 importer/golden 的 BF16 scale、bias/RoPE 舍入、RMSNorm/softmax 等
  数值测试继续通过；原量化与舍入算法没有修改。
- 小型离线随机 checkpoint 的真实 Llama/Qwen2 实现均能通过同一个 HF loader
  和 logits wrapper 捕获，框架与捕获结果通过 `torch.testing.assert_close`。
  测试包含非平凡 attention mask、Qwen2 projection bias、Llama tied weights、
  RoPE 非持久 buffer 和 BF16 checkpoint → float32 加载的实际 cast 记录。
- 隐式 buffer/input mutation 被拒绝，显式 tensor 状态输入/输出被保留。
- 配置错误、转换错误和非空输出保护测试确认 traceback/捕获图及既有文件
  不被覆盖。

### 真实 pretrained TinyLlama

使用已有 `.cache/hf/TinyLlama-1.1B-Chat-v1.0` checkpoint，只修改配置的模型
目录、local-files-only 和 `1×2` 输入 shape，复用默认 float32/eager/logits
入口。没有增加 TinyLlama 专用数学实现或入口 wrapper。

本地观察脚本 `out/frontend-validation/validate_tinyllama.py` 调用真实 HF loader
一次，在实际 exporter 调用前观察数值，随后调用生产 `torch_xla_export`，
不使用模拟的 XLA converter。结果：

| 检查 | 结果 |
| --- | --- |
| HF 加载 | 通过，约 7.88 秒 |
| 完整 forward torch.export 捕获 | 通过，约 15.84 秒 |
| 完整 logits：框架 vs 捕获 | `1×2×32000` float32，最大绝对误差 0 |
| 真实 layer 0 MLP：框架 vs 捕获 | `1×2×2048` float32，最大绝对误差 0 |
| XLA 转换 | 失败：`ModuleNotFoundError: No module named 'torch_xla'` |
| StableHLO bundle 与导出执行数值 | 未完成 |

原始日志、捕获图、签名、数值观察和错误位于忽略目录
`out/frontend-validation/tinyllama-capture/`。报告明确是 conversion 失败，
没有 `manifest.json` 或成功 bundle。这些本地产物没有加入源码。

### Reader 迁移前后

迁移前从 HEAD 保存旧 reader 到忽略目录，在真实 TinyLlama checkpoint 上
比较以下张量的 float32 位模式：

- `model.layers.0.input_layernorm.weight`
- `model.layers.0.self_attn.q_proj.weight`
- `model.layers.0.mlp.gate_proj.weight`

全部逐位相等。Q projection 和 FFN gate 的原 `quantize_linear` 输出 int8
及 scale 也完全相等。结果保存在
`out/frontend-validation/reader_comparison.json` 和
`out/frontend-validation/quantization_comparison.json`。

### Compiler 解析和选定 lowering

```powershell
python compiler/tests/stablehlo_matmul_tensor_mlir_test.py `
  --tool build-codex-macro-vs-verify/compiler/Release/ftlpu_opt.exe `
  --input compiler/examples/iree_frontend/simple_stablehlo.mlir `
  --output out/frontend-validation/matmul.tensor.mlir
```

退出码 0，验证了现有 StableHLO parser、matmul → Tensor IR、物理 allocation
和 parse/print roundtrip。输入为仓库已有 StableHLO fixture；这不能代替自动
导出的 pretrained 模型图的后端验证。

## 没有完成或尚不支持

- 本机没有执行真实 PyTorch/XLA `save()`、portable bytecode 反序列化、StableHLO
  执行数值检查或自动模型图的 compiler lowering。原生布局和 metadata 已核对
  上游源码，协议测试不作为真实导出成功的证据。
- 未下载/执行 SmolLM2-135M 本身；保留了旧入口默认值，并通过同架构真实
  TinyLlama 和离线 HF 模型检查连接行为。Linux 上仍需执行 SmolLM2/XLA 验证。
- 未执行 BF16 的 XLA 导出；上游默认 `.numpy()` 序列化能力可能限制该路径。
- 不支持通用全图 W8A16、HF 量化加载、隐式可变状态、动态 shape、多公开函数
  或未经适配的 exporter metadata。量化请求明确报错。
- bundle 的文本 signature reader 不验证整个 MLIR body；应额外运行
  `stablehlo-opt` 或现有 compiler。原生 bytecode 目前只检查存在性和摘要。
- 未修改 runtime、ModelPackage、后端架构、数学实现或全部旧脚本；原先已有
  compiler/runtime 修改与 importer 的 `token_id_offset` 工作均被保留。

Linux 下一步：安装匹配 torch/torch-xla，设置 `PJRT_DEVICE=CPU`，运行
`tools/export_hf_stablehlo.py --config tools/configs/smollm2_stablehlo.json
--output <新目录> --verify-numerics`，再用现有 compiler 的选定 pipeline 检查
生成的 `functions/forward.mlir`。XLA 集成测试已提供，依赖存在时会真实执行
HF → capture → XLA save → bundle 校验 → 重载产物数值对比。
