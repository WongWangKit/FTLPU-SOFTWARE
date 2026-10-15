# 配置驱动的 Hugging Face → StableHLO 前端

统一入口为 `tools/export_hf_stablehlo.py`。JSON 指定 Hugging Face 模型 ID
或本地 `save_pretrained` 目录、加载选项和静态输入规格。前端调用真实的
Transformers 模型，通过 `torch.export` 捕获，再用 PyTorch/XLA 导出原生
StableHLO bundle。默认入口输出 logits，关闭 KV cache；分词、采样和生成循环
由调用方处理。

## 使用

需要 Linux 上版本匹配的 PyTorch 和 PyTorch/XLA；Windows 请使用 WSL2。
环境和 PowerShell 包装器见 [WSL/Linux 指南](README.xla-linux.zh-CN.md)。

在仓库根目录运行：

```bash
PJRT_DEVICE=CPU python tools/export_hf_stablehlo.py \
  --config tools/configs/smollm2_stablehlo.json \
  --output artifacts/smollm2-stablehlo --verify-numerics
```

默认示例允许从 HF 下载。使用本地模型时创建自己的 JSON，例如：

```json
{
  "schema_version": 1,
  "model": "/models/my-llama-checkpoint",
  "loader": "AutoModelForCausalLM",
  "load_dtype": "float32",
  "load_options": {"attn_implementation": "eager", "local_files_only": true},
  "entry": "causal_lm_logits",
  "inputs": [
    {"name": "input_ids", "shape": [1, 16], "dtype": "int64", "fill": 1},
    {"name": "attention_mask", "shape": [1, 16], "dtype": "int64", "fill": 1}
  ],
  "quantization": "none",
  "exporter": "torch_xla"
}
```

HF 模型 ID 可以设置 `revision` 固定提交，以及
`load_options.cache_dir` 指定缓存。完全离线时再设置
`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`。JSON 支持 UTF-8 和 UTF-8 BOM；
WSL 中的模型目录、缓存目录必须使用 Linux 可访问的路径。
所有相对路径按命令的工作目录解析。输出目录必须为空，防止覆盖已有结果。

`tools/configs/*.verified.json` 固定了 SmolLM2 的 revision，并默认使用
仓库工作目录下的 `.cache/hf-hub`。文件名表示实验中的验证配置，使用前仍需准备
对应缓存并在当前环境重新验证。`tinyllama_stablehlo.json` 提供另一种模型配置。

## 配置约束与入口扩展

- `load_dtype` 支持 `float32`、`float16`、`bfloat16`。检查点存储类型与实际加载
  类型分别记录在 manifest，不能将 FP32 加载描述为保持 BF16 存储类型。
- 默认 `causal_lm_logits` 要求两个同形状、二维 `int64` 输入。
  `fill` 生成静态示例输入；导出的 ABI 接受同形状和类型的其他张量。
  数值验证只覆盖这些示例输入。
- `strict` 默认 `false`，交给 `torch.export`；未知顶层字段直接报错。
  模型本身是否能捕获、算子是否受 XLA 支持，需要实际导出验证。
- `load_options` 透传 HF 加载参数；不要在其中重复指定 `revision`、
  `torch_dtype` 或 `dtype`。当前 CPU 捕获路径拒绝 `device_map`、`state_dict`
  和隐式 HF 量化加载。
- 自定义入口写为 `module:factory`。工厂接收 `(model, entry_options)`，
  返回 `torch.nn.Module`，输入按 JSON 顺序传入。模块必须在 Python 导入路径中。
  工厂是会执行的 Python 代码，只使用可信配置和模块。
- 真实子模块可使用 `ftlpu_frontend.hf:submodule_entry`，并设置
  `entry_options.module_path`，例如 `model.layers.0.mlp`；输入规格需改为
  该子模块的实际张量接口。

默认 `quantization=none`。附带的 `legacy_w8a16_bf16` 是显式 decoder profile，
仅用于与已有 W8A16/BF16 路径对齐；它不代表任意 HF 模型都支持量化导出。
对应配置为 `smollm2_decoder_w8a16.verified.json`。
`legacy_bf16_checkpoint` 使用单独记录的历史 BF16 判据，不能等同于默认的
`strict_torch_close` 验证通过。

## 产物与验证

供后续编译器读取的 StableHLO 文本是 **`functions/forward.mlir`**，而非固定名
`stablehlo.mlir`。bundle 保留 XLA 的原生目录结构：

| 文件 | 用途 |
| --- | --- |
| `functions/forward.mlir`、`forward.bytecode`、`forward.meta` | IR、可移植字节码及实际函数 ABI |
| `data/`、`constants/` | 原生参数、buffer 与常量文件 |
| `manifest.json` | 来源、类型变化、张量和函数参数的位置映射 |
| `configuration.json`、`capture_signature.json`、`capture_graph.txt` | 有效配置和捕获证据 |
| `report.json`、`frontend.log` | 分阶段结果、工具版本和耗时 |
| `error.txt` | 失败时保留原异常和调用栈 |
| `validation/`、`bundle_checksums.json` | 数值验证输入/输出与通过验证后的 SHA-256 清单 |

映射依据 XLA `input_locations`，不能假设参数顺序等于 HF `state_dict` 顺序。
捕获阶段拒绝隐式输入/buffer 修改；有状态接口需将状态显式作为张量输入输出。

```bash
# 结构、ABI、张量文件、类型及已有校验和检查；不加载 HF
python tools/export_hf_stablehlo.py --validate-bundle artifacts/smollm2-stablehlo

# 重载已通过数值验证的 XLA bundle 并比对保存的参考输出；不重新捕获 HF
PJRT_DEVICE=CPU python tools/run_saved_hf_bundle.py \
  --bundle artifacts/smollm2-stablehlo
```

`--verify-numerics` 比较原始 HF、捕获图和从磁盘重载的 XLA 程序。
默认判据为 `rtol=1e-4, atol=1e-5`。只有序列化程序验证成功才生成校验和清单。
结构验证检查函数签名和产物一致性，不能代替完整 MLIR 验证、数值验证或后端验证。

可以传 `--backend-tool /path/to/ftlpu-opt` 和 `--backend-pipeline` 做额外的
后端尝试，结果独立记录。前端导出成功不保证 main 能将整个模型降到 Schedule
或在 CModel 执行；本次迁移未包含实验分支的后端适配和 ModelPackage 桥接。

原有 `tools/export_smollm2_stablehlo.py` 的参数、默认值和 `export.json` 保留，
内部复用同一导出实现。

## 回归测试

普通 Python 环境安装 `pytest numpy` 可运行协议、JSON 和命令行检查：

```bash
python -m pytest -q -p no:cacheprovider \
  compiler/tests/frontend_bundle_test.py compiler/tests/frontend_entry_test.py -k 'not xla'
```

配置 CMake 时检测到这些依赖，会注册 `hf_frontend_protocol_test`。
权重兼容测试还需 `safetensors>=0.8 ml_dtypes`。完整导出测试需 XLA 环境：

```bash
PJRT_DEVICE=CPU HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -m pytest -q -p no:cacheprovider compiler/tests/frontend_*test.py \
  compiler/tests/import_hf_decoder_layer_math_test.py
```

完整测试会在临时目录生成小型真实 HF Llama/Qwen 检查点，验证捕获、原生 XLA
保存、数值比对、复用与损坏检测，不下载预训练模型。
