# 配置驱动的 HF → StableHLO 前端

这是一层轻量连接代码，位于 `compiler/tools/ftlpu_frontend/`。它连接 HF
`from_pretrained`、真实模型的薄入口、`torch.export.export(strict=False)`、
PyTorch/XLA 默认 `exported_program_to_stablehlo()` / `save()`，然后建立和校验
bundle 索引。Attention、mask、RoPE、FFN 和 bias 使用 Transformers 实现。

`tools/export_smollm2_stablehlo.py` 保留原来的 CLI、float32、eager attention、
固定 shape、logits-only、无 KV cache 的默认行为，以及旧 `export.json`。
新旧入口共享连接代码；没有新增 exporter 实现、量化框架或 runtime ABI。

2026-10-14 已配置独立的 `FTLPU-XLA` WSL2 环境，完成真实 SmolLM2 导出、
原生保存及独立进程复用。集中环境、依赖锁、已验证 bundle 和日常命令见
[Linux 导出环境与 bundle 复用](README.xla-linux.zh-CN.md)。

## 配置与运行

在支持 PyTorch/XLA 的 Linux/WSL 环境安装匹配版本的 `torch` / `torch-xla`，
以及 `tools/requirements.frontend.txt` 中的其他依赖。Windows 原生缺少 XLA
时可以运行协议和读取测试，但不能把捕获成功视为 StableHLO 导出成功。

```bash
PJRT_DEVICE=CPU python tools/export_hf_stablehlo.py \
  --config tools/configs/smollm2_stablehlo.json \
  --output out/smollm2-bundle \
  --verify-numerics

# 相同接口模型只修改 model / revision / shape / load_options。
PJRT_DEVICE=CPU python tools/export_hf_stablehlo.py \
  --config tools/configs/tinyllama_stablehlo.json \
  --output out/tinyllama-bundle

python tools/export_hf_stablehlo.py --validate-bundle out/smollm2-bundle
```

输出必须是新目录或空目录，失败产物也不会被下一次运行覆盖。修改 shape 后
导出到另一个目录。模型可使用本地目录；设置 `load_options.local_files_only=true`
禁止下载。`revision` 可使用固定 commit；manifest 同时记录请求 revision 和 HF
实际解析的 commit（本地目录可能没有 commit）。默认样例值为零 token 和全一 mask；
数值检查只覆盖生成的样例输入，不能代替多输入模型验证。

核心配置示例：

```json
{
  "schema_version": 1,
  "model": "HuggingFaceTB/SmolLM2-135M",
  "revision": null,
  "loader": "AutoModelForCausalLM",
  "load_dtype": "float32",
  "load_options": {"attn_implementation": "eager", "local_files_only": true},
  "entry": "causal_lm_logits",
  "inputs": [
    {"name": "input_ids", "shape": [1, 16], "dtype": "int64", "fill": 0},
    {"name": "attention_mask", "shape": [1, 16], "dtype": "int64", "fill": 1}
  ],
  "quantization": "none",
  "exporter": "torch_xla",
  "strict": false
}
```

`load_options` 直接传给 HF loader，可设置 `cache_dir`、`use_safetensors`、
`variant` 或显式 `trust_remote_code`。dtype 和 revision 使用专用字段。
当前 CPU 捕获不支持 `device_map`、外部 `state_dict` 或量化加载选项。

接口不同的模型使用 `entry="your_module:make_entry"`。工厂签名为
`make_entry(model, entry_options) -> torch.nn.Module`，配置中的 `inputs` 按列表
顺序作为位置参数传入。wrapper 负责真实模型的输入/输出整理；所有叶子必须
是 tensor，shape 静态。工厂模块可放在 `compiler/tools/`，或将其目录加入
`PYTHONPATH`。比如使用真实子模块而不重写其数学：

```python
def make_entry(model, options):
    return model.get_submodule(options["module_path"])
```

状态入口应以 tensor 输入并以 tensor 输出返回新状态。当前 XLA 转换会丢弃
隐式 buffer mutation 输出，因此前端会明确拒绝这类捕获图，也拒绝隐式输入
mutation。不要删掉状态更新或将 `use_cache=False` 当作有状态模型的适配方法。
内置 `causal_lm_logits` 本来就是无 cache 的 forward 接口。

其他 exporter 的 Python 接口保持简单：

```python
from ftlpu_frontend import export_model

export_model(config, output, exporter=my_exporter)
# my_exporter(exported_program, output: Path) 写入下述原生产物布局。
# 如需 --verify-numerics 对应能力，可返回与 wrapper 同接口的可调用对象。
```

这是一个可注入函数，不是插件注册系统；本次只实现默认 XLA exporter。

## Bundle v1

已核对上游 [PyTorch/XLA save 与 metadata 实现](https://github.com/pytorch/xla/blob/v2.7.0/torch_xla/stablehlo.py)，
并与当前上游实现比对。原生格式已经提供权重、额外常量、函数输入来源及
输入输出 shape/dtype；本项目保留这些产物，补充来源、捕获签名、数值记录和
版本化索引。升级 XLA 后若格式不同，校验会报错，不能猜测新的 ABI。

```text
bundle/
  functions/forward.mlir       # 原生 StableHLO 文本，内部入口通常是 @main
  functions/forward.bytecode   # 原生 portable artifact
  functions/forward.meta       # 原生输入位置、shape/dtype、unused_inputs、pytree
  data/<captured_target>       # 原生 NPY，文件没有 .npy 后缀
  constants/<index>            # 原生额外常量 NPY；内联常量留在图中
  manifest.json               # ftlpu.frontend.bundle，schema_version=1
  configuration.json          # 完整生效配置
  capture_signature.json      # 捕获图的参数/buffer/constant/user input/output
  capture_graph.txt           # 原始捕获图，转换失败时仍保留
  conversion.mlir             # XLA 已返回图时保存的转换诊断副本
  report.json / frontend.log  # 阶段、耗时、版本、状态、路径
  error.txt                   # 失败时的原始 traceback
  validation/reference_*.npy  # 数值检查通过后保存的框架参考输出
  bundle_checksums.json       # 数值验证通过后生成的内容完整性记录
```

`manifest.functions[].arguments[].position` 是实际 StableHLO 函数参数位置。
其 `source` 来自 XLA `.meta.input_locations`，可以是 parameter、input_arg 或
constant。`tensors[].function_arguments` 用捕获签名的 target 与这些位置关联，
不按 state_dict、checkpoint 或名称排序猜测。HF 参数别名根据真实 tensor
对象身份记录，包含 tied parameters。未被 exporter metadata 引用的状态有空
位置列表和显式标记，不推断它究竟是被剪枝还是内联。

校验包含：必要文件非空、NPY 的 shape/dtype、捕获状态完整性、函数参数来源
完整性、未使用输入记录、输出规格、实际文本 IR 函数签名以及图文件摘要。
读取原生 NPY 使用 NumPy mmap，只读取 header，不再次扫描全部权重。
签名检查仅支持静态 ranked tensor 和一个公开入口，不替代 MLIR body verifier；
bytecode 检查存在性及保存后摘要，未独立反序列化验证其 body。
`StableHLOFunctionMeta.stablehlo_version` 的上游值可能为 `0.0.0`，不把它当作
实际工具版本；另行记录安装的 Python、torch、torch-xla、Transformers 等版本。

当前 compiler 继续消费 `manifest.functions[0].mlir`，不必修改 C++ driver：

```bash
stablehlo-opt out/smollm2-bundle/functions/forward.mlir -o /dev/null
ftlpu-opt --input out/smollm2-bundle/functions/forward.mlir \
  --output out/smollm2.kernel.mlir --pipeline ftlpu-stablehlo-to-kernel

# 也可让前端运行一次现有后端并保存其原始 stdout/stderr 和失败阶段：
python tools/export_hf_stablehlo.py --config tools/configs/smollm2_stablehlo.json \
  --output out/smollm2-with-backend --backend-tool /path/to/ftlpu-opt \
  --backend-pipeline ftlpu-stablehlo-to-kernel
```

manifest 是前端逻辑关联协议。后端按函数参数位置选择对应 tensor，继续负责
物理 layout、SRAM placement、量化 lowering 和最终 runtime binding。原生 NPY
不能直接认作硬件 runtime buffer。显式 decoder profile 可以经
[`lower_hf_bundle.py`](lower_hf_bundle.py) 复用现有 ModelPackage 写入器，详见
[连接与验证](../docs/hf_connection_validation.zh-CN.md)。现有后端仍可能
不支持全模型图中的部分算子或 dtype，失败会记录在 `backend` 阶段。

## 数值阶段与权重读取

manifest 区分四个阶段：

1. `checkpoint_storage` 通过官方 Safetensors metadata 记录存储 dtype/shape。
   使用非 Safetensors checkpoint 时标记 unknown，不拿 HF config dtype 猜测。
2. `requested_load_dtype` 与 `observed_state_dtypes` 记录 HF 的加载/运行类型。
   只对能由捕获 tensor 身份与 checkpoint 名称关联的实际 dtype cast 生成记录。
3. 默认为 `quantization=none`。另外提供显式 `legacy_w8a16_bf16` decoder profile，
   复用旧 quantizer，同步变换七个 Linear 的计算、INT8 状态、scale、舍入和捕获映射。
   HF 实现继续负责模型数学。它不支持通用全图 W8A16；任意量化配置、HF 量化加载
   及未经支持的已量化模型仍拒绝，不能只换权重文件。
4. `export_compute_policy` 保留 XLA 默认转换；`export_signature_dtypes` 记录
   函数 ABI，`export_tensor_dtypes` 记录文本 IR 中实际 tensor 类型，包括混合类型。
   不执行额外全图 dtype 改写，也不宣称做了硬件 BF16/FP16 舍入。

默认无量化 float32 路径保持原转换/save 行为。显式 decoder profile 在固定的
torch-xla 2.9 上用一个 BF16 NumPy 互操作适配器：转换时设 `export_weights=False`，
随后按捕获签名提供 BF16 bits，并继续调用原生 `save()`。保存的 V2 NPY 只有在
捕获/原生 metadata/IR ABI 都声明 BF16 时才恢复 BF16。其他版本明确拒绝这个
适配器；没有静默改成 FP32。该路径已实际导出、重载和复用验证。

`import_hf_decoder_layer.SafeTensorFile/Store` 的构造与 `read(name)` 接口保留，
改用官方 `safe_open`，薄封装只负责 shard index 路由及原有 float32 返回值。
BF16 使用 `ml_dtypes` 注册 NumPy dtype，并保持原位展开，包括 NaN payload。
需要安装 `safetensors` 和 `ml_dtypes`。原 `quantize_linear`、`bf16_bits`、golden
算法及默认数值行为没有改动；依赖它的 FFN/stack 脚本自然复用 reader，不需要
一次重写全部旧脚本。前端 HF 路径不会再次 `read()` 所有已加载权重。

## 验证

构建启用静态 mask reference interpreter 的 compiler 前，先按
[StableHLO 兼容补丁说明](../compiler/patches/README.md) 处理固定版本的 MLIR API
兼容性；父仓库记录补丁和重放命令，不把本地脏子模块当作已上传的修改。

```bash
python -m pip install pytest
python -m pytest -q compiler/tests/frontend_bundle_test.py \
  compiler/tests/frontend_weights_test.py compiler/tests/frontend_capture_test.py \
  compiler/tests/import_hf_decoder_layer_math_test.py
```

HF 捕获测试用离线、小型随机权重的真实 Llama/Qwen2 实现验证配置接入、mask、
bias、torch.export 数值一致性和显式状态接口，不等同于 pretrained 模型验证。
实际本机运行结果与没有完成的检查见
[前端验证记录](../docs/hf_frontend_validation.zh-CN.md)。
