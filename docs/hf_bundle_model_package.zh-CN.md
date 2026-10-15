# 捕获 bundle 到现有 ModelPackage 的连接

`compiler/tools/ftlpu_frontend/package_bridge.py` 和 `tools/lower_hf_bundle.py`
以真实 HF/XLA bundle 的 `functions/*.mlir` 为编译器输入。它们复用旧 decoder
数值参考与 ModelPackage 写入器，不使用旧手写 StableHLO 替换捕获图。

## 数值与语义边界

当前连接明确接受 `legacy_w8a16_bf16` profile v1：七个 Linear 的 INT8 权重、
各自的 scale、BF16 norm/bias/activation，位置从零开始、无 padding 的静态 causal
prefill、标准未 scaling 的 RoPE。shape 从 HF 配置取得；参数 target 从捕获签名取得。
padding mask、非零 position offset、RoPE scaling、修改过的 inv_freq、未保留的
Attention bias、未声明的权重存储布局都明确拒绝。

Profile 位于独立转换 bundle 的 `backend_profile.json`，包含：

- `architecture`：真实 HF config；`layer`、`sequence_length`：边界；
- `roles`：每个语义角色的捕获 target、HF 名、明确的 `in_out`/`out_in` 布局及 scale；
- `activation`：实际捕获输入的逻辑名和 `validation/input_0.npy`；
- `mask`、`position_range`、`rope.theta`、实际 `rope.inv_freq`；
- `immutable_targets`：允许特化为常量的明确 target，通常只有 inv_freq；
- `transformations`：实际执行的量化与 dtype/scale 舍入；
- `fp32_reference_file`：可选的未量化真实 HF 子图输出，用于区分量化和 lowering 误差。

旧后端有自己的 RMSNorm、Softmax LUT 和中间舍入规则。框架转换后的输出与旧 backend
golden 分别保存、分别比较，不宣称逐位等价。连接层不调整旧 CModel 验证阈值。

## 参数绑定

绑定分三步完成，不能用旧固定参数顺序代替新图签名：

1. Profile 的 target 与 bundle 原生 exporter metadata 相连，取得原函数参数位置。
2. 编译器的 `abi_map.json` 显式记录常量特化与未使用参数裁剪后的位置。
3. 读取实际 Command IR 的 input/output binding，核对每个逻辑角色的 index、shape、
   dtype、bytes，并检查完整覆盖。物理 slices、MEM row 和驻留分配仍由后端处理。

核对结果保存为 `command_binding_map.json`。旧
`build_decoder_layer_package.py` 仅增加可选的 metadata 字段
`executable_input_bindings`；不提供时仍使用旧 ABI 顺序。

BF16 NPY 会由 NumPy 保存为 `V2` header。只有 bundle/ABI 明确声明 BF16 且 payload
是两字节 void 时，连接层才恢复 `ml_dtypes.bfloat16` view；不会把任意两字节数据
猜成 BF16。此过程保留所有 BF16 bits，包括负零及 NaN payload。

## 运行

Linux/WSL 继续负责 HF/XLA 导出；Windows 上运行 compiler 和 CModel。使用现有
Windows 依赖目录，无需重新安装 XLA：

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) '.cache/frontend-python'
python tools/lower_hf_bundle.py `
  --bundle out/hf-bundles/smollm2-layer0-w8a16-s128-connection-v5-20261014 `
  --opt build-codex-macro-vs-verify/compiler/Release/ftlpu_opt.exe `
  --translate build-codex-macro-vs-verify/compiler/Release/ftlpu-translate.exe `
  --runtime-test build-codex-macro-vs-verify/runtime/Release/hf_decoder_layer_model_session_test.exe `
  --target-config ../FTLPU-CMODEL/config/ftlpu-lpu32.json `
  --output-dir out/hf-connection-20261014/smollm2-layer0
```

输出目录必须尚不存在。入口保存生效 profile、工具 SHA256、target SHA256，生成
constant/profile JSON，使用 `--kernel-mode complete`，保留 Kernel、Tensor、Stream、
compressed Schedule、独立 Schedule verification、Command IR，以及各阶段的快照、耗时、原始 stdout/stderr 和
失败 traceback。任何阶段失败都会停止后续工作，不把混合 IR 当成成功。

达到 Command 后，连接层从已保存的 bundle 权重生成旧格式 `package_inputs/`，
调用旧 package writer，再由现有 `ModelSession -> CModel` 执行。CModel 测试新增可选
输出文件参数，默认旧调用与阈值不变。产物包含 `model.ftlpum`、`decoder.ftlpu`、
完整 `cmodel_output.bf16.bin` 和 `numerical_comparisons.json`。

完整数组分别比较 CModel、转换后的真实框架参考、旧 backend golden；提供 FP32
子图参考时再增加 FP32 对照。报告中的 `measured` 指误差已测量，接受标准是既有
CModel test 的标准，不等同于任意输入或完整模型已得到证明。

多个成功的捕获层 executable 可以继续用现有 `build_decoder_layer_package.py` 的
重复 `--golden-dir`/`--executable` 参数形成层链。它已有驻留权重、层间 device alias、
host embedding/LM-head、final norm 接口。整模型及这些边界的通过情况必须另有真实
捕获和执行证据，单层通过不能作为完整 30 层执行结论。

## 已执行的连接层验证

2026-10-14，WSL pytest 17 项通过，覆盖真实映射顺序、显式 ABI 裁剪、Command
shape/dtype/bytes、bias/mask/RoPE/布局负例、旧 package 默认 ABI、BF16 bits、完整输出
三方比较和原始失败诊断保存。报告在
`out/hf-connection-20261014/package-bridge-tests/report.json`。

同日重新执行旧 seq128/vector/W8A16/BF16 raw Command CModel baseline，73,728 个 BF16
输出通过，最大误差 `0.03125`，`max_cycle=315681`。原始输出及报告位于
`out/hf-connection-20261014/old-baseline/`。这是旧手写图回归证据；新捕获图的实际
Command/CModel 成功与失败必须引用独立 bridge `report.json`。

真实 pretrained SmolLM2 layer-0 的捕获图已完成整条路径和 CModel 验证：
`out/hf-connection-20261014/smollm2-layer0-final/`。最大/平均 CModel 对 backend golden
误差为 `0.25 / 0.00340047799`，通过原有 `0.35 / 0.035` 阈值。它不是完整 30 层模型
验证，也不是 FP32 逐位验证。量化、框架转换、backend LUT/舍入及执行误差分别见
[连接验证报告](hf_connection_validation.zh-CN.md)。

第二个实际 HF Qwen2 接口沿用相同 wrapper/配置，保留非零 Q/K/V bias，导出和
Kernel/Tensor/Stream 通过；当前 target 的 compact RoPE/bias MEM 资源冲突阻止 Schedule。
未删 bias，未将这次运行标为 Command/CModel 成功。该 fixture 是固定种子的随机权重
本地 Safetensors 模型，不能作为 pretrained Qwen2 成功证据。
