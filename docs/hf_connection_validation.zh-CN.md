# HF/XLA 捕获图到 Command/CModel 的连接验证（2026-10-14）

真实 pretrained SmolLM2-135M decoder layer-0 已完成完整 Kernel、Tensor、Stream、
压缩 Schedule、独立 Schedule verification、Command、二进制、ModelPackage 和
CModel 执行。首轮使用旧成功路径的 `S=128,H=576,I=1536,Q=9,KV=3,D=64`、
W8A16/BF16 profile、vector MXM、tail FFN、ICU macro 和
`../FTLPU-CMODEL/config/ftlpu-lpu32.json`。全部 73,728 个输出参与比较。

这份结论限于真实单个 decoder 子图及其现有 ModelSession 执行。尚未证明真实
30 层模型连同 host embedding、final norm、LM-head 的完整执行。原有 FP32 forward
bundle 保持独立，完整性重新校验通过，没有覆写其图、权重或参考输出。

## 实际改动

- `compiler/tools/ftlpu_frontend/profiles.py`：薄 decoder 边界调用真实 HF layer、
  rotary 和 mask helper。仅 selected Linear 的权重表示及明确的计算 dtype 使用旧
  quantizer；保留 bias、残差、mask、位置和数学实现。入口不启用 cache/state。
- `NormalizeStableHlo.cpp`：布局规则及 MLIR CSE；静态 mask 的实际整数/bool/布局
  op 交给上游 StableHLO reference interpreter。没有浮点重关联。保留函数 ABI；
  常量特化与可选裁剪另生成逐参数 `abi_map.json`。
- `StableHloToKernel.cpp`：从实际 shape、属性、类型和 operand 身份证明 HF RMSNorm、
  半分 RoPE、稳定 Softmax、Q/K/V/O、GQA、SiLU/SwiGLU/FFN 和残差连接。新数值规则
  仅在明确的旧 profile 中启用，不按模型名称、层号或一份文本匹配。不引入整块
  不透明 Attention/FFN 操作，继续交给 AttentionGraph/FfnGraph 和旧规划器。
- `VerifyKernelCoverage.cpp`：完整模式拒绝所有没有已知下游消费者的计算，包括
  generic batch_matmul 及未组成完整结构的 Kernel 辅助操作。partial 仅用于 Kernel
  调试，不能进入 Tensor。名称、数量、位置、拒绝原因保存在 IR、stderr 和 JSON。
- `xla_compat.py`/`tensor_io.py`：补齐 torch-xla 2.9 的 BF16 NumPy 互操作缺口，仍由
  XLA 转换并原生 save/load。BF16 bits 只在捕获签名、原生 metadata 和 IR 类型明确
  声明时恢复。FP32 默认调用不变。
- `package_bridge.py`/`tools/lower_hf_bundle.py`：唯一模型 IR 输入是保存的捕获图；
  使用原生逻辑映射、编译器 ABI map 和实际 Command binding 三方核对，然后复用
  原 package writer、权重驻留和 ModelSession/CModel。没有替换为手写模型图。
  Package writer 只增加可选角色到实际 binding index 的字段，缺省旧 ABI 不变。
- `ScheduleToCommand.cpp`：实际执行暴露计算依赖被误当成 MEM 存储别名。重定位
  现在只沿 Route/明确 input binding 追踪；隐式 binding 的角色按读操作确定，
  不再假设 arg0 是 activation。未改 runtime 架构。
- `KernelToTensor.cpp`/`ftlpu_opt.cpp`：MEM 失败补充参数、type、字节数、类别、
  slices/banks/capacity；各阶段保存耗时、IR 和原始诊断。JSON 诊断拥有符号字符串，
  防止 MLIRContext 销毁后的悬空引用；partial 警告明确打印。

为使用当前 checkout 的上游 reference interpreter，StableHLO 子模块有一处
DenseElements/MLIR complex API 的构造类型兼容修改；数学算法未改变。可重放补丁在
`compiler/patches/stablehlo-reference-mlir-complex.patch`。未覆盖此前 runtime、TinyLlama、
Command binary 和其他未提交工作。

## 数值阶段和实际结果

checkpoint 存储 dtype 来自官方 Safetensors metadata，HF 此次加载为 FP32。
七个 decoder Linear 复用旧 `quantize_linear`：每 tensor 对称 scale，`max(abs)/127`，
`numpy.rint` 最近偶数、范围 `[-127,127]`，INT8 `[in,out]` 存储。scale 复用旧 BF16 RNE。
有效权重为 INT8×舍入后的 scale，再舍入 BF16。Attention projection 使用 BF16；
FFN gate/up/down 的 FP32 dot 和 BF16 中间边界分别显式记录。norm/bias/input 使用 BF16。
embedding/LM-head 未被机械量化，原模型 tied weight 身份没有被改变。

原 HF FP32 子图、转换后的真实 HF、保存的 StableHLO、旧 backend golden 和 CModel
均保留。子图比较使用同一实际 BF16 hidden 输入，FP32 子图把这个输入转换为 FP32；
因此这里不包含 host embedding 从 FP32 到 BF16 的额外误差。

| 比较 | 最大绝对误差 | 平均绝对误差 | 结论 |
|---|---:|---:|---|
| 转换 HF → torch.export | 0 | 0 | 严格 capture 比较通过 |
| 保存 StableHLO → 转换 HF | 0.5 | 0.00439962 | 旧 BF16 checkpoint 标准通过 |
| 原生保存图 → compiler 重序列化图 | 0 | 0 | XLA 零容差通过 |
| 重序列化图 → 布局规范化/实际常量特化图 | 0 | 0 | XLA 零容差通过 |
| 转换 HF → 同输入 FP32 HF | 1.1914463 | 0.05522831 | 显式量化/dtype 阶段误差 |
| backend golden → 转换 HF | 0.5 | 0.00634923 | 旧 LUT/中间舍入契约差异 |
| CModel → backend golden | 0.25 | 0.00340048 | 原有 CModel 阈值通过 |
| CModel → 转换 HF | 0.5 | 0.00643289 | 全输出测量，不称逐位等价 |
| CModel → 同输入 FP32 HF | 1.5368576 | 0.05524171 | 包含转换及 backend 差异 |

FP32 默认 `rtol=1e-4,atol=1e-5` 没有放宽。BF16 profile 明确选择现有
`hf_decoder_stack_checkpoint_test.cpp` 的标准：每元素界限 `max(0.25,4 BF16 ULP)`，
平均误差 ≤0.04。StableHLO 对转换 HF 的严格 FP32 容差仍有 44,155 个元素超限，
记录保留，不能称为 FP32 严格验证通过。CModel 对 backend golden 使用原有
最大 ≤0.35、平均 ≤0.035，代码与阈值未改。

上游 torch.export 的可选 `.pt2` archive 因 HF mask helper 中非序列化 functorch
初始化节点失败，原 traceback 在 `capture_archive_error.txt`。原生 XLA 图、参数、
metadata 与重载执行成功，保存复用不依赖这个可选 archive。

## 产物和协议

FP32 参考：`out/hf-bundles/smollm2-135m-f32-b1-s16-xla2.9.0-20261014/`。
新独立转换 bundle：`out/hf-bundles/smollm2-layer0-w8a16-s128-connection-v5-20261014/`。
最终完整执行：`out/hf-connection-20261014/smollm2-layer0-final/`。

原生 bundle 仍包含 `functions/forward.{mlir,bytecode,meta}`、`data/` 和 `constants/`。
v1 manifest 索引图、原生参数/常量、真实函数签名、捕获来源、实际数值转换及
工具版本；补充 `backend_profile.json`、实际 `validation/input_0.npy` 和多方数值证据。
`bundle_checksums.json` 封存所有这些产物。未创建新的权重格式或缓存系统。

实际函数有 13 个参数，七个 INT8 权重、两份 BF16 gamma、BF16 hidden、真实 FP32
inv_freq、捕获 lifted BF16 零常量、原生 bool 常量。后三项按 exporter/capture 的来源
绑定为真实常量，裁剪后 10 个逻辑输入。参数顺序为 exporter 的实际顺序，不同于
旧手写 ABI：例如 down 是最终 arg0，hidden 是 arg4，query 是 arg7。
`command_binding_map.json` 对每个角色核对位置、shape、dtype 和 bytes，所有实际
input/output 完整覆盖。物理布局、MEM 行和最终 runtime 绑定由后端确定。

执行目录含各阶段 IR/诊断、`decoder.ftlpu`、`model.ftlpum`、`package_inputs/`、完整
`cmodel_output.bf16.bin`、`numerical_comparisons.json` 和结构化 `report.json`。
保存 bundle 已在独立进程复用，未调用 HF loader、torch.export 或 XLA conversion。

## 配置、运行与调试

复用已经配置的 FTLPU-XLA WSL 环境（torch/XLA 2.9.0、Transformers 4.57.1、CPU）：

```powershell
.\tools\run_xla.ps1 tools/export_hf_stablehlo.py `
  --config tools/configs/smollm2_decoder_w8a16.verified.json `
  --output out/hf-bundles/my-layer0-w8a16 --verify-numerics

$savedPythonPath = $env:PYTHONPATH
try {
  $env:PYTHONPATH = Join-Path (Get-Location) '.cache/frontend-python'
  python tools/lower_hf_bundle.py `
    --bundle out/hf-bundles/smollm2-layer0-w8a16-s128-connection-v5-20261014 `
    --opt build-codex-macro-vs-verify/compiler/Release/ftlpu_opt.exe `
    --translate build-codex-macro-vs-verify/compiler/Release/ftlpu-translate.exe `
    --runtime-test build-codex-macro-vs-verify/runtime/Release/hf_decoder_layer_model_session_test.exe `
    --target-config ../FTLPU-CMODEL/config/ftlpu-lpu32.json `
    --output-dir out/hf-connection/my-layer0
} finally { $env:PYTHONPATH = $savedPythonPath }
```

`entry_options.layer` 选择真实层，`model/revision/load_dtype/load_options` 选择来源和
加载行为。shape 必须与所选 profile/target 相符；不能通过只改输入规格改变数学。
同接口模型使用相同 `ftlpu_frontend.profiles:decoder_entry`。输出目录必须为空。
普通 FP32 入口配置和 `tools/export_smollm2_stablehlo.py` 默认行为保持。

只调试局部转换时使用 `--kernel-mode partial --pipeline ftlpu-stablehlo-to-kernel`；
完整连接入口强制 complete。规范化测试和 portable 数值复现见
[规范化说明](hf_normalization_legality.zh-CN.md)。真实捕获负例入口为
`compiler/tests/hf_decoder_capture_negative_test.py`，接受 bundle/tool/target/output-dir。

## 验证、已知基线与限制

- 前端/bundle/Safetensors/profile/映射回归 59 项通过；桥最终回归 18 项通过。
- 规范化 6 个结构用例及 6 个 XLA 零容差数值用例通过，8 个负例和 2 个常量/ABI
  用例通过。rank-2 非默认 precision/algorithm 也明确拒绝，防止 singleton 规范化
  后丢失精度契约。真实 decoder 的 inv_freq、theta、position offset、padding mask 和
 非默认 dot precision 五个语义负例均在 Kernel 阶段拒绝。超大输入的 MEM 负例
 验证参数/type/bytes/category/capacity 诊断。
- HF 结构 10 项及 rank-3 12 项通过；最终选定 CTest 六项通过。首次 rank-3 test 把
 诊断属性中的 op 名计作第二条 BMM，已修正为只统计实际 op 行。
- 旧 Attention Tensor scratch fixture 仍失败；旧/新工具均已对照复现同一错误。
 两个旧 FFN IR test 的过期 placement 字符串断言也在 2026-09-22 旧工具上同样失败。
 没改这些旧断言来粉饰全绿，原始日志与对照产物保留。
- 旧成功 seq128/vector raw Command 经当前 Command pass/translator 重新生成并由
 更新到当前 CModel ABI 的旧 decoder test 验证，73,728 个输出最大误差 0.03125。
 初次使用未重链接的旧 test 因 macro-context 160/256 硬件 ABI 不同而失败；原错误
 和旧 test 二进制已保留。它不是数值失败。
- 旧手写 RoPE 表达与旧 backend 的旋转约定不能直接当作新图的等价证明。
 新入口证明实际 HF 半分旋转及实际 position/inv_freq；旧 regression 只证明原有
 数值契约兼容，新的数值结果独立对照真实 HF。
- 第二接口测试使用实际 HF Qwen2、小型单层、固定种子随机权重、官方 Safetensors
 本地 checkpoint 和非零 Q/K/V bias，没有新模型 wrapper。导出/capture/保存重载、
 bundle 验证、完整 Kernel/Tensor/Stream 通过。Schedule 明确失败于
 `compact RoPE product placement conflicts with a MEM ICU/read-write resource`。
 该模式未到 Command/CModel；原失败 IR 在 `qwen2-schedule-failure-diagnostics/`。
- padding、非零位置 offset、RoPE scaling/不同 inv_freq、cache/state、非 SiLU、
 FFN/O projection bias、未知精度算法和任意全图量化目前不支持。保持语义并拒绝，
 不删 bias/mask/状态。多个 captured layer 可使用现有 package writer 的层链接口，
 但真实 30 层驻留/paging、层间输入 golden、host 边界仍需独立接入验证。

下一处明确阻塞是上述保留 QKV bias 的 compact RoPE MEM 调度冲突；进一步的全模型
工作还须验证真实层链及权重分页。它们没有被混合 IR 或单层成功替代。

最后补充 rank-2 非默认精度拒绝后，六项选定 CTest 再次通过。真实 SmolLM2 的
Kernel IR SHA256 与已完成 CModel 的版本完全相同，证据在
`out/hf-connection-20261014/post-precision-guard-validation.json`；此修改只增加不支持
数值契约的拒绝，没有改变已经验证的默认计算。
