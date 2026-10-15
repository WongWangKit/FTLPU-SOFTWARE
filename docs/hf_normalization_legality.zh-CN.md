# HF StableHLO 规范化和 Kernel 完整性边界

`ftlpu-opt` 默认在 Kernel 转换之前执行轻量布局规范化和上游 MLIR CSE。
布局规则只消除/组合静态 reshape、transpose、broadcast，根据实际 dot dimension
numbers 整理二维收缩轴，并在 batch 维两侧均为 1 时移除和恢复该维。
reshape 穿过 add/multiply/convert 的规则要求每个输入都来自相同二维 shape 的
reshape；不穿过 reduce，不修改广播轴，不省略类型转换。

这些规则不做浮点算术重排、不更换 dot precision 或 algorithm、不修改 dtype、
mask 或参数顺序。源/目标 StableHLO 的官方 portable 序列化可以由 XLA 分别执行，
用于检查实际数值；结构检查本身不等于数值执行成功。

## 完整模式与调试模式

`--kernel-mode complete` 是默认值。Kernel pass 后检查所有残留 StableHLO
计算，并检查 Kernel 辅助算子是否被已有完整 AttentionGraph/FfnGraph 消费。
当前没有独立 Tensor lowering 的 generic batch_matmul、独立 Softmax/RoPE/GQA、
布局或 bias 算子不能仅因进入 Kernel dialect 就被当作完整转换。
检查失败时，Tensor/Stream/Schedule/Command 不会启动。

`--kernel-mode partial` 只允许 `ftlpu-stablehlo-to-kernel` 调试入口。
产物记录 `ftlpu.kernel_complete = false` 和每个残留 op 的名称、数量、所有位置、
拒绝原因；stderr 同时警告。该输出不表示完整 lowering，不能用 partial 模式
启动 Tensor 及之后的 pipeline。把 partial IR 另存后重新输入完整 pipeline，
依然必须重新通过完整性检查。

`ftlpu-compile` 的 StableHLO 入口同样默认执行规范化、CSE 和完整性检查。

## 诊断和参数绑定

`--diagnostic-dir` 必须指向尚未包含 `report.json` 的目录，避免覆盖既有证据。
目录保存输入、绑定后的输入、每个 pass 后的 IR 和失败时的 IR；`report.json`
包含实际配置、LLVM 版本、pass 耗时、失败阶段、原始 MLIR 消息和结构化 Kernel
coverage。MLIR 原始 stderr 保持输出。调用 Python 桥时，桥再保存完整 stdout/stderr。

`--constant-bindings bindings.json` 只接受明确的静态小型 f32/BF16/bool 常量，检查实际函数
符号、原参数位置、shape、dtype、元素数量和有限值。它用于将 bundle 中实际保存的
不可变 inv_freq 等内容提供给结构证明，不能用猜测的 theta 重建并替换真实值。
该选项本身保留函数 ABI。BF16 使用 LLVM APFloat 的对应 dtype 和最近偶数转换；
bool 必须是 JSON boolean，不能通过猜测原始字节解释。

`--fold-static-mask on` 显式启用静态 mask 求值，默认关闭以兼容旧手写图的结构
识别。仅处理 select 数据流中的已知常量：整数/bool 运算及布局、select、类型
转换。逐个实际 op（含真实属性/region）交给上游 StableHLO reference interpreter，
不实现新的 gather/reduce_window 数学，不折叠浮点算术。每个 tensor 的元素上限
为 65536。动态 mask 不能求值，仍须通过结构证明或明确拒绝。最终 mask 值保留
为 dense constant，不能凭配置声明它是 causal。

```json
{
  "schema_version": 1,
  "functions": {
    "main": [
      {"argument_index": 3, "shape": [2], "dtype": "float32", "values": [1.0, 0.01]}
    ]
  }
}
```

上例只是格式示例；实际参数位置和值必须从 bundle manifest 和保存的参数读取。
PyTorch/XLA 文件 `functions/forward.mlir` 中的函数符号通常是 `main`，不得把文件名
当成函数符号。

显式 `--prune-unused-inputs` 删除绑定后未使用的参数，必须同时指定诊断目录。
`abi_map.json` 对每个原参数记录 `original_index`、`final_index`（删除时为 null）、
原类型和是否常量特化。ModelPackage 的 Command input binding 必须通过该映射
回到 bundle 的原逻辑参数，不能依赖权重文件顺序。当前不裁剪被内部 func.call
调用的函数，避免改变调用 ABI。

`--backend-profile profile.json` 记录显式数值 profile 和实际模型的 RoPE theta：

```json
{"schema_version": 1, "profile": "legacy_w8a16_bf16", "rope_theta": 100000.0}
```

它附加 `ftlpu.backend_numeric_profile` 和 `ftlpu.rope_theta` 元数据；结构 matcher
仍须验证计算图和实际常量。profile 元数据不能证明 mask、位置、scaling 或 bias
可以省略。

## 运行和数值检查

```powershell
.\build-codex-macro-vs-verify\compiler\Release\ftlpu_opt.exe `
  --input input.stablehlo.mlir --output normalized.stablehlo.mlir `
  --pipeline ftlpu-normalize-stablehlo `
  --portable-output normalized.bytecode --portable-version 1.13.0 `
  --diagnostic-dir out/normalization-evidence

python compiler/tests/stablehlo_normalization_legality_test.py `
  --tool build-codex-macro-vs-verify/compiler/Release/ftlpu_opt.exe `
  --output-dir out/normalization-regression

.\tools\run_xla.ps1 compiler/tests/stablehlo_normalization_legality_test.py `
  --output-dir out/normalization-regression --verify-numerics
```

测试分别用 upstream StableHLO serialization API 保存未规范化和规范化后的
portable artifact，再使用已配置的 PyTorch/XLA CPU 执行，数值 case 使用固定随机
输入和 `rtol=0, atol=0`。若测试目录非空，结构测试会选择新的时间戳子目录；
数值检查应使用结构测试打印的实际目录。

数值结果保存在 `numerical_report.json`。未执行数值检查时，结构测试报告明确
记录 `numerical_verification: not_run`。该局部规范化验证不代替真实模型的量化
golden、完整 Command lowering 和 CModel 执行验证。
