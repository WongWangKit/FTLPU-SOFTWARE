# HF/XLA rank-3 dot 局部修补

2026-10-14，复用已验证的 SmolLM2-135M float32、batch=1、sequence=16 bundle，
修补真实 XLA 图进入 Kernel pass 时的 rank-3 dot 报错。

## 修改范围

`StableHloToKernel.cpp` 读取官方 StableHLO dot dimension numbers，不根据形状猜测收缩轴。
新增路径只接受静态 rank-3、两侧 batch 轴均为 0、左侧收缩轴为 2、右侧收缩轴为 1 或 2。

- batch=1：移除 singleton batch，调用现有二维 `kernel.matmul`，恢复原输出 shape；
  右侧收缩轴为 2 时保留必要的转置。真实 RoPE 的 `[1,32,1] × [1,1,16]`
  外积使用此路径。
- batch>1：复用现有 `kernel.batch_matmul`，保留 batch 和 RHS 转置标志。
  新增 `generic` 角色，表示尚未识别为完整 Attention 图的批量乘法；
  不冒用专用 `qk`/`pv` 角色，既有 Attention matcher 保持其检查。
- 新 rank-3 路径拒绝非 DEFAULT precision 和显式 dot algorithm，因为当前 Kernel IR
  无法表达这些数值约束。缺省 precision 可用。
- 二维路径补充无 batch、收缩轴 `[1] × [0]` 的校验，避免方阵形状掩盖错误的轴解释。

没有修改 HF 模型数学逻辑、dtype、权重、mask、RoPE 或 bundle 参数映射。
原生 bundle 及其内容校验文件保持可复用；本次产物另存于 `out/hf-lowering/`。

## 实测结果

更新后的 Windows Release compiler 成功构建。
`stablehlo_batched_dot_mlir_test.py` 的 12 个真实 CLI case 全部通过：
3 个 singleton-batch case、2 个 batch=9 case、7 个明确拒绝 case。
覆盖转置、缺省 precision、维度约束及不能表达的数值选项；已注册 CTest。

5 项针对性 CTest 中 4 项通过（新 batched dot、旧二维 Kernel/Tensor、旧 Attention
Kernel），旧 Attention Tensor 测试因 activation scratch 容量失败。
2026-09-22 的旧二进制与本次二进制在默认配置及相同显式 legacy 配置下均复现
相同错误；两者生成的 Kernel 函数体一致。这是既有内存规划限制。
对照证据在
[`out/rank3-review-baseline-20261014/report.json`](../out/rank3-review-baseline-20261014/report.json)。

完整 SmolLM2 图的 Kernel pass 返回 0，所有 272 条 `stablehlo.dot_general` 均已转换：
212 条 `kernel.matmul`、60 条 `kernel.batch_matmul`。
比较函数签名后，所有参数、输入输出 tensor shape/dtype 保持一致。
原始 bundle 的 manifest、文件和内容 seal 校验再次通过。

Kernel 输出仍保留 gather、reduce、mask、sin/cos、logistic 等 StableHLO 操作。
这表示 dot lowering 已通过，不表示完整 Kernel 合法化、Tensor lowering 或硬件执行已通过。
本次没有执行 Kernel/Tensor IR 数值验证，前端保存 StableHLO 的数值验证结果见
[Linux 环境说明](../tools/README.xla-linux.zh-CN.md)。

继续完整模型的 `ftlpu-stablehlo-to-tensor` 后返回 1，报错：

```text
cannot allocate function inputs in the east MEM hemisphere
```

当前 Tensor 入口把函数参数作为同时驻留的模型输入分配；完整 float32 模型的参数和
输入不能满足该分配方式。新 `generic` batch matmul 也尚无通用 Tensor lowering。
本次没有通过改变物理内存配置或删除模型计算来绕过这些限制。

结构化结果及原始诊断在
[`out/hf-lowering/rank3-fix-20261014/report.json`](../out/hf-lowering/rank3-fix-20261014/report.json)，
Kernel 图在同目录的 `smollm2-135m.kernel.mlir`。
独立回归输入、结果和原始错误在 `out/rank3-dot-regression-20261014/`。

## 复现

在仓库根目录使用更新后的 compiler；输出放在 bundle 目录之外：
当前默认 complete 模式会拒绝剩余 StableHLO；下面显式使用 partial 复现当时的
dot 局部转换。它只允许 Kernel 调试输出，不能作为完整 Tensor lowering 输入。

```powershell
.\build-codex-macro-vs-verify\compiler\Release\ftlpu_opt.exe `
  --input out/hf-bundles/smollm2-135m-f32-b1-s16-xla2.9.0-20261014/functions/forward.mlir `
  --output out/smollm2.rank3-fixed.kernel.mlir `
  --pipeline ftlpu-stablehlo-to-kernel --kernel-mode partial

python compiler/tests/stablehlo_batched_dot_mlir_test.py `
  --tool build-codex-macro-vs-verify/compiler/Release/ftlpu_opt.exe `
  --output-dir out/rank3-dot-regression
```

上述 MEM 错误是引入严格完整性检查之前的历史结果。当前直接改为 Tensor
pipeline 会先在完整性检查拒绝未支持的计算，不能带 partial 绕过该检查。
