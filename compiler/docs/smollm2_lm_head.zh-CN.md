# SmolLM2-135M 量化 LM Head

## 测试范围

该测试读取 Hugging Face `SmolLM2-135M` 的真实 checkpoint。模型的 LM head
与 embedding 权重绑定，因此使用真实的 `model.embed_tokens.weight` 作为
LM-head 权重，覆盖完整模型尺寸：

- hidden size：576
- vocabulary size：49152
- 激活批次：32 行
- 权重存储：对称 per-tensor INT8
- 计算和输出格式：BF16

lowering 不是 SmolLM2 专用逻辑。输入使用标准 StableHLO 表达 INT8 权重
转换、标量反量化和 `dot_general`，compiler 将其识别为通用 W8A16
linear projection。

## Lowering 流程

```text
StableHLO dot_general
  -> ftlpu.kernel.matmul（M/N/K 和 rhs_scale）
  -> ftlpu.tensor.projection_task kind="linear"
  -> ftlpu.stream.projection_task
  -> 精确 cycle 的 Schedule IR
  -> Command IR
  -> .ftlpu 二进制
```

Tensor IR 分配两条 BF16 activation slice、跨东西半球的八条 INT8
weight slice 和四条 BF16 result slice。Stream IR 表达 MEM 到 MXM 的
激活路径、MEM 到 VXM 再到 MXM 的权重反量化路径，以及 MXM accumulator
到 MEM 的结果路径。

每个 `32x576 @ 576x4096` executable 包含 18 次 K 方向 partial
accumulation。权重以八条 INT8 stream 读出，由 VXM 反量化并 cast 到
BF16，送入东西半球各一个 MXM，在 MXM accumulator SRAM 中累加，最后
cast 成 BF16 并写回 MEM。

## 词表分片

完整 `576x49152` 权重拆成 12 个连续的 4096 列 shard。所有 shard 共用
同一个 executable 和同一个全局量化 scale。Runtime 对每个 shard
重新装载 ICU program、绑定权重、运行 CModel，并拼接 12 份结果。这样
可以控制 Schedule IR 体积，同时保证 49152 个词表列全部在 LPU 上执行。

## 真实 Checkpoint 结果

测试生成器读取：

```text
.cache/hf/SmolLM2-135M/model.safetensors
tensor: model.embed_tokens.weight
```

完整词表实测结果：

- 12 次 CModel 执行
- 检查 1,572,864 个 BF16 logits
- 总计 998,388 cycles，包含每个 shard 末尾 64 个 drain cycles
- cosine similarity：1.0
- 平均绝对误差：约 `1.0e-7`
- 最大绝对误差：`0.03125`
- 第 31 行 top-1：CModel 和 golden 均为 token 32
- 非零 logits：1,572,864

## 运行方式

构建并运行完整真实权重流程：

```powershell
cmake --build build-ftlpu-vs2026 --config Release --target smollm2_135m_lm_head_pipeline
```

逐层生成物位于：

```text
build-ftlpu-vs2026/compiler/ftlpu_lower/smollm2_135m_lm_head/
  data/
  ir/stablehlo/
  ir/kernel/
  ir/tensor/
  ir/stream/
  ir/schedule/
  ir/command/
  smollm2_135m_lm_head_shard.ftlpu
  cmodel_result.txt
```

日常快速检查可向
`compiler/tests/smollm2_lm_head_binary_runtime_test.py` 传入
`--shard-count 1`，只执行一个真实 shard。

## 完整 Prefill 组合测试

`smollm2_full_prefill_lm_head_e2e_test` 在测试时替换模型包里的 host LM
head，把 30 层 decoder 和最终 RMSNorm 的真实 CModel 输出接到量化 LPU LM
head。最后一个 token 被放入一个补零的 32 行硬件 tile 的第 31 行，随后
12 个词表 shard 全部经过二进制加载、runtime binding、ICU 队列和 CModel
执行。

测试检查三个边界：

- LPU 最终 RMSNorm 与导入的 decoder golden 对比
- LPU LM-head logits 与 BF16 反量化后的 INT8 精确 golden 对比
- 最终 49152 维 logits 和 top-5 与原始 tied BF16 模型权重对比

生成完整 prefill package 和 LM-head 产物后运行：

```powershell
python compiler/tests/smollm2_full_prefill_lm_head_e2e_test.py `
  --runtime-test build-ftlpu-vs2026/runtime/Release/smollm2_full_prefill_lm_head_e2e_test.exe `
  --package build-ftlpu-vs2026/full_prefill_tail/smollm2_135m_seq128_tail_bf16.ftlpum `
  --lm-head-binary build-ftlpu-vs2026/ftlpu_lower/smollm2_135m_lm_head/smollm2_135m_lm_head_shard.ftlpu `
  --weight build-ftlpu-vs2026/ftlpu_lower/smollm2_135m_lm_head/data/lm_head_weight.i8 `
  --metadata build-ftlpu-vs2026/ftlpu_lower/smollm2_135m_lm_head/data/metadata.json `
  --output build-ftlpu-vs2026/full_prefill_tail/lpu_lm_head_e2e_result.txt
```
