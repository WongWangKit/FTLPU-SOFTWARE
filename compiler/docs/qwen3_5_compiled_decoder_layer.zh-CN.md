# Qwen3.5 dense DeltaNet decoder 层验证

`qwen3_5_compiled_layer.py` 已连接完整的 `linear_attention` decoder 层：

1. 零中心 RMSNorm，QKV/Z/B/A projection。
2. kernel-4 causal DepthwiseConv1D + SiLU，Q/K 拆分、head 重复及 L2 归一化。
3. beta sigmoid、FP32 decay、gated delta-rule 和 FP32 recurrent state。
4. 每 head RMSNorm × SiLU(z)、out projection、第一次 residual add。
5. 第二次零中心 RMSNorm、Gate/Up projection、SwiGLU、Down projection、第二次 residual add。

所有运行时算术均由 Kernel IR lowering 生成的 ICU/FU 指令执行。CModel 中没有
Qwen/DeltaNet decoder 层专用执行函数。Python 只做 reshape、slice、head 重复、
broadcast、padding 和阶段输入输出传输；NumPy 数学仅在独立测试参考中使用。

当前这是**串行分阶段 correctness bring-up**，尚不是单个驻留在 MEM 的整层
program。向量 SSA 子图内部的中间值留在 MEM，不再逐条运算下载/上传：
RMSNorm/L2Norm、beta/decay 控制链、每 head norm × SiLU(z)、SwiGLU
各自编译为一个程序。编译器根据 SSA 最后使用位置复用 slice/bank，广播直接
由 MEM READ3D 的地址步进完成；同一输入相乘只读一次，ALU 两个操作数复用 stream。
阶段之间仍经过主机下载/上传；因此阶段 cycle 相加不代表真实 TTFT，
也不能用它评估 C2C overlap。大矩阵会在 lowering 前按 SRAM/iMEM 约束切块。
超过 4096 的 projection K 轴分块时，各 MXM 部分和先转 BF16，再由 VXM
以 FP32 相加；它存在额外的 BF16 舍入边界。真实大模型 checkpoint 尚需单独验证。

## 测试

```powershell
python compiler/tests/qwen3_5_compiled_decoder_layer_test.py `
  --compile build-ftlpu-vs2026/compiler/ftlpu-compile.exe `
  --runner build-ftlpu-vs2026/runtime/compiled_stage_file_runner.exe `
  --output-dir results/qwen3_5_decoder_layer_prefill
```

该测试使用非零合成权重和 seq32、hidden64、FFN128、1 个 key head、2 个
value head、head_dim32，覆盖完整层及 Conv/recurrent 两种状态。
另外用 [-12,12] 的输入验证 softplus 的两个分支。
目录中保存每阶段 Kernel IR、真实 `.ftlpu` 指令文件、输入输出、`stages.json`
和包含中间误差的 `summary.json`。CTest 名称为
`qwen3_5_compiled_decoder_layer_test`。

这组 mini seq32 测试的 decoder 独立执行阶段从 84 个减少为 19 个；最终输出
最大绝对误差 0.003445，FP32 recurrent state 最大绝对误差 0.0001654，
与逐运算版本相同。当前子图驻留路径要求单个输入不超过 4096 个元素、最多
四个外部输入；更大形状仍使用分块路径。该结果使用合成权重，不代表真实
checkpoint 或端到端性能验证。`vector_alu_binary_runtime_test` 另外覆盖
分支保活、scratch 复用、二维广播以及全部算子共享 `loc(unknown)` 的调度。

## 导入权重

先用 `import_hf_qwen3_5_layer.py` 导入 checkpoint，再运行：

```powershell
python compiler/tools/qwen3_5_compiled_layer.py `
  --package <import-directory> `
  --compile build-ftlpu-vs2026/compiler/ftlpu-compile.exe `
  --runner build-ftlpu-vs2026/runtime/compiled_stage_file_runner.exe `
  --output-dir <result-directory>
```

CLI 仅报告执行完成，未自动把真实 checkpoint 的误差判定为通过。
目前仅支持空 cache 开始的 prefill，tokens/hidden/intermediate 必须按 32 对齐。
`full_attention` 层不属于此 DeltaNet 路径。

## 硬件映射中的修复

- `vector_alu` 是通用映射/行归约，不包含模型字段。行归约使用 accumulator
  首次清零、最后发出；EXP 位于 ALU1，RECIP/RSQRT 位于 ALU3。
- softplus 使用 `max(x,0) + 2 atanh(t)`，其中
  `t=exp(-abs(x))/(2+exp(-abs(x)))`，六项奇次多项式的截断误差小于 1.1e-7。
  实际误差还包含 LUT 插值误差。
- 独立 W8A16 projection 连续权重读取、MXM load、ACC read 和结果写回直接
  生成循环域；全部 K 块保持 FP32 累加，最后 ACC→MEM 转 BF16。
- projection 结果使用 `fp16_hemisphere_pair_planar`：每 32 列交替存 East/West，
  各使用同一 byte-plane pair。结果 bank1 避免覆盖 bank0 后续权重。


## 整层单程序驻留执行

`qwen3_5_resident_decoder_layer_test` 将完整 linear-attention decoder 层生成
一个 Schedule IR 和一个 `.ftlpu` 文件。19 个算子子图只作为编译期 lowering
输入；运行时只 load 一次、run 一次。中间激活和状态保留在 SRAM，host
只上传初始激活、权重/常量，最后下载输出、recurrent state 和 Conv state。
中间 host 传输为 0，CModel 内没有 Qwen 算子实现。

```powershell
python compiler/tests/qwen3_5_resident_decoder_layer_test.py `
  --opt build-ftlpu-vs2026/compiler/ftlpu_opt.exe `
  --compile build-ftlpu-vs2026/compiler/ftlpu-compile.exe `
  --runner build-ftlpu-vs2026/runtime/compiled_stage_file_runner.exe `
  --output-dir results/qwen3_5_decoder_layer_resident
```

已通过 seq32、hidden64、FFN128、1 key head / 2 value heads、head_dim32
的非零合成 W8A16 权重测试。单文件大小 676,444 bytes；输出最大绝对误差
0.00344491，FP32 recurrent state 0.000165386，Conv state 0.00368208。
共执行 4,899,791 cycles；布局转换按保守串行调度，这个周期数不能当作
真实 checkpoint 的 TTFT。真实尺寸的分页、C2C overlap 和 full-attention 层
还未在此单程序路径验证。

通用驻留链接器直接用张量形状/索引视图生成 MEM 3D、SXM RUN_2D 和
VXM RUN_2D 域，不先展开 FU 轨迹。静态参数在 bank1 下半区，中间张量
在 bank1 上半区按生命周期分配；算子本身保留原生 scratch 布局。同侧
packed→scalar 转换由 SXM 直接写 MEM，跨侧搬运使用 VXM。FP32 同侧
转换逐 byte-plane 保留原始比特；独立布局回归测试逐比特检查此转换。

SXM 使用已有 4-bit selector 的保留编码：Transpose input-row 9 表示把
一个输入 byte pair 广播到 8 行；Permute output-row 9..15 表示选择行
0..6 并释放整个输入块，行 7 原本就释放。旧 selector 0..8 保持原意，
Transpose/Permute 仍分别使用独立队列，包仍为 2/4 个 96-bit 字。
Block Permute 允许整 superlane fanout，4 个输出 tile 逐周期发出；没有
增加模型语义指令或 ALU delay op。


验证补充：布局转换四组逐比特检查、vector/reduction/state 回归、SXM codec、
启用断言的 SXM slice 单元测试和 attention transpose 测试通过。原有
`sxm_mem_transpose_test` 在 cycle23 出现 buffer-full；将 SXM slice 替换为
Git HEAD 版本后也复现同一失败，未将其计入通过项。结果目录的
`validation.json` 保存此记录。


## Runtime pipeline viewer

整层文件 runner 的第四个参数可选，用于在 `load` 前启用真实 CModel
execution trace，并在运行后导出 pipeline viewer 格式的 CSV：

```powershell
build-ftlpu-vs2026/runtime/compiled_stage_file_runner.exe `
  results/qwen3_5_decoder_layer_resident/decoder_layer.ftlpu `
  results/qwen3_5_decoder_layer_resident/inputs.txt `
  results/qwen3_5_decoder_layer_resident/resident `
  results/qwen3_5_decoder_layer_resident/runtime.pipeline.csv
```

完整测试也接受 `--pipeline-csv <path>`。该 CSV 来自逐 physical-cycle
采样的 ICU 实际发射，包含 MEM、SXM、MXM 和 VXM，不能用静态 Schedule
CSV 替代。这个 resident 测试的初始输入直接上传到 SRAM，不经过 DDR/C2C，
因此 trace 不包含虚构的冷启动传输段。

本机 HTTP viewer 支持通过 `?csv=/results/.../runtime.pipeline.csv` 自动加载；
也可直接打开 `tools/pipeline_viewer/index.html`，点击“打开 CSV”选择该文件。

## 官方 0.8B checkpoint 的真实尺寸验证

使用 `Qwen/Qwen3.5-0.8B` 第 0 个 linear-attention 层，hidden=1024、
intermediate=3584、16 个 K/V head、head-dim=128、Conv1D kernel=4，seq32。
下载器按 HTTP Range 读取这一层全部原始参数和 token-id 0..31 的真实
embedding 行；不下载其余层、视觉参数或未使用的 embedding。输入是
确定性的 token-id 测试序列，不是自然语言 prompt。`provenance.json`
记录不可变 revision、原始形状、下载范围、各张量 SHA256 和子集 SHA256；
导入前检查子集 SHA256。

```powershell
D:/Apps/Anaconda3/python.exe compiler/tools/download_hf_layer.py `
  --repo Qwen/Qwen3.5-0.8B `
  --revision 2fc06364715b967f1860aea9cf38778875588b17 `
  --layer-prefix model.language_model.layers.0 `
  --embedding model.language_model.embed_tokens.weight `
  --embedding-rows 32 --output-dir tmp/models/Qwen3.5-0.8B-layer0
D:/Apps/Anaconda3/python.exe compiler/tools/import_hf_qwen3_5_layer.py `
  --model-dir tmp/models/Qwen3.5-0.8B-layer0 `
  --output-dir results/qwen3_5_0_8b_decoder_layer_prefill/package --seq-len 32
D:/Apps/Anaconda3/python.exe compiler/tests/qwen3_5_real_decoder_layer_test.py `
  --package results/qwen3_5_0_8b_decoder_layer_prefill/package `
  --compile build-ftlpu-vs2026/compiler/ftlpu-compile.exe `
  --runner build-ftlpu-vs2026/runtime/compiled_stage_file_runner.exe `
  --output-dir results/qwen3_5_0_8b_decoder_layer_prefill/compiled_cached
```

线性权重按现有硬件 W8A16 per-tensor 格式量化，总计 20.53125 MiB；
不是随机权重，也不是原生 BF16 权重执行。FP32 recurrent state 为 1 MiB。
矩阵文件采用 `[in,out]`，验证接口按转置恢复 `[out,in]`，并用矩形矩阵
回归检查方向。Conv1D 在通道维按 1024 分块；DeltaNet 在 head/time 维
分块，每个 head 内严格串行传递 FP32 state。独立 head 可以在不同
CModel 实例中并行验证以减少测试墙钟时间，不能据此声称硬件并行速度。

这个真实尺寸测试目前使用逐算子/分块 ICU 程序，存在阶段间主机布局
传输；主机不计算中间算术结果。重复程序按源码和编译器身份缓存，已
执行阶段按程序、runner、所有输入和输出哈希校验后续跑。最终
`real_validation.json` 必须明确 `single_resident_program=false`、
`ddr_c2c_paging=false`，不能作为整层连续 runtime pipeline 或 TTFT 结果。
通过标准在运行前固定：output、recurrent-state、conv-state 的归一化
RMS 误差 ≤3%、cosine ≥0.999。参考是同样量化权重的 FP32 NumPy 方程，
不是未经量化的官方 BF16 模型。

`quantization_reference.json` 另存原始 BF16 线性权重与 W8 per-tensor
权重的 FP32 方程对照。当前 token-id 0..31 测试序列的输出量化 NRMSE
约 14.08%、cosine 0.990315，state NRMSE 约 4.79%；这是量化误差，
不是 CModel 与同量化 reference 的执行误差。不能将硬件数值通过表述为
原始 BF16 模型等精度通过；后续精度优化需评估更细的量化 scale 粒度。

整层 resident 测试新增 `--package` 可以直接尝试真实 checkpoint。
当前真实尺寸输入 RMSNorm 会明确报错：`vector scalar-broadcast storage
exceeds SRAM; tile the vector before lowering`。mini 的单程序成功不能
用于宣称真实尺寸单程序已通过；还需补分块链接、packed 中间存储和
权重分页。失败尝试不得生成 passed summary。

真实尺寸分块运行已通过：633 个程序执行，output NRMSE 0.8451%、
cosine 0.9999665；recurrent state NRMSE 0.5415%、cosine 0.9999881；
conv state NRMSE 0.3413%、cosine 0.9999972。结果在
`results/qwen3_5_0_8b_decoder_layer_prefill/summary.json`，阶段检查在
`compiled_cached/real_validation.json`。这里的通过只覆盖分块数值验证，
不覆盖真实尺寸连续 resident 程序或 DDR/C2C 分页。
