# Qwen3.5 dense DeltaNet decoder 层验证

## 当前整层入口：C++ 图级 lowering

整层验收使用 `qwen3_5_resident_decoder_layer_test.py` 的真实 checkpoint
路径。`qwen3_5_logical_graph.py` 只生成 19 个逻辑 Kernel 子图、张量
reshape/slice/repeat 视图和参数文件；不生成 Schedule，不决定 SRAM
slice/bank、分页、Conv 分块、ALU 分配或周期。

`ftlpu-graph-compile` 调用编译器库中的
`LogicalKernelGraphToSchedule.cpp`，负责图级规划、生命周期分配、布局
衔接和硬件分块，再复用原生 Kernel→Tensor→Stream→Schedule lowering。
projection 的 N 分页、Conv 的双半球分块、归约的行分块和 Delta rule
的逐 token 状态传递都在 C++ 中完成。生成的都是闭式 MEM 3D、SXM/
VXM 2D 和 MXM 3D 域，不先展开 FU 指令再压缩。

整层当前采用与 dense attention 相同的图级存储规划原则：先将原生
算子降到 Tensor task，收集整张图的 producer/consumer、生命周期和
固定工作区，再统一分配保留输出；之后才执行 Stream/Schedule
lowering。projection 的 `memory_plan.result`、Conv/RMSNorm 的
`result_allocations`、VXM 的结果布局都在生成 ICU 指令前确定，原生
emitter 直接写到该位置。没有逐算子 `archive()` 或输出归档阶段。
这不是生成指令后重定位 MEM 写地址，也不经过 Python 硬件 linker。

完全兼容的输入直接引用 producer 的保留区。布局不兼容的边仍然需要
消费者侧转换，Conv 双半球拼接和 scalar/packed 转换也保留；因此
“零输出归档”不等于“所有 MEM 搬运为零”。分配还检查同一算子读写
是否占用同一 MEM ICU，地址不同不能消除单 context 队列冲突。

逐 token Delta rule 的 FP32 state 由图级规划保留一份，后续 token
继续原地更新。没有外部快照/分支消费者时，final state 与 carried
state 指向同一区域，原生 recurrent emitter 不再生成自复制。
有快照或分支时分配独立状态区，不能用原地更新覆盖仍存活的值。

```powershell
cmake --build build-ftlpu-vs2026 --config Release --target ftlpu_graph_compile ftlpu_compile compiled_stage_file_runner
D:/Apps/Anaconda3/python.exe compiler/tests/qwen3_5_resident_decoder_layer_test.py `
  --package results/qwen3_5_0_8b_decoder_layer_prefill/package `
  --opt build-ftlpu-vs2026/compiler/ftlpu_opt.exe `
  --compile build-ftlpu-vs2026/compiler/ftlpu-compile.exe `
  --runner build-ftlpu-vs2026/runtime/compiled_stage_file_runner.exe `
  --target-config ../FTLPU-CMODEL/config/ftlpu-lpu32.json `
  --output-dir results/qwen3_5_0_8b_decoder_layer_prefill/continuous `
  --pipeline-csv results/qwen3_5_0_8b_decoder_layer_prefill/continuous/runtime.pipeline.csv
```

输入为官方 Qwen3.5-0.8B 第 0 层，seq32、hidden1024、FFN3584。
整层包括输入 RMSNorm、QKV/Conv/SiLU、Q/K L2、Z/B/A 和 beta/decay、
FP32 recurrent state 更新、head norm/gate、out projection，以及两次
residual、post-attention RMSNorm 和完整 Gate/Up/SwiGLU/Down FFN。
只有一个连续计算程序，中间激活和状态不经过主机。

当前图级调度保守串行执行算子，在每个 projection 前预留独占的 C2C
ingress 窗口。compiler 的 `MEM_WRITE_SYNC` 和 page `SYNC` 承担真实
DDR 延迟等待；后续算子可以复用 SR，不会把其未来使用错误地作为先前
权重页的启动下限。窗口以 `c2c.window.binding:<index>.page:<page>`
timeline 元数据传给页规划器，不改变硬件指令字段。该路径目前不宣称
跨算子预取重叠或最优 TTFT。

下文的串行 host-stage 及 Python resident linker 数据保留为历史测试记录。
它们不再作为当前整层 lowering 入口。新的通过结论以 `continuous/summary.json`
和同次执行的 `resident.runtime.ftlpu`、`runtime.pipeline.csv` 为准。

图级输出直写改动前，2026-10-14 的真实 seq32 整层基线通过同 W8A16
权重的 FP32 参考检查：
output NRMSE 0.8603%、recurrent state 0.5577%、Conv state 0.3735%，
cosine 均大于 0.999。连续计算程序执行一次，中间 host 传输为 0；14 个
epoch、13 次动态权重分页，21,626,880 bytes 经 DDR/C2C 装载。
实际 runtime 文件为 1,214,842 bytes，主程序物理 iMEM payload 为
748,764 bytes，所有队列和单 context 检查通过。

`runtime_icu_programs/index.csv` 索引 240 个真实 ICU 文件，包含初始
输入装载阶段和主计算程序的 `MEM_WRITE_SYNC`、普通 `SYNC`、page
`SYNC`。CSV 的周期提示不能替代运行时等待；实际 FU 发射时间看
`runtime.pipeline.csv`。`operator_cycles.csv` 通过物理队列和 PC 匹配
同次真实轨迹，给出各原生算子的首尾发射区间；独立布局搬运不计入
算子区间。C++ 输出的 `graph_plan.json` 保存节点来源和编译周期域。

同日的图级输出直写版本已完成真实权重整层验证，输出、recurrent
state、Conv state 三个文件与上述基线逐字节一致。三项 NRMSE 仍为
0.8603%、0.5577%、0.3735%。静态执行周期（含 drain）从
25,717,506 降为 21,662,582，减少 15.77%；包含初始输入、真实等待和
最终输出的 runtime trace 终点为 21,832,145。它仍为 14 个 epoch、
13 次权重分页和零中间 host 操作，算子间保持串行。

该版本 runtime 文件为 1,232,879 bytes，主程序物理 iMEM 为
761,352 bytes；指令体积并未随周期同步降低。VXM 最大使用
1862/2048 slots，SXM permute 最大 1950/2048 slots；所有队列、
单 context 和实际数值检查均通过。`graph_plan.json` 的
`planning = tensor_subgraph_before_schedule`、
`native_output_archive_copies = 0` 及各节点 `direct_output_layouts`
记录实际采用的编译方案。

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

整层 resident 测试的 `--package` 路径使用真实 checkpoint，RMSNorm 按
SRAM 容量分块、逐元素图使用 packed 布局，中间 tensor view 在设备上
由 MEM/VXM/SXM 搬运。分块发生在 lowering 前，根据算子形状直接产生
3D/2D 域。失败尝试不得生成 passed summary。

真实尺寸分块运行已通过：633 个程序执行，output NRMSE 0.8451%、
cosine 0.9999665；recurrent state NRMSE 0.5415%、cosine 0.9999881；
conv state NRMSE 0.3413%、cosine 0.9999972。结果在
`results/qwen3_5_0_8b_decoder_layer_prefill/serial_staged_summary.json`，阶段检查在
`compiled_cached/real_validation.json`。这里的通过只覆盖分块数值验证，
不覆盖真实尺寸连续 resident 程序或 DDR/C2C 分页。

## 真实尺寸连续程序和 DDR/C2C 分页

```powershell
D:/Apps/Anaconda3/python.exe compiler/tests/qwen3_5_resident_decoder_layer_test.py `
  --package results/qwen3_5_0_8b_decoder_layer_prefill/package `
  --opt build-ftlpu-vs2026/compiler/ftlpu_opt.exe `
  --compile build-ftlpu-vs2026/compiler/ftlpu-compile.exe `
  --runner build-ftlpu-vs2026/runtime/compiled_stage_file_runner.exe `
  --output-dir results/qwen3_5_0_8b_decoder_layer_prefill/continuous `
  --pipeline-csv results/qwen3_5_0_8b_decoder_layer_prefill/continuous/runtime.pipeline.csv
```

123 个编译节点链接为一个计算程序。QKV/Z/B/A、out、gate/up/down 的
8 个线性矩阵按 N 轴产生 13 个页；每页至多 4096 个 SRAM row，分别
读取 DDR、经 C2C RX 和 MEM_WRITE_SYNC 写入原生权重布局。当前各页
保守复用 bank0，激活档案使用 bank1；bank0 也有原生算子 scratch，
页装载必须等对应 MEM 队列的完整域退出后才能复用。连续 projection
页即使复用激活，也保留足够容纳一条同步写的窗口；DDR 超时由事件
SYNC 等待吸收。空 recurrent state 在第一个 delta-rule 前通过 ICU
初始化，避免被早期 projection 权重页覆盖。布局搬运留出 W24–W31
给 C2C。没有增加 SRAM、iMEM 深度或 ICU context 数。

compiler binary 包含 MEM_WRITE_SYNC 和页边界 SYNC_PAGE。DDR 地址及
对应 C2C DMA/RX 程序在 ModelSession 中链接，页 ready 仅由实际写入
完成触发。一个 ICU 队列先完成整条域，再执行下一条；没有跨域交错。
初始常量、激活和最终输出/state 也通过 C2C 传输，中间主机传输为 0。
Conv state 的输出采用 `[4,6144]` token/channel 形式，recurrent state
保留 FP32，并按 `[2048,128]` 输出。

编译产物 `decoder_layer.ftlpu` 与实际装载的 `resident.runtime.ftlpu`
分别保留。`runtime.pipeline.csv` 来自实际运行采样，包括初始传入、
MEM、MXM、VXM、SXM 和 C2C；布局转换仍为保守串行调度，不能当作
优化后的 TTFT。只有 `continuous/summary.json` 的 `status=passed`
才表示此路径通过固定的数值判据，旧分块测试的通过不能替代它。

导出实际 ICU 文件时使用 runtime image 及初始传入的实际 snapshots：

```powershell
build-ftlpu-vs2026/runtime/ftlpu_icu_program_export.exe `
  results/qwen3_5_0_8b_decoder_layer_prefill/continuous/resident.runtime.ftlpu `
  results/qwen3_5_0_8b_decoder_layer_prefill/continuous/runtime_icu_programs `
  --pre-execution-dir results/qwen3_5_0_8b_decoder_layer_prefill/continuous/resident.pre_execution
```

每个物理 ICU 一个 CSV，保留实际指令字、wait_cycle、同步 tag 及初始
传入的 load_id。CSV 中的指令起止周期是队列重放周期；SYNC 的真实
等待及 FU 实际发射周期看 runtime pipeline。

外部输出传输也使用原生 MEM_READ_SYNC，逐向量等待 C2C TX 的读请求。
DDR 写端会按 lane 串行接收，即使总带宽足够，也不能用固定周期的
MEM Read/Repeat 同时向多个 lane 连续灌数据。真实输出布局的回归
覆盖 West bank1 的 `[32,1024]` BF16 双平面，以及 East bank1 的
`[2048,128]` FP32 四平面（1 MiB），均要求 C2C 往返逐字节一致。

2026-10-13 的真实尺寸连续运行已通过固定判据：output NRMSE 0.8451%、
cosine 0.9999665；recurrent-state NRMSE 0.5415%、cosine 0.9999883；
conv-state NRMSE 0.3413%、cosine 0.9999973。一次计算程序运行，0 次中间
主机传输，13 个动态权重页，共 20.625 MiB，14 个执行 epoch。
所有 ICU iMEM 和单 context 检查通过；编译文件 1,458,620 bytes，实际
runtime 文件 1,476,302 bytes，主计算程序的物理 iMEM payload 为
923,052 bytes。主程序有 208 条 MEM_WRITE_SYNC 和 1,643 条 SYNC_PAGE。
导出的 246 个 ICU CSV 同时包含初始 C2C 传入快照，统计时应按 phase
区分初始装载和主程序。最后的外部输出使用独立 C2C I/O 事务，pipeline
中以 HostOutput 区间记录，不包含在主程序的 READ_SYNC 计数内。

当前结果入口 `results/qwen3_5_0_8b_decoder_layer_prefill/summary.json`
已更新为连续程序验证。运行图覆盖 MEM/MXM/VXM/SXM/C2C，共 47 条资源
行，含初始传入和最终输出，总跨度 29,042,780 个实际仿真周期。
这是保守调度的一层 linear-attention decoder，不是完整模型的 TTFT。

## 独立 Input RMSNorm

`compiler/tests/qwen3_5_input_rmsnorm_runtime_test.py` 单独运行一个逻辑
Input RMSNorm，使用真实 token-id 0..31 的 `[32,1024]` 激活、
`BF16(1 + input_layernorm.weight)` 和 epsilon `1e-6`。
使用 dense transformer 共用的 `kernel.rms_norm` → Tensor/Stream RMSNorm
task → 专用 `vxm_feedback` lowering，直接生成输入转置、反馈归约与
归一化、输出布局恢复。整个 `[32,1024]` 对应一个算子，无需原来的
8 个 scalar-broadcast 块及额外输出 join，只有一个计算 program，
中间主机传输为 0。默认 target 与 dense 测试相同，来自
`FTLPU-CMODEL/config/ftlpu-lpu32.json`，也可以通过 `--target-config`
显式指定。输入输出均走 C2C。结果目录为
`results/qwen3_5_0_8b_input_rmsnorm`，`runtime_icu_programs/index.csv`
包含实际装载的主程序和初始传入快照。

```powershell
D:/Apps/Anaconda3/python.exe compiler/tests/qwen3_5_input_rmsnorm_runtime_test.py `
  --package results/qwen3_5_0_8b_decoder_layer_prefill/package `
  --opt build-ftlpu-vs2026/compiler/ftlpu_opt.exe `
  --compile build-ftlpu-vs2026/compiler/ftlpu-compile.exe `
  --runner build-ftlpu-vs2026/runtime/compiled_stage_file_runner.exe `
  --output-dir results/qwen3_5_0_8b_input_rmsnorm
```

2026-10-13 共享路径运行通过：NRMSE 0.1879%，cosine 0.9999983；actual
runtime program 15,164 bytes，主程序物理 iMEM payload 8,640 bytes，
容量和单 context 检查通过。数值参考使用相同 BF16 激活和 gamma
的 FP32 方程。此处的结果更新只针对独立 Input RMSNorm，上述整层
连续程序的历史数值与性能结果仍对应当时的实现，未重新运行整层。

Qwen3.5 的 serial/resident 两种 graph harness 的普通 RMSNorm 都使用
相同的 `common_kernel_graph.rms_norm_source`，gamma 的 `1 + weight`
由模型适配器提供。L2 normalization 和 DeltaNet gated normalization
仍是独立的算子语义。Full/Linear Attention 的图展开可以不同，共有
算子不再因 attention 类型选择不同的硬件 lowering。CModel 无新增
模型级操作，只执行原生 ICU/FU。

连续布局衔接回归 `shared_rmsnorm_resident_runtime_test.py` 在单个程序
内执行两个 `[32,1024]` RMSNorm（epsilon 分别 `1e-6`/`1e-5`）及一个
residual add，NRMSE 0.2768%，中间主机传输为 0。

## 连续 Prefill 前端：RMSNorm、QKV、Conv1D + SiLU

`compiler/tests/qwen3_5_prefill_frontend_runtime_test.py` 在共享 Input RMSNorm
之后加入真实 layer-0 的 `in_proj_qkv` 和 causal Depthwise Conv1D + SiLU。
形状为 seq32、hidden1024、Q/K/V 各 2048 通道，总宽度 6144，卷积为
4 tap，初始历史为零。Projection 使用真实 checkpoint 导入的 W8A16
权重，RMSNorm gamma 和卷积权重为 BF16。FP32 参考使用同样的量化
权重，不代表与未量化 checkpoint 的误差。

这三个逻辑算子生成一个连续 ICU program，QKV 按 SRAM 几何分为
3 个 2048-column 权重页，共 6 MiB，通过动态 DDR/C2C 传入；卷积按
1024-channel 域生成 6 块。MEM/MXM/VXM/SXM 仍执行原生 ICU/FU 指令，
没有中间主机计算或回传。测试结束时额外导出 RMSNorm/QKV 诊断结果，
因此 runtime pipeline 的末尾还包含这些诊断输出传输。

```powershell
D:/Apps/Anaconda3/python.exe compiler/tests/qwen3_5_prefill_frontend_runtime_test.py `
  --package results/qwen3_5_0_8b_decoder_layer_prefill/package `
  --opt build-ftlpu-vs2026/compiler/ftlpu_opt.exe `
  --compile build-ftlpu-vs2026/compiler/ftlpu-compile.exe `
  --runner build-ftlpu-vs2026/runtime/compiled_stage_file_runner.exe `
  --output-dir results/qwen3_5_0_8b_prefill_frontend
```

同一目录持续更新此递增前端测试。`runtime.pipeline.csv` 是包含初始
C2C 输入的实际运行轨迹，`frontend.runtime.ftlpu` 是实际装载的程序，
`runtime_icu_programs/index.csv` 索引每个物理 ICU 的 CSV，包括初始
输入的执行快照。与原有单算子及整层历史结果分开保存。

2026-10-13 双半球卷积修改前运行通过：RMSNorm NRMSE 0.1879%、QKV 0.3546%、
Conv1D + SiLU 0.3900%，cosine 均大于 0.99999。runtime program
68,198 bytes，物理 iMEM payload 40,896 bytes，容量及单 context
检查通过。测试为一个 program、4 个 epoch、3 个动态权重页。

Resident 分配器从原生 MEM write/write_tap 域推导 bank1 workspace
范围，避开这些范围存放不可变参数与归档值。不能假定 projection
结果永远在 slice28/29：当前 target 的结果在 slice0/1，若把卷积
权重预置于其 row0..1023 内，就会被 projection 覆盖。

前端测试的初始激活通过 `direct_initial_inputs=[(0,0)]` 直接由 C2C
传入 RMSNorm 的原生 distributed16 工作区：两侧 slice0..15、bank0、
row0..127。无需先存 bank1 再经过 VXM 搬回 bank0。此选项要求第一
算子的外部输入没有后续外部消费者，且不与预取权重存储重叠；其余
输入仍可通过设备 materialize 保留内部 scratch 的原生 bank/地址。
整个初始搬运域被移除，RMSNorm 的实际 FU 发射区间从
`[16382,18810)` 提前至 `[5846,8274)`，减少 10,536 个调度周期。
三段数值误差保持不变。原 cycle5813..5941 的 VXM 输入搬运已消失。

RMSNorm 到 projection 的边通过 `direct_edges=[((0,0),(1,0))]` 直接
物化到 projection 的 `fp16_pair_planar` 输入：两侧 slice0/1、bank0、
row0..1023。其后的三个 projection 页直接复用该区，不再经过 West
slice32/33 中间归档。诊断输出也从这个保持存活的输入区读取。
直接边要求完整、类型匹配的张量，目标不能在存活期间被后续原生域写坏。

此前的纯搬运桥统一发两个 VXM cast（硬件 opcode 为 bypass）。
`raw_sr_copies=True` 改为字节级 MEM/SR 通路：同 fabric 的可达方向使用
普通 SR hop；跨 fabric 通过现有 VXM 区域的被动 SR 链传输，但不配置
VXM ALU，也不做 BF16/FP32 解码转换。使用 SR20..23，避开当前 RMSNorm
和 Conv 的 ALU 输入，包括镜像 RMSNorm 使用的 SR16..19；C2C 规划能
看到生成的 MEM 流占用。同 fabric、同 MEM 组没有直接 U-turn 通路，
仍保留原桥接 fallback。这个选项用于当前前端的串行桥接阶段。

RMSNorm 到 projection 之间已无 VXM bypass。相同位宽但不同地址布局
的转换仍由几何生成的 MEM read/write 3-D 域完成，不需要数值计算。
静态调度由 465,091 降至 453,204 cycle，减少 11,887 cycle；这不是
包含 DDR 等待及诊断输出传输的实际总时延。三段 NRMSE 保持不变。
`resident_sr_copy_runtime_test.py` 覆盖同 fabric 双向、同组跨 fabric
及 FP32 跨 fabric，任意字节（含 NaN 位型）逐位一致，且无 VXM 指令。

### Conv1D + SiLU 双半球镜像并行

当前前端把 Q、K、V 各自的 2048 通道分成两个 1024-channel 域。
三轮分别处理 Q、K、V，每轮两侧同时执行，共覆盖 6144 通道。
主链使用 C0..C7 的卷积乘加树、C0..C5 的 SiLU；镜像链使用
C15..C8 的卷积乘加树、C15..C10 的 SiLU。MEM 激活、权重、历史
padding、scratch 和结果放置均反射到另一侧，内部 FP32 数据走
反向 SR，避免与主链输入流碰撞。两侧的数据彼此独立。

`ftlpu.depthwise_conv_mirror = true` 由原生 DepthwiseConv1D emitter
直接生成镜像调度域。Resident linker 在同一 `parallel_group` 内
先准备两侧输入，再同时发射两侧原生域，等两侧完成后分别归档结果。
每个 ICU 仍只有一个活动 context，没有交错执行不同 ICU 指令。
两侧结果的布局 join 目前仍是单独阶段，因此整段前端不会减半。

C2C 的 SR 生命周期摘要只需要覆盖最后一个分页权重 consumer
及其之前开始的域；之后才开始的 Conv 流不能成为此前 QKV 页的
预取下限。实际 DDR 延迟由 compiler 发出的 page SYNC 保护，
较晚的传输会暂停后续计算 ICU，直到权重页完成。保留域的完整
结束时间和 SR drain，不能把正在使用的流提前释放。

2026-10-13 镜像并行版真实 seq32 前端运行通过：RMSNorm NRMSE
0.1879%、QKV 0.3546%、Conv1D + SiLU 0.3900%，与修改前相同。
`conv_mirrored_runtime.csv` 从真实运行轨迹确认 C0/C15 的 MAC 与
SiLU 启动区间完全一致：

| 通道组 | 两侧 MAC 发射区间 | 两侧 SiLU NEG 发射区间 |
|---|---|---|
| Q | [322479,323503) | [326639,327663) |
| K | [361040,362064) | [365200,366224) |
| V | [399663,400687) | [403823,404847) |

以上是 2026-10-13 对应 ALU 指令的 FU 发射区间，不包含整个算子的其他 ALU、
MEM 完成延迟和布局搬运。当时实际装载 program 为 71,196 bytes，
主程序物理 iMEM payload 为 42,696 bytes，容量与单 context 检查
通过。静态调度加 drain 为 413,935 cycle，修改前为 453,204，
减少 39,269 cycle（8.67%）；这不等于包含 DDR 及诊断输出的 TTFT。
实际 DDR/C2C 等待仍为 55,274 cycle，共三个动态权重页、四个 epoch。

`depthwise_conv_mirrored_runtime_test.py` 覆盖独立镜像链以及两条
1024-channel 链同时执行，NRMSE 分别为 0.2392% 和 0.2405%。
共享 RMSNorm→RMSNorm→residual 回归也通过（NRMSE 0.2768%）。

### 移除 Conv 附近的纯复制 ALU bypass

2026-10-14 更新：真实前端 Conv 附近的 VXM bypass 域由 44 降至
18。移除的 26 个域包括六条通道链各两个输入复制 cast，以及
七次布局搬运各两个 cast。卷积输入仍要复制到四组 causal window
MEM slice，但改为字节级 MEM read → 被动 SR 链 → MEM write_tap/
write，不配置 ALU，不做 BF16→FP32→BF16 的往返转换。

Resident 分配器优先选择与 producer、consumer 位于不同 MEM 组的
归档地址，使同 fabric 搬运可以沿单一方向通过 SR。布局 join
及最终输出也采用相同规则，无需增加中间复制。若 SRAM 只有同组
位置可用，通用分配器仍保留原有桥接 fallback；当前前端整个
Schedule 中的 `%dummy` VXM 搬运域为零。

当前仍显示为硬件 `bypass` 的 18 个域承担数值转换，而非纯复制：

| 用途 | ALU | 域数 |
|---|---|---|
| causal 前三行零填充结果 FP32→BF16 | C1、C14 | 6 |
| 四 tap 卷积归约结果 FP32→BF16 | C7、C8 | 6 |
| SiLU 结果 FP32→BF16 | C5、C10 | 6 |

硬件 opcode 把 `cast` 编码为 `bypass`，输出格式控制实际的 BF16
舍入与编码，因此不能把这些域当成单纯直通直接删除。
`conv_bypass_runtime.csv` 逐条记录其实际周期、ALU 和用途；
`conv_mirrored_runtime.csv` 继续验证两侧 MAC/SiLU 同时执行。

seq32 真实权重前端数值保持不变：RMSNorm NRMSE 0.1879%、QKV
0.3546%、Conv1D+SiLU 0.3900%。独立镜像、双链并行以及五个
tokens/channels/activation 回归均通过。实际 runtime program
69,870 bytes，主程序物理 iMEM payload 41,760 bytes，容量及单
context 检查通过。当前静态调度加 drain 为 413,842 cycle；
主要收益是释放 ALU 和缩小程序，MEM 数据复制本身仍需要执行。
所有最新结果继续更新 `results/qwen3_5_0_8b_prefill_frontend`。
