# Qwen2.5-1.5B decoder 层

本目录提供 `seq_len=32` 和 `seq_len=128` 两套标准 StableHLO decoder 层输入。
两者均采用 Qwen2.5-1.5B 的真实尺寸：hidden size 1536、intermediate size
8960、12 个 Q head、2 个 KV head、head dimension 128、RoPE theta 1,000,000，
以及 RMSNorm epsilon 1e-6。

## 当前 seq32 状态

完整层流程现在把标准 StableHLO 依次 lower 到 Kernel IR、Tensor IR、Stream IR，
再直接生成各功能单元的闭式 Schedule 域和 raw hardware Command 包。标准路径固定
设置 `ftlpu.command_lowering = "direct"`、`ftlpu.icu_compression = "none"` 和
`ftlpu.mem_slice_program = false`；不会先生成逐 cycle 细指令再做压缩。

以下已验证数据使用 `--projection-rope-overlap on`，即 projection 与 RoPE
按输出组流水重叠；切换到默认的 `off` 后，指令数、镜像大小和周期数会变化。
已验证的 `seq_len=32`、weight bank 1、KV capacity 256 构建结果如下：

| 项目 | 结果 |
| --- | --- |
| Binary 大小 | 1,161,429 bytes（1.108 MiB） |
| 序列化物理队列数 | 226 |
| Raw FU 域总数 | 19,876 |
| 逻辑 ICU 指令总数 | 38,880 = 19,876 条工作包 + 19,004 条时长 NOP |
| MEM READ/WRITE_3D / WRITE_READ_2D / MXM load / dequant / compute 域 | 17,944 / 384 / 292 / 232 / 226 |
| VXM / SXM 域 | 662 / 136 |
| 物理 iMEM word | 78,290 |
| counter 展开后的工作量 | 6,652,186 次发射 |
| Binary `max_cycle` / measured end | 768,137 / 768,201 |
| Raw FU-ICU 物理分析 | `deployable=yes`，每个队列 peak context 均为 1 |
| 完整 decoder CModel | 通过，49,152 个 BF16 输出，49,116 个非零，最大误差 0.0625 |
| 动态 C2C 权重页 | 通过，20 次使用、21 次预取，运行时 page-ready 等待 0 cycle |

第一版硬件合法的统一 MEM 基线包含 26,010 个 raw 域，其中 MEM 域 23,776 个，
binary 为 1,554,521 bytes。选择性归纳先把它降到 1,458,329 bytes、24,258 个
raw 域和 22,104 个 MEM 域。projection 的最大闭式域随后把它降到 1,246,745 bytes、
21,202 个 raw 域和 19,576 个 MEM 域。在 `--projection-rope-overlap on` 下加入激活区
ping-pong staging 后，Q projection 可以跨 RoPE 连续执行，不再保留原来的 `2+46`
边界。该模式此前的镜像为
1,219,763 bytes、20,852 个 raw 域和 19,304 个 MEM 域。QKV 权重 READ 从最初的
720 条降到 256 条，O projection 从 384 条降到 16 条，Gate 和 Up 合计从 2,304 条
降到 192 条；Down 因同队列 stream 和 weight-buffer 复用形成真实边界，仍为
384 条。counter 展开后的 FU 工作量为 6,652,186 次。

当前 Q projection 用三个 96-bit word 的 `WRITE_READ_2D` MEM ICU 指令，把
MXM 原始结果写入和随后的跨半球镜像读取配成一条指令。共 384 条（12 个输出组 ×
2 个半球 × 16 个 slice），直接从闭式域生成，不先展开逐周期 FU 指令。镜像流号按
物理源 slice 固定；最后一组 Q 使用另一段流号，避开紧随其后的 V projection。

这些域由各算子 emitter 直接生成。QKV 权重队列使用完整的
`wave_count=48, group_count=2` 域，不再把一次 reduction 拆成 `2+46`。第一段与
RoPE 重叠的 Q projection MXM 域（`on` 模式）从 cycle 8,294 开始，以
32-cycle 间隔覆盖全部
48 个 reduction，同时上一 projection 的 RoPE 继续运行。activation read 在激活
SRAM 中使用两份副本：bank 0 的 slice 8/9 是 primary，bank 0 的 slice 0/1 是 pong。
direct lowering 只把会与同 MEM 队列 RoPE write 冲突的 read 区间路由到 pong，因此
MXM 域可以连续运行，每个 MEM ICU 仍然只有一个 live context。

对于其他 projection，每条 O-projection 权重队列只需一条
`counts=(4,48,24)`、cycle stride 为 `(1,32,1578)` 的 READ；Gate 和 Up 使用
MEM ICU 的 blocked-outer 地址模式表达交替 weight buffer，不再把每个 pair 拆开。
只有 operand read 与 result write 落在同一物理 MEM 队列时，residual block 才在
direct lowering 阶段拆域，并由 VXM 流水延迟结果，保证 read 域退出后再执行
write 域。

每个物理 `(hemisphere, slice, bank)` MEM ICU 只有一份 iMEM、一个指令 FIFO 和一个
PC。`READ_3D`、`WRITE_3D`、`WRITE_TAP_3D` 以及外部链接进来的
`MEM_WRITE_SYNC` 都在这一个队列中执行。SRAM 即使有独立读写端口，也不能交错执行
两条 ICU 粗指令。RoPE pair 域和发生别名的 residual 域由各自 schedule emitter
直接按硬件顺序生成，因此每个队列最多只需要一个 live FU-domain context。CModel
分阶段最大误差为 FFN 0.03125、attention context 0.0195312、residual 0、
RMSNorm 2 为 0.015625。

### 动态 C2C 单层执行

同一个 seq32 runtime 测试还会通过 `ModelSession(C2cDmaSystem)` 执行整层。
executable 声明 20 个权重页使用项：Q、K、V、O 各 1 页，Gate 2 页，Up 2 页，
Down 12 页。ModelPackage 另建 1 个参数页，装入两组 RMSNorm scale 和 Q/K/V bias。
该参数页不计入 20 个 executable page use，因此整次运行共有 21 次物理预取。

在加载链接后的 executable 之前，`ModelSession` 会同步搬入参数页，以及 planner
标为 `pre_execution` 的 6 个 executable 页，即 Q/K/V/O 和两个 Up 页。其余
14 个 executable 页（两个 Gate 页和全部 12 个 Down 页）会链接进静态程序的空闲
窗口。linker 向 C2C DMA/RX 队列和普通 MEM ICU 队列加入指令；后者使用
`MEM_WRITE_SYNC`，与普通读写共享同一物理队列。page fence 会保留目标队列直到首个
consumer 边界，page-ready 同步负责吸收更晚的传输完成时间，避免逻辑静态时序偏移。

启动前传输程序是临时程序：它们在 `runtime.load()` 之前完成，不会复制到
`ModelSession::last_linked_program()`。任何 `BinaryProgram` 都不序列化页面 payload；
所有页数据仍保存在 `ModelPackage`/外部 DDR backing store 中。因此，链接后的
`.ftlpu` 包含原静态 executable 和 14 个重叠页的传输指令，同时仍保留原有 20 个
page-use descriptor 作为 metadata。统计导出的 linked image 指令时必须区分这两类页面。

已验证的动态运行统计如下：

| 动态 C2C 项目 | 结果 |
| --- | --- |
| 页面计划 | 20 个 executable 页 + 1 个参数页 = 21 次预取 |
| 物理传输字节数 | 47,194,112 = 46,792,704 executable + 401,408 parameter |
| 启动页 / 初始等待 | 7 页 / 217,183 cycles |
| 运行期链接页面 | 14 个 executable 页 |
| 链接后指令镜像 | 1,321,249 bytes、230 个序列化队列、85,242 个物理 iMEM slot |
| 链接后的 MEM 同步 | 256 条 `MEM_WRITE_SYNC` 指令 |
| 同步 MEM FU 工作量 | 860,160 次 vector 写入 = 27,525,120 bytes |
| executable page-ready 等待 | 0 cycles |
| 数值结果 | 与 direct run 逐字节一致；49,152 个 BF16 值、49,116 个非零、最大误差 0.0625 |

## 编译和检查

`ftlpu-opt` 和 `ftlpu-compile` 均接受 `--projection-rope-overlap on|off`，省略时
默认为 `off`。`off` 模式先完成所有 Q projection，并将全部 Q 结果写入独立的
RoPE staging 地址；最后一次写入完成后，才统一读取这些结果并执行所有 Q RoPE。
K projection 和 K RoPE 也按同样顺序执行。`on` 模式保留按输出组交叠的
projection/RoPE 流水。这个开关控制 projection 与 RoPE 的跨阶段重叠；已有的
MXM 内部执行流水和动态 C2C 页面传输并不由它控制。分阶段使用 `ftlpu-opt` 时，
可在 StableHLO 到 Stream 阶段设置一次；后续工具会从 IR 中继承
`ftlpu.projection_rope_overlap` 属性，也可以再次用命令行显式覆盖。

在 seq32 串行配置下，Q、K、V projection 的激活和权重读取现已直接 lower
为每个参与的物理 MEM ICU 各一条 `READ_3D`。第三维覆盖全部输出半组：Q 为
24，K/V 各为 4。V 的跨半球输出搬运也移到完整 V projection 之后，避免打断
读取域；C2C 权重页同步及结果写入仍是独立的 MEM ICU 指令。

例如，直接生成串行版本的 seq32 Schedule，并对比开启流水的版本：

```powershell
build-ftlpu-vs2026-direct/compiler/ftlpu_opt.exe `
  --input compiler/examples/qwen2_5_1_5b_decoder_layer/decoder_layer_seq32.stablehlo.mlir `
  --output build-ftlpu-vs2026-direct/qwen2_5_seq32_serial.schedule.mlir `
  --pipeline ftlpu-stablehlo-to-schedule `
  --target-config ../FTLPU-CMODEL/config/ftlpu-lpu32.json `
  --weight-bank 0 --mxm-execution vector `
  --projection-rope-overlap off

build-ftlpu-vs2026-direct/compiler/ftlpu_opt.exe `
  --input compiler/examples/qwen2_5_1_5b_decoder_layer/decoder_layer_seq32.stablehlo.mlir `
  --output build-ftlpu-vs2026-direct/qwen2_5_seq32_overlap.schedule.mlir `
  --pipeline ftlpu-stablehlo-to-schedule `
  --target-config ../FTLPU-CMODEL/config/ftlpu-lpu32.json `
  --weight-bank 0 --mxm-execution vector `
  --projection-rope-overlap on
```

下面的完整层 pipeline 测试为了保留上文已验证的流水基线，会显式使用 `on`。
检查两种模式下 Q projection 与 Q RoPE 的实际发射顺序，可运行
`ctest --test-dir build-ftlpu-vs2026-direct -C Release -R qwen2_5_projection_rope_overlap_test --output-on-failure`。

使用以下命令生成 `seq_len=32` raw FU executable：

```powershell
python compiler/tests/qwen2_5_1_5b_decoder_layer_pipeline_test.py `
  --opt build-ftlpu-vs2026-direct/compiler/ftlpu_opt.exe `
  --compile build-ftlpu-vs2026-direct/compiler/ftlpu-compile.exe `
  --input compiler/examples/qwen2_5_1_5b_decoder_layer/decoder_layer_seq32.stablehlo.mlir `
  --target-config ../FTLPU-CMODEL/config/ftlpu-lpu32.json `
  --weight-bank 1 `
  --kv-cache-capacity 256 `
  --output-dir build-ftlpu-vs2026-direct/compiler/ftlpu_lower/qwen2_5_1_5b_decoder_layer
```

运行 direct-domain 重点检查：

```powershell
python compiler/tests/qwen2_5_1_5b_decode_reference_test.py
ctest --test-dir build-ftlpu-vs2026-direct -C Release `
  -R "ffn_up_3d_lowering_test|standalone_ffn_up_3d_compile_test|ffn_swish_emitter_test|schedule_trace_weight_page_test" `
  --output-on-failure
```

完整 CModel 检查：

```powershell
build-ftlpu-vs2026-direct/runtime/compiled_qwen2_5_1_5b_decoder_layer_seq32_runtime_test.exe `
  build-ftlpu-vs2026-direct/compiler/ftlpu_lower/qwen2_5_1_5b_decoder_layer/decoder_layer.ftlpu
```

把整个 prefill 的静态 ICU 程序按物理 ICU 拆成独立文件：

```powershell
build-ftlpu-vs2026-direct/runtime/ftlpu_icu_program_export.exe `
  build-ftlpu-vs2026-direct/compiler/ftlpu_lower/qwen2_5_1_5b_decoder_layer/decoder_layer.ftlpu `
  build-ftlpu-vs2026-direct/compiler/ftlpu_lower/qwen2_5_1_5b_decoder_layer/icu_programs
```

`icu_programs/index.csv` 给出资源、队列号、物理位置、粗指令数、i-MEM 字数和
counter 展开工作量。每个 `*.icu.csv` 对应一个物理 ICU；每行是一条 ICU 粗指令，
包含物理 `pc_word`、循环域、解码字段和完整 96/128-bit 原始编码。没有静态任务的
ICU 也保留为空文件。每个物理 `(hemisphere, slice, bank)` MEM ICU 只有一个 CSV，
`READ_3D`、`WRITE_3D`、`WRITE_TAP_3D` 和 `MEM_WRITE_SYNC` 按 PC 顺序放在同一文件中。
带 `_read.icu.csv` 或 `_write.icu.csv` 后缀的是统一前的旧导出；重新运行 exporter 时会
先清掉输出目录内残留的 `*.icu.csv`。这个基础镜像的 C2C 文件为空，因为它只包含
静态计算程序和 page-use metadata，尚未由 `ModelSession` 链接动态 C2C 指令。

设置 `FTLPU_QWEN_C2C_LINKED_BINARY` 可以导出 runtime 实际加载的链接后程序，
再按物理 ICU 拆分该镜像：

```powershell
$env:FTLPU_QWEN_C2C_LINKED_BINARY = `
  "build-ftlpu-vs2026-direct/compiler/ftlpu_lower/qwen2_5_1_5b_decoder_layer/decoder_layer.linked.ftlpu"
build-ftlpu-vs2026-direct/runtime/compiled_qwen2_5_1_5b_decoder_layer_seq32_runtime_test.exe `
  build-ftlpu-vs2026-direct/compiler/ftlpu_lower/qwen2_5_1_5b_decoder_layer/decoder_layer.ftlpu
Remove-Item Env:FTLPU_QWEN_C2C_LINKED_BINARY

build-ftlpu-vs2026-direct/runtime/ftlpu_icu_program_export.exe `
  build-ftlpu-vs2026-direct/compiler/ftlpu_lower/qwen2_5_1_5b_decoder_layer/decoder_layer.linked.ftlpu `
  build-ftlpu-vs2026-direct/compiler/ftlpu_lower/qwen2_5_1_5b_decoder_layer/linked_icu_programs
```

linked export 中的 C2C DMA/RX 队列不再为空，MEM 队列包含 14 个重叠页对应的
256 条同步指令。7 个启动页在 linked executable 加载前已经到达 SRAM，因此其临时
传输程序不会出现在该镜像中。测试成功时会写出这个诊断镜像；如果执行在完成链接后
失败，也会尝试保留该镜像。

生成 MEM 专用逐周期 CSV：

```powershell
$env:FTLPU_QWEN_MEM_CSV = `
  "build-ftlpu-vs2026-direct/compiler/ftlpu_lower/qwen2_5_1_5b_decoder_layer/decoder_layer.mem.csv"
build-ftlpu-vs2026-direct/runtime/compiled_qwen2_5_1_5b_decoder_layer_seq32_runtime_test.exe `
  build-ftlpu-vs2026-direct/compiler/ftlpu_lower/qwen2_5_1_5b_decoder_layer/decoder_layer.ftlpu
Remove-Item Env:FTLPU_QWEN_MEM_CSV
```

用浏览器打开 `tools/pipeline_viewer/mem.html`，再载入
`decoder_layer.mem.csv`。该 CSV 逐项记录 MEM ICU 状态、tile 0..3 流水和实际 SRAM
读写；完整 seq32 层会产生较大的文件。

binary inspector 可以为 Pipeline Viewer 生成 raw-domain CSV：

```powershell
build-ftlpu-vs2026-direct/runtime/ftlpu_binary_inspect.exe `
  build-ftlpu-vs2026-direct/compiler/ftlpu_lower/qwen2_5_1_5b_decoder_layer/decoder_layer.ftlpu `
  --all-queues `
  --trace build-ftlpu-vs2026-direct/compiler/ftlpu_lower/qwen2_5_1_5b_decoder_layer/decoder_layer.pipeline.csv
```

用浏览器打开 `tools/pipeline_viewer/index.html`，载入
`decoder_layer.pipeline.csv`。第三层 counter 的每个切片写一行，前两层 counter
继续保留为 `repeat`/`repeat2d` pattern，由 Viewer 只在当前窗口按需展开。这个 CSV
是直接从实际 raw packet 解码得到的离线视图；上面的 CModel 结果提供执行和数值验证。

## 单 token decode reference

`compiler/tests/qwen2_5_1_5b_decode_reference_test.py` 使用真实的 Qwen2.5-1.5B
层尺寸，先为 32 个 token 生成 RoPE 后的 K cache 和 V cache，再在绝对位置 32
执行一次单 token decode。测试验证 GQA 12:2 head 映射、KV append、cache 前缀保持、
decode attention、残差和 1536/8960 FFN，并与 33-token 单层数学 golden 的最后一个
token 逐位比较。

该测试建立 compiler decode lowering 所需的数值规范；当前生成的 `.ftlpu`
executable 尚未包含编译后的 KV cache read/write command，因此它不代表 compiled
CModel decode 已经实现。

## 历史兼容路径数值基线

早期兼容路径的 CModel 测试使用确定性稀疏 INT8 权重，49,152 个 BF16 输出全部
通过，其中 49,127 个非零。分阶段最大误差为：FFN 0.03125、attention residual 0、
第二次 RMSNorm 0.03125。

真实 checkpoint FFN 流程使用 `compiler/tools/import_hf_ffn.py` 导入 Hugging Face
Qwen2.5-1.5B 第 0 层 Gate、Up、Down 权重和 embedding 产生的 BF16 输入，执行
per-tensor INT8 量化，并模拟 BF16 MXM 反量化、FP32 partial 累加以及 VXM FP16
LUT 实现的 SwiGLU。早期 v22 兼容路径上传 26 个权重页，并在 cycle 770,646 前完成
全部 49,152 个 BF16 输出的数值比较：超阈值 mismatch 为 0，MAE 为
`6.54569e-05`，RMSE 为 `0.000266376`，最大误差为 `0.00418091`。

这些结果只保留为兼容 schedule 的历史数值基线。当前 direct binary 及其每队列单
context 验证见上文；两类结果都不是 Hugging Face 原始 checkpoint 全部 28 层的执行。
