# FTLPU 软件栈开发规范

版本：0.3 工作草案。基线：2026-09-18 的 FTLPU-SOFTWARE 与 FTLPU-CMODEL 工作区；可执行文件格式 v32，模型包格式 v6。本文面向刚接手编译器、runtime 或 CModel 联调的开发者；读完后应能独立构建、编译并运行一层 Qwen2.5 prefill，查明当前算子的支持范围与物理映射，定位一条 ICU 指令和一次 C2C 搬运，并知道改动应落在哪一层。

本文使用“必须”描述跨层契约，使用“当前”描述已实现行为，使用“待实现”描述尚未完成的硬件或算子能力。代码和 target JSON 是发生版本差异时的最终校验依据；不要用旧实验的 binary 大小、cycle 或 slice 常量替代当前目标配置。

## 1 先看结论和范围

FTLPU 软件栈把 StableHLO 模型计算图变成按物理 ICU 分队列的 `.ftlpu` 指令程序；`.ftlpum` 在它外面封装模型权重、调用顺序和持久状态。编译器决定数据放在哪里、沿哪条 stream 走、何时由哪条 FU 队列发射；runtime 检查 ABI、装载程序、经 DDR/C2C 搬运数据并处理 page-ready；CModel 按 cycle 解码 ICU 指令并执行 FU。硬件 ICU 不读取编译器的全局绝对 cycle。

目前最完整的编译执行闭环是 Qwen2.5-1.5B、序列长度 32 的单层 prefill。单 token decode 已有数学 reference 与 KV state ABI，但编译后的 executable 尚未完成完整 KV cache read、append 和历史前缀 attention。Embedding lookup 和 LM head 在模型包中仍是显式 host operation；它们不是 ICU 指令。当前可执行参考是 CModel，RTL/芯片的下载寄存器和硬件事件协议仍待冻结。

| 名称 | 本文含义 | 常见误解 |
| --- | --- | --- |
| ICU 指令 | 装入某个物理 ICU 的粗粒度包，带本地等待、循环和操作数归纳 | 不是 projection、attention 或完整算子 program |
| FU 指令 | ICU 在某一 cycle 向 MEM、MXM、VXM、SXM、C2C 发出的细操作 | 不需要逐条存进 executable |
| 逻辑 cycle | 编译期调度与 binary 中 consumer/release 提示的时间轴 | 不是硬件可读的全局时钟 |
| 物理 cycle | CModel 或设备真正推进的时间轴 | C2C 页面未就绪时会比逻辑时间更晚 |
| Binding | 逻辑值与物理 SRAM layout、slice、bank、row 的契约 | 不是单纯的 host 指针 |
| Target ABI | 共享目标配置对拓扑、容量、编码和时序参数的身份校验 | 相同模型 shape 不代表 ABI 兼容 |

## 2 十分钟上手路径

两个仓库应放在同一父目录：`FTLPU-SOFTWARE` 与 `FTLPU-CMODEL`。在 Windows 上使用 Visual Studio 2026 Developer PowerShell，或 MSVC 19.36 及以上工具集；需要 CMake 3.20 及以上、Ninja、Python 3、Git 和充足磁盘空间。编译器直接依赖仓库固定的 LLVM/MLIR 与 StableHLO 子模块；IREE 是参考工程，不是构建必需依赖。

首次配置先初始化依赖，再构建 MLIR。下面命令在 `FTLPU-SOFTWARE` 根目录执行；如果已有兼容的 MLIR build，可跳过第一段并把 `FTLPU_MLIR_DIR` 指向其 `lib/cmake/mlir`。

```powershell
git submodule update --init --recursive third_party/llvm-project third_party/stablehlo
cmake -S third_party/llvm-project/llvm -B build-mlir-dev -G Ninja `
  -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_PROJECTS=mlir `
  -DLLVM_TARGETS_TO_BUILD=Native -DLLVM_ENABLE_RTTI=ON `
  -DLLVM_ENABLE_EH=ON -DLLVM_INCLUDE_TESTS=OFF `
  -DLLVM_INCLUDE_BENCHMARKS=OFF
cmake --build build-mlir-dev -j 8
cmake -S . -B build-dev -G Ninja -DCMAKE_BUILD_TYPE=Release `
  -DFTLPU_MLIR_DIR="$PWD/build-mlir-dev/lib/cmake/mlir" `
  -DFTLPU_CMODEL_DIR="../FTLPU-CMODEL" `
  -DFTLPU_HARDWARE_CONFIG="../FTLPU-CMODEL/config/ftlpu-lpu32.json"
cmake --build build-dev --target ftlpu_opt ftlpu_compile `
  ftlpu_binary_inspect ftlpu_icu_program_export `
  compiled_qwen2_5_1_5b_decoder_layer_seq32_runtime_test -j 8
```

`FTLPU-SOFTWARE` 的 CMake 会编译所需的 CModel 静态代码并生成与 target 匹配的编译期硬件常量；单独构建 CModel 不是软件栈构建的前置步骤。改变共享 target JSON 后，必须重新配置并构建两个工程，不得复用旧 ABI 的 executable。先运行轻量回归，再编译完整单层：

```powershell
ctest --test-dir build-dev --output-on-failure `
  -R "ffn_up_3d_lowering_test|stream_fabric_scheduler_test|schedule_trace_weight_page_test"
python compiler/tests/qwen2_5_1_5b_decoder_layer_pipeline_test.py `
  --opt build-dev/compiler/ftlpu_opt.exe `
  --compile build-dev/compiler/ftlpu-compile.exe `
  --input compiler/examples/qwen2_5_1_5b_decoder_layer/decoder_layer_seq32.stablehlo.mlir `
  --target-config ../FTLPU-CMODEL/config/ftlpu-lpu32.json `
  --weight-bank 1 --kv-cache-capacity 256 `
  --output-dir build-dev/qwen2_5_seq32
build-dev/runtime/compiled_qwen2_5_1_5b_decoder_layer_seq32_runtime_test.exe `
  build-dev/qwen2_5_seq32/decoder_layer.ftlpu
```

该 Python pipeline 固定使用 `--mxm-execution vector`、`--ffn-schedule fused` 和 `--projection-rope-overlap on`，会输出 StableHLO、Stream、Schedule、Command 与最终 `.ftlpu`。最后一个命令执行 CModel 的完整单层数值检查；模型尺寸为 hidden 1536、FFN hidden 8960、12 个 query head、2 个 KV head、head dimension 128。权重使用测试 fixture；真实 checkpoint 导入是另一条部署流程，不能把此回归解释为完整 28 层模型已经运行。

## 3 仓库与代码入口

| 需求 | 首先查看的代码 | 主要职责 |
| --- | --- | --- |
| 新增 StableHLO primitive | `compiler/src/Dialect/Kernel/Transforms/StableHloToKernel.cpp`、`compiler/src/Dialect/Kernel/Analysis` | 算子语义、pattern、SSA 图识别 |
| 分配 SRAM 或改 layout | `compiler/src/Dialect/Tensor/Transforms`、`compiler/src/Dialect/Tensor/Analysis/PhysicalMemoryAllocator.cpp` | binding、scratch、lifetime、page placement |
| 调整片上 route | `compiler/src/Dialect/Stream/Transforms`、`compiler/src/Target/LPUTargetModel.cpp` | endpoint、stream ID、传播延迟、布局策略 |
| 消除流水气泡或加闭式域 | `compiler/src/Dialect/Schedule/Transforms`、`compiler/src/Dialect/Schedule/Transforms/VerifySchedule.cpp` | 算子 schedule emitter、资源预留、同队列合法性 |
| 改 ICU 指令生成 | `compiler/src/Dialect/Command/Transforms/ScheduleToCommand.cpp`、`Fu3DCommandMaterializer.cpp`、`compiler/src/Target/CommandBinary.cpp` | direct lowering、packet 构造、序列化 |
| 改 packet 或 CModel ICU | `FTLPU-CMODEL/include/ftlpu/icu`、`FTLPU-CMODEL/include/ftlpu/core` | codec、每类 ICU decoder、FU 逐周期发射 |
| 改 `.ftlpu` ABI | `runtime/include/ftlpu/software/runtime/binary.hpp`、`runtime/src/runtime/binary.cpp` | 版本、binding、queue、relocation 的读写 |
| 改模型装载与执行 | `runtime/src/runtime/model_package.cpp`、`model_session.cpp`、`session_memory_planner.cpp` | `.ftlpum`、跨 invocation 生命周期、状态 |
| 改 C2C page 流水 | `runtime/src/runtime/c2c_weight_pager.cpp`、`weight_prefetch_plan.cpp`、`weight_page_builder.cpp` | 离线页、DMA/RX/MEM 同步写与 fence |
| 看实际发射与 CSV | `runtime/src/runtime/runtime_execution_trace.cpp`、`schedule_trace.cpp`、`runtime/examples/icu_program_export.cpp` | 动态/静态轨迹、每个 ICU 的程序 |

`compiler/tools/ftlpu_opt.cpp` 选择 MLIR pass pipeline，`compiler/tools/ftlpu_compile.cpp` 直接编译到 executable，`compiler/tools/ftlpu_translate.cpp` 序列化 Command IR。`runtime/examples/binary_inspect.cpp` 和 `icu_program_export.cpp` 是排查二进制最常用的命令行入口。测试通常在 `compiler/tests` 或 `runtime/tests/runtime`；修改哪一层就先运行该层针对性测试，再跑单层端到端回归。

## 4 从模型到执行的交付物

标准路径是 `StableHLO → Kernel IR → Tensor IR → Stream IR → Schedule IR → Command IR → .ftlpu → ModelSession/CModel`。每一层必须只决定自己负责的事实；不能在 runtime 重新发明 schedule，也不能在硬件中出现“projection program”这样的模型专用 opcode。

| 边界 | 必须包含的信息 | 负责检查的错误 |
| --- | --- | --- |
| StableHLO | 静态 shape、元素类型、broadcast、dot 和算子语义 | 不支持的动态 shape 或类型 |
| Kernel IR | matmul、batch matmul、RMSNorm、RoPE、softmax、Swish、elementwise 等 primitive SSA 图 | 缺失生产者、非法 head/维度关系 |
| Tensor IR | 物理 rank-6 地址 `[device, hemisphere, slice, bank, word, byte]`、binding 和 scratch | 容量不足、活跃值覆盖、layout 不匹配 |
| Stream IR | MEM/FU endpoint、方向、stream 范围、route、固定 transport latency | 路由越界、来源/目的不匹配 |
| Schedule IR | 逻辑 cycle、闭式 launch 域、FU/端口/SR 窗口、page ready/release | 资源冲突、读前写、同队列交错 |
| Command IR | 物理 ICU 队列、raw packet、等待与 relocation | 字段超宽、不能由本 FU decoder 表达 |
| `.ftlpu` | target ABI、typed binding、队列、page/sync metadata、iMEM word | 版本/ABI/队列/packet 非法 |
| `.ftlpum` | 常量、页镜像、executable 模板、命名值、持久 state、invocation | 引用悬空、shape/type 不兼容 |

`ftlpu-opt` 可在任意中间 IR 停下：`ftlpu-stablehlo-to-kernel`、`-tensor`、`-stream`、`-schedule`、`-commands`；已有 Stream IR 可用 `ftlpu-stream-to-schedule`，已有 Schedule IR 可用 `ftlpu-schedule-to-commands`。标准 schedule pipeline 设置 `ftlpu.command_lowering = "direct"`、`ftlpu.icu_compression = "none"`、`ftlpu.mem_slice_program = false`。显式选择的旧压缩 pipeline 仅用于兼容和对比；产品路径不先生成逐 cycle FU 列表再反向压缩。

## 5 共享 target 与默认物理模型

当前两仓库共用 `FTLPU-CMODEL/config/ftlpu-lpu32.json`。CMake 校验此 JSON 并生成编译期常量；compiler `--target-config` 读取同一配置。`target_abi` 把影响程序解释的拓扑、容量、布局、吞吐与延迟参数绑定到二进制。探索配置可以编译出不同 schedule，但只有 runtime/CModel 也支持同一 ABI 才能执行。

| 默认项目 | 值与推导 | 开发影响 |
| --- | --- | --- |
| 半球与向量 | 2 个半球；4 tile × 8 lane × 1 byte = 32-byte 物理向量 | 双半球镜像，跨半球要显式 route |
| MEM | 每半球 52 slice；每 slice 2 bank；每 bank 8192 行 × 32 byte = 256 KiB | 全芯片 52 MiB SRAM，而不是 208 MiB |
| MEM ICU | 2 × 52 × 2 = 208 条每 bank 独立队列 | 同一 bank 的 read/write 共享一个 PC/context |
| Stream | 32 条 eastward + 32 条 westward；普通 SR 每跨一列延迟 1 cycle | 路径中每列均占资源，端点延迟另算 |
| 算术 | 物理四个 32×32 MXM、中心 VXM、每半球一个 SXM | 启用/吞吐由 target 配置与算子策略约束 |
| MXM accumulator | 默认每 MXM 256 个 32×32 FP32 block，约 1 MiB | 位于 MXM 内部，不占 MEM slice |
| 外部内存 | LPU 500 MHz；DDR 峰值 51.2 GB/s，规划按 90% 即 92.16 byte/cycle | C2C 链路峰值不等于 DDR 可持续供数速度 |
| DDR latency | 读 35 加 0..15 cycle 抖动；写 25 加 0..10 cycle 抖动 | 动态 page-ready 可晚于静态预测 |

默认专用 slice 角色把每半球 local slice `0..19` 用于 activation/工作区，`20..51` 用于权重；其中 `0..15` 可形成 distributed-16 activation 平面，`16..19` 是辅助工作区。两 bank 使用相同 slice 角色。角色是规划约束，具体 row、bank 和可复用区间仍以当前 binding、memory floor 与 page metadata 为准。

## 6 内存放置与跨层生命周期

Tensor lowering 负责选择 layout、hemisphere、slice、bank、初始 base row、有符号地址 stride，以及 activation、weight、输出和 scratch 的生命周期。普通 SSA value 按最后一次使用释放；Attention scratch 必须按半开生命周期和 row 区间检查重叠；专用算子 profile 可预留固定区域。分配 output 时仍在消费的 operand 必须保持有效，不能因“行号可复用”覆盖输入。

`BinaryBinding` 包含 access class、元素类型、逻辑 shape、字节数、物理 layout、slice 集合、bank、base row、instruction_count、address_stride、role/name、ready cycle 和可选的 page metadata。runtime 只对有 relocation record 的物理地址调整 `base_row`；不能改 slice topology、layout、元素类型或 loop 次序。对有符号 stride，实际保留的半开 row 区间是 `[base_row + min(0,(count−1)×stride), base_row + max(0,(count−1)×stride)+1)`，所以反向遍历权重也必须完整占位。

`BinaryMemoryFloor` 逐 `(hemisphere,slice,bank)` 保留匿名 command scratch 的最低不可分配区域。`SessionMemoryPlanner` 再根据 package 的 producer 到最后 consumer 生命周期，先保留 scratch、普通 non-resident binding 和 state staging，随后在剩余共同 row 区间放置 resident 常量。同一个值若在后继 invocation 中拥有完全兼容的 binding，可保留 DeviceAlias；layout 不一致时必须执行显式 DeviceCopy 或外部传输，不能只重解释字节。

物理空间合法与 cycle 合法是独立证明。同一 bank 上两个不重叠 row 的 C2C write 和 compute read 仍争用一个单端口与一个 MEM ICU；反过来，两个不同 bank 可在 stream/端点不冲突时同 cycle 工作。排查容量错误先看 binding、memory floor、页目标 row；排查发射错误再看队列和 SR 时间窗。

## 7 Stream 和 Schedule 的证明义务

Stream route 声明生产端、消费端、方向、stream 范围、物理边界列和传输延迟。SR fabric 的基本占用键是 `(cycle, column, E/W, stream_id)`。不同 token 不得写同一个 cell；同 token 可以被多个消费者读取；Tap 保留继续传播，Consume 终止传播。东西两个方向分别建模，不能用“终端端口空闲”代替整条路径空闲。

`SchedulePlan` 以稳定 ID 和名称构建 task DAG；规划阶段检查重复名称、非法依赖与依赖环。`ResourceScheduler` 给 task 寻找 FU、MEM bank、weight buffer 和流路均合法的最早时间，`StreamFabricScheduler` 预留每一列每一拍的路径。生产者发射到消费者可见的固定 transport latency 来自 target，不等于经过的列数；CModel tick phase 也计入端点延迟。`VerifySchedule` 在生成 Command 之前复核资源和依赖。

每个 emitter 必须静态证明：SR 数据在 consumer 读取时已到达、端口单周期最多承载允许的 FU 操作、MEM 写回不会覆盖活跃输入、MXM accumulator 和 weight buffer 不被早复用、page 在首个 consumer 前有同步点、以及同一队列的粗指令 live interval 不重叠。若闭式域中间有空拍，该域的 context 仍被占用；不能把另一条同队列指令插进空拍。诊断应指出算子、物理队列、冲突区间和相关 binding。

## 8 ICU 的本地执行模型

每个物理 ICU 有自己的 iMEM、FIFO、PC、等待计数器和至多一个 active decoded context。整包 word 到齐且 header 合法后才激活；当前指令的最后一次 FU 发射结束前，不激活下一条。不同 ICU 可并行，同一个 MEM bank 的 `READ_3D`、`WRITE_3D`、`WRITE_TAP_3D`、`WRITE_READ_2D` 与 `MEM_WRITE_SYNC` 必须按同一个 PC 顺序执行。硬件不需要知道某条指令的“全局第几 cycle”。

MEM/MXM 的公共三维循环以 counter 0 为最内层：`issue_offset = wait_cycle + i0×S0 + i1×S1 + i2×S2`，`0≤id<count[d]`。`wait_cycle` 是包激活后的相对等待；各维 stride 同时描述循环内部及循环边界的间隔。公共编码中 wait 为 24 bit，每维 `count−1` 为 16 bit、cycle stride 为 24 bit；count 范围 1..65536，最后一次 offset 必须可编码。下一包启动前的长空档优先折入下一包 wait；超过字段宽度才插独立 NOP。Schedule/Command 中的绝对 cycle 只用来计算这些本地等待。

| 物理队列 | 当前主要 ICU 包 | 每条包的本地 iMEM 占用 | 循环层次 |
| --- | --- | --- | --- |
| MEM bank | `READ_3D`、`WRITE_3D`、`WRITE_TAP_3D`、`WRITE_READ_2D` | 3 × 96 bit | 3D 仿射或写读双 2D |
| MEM bank | `MEM_WRITE_SYNC` | 2 × 96 bit | tagged C2C vector burst |
| MXM load | `LOAD_3D` | 2 × 128 bit | 3D；归纳 weight column/buffer |
| MXM dequant | `DEQUANT_3D` | 2 × 128 bit | 3D；scale 固定 |
| MXM compute | `COMPUTE_3D`、`ACCUMULATOR_READ_3D` | 2 × 128 bit | 3D；归纳 accumulator，支持 terminal mode |
| VXM | `RUN_2D` 与 compact config | 3 × 96 bit | 外层 2D；config 自带连续 repeat |
| SXM | `RUN_2D` 与 tile-local config | 6 × 96 bit | 外层 2D；tile/lane map 留在 payload |
| C2C | DMA、RX、TX burst 与事件控制 | 按端点格式；RX 1 × 96 bit、DMA 2 × 96 bit | vector count 与完成事件 |

不能把所有 FU 强行套同一个 4D decoder。MEM/MXM 的三个硬件 counter 对应连续项、tile 和外层 group；VXM 的连续元素已在 compact config 内；SXM 的 lane map 是本地 payload；C2C 用 burst 和事件。出现第四个逻辑轴时，lowering 先尝试在字段范围内折叠，或只在 bank/page/FU 模板等真实边界切域。切域必须保持队列顺序和原 FU 时序。

## 9 MEM 读写包与 WRITE_READ_2D

普通 MEM 三维包包含固定 stream selector、bank-local base row、三个有符号地址 stride；blocked outer 模式再用 `outer_group_size` 与组间 stride 描述交替权重 buffer。地址为 `base + i0×a0 + i1×a1 + (i2 mod G)×a2 + floor(i2/G)×ag`；`G=1` 时就是普通三维仿射域。base row 为 13 bit，stream selector 6 bit，地址 stride 为有符号 20 bit。编译器必须验证所有迭代地址都在目标 bank 内。

`WRITE_READ_2D` 为同一 bank 的每个 `(i0,i1)` 坐标各发射一次 write 和一次 read，两者共享地址域但有独立时间域：`write = start_wait+i0×Sw0+i1×Sw1`；`read = start_wait+read_start_offset+i0×Sr0+i1×Sr1`；`addr = base+i0×a0+i1×a1`。write stream 固定，read stream 可按外层坐标改变。两维 count 为 16 bit，四个 cycle stride 与 read offset 为 24 bit，base row 13 bit，两个地址 stride 为有符号 20 bit，stream 字段各 6 bit。它仍是一条 3×96-bit ICU 指令；每 cycle 至多发一条 MEM FU 指令。

这个包只适用于写读迭代次数相同、地址公式相同且每个坐标恰好一次写一次读的区域。编译器必须证明时间不冲突、先写后读、读前未被覆盖、stream 数据按时到达；硬件不会通过 FIFO 空满自动重排。Q projection 的 MXM 结果写入与随后跨半球镜像读取是当前实例；不同写源、独立读写地址或多次读同一坐标应使用其他闭式包或分段域。

`MEM_WRITE_SYNC` 与普通 MEM 指令共用 bank ICU。它携带 vector count、16-bit sync tag、SR 传播延迟、起始 row/有符号 row stride 和 24-bit 预留窗口。指令到队首后等待匹配 token 与 SR 数据；提前写完仍占有预留窗口，迟到则等最后一次 SRAM commit。后续普通 read/write 不能越过；因此 runtime 不能通过“DDR 已完成”提前推进静态计算排程。

## 10 MXM VXM SXM 与算子 direct lowering

MXM 的 load、dequant、compute 分别有自己的 ICU 队列。Load 包指定 weight buffer、可归纳的 weight column、输入 stream 和数据格式；Dequant 包指定 BF16 scale；Compute 包指定 activation/result stream、accumulator base/stride、row stride、格式与 destination/clear/output mode。terminal mode 可在某个 loop 维度的最后一点切换输出/清零行为，让 reduction 末步仍在同一个闭式域内。`ACCUMULATOR_READ_3D` 从 compute ICU 独立读 accumulator，不能被误算成 MEM read。

VXM `RUN_2D` 的每个外层坐标 launch 一次 96-bit compact config，config 的 `repeat_count` 再驱动连续元素；SXM `RUN_2D` 每次 launch 一个含 transpose/permute、stream 列表和 lane map 的 tile-local config。算子 emitter 必须从 shape、placement 与闭式 timeline 直接生成这些包，不得先为每个 token/reduction 建 FU event 再扫描压缩。`ftlpu-compile` 的 direct 路径遇到旧 `CommandSequence` 会报错，防止静默退回兼容模式。

Attention 的 primitive 图包含 Q/K/V projection、Q/K RoPE、QK batch matmul、causal softmax、probability/value transpose、PV、output projection、residual 和 RMSNorm。FFN 包含 gate/up projection、Swish、multiply、down projection 与 residual。Tensor 层负责各 task 的 placement，Stream 层把真实数据路径挂在对应 task，Schedule emitter 负责 projection、RoPE、softmax、PV、FFN 等阶段的闭式域。硬件看到的始终是 MEM/MXM/VXM/SXM/C2C 本地指令。

Qwen2.5 seq32 的 projection/RoPE 默认开关为 `off`：先完成 projection，再集中做 RoPE；测试 pipeline 显式使用 `on`，按输出组交叠。`on` 模式把冲突的 activation read 从 primary slice 8/9 导向 pong slice 0/1，让 MXM 48 次 reduction 维保持连续；只在真正 bank/stream/FU 模板边界切域。`WRITE_READ_2D` 在 Q 输出镜像路径中把写读合成同一 context。FFN Up 的 direct-lowering 单元测试曾以 `M=32,K=1536,N=8960` 生成 100 条 MEM 包和每类 MXM 各 2 条包；这个数是特定 placement/target 的回归观察，不是 ISA 规定。

### 10.1 当前算子支持矩阵

下面区分三个层次：前端可识别为 Kernel primitive、可排成 Tensor/Stream/Schedule、以及已有 raw ICU image 与 CModel 数值闭环。“图内”表示只在相应 Attention/FFN matcher 的布局和时序上下文中已实现，不能理解为该 primitive 的任意独立调用都可执行。所有项仍须满足 target、静态 shape、元素类型和布局校验。

| 算子或语义 | 当前入口与限制 | 硬件映射及支持状态 |
| --- | --- | --- |
| rank-2 Matmul / Linear | 静态 M×K 与 K×N；普通矩阵乘、W8A16 projection | MEM + MXM，必要时 VXM dequant；独立 |
| Projection bias | Attention Q/K/V 的 rank-1 bias 广播 | Q/K 融入 RoPE VXM，V 经 VXM add/bypass；图内 |
| Reshape | 静态 shape，元素数和类型不变 | 逻辑 view；物理排布变化须显式搬运；图内结构语义 |
| Transpose / Permute | Attention 的 head、概率与 value 布局 | SXM + MEM/stream；图内结构语义 |
| RoPE | Q/K，偶数 head_dim、静态位置 | VXM 乘加 + MEM staging/写回；Attention 图内 |
| GQA head 展开 | query_heads 为 kv_heads 的整数倍 | Head 布局与复用计划；Attention 图内 |
| QK 与 PV batch matmul | rank-3，role 限定 qk 或 pv | MEM + MXM Vector compute + SXM；Attention 图内 |
| Scaled causal Softmax | Attention 行尾维、静态序列 | VXM 归约、exp、倒数、归一化 + MEM/SXM；图内 |
| RMSNorm | 静态 rank-2，末维 gamma、正 epsilon | MEM + SXM + VXM feedback/reduction；独立 |
| Elementwise Add | 同形 16-bit float rank-2，tile 对齐 | MEM + VXM + MEM；独立 |
| Elementwise Multiply | FFN 的 Swish×Up 等图形 | VXM 链及 MEM；仅 FFN 图内 |
| Swish / SiLU | Gate 激活或量化 SwiGLU 阶段 | VXM 多级链；仅 FFN 图内 |
| Gate/Up/Down FFN | Gate/Up 投影、Swish、乘法、Down 投影 | MXM + VXM + MEM，必要时 SXM；组合图闭环 |
| INT8 SwiGLU primitive | 静态 rank-2 INT8 输入、权重、输出及 scale | MXM + 量化 VXM；专用兼容路径 |

StableHLO 前端会识别标准 Attention/FFN/RMSNorm 子图、rank-2 dot_general 与同形 rank-2 add/multiply。Softmax、RoPE、GQA 等由匹配的 Attention 图建立 Kernel primitive，而不是逐条直译任意 StableHLO 序列。Elementwise Multiply 虽可进入 Kernel IR，当前独立 Schedule emitter 仅接受 tile-aligned 16-bit float Add；Multiply 主要由 FFN 专用图执行。Reshape/Transpose 也不是天然零成本：布局未变时可视为 view，生产和消费所需物理布局不同则须安排 SXM/MEM/stream 转换。

### 10.2 Matmul 与有权重 Projection

Matmul 计算 C[m,n] = Σk A[m,k]×W[k,n]。Tensor 层将 A、W、C 放到可组成 MXM tile 的 slice/layout；Stream 层安排激活和权重从 MEM 到 MXM、结果从 MXM 回 MEM。Schedule 按 K reduction、M tile、N output group 建闭式域：参与的权重 MEM 队列发 READ_3D，MXM load/dequant/compute 队列分别发 LOAD_3D、可选 DEQUANT_3D、COMPUTE_3D，结果由 MEM WRITE_3D 或 WRITE_TAP_3D 接收。MXM accumulator 在 reduction 中保留 FP32 部分和；最后一项用 terminal mode 输出和清零。独立 accumulator 读取使用 ACCUMULATOR_READ_3D，不是 MEM read。

当前通用 W8A16 linear projection 要求 M、K 对齐 MXM tile，N 对齐两个 tile；默认 tile 为 32，常见条件是 M%32=0、K%32=0、N%64=0。BF16 activation 与 INT8 weight 且 target 能力满足时，auto 策略原子选择 MXM 本地反量化与 Block8 compute：原始 INT8 weight stream 经 DEQUANT_3D 进入 MXM，每个 32 行输出块由四次 8 行 compute 完成。条件不满足则整体回退到 VXM dequant、Direct16 load、Vector compute，不能生成半套 Block8/半套 legacy 的混合。auto、legacy、block8 是编译策略，不会开启 target 缺少的物理能力。

一个 projection 在某个 MEM ICU 中能否合成一条 READ_3D，取决于地址、stream、FU 模板和相对时间能否构成合法闭式域。Qwen2.5 的 Q/K/V、O、Gate/Up 在合适队列上可跨 output group 归纳；同一 bank 若中途必须写结果、改 stream 或换模板，就只能在 lowering 时于该真实边界切域。不同 ICU 可流水重叠，同一 MEM bank 不能用两条活动粗指令交错隐藏数据搬运。

### 10.3 RMSNorm、Elementwise 与 FFN

RMSNorm 计算 y = x×rsqrt(mean(x²)+epsilon)×gamma。当前固定使用 VXM feedback：MEM 分布式读输入，必要时 SXM 转换/恢复布局，VXM 完成平方、FP32 累积归约、倒平方根和 gamma 乘法，再写回 MEM。独立路径要求静态 rank-2、最后一维 gamma 长度匹配、epsilon 为正；Qwen3 的 Q/K 逐 head norm 还依赖 Attention 图提供 head reshape 与 gamma placement。不能从这一实现推论任意 axis 的通用归约已经可执行。

独立 Elementwise Add 要求同形、tile 对齐的 16-bit float rank-2。两个 operand 经 MEM/SR 到 VXM，结果写回 MEM。Residual add 属于这一类，但源读与结果写若落在同一 bank 队列，emitter 必须先退出读域，再安排 VXM 流水和写域；不能让两个 ICU context 交错。FFN 的 Multiply 与 Swish 在专用 FFN 计划内绑定 Gate、Up、Down 的中间值和生命周期，不代表已有通用独立 Multiply 或 Swish Schedule 路径。

FFN 的语义是 Down(Swish(Gate(x))×Up(x))。Gate 和 Up 是两个共享输入的投影；MXM 结果进入 VXM 的 Swish/乘法链，中间 hidden 值写到 Down 可消费的布局，必要时经普通 stream 镜像到另一 hemisphere。Down 再读 hidden 和自身权重执行 projection。编译器须同时证明 Gate/Up 权重 buffer、MXM compute、VXM 链、hidden MEM 生命周期和 Down 读数的相对时间。Qwen FFN 通常是 BF16 activation + INT8 weight；独立 INT8 SwiGLU primitive 属特定量化兼容路径，不能与 BF16 图的输入输出精度契约混写。

### 10.4 Attention 组合图

当前 Attention 依赖链为 Q/K/V projection → 可选 Q/K bias 与逐 head RMSNorm → Q/K RoPE → GQA head 布局 → QK → scaled causal Softmax → PV → context 布局转换 → O projection。Kernel 层保留 primitive SSA 图，Tensor/Stream/Schedule 分别确定内存、路由和执行计划；最终没有 compound Attention ICU 指令。

Q/K/V 与 O 用有权重 MXM projection。Q/K bias 在 MXM 结果舍入 BF16 后融入 RoPE 的 VXM product FMA；V 若有 bias，则先经 VXM add/bypass，再写入 distributed-16 MEM。RoPE 按成对旋转维进行 sin/cos 乘加。显式启用 projection–RoPE overlap 时，每个 Q/K 输出组进入 staging 后尽快启动 RoPE，让下一组 MXM compute 与当前组 VXM/MEM 工作并行；冲突的 activation read 可转向 pong slice，避免 RoPE write 打断同一 bank 的 read。没有合法独立 bank/stream 窗口时必须保留空档或切域。

QK 与 PV 的 Kernel op 虽名为 batch_matmul，但 role 只接受 qk/pv；当前以激活乘激活的 MXM Vector compute、MEM 供数和 SXM 重排实现，不走 INT8 权重的 Block8 projection。GQA 要求 query_heads 是 kv_heads 的整数倍，软件按 head 组复用 K/V，计划 K 跨 hemisphere 的镜像和 V 的归属位置；硬件没有“复制 KV head”的模型级指令。

Softmax 对每个 score row 施加 1/sqrt(head_dim) 缩放与 causal mask，然后进行稳定的 max、exp、sum、reciprocal、normalize。VXM 的 FP32 局部 accumulator 保存归约，MEM 暂存 score/exp/mask/probability，SXM 将概率转换为 PV 可用布局；一次数学 Softmax 对应多条各 FU ICU 指令。PV context 经必要的 transpose/reshape 后成为 O projection activation，只有真实布局变化才生成 SXM/MEM 工作。

### 10.5 已验证配置和不支持范围

| 配置 | 已覆盖的设备路径 | 边界 |
| --- | --- | --- |
| Qwen2.5-1.5B seq32 | hidden 1536、FFN 8960、Q:KV head 12:2、head_dim 128；单层 prefill Attention/FFN、KV state 与 C2C 分页 | 完整单 token decode 未闭环 |
| Qwen3-0.6B seq32 | hidden 1024、FFN 3072、Q:KV head 16:8、head_dim 128；Q/K 逐 head RMSNorm 与完整单层 prefill | 不能把这一 fixture 推广为所有 Qwen3 变体 |
| SmolLM2-135M | Attention、FFN、decoder layer 与单独 LM-head shard 的编译/CModel 回归 | 模型包默认 embedding/LM head 仍为 host operation |

模型包默认顺序是 host embedding → 设备 decoder executable → host LM head。可单独编译的 LM-head shard 不会自动替换 ModelSession 的 host LM head。KV cache 已有逻辑 state、外存 backing、resident window、prefill K/V page-in/out 和数值测试；编译后的单 token executable 尚缺动态 position、历史前缀 QK/PV、append-only K/V 写入与移动页偏移。数学 reference 通过不能算作设备 decode 支持。以上矩阵是当前已验证的形状与路径，不是对所有序列长度、数据类型或模型变体的承诺。

## 11 `.ftlpu` 可执行程序格式与 loader

当前 writer 写 `.ftlpu` v32，magic 为 `FTLPUB01`，标量字段采用 little-endian。文件头携带 version、target name/ABI、完整 executable hardware config、`max_cycle` 和各类记录数量。后续依次保存 typed binding、timeline、逐 bank memory floor、weight page use、stream release cycle、物理 queue、scale/address relocation。Queue 记录 kind、物理 index、encoding mode 与命令数；raw 包按其本地 iMEM word 序列保存。文件容器可以有兼容旧描述符，但它们不是另一个硬件 ISA。

`BinaryWeightPageUse` 给出 binding/page/bank、首个 consumer 的逻辑 `ready_cycle` 和可释放的 `release_cycle`。`stream_release_cycles` 告知 runtime 哪些普通 SR 在何时可供 C2C lookahead 借用。`BinaryAddressRelocation` 指向 binding、queue、command 和读/写端，`BinaryScaleRelocation` 指向 VXM 立即数；修改包格式时必须同步更新 relocation、reader/writer、容量分析、trace、CModel codec 与 roundtrip 测试。

Reader 允许读取旧版本用于检查，不表示能部署执行。Loader 必须在写任何 iMEM 之前验证 magic/version、target ABI、队列种类与容量、packet header/长度、binding 范围、relocation 目标、单队列 context 与页同步。不能把不同 target 的二进制默默交给默认 CModel。`max_cycle` 是编译期逻辑时间/排空参考，不是一个写进各 ICU 的全局触发时刻。

## 12 `.ftlpum` 模型包与 ModelSession

当前模型包 writer 写 `.ftlpum` v6，magic 为 `FTLPUM01`。它保存模型名/架构、命名 tensor（raw、对称 INT8 或 target-packed SRAM vector）、命名外部值、一个或多个嵌入 executable、host embedding/LM-head operation、持久 state、weight page 和有序 invocation。可复用 executable 与每层不同的 tensor payload 分开，避免把同一类 ICU 程序按层完整展开。LazyExecutables 模式先读 metadata，dispatch 时才物化某个 executable。

`ModelSession::load()` 先验证 package、binding、shape/type、target ABI 与调用引用，调用 `SessionMemoryPlanner` 算出跨 invocation 的 resident/alias/transfer，再搬入常量并为每个持久 state 分配清零的逻辑片外 backing。`set_input(name, bytes)` 只接受声明为 external input 的值。`run()` 清理临时值映射，执行 host embedding，按 invocation 顺序装载/运行 executable，最后执行 host LM head，并只下载外部输出。`run_invocation(index)` 是单次设备调试入口，不自动替代整模型前后处理。

每次 invocation 会物化并有限 relocation executable，搬入本层 K/V resident window，重置并重新装载 ICU 队列但保留 MEM SRAM，解析输入是 Resident、HostUpload、DeviceAlias 或 DeviceCopy，执行静态程序与排空，再把更新的 K/V window 搬回片外逻辑 backing。`value(name)` 取模型输出，`read_state(name)` 读取逻辑 state，`reset_states()` 显式清零；普通 `run()` 不自动清空 KV。`stats()` 区分 resident、host I/O、state page、C2C 与 page-ready wait。

生产模型的外部输入、输出、权重和 state 必须经 DDR/C2C，不准从 host 直接修改 LPU MEM。低层 `CModelRuntime::upload_input()` 等 SRAM helper 可用于单元测试，但不能成为 ModelSession 设备边界的捷径。当前某些布局不一致的 DeviceCopy 仍由 CModel 内部操作完成；移植真实硬件时需替换成显式 ICU adapter executable。

## 13 C2C 权重分页与动态同步

离线 packer 先把量化后的逻辑权重重排成 target-packed SRAM row image；runtime 只搬运页，不在运行时重新排列完整大权重。每页的 segment 记录 tensor、byte offset、hemisphere、slice、bank、base row、vector count 和目标 stream。page 的物理 slot 可以在不同 bank 乒乓，也可以在同 bank 上复用不重叠的 slice/row 区间；但同 bank 的 C2C write 与计算 read 仍不能同 cycle 争端口。

一次外部输入的真实路径是 `host backing → DDR → C2C DMA → C2C RX → 普通 west SR → MEM Write → SRAM`，输出反向。默认 8 条 C2C lane 每方向每 cycle 最多 8×32=256 byte，但默认 DDR 规划持续值仅 92.16 byte/cycle，不能用 C2C 峰值推算稳定预取时间。一个连续 segment 只需一条 `C2C_DMA_BURST`、一条 `C2C_RX_BURST` 和目标 bank 一条 `MEM_WRITE_SYNC`；数量按 segment 而非 vector 增长。RX 发带 tag 的点对点 token，目标 MEM ICU 在 token 和 SR 数据同时就绪时才提交 SRAM write。

Compiler 在 page 的 `consumer_cycle` 前预留可行的 C2C/stream/MEM 窗口；runtime 在这些窗口中链接 DMA/RX 与同步写指令，并核对与 compiler MEM 指令计划一致。page-ready 的唯一判据是该页最后一个目标 SRAM write 已提交，不是 DDR/RX 完成。若首个 consumer 到达但页面未 ready，当前实现用计算侧全局 issue gate 暂停普通 ICU，同时继续 tick DDR/C2C/RX/同步写；page fence 完成后恢复。因此带宽不足造成真实等待，不造成静态程序错过 deadline。

`ModelSession` 的 `weight_page_initial_wait_cycles`、`weight_page_boundary_wait_cycles`、`weight_page_runtime_wait_cycles` 和 `weight_page_hidden_prefetches` 分别描述冷启动、层边界、executable 内同步及被计算隐藏的预取。更细粒度的 task 级等待尚需硬件 event ID/scoreboard 设计；目前不能假设每个 FU ICU 可以独立在任意流水位置等待不同 page。

## 14 KV cache 与 decode 当前边界

编译选项 `--kv-cache-capacity N` 让 decoder executable 声明逻辑缓存容量，N 不能小于本次编译的序列长度。每层 K/V 在模型包中分别保存 `[capacity, kv_heads, head_dim]` 的逻辑 state；`page_tokens` 与 `resident_tokens` 把片外逻辑容量和本次 executable 的 SRAM 窗口分开。当前固定 internal binding index：K 为 65536、role `state.kv.key`；V 为 65537、role `state.kv.value`。K 为 head-planar 并可跨半球复制，V 为 PV 消费布局；具体 slice/row 由 target 和 binding 决定。

Qwen2.5 seq32、capacity 256、2 KV head、head_dim 128 时，每层逻辑 K 与 V 各 128 KiB，单次 resident window 各 16 KiB。28 层逻辑缓存合计约 7 MiB，但顺序执行的层可以复用 SRAM staging slot。`ModelSession` 在每次调用前经 C2C page-in，之后 page-out；`read_state` 可验证 cache 内容。当前已测试 seq32 prefill 的 K/V 输出及 state reset。单 token decode 数学 reference 验证 GQA 12:2、KV append 和位置 32 的结果；编译后的 QK/PV 有效前缀、动态 token position、cache page offset 与实际 append command 尚待实现，所以不能把现有 prefill binary 当作 decode executable。

## 15 开发时如何生成与阅读指令

先用 `ftlpu_binary_inspect` 看目标 ABI、队列数、packet/iMEM slot、page use、live context 和展开后的 FU 工作量；它的 `--trace` 是从 binary 解码得到的离线计划，不是动态测量。再用 `ftlpu_icu_program_export` 生成 `index.csv` 和每个物理 ICU 的 `*.icu.csv`；每一行是一条 ICU 指令，包含 PC word、opcode、循环域、raw 96/128-bit 编码。MEM 的 read/write/sync 已合到同一个 `(hemisphere,slice,bank)` CSV；旧的 `_read`/`_write` 分离文件不代表当前硬件队列。

```powershell
build-dev/runtime/ftlpu_binary_inspect.exe `
  build-dev/qwen2_5_seq32/decoder_layer.ftlpu --all-queues `
  --trace build-dev/qwen2_5_seq32/plan.csv
build-dev/runtime/ftlpu_icu_program_export.exe `
  build-dev/qwen2_5_seq32/decoder_layer.ftlpu `
  build-dev/qwen2_5_seq32/icu_programs
```

浏览器打开 `tools/pipeline_viewer/index.html` 并载入 `plan.csv` 可检查计划中的 domain、资源和气泡；Viewer 仅按当前可见窗口展开 repeat。为了观察实际运行，使用 `ModelSession::enable_execution_trace()` 与 `write_execution_trace_csv()`，或在 Qwen runtime 测试中设置 `FTLPU_QWEN_C2C_PIPELINE_CSV`；动态 CSV 记录 physical cycle、DDR 抖动、C2C commit 和 `ICU.PageReadyWait`。要查某个 MEM bank 每周期状态，设置 `FTLPU_QWEN_MEM_CSV` 并在 `tools/pipeline_viewer/mem.html` 载入；该 CSV 是稀疏事件表，包含 ICU issue/wait、tile 0..3 流水和 SRAM read/write commit。不能用计划 CSV 断言“运行时没有 page wait”。

`FTLPU_QWEN_C2C_LINKED_BINARY` 可保存 runtime 插入传输指令后的真实 executable，再导出其 C2C DMA/RX 和 MEM 同步写队列；`FTLPU_QWEN_C2C_PRE_EXECUTION_DIR` 可保存启动前临时传输程序。主程序 linked image 不包含已在加载前完成的权重/输入传输，因此需要同时看两阶段，才能解释全部 ICU 指令和时间。

## 16 回归矩阵与验收口径

| 改动类型 | 最小回归 | 还需检查 |
| --- | --- | --- |
| 新 packet 或 bit 字段 | CModel codec roundtrip、非法字段/截断、runtime binary roundtrip | iMEM word 数、loader 拒绝路径、每 ICU CSV raw bits |
| 新算子/闭式 lowering | 对应 emitter 单测、Schedule verifier、FU 展开时序对照 | direct 标记、没有旧 Sequence 回退、单 context |
| placement/target 改动 | PhysicalMemoryAllocator、target 配置校验、binding relocation | 跨 invocation alias、row floor、ABI 不兼容拒绝 |
| stream 或延迟改动 | StreamFabricScheduler、route test、单算子 CModel | 每列每拍占用、producer-to-consumer latency |
| C2C/page 改动 | page planner、ping-pong、runtime page-ready sync、动态 trace | 最后 SRAM commit 才 ready、物理 wait 与统计一致 |
| Decoder 算子映射改动 | Qwen2.5/Qwen3 或 SmolLM2 对应单层 pipeline、数值与 KV state 测试 | Attention/FFN 分阶段误差、shape/type 拒绝、实际 C2C 重叠 |

测试中的 FU 展开只用于验证 `queue/cycle/opcode/operand` 与参考 schedule 等价；产品编译不能以这份展开表作为生成粗指令的输入。完整层输出按注明的 BF16 容差与 golden 比较，不能只看程序正常退出或非零个数。每次改动记录 target JSON、编译选项、binary 版本、模型 shape、输出误差、物理 cycle 与主要 queue 的 iMEM word，避免把不同选项的数字并排比较。

## 17 常见故障的定位顺序

| 现象 | 首先看 | 常见根因 |
| --- | --- | --- |
| Loader 拒绝 binary | magic/version、target name/ABI、queue/packet header | 用旧配置构建、packet 改了但 reader/CModel 未同步 |
| MEM 指令被切很多条 | 同 bank 的 write/read/sync live interval、stream/FU 模板 | 真实同队列交错、page/bank 边界或闭式地址无法统一 |
| MXM 有空泡 | execution CSV 的 activation/weight 到达和 buffer 切换 | MEM bank 冲突、SR 路径冲突、transport latency 未满足 |
| 页已 RX 完仍等待 | MEM_WRITE_SYNC 最后写提交与 page fence | RX 完成不等于 SRAM-ready；端口仍被计算占用 |
| 输出数值偏差 | 分阶段 golden、binding layout、dequant scale、terminal mode | 错误的 BF16/FP16 解释、lane 顺序或 accumulator 清零 |
| 计划快而运行慢 | 动态 execution CSV、DDR 供数与 `PageReadyWait` | 带宽、latency jitter、C2C 同 bank 串行化 |
| iMEM 超容量 | 每 ICU CSV 的 `pc_word` 与 inspector 的物理 slot | 把包数误当 word 数、长等待 NOP 或真实切域过多 |

## 18 改动规范与仍待冻结的接口

新增算子时，先在 Kernel 级确定数学语义和 supported shape；再定义 Tensor placement/binding、Stream route、Schedule 资源与闭式域；随后在 Command 层直接编码现有 FU 指令，最后用 CModel 数值和动态 trace 验收。若当前 ISA 无法表达所需行为，应先明确缺少的 ICU 本地状态/字段与单 context 约束，再设计 opcode；不能在 runtime 插入隐式 host 计算替代硬件算子。

改二进制或物理 target 时，至少同时修改 writer/reader、target ABI、compiler verifier、runtime relocation/容量分析、CModel decoder、CSV 导出与相应测试。兼容旧文件是独立的 reader/adapter 责任，不能为了接受旧数据放松当前 raw 硬件语义。对外报告的“ICU 指令数”“物理 iMEM word”“序列化文件字节数”“FU 展开发射次数”必须分别计数。

RTL/芯片接入前仍需冻结每类 packet 的全部 bit 与非法编码行为、iMEM 下载/启动/停止接口、page-ready/event tag 的硬件时序、错误中断与状态读取、真实 DDR/C2C DMA 描述符和吞吐限制，以及目标配置升级规则。当前 CModel 通过不等于这些接口已定版。后续跨 compiler、runtime、CModel、RTL 的一致性测试应以同一可执行程序和同一 target ABI 为输入。
