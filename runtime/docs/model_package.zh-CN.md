# ModelPackage 与 ModelSession

`ModelPackage` 是位于 FTLPU command binary 之上的模型级容器。它让 decoder
layer 的指令可以复用，避免将所有层静态展开成一个巨大的 ICU 程序。

## 模型包内容

第一版 `.ftlpum` 格式包含：

- 模型名和架构标识；
- 命名常量 tensor；
- raw 或对称 INT8 量化元数据；
- 命名的外部输入、输出和中间值；
- 一个或多个内嵌 `.ftlpu` executable；
- 将命名 value 映射到 executable binding 的有序 invocation 列表。

文件 magic 为 `FTLPUM01`。量化 tensor 元数据包括 encoding、axis、block
size 和 scale。量化描述不决定物理 SRAM 布局，物理布局仍以内嵌
executable 的 `BinaryBinding` 为准。

## Session 执行

`ModelSession` 加载模型包、接收命名外部输入，并按顺序执行 invocation。
每个 invocation 会：

1. reset 并重新装载 CModel ICU 队列；
2. 按名称解析模型常量或前序计算结果；
3. 执行 planner 选择的 host upload、device alias 或 device layout copy；
4. 执行 command program 和 drain cycle；
5. 将中间输出保留在 MEM，只下载外部输出。

`SessionMemoryPlanner` 在执行前计算 producer/consumer lifetime。物理 binding
兼容时直接 alias；FP16 layout 不兼容时，在 CModel MEM 内完成 layout transfer，
不会在 host 中物化完整逻辑 tensor。该 transfer 是明确的 backend 操作，后续在
硬件上可以替换成 ICU MEM/SXM adapter executable。

每个 executable 边界会清空 ICU、stream、MXM、VXM 和 SXM 执行状态，但保留
MEM SRAM。因此每份 command binary 都从 cycle 0 状态开始，同时中间值继续驻留
在设备内。

## Executable 内部 binding

`access = internal` 的 executable binding 是 executable 自己拥有的物理存储，
不是由 `ModelSession` 提供的输入槽位。二进制格式 v7 为它增加了 typed
initializer 和初始化参数。目前 runtime 支持清零、分块 causal mask 生成和
FP16 RoPE cosine/sine 表生成。编译器给出表的 shape、物理 slice、`theta`
和 head dimension；`CModelRuntime::load` 会在 ICU 时钟启动前物化这些数据。

这样，算法常量由 compiler/runtime ABI 明确定义，同时不需要在 binary
中反复携带大块常量，也不再需要测试代码手工初始化 SRAM。

## 真实 SmolLM2 单层 Golden

`import_hf_decoder_layer.py` 不依赖 PyTorch，直接读取标准 Llama-compatible
`config.json` 和 safetensors checkpoint。它可以抽取任意 decoder layer，
将 Hugging Face Linear 权重转置为 StableHLO 使用的布局，执行对称
per-tensor INT8 量化，并生成 NumPy 参考结果。

SmolLM2-135M 第 0 层、`seq_len=128` 测试包包含：

- 编译后的可复用 decoder-layer executable；
- 真实第 0 层 Q/K/V/O 和 gate/up/down 权重；
- 两个真实 RMSNorm 权重；
- 从 embedding 得到的输入和量化 golden 输出。

`ModelSession -> ICU -> CModel` 的 73,728 个 FP16 输出全部完成比较，最大
绝对误差为 `0.03125`，平均绝对误差为 `0.000815428`。测试现在完全通过
typed internal binary binding 获得 RoPE 表。

## 真实双层 Golden

`seq_len=128` 的第 0/1 层测试分别编译 executable，因为两层的 W8A16 scale
不同。`hidden.1` 在 invocation 之间保留于 CModel MEM。采用 `vxm_feedback`
RMSNorm 策略的 decoder 以 `fp16_mxm_distributed_16` 作为持久化外部激活 ABI。
最终 residual add 直接写入下一次 decoder invocation 所需的相同 slices 和
base row，因此 planner 选择 device alias，不再执行 layout copy。

Session 共执行 19 次 host upload，用于模型入口和两层常量；只执行一次最终
host download，并执行一次 device alias、零次 device copy；`hidden.1` 不经过
host。最终 73,728 个 FP16 值全部与 NumPy 双层 golden 对比通过，最大绝对误差为
`0.0625`，平均绝对误差为 `0.00147592`。
