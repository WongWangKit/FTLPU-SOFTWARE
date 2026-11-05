# MoE 逐周期与指令视图

`moe_ffn_demo` 运行时采样真实 CModel 状态，输出 `instructions.cycles.csv`。
`tools/render_moe_cycles.py` 将其封装成可直接打开的 `cycles.html`；无需服务器。
原 `pipeline.html` 保留全程概览，工具栏可切换到逐周期视图。

封装页面同时输出 `moe_full_cycles.html`（与 `cycles.html` 内容相同，便于重新打开
避免旧预览缓存）。数据以字符串字典形式直接内嵌，通过普通 JavaScript 还原，
不依赖网络、`fetch` 或 `DecompressionStream`。顶部显示加载状态和脚本错误；
另附内嵌静态全程图，即使预览未执行脚本也可查看。若交互未启动，可把该 HTML
文件直接拖入 Edge / Chrome。封装测试会执行最终 HTML 的全部可执行脚本，
校验还原后的全部事件与原始 trace 一致；该检查不等同于实际浏览器截图验证。

## 查看方法

- 页面顶部展示从程序装载到加权回并的全流程，包含 Router、Top-K / PLAN、
  6 次专家输入 Gather、4 次专家权重落 SRAM，以及全部专家块的 Gate / Up /
  SwiGLU / Down。点击总览可直接定位物理周期，悬停可查看阶段起止区间。
- 输入物理 cycle、拖动滑块，或使用上一／下一周期、左右方向键、播放按钮。
- 点击波形选择周期；按功能单元筛选，或只显示指令发射。
- 使用 Router、Top-K、专家权重 DMA、专家 FFN、加权回并按钮跳转。
- 点击事件，查看完整编码字段、地址、操作数和比较输入。
- 右侧路由表仅显示该周期已经产生的分数、Top-K 候选、选择结果、计数、
  分段偏移及反向索引。空白表示结果尚未产生。

总览中的阶段跨度包含阶段内等待，不表示功能单元每周期都执行运算；下方窄窗口
按实际事件标记工作和等待。进入页面默认定位 Router 打分；程序装载、PLAN、
专家输入、权重 DMA、FFN 和回并均有独立跳转按钮。

## 开头的 DMA 是什么

该案例的 `[0,4993)` 区间是 `load_timed()` 触发的程序和描述符装载：
程序镜像在 DDR，经 ProgramLoader 的 DMA 及前端装载通路提交至各队列 iMEM
和相关配置存储，等待提交及前端就绪后才启动程序。它包含装载等待，不是整个
区间每周期都有 DDR 数据传输。共有程序模板在这里装载，不按专家重复加载程序。

原始 token 激活和 Router 权重由测试初始化到 SRAM，不计为这段 DMA。
随后设备选出专家，约从 cycle 6203 开始，才通过两侧 DMA、C2C RX 和 MEM
写入通路，把所需专家权重从 DDR 搬进 SRAM。专家输入 Gather 则读取已有 SRAM
激活并重新组织为专家输入块。冷启动程序装载与运行中的专家权重搬运分开分析。

## 记录范围与含义

| 单元 | 记录内容 |
| --- | --- |
| RCU | 原生指令发射、描述符等待、逐次比较交换、计数、前缀和、assignment、地址生成及回并控制步骤 |
| MXM | 指令发射、各 tile 的 load/compute 控制流水、实际 MAC 活动和列结果 |
| VXM | ICU 配置发射、回并控制器配置发射，各 tile、各 stage 的 lane 0 操作数接收、流水和结果 |
| DMA | ProgramLoader 和两侧 DMA 的命令、DDR 请求、返回、信用等待、在途及 FIFO 等待 |
| MEM | 指令发射、tile 流水、实际 SRAM/SR 读写提交，以及 RCU 共享控制口访问 |
| ICU | 程序存储及描述符写入提交；各执行队列的指令发射归到对应功能单元 |

实色表示发射、执行或提交，浅色表示等待、在途或控制装载；空白表示无已记录活动。
RCU 的一个 ISA 命令可以持续多个周期，内部步骤不会伪装成新的 ISA 指令。
有实际源 PC 的发射记录显示 PC，内部流水不推测源 PC。
MEM Gather 指令中的地址字段与间接读取的实际 SRAM 行地址分开显示，以提交记录为准。
VXM 使用 lane 0 作为广播指令的数据行为代表，不提供全部 lane 的数值追踪。
描述符等待显示当前模型已有的延迟，不能解释为新增的逐端口物理 RAM 模型。
这些记录代表 CModel 已建模行为，不是 RTL 门级活动，也不引入新的硬件或 ISA。
当前追踪器接在 MoE 示例中，并非自动为所有 runtime 程序开启。

## 示例和复现

构造案例为 5 token、4 专家、Top-2、H=F=32 的 SwiGLU MoE。
Router 经真实 MXM 计算；选择出的正 FP32 logits 直接作为权重，没有 softmax 或重归一化。
专家数据采用 BF16 存储及现有 VXM 运算精度。不是从完整模型提取的子图。

在软件仓库中运行：

```powershell
cmake --build build-program-load --config Release --target moe_ffn_demo
build-program-load/runtime/Release/moe_ffn_demo.exe <输出目录>
python tools/render_moe_demo.py <输出目录>
python tools/render_moe_cycles.py <输出目录>
```

先生成概览，再运行逐周期封装。后一脚本同时检查路由状态回放、事件数量及 UI
跳转逻辑。源码页面 `tools/pipeline_viewer/cycles.html` 也可手动打开 CSV / JSON。

当前案例为 29481 周期、79698 条记录；40 次 Top-K 比较、10 项选中结果、
80 条专家权重 DMA、784 个权重返回向量。开启采样前后周期不变，160 个 BF16
输出与 BF16/LUT 参考逐位一致。`cycle_trace_checks.json` 保存该次运行的检查数据。
