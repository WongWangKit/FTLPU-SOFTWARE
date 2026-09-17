# FTLPU Pipeline Viewer

Pipeline Viewer 是一个无外部依赖的 Canvas 波形工作台，可以查看两类 CSV：

- `ModelSession::write_execution_trace_csv()` / `RuntimeExecutionTrace` 在 CModel
  逐 cycle 执行时采样 ICU 实际发射，记录真实 physical cycle、DDR 抖动和同步等待；
- `write_schedule_trace_csv()` 不运行 CModel，只从 binary 生成离线计划视图，适合
  检查编译结果，但不能作为实际执行时序的证据。

浏览器直接打开 `index.html` 后载入任意一种 CSV。性能分析和流水验证应优先使用
第一种实际运行 trace。

## MEM 每周期 Viewer

`mem.html` 是 MEM 专用的动态查看器。它按 `hemisphere / slice / bank /
read-write port` 展开物理端口，并在每一行内分别显示：

- MEM ICU 当周期的 issue、wait、同步等待、3D decode 和 program gate；
- 指令经过 tile 0 到 tile 3 的流水位置；
- SRAM 实际完成的 `read_to_sr` 或 `write_commit`，包括 row address、SR
  column、stream、vector tag 和 8-byte 数据。

运行前启用专用 trace，运行后写 CSV：

```cpp
CModelRuntime runtime(system);
runtime.enable_mem_execution_trace();
runtime.load(program);
runtime.run_cycles(cycles);
runtime.write_mem_execution_trace_csv("run.mem.csv");
```

`ModelSession` 提供同名的 `enable_mem_execution_trace()` 和
`write_mem_execution_trace_csv()`。Qwen2.5/Qwen3 decoder runtime 测试也可通过
环境变量直接输出：

```powershell
$env:FTLPU_QWEN_MEM_CSV = "decoder_layer.mem.csv"
```

专用 CSV 是稀疏事件表；没有行为的端口周期不写入文件。`mem.html` 会把空白自动
解释为空闲，并可用“显示空闲物理端口”展开完整的 2 hemisphere × 32 slice ×
2 bank × 2 port 视图。播放控制会逐周期列出当时所有 MEM 行为。

对于包含多个 invocation 的 `ModelSession`，runtime segment 会追加到同一条 session
物理 cycle 时间轴。`Session.Invocation` 标出每个 executable；
`C2C.ModelWeightPage`、`C2C.HostInput` 和 `C2C.HostOutput` 保留发生在 executable
局部 ICU cycle 之外的传输时间。

旧版四列输入格式继续兼容：

```csv
start,end,resource,detail
0,32,"MEM.E.Read","slice=0 addr=0 stream=E0"
32,64,"MXM.E0.Compute","Compute buffer=0 act=E0 out=W0"
```

新版 trace 增加结构化压缩字段：

```csv
start,end,resource,detail,pattern,inner_count,inner_interval,inner_stride,outer_count,outer_interval,outer_stride,skip_first,induction,base_delta
0,1,"MEM.E.Read","slice=0 addr=0 stream=E0","repeat",128,1,1,1,0,0,0,"mem_address",0
```

`repeat` 和 `repeat2d` 行描述迭代空间，不再提前展开所有 event。Viewer 只展开
与当前可见 cycle 窗口相交的实例；`induction` 指明 stride 修改的数值字段。

对于 raw FU 3D packet，离线 writer 为第三层 counter 的每个坐标输出一行，前两层
counter 继续保留为 `repeat` 或 `repeat2d` pattern。这样既完整表示硬件 launch，
又不需要恢复逐条细指令，CSV 大小也保持在可加载范围内。

实际运行 trace 中的 `C2C.E.Prefetch`、`C2C.W.Prefetch`、共享 SR 和
`MEM.*.C2CWrite` 区间来自 CModel 已完成的 DMA/RX/MEM write；`detail` 使用
`source=runtime`，并记录 `consumer_cycle` 与 `actual_ready`。当页面没有及时完成时，
`ICU.PageReadyWait` 显示计算侧 ICU 被同步屏障阻塞的真实 physical-cycle 区间。
离线计划 CSV 仍使用 `planned=true`，其区间只是目标带宽模型的预测。

`C2C.E/W.DMA` 是东西半球各自的 DDR-to-C2C DMA 命令发射行；
`C2C.E/W.RX` 是对应半球的 C2C 接收命令发射行，它把 lane 绑定到目标
MEM slice/bank。它们合计四行，是“两半球 x DMA/RX 两级”的 ICU 命令轨迹；
完整数据搬运持续时间看 `C2C.E/W.Prefetch`，最终 SRAM 写入看
`MEM.E/W.C2CWrite`。

操作：

- 鼠标滚轮：以指针所在 cycle 为中心缩放。
- 鼠标拖拽：水平移动可见时间窗口。
- Shift + 滚轮：水平移动。
- Ctrl/Alt + 滚轮：垂直滚动功能单元。
- 选择 A/B 后单击：放置测量游标。
- 单击 Overview：快速跳转到对应时间区域。
- 搜索框和功能单元菜单：筛选 C2C/MEM/MXM/VXM/SXM 资源与指令详情。

CSV 由 Web Worker 流式解码，浏览器不会再创建包含整个文件的单个字符串。
远景下查看器会自动启用 LOD，把每个可见资源行的绘制数量限制在合理范围；
放大后自动恢复逐 event 绘制。状态栏中的 `LOD` 数字表示当前实际绘制的 event
数量，点击命中和详情查询按需展开压缩 pattern。

现有 runtime 测试会生成可直接载入的 trace，例如：

```text
build-ftlpu-vs2026/compiler/ftlpu_lower/
  smollm2_135m_ffn_seq128_pipeline/ffn.runtime.csv
  smollm2_attention_pipeline/attention.runtime.csv
  smollm2_decoder_layer_binary_runtime/decoder_layer.runtime.csv
  qwen2_5_1_5b_decoder_layer/decoder_layer.actual.runtime.csv
```

查看器只绘制当前可见的资源行和 cycle 区间，因此长调度不再需要生成超宽
SVG。事件按资源建立时间索引，Overview 使用缓存的有界采样，因此几十万条
event 的整层 trace 仍可流畅缩放和拖动。原有 SVG/PNG 脚本继续用于文档中的
静态快照。
