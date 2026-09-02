# FTLPU Pipeline Viewer

Pipeline Viewer 是一个无外部依赖的 Canvas 波形工作台，可以查看两类 CSV：

- `ModelSession::write_execution_trace_csv()` / `RuntimeExecutionTrace` 在 CModel
  逐 cycle 执行时采样 ICU 实际发射，记录真实 physical cycle、DDR 抖动和同步等待；
- `write_schedule_trace_csv()` 不运行 CModel，只从 binary 生成离线计划视图，适合
  检查编译结果，但不能作为实际执行时序的证据。

浏览器直接打开 `index.html` 后载入任意一种 CSV。性能分析和流水验证应优先使用
第一种实际运行 trace。

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
