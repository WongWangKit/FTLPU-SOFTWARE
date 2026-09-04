# FTLPU Pipeline Viewer

The pipeline viewer is a dependency-free Canvas waveform workspace for two
kinds of CSV trace:

- `ModelSession::write_execution_trace_csv()` and `RuntimeExecutionTrace`
  sample instructions actually issued by the ICUs while CModel advances cycle
  by cycle. They include physical cycles, DDR jitter, and synchronization stalls.
- `write_schedule_trace_csv()` produces an offline plan from a binary without
  running CModel. It is useful for compiler inspection, but it is not evidence
  of actual execution timing.

Open `index.html` and load either format. Performance and pipeline validation
should use the runtime execution trace.

For a multi-invocation `ModelSession`, runtime segments are appended on one
session-wide physical-cycle axis. `Session.Invocation` marks each executable;
`C2C.ModelWeightPage`, `C2C.HostInput`, and `C2C.HostOutput` retain transfer
time outside an executable's local ICU cycle counter.

The original four-column CSV remains supported:

```csv
start,end,resource,detail
0,32,"MEM.E.Read","slice=0 addr=0 stream=E0"
32,64,"MXM.E0.Compute","Compute buffer=0 act=E0 out=W0"
```

New traces add compact pattern columns:

```csv
start,end,resource,detail,pattern,inner_count,inner_interval,inner_stride,outer_count,outer_interval,outer_stride,skip_first,induction,base_delta
0,1,"MEM.E.Read","slice=0 addr=0 stream=E0","repeat",128,1,1,1,0,0,0,"mem_address",0
```

`repeat` and `repeat2d` rows describe iteration spaces rather than expanded
events. The viewer expands only the instances intersecting the visible cycle
window. `induction` identifies the numeric field changed by the strides.

In an execution trace, `C2C.E.Prefetch`, `C2C.W.Prefetch`, shared-SR, and
`MEM.*.C2CWrite` intervals come from completed CModel DMA/RX/MEM-write work.
Their details carry `source=runtime`, `consumer_cycle`, and `actual_ready`.
When a page finishes late, `ICU.PageReadyWait` shows the physical-cycle interval
for which compute-side ICU issue was held at the synchronization barrier.
Offline-plan rows retain `planned=true` and are bandwidth-model predictions.

`C2C.E/W.DMA` are the DDR-to-C2C DMA command-issue rows for the east and west
hemispheres. `C2C.E/W.RX` are the matching receive command-issue rows that bind
each lane to a destination MEM slice/bank. The four rows are therefore two
hemispheres times two command stages. Use `C2C.E/W.Prefetch` for the completed
transfer interval and `MEM.E/W.C2CWrite` for the final SRAM-write interval.

Controls:

- Mouse wheel: zoom around the pointer.
- Drag: pan the visible cycle window.
- Shift + wheel: horizontal pan.
- Ctrl/Alt + wheel: vertical resource scrolling.
- A/B buttons followed by a click: place measurement cursors.
- Overview click: recenter the visible window.
- Overview drag or wheel: pan horizontally; hold Shift for fine control.
- Focus the overview and use Left/Right; hold Shift for smaller steps.
- Search and unit selector: filter C2C/MEM/MXM/VXM/SXM resources and details.

CSV files are decoded incrementally in a Web Worker, so the browser never
creates one string containing the entire trace. The viewer automatically
enables level-of-detail sampling in wide views and returns to exact per-event
drawing as you zoom in. The status strip reports the LOD draw count; hit testing
and event inspection expand compact patterns on demand.

Runtime tests already generate traces such as:

```text
build-ftlpu-vs2026/compiler/ftlpu_lower/
  smollm2_135m_ffn_seq128_pipeline/ffn.runtime.csv
  smollm2_attention_pipeline/attention.runtime.csv
  smollm2_decoder_layer_binary_runtime/decoder_layer.runtime.csv
  qwen2_5_1_5b_decoder_layer/decoder_layer.actual.runtime.csv
```

The viewer renders only visible rows and cycle intervals, so long schedules do
not require an ultra-wide SVG. Per-resource time indexes and a cached, bounded
overview keep full-layer traces with hundreds of thousands of events
interactive. Existing SVG/PNG renderers remain useful for documentation
snapshots.
