# FTLPU Pipeline Viewer

The pipeline viewer is a dependency-free Canvas waveform workspace for runtime
schedule traces. Open `index.html` in a browser and load the CSV emitted by
`write_schedule_trace_csv()`.

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

Paged binaries also include `C2C.E.Prefetch` and `C2C.W.Prefetch` rows. Their
intervals are derived from binary page readiness and the target's dedicated
C2C bandwidth; `detail` records the page, bank, bindings, bytes, lane count,
and `planned=true` provenance.

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
```

The viewer renders only visible rows and cycle intervals, so long schedules do
not require an ultra-wide SVG. Per-resource time indexes and a cached, bounded
overview keep full-layer traces with hundreds of thousands of events
interactive. Existing SVG/PNG renderers remain useful for documentation
snapshots.
