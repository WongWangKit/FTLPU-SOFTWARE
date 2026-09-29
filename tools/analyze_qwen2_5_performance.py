#!/usr/bin/env python3
"""Plot Qwen2.5-1.5B prefill/decode sweep results and estimate model latency."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[1]
PREFILL_CSV = ROOT / "results" / "qwen2_5_decoder_layer_prefill_sweep.csv"
DECODE_CSV = ROOT / "results" / "qwen2_5_decoder_layer_decode" / "sweep.csv"
OUT = ROOT / "results" / "qwen2_5_decoder_layer_decode"

LAYERS = 28
PROMPT_TOKENS = 32
VOCAB_SIZE = 151_936
HIDDEN_SIZE = 1_536
DDR_EFFICIENCY = 0.90
LM_HEAD_BYTES = VOCAB_SIZE * HIDDEN_SIZE  # W8 [vocab, hidden].
LAYER_WEIGHT_BYTES = 47_194_112


@dataclass(frozen=True)
class Point:
    clock: int
    bandwidth: int
    schedule_cycles: int
    initial_cycles: int
    runtime_cycles: int
    ingress_cycles: int = 0
    egress_cycles: int = 0

    @property
    def schedule_ms(self) -> float:
        return self.schedule_cycles / (self.clock * 1000.0)

    @property
    def initial_ms(self) -> float:
        return self.initial_cycles / (self.clock * 1000.0)

    @property
    def runtime_ms(self) -> float:
        return self.runtime_cycles / (self.clock * 1000.0)

    @property
    def transport_ms(self) -> float:
        return (self.ingress_cycles + self.egress_cycles) / (self.clock * 1000.0)

    @property
    def cold_layer_ms(self) -> float:
        return self.schedule_ms + self.initial_ms + self.runtime_ms + self.transport_ms

    @property
    def lm_head_ms(self) -> float:
        effective_bytes_per_second = self.bandwidth * 1_000_000 * DDR_EFFICIENCY
        return LM_HEAD_BYTES / effective_bytes_per_second * 1000.0

    @property
    def layer_weight_ms(self) -> float:
        effective_bytes_per_second = self.bandwidth * 1_000_000 * DDR_EFFICIENCY
        return LAYER_WEIGHT_BYTES / effective_bytes_per_second * 1000.0

    @property
    def stack_pipelined_ms(self) -> float:
        # First layer exposes initial preload. Later-layer preload is assumed to
        # overlap the preceding layer; in-layer runtime stalls remain exposed.
        # Cross-layer overlap must still respect total DDR byte throughput.
        scheduled = (self.transport_ms + self.initial_ms +
                     LAYERS * (self.schedule_ms + self.runtime_ms))
        ddr_floor = LAYERS * self.layer_weight_ms
        return max(scheduled, self.transport_ms + ddr_floor)

    @property
    def stack_serial_ms(self) -> float:
        serialized = self.transport_ms + LAYERS * (
            self.schedule_ms + self.initial_ms + self.runtime_ms)
        return max(serialized, self.transport_ms + LAYERS * self.layer_weight_ms)


def read_points(path: Path, cycle_column: str) -> list[Point]:
    points: list[Point] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["status"] != "passed":
                continue
            points.append(
                Point(
                    clock=int(row["clock_mhz"]),
                    bandwidth=int(row["bandwidth_mbytes_per_second"]),
                    schedule_cycles=int(row[cycle_column]),
                    initial_cycles=int(row["initial_wait_cycles"]),
                    runtime_cycles=int(row["runtime_wait_cycles"]),
                    ingress_cycles=int(row.get("c2c_ingress_cycles") or 0),
                    egress_cycles=int(row.get("c2c_egress_cycles") or 0),
                )
            )
    return sorted(points, key=lambda p: (p.clock, p.bandwidth))


def by_clock(points: list[Point]) -> dict[int, list[Point]]:
    return {
        clock: sorted((p for p in points if p.clock == clock), key=lambda p: p.bandwidth)
        for clock in sorted({p.clock for p in points})
    }


def svg_line_panel(parts: list[str], box: tuple[int, int, int, int], title: str,
                   ylabel: str, series: list[tuple[str, list[float], list[float], str, bool]],
                   note: str = "") -> None:
    x0, y0, width, height = box
    left, right, top, bottom = 62, 18, 42, 52
    px0, py0 = x0 + left, y0 + top
    pw, ph = width - left - right, height - top - bottom
    all_x = [x for _, xs, _, _, _ in series for x in xs]
    all_y = [y for _, _, ys, _, _ in series for y in ys]
    xmin, xmax = min(all_x), max(all_x)
    ymin = 0.0
    ymax = max(all_y) * 1.12
    if ymax <= 0:
        ymax = 1.0

    sx = lambda v: px0 + (v - xmin) / (xmax - xmin) * pw
    sy = lambda v: py0 + ph - (v - ymin) / (ymax - ymin) * ph
    parts.append(f'<rect x="{x0}" y="{y0}" width="{width}" height="{height}" rx="7" fill="#ffffff" stroke="#d9dee7"/>')
    parts.append(f'<text x="{x0 + 14}" y="{y0 + 25}" class="title">{escape(title)}</text>')
    for i in range(6):
        value = ymin + (ymax - ymin) * i / 5
        yy = sy(value)
        parts.append(f'<line x1="{px0}" y1="{yy:.1f}" x2="{px0 + pw}" y2="{yy:.1f}" class="grid"/>')
        parts.append(f'<text x="{px0 - 8}" y="{yy + 4:.1f}" class="tick" text-anchor="end">{value:.1f}</text>')
    for value in sorted(set(all_x)):
        xx = sx(value)
        parts.append(f'<line x1="{xx:.1f}" y1="{py0}" x2="{xx:.1f}" y2="{py0 + ph}" class="grid"/>')
        parts.append(f'<text x="{xx:.1f}" y="{py0 + ph + 20}" class="tick" text-anchor="middle">{value:g}</text>')
    parts.append(f'<line x1="{px0}" y1="{py0 + ph}" x2="{px0 + pw}" y2="{py0 + ph}" class="axis"/>')
    parts.append(f'<line x1="{px0}" y1="{py0}" x2="{px0}" y2="{py0 + ph}" class="axis"/>')
    for label, xs, ys, color, dashed in series:
        coords = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(xs, ys))
        dash = ' stroke-dasharray="7 5" opacity="0.65"' if dashed else ""
        parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2.5"{dash}/>')
        if not dashed:
            for x, y in zip(xs, ys):
                parts.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="4" fill="{color}"/>')
    parts.append(f'<text x="{px0 + pw / 2:.1f}" y="{y0 + height - 10}" class="label" text-anchor="middle">DDR peak bandwidth (GB/s)</text>')
    parts.append(f'<text x="{x0 + 16}" y="{py0 + ph / 2:.1f}" class="label" text-anchor="middle" transform="rotate(-90 {x0 + 16} {py0 + ph / 2:.1f})">{escape(ylabel)}</text>')
    if note:
        parts.append(f'<text x="{px0 + 8}" y="{py0 + 16}" class="note">{escape(note)}</text>')
    legend_x = px0 + pw - 100
    legend_y = py0 + 14
    shown = set()
    for label, _, _, color, dashed in series:
        if dashed or label in shown:
            continue
        shown.add(label)
        parts.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 22}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{legend_x + 28}" y="{legend_y + 4}" class="legend">{escape(label)}</text>')
        legend_y += 18


def save_overview(prefill: list[Point], decode: list[Point]) -> Path:
    colors = {500: "#377eb8", 750: "#ff7f00", 1000: "#4daf4a"}
    pre_layer, dec_layer, ttft, toks = [], [], [], []
    for clock, points in by_clock(prefill).items():
        xs = [p.bandwidth / 1000 for p in points]
        pre_layer.append((f"{clock} MHz", xs, [p.cold_layer_ms for p in points], colors[clock], False))
        ttft.append((f"{clock} MHz", xs, [p.stack_pipelined_ms + p.lm_head_ms for p in points], colors[clock], False))
        ttft.append((f"{clock} MHz serial", xs, [p.stack_serial_ms + p.lm_head_ms for p in points], colors[clock], True))
    for clock, points in by_clock(decode).items():
        xs = [p.bandwidth / 1000 for p in points]
        dec_layer.append((f"{clock} MHz", xs, [p.cold_layer_ms for p in points], colors[clock], False))
        pipe = [p.stack_pipelined_ms + p.lm_head_ms for p in points]
        serial = [p.stack_serial_ms + p.lm_head_ms for p in points]
        toks.append((f"{clock} MHz", xs, [1000 / v for v in pipe], colors[clock], False))
        toks.append((f"{clock} MHz serial", xs, [1000 / v for v in serial], colors[clock], True))

    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="900" viewBox="0 0 1400 900">',
             '<style>text{font-family:Segoe UI,Arial,sans-serif;fill:#263238}.title{font-size:17px;font-weight:600}.tick,.legend{font-size:11px}.label{font-size:12px}.note{font-size:10px;fill:#666}.grid{stroke:#e7ebf0;stroke-width:1}.axis{stroke:#76818b;stroke-width:1.2}</style>',
             '<rect width="1400" height="900" fill="#f5f7fa"/>']
    svg_line_panel(parts, (20, 20, 670, 410), "Prefill seq32: measured cold layer latency", "Latency (ms/layer)", pre_layer)
    svg_line_panel(parts, (710, 20, 670, 410), "Decode token: measured cold layer latency", "Latency (ms/layer)", dec_layer)
    svg_line_panel(parts, (20, 450, 670, 410), "Estimated TTFT (28 layers + W8 LM head)", "TTFT (ms)", ttft, "solid: cross-layer overlap; dashed: serialized")
    svg_line_panel(parts, (710, 450, 670, 410), "Estimated single-stream decode throughput", "tokens/s", toks, "solid: cross-layer overlap; dashed: serialized")
    parts.append('</svg>')
    path = OUT / "qwen2_5_performance_overview.svg"
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def svg_bar_panel(parts: list[str], box: tuple[int, int, int, int], title: str,
                  points: list[Point]) -> None:
    x0, y0, width, height = box
    left, right, top, bottom = 62, 18, 45, 65
    px0, py0 = x0 + left, y0 + top
    pw, ph = width - left - right, height - top - bottom
    ymax = max(p.cold_layer_ms for p in points) * 1.12
    sy = lambda v: py0 + ph - v / ymax * ph
    parts.append(f'<rect x="{x0}" y="{y0}" width="{width}" height="{height}" rx="7" fill="#fff" stroke="#d9dee7"/>')
    parts.append(f'<text x="{x0 + 14}" y="{y0 + 27}" class="title">{escape(title)}</text>')
    for i in range(6):
        value = ymax * i / 5
        yy = sy(value)
        parts.append(f'<line x1="{px0}" y1="{yy:.1f}" x2="{px0 + pw}" y2="{yy:.1f}" class="grid"/>')
        parts.append(f'<text x="{px0 - 8}" y="{yy + 4:.1f}" class="tick" text-anchor="end">{value:.1f}</text>')
    step = pw / len(points)
    barw = step * 0.62
    colors = ("#4c78a8", "#f58518", "#e45756", "#72b7b2")
    for i, p in enumerate(points):
        xx = px0 + step * i + (step - barw) / 2
        bottom = 0.0
        for value, color in zip(
                (p.schedule_ms, p.initial_ms, p.runtime_ms, p.transport_ms), colors):
            yy = sy(bottom + value)
            hh = sy(bottom) - yy
            parts.append(f'<rect x="{xx:.1f}" y="{yy:.1f}" width="{barw:.1f}" height="{max(hh, 0):.1f}" fill="{color}"/>')
            bottom += value
        label = f"{p.clock}/{p.bandwidth / 1000:g}"
        tx, ty = xx + barw / 2, py0 + ph + 13
        parts.append(f'<text x="{tx:.1f}" y="{ty:.1f}" class="tick" text-anchor="end" transform="rotate(-40 {tx:.1f} {ty:.1f})">{label}</text>')
    legend = (("Program span", colors[0]),
              ("Initial preload", colors[1]),
              ("Page-ready wait", colors[2]),
              ("C2C I/O", colors[3]))
    lx = px0 + pw - 570
    for label, color in legend:
        parts.append(f'<rect x="{lx}" y="{py0 + 5}" width="12" height="12" fill="{color}"/>')
        parts.append(f'<text x="{lx + 17}" y="{py0 + 15}" class="legend">{label}</text>')
        lx += 40 + len(label) * 5.8
    parts.append(f'<text x="{x0 + 16}" y="{py0 + ph / 2:.1f}" class="label" text-anchor="middle" transform="rotate(-90 {x0 + 16} {py0 + ph / 2:.1f})">ms/layer</text>')


def save_breakdown(prefill: list[Point], decode: list[Point]) -> Path:
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="900" viewBox="0 0 1400 900">',
             '<style>text{font-family:Segoe UI,Arial,sans-serif;fill:#263238}.title{font-size:18px;font-weight:600}.tick,.legend{font-size:11px}.label{font-size:12px}.grid{stroke:#e7ebf0;stroke-width:1}</style>',
             '<rect width="1400" height="900" fill="#f5f7fa"/>']
    svg_bar_panel(parts, (20, 20, 1360, 410), "Prefill seq32 single-layer latency breakdown", prefill)
    svg_bar_panel(parts, (20, 450, 1360, 410), "Decode single-layer latency breakdown", decode)
    parts.append('</svg>')
    path = OUT / "qwen2_5_latency_breakdown.svg"
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def pct_improvement(old: float, new: float) -> float:
    return (old - new) / old * 100.0


def point(points: list[Point], clock: int, bandwidth: int) -> Point:
    return next(p for p in points if p.clock == clock and p.bandwidth == bandwidth)


def write_report(prefill: list[Point], decode: list[Point]) -> Path:
    best_prefill = min(prefill, key=lambda p: p.stack_pipelined_ms + p.lm_head_ms)
    best_decode = min(decode, key=lambda p: p.stack_pipelined_ms + p.lm_head_ms)
    p500_256 = point(prefill, 500, 25_600)
    p1000_256 = point(prefill, 1000, 25_600)
    p500_856 = point(prefill, 500, 85_600)
    p1000_856 = point(prefill, 1000, 85_600)
    d500_256 = point(decode, 500, 25_600)
    d1000_256 = point(decode, 1000, 25_600)
    d500_856 = point(decode, 500, 85_600)
    d1000_856 = point(decode, 1000, 85_600)

    rows = []
    decode_map = {(p.clock, p.bandwidth): p for p in decode}
    for p in prefill:
        d = decode_map[(p.clock, p.bandwidth)]
        ttft = p.stack_pipelined_ms + p.lm_head_ms
        ttft_upper = p.stack_serial_ms + p.lm_head_ms
        decode_ms = d.stack_pipelined_ms + d.lm_head_ms
        decode_upper = d.stack_serial_ms + d.lm_head_ms
        rows.append(
            f"| {p.clock} | {p.bandwidth / 1000:.1f} | {p.cold_layer_ms:.3f} | "
            f"{ttft:.2f}–{ttft_upper:.2f} | {d.cold_layer_ms:.3f} | "
            f"{decode_ms:.2f}–{decode_upper:.2f} | "
            f"{1000 / decode_upper:.1f}–{1000 / decode_ms:.1f} |"
        )

    best_ttft = best_prefill.stack_pipelined_ms + best_prefill.lm_head_ms
    best_ttft_upper = best_prefill.stack_serial_ms + best_prefill.lm_head_ms
    best_decode_ms = best_decode.stack_pipelined_ms + best_decode.lm_head_ms
    best_decode_upper = best_decode.stack_serial_ms + best_decode.lm_head_ms
    head_mib = LM_HEAD_BYTES / (1024 * 1024)

    report = f"""# Qwen2.5-1.5B prefill/decode 性能分析

数据源为最新的 12 组 prefill 和 12 组 decode ICU/CModel sweep。所有组合均完成执行；本轮按用户要求未用数值误差门限判定结果。

![性能总览](qwen2_5_performance_overview.svg)

![单层延迟分解](qwen2_5_latency_breakdown.svg)

## 估算口径

- 模型按 28 个相同 decoder layer 估算；prefill prompt 长度为 32，decode 为单 token、逻辑输入输出 `1×1536`、MXM native4。
- 单层冷启动延迟为 `(schedule cycles + initial wait + runtime page wait + C2C ingress + C2C egress) / clock`。其中 schedule cycles 是静态 ICU program span，包含片上资源和数据依赖造成的固定间隔，但不再包含按 DDR 带宽估算的等待。decode 日志已导出 C2C 首尾周期；当前 prefill 日志尚未导出这两项，因此 prefill 单层值暂未计外部 C2C I/O。
- 表中低延迟端假设层间权重预加载能与上一层执行重叠，因此调度式只暴露首层 initial wait，runtime page wait 仍逐层计入；模型输入输出的 C2C ingress/egress 只在 28 层堆叠的首尾各计一次，同时取 28 层总权重流量的 DDR 时间作为物理下限。
- 高延迟端把每层 initial wait 都串行计入，是没有跨层预取隐藏时的保守上界。
- LM head 按 Qwen2.5-1.5B 的 `151936×1536` W8 权重估算，权重流量 {head_mib:.1f} MiB，使用目标配置中的 90% DDR 调度效率。
- TTFT 包含 28 层 prefill 和一次 LM head；decode tokens/s 包含每个 token 的 28 层和一次 LM head。未计 tokenizer、host 调度、采样、完整多层模型控制开销。

## 结果

| MHz | DDR GB/s | prefill 单层 ms | TTFT ms（流水–串行） | decode 单层 ms | 每 token ms（流水–串行） | tokens/s（串行–流水） |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

prefill 最优点是 {best_prefill.clock} MHz / {best_prefill.bandwidth / 1000:.1f} GB/s：TTFT 估计 **{best_ttft:.2f}–{best_ttft_upper:.2f} ms**，其中前者是层间预取充分重叠的目标值，后者是逐层冷启动的保守上界；对应 32-token prompt 的 accelerator 侧处理率约 **{PROMPT_TOKENS / best_ttft * 1000:.0f} tokens/s**。decode 最优点是 {best_decode.clock} MHz / {best_decode.bandwidth / 1000:.1f} GB/s：估计 **{best_decode_ms:.2f}–{best_decode_upper:.2f} ms/token**，即 **{1000 / best_decode_upper:.1f}–{1000 / best_decode_ms:.1f} tokens/s**；吞吐区间前者是保守值，后者是预取重叠目标值。

## 性能瓶颈

1. **Decode 的主瓶颈是 DDR 权重带宽。** 在 25.6 GB/s 下，把频率从 500 MHz 提到 1000 MHz，单层只改善 {pct_improvement(d500_256.cold_layer_ms, d1000_256.cold_layer_ms):.1f}%（{d500_256.cold_layer_ms:.3f} → {d1000_256.cold_layer_ms:.3f} ms）；85.6 GB/s 下反而增加 {(d1000_856.cold_layer_ms / d500_856.cold_layer_ms - 1) * 100:.1f}%（{d500_856.cold_layer_ms:.3f} → {d1000_856.cold_layer_ms:.3f} ms），原因是更早到达 `SYNC_PAGE` 后暴露了更多动态等待。单 token 的矩阵 M=1，权重复用很低，47,194,112 B 的单层权重流和 {head_mib:.1f} MiB LM head 决定吞吐上限。

2. **Prefill 是计算与带宽混合受限。** 85.6 GB/s 下，500 → 1000 MHz 让单层改善 {pct_improvement(p500_856.cold_layer_ms, p1000_856.cold_layer_ms):.1f}%（{p500_856.cold_layer_ms:.3f} → {p1000_856.cold_layer_ms:.3f} ms），说明 seq32 仍有可随频率缩短的 MXM/VXM 工作；25.6 GB/s 下只改善 {pct_improvement(p500_256.cold_layer_ms, p1000_256.cold_layer_ms):.1f}%，且 1 GHz 出现 {p1000_256.runtime_ms:.3f} ms runtime page wait，已经被分页供数压住。

3. **LM head 是整模型不可忽略的固定带宽成本。** 它每次 TTFT/每个 decode token 需要读取约 {head_mib:.1f} MiB W8 权重；在 25.6/51.2/68.0/85.6 GB/s 下分别约为 {Point(1, 25600, 0, 0, 0).lm_head_ms:.2f}/{Point(1, 51200, 0, 0, 0).lm_head_ms:.2f}/{Point(1, 68000, 0, 0, 0).lm_head_ms:.2f}/{Point(1, 85600, 0, 0, 0).lm_head_ms:.2f} ms。若 LM head 最终不是 W8 或不能达到 90% DDR 效率，应按实际字节数和效率同比修正。

4. **次要瓶颈是分页供数造成的 runtime wait。** 编译器不再根据 DDR 带宽插入长等待；当某个权重页尚未完成 DDR→C2C→MEM 传输时，消费者会在 `SYNC_PAGE` 上动态停顿。后续优化应优先移动 page-in 起点、增加可用驻留空间或调整 bank 交替，缩短实际 page-ready wait。

   图中的 dynamic page-ready wait 是各页在 `SYNC_PAGE` 上造成的实际全局延迟。Pipeline Viewer 中没有 FU 事件的空白还可能包括片上固定 NOP 和该功能单元在其他算子阶段未被使用的时间；这些时间已包含在 program span 中，不能再次累加。

5. **KV/C2C 目前不是主要瓶颈。** decode 日志中的 KV page-in/page-out 各 98,304 B，C2C ingress/egress 各 104,448 B；在 past length 32 下远小于单层权重流。上下文增长后 attention 和 KV 流量会线性增加，届时需要重新 sweep，不能直接沿用本表 tokens/s。

## 建议

- Decode 优先提高有效 DDR 带宽、压缩/常驻权重、减少 LM head 流量；单纯继续提高 LPU 时钟收益很小。
- Prefill 在 68–85.6 GB/s 区间继续提高频率仍有效；在 25.6 GB/s 下先消除 runtime page wait。
- 实现完整 28 层运行后，用真实跨层预取测量值替换“流水–串行”区间；当前低延迟端是调度目标估计，高延迟端是逐层冷启动保守估计。
"""
    path = OUT / "qwen2_5_performance_analysis.md"
    path.write_text(report, encoding="utf-8")
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prefill = read_points(PREFILL_CSV, "max_cycle")
    decode = read_points(DECODE_CSV, "compute_cycles")
    if len(prefill) != 12 or len(decode) != 12:
        raise RuntimeError(f"expected 12+12 passed rows, got {len(prefill)}+{len(decode)}")
    outputs = [save_overview(prefill, decode), save_breakdown(prefill, decode),
               write_report(prefill, decode)]
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
