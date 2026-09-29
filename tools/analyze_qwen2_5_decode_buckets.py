#!/usr/bin/env python3
"""Plot Qwen2.5 decode performance across exact past-length buckets."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results" / "qwen2_5_decoder_layer_decode_buckets"
PAST_LENGTHS = (32, 64, 128, 224)
LAYERS = 28
VOCAB_SIZE = 151_936
HIDDEN_SIZE = 1_536
DDR_EFFICIENCY = 0.90
LM_HEAD_BYTES = VOCAB_SIZE * HIDDEN_SIZE
LAYER_WEIGHT_BYTES = 47_194_112


@dataclass(frozen=True)
class Point:
    past_len: int
    clock: int
    bandwidth: int
    compute_cycles: int
    initial_cycles: int
    runtime_cycles: int
    ingress_cycles: int
    egress_cycles: int
    state_in_cycles: int
    state_out_cycles: int

    def ms(self, cycles: int) -> float:
        return cycles / (self.clock * 1000.0)

    @property
    def program_ms(self) -> float:
        return self.ms(self.compute_cycles)

    @property
    def initial_ms(self) -> float:
        return self.ms(self.initial_cycles)

    @property
    def runtime_ms(self) -> float:
        return self.ms(self.runtime_cycles)

    @property
    def transport_ms(self) -> float:
        return self.ms(self.ingress_cycles + self.egress_cycles)

    @property
    def cold_layer_ms(self) -> float:
        return self.program_ms + self.initial_ms + self.runtime_ms + self.transport_ms

    @property
    def layer_weight_ms(self) -> float:
        rate = self.bandwidth * 1_000_000 * DDR_EFFICIENCY
        return LAYER_WEIGHT_BYTES / rate * 1000.0

    @property
    def lm_head_ms(self) -> float:
        rate = self.bandwidth * 1_000_000 * DDR_EFFICIENCY
        return LM_HEAD_BYTES / rate * 1000.0

    @property
    def token_pipelined_ms(self) -> float:
        scheduled = (
            self.transport_ms
            + self.initial_ms
            + LAYERS * (self.program_ms + self.runtime_ms)
        )
        ddr_floor = self.transport_ms + LAYERS * self.layer_weight_ms
        return max(scheduled, ddr_floor) + self.lm_head_ms

    @property
    def token_serial_ms(self) -> float:
        scheduled = self.transport_ms + LAYERS * (
            self.program_ms + self.initial_ms + self.runtime_ms
        )
        ddr_floor = self.transport_ms + LAYERS * self.layer_weight_ms
        return max(scheduled, ddr_floor) + self.lm_head_ms


def value(row: dict[str, str], name: str) -> int:
    text = row.get(name, "")
    return int(text) if text else 0


def read_points() -> list[Point]:
    points: list[Point] = []
    for past_len in PAST_LENGTHS:
        path = RESULT_ROOT / f"past_{past_len:04d}" / "sweep.csv"
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        passed = [row for row in rows if row["status"] == "passed"]
        if len(passed) != 12:
            raise RuntimeError(f"{path} has {len(passed)} passed rows, expected 12")
        for row in passed:
            points.append(
                Point(
                    past_len=past_len,
                    clock=value(row, "clock_mhz"),
                    bandwidth=value(row, "bandwidth_mbytes_per_second"),
                    compute_cycles=value(row, "compute_cycles"),
                    initial_cycles=value(row, "initial_wait_cycles"),
                    runtime_cycles=value(row, "runtime_wait_cycles"),
                    ingress_cycles=value(row, "c2c_ingress_cycles"),
                    egress_cycles=value(row, "c2c_egress_cycles"),
                    state_in_cycles=value(row, "state_page_in_cycles"),
                    state_out_cycles=value(row, "state_page_out_cycles"),
                )
            )
    return sorted(points, key=lambda p: (p.past_len, p.clock, p.bandwidth))


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.facecolor": "white",
            "figure.facecolor": "#f5f7fa",
            "axes.edgecolor": "#9aa5b1",
            "axes.grid": True,
            "grid.color": "#e5e9ef",
            "grid.linewidth": 0.8,
            "axes.titleweight": "semibold",
        }
    )


def save_sweep(points: list[Point]) -> None:
    colors = {500: "#377eb8", 750: "#ff7f00", 1000: "#4daf4a"}
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for ax, past_len in zip(axes.flat, PAST_LENGTHS):
        selected = [p for p in points if p.past_len == past_len]
        for clock in sorted({p.clock for p in selected}):
            series = sorted(
                (p for p in selected if p.clock == clock),
                key=lambda p: p.bandwidth,
            )
            ax.plot(
                [p.bandwidth / 1000 for p in series],
                [p.cold_layer_ms for p in series],
                marker="o",
                linewidth=2.1,
                color=colors[clock],
                label=f"{clock} MHz",
            )
        ax.set_title(f"past_len={past_len}, resident KV={past_len + 1}")
        ax.set_xlabel("DDR peak bandwidth (GB/s)")
        ax.set_ylabel("Measured cold latency (ms/layer)")
        ax.set_xticks([25.6, 51.2, 68.0, 85.6])
        ax.set_ylim(bottom=0)
        ax.legend(frameon=False)
    fig.suptitle("Qwen2.5-1.5B decode bucket performance", fontsize=17, weight="bold")
    for suffix in ("png", "svg"):
        fig.savefig(RESULT_ROOT / f"qwen2_5_decode_bucket_performance.{suffix}", dpi=180)
    plt.close(fig)


def save_scaling(points: list[Point]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    bandwidth_colors = {
        25_600: "#377eb8",
        51_200: "#ff7f00",
        68_000: "#984ea3",
        85_600: "#4daf4a",
    }
    for ax, clock in zip(axes[0], (500, 1000)):
        selected = [p for p in points if p.clock == clock]
        for bandwidth in sorted({p.bandwidth for p in selected}):
            series = sorted(
                (p for p in selected if p.bandwidth == bandwidth),
                key=lambda p: p.past_len,
            )
            ax.plot(
                [p.past_len for p in series],
                [p.cold_layer_ms for p in series],
                marker="o",
                linewidth=2.1,
                color=bandwidth_colors[bandwidth],
                label=f"{bandwidth / 1000:g} GB/s",
            )
        ax.set_title(f"Context scaling at {clock} MHz")
        ax.set_xlabel("Past KV tokens")
        ax.set_ylabel("Measured cold latency (ms/layer)")
        ax.set_xticks(PAST_LENGTHS)
        ax.set_ylim(bottom=0)
        ax.legend(frameon=False)

    ax = axes[1, 0]
    selected = sorted(
        (p for p in points if p.clock == 1000 and p.bandwidth == 85_600),
        key=lambda p: p.past_len,
    )
    x = [p.past_len for p in selected]
    bottoms = [0.0] * len(selected)
    components = (
        ("Program span", [p.program_ms for p in selected], "#4c78a8"),
        ("Initial preload", [p.initial_ms for p in selected], "#f58518"),
        ("Page-ready wait", [p.runtime_ms for p in selected], "#e45756"),
        ("C2C I/O", [p.transport_ms for p in selected], "#72b7b2"),
    )
    for label, values, color in components:
        ax.bar(x, values, bottom=bottoms, width=18, color=color, label=label)
        bottoms = [a + b for a, b in zip(bottoms, values)]
    ax.set_title("Latency breakdown at 1000 MHz / 85.6 GB/s")
    ax.set_xlabel("Past KV tokens")
    ax.set_ylabel("Measured cold latency (ms/layer)")
    ax.set_xticks(PAST_LENGTHS)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1, 1]
    for bandwidth in (25_600, 51_200, 68_000, 85_600):
        series = sorted(
            (
                p
                for p in points
                if p.clock == 1000 and p.bandwidth == bandwidth
            ),
            key=lambda p: p.past_len,
        )
        ax.plot(
            [p.past_len for p in series],
            [1000.0 / p.token_pipelined_ms for p in series],
            marker="o",
            linewidth=2.1,
            color=bandwidth_colors[bandwidth],
            label=f"{bandwidth / 1000:g} GB/s",
        )
    ax.set_title("Estimated 28-layer single-stream throughput at 1000 MHz")
    ax.set_xlabel("Past KV tokens")
    ax.set_ylabel("tokens/s")
    ax.set_xticks(PAST_LENGTHS)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    fig.suptitle("Qwen2.5-1.5B decode context-length scaling", fontsize=17, weight="bold")
    for suffix in ("png", "svg"):
        fig.savefig(RESULT_ROOT / f"qwen2_5_decode_bucket_scaling.{suffix}", dpi=180)
    plt.close(fig)


def write_combined_csv(points: list[Point]) -> Path:
    path = RESULT_ROOT / "bucket_sweep.csv"
    fields = [
        "past_len",
        "resident_kv_tokens",
        "clock_mhz",
        "bandwidth_mbytes_per_second",
        "compute_cycles",
        "initial_wait_cycles",
        "runtime_wait_cycles",
        "c2c_ingress_cycles",
        "c2c_egress_cycles",
        "state_page_in_cycles",
        "state_page_out_cycles",
        "cold_layer_ms",
        "estimated_pipelined_tokens_per_second",
        "estimated_serial_tokens_per_second",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for p in points:
            writer.writerow(
                {
                    "past_len": p.past_len,
                    "resident_kv_tokens": p.past_len + 1,
                    "clock_mhz": p.clock,
                    "bandwidth_mbytes_per_second": p.bandwidth,
                    "compute_cycles": p.compute_cycles,
                    "initial_wait_cycles": p.initial_cycles,
                    "runtime_wait_cycles": p.runtime_cycles,
                    "c2c_ingress_cycles": p.ingress_cycles,
                    "c2c_egress_cycles": p.egress_cycles,
                    "state_page_in_cycles": p.state_in_cycles,
                    "state_page_out_cycles": p.state_out_cycles,
                    "cold_layer_ms": f"{p.cold_layer_ms:.6f}",
                    "estimated_pipelined_tokens_per_second": f"{1000 / p.token_pipelined_ms:.3f}",
                    "estimated_serial_tokens_per_second": f"{1000 / p.token_serial_ms:.3f}",
                }
            )
    return path


def write_report(points: list[Point]) -> Path:
    canonical = sorted(
        (p for p in points if p.clock == 1000 and p.bandwidth == 85_600),
        key=lambda p: p.past_len,
    )
    rows = []
    for p in canonical:
        rows.append(
            f"| {p.past_len} | {p.past_len + 1} | {p.compute_cycles:,} | "
            f"{p.initial_cycles:,} | {p.runtime_cycles:,} | "
            f"{p.ingress_cycles + p.egress_cycles:,} | {p.cold_layer_ms:.4f} | "
            f"{1000 / p.token_serial_ms:.1f}–{1000 / p.token_pipelined_ms:.1f} |"
        )
    first, last = canonical[0], canonical[-1]
    growth = (last.cold_layer_ms / first.cold_layer_ms - 1) * 100
    report = f"""# Qwen2.5-1.5B decode 分桶性能

数据来自 4 个精确 past-length 桶、3 个 LPU 时钟和 4 个 DDR 带宽，共 48 次真实 ICU/CModel 单层 decode。所有运行都包含 DDR 权重加载、KV page-in/page-out、C2C 传输和更新后 KV 写回；性能 sweep 跳过数值门限，桶的独立数值测试另行保留。

![四桶 DDR/时钟 sweep](qwen2_5_decode_bucket_performance.svg)

![上下文长度缩放](qwen2_5_decode_bucket_scaling.svg)

## 1000 MHz / 85.6 GB/s 对比

| past_len | resident KV | program cycles | initial wait | runtime wait | C2C cycles | cold ms/layer | 估算 tokens/s（串行–跨层预取） |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

`cold ms/layer` 使用与之前性能图一致的口径：`(program cycles + initial wait + runtime page wait + C2C ingress + C2C egress) / clock`。从 past_len={first.past_len} 增长到 {last.past_len}，该配置下单层冷延迟变化 {growth:+.1f}%。

吞吐是按 28 个同构 decoder layer 加一次 W8 LM head 估算；区间左端逐层暴露 initial preload，右端假设后续层权重预取可与前一层重叠。它用于比较桶之间的趋势，不等同于完整模型实测。
"""
    path = RESULT_ROOT / "qwen2_5_decode_bucket_performance.md"
    path.write_text(report, encoding="utf-8")
    return path


def main() -> None:
    style()
    points = read_points()
    if len(points) != 48:
        raise RuntimeError(f"expected 48 points, got {len(points)}")
    save_sweep(points)
    save_scaling(points)
    outputs = [write_combined_csv(points), write_report(points)]
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
