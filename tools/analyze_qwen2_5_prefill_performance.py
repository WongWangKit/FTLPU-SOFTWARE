#!/usr/bin/env python3
"""Render the measured Qwen2.5 seq32 prefill sweep."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "qwen2_5_decoder_layer_prefill_sweep.csv"
OUT = ROOT / "results"
LAYERS = 28
PROMPT_TOKENS = 32
LAYER_WEIGHT_BYTES = 47_194_112
LM_HEAD_BYTES = 151_936 * 1_536
DDR_EFFICIENCY = 0.90


@dataclass(frozen=True)
class Point:
    clock_mhz: int
    bandwidth_mbps: int
    program_cycles: int
    initial_wait_cycles: int
    runtime_wait_cycles: int

    def ms(self, cycles: int) -> float:
        return cycles / (self.clock_mhz * 1000.0)

    @property
    def program_ms(self) -> float:
        return self.ms(self.program_cycles)

    @property
    def initial_wait_ms(self) -> float:
        return self.ms(self.initial_wait_cycles)

    @property
    def runtime_wait_ms(self) -> float:
        return self.ms(self.runtime_wait_cycles)

    @property
    def cold_layer_ms(self) -> float:
        return self.program_ms + self.initial_wait_ms + self.runtime_wait_ms

    @property
    def layer_weight_ms(self) -> float:
        return LAYER_WEIGHT_BYTES / (
            self.bandwidth_mbps * 1_000_000 * DDR_EFFICIENCY) * 1000.0

    @property
    def lm_head_ms(self) -> float:
        return LM_HEAD_BYTES / (
            self.bandwidth_mbps * 1_000_000 * DDR_EFFICIENCY) * 1000.0

    @property
    def ttft_overlap_ms(self) -> float:
        scheduled = self.initial_wait_ms + LAYERS * (
            self.program_ms + self.runtime_wait_ms)
        ddr_floor = LAYERS * self.layer_weight_ms
        return max(scheduled, ddr_floor) + self.lm_head_ms

    @property
    def ttft_serial_ms(self) -> float:
        serialized = LAYERS * self.cold_layer_ms
        ddr_floor = LAYERS * self.layer_weight_ms
        return max(serialized, ddr_floor) + self.lm_head_ms


def read_points() -> list[Point]:
    points: list[Point] = []
    with SOURCE.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["status"] != "passed":
                continue
            points.append(Point(
                int(row["clock_mhz"]),
                int(row["bandwidth_mbytes_per_second"]),
                int(row["max_cycle"]),
                int(row["initial_wait_cycles"]),
                int(row["runtime_wait_cycles"]),
            ))
    points.sort(key=lambda point: (point.clock_mhz, point.bandwidth_mbps))
    if len(points) != 12:
        raise RuntimeError(f"expected 12 passed points, got {len(points)}")
    return points


def write_metrics(points: list[Point]) -> Path:
    path = OUT / "qwen2_5_decoder_layer_prefill_performance.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "clock_mhz", "bandwidth_gbytes_per_second", "program_ms",
            "initial_wait_ms", "runtime_wait_ms", "cold_layer_ms",
            "ttft_overlap_ms", "ttft_serial_ms",
            "prompt_tokens_per_second_overlap",
        ])
        for point in points:
            writer.writerow([
                point.clock_mhz, point.bandwidth_mbps / 1000,
                f"{point.program_ms:.6f}", f"{point.initial_wait_ms:.6f}",
                f"{point.runtime_wait_ms:.6f}", f"{point.cold_layer_ms:.6f}",
                f"{point.ttft_overlap_ms:.6f}",
                f"{point.ttft_serial_ms:.6f}",
                f"{PROMPT_TOKENS / point.ttft_overlap_ms * 1000:.3f}",
            ])
    return path


def render(points: list[Point]) -> tuple[Path, Path]:
    colors = {500: "#2878B5", 750: "#F39C34", 1000: "#3A923A"}
    grouped = {
        clock: [point for point in points if point.clock_mhz == clock]
        for clock in sorted({point.clock_mhz for point in points})
    }
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    fig.suptitle("Qwen2.5-1.5B Seq32 Prefill Performance — One Decoder Layer",
                 fontsize=17, fontweight="bold")

    ax = axes[0, 0]
    for clock, values in grouped.items():
        ax.plot([p.bandwidth_mbps / 1000 for p in values],
                [p.cold_layer_ms for p in values], marker="o", linewidth=2.2,
                color=colors[clock], label=f"{clock} MHz")
    ax.set(title="Measured cold single-layer latency",
           xlabel="DDR peak bandwidth (GB/s)", ylabel="Latency (ms)")
    artifact = next(point for point in points
                    if point.clock_mhz == 1000
                    and point.bandwidth_mbps == 25_600)
    ax.annotate("32 B/vector CModel\ntransport artifact",
                xy=(25.6, artifact.cold_layer_ms), xytext=(35, 2.27),
                fontsize=8, color="#8C2D2D",
                arrowprops={"arrowstyle": "->", "color": "#8C2D2D"})
    ax.legend()

    ax = axes[0, 1]
    labels = [f"{p.clock_mhz}/{p.bandwidth_mbps / 1000:g}" for p in points]
    x = np.arange(len(points))
    program = np.array([p.program_ms for p in points])
    initial = np.array([p.initial_wait_ms for p in points])
    runtime = np.array([p.runtime_wait_ms for p in points])
    ax.bar(x, program, label="ICU program span", color="#4C78A8")
    ax.bar(x, initial, bottom=program, label="Initial page wait", color="#F58518")
    ax.bar(x, runtime, bottom=program + initial,
           label="Runtime page wait", color="#E45756")
    ax.set(title="Measured single-layer latency breakdown",
           xlabel="MHz / DDR GB/s", ylabel="Latency (ms)")
    ax.set_xticks(x, labels, rotation=45, ha="right")
    ax.legend(fontsize=9)

    ax = axes[1, 0]
    for clock, values in grouped.items():
        bandwidth = [p.bandwidth_mbps / 1000 for p in values]
        ax.plot(bandwidth, [p.ttft_overlap_ms for p in values], marker="o",
                linewidth=2.2, color=colors[clock], label=f"{clock} MHz overlap")
        ax.plot(bandwidth, [p.ttft_serial_ms for p in values], linestyle="--",
                linewidth=1.7, color=colors[clock], alpha=0.65,
                label=f"{clock} MHz serialized")
    ax.set(title="Estimated TTFT: 28 layers + W8 LM head",
           xlabel="DDR peak bandwidth (GB/s)", ylabel="TTFT (ms)")
    ax.legend(fontsize=8, ncol=2)

    ax = axes[1, 1]
    for clock, values in grouped.items():
        ax.plot([p.bandwidth_mbps / 1000 for p in values],
                [PROMPT_TOKENS / p.ttft_overlap_ms * 1000 for p in values],
                marker="o", linewidth=2.2, color=colors[clock],
                label=f"{clock} MHz")
    ax.set(title="Estimated accelerator-side prompt throughput",
           xlabel="DDR peak bandwidth (GB/s)", ylabel="Prompt tokens/s")
    ax.legend()

    for ax in axes.flat:
        ax.set_axisbelow(True)
        ax.grid(alpha=0.28)

    png = OUT / "qwen2_5_decoder_layer_prefill_performance.png"
    svg = OUT / "qwen2_5_decoder_layer_prefill_performance.svg"
    fig.savefig(png, dpi=180, facecolor="#F7F9FC")
    fig.savefig(svg, facecolor="#F7F9FC")
    plt.close(fig)
    return png, svg


def write_report(points: list[Point]) -> Path:
    best = min(points, key=lambda point: point.ttft_overlap_ms)
    rows = []
    for point in points:
        rows.append(
            f"| {point.clock_mhz} | {point.bandwidth_mbps / 1000:.1f} | "
            f"{point.cold_layer_ms:.3f} | {point.initial_wait_ms:.3f} | "
            f"{point.runtime_wait_ms:.3f} | {point.ttft_overlap_ms:.2f}–"
            f"{point.ttft_serial_ms:.2f} |")
    report = f"""# Qwen2.5-1.5B seq32 单层 prefill 性能

12 个点均来自当前代码重新编译后的真实 CModel 单层运行，并全部启用数值校验且通过。

![性能图](qwen2_5_decoder_layer_prefill_performance.png)

| MHz | DDR GB/s | 冷启动单层 ms | 初始等待 ms | 运行时等待 ms | 28 层 TTFT ms（重叠–串行） |
| ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

TTFT 按 28 个相同 decoder layer 和一次 `151936×1536` W8 LM head 估算。重叠端只暴露首层初始权重等待，并受 28 层总权重 DDR 流量下限约束；串行端逐层计入初始等待。DDR 有效带宽按峰值的 90% 计算，未计 tokenizer、host 调度和采样。

1000 MHz / 25.6 GB/s 的单层点包含 CModel 的 32 B C2C vector 离散化效应：此时 DDR 预算为 25.6 B/cycle，低于一个 vector；750 MHz 时为 34.13 B/cycle。逐页诊断显示 Gate 页的有效带宽由约 25.15 GB/s 降至 23.33 GB/s，因此该点不能解释为真实硬件上提高频率会降低性能。

本轮最优组合为 **{best.clock_mhz} MHz / {best.bandwidth_mbps / 1000:.1f} GB/s**：单层冷启动 **{best.cold_layer_ms:.3f} ms**，28 层 TTFT 估计 **{best.ttft_overlap_ms:.2f}–{best.ttft_serial_ms:.2f} ms**。
"""
    path = OUT / "qwen2_5_decoder_layer_prefill_performance.md"
    path.write_text(report, encoding="utf-8")
    return path


def main() -> None:
    points = read_points()
    for path in (*render(points), write_metrics(points), write_report(points)):
        print(path)


if __name__ == "__main__":
    main()
