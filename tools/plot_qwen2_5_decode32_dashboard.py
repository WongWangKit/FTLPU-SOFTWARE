#!/usr/bin/env python3
"""Plot the four-panel Qwen2.5 past32 decode performance dashboard."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results" / "qwen2_5_decoder_layer_decode_buckets"
SWEEP = RESULT_ROOT / "past_0032" / "sweep.csv"
CLOCKS = (500, 750, 1000)
BANDWIDTHS = (25_600, 51_200, 68_000, 85_600)
LAYERS = 28
VOCAB_SIZE = 151_936
HIDDEN_SIZE = 1_536
LAYER_WEIGHT_BYTES = 47_194_112
DDR_EFFICIENCY = 0.90


@dataclass(frozen=True)
class Point:
    clock: int
    bandwidth: int
    program_cycles: int
    initial_cycles: int
    runtime_cycles: int
    ingress_cycles: int
    egress_cycles: int

    def ms(self, cycles: int) -> float:
        return cycles / (self.clock * 1000.0)

    @property
    def program_ms(self) -> float:
        return self.ms(self.program_cycles)

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
        return LAYER_WEIGHT_BYTES / (
            self.bandwidth * 1_000_000 * DDR_EFFICIENCY
        ) * 1000.0

    @property
    def lm_head_ms(self) -> float:
        return VOCAB_SIZE * HIDDEN_SIZE / (
            self.bandwidth * 1_000_000 * DDR_EFFICIENCY
        ) * 1000.0

    @property
    def token_overlap_ms(self) -> float:
        scheduled = (
            self.transport_ms + self.initial_ms
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


def read_points() -> list[Point]:
    with SWEEP.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["status"] == "passed"]
    if len(rows) != 12:
        raise RuntimeError(f"{SWEEP} has {len(rows)} passed rows, expected 12")
    return sorted(
        (
            Point(
                clock=int(row["clock_mhz"]),
                bandwidth=int(row["bandwidth_mbytes_per_second"]),
                program_cycles=int(row["compute_cycles"]),
                initial_cycles=int(row["initial_wait_cycles"]),
                runtime_cycles=int(row["runtime_wait_cycles"]),
                ingress_cycles=int(row["c2c_ingress_cycles"]),
                egress_cycles=int(row["c2c_egress_cycles"]),
            )
            for row in rows
        ),
        key=lambda point: (point.clock, point.bandwidth),
    )


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "figure.facecolor": "#f5f7fa",
        "axes.facecolor": "white",
        "axes.edgecolor": "#9aa5b1",
        "axes.grid": True,
        "grid.color": "#e5e9ef",
        "grid.linewidth": 0.8,
    })


def main() -> None:
    configure_style()
    points = read_points()
    colors = {500: "#2878b5", 750: "#f28e1c", 1000: "#2f8f35"}
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)

    ax = axes[0, 0]
    for clock in CLOCKS:
        selected = [point for point in points if point.clock == clock]
        ax.plot(
            [point.bandwidth / 1000 for point in selected],
            [point.cold_layer_ms for point in selected],
            marker="o", linewidth=2.4, color=colors[clock], label=f"{clock} MHz",
        )
    ax.set_title("Measured cold single-layer latency")
    ax.set_xlabel("DDR peak bandwidth (GB/s)")
    ax.set_ylabel("Latency (ms)")
    ax.set_xticks([bandwidth / 1000 for bandwidth in BANDWIDTHS])
    ax.legend(frameon=False)

    ax = axes[0, 1]
    labels = [f"{point.clock}/{point.bandwidth / 1000:g}" for point in points]
    x = list(range(len(points)))
    bottoms = [0.0] * len(points)
    components = (
        ("ICU program span", [point.program_ms for point in points], "#4c78a8"),
        ("Initial page wait", [point.initial_ms for point in points], "#f58518"),
        ("Runtime page wait", [point.runtime_ms for point in points], "#e45756"),
        ("C2C input/output", [point.transport_ms for point in points], "#72b7b2"),
    )
    for label, values, color in components:
        ax.bar(x, values, bottom=bottoms, color=color, label=label)
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]
    ax.set_title("Measured single-layer latency breakdown")
    ax.set_xlabel("MHz / DDR GB/s")
    ax.set_ylabel("Latency (ms)")
    ax.set_xticks(x, labels, rotation=48, ha="right")
    ax.legend(frameon=False)

    ax = axes[1, 0]
    for clock in CLOCKS:
        selected = [point for point in points if point.clock == clock]
        bandwidths = [point.bandwidth / 1000 for point in selected]
        ax.plot(
            bandwidths, [point.token_overlap_ms for point in selected],
            marker="o", linewidth=2.4, color=colors[clock],
            label=f"{clock} MHz overlap",
        )
        ax.plot(
            bandwidths, [point.token_serial_ms for point in selected],
            linewidth=1.8, linestyle="--", alpha=0.7, color=colors[clock],
            label=f"{clock} MHz serialized",
        )
    ax.set_title("Estimated decode token latency: 28 layers + W8 LM head")
    ax.set_xlabel("DDR peak bandwidth (GB/s)")
    ax.set_ylabel("Latency (ms/token)")
    ax.set_xticks([bandwidth / 1000 for bandwidth in BANDWIDTHS])
    ax.legend(frameon=False, ncol=2, fontsize=9)

    ax = axes[1, 1]
    for clock in CLOCKS:
        selected = [point for point in points if point.clock == clock]
        ax.plot(
            [point.bandwidth / 1000 for point in selected],
            [1000.0 / point.token_overlap_ms for point in selected],
            marker="o", linewidth=2.4, color=colors[clock], label=f"{clock} MHz",
        )
    ax.set_title("Estimated single-stream decode throughput")
    ax.set_xlabel("DDR peak bandwidth (GB/s)")
    ax.set_ylabel("Decode tokens/s")
    ax.set_xticks([bandwidth / 1000 for bandwidth in BANDWIDTHS])
    ax.legend(frameon=False)

    fig.suptitle(
        "Qwen2.5-1.5B Past32 Decode Performance — One Decoder Layer",
        fontsize=18, weight="bold",
    )
    for suffix in ("png", "svg"):
        fig.savefig(RESULT_ROOT / f"qwen2_5_decode32_performance.{suffix}", dpi=180)
    plt.close(fig)
    print(RESULT_ROOT / "qwen2_5_decode32_performance.png")


if __name__ == "__main__":
    main()
