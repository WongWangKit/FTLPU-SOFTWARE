#!/usr/bin/env python3
"""Plot seq32 prefill versus past32 decode and four decode buckets."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
BUCKET_ROOT = RESULTS / "qwen2_5_decoder_layer_decode_buckets"
CLOCKS = (500, 750, 1000)
BANDWIDTHS = (25_600, 51_200, 68_000, 85_600)
PAST_LENGTHS = (32, 64, 128, 224)


@dataclass(frozen=True)
class Point:
    clock: int
    bandwidth: int
    latency_ms: float


def rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        selected = [row for row in csv.DictReader(handle) if row["status"] == "passed"]
    if len(selected) != 12:
        raise RuntimeError(f"{path} has {len(selected)} passed rows, expected 12")
    return selected


def prefill_points() -> list[Point]:
    result = []
    path = RESULTS / "qwen2_5_decoder_layer_prefill_sweep.csv"
    for row in rows(path):
        clock = int(row["clock_mhz"])
        total_cycles = (
            int(row["max_cycle"])
            + int(row["initial_wait_cycles"])
            + int(row["runtime_wait_cycles"])
        )
        result.append(Point(clock, int(row["bandwidth_mbytes_per_second"]),
                            total_cycles / (clock * 1000.0)))
    return result


def decode_points(past_len: int) -> list[Point]:
    path = BUCKET_ROOT / f"past_{past_len:04d}" / "sweep.csv"
    return [
        Point(int(row["clock_mhz"]), int(row["bandwidth_mbytes_per_second"]),
              float(row["cold_layer_ms"]))
        for row in rows(path)
    ]


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "figure.facecolor": "#f5f7fa",
        "axes.facecolor": "white",
        "axes.edgecolor": "#9aa5b1",
        "axes.grid": True,
        "grid.color": "#e5e9ef",
        "grid.linewidth": 0.8,
        "axes.titleweight": "semibold",
    })


def series(points: list[Point], clock: int) -> list[Point]:
    return sorted((point for point in points if point.clock == clock),
                  key=lambda point: point.bandwidth)


def save_prefill_decode32(prefill: list[Point], decode32: list[Point]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    all_latency = [point.latency_ms for point in prefill + decode32]
    ymax = max(all_latency) * 1.08
    workload_style = {
        "Prefill seq=32": ("#377eb8", "o", "-"),
        "Decode past=32": ("#e66101", "s", "--"),
    }
    for ax, clock in zip(axes.flat[:3], CLOCKS):
        for label, points in (("Prefill seq=32", prefill),
                              ("Decode past=32", decode32)):
            selected = series(points, clock)
            color, marker, line = workload_style[label]
            ax.plot(
                [point.bandwidth / 1000 for point in selected],
                [point.latency_ms for point in selected],
                label=label, color=color, marker=marker, linestyle=line,
                linewidth=2.3, markersize=6,
            )
        ax.set_title(f"{clock} MHz")
        ax.set_xlabel("DDR peak bandwidth (GB/s)")
        ax.set_ylabel("Cold latency (ms/layer)")
        ax.set_xticks([bandwidth / 1000 for bandwidth in BANDWIDTHS])
        ax.set_ylim(0, ymax)
        ax.legend(frameon=False)

    ax = axes.flat[3]
    colors = {500: "#377eb8", 750: "#ff7f00", 1000: "#4daf4a"}
    for clock in CLOCKS:
        prefill_by_bw = {p.bandwidth: p.latency_ms for p in series(prefill, clock)}
        selected = series(decode32, clock)
        ax.plot(
            [point.bandwidth / 1000 for point in selected],
            [prefill_by_bw[point.bandwidth] / point.latency_ms for point in selected],
            label=f"{clock} MHz", color=colors[clock], marker="o", linewidth=2.3,
        )
    ax.axhline(1.0, color="#6b7280", linewidth=1.2, linestyle=":")
    ax.set_title("Prefill / decode latency ratio")
    ax.set_xlabel("DDR peak bandwidth (GB/s)")
    ax.set_ylabel("Latency ratio (higher means decode is faster)")
    ax.set_xticks([bandwidth / 1000 for bandwidth in BANDWIDTHS])
    ax.legend(frameon=False)
    fig.suptitle(
        "Qwen2.5-1.5B: seq32 prefill vs past32 decode",
        fontsize=17, weight="bold",
    )
    for suffix in ("png", "svg"):
        fig.savefig(BUCKET_ROOT / f"qwen2_5_prefill_vs_decode32.{suffix}", dpi=180)
    plt.close(fig)


def save_four_buckets(bucket_points: dict[int, list[Point]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.7), sharey=True,
                             constrained_layout=True)
    colors = {
        25_600: "#377eb8",
        51_200: "#ff7f00",
        68_000: "#984ea3",
        85_600: "#4daf4a",
    }
    all_points = [point for points in bucket_points.values() for point in points]
    ymax = max(point.latency_ms for point in all_points) * 1.08
    for ax, clock in zip(axes, CLOCKS):
        for bandwidth in BANDWIDTHS:
            values = []
            for past_len in PAST_LENGTHS:
                match = next(
                    point for point in bucket_points[past_len]
                    if point.clock == clock and point.bandwidth == bandwidth
                )
                values.append(match.latency_ms)
            ax.plot(
                PAST_LENGTHS, values, marker="o", linewidth=2.3,
                color=colors[bandwidth], label=f"{bandwidth / 1000:g} GB/s",
            )
        ax.set_title(f"{clock} MHz")
        ax.set_xlabel("Past KV tokens")
        ax.set_xticks(PAST_LENGTHS)
        ax.set_ylim(0, ymax)
        ax.legend(frameon=False)
    axes[0].set_ylabel("Cold latency (ms/layer)")
    fig.suptitle(
        "Qwen2.5-1.5B decode: four exact context buckets",
        fontsize=17, weight="bold",
    )
    for suffix in ("png", "svg"):
        fig.savefig(BUCKET_ROOT / f"qwen2_5_decode_4bucket_comparison.{suffix}", dpi=180)
    plt.close(fig)


def main() -> None:
    configure_style()
    prefill = prefill_points()
    buckets = {past_len: decode_points(past_len) for past_len in PAST_LENGTHS}
    save_prefill_decode32(prefill, buckets[32])
    save_four_buckets(buckets)
    print(BUCKET_ROOT / "qwen2_5_prefill_vs_decode32.png")
    print(BUCKET_ROOT / "qwen2_5_decode_4bucket_comparison.png")


if __name__ == "__main__":
    main()
