#!/usr/bin/env python3
"""Render FFN weight-page compute and planned C2C prefetch windows."""

from __future__ import annotations

import argparse
import html
import math
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Binding:
    index: int
    name: str
    byte_size: int
    shape: tuple[int, ...]
    page_count: int


@dataclass(frozen=True)
class PageUse:
    binding: int
    page: int
    bank: int
    ready: int
    release: int


@dataclass(frozen=True)
class Round:
    stage: str
    page: int
    bank: int
    ready: int
    release: int
    byte_size: int
    transfer_cycles: int


def integer_attr(text: str, name: str) -> int:
    match = re.search(rf"\b{re.escape(name)} = (\d+) : i\d+", text)
    if not match:
        raise ValueError(f"missing integer attribute {name}")
    return int(match.group(1))


def string_attr(text: str, name: str) -> str:
    match = re.search(rf'\b{re.escape(name)} = "([^"]+)"', text)
    if not match:
        raise ValueError(f"missing string attribute {name}")
    return match.group(1)


def parse_command(path: Path) -> tuple[list[Binding], list[PageUse], int, int]:
    bindings: list[Binding] = []
    pages: list[PageUse] = []
    c2c_lanes = 0
    bytes_per_lane = 0
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not c2c_lanes:
                match = re.search(r"c2c_streams_per_direction = (\d+) : i\d+", line)
                if match:
                    c2c_lanes = int(match.group(1))
                match = re.search(r"c2c_bytes_per_stream_per_cycle = (\d+) : i\d+", line)
                if match:
                    bytes_per_lane = int(match.group(1))
            if "ftlpu.command.binding" in line and "paged_weight = true" in line:
                shape_match = re.search(r"\bshape = \[([^]]*)\]", line)
                page_match = re.search(r"\bpage_count = (\d+) : i\d+", line)
                if not shape_match or not page_match:
                    raise ValueError("paged weight binding lacks shape or page_count")
                shape = tuple(
                    int(value.strip())
                    for value in shape_match.group(1).split(",") if value.strip()
                )
                bindings.append(Binding(
                    integer_attr(line, "index"), string_attr(line, "name"),
                    integer_attr(line, "bytes"), shape, int(page_match.group(1))))
            elif "ftlpu.command.weight_page" in line:
                pages.append(PageUse(
                    integer_attr(line, "binding_index"),
                    integer_attr(line, "page_index"),
                    integer_attr(line, "bank"),
                    integer_attr(line, "ready_cycle"),
                    integer_attr(line, "release_cycle")))
    if c2c_lanes <= 0 or bytes_per_lane <= 0:
        raise ValueError("command module lacks the target C2C bandwidth")
    return bindings, pages, c2c_lanes, bytes_per_lane


def page_bytes(binding: Binding, page: int) -> int:
    quotient, remainder = divmod(binding.byte_size, binding.page_count)
    return quotient + (1 if page < remainder else 0)


def build_rounds(bindings: list[Binding], uses: list[PageUse],
                 lanes_per_direction: int, bytes_per_lane: int) -> list[Round]:
    uses_by_binding: dict[int, dict[int, PageUse]] = {}
    for use in uses:
        uses_by_binding.setdefault(use.binding, {})[use.page] = use
    bindings = [binding for binding in bindings if binding.index in uses_by_binding]
    shape_groups: dict[tuple[int, ...], list[Binding]] = {}
    for binding in bindings:
        shape_groups.setdefault(binding.shape, []).append(binding)
    projection = next(
        (group for group in shape_groups.values() if len(group) == 2), None)
    if projection is None:
        raise ValueError("expected two paged FFN projection bindings with one shape")
    down = next(
        (binding for binding in bindings if binding not in projection), None)
    if down is None:
        raise ValueError("expected one paged FFN down-projection binding")

    total_bandwidth = 2 * lanes_per_direction * bytes_per_lane
    rounds: list[Round] = []
    projection_pages = min(binding.page_count for binding in projection)
    for page in range(projection_pages):
        page_uses = [uses_by_binding[binding.index][page] for binding in projection]
        if len({use.bank for use in page_uses}) != 1:
            raise ValueError("Gate and Up page banks do not match")
        byte_size = sum(page_bytes(binding, page) for binding in projection)
        rounds.append(Round(
            "Gate + Up", page, page_uses[0].bank,
            min(use.ready for use in page_uses),
            max(use.release for use in page_uses), byte_size,
            math.ceil(byte_size / total_bandwidth)))
    for page in range(down.page_count):
        use = uses_by_binding[down.index][page]
        byte_size = page_bytes(down, page)
        rounds.append(Round(
            "Down", page, use.bank, use.ready, use.release, byte_size,
            math.ceil(byte_size / total_bandwidth)))
    return rounds


def esc(text: str) -> str:
    return html.escape(text, quote=True)


def format_mib(byte_size: int) -> str:
    return f"{byte_size / (1024 * 1024):.3g} MiB"


def render(rounds: list[Round], output: Path, lanes: int,
           bytes_per_lane: int, sequence_length: int) -> None:
    initial = rounds[0].transfer_cycles
    axis_start = -initial
    axis_end = max(item.release for item in rounds)
    width, height = 1920, 590
    left, right = 250, 50
    plot_width = width - left - right
    scale = plot_width / (axis_end - axis_start)
    lanes_y = {
        f"C2C East ({lanes} lanes)": 176,
        f"C2C West ({lanes} lanes)": 216,
        "MEM weight bank 0": 282,
        "MEM weight bank 1": 322,
        "MXM Gate + Up": 388,
        "VXM SwiGLU tail": 428,
        "MXM Down": 468,
    }
    bank_colors = {0: "#66b59a", 1: "#6f9fd8"}
    c2c_color = "#e9a24b"
    swiglu_start = max(item.release for item in rounds if item.stage == "Gate + Up")
    swiglu_end = min(item.ready for item in rounds if item.stage == "Down")

    def x(cycle: int) -> float:
        return left + (cycle - axis_start) * scale

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<defs>",
        '<pattern id="c2c" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">',
        f'<rect width="8" height="8" fill="{c2c_color}"/><line x1="0" y1="0" x2="0" y2="8" stroke="#fff" stroke-width="2" opacity="0.45"/>',
        "</pattern>",
        "</defs>",
        "<style>",
        ".title{font:700 28px 'Segoe UI',Arial,sans-serif;fill:#17212b}",
        ".sub{font:14px 'Segoe UI',Arial,sans-serif;fill:#56636f}",
        ".laneLabel{font:600 13px 'Segoe UI',Arial,sans-serif;fill:#2d3944}",
        ".tick{font:11px 'Segoe UI',Arial,sans-serif;fill:#64717d}",
        ".bar{font:600 10px 'Segoe UI',Arial,sans-serif;fill:#17212b}",
        ".grid{stroke:#dce3e8;stroke-width:1}",
        ".lane{fill:#f9fafb;stroke:#e0e5e9;stroke-width:1}",
        "</style>",
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<text x="42" y="43" class="title">Qwen2.5-1.5B Vector FFN: C2C Weight-Page Pipeline</text>',
        f'<text x="42" y="69" class="sub">seq_len={sequence_length}; dedicated C2C bandwidth per direction: {lanes} lanes x {bytes_per_lane} B/cycle = {lanes * bytes_per_lane} B/cycle.</text>',
        '<rect x="42" y="88" width="18" height="12" rx="2" fill="url(#c2c)" stroke="#805d29"/>',
        '<text x="68" y="99" class="sub">planned C2C prefetch</text>',
        '<rect x="242" y="88" width="18" height="12" rx="2" fill="#66b59a" stroke="#52606c"/>',
        '<text x="268" y="99" class="sub">bank 0 resident / compute</text>',
        '<rect x="490" y="88" width="18" height="12" rx="2" fill="#6f9fd8" stroke="#52606c"/>',
        '<text x="516" y="99" class="sub">bank 1 resident / compute</text>',
        '<rect x="742" y="88" width="18" height="12" rx="2" fill="#df8167" stroke="#52606c"/>',
        '<text x="768" y="99" class="sub">SwiGLU tail</text>',
        f'<text x="{width - right}" y="99" text-anchor="end" class="sub">19 rounds: 7 Gate/Up + 12 Down</text>',
    ]

    for tick in range(11):
        cycle = round(axis_start + (axis_end - axis_start) * tick / 10)
        px = x(cycle)
        lines.append(f'<line x1="{px:.2f}" y1="142" x2="{px:.2f}" y2="508" class="grid"/>')
        lines.append(f'<text x="{px:.2f}" y="132" text-anchor="middle" class="tick">{cycle:,}</text>')
    zero_x = x(0)
    lines.append(f'<line x1="{zero_x:.2f}" y1="138" x2="{zero_x:.2f}" y2="508" stroke="#263746" stroke-width="1.6"/>')
    lines.append(f'<text x="{zero_x + 5:.2f}" y="151" class="tick">ICU cycle 0</text>')

    for label, y in lanes_y.items():
        lines.append(f'<rect x="{left}" y="{y}" width="{plot_width}" height="28" class="lane"/>')
        lines.append(f'<text x="{left - 14}" y="{y + 19}" text-anchor="end" class="laneLabel">{esc(label)}</text>')

    for item in rounds:
        transfer_start = item.ready - item.transfer_cycles
        transfer_width = max(2.0, x(item.ready) - x(transfer_start))
        transfer_label = f'{"GU" if item.stage == "Gate + Up" else "D"}{item.page}'
        per_direction = item.byte_size // 2
        tooltip = (
            f"{item.stage} page {item.page}: C2C {transfer_start}..{item.ready}; "
            f"{format_mib(per_direction)} per direction; bank {item.bank}"
        )
        for y in (lanes_y[f"C2C East ({lanes} lanes)"],
                  lanes_y[f"C2C West ({lanes} lanes)"]):
            lines.append(
                f'<rect x="{x(transfer_start):.2f}" y="{y + 4}" width="{transfer_width:.2f}" height="20" '
                f'rx="2" fill="url(#c2c)" stroke="#805d29" stroke-width="0.7"><title>{esc(tooltip)}</title></rect>')
            if transfer_width >= 25:
                lines.append(f'<text x="{x(transfer_start) + transfer_width / 2:.2f}" y="{y + 18}" text-anchor="middle" class="bar">{transfer_label}</text>')

        compute_width = max(2.0, x(item.release) - x(item.ready))
        mem_y = lanes_y[f"MEM weight bank {item.bank}"]
        label = f'{"Gate/Up" if item.stage == "Gate + Up" else "Down"} p{item.page}'
        compute_tooltip = (
            f"{item.stage} page {item.page}: ready {item.ready}, release {item.release}; "
            f"{format_mib(item.byte_size)}; bank {item.bank}"
        )
        for y in (mem_y, lanes_y[f"MXM {item.stage}"]):
            lines.append(
                f'<rect x="{x(item.ready):.2f}" y="{y + 4}" width="{compute_width:.2f}" height="20" '
                f'rx="2" fill="{bank_colors[item.bank]}" stroke="#46545f" stroke-width="0.7"><title>{esc(compute_tooltip)}</title></rect>')
            if compute_width >= 42:
                lines.append(f'<text x="{x(item.ready) + compute_width / 2:.2f}" y="{y + 18}" text-anchor="middle" class="bar">{label}</text>')

    swiglu_width = max(2.0, x(swiglu_end) - x(swiglu_start))
    lines.append(
        f'<rect x="{x(swiglu_start):.2f}" y="{lanes_y["VXM SwiGLU tail"] + 4}" width="{swiglu_width:.2f}" height="20" '
        f'rx="2" fill="#df8167" stroke="#6d493f" stroke-width="0.7"><title>SwiGLU tail: cycles {swiglu_start}..{swiglu_end}</title></rect>')
    if swiglu_width >= 55:
        lines.append(f'<text x="{x(swiglu_start) + swiglu_width / 2:.2f}" y="{lanes_y["VXM SwiGLU tail"] + 18}" text-anchor="middle" class="bar">SwiGLU tail</text>')

    total_bytes = sum(item.byte_size for item in rounds)
    transfer_summary = ", ".join(
        f"{stage}: {next(item.transfer_cycles for item in rounds if item.stage == stage):,} cycles/page"
        for stage in ("Gate + Up", "Down")
    )
    lines.extend([
        f'<text x="42" y="542" class="sub">Cold start: GU0 is prefetched before ICU cycle 0. During each compute round, the next page is loaded into the other SRAM bank.</text>',
        f'<text x="42" y="566" class="sub">C2C windows are derived from Command IR readiness and target bandwidth ({transfer_summary}); compute windows are executable residency intervals. Total weights: {format_mib(total_bytes)}.</text>',
        "</svg>",
    ])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="FFN Command MLIR")
    parser.add_argument("output", type=Path, help="output SVG")
    parser.add_argument("--sequence-length", type=int, default=32)
    args = parser.parse_args()
    bindings, pages, lanes, bytes_per_lane = parse_command(args.input)
    rounds = build_rounds(bindings, pages, lanes, bytes_per_lane)
    render(rounds, args.output, lanes, bytes_per_lane, args.sequence_length)


if __name__ == "__main__":
    main()
