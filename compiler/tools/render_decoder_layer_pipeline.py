#!/usr/bin/env python3
"""Render a complete decoder-layer runtime trace in CModel detail style."""

from __future__ import annotations

import argparse
import csv
import html
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Event:
    start: int
    end: int
    resource: str
    detail: str


@dataclass(frozen=True)
class Stage:
    start: int
    end: int
    name: str
    color: str


@dataclass(frozen=True)
class Window:
    start: int
    end: int
    title: str


STAGES = (
    Stage(0, 7566, "RMSNorm 1", "#d7c6e6"),
    Stage(7566, 30429, "Q/K/V projection", "#c8dcef"),
    Stage(30429, 33810, "QK", "#a9c9e5"),
    Stage(33810, 38786, "Softmax", "#f0cba8"),
    Stage(38786, 51257, "PV", "#9fc4df"),
    Stage(51257, 73685, "O projection", "#bdd6e9"),
    Stage(73695, 76125, "Residual 1", "#f3d69b"),
    Stage(76143, 83655, "RMSNorm 2", "#d7c6e6"),
    Stage(83655, 146032, "Gate / Up projection", "#c7dfd0"),
    Stage(146032, 158525, "SwiGLU", "#add4bc"),
    Stage(158525, 188810, "Down projection", "#c7dfd0"),
    Stage(188810, 191312, "Residual 2", "#f3d69b"),
    Stage(191312, 192047, "Drain", "#e7eaed"),
)

WINDOWS = (
    Window(0, 192047, "Complete decoder layer"),
    Window(0, 33850, "RMSNorm 1 -> Q/K/V projection -> first QK work"),
    Window(73000, 84500, "O projection -> residual 1 -> RMSNorm 2 -> FFN"),
    Window(83500, 192047, "FFN and final residual"),
)

COLORS = {
    "read": "#8fc8a7",
    "write": "#f0ca78",
    "load": "#91b7e5",
    "compute": "#6f9fd8",
    "acc_sram": "#c5a3d9",
    "acc_stream": "#e16b6f",
    "vxm": "#ed996d",
    "sxm": "#71c3bc",
}

ROWS = (
    "MEM.E",
    "MEM.W",
    "MXM0",
    "MXM1",
    "MXM2",
    "MXM3",
    "VXM.ALU0-7",
    "VXM.ALU8-15",
    "SXM.E",
    "SXM.W",
)


def esc(value: str) -> str:
    return html.escape(value, quote=True)


def load_events(path: Path) -> list[Event]:
    events = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            resource = row["resource"]
            if resource.endswith(".Tail"):
                continue
            events.append(Event(
                int(row["start"]),
                int(row["end"]),
                resource,
                row["detail"],
            ))
    return events


def display_row(event: Event) -> str | None:
    resource = event.resource
    if resource.startswith("MEM") and resource.endswith(".Accumulate"):
        match = re.search(r"\bslice=(\d+)\b", event.detail)
        if match:
            local_mxm = 0 if int(match.group(1)) < 40 else 1
            return f"MXM{local_mxm + (2 if resource.startswith('MEM.W') else 0)}"
    if resource.startswith("MEM.E"):
        return "MEM.E"
    if resource.startswith("MEM.W"):
        return "MEM.W"
    match = re.match(r"MXM\.([EW])(\d+)", resource)
    if match:
        unit = int(match.group(2)) + (2 if match.group(1) == "W" else 0)
        return f"MXM{unit}"
    match = re.match(r"VXM\.ALU(\d+)", resource)
    if match:
        return "VXM.ALU0-7" if int(match.group(1)) < 8 else "VXM.ALU8-15"
    if resource.startswith("SXM.E"):
        return "SXM.E"
    if resource.startswith("SXM.W"):
        return "SXM.W"
    return None


def event_kind(event: Event) -> str:
    resource = event.resource
    if resource.startswith("MEM"):
        if resource.endswith(".Read"):
            return "read"
        if resource.endswith(".Write"):
            return "write"
        if "stream+clear" in event.detail or "stream + clear" in event.detail:
            return "acc_stream"
        return "acc_sram"
    if resource.endswith(".Load"):
        return "load"
    if resource.endswith(".Compute"):
        if event.detail.startswith("Compute "):
            return (
                "compute_stream"
                if "dst=stream" in event.detail
                else "compute_sram"
            )
        if event.detail.startswith("AccumulatorRead "):
            return (
                "acc_stream"
                if " clear" in event.detail
                else "acc_sram"
            )
        return "compute"
    if resource.startswith("VXM"):
        return "vxm"
    return "sxm"


def merge_events(events: list[Event], window: Window) -> list[tuple[str, str, int, int, int]]:
    groups: dict[tuple[str, str], list[Event]] = {}
    for event in events:
        if event.end <= window.start or event.start >= window.end:
            continue
        row = display_row(event)
        if row is None:
            continue
        groups.setdefault((row, event_kind(event)), []).append(event)

    merged = []
    # Gaps narrower than roughly two output pixels are not visually
    # distinguishable in this layer-level view.
    merge_gap = max(1, (window.end - window.start) // 700)
    for (row, kind), group in groups.items():
        ordered = sorted(group, key=lambda event: (event.start, event.end))
        start = ordered[0].start
        end = ordered[0].end
        count = 1
        for event in ordered[1:]:
            if event.start <= end + merge_gap:
                end = max(end, event.end)
                count += 1
            else:
                merged.append((row, kind, start, end, count))
                start, end, count = event.start, event.end, 1
        merged.append((row, kind, start, end, count))
    return sorted(merged, key=lambda item: (item[2], ROWS.index(item[0])))


def render(events: list[Event], output: Path) -> None:
    width = 1900
    left = 185
    right = 45
    plot_width = width - left - right
    row_height = 27
    panel_header = 66
    panel_gap = 45
    header = 132
    panel_height = panel_header + len(ROWS) * row_height
    height = header + len(WINDOWS) * (panel_height + panel_gap) + 25

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        ".title{font:700 29px 'Segoe UI',Arial,sans-serif;fill:#17212b}",
        ".sub{font:14px 'Segoe UI',Arial,sans-serif;fill:#5d6874}",
        ".panel{font:700 18px 'Segoe UI',Arial,sans-serif;fill:#25313c}",
        ".row{font:600 12px 'Segoe UI',Arial,sans-serif;fill:#34404b}",
        ".tick{font:11px 'Segoe UI',Arial,sans-serif;fill:#66717d}",
        ".stage{font:600 11px 'Segoe UI',Arial,sans-serif;fill:#25313c}",
        ".grid{stroke:#dce2e8;stroke-width:1}",
        ".lane{fill:#fafbfc;stroke:#e1e6eb;stroke-width:1}",
        "</style>",
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<text x="42" y="43" class="title">'
        "SmolLM2-135M Decoder Layer: Runtime ICU Schedule</text>",
        '<text x="42" y="70" class="sub">'
        "X[128,576] -> RMSNorm -> Attention -> residual -> RMSNorm -> "
        "SwiGLU FFN -> residual. Decoded from the generated .ftlpu binary."
        "</text>",
    ]

    legend = (
        ("MEM read", COLORS["read"]),
        ("MEM write", COLORS["write"]),
        ("MXM load", COLORS["load"]),
        ("MXM compute", COLORS["compute"]),
        ("ACC -> SRAM", COLORS["acc_sram"]),
        ("ACC -> stream + clear", COLORS["acc_stream"]),
        ("VXM", COLORS["vxm"]),
        ("SXM", COLORS["sxm"]),
    )
    x = 42
    for label, color in legend:
        lines.append(
            f'<rect x="{x}" y="88" width="17" height="12" rx="2" '
            f'fill="{color}" stroke="#52606c" stroke-width="0.7"/>'
        )
        lines.append(
            f'<text x="{x + 24}" y="99" class="sub">{esc(label)}</text>'
        )
        x += 95 + len(label) * 5.8

    y0 = header
    for window in WINDOWS:
        plot_y = y0 + panel_header
        scale = plot_width / (window.end - window.start)
        lines.append(
            f'<text x="42" y="{y0 + 24}" class="panel">'
            f"{esc(window.title)}</text>"
        )
        lines.append(
            f'<text x="{width - right}" y="{y0 + 24}" text-anchor="end" '
            f'class="sub">cycles {window.start}..{window.end} '
            f"({window.end - window.start} cycles)</text>"
        )

        for stage in STAGES:
            start = max(stage.start, window.start)
            end = min(stage.end, window.end)
            if end <= start:
                continue
            sx = left + (start - window.start) * scale
            sw = (end - start) * scale
            lines.append(
                f'<rect x="{sx:.2f}" y="{y0 + 34}" width="{sw:.2f}" '
                f'height="21" fill="{stage.color}" opacity="0.9"/>'
            )
            if sw >= 45:
                lines.append(
                    f'<text x="{sx + sw / 2:.2f}" y="{y0 + 49}" '
                    f'text-anchor="middle" class="stage">{esc(stage.name)}</text>'
                )

        for tick in range(7):
            cycle = window.start + round(
                (window.end - window.start) * tick / 6
            )
            tx = left + plot_width * tick / 6
            lines.append(
                f'<line x1="{tx:.2f}" y1="{plot_y - 5}" x2="{tx:.2f}" '
                f'y2="{plot_y + len(ROWS) * row_height}" class="grid"/>'
            )
            lines.append(
                f'<text x="{tx:.2f}" y="{plot_y - 10}" '
                f'text-anchor="middle" class="tick">{cycle}</text>'
            )

        row_y = {}
        for index, row in enumerate(ROWS):
            y = plot_y + index * row_height
            row_y[row] = y
            lines.append(
                f'<rect x="{left}" y="{y}" width="{plot_width}" '
                f'height="{row_height - 2}" class="lane"/>'
            )
            lines.append(
                f'<text x="{left - 11}" y="{y + 17}" text-anchor="end" '
                f'class="row">{esc(row)}</text>'
            )

        for row, kind, start, end, count in merge_events(events, window):
            clipped_start = max(start, window.start)
            clipped_end = min(end, window.end)
            bx = left + (clipped_start - window.start) * scale
            bw = max(1.5, (clipped_end - clipped_start) * scale)
            by = row_y[row] + 4
            tooltip = (
                f"{row}: cycles {start}..{end}; {kind}; "
                f"{count} coalesced commands"
            )
            fill = COLORS.get(kind, COLORS["compute"])
            lines.append(
                f'<rect x="{bx:.2f}" y="{by}" width="{bw:.2f}" '
                f'height="{row_height - 10}" rx="2" '
                f'fill="{fill}" stroke="#52606c" '
                f'stroke-width="0.55" opacity="0.92">'
                f"<title>{esc(tooltip)}</title></rect>"
            )
            if kind in ("compute_sram", "compute_stream"):
                accumulator_color = COLORS[
                    "acc_stream"
                    if kind == "compute_stream"
                    else "acc_sram"
                ]
                lines.append(
                    f'<rect x="{bx:.2f}" y="{by + row_height - 15}" '
                    f'width="{bw:.2f}" height="4" rx="1" '
                    f'fill="{accumulator_color}" opacity="0.96"/>'
                )

        y0 += panel_height + panel_gap

    lines.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    render(load_events(args.input), args.output)


if __name__ == "__main__":
    main()
