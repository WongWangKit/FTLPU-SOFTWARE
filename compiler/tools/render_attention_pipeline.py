#!/usr/bin/env python3
"""Render semantic Attention windows from an ICU runtime trace as an SVG."""

from __future__ import annotations

import argparse
import csv
import html
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Event:
    start: int
    end: int
    resource: str
    detail: str


@dataclass(frozen=True)
class Window:
    start: int
    end: int
    title: str


COLORS = {
    "MEM.Read": "#8fc8a7",
    "MEM.Write": "#f0ca78",
    "MEM.Accumulate.Sram": "#c5a3d9",
    "MEM.Accumulate.Stream": "#e16b6f",
    "SXM.Transpose": "#71c3bc",
    "SXM.Permute": "#4ca9a0",
    "SXM.Tail": "#c7e5e1",
    "VXM": "#ed996d",
    "MXM.Load": "#91b7e5",
    "MXM.Compute": "#6f9fd8",
    "MXM.Tail": "#dce5ef",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--window",
        action="append",
        default=[],
        metavar="START:END:TITLE",
        help="Override a semantic cycle window; may be repeated.",
    )
    parser.add_argument(
        "--title",
        default="SmolLM2-135M Attention: Compiler Runtime ICU Schedule Windows",
    )
    parser.add_argument(
        "--subtitle",
        default=(
            "Decoded .ftlpu commands grouped by functional unit; MXM/SXM drain tails "
            "are omitted. Hover for exact cycles, streams, addresses, and repeats."
        ),
    )
    return parser.parse_args()


def parse_windows(values: list[str]) -> tuple[Window, ...]:
    windows = []
    for value in values:
        fields = value.split(":", 2)
        if len(fields) != 3:
            raise ValueError(f"invalid window: {value}")
        start, end = int(fields[0]), int(fields[1])
        if end <= start:
            raise ValueError(f"window end must exceed start: {value}")
        windows.append(Window(start, end, fields[2]))
    return tuple(windows)


def load_events(path: Path) -> list[Event]:
    events = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["resource"].endswith(".Tail"):
                continue
            events.append(
                Event(
                    int(row["start"]),
                    int(row["end"]),
                    row["resource"],
                    row["detail"],
                )
            )
    return group_dequant_lanes(events)


def group_dequant_lanes(events: list[Event]) -> list[Event]:
    """Fold the 16 weight-dequant ALUs into multiply and cast bands."""
    candidates: dict[tuple[int, int, str, int], list[tuple[int, Event]]] = (
        defaultdict(list)
    )
    untouched: list[Event] = []
    for event in events:
        match = re.fullmatch(r"VXM\.ALU(\d+)", event.resource)
        if not match:
            untouched.append(event)
            continue
        alu = int(match.group(1))
        if event.detail == "multiply" and alu < 8:
            candidates[(event.start, event.end, "multiply", 0)].append(
                (alu, event)
            )
        elif event.detail.startswith("cast ->") and alu >= 8:
            candidates[(event.start, event.end, "cast", 8)].append((alu, event))
        else:
            untouched.append(event)

    grouped = list(untouched)
    for (start, end, operation, base), group in candidates.items():
        if {alu for alu, _ in group} == set(range(base, base + 8)):
            detail = (
                "dequant int8 multiply"
                if operation == "multiply"
                else "dequant cast FP16"
            )
            grouped.append(
                Event(start, end, f"VXM.ALU{base}-{base + 7}", detail)
            )
        else:
            grouped.extend(event for _, event in group)
    return grouped


def first_start(events: list[Event], predicate) -> int:
    matches = [event.start for event in events if predicate(event)]
    if not matches:
        raise ValueError("runtime trace is missing an Attention scheduling landmark")
    return min(matches)


def merge_intervals(events: list[Event], max_gap: int) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for event in sorted(events, key=lambda item: (item.start, item.end)):
        if not merged or event.start > merged[-1][1] + max_gap:
            merged.append([event.start, event.end])
        else:
            merged[-1][1] = max(merged[-1][1], event.end)
    return [(start, end) for start, end in merged]


def discover_windows(events: list[Event]) -> tuple[Window, ...]:
    """Locate stable semantic landmarks instead of pinning windows to one schedule."""
    softmax_start = first_start(
        events,
        lambda event: event.resource.startswith("VXM.")
        and event.detail in ("multiply -> E8", "add -> E8"),
    )
    projection_compute = [
        event
        for event in events
        if event.resource.endswith(".Compute") and event.start < softmax_start
    ]
    qk_start = merge_intervals(projection_compute, max_gap=64)[-1][0]

    rope_start = first_start(
        events,
        lambda event: event.resource.startswith("VXM.")
        and event.detail == "subtract -> E0",
    )
    packed_p_start = first_start(
        events,
        lambda event: event.start > softmax_start
        and event.resource.startswith("VXM.")
        and event.detail == "pass -> E0",
    )
    sxm_start = first_start(
        events, lambda event: event.resource.startswith("SXM.")
    )
    pv_compute_start = first_start(
        events,
        lambda event: event.start > sxm_start
        and event.resource.endswith(".Compute"),
    )
    post_pv_compute = []
    for event in events:
        if (
            event.start > pv_compute_start
            and event.resource.endswith(".Compute")
            and event.detail.startswith("Compute ")
        ):
            accumulator = re.search(r"\bacc=(\d+)\b", event.detail)
            if accumulator:
                post_pv_compute.append((event, int(accumulator.group(1))))
    if not post_pv_compute:
        raise ValueError("runtime trace is missing post-PV MXM computes")
    # O projection reuses one high accumulator block after each completed
    # output group. PV uses lower context-addressed blocks.
    o_accumulator_base = max(accumulator for _, accumulator in post_pv_compute)
    o_compute_start = min(
        event.start
        for event, accumulator in post_pv_compute
        if accumulator == o_accumulator_base
    )

    cast_events = [
        event
        for event in events
        if event.start < qk_start
        and event.resource.startswith("VXM.")
        and event.detail.startswith("FP32->FP16 output cast")
    ]
    cast_clusters = merge_intervals(cast_events, max_gap=64)
    if not cast_clusters:
        raise ValueError("runtime trace is missing projection output casts")
    final_projection_cast = cast_clusters[-1][0]

    return (
        Window(0, 204, "Q projection: first reduction block"),
        Window(
            max(0, rope_start - 8),
            rope_start + 76,
            "Q RoPE: FP32 rotate, FP16 cast, and MEM writeback",
        ),
        Window(
            final_projection_cast,
            min(qk_start, final_projection_cast + 170),
            "V projection: FP16 cast -> packed 16-stream MEM layout",
        ),
        Window(
            qk_start + 450,
            min(softmax_start, qk_start + 1094),
            "QK: four MXMs, independent query blocks",
        ),
        Window(
            softmax_start + 609,
            softmax_start + 1049,
            "Softmax: P1 scale/mask/max, P2 exp/sum, P3 normalize/cast",
        ),
        Window(
            packed_p_start + 593,
            packed_p_start + 997,
            "Post-softmax P layout: packed x16 blocks at II=4",
        ),
        Window(
            max(sxm_start, pv_compute_start - 36),
            pv_compute_start + 288,
            "P x V: passive cross-hemisphere streams, V IW, and direct P replay",
        ),
        Window(
            max(0, o_compute_start - 36),
            o_compute_start + 276,
            "O projection: first four-MXM reduction wave",
        ),
    )


def resource_key(resource: str) -> tuple[int, int, int, str]:
    hemisphere = 0 if ".E" in resource else 1 if ".W" in resource else 2
    if resource.startswith("MEM"):
        if resource.endswith("Accumulate"):
            return (3, hemisphere, 3, resource)
        operation = 0 if resource.endswith("Read") else 1
        return (0, hemisphere, operation, resource)
    if resource.startswith("SXM"):
        operation = (
            0
            if resource.endswith("Transpose")
            else 1
            if resource.endswith("Permute")
            else 2
        )
        return (1, hemisphere, operation, resource)
    if resource.startswith("VXM"):
        match = re.search(r"ALU(\d+)", resource)
        alu = int(match.group(1)) if match else 20
        return (2, 0, alu, resource)
    if resource.startswith("MXM"):
        operation = (
            0
            if resource.endswith("Load")
            else 1
            if resource.endswith("Compute")
            else 2
        )
        return (3, hemisphere, operation, resource)
    return (4, hemisphere, 0, resource)


def accumulator_stream_destination(detail: str) -> bool:
    return "dst=stream+clear" in detail or "-> stream + clear" in detail


def event_color(event: Event) -> str:
    resource = event.resource
    if resource.startswith("MEM"):
        if resource.endswith(".Accumulate"):
            destination = (
                "Stream" if accumulator_stream_destination(event.detail) else "Sram"
            )
            return COLORS["MEM.Accumulate." + destination]
        return COLORS["MEM." + resource.rsplit(".", 1)[-1]]
    if resource.startswith("SXM"):
        return COLORS["SXM." + resource.rsplit(".", 1)[-1]]
    if resource.startswith("VXM"):
        return COLORS["VXM"]
    if resource.endswith(".Load"):
        return COLORS["MXM.Load"]
    if resource.endswith(".Compute"):
        return COLORS["MXM.Compute"]
    return COLORS["MXM.Tail"]


def detail_class(event: Event) -> str:
    if event.resource.startswith("MEM"):
        if event.resource.endswith(".Read"):
            return "continuous reads"
        if event.resource.endswith(".Write"):
            return "continuous writes"
        destination = (
            "stream + clear"
            if accumulator_stream_destination(event.detail)
            else "SRAM"
        )
        return f"continuous accumulates -> {destination}"
    if event.resource.endswith(".Load"):
        return re.sub(r" column=\d+", "", event.detail)
    return event.detail


def coalesce(events: list[Event]) -> list[tuple[Event, int]]:
    grouped: dict[tuple[str, str], list[Event]] = defaultdict(list)
    for event in events:
        grouped[(event.resource, detail_class(event))].append(event)

    merged: list[tuple[Event, int]] = []
    for (resource, detail), group in grouped.items():
        ordered = sorted(group, key=lambda event: (event.start, event.end))
        if resource.endswith(".Compute"):
            merged.extend(
                (Event(event.start, event.end, resource, detail), 1)
                for event in ordered
            )
            continue
        start = ordered[0].start
        end = ordered[0].end
        count = 1
        for event in ordered[1:]:
            if event.start <= end:
                end = max(end, event.end)
                count += 1
                continue
            merged.append((Event(start, end, resource, detail), count))
            start, end, count = event.start, event.end, 1
        merged.append((Event(start, end, resource, detail), count))

    return sorted(
        merged, key=lambda item: (item[0].start, resource_key(item[0].resource))
    )


def esc(value: str) -> str:
    return html.escape(value, quote=True)


def render(
    events: list[Event],
    windows: tuple[Window, ...],
    output: Path,
    title: str,
    subtitle: str,
) -> None:
    width = 1800
    left = 245
    right = 45
    plot_width = width - left - right
    row_height = 26
    panel_gap = 54
    header_height = 124

    panel_data = []
    total_height = header_height
    for window in windows:
        selected = [
            event
            for event in events
            if event.end > window.start and event.start < window.end
        ]
        resources = sorted(
            {event.resource for event in selected}, key=resource_key
        )
        panel_height = 55 + len(resources) * row_height
        panel_data.append((window, selected, resources, total_height, panel_height))
        total_height += panel_height + panel_gap
    total_height += 36

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{total_height}" viewBox="0 0 {width} {total_height}">',
        "<style>",
        ".title{font:700 29px 'Segoe UI',Arial,sans-serif;fill:#17212b}",
        ".sub{font:14px 'Segoe UI',Arial,sans-serif;fill:#5d6874}",
        ".panel{font:700 18px 'Segoe UI',Arial,sans-serif;fill:#25313c}",
        ".row{font:600 12px 'Segoe UI',Arial,sans-serif;fill:#34404b}",
        ".tick{font:11px 'Segoe UI',Arial,sans-serif;fill:#66717d}",
        ".bar{font:600 10px 'Segoe UI',Arial,sans-serif;fill:#17212b}",
        ".grid{stroke:#dce2e8;stroke-width:1}",
        ".lane{fill:#fafbfc;stroke:#e1e6eb;stroke-width:1}",
        "</style>",
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text x="42" y="43" class="title">{esc(title)}</text>',
        f'<text x="42" y="69" class="sub">{esc(subtitle)}</text>',
        '<rect x="42" y="82" width="18" height="12" rx="2" '
        'fill="#c5a3d9" stroke="#52606c" stroke-width="0.7"/>',
        '<text x="68" y="93" class="sub">'
        "Accumulator -&gt; SRAM (keep partial sum)</text>",
        '<rect x="340" y="82" width="18" height="12" rx="2" '
        'fill="#e16b6f" stroke="#52606c" stroke-width="0.7"/>',
        '<text x="366" y="93" class="sub">'
        "Accumulator -&gt; stream + clear (final sum)</text>",
    ]

    for window, selected, resources, y0, panel_height in panel_data:
        scale = plot_width / (window.end - window.start)
        lines.append(
            f'<text x="42" y="{y0 + 24}" class="panel">{esc(window.title)}</text>'
        )
        lines.append(
            f'<text x="{width - 45}" y="{y0 + 24}" text-anchor="end" '
            f'class="sub">cycles {window.start}..{window.end} '
            f"({window.end - window.start} cycles)</text>"
        )
        plot_y = y0 + 42
        for tick in range(6):
            cycle = window.start + round(
                (window.end - window.start) * tick / 5
            )
            x = left + plot_width * tick / 5
            lines.append(
                f'<line x1="{x:.2f}" y1="{plot_y}" x2="{x:.2f}" '
                f'y2="{y0 + panel_height}" class="grid"/>'
            )
            lines.append(
                f'<text x="{x:.2f}" y="{plot_y - 7}" text-anchor="middle" '
                f'class="tick">{cycle}</text>'
            )

        resource_y = {}
        for index, resource in enumerate(resources):
            y = plot_y + index * row_height
            resource_y[resource] = y
            lines.append(
                f'<rect x="{left}" y="{y}" width="{plot_width}" '
                f'height="{row_height - 2}" class="lane"/>'
            )
            lines.append(
                f'<text x="{left - 12}" y="{y + 17}" text-anchor="end" '
                f'class="row">{esc(resource)}</text>'
            )

        for event, count in coalesce(selected):
            clipped_start = max(event.start, window.start)
            clipped_end = min(event.end, window.end)
            x = left + (clipped_start - window.start) * scale
            bar_width = max(1.5, (clipped_end - clipped_start) * scale)
            y = resource_y[event.resource] + 3
            tooltip = (
                f"{event.resource}: cycles {event.start}..{event.end}; "
                f"{event.detail}"
            )
            if count > 1:
                tooltip += f"; {count} coalesced events"
            lines.append(
                f'<rect x="{x:.2f}" y="{y}" width="{bar_width:.2f}" '
                f'height="{row_height - 8}" rx="2" fill="{event_color(event)}" '
                f'stroke="#52606c" stroke-width="0.7" opacity="0.92">'
                f"<title>{esc(tooltip)}</title></rect>"
            )
            if bar_width >= 58:
                label = event.detail + (f" x{count}" if count > 1 else "")
                max_chars = max(5, int(bar_width / 7.2))
                if len(label) > max_chars:
                    label = label[: max_chars - 1] + "..."
                lines.append(
                    f'<text x="{x + bar_width / 2:.2f}" y="{y + 13}" '
                    f'text-anchor="middle" class="bar" pointer-events="none">'
                    f"{esc(label)}</text>"
                )

    lines.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    events = load_events(args.input)
    windows = parse_windows(args.window) or discover_windows(events)
    render(events, windows, args.output, args.title, args.subtitle)


if __name__ == "__main__":
    main()
