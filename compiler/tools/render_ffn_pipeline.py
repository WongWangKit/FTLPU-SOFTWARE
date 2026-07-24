#!/usr/bin/env python3
"""Render runtime FFN ICU commands in the CModel schedule-detail layout."""

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
    "VXM": "#ed996d",
    "MXM.Load": "#91b7e5",
    "MXM.Compute": "#6f9fd8",
}

MXM_TILE_CYCLES = 32


def load_events(path: Path) -> list[Event]:
    with path.open(newline="", encoding="utf-8") as stream:
        events = [
            Event(int(row["start"]), int(row["end"]), row["resource"], row["detail"])
            for row in csv.DictReader(stream)
        ]
    return compact_repeats(events)


def normalized_detail(detail: str) -> str:
    return re.sub(r" count=\d+ interval=\d+ stride=-?\d+$", "", detail)


def repeat_signature(detail: str) -> str:
    """Match a repeated MEM command even though its trace shows the next address."""
    return re.sub(r" addr=-?\d+", " addr=*", normalized_detail(detail))


def compact_repeats(events: list[Event]) -> list[Event]:
    """Fold serialized Instruction+Repeat pairs back into one issue interval."""
    by_resource: dict[str, list[Event]] = defaultdict(list)
    for event in events:
        by_resource[event.resource].append(event)
    compacted: list[Event] = []
    for resource, group in by_resource.items():
        pending: list[Event] = []
        for event in sorted(group, key=lambda item: (item.start, item.end)):
            detail = normalized_detail(event.detail)
            is_repeat = detail != event.detail
            match = None
            if is_repeat:
                signature = repeat_signature(event.detail)
                match = next(
                    (index for index in range(len(pending) - 1, -1, -1)
                     if pending[index].end == event.start
                     and repeat_signature(pending[index].detail) == signature),
                    None,
                )
            if match is None:
                pending.append(Event(event.start, event.end, resource, detail))
                continue
            previous = pending[match]
            pending[match] = Event(previous.start, event.end, resource, previous.detail)
        compacted.extend(pending)
    return group_compute_tiles(group_dequant_lanes(compacted))


def group_compute_tiles(events: list[Event]) -> list[Event]:
    """Reassemble command fragments into the 32-cycle logical MXM tiles."""
    by_resource: dict[str, list[Event]] = defaultdict(list)
    untouched: list[Event] = []
    for event in events:
        if event.resource.endswith(".Compute"):
            by_resource[event.resource].append(event)
        else:
            untouched.append(event)

    grouped = list(untouched)
    for resource, group in by_resource.items():
        tile: list[Event] = []
        tile_cycles = 0

        def flush() -> None:
            nonlocal tile, tile_cycles
            if not tile:
                return
            segments = []
            for segment in tile:
                match = re.search(r"act=([EW]\d+)", segment.detail)
                label = match.group(1) if match else segment.detail
                segments.append(f"{segment.end - segment.start}c {label}")
            detail = tile[0].detail if len(tile) == 1 else (
                "Compute tile; segments=" + "/".join(segments)
            )
            grouped.append(Event(tile[0].start, tile[-1].end, resource, detail))
            tile = []
            tile_cycles = 0

        for event in sorted(group, key=lambda item: (item.start, item.end)):
            if tile and (event.start != tile[-1].end
                         or tile_cycles + event.end - event.start > MXM_TILE_CYCLES):
                flush()
            tile.append(event)
            tile_cycles += event.end - event.start
            if tile_cycles == MXM_TILE_CYCLES:
                flush()
        flush()
    return sorted(grouped, key=lambda event: (event.start, resource_key(event.resource)))


def group_dequant_lanes(events: list[Event]) -> list[Event]:
    """Use the same ALU0-7/ALU8-15 dequant bands as the CModel detail view."""
    candidates: dict[tuple[int, int, str, int], list[tuple[int, Event]]] = defaultdict(list)
    untouched: list[Event] = []
    for event in events:
        match = re.fullmatch(r"VXM\.ALU(\d+)", event.resource)
        if not match:
            untouched.append(event)
            continue
        alu = int(match.group(1))
        if event.detail == "multiply" and alu < 8:
            candidates[(event.start, event.end, "multiply", 0)].append((alu, event))
        elif event.detail.startswith("cast ->") and alu >= 8:
            candidates[(event.start, event.end, "cast", 8)].append((alu, event))
        else:
            untouched.append(event)

    grouped = list(untouched)
    for (start, end, operation, base), group in candidates.items():
        if {alu for alu, _ in group} == set(range(base, base + 8)):
            detail = "dequant int8 multiply" if operation == "multiply" else "dequant cast FP16"
            grouped.append(Event(start, end, f"VXM.ALU{base}-{base + 7}", detail))
        else:
            grouped.extend(event for _, event in group)
    return sorted(grouped, key=lambda event: (event.start, resource_key(event.resource)))


def detect_schedule_policy(events: list[Event]) -> str:
    fused_hidden_streams = {"E30", "E31"}
    for event in events:
        if not event.resource.startswith("MEM") or not event.resource.endswith(".Write"):
            continue
        match = re.search(r"stream=([EW]\d+)", event.detail)
        if match and match.group(1) in fused_hidden_streams:
            return "fused"
    return "tail"


def discover_windows(events: list[Event], policy: str) -> tuple[Window, ...]:
    if not events:
        raise ValueError("runtime trace contains no events")
    end_cycle = max(event.end for event in events)
    swiglu = min(
        event.start for event in events
        if event.resource.startswith("VXM") and "negate" in event.detail
    )
    hidden_write_end = max(
        event.end for event in events
        if event.resource.startswith("MEM")
        and event.resource.endswith(".Write")
        and re.search(r"slice=(21|22|23|29)\b", event.detail)
    )
    down = min(
        event.start for event in events
        if event.start >= hidden_write_end and event.resource.endswith(".Load")
    )

    if policy == "fused":
        transition = "Fused transition: ACC tile complete -> SwiGLU"
        swiglu_finish = "SwiGLU: final staged tile and hidden writeback"
    else:
        transition = "Tail transition: Gate/Up projection complete -> SwiGLU"
        swiglu_finish = "Tail SwiGLU: final tile and hidden writeback"

    return (
        Window(0, min(190, end_cycle),
               "Gate/Up projection: startup and first K-reduction waves"),
        Window(max(0, swiglu - 278), min(end_cycle, swiglu + 78),
               transition),
        Window(swiglu, min(end_cycle, swiglu + 276),
               "SwiGLU: first scheduled tile across East/West"),
        Window(max(0, down - 277), min(end_cycle, down + 1),
               swiglu_finish),
        Window(down, min(end_cycle, down + 210),
               "Down projection: startup and first output waves"),
        Window(max(0, end_cycle - 264), end_cycle,
               "Down projection: final accumulation, cast, and writeback"),
    )


def resource_key(resource: str) -> tuple[int, int, int, str]:
    hemisphere = 0 if ".E" in resource else 1 if ".W" in resource else 2
    if resource.startswith("MEM"):
        if resource.endswith("Accumulate"):
            return 3, hemisphere, 3, resource
        operation = 0 if resource.endswith("Read") else 1
        return 0, hemisphere, operation, resource
    if resource.startswith("SXM"):
        operation = 0 if resource.endswith("Transpose") else 1
        return 1, hemisphere, operation, resource
    if resource.startswith("VXM"):
        match = re.search(r"ALU(\d+)", resource)
        return 2, 0, int(match.group(1)) if match else 20, resource
    if resource.startswith("MXM"):
        operation = 0 if resource.endswith("Load") else 1
        return 3, hemisphere, operation, resource
    if resource.startswith("ACC."):
        operation = 0 if "Gate" in resource or "MXM0" in resource else 1
        return 4, hemisphere, operation, resource
    return 4, hemisphere, 0, resource


def event_color(event: Event) -> str:
    resource = event.resource
    if resource.startswith("ACC."):
        destination = "Stream" if "stream + clear" in event.detail else "Sram"
        return COLORS["MEM.Accumulate." + destination]
    if resource.startswith("MEM"):
        if resource.endswith(".Accumulate"):
            destination = "Stream" if "dst=stream+clear" in event.detail else "Sram"
            return COLORS["MEM.Accumulate." + destination]
        return COLORS["MEM." + resource.rsplit(".", 1)[-1]]
    if resource.startswith("SXM"):
        return COLORS["SXM." + resource.rsplit(".", 1)[-1]]
    if resource.startswith("VXM"):
        return COLORS["VXM"]
    if resource.endswith(".Load"):
        return COLORS["MXM.Load"]
    return COLORS["MXM.Compute"]


def detail_class(event: Event) -> str:
    if event.resource.startswith("ACC."):
        return event.detail
    if event.resource.startswith("MEM"):
        if event.resource.endswith(".Read"):
            return "continuous reads"
        if event.resource.endswith(".Write"):
            return "continuous writes"
        destination = "stream + clear" if "dst=stream+clear" in event.detail else "SRAM"
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
        if (resource.endswith(".Compute") or resource.endswith(".Accumulate")
                or resource.startswith("ACC.")):
            intervals: dict[tuple[int, int], int] = defaultdict(int)
            for event in ordered:
                intervals[(event.start, event.end)] += 1
            merged.extend(
                (Event(start, end, resource, detail), count)
                for (start, end), count in intervals.items()
            )
            continue
        start, end, count = ordered[0].start, ordered[0].end, 1
        for event in ordered[1:]:
            if event.start <= end:
                end = max(end, event.end)
                count += 1
            else:
                merged.append((Event(start, end, resource, detail), count))
                start, end, count = event.start, event.end, 1
        merged.append((Event(start, end, resource, detail), count))
    return sorted(merged, key=lambda item: (item[0].start, resource_key(item[0].resource)))


def esc(value: str) -> str:
    return html.escape(value, quote=True)


def render(events: list[Event], windows: tuple[Window, ...], output: Path,
           sequence_length: int, policy: str) -> None:
    width, left, right = 1800, 245, 45
    plot_width = width - left - right
    row_height, panel_gap, header_height = 26, 54, 124
    panel_data = []
    total_height = header_height
    for window in windows:
        selected = [event for event in events
                    if event.end > window.start and event.start < window.end]
        selected = [
            display_accumulator_event(event)
            if event.resource.endswith(".Accumulate") else event
            for event in selected
        ]
        resources = sorted({event.resource for event in selected}, key=resource_key)
        panel_height = 55 + len(resources) * row_height
        panel_data.append((window, selected, resources, total_height, panel_height))
        total_height += panel_height + panel_gap
    total_height += 36

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{total_height}" viewBox="0 0 {width} {total_height}">',
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
        f'<text x="42" y="43" class="title">Dual-Hemisphere W8A16 FFN: {policy.title()} Runtime Schedule</text>',
        f'<text x="42" y="69" class="sub">X[{sequence_length},576] -> gate/up[{sequence_length},1536] -> SwiGLU[{sequence_length},1536] -> down[{sequence_length},576]. Decoded from the runtime .ftlpu binary; hover bars for exact commands.</text>',
        '<rect x="42" y="82" width="18" height="12" rx="2" fill="#c5a3d9" stroke="#52606c" stroke-width="0.7"/>',
        '<text x="68" y="93" class="sub">Accumulator -> SRAM (keep partial sum)</text>',
        '<rect x="340" y="82" width="18" height="12" rx="2" fill="#e16b6f" stroke="#52606c" stroke-width="0.7"/>',
        '<text x="366" y="93" class="sub">Accumulator -> stream + clear (final sum)</text>',
    ]

    for window, selected, resources, y0, panel_height in panel_data:
        scale = plot_width / (window.end - window.start)
        lines.append(f'<text x="42" y="{y0 + 24}" class="panel">{esc(window.title)}</text>')
        lines.append(
            f'<text x="{width - right}" y="{y0 + 24}" text-anchor="end" class="sub">'
            f'cycles {window.start}..{window.end} ({window.end - window.start} cycles)</text>'
        )
        plot_y = y0 + 42
        for tick in range(6):
            cycle = window.start + round((window.end - window.start) * tick / 5)
            x = left + plot_width * tick / 5
            lines.append(f'<line x1="{x:.2f}" y1="{plot_y}" x2="{x:.2f}" y2="{y0 + panel_height}" class="grid"/>')
            lines.append(f'<text x="{x:.2f}" y="{plot_y - 7}" text-anchor="middle" class="tick">{cycle}</text>')

        resource_y = {}
        for index, resource in enumerate(resources):
            y = plot_y + index * row_height
            resource_y[resource] = y
            lines.append(f'<rect x="{left}" y="{y}" width="{plot_width}" height="{row_height - 2}" class="lane"/>')
            lines.append(f'<text x="{left - 12}" y="{y + 17}" text-anchor="end" class="row">{esc(resource)}</text>')

        for event, count in coalesce(selected):
            clipped_start = max(event.start, window.start)
            clipped_end = min(event.end, window.end)
            x = left + (clipped_start - window.start) * scale
            bar_width = max(1.5, (clipped_end - clipped_start) * scale)
            y = resource_y[event.resource] + 3
            tooltip = f"{event.resource}: cycles {event.start}..{event.end}; {event.detail}"
            if count > 1:
                tooltip += f"; {count} coalesced events"
            lines.append(
                f'<rect x="{x:.2f}" y="{y}" width="{bar_width:.2f}" height="{row_height - 8}" '
                f'rx="2" fill="{event_color(event)}" stroke="#52606c" stroke-width="0.7" opacity="0.92">'
                f'<title>{esc(tooltip)}</title></rect>'
            )
            if bar_width >= 58:
                label = event.detail + (f" x{count}" if count > 1 else "")
                max_chars = max(5, int(bar_width / 7.2))
                if len(label) > max_chars:
                    label = label[:max_chars - 1] + "..."
                lines.append(
                    f'<text x="{x + bar_width / 2:.2f}" y="{y + 13}" text-anchor="middle" '
                    f'class="bar" pointer-events="none">{esc(label)}</text>'
                )
    lines.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def accumulator_slice(event: Event) -> int:
    match = re.search(r"slice=(\d+)", event.detail)
    if not match:
        raise ValueError(f"accumulator event has no slice: {event.detail}")
    return int(match.group(1))


def display_accumulator_event(event: Event) -> Event:
    side = "E" if ".E." in event.resource else "W"
    bank0 = accumulator_slice(event) < 40
    label = "Gate/MXM0" if bank0 else "Up/MXM1"
    stream = "dst=stream+clear" in event.detail
    detail = "accumulate -> stream + clear" if stream else "accumulate -> SRAM"
    return Event(event.start, event.end, f"ACC.{label}.{side}", detail)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sequence-length", type=int, default=128)
    args = parser.parse_args()
    events = load_events(args.input)
    policy = detect_schedule_policy(events)
    render(
        events,
        discover_windows(events, policy),
        args.output,
        args.sequence_length,
        policy,
    )


if __name__ == "__main__":
    main()
