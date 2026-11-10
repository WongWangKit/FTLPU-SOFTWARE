"""Measure cycle coverage (not lane utilization) from an executed pipeline CSV."""
import argparse
import csv
import json
import re
from pathlib import Path


def measure(path):
    active = [bytearray() for _ in range(4)]
    end_cycle = 0
    phases = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            start, end = int(row["start"]), int(row["end"])
            ni, no = int(row["inner_count"]), int(row["outer_count"])
            di, do = int(row["inner_interval"]), int(row["outer_interval"])
            skip = int(row["skip_first"]) != 0
            if ni * no <= int(skip):
                continue
            last = end + (ni - 1) * di + (no - 1) * do
            end_cycle = max(end_cycle, last)
            detail, resource = row["detail"], row["resource"]
            label = detail.split(" ", 1)[0]
            phase = label[6:] if label.startswith("phase=") else "program"
            span = phases.setdefault(phase, [start, last])
            span[0], span[1] = min(span[0], start), max(span[1], last)
            if resource.startswith("MXM.") and resource.endswith(".Compute"):
                mask = active[0]
            elif re.fullmatch(r"VXM\.C\d+", resource) and "bypass" not in detail.lower():
                mask = active[1]
            elif resource == "C2C.HostInput":
                mask = active[2]
            elif resource == "C2C.HostOutput":
                mask = active[3]
            else:
                continue
            if last > len(mask):
                mask.extend(bytes(last - len(mask)))
            width = end - start
            fill = b"\x01" * width
            for outer in range(no):
                first = 1 if skip and outer == 0 else 0
                base = start + outer * do
                if di == width and ni > first:
                    mask[base + first*di:base + (ni-1)*di + width] = fill * (ni-first)
                else:
                    for inner in range(first, ni):
                        t = base + inner * di
                        mask[t:t + width] = fill
    def stats(lo, hi):
        mx, vx, rx, tx = (m[lo:hi] for m in active)
        both = bytes(a & b for a, b in zip(mx, vx))
        first = both.find(b"\x01")
        return dict(start=lo, end=hi, cycles=hi-lo, mxm_compute=sum(mx),
                    vxm_non_bypass=sum(vx), simultaneous=both.count(b"\x01"),
                    first_simultaneous_cycle=lo + first if first >= 0 else None,
                    host_input_window=sum(rx), host_output_window=sum(tx))
    return dict(total=stats(0, end_cycle),
                phase_windows={p:stats(*s) for p, s in phases.items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    args = parser.parse_args()
    print(json.dumps(measure(args.csv), indent=2))
