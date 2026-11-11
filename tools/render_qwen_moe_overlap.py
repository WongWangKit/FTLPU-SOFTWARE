"""Join real CModel trace records and render the serial/overlap comparison."""
import argparse
from collections import defaultdict
import csv
from html import escape
import json
from pathlib import Path
from run_dense_pipelines import embedded_viewer

ROOT = Path(__file__).resolve().parents[1]


def records(path):
    with path.open(encoding="utf-8", newline="") as file:
        yield from csv.DictReader(file)


def span(cycles):
    return (min(cycles), max(cycles) + 1) if cycles else None


def cycles(intervals):
    return {cycle for begin, end in intervals for cycle in range(begin, end)}


def trace_row(begin, end, name, detail):
    return dict(start=begin, end=end, resource="MoE.View." + name, detail=detail,
                pattern="single", inner_count=1, inner_interval=0, inner_stride=0,
                outer_count=1, outer_interval=0, outer_stride=0, skip_first=0,
                induction="none", base_delta=0)


def analyze(output, mode):
    directory = output / mode
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "passed" and summary["bit_mismatches"] == 0
    tasks = [{key: int(value) for key, value in row.items()}
             for row in records(directory / "expert_tasks.csv")]
    ops = list(records(directory / "rcu_operations.csv"))
    base_trace = list(records(directory / "runtime.pipeline.csv"))
    program = next(row for row in base_trace if row["resource"] == "MoE.ProgramLoad")
    start = int(program["end"])
    rows = defaultdict(list)
    rows["Program DMA"].append((int(program["start"]), start, "iMEM and descriptors"))
    gathers = [row for row in ops if row["opcode"] == "ROUTE_ADDR"]
    assert len(gathers) == len(tasks)
    for task, row in zip(tasks, gathers, strict=True):
        task["gather"] = [int(row["start"]), int(row["end"])]
        buffer = "AB"[task.get("weight_buffer", 0)]
        label = f'task {task["task"]} / expert {task["expert"]} / buffer {buffer} / {task["rows"]} rows'
        rows["Gather"].append((*task["gather"], label))
        rows["Expert FFN"].append((task["start"], task["end"], label))
    for row in ops:
        if row["opcode"] == "ROUTE_PLAN":
            rows["PLAN"].append((int(row["start"]), int(row["end"]), row["opcode"]))

    pending = {}
    dma = defaultdict(list)
    ddr_activity = defaultdict(set)
    gather_activity = defaultdict(set)
    weight_commits = defaultdict(set)
    ffn_activity = defaultdict(set)
    prefetch_tasks = set()
    router = set()
    topk = set()
    combine = set()
    for row in records(directory / "instructions.cycles.csv"):
        cycle = int(row["cycle"])
        unit, resource, stage = row["unit"], row["resource"], row["stage"]
        if unit == "DMA" and resource in ("DMA.E", "DMA.W"):
            payload = json.loads(row["payload"])
            if payload.get("sync_tag") != "65535":
                continue
            key = resource, payload["base"]
            task = int(payload["task"])
            if stage == "activate":
                if payload.get("prefetch") == "true":
                    prefetch_tasks.add(task)
                assert key not in pending
                pending[key] = (task, cycle)
            elif stage == "complete":
                owner, begin = pending.pop(key)
                assert owner == task
                dma[task].append((begin, cycle + 1))
            elif stage in ("ddr_read_request", "data_return"):
                ddr_activity[task].add(cycle)
        elif unit == "MEM" and stage in ("read_commit", "write_commit"):
            payload = json.loads(row["payload"])
            if resource.startswith("MEM.W.S") and resource.endswith(".B0"):
                slice_id = int(resource.split(".")[2][1:])
                if stage == "write_commit" and 40 <= slice_id <= 43 and 500 <= int(payload["row"]) < 505:
                    router.add(cycle)
            if "task" in payload:
                task = int(payload["task"])
                if payload.get("stream") == "E31":
                    gather_activity[task].add(cycle)
                if stage == "write_commit" and payload.get("stream") == "W30":
                    weight_commits[task].add(cycle)
        elif resource.startswith("MXM.W0.Compute") and row["opcode"] == "COMPUTE":
            router.add(cycle)
        elif resource == "VXM.TOPK" and stage in ("scan", "merge", "submit"):
            topk.add(cycle)
        elif resource == "RCU.Combine":
            combine.add(cycle)
        elif stage == "execute" and ((unit == "MXM" and resource.endswith(".MAC")) or unit == "VXM"):
            payload = json.loads(row["payload"])
            if "task" in payload:
                ffn_activity[int(payload["task"])].add(cycle)
    assert not pending
    for name, observed in (("Router MXM", router), ("VXM Top-K", topk), ("Combine", combine)):
        interval = span(observed)
        assert interval, f"missing observed phase {name}"
        rows[name].append((*interval, "observed activity"))

    for task in tasks:
        identity = task["task"]
        gather = set(range(*task["gather"]))
        task["dma"] = dma[identity]
        task["overlap"] = len(cycles(dma[identity]) & gather)
        task["ddr_and_gather_activity"] = len(ddr_activity[identity] & gather_activity[identity])
        task["sram_commit_overlap"] = len(weight_commits[identity] & gather_activity[identity])
        for begin, end in task["dma"]:
            kind = "prefetch" if identity in prefetch_tasks else "demand"
            rows["Expert weight DMA"].append((begin, end,
                f'task {identity} / expert {task["expert"]} / buffer {"AB"[task.get("weight_buffer", 0)]} / {kind}'))
            first, last = max(begin, task["gather"][0]), min(end, task["gather"][1])
            if first < last:
                rows["Gather overlap"].append((first, last, f'task {identity} / DMA in flight while Gather runs'))

    cross = []
    for compute in tasks:
        a = compute["task"]
        work = ffn_activity[a] & set(range(compute["start"], compute["end"]))
        for load in tasks:
            b = load["task"]
            if b <= a or compute["expert"] == load["expert"]:
                continue
            overlaps = [(max(begin, compute["start"]), min(end, compute["end"]))
                        for begin, end in dma[b] if begin < compute["end"] and end > compute["start"]]
            if not overlaps:
                continue
            assert b in prefetch_tasks
            assert compute["weight_buffer"] != load["weight_buffer"]
            for begin, end in overlaps:
                rows["Cross-expert overlap"].append((begin, end,
                    f'FFN expert {compute["expert"]} / prefetch expert {load["expert"]}'))
            cross.append(dict(compute_task=a, compute_expert=compute["expert"],
                              compute_buffer=compute["weight_buffer"], prefetch_task=b,
                              prefetch_expert=load["expert"], prefetch_buffer=load["weight_buffer"],
                              intervals=overlaps, inflight_cycles=len(cycles(overlaps)),
                              ddr_and_ffn=len(ddr_activity[b] & work),
                              sram_commit_and_ffn=len(weight_commits[b] & work)))
    base_trace.extend(trace_row(begin, end, name, detail)
                      for name, intervals in rows.items() for begin, end, detail in intervals)
    full = directory / "pipeline.csv"
    with full.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=base_trace[0].keys())
        writer.writeheader()
        writer.writerows(sorted(base_trace, key=lambda row: (int(row["start"]), row["resource"])))
    path = directory / "pipeline.html"
    embedded_viewer(ROOT / "tools/pipeline_viewer", full, path,
                    f"Qwen3-30B-A3B 权重切片 · {mode}")
    source = path.read_text(encoding="utf-8").replace(
        'dom.interactionState.textContent = "实际运行 · 包含最初数据读入";',
        'dom.searchInput.value="MoE.View"; state.query="MoE.View"; applyFilters(); fitAll();\n'
        'dom.interactionState.textContent = "CModel 实际周期 · 含程序 DMA";')
    path.write_text(source, encoding="utf-8")
    return dict(summary=summary, start=start, rows=dict(rows), tasks=tasks,
                overlap=sum(task["overlap"] for task in tasks),
                ddr_and_gather_activity=sum(task["ddr_and_gather_activity"] for task in tasks),
                sram_commit_overlap=sum(task["sram_commit_overlap"] for task in tasks),
                cross_expert=cross,
                cross_expert_overlap=sum(pair["inflight_cycles"] for pair in cross),
                ddr_and_ffn=sum(pair["ddr_and_ffn"] for pair in cross),
                sram_commit_and_ffn=sum(pair["sram_commit_and_ffn"] for pair in cross))


COLORS = {
    "Router MXM": "#326fa8", "VXM Top-K": "#875cad", "PLAN": "#587285",
    "Gather": "#128679", "Expert weight DMA": "#bb7518", "Expert FFN": "#3568bb",
    "Combine": "#915268", "Gather overlap": "#9d4d7e", "Cross-expert overlap": "#bb3057"
}


def svg(result, lo, hi, lanes):
    x0, width = 164, 886
    height = 34 + len(lanes) * 24
    output = [f'<svg viewBox="0 0 1070 {height}" role="img" xmlns="http://www.w3.org/2000/svg">']
    for i in range(6):
        x = x0 + width * i / 5
        cycle = round(lo + (hi - lo) * i / 5)
        output.append(f'<line x1="{x}" x2="{x}" y1="15" y2="{height}" stroke="#dce3eb"/>')
        output.append(f'<text x="{x}" y="12" text-anchor="middle">{cycle:,}</text>')
    for i, name in enumerate(lanes):
        y = 20 + i * 24
        output.append(f'<text x="3" y="{y+13}">{escape(name)}</text>')
        for begin, end, detail in result["rows"].get(name, []):
            begin -= result["start"]
            end -= result["start"]
            if begin >= hi or end <= lo:
                continue
            x = x0 + (max(begin, lo) - lo) * width / (hi - lo)
            bar = (min(end, hi) - max(begin, lo)) * width / (hi - lo)
            tip = escape(f"{detail} · [{begin}, {end}) · {end-begin} cycles")
            output.append(f'<rect x="{x:.2f}" y="{y}" width="{max(0.7,bar):.2f}" height="16" '
                          f'rx="2" fill="{COLORS[name]}"><title>{tip}</title></rect>')
    output.append("</svg>")
    return "".join(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    output = parser.parse_args().output_dir
    modes = ("serial", "overlap", "double-buffered")
    reports = {mode: analyze(output, mode) for mode in modes}
    serial, overlap, double = reports.values()
    assert serial["overlap"] == 0 and overlap["overlap"] > 0
    assert serial["cross_expert_overlap"] == overlap["cross_expert_overlap"] == 0
    assert double["cross_expert_overlap"] > 0 and double["ddr_and_ffn"] > 0
    assert double["summary"]["prefetches"] == double["summary"]["prefetch_hits"] > 0
    for metric in ("dma_segments", "committed_weight_vectors"):
        assert len({report["summary"][metric] for report in reports.values()}) == 1
    for file in ("program.ftlpu", "actual.bf16.bin", "routing.csv"):
        expected = (output / "serial" / file).read_bytes()
        assert all((output / mode / file).read_bytes() == expected for mode in modes[1:])
    limit = max(run["summary"]["physical_cycles"] - run["start"] for run in reports.values())
    lanes = list(COLORS)
    text = ['''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Qwen MoE 流水重叠实测</title><style>
body{font:16px "Microsoft YaHei",Inter,sans-serif;background:#f3f6fa;color:#182638;max-width:1140px;margin:32px auto;padding:0 20px}
section{background:white;border:1px solid #dbe4ef;border-radius:12px;padding:14px 18px;margin:14px 0}
h1{font-size:30px;margin-bottom:8px}h2{font-size:18px}svg{width:100%;display:block}svg text{font:12px Inter,Arial,sans-serif;fill:#34495c}a{color:#18588e}p{line-height:1.6}
table{width:100%;border-collapse:collapse}td,th{text-align:left;padding:8px;border-bottom:1px solid #dbe4ef}
</style><h1>Qwen MoE 流水重叠实测</h1>
<p>Qwen3-30B-A3B 第 0 层官方 BF16 权重切片，5 token、4 专家、Top-2，H=F=32。横轴以程序启动为周期 0，程序 DMA 另计。阶段条可悬停查看起止周期。</p>''']
    for name, key in zip(("串行", "同块 Gather + DMA", "双缓冲跨专家预取"), modes, strict=True):
        run = reports[key]
        duration = run["summary"]["physical_cycles"] - run["start"]
        text.append(f'<section><h2>{name} · {duration:,} 执行周期 · '
                    f'{run["cross_expert_overlap"]:,} 跨专家重叠周期</h2>'
                    + svg(run, 0, limit, lanes)
                    + f'<p><a href="{key}/pipeline.html">打开交互流水</a> · '
                      f'<a href="{key}/instructions.cycles.csv">逐周期指令记录</a></p></section>')
    pair = double["cross_expert"][0]
    compute = next(task for task in double["tasks"] if task["task"] == pair["compute_task"])
    lo, hi = compute["start"], compute["end"]
    text.append(f'<section><h2>放大 · 专家 {pair["compute_expert"]} 执行时预取专家 {pair["prefetch_expert"]}</h2>'
                + svg(double, lo - double["start"] - 30, hi - double["start"] + 30,
                      ["Expert FFN", "Expert weight DMA", "Cross-expert overlap"])
                + '<p>跨专家重叠取实际 DMA 命令 activate–complete 区间和上一专家 FFN 区间的交集。'
                  '同周期 DDR 和 FFN 活动另按逐周期 trace 统计。</p>'
                  '<table><thead><tr><th>当前 FFN</th><th>预取权重</th><th>DMA 在途 ∩ FFN</th>'
                  '<th>DDR 事件 ∩ FFN 运算</th><th>权重写 SRAM ∩ FFN 运算</th></tr></thead><tbody>')
    for entry in double["cross_expert"]:
        text.append(f'<tr><td>专家 {entry["compute_expert"]} / task {entry["compute_task"]} / '
                    f'{"AB"[entry["compute_buffer"]]}</td><td>专家 {entry["prefetch_expert"]} / '
                    f'{"AB"[entry["prefetch_buffer"]]}</td><td>{entry["inflight_cycles"]}</td>'
                    f'<td>{entry["ddr_and_ffn"]}</td><td>{entry["sram_commit_and_ffn"]}</td></tr>')
    text.append('</tbody></table></section>')
    (output / "pipeline_compare.html").write_text("".join(text) + "</html>", encoding="utf-8")
    (output / "comparison.json").write_text(json.dumps(
        reports, indent=2) + "\n", encoding="utf-8")
    for mode, report in reports.items():
        print(mode, report["summary"]["physical_cycles"], "cross-expert overlap",
              report["cross_expert_overlap"], "DDR and FFN active", report["ddr_and_ffn"])


if __name__ == "__main__":
    main()
