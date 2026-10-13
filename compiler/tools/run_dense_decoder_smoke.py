#!/usr/bin/env python3
"""Run one full-shape dense decoder layer through importer, compiler binary, and CModel."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def run(command: list[str], log: Path) -> subprocess.CompletedProcess[str]:
    print("+ " + " ".join(command), flush=True)
    output = []
    environment = os.environ.copy()
    environment.setdefault("FTLPU_SESSION_PROGRESS", "1")
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(command, text=True, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, errors="replace",
                                   env=environment)
        for line in process.stdout:
            output.append(line)
            stream.write(line)
            stream.flush()
            print(line, end="", flush=True)
        return subprocess.CompletedProcess(command, process.wait(), "".join(output), "")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--component", choices=("zero", "attention", "ffn", "both"),
        default="both",
    )
    parser.add_argument("--executable", type=Path)
    parser.add_argument("--qk-pattern", choices=("zero", "nonzero"), default="zero")
    parser.add_argument("--compiler", type=Path,
                        help="Generate and compile the model fixture before testing")
    parser.add_argument("--target-config", type=Path, default=(
        Path(__file__).resolve().parents[3] / "FTLPU-CMODEL"
        / "config" / "ftlpu-lpu32.json"))
    parser.add_argument("--runtime-test", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    creator = here / "create_dense_decoder_smoke_checkpoint.py"
    importer = here / "import_hf_decoder_layer.py"
    packager = here / "build_decoder_layer_package.py"
    run_name = args.model if args.component == "both" else (
        f"{args.model}_{args.component}"
    )
    model_work = args.work_dir / run_name
    model_work.mkdir(parents=True, exist_ok=True)
    compile_commands = []
    if args.compiler:
        from create_dense_decoder_smoke_checkpoint import SPECS
        from generate_dense_decoder_layer import generate
        spec = SPECS[args.model]
        source = model_work / "decoder.stablehlo.mlir"
        source.write_text(generate(
            here.parent / "examples" / "qwen2_5_1_5b_decoder_layer"
            / "decoder_layer_seq32.stablehlo.mlir",
            name=args.model, hidden=spec.hidden, intermediate=spec.intermediate,
            query_heads=spec.query_heads, kv_heads=spec.kv_heads,
            head_dim=spec.head_dim, rope_theta=spec.rope_theta,
            rms_norm_eps=spec.rms_epsilon,
            rotary_dim=96 if spec.model_type == "phi3" else spec.head_dim,
            rope_attention_factor=math.sqrt(1.0 + math.log(32.0) / math.log(4096.0))
                if spec.model_type == "phi3" else 1.0,
            residual_scale=1.4 / math.sqrt(spec.layers)
                if spec.model_type == "minicpm" else 1.0), encoding="utf-8")
        args.executable = args.executable or model_work / "decoder.ftlpu"
        compile_commands.append(("compile", [str(args.compiler),
            "--input", str(source), "--output", str(args.executable),
            "--target-config", str(args.target_config), "--weight-bank", "1",
            "--ffn-schedule", "fused", "--attention-schedule", "fused",
            "--projection-rope-overlap", "off"]))
    elif not args.executable:
        parser.error("provide --compiler or --executable")
    checkpoint = model_work / "checkpoint"
    golden = model_work / "golden"
    package = model_work / f"{args.model}.ftlpum"
    result_dir = args.results_dir / run_name
    result_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    linked = Path(str(package) + ".linked.ftlpu")
    # A failed new run must not inherit an earlier run's executed ICU image.
    linked.unlink(missing_ok=True)
    (result_dir / "linked.ftlpu").unlink(missing_ok=True)
    (result_dir / "result.json").write_text(json.dumps({
        "model": args.model, "component": args.component,
        "sequence_length": 32, "checkpoint_kind": "synthetic_full_shape",
        "qk_pattern": args.qk_pattern,
        "status": "running", "started_at": started_at,
    }, indent=2), encoding="utf-8")

    phases: list[dict[str, object]] = []
    attention_coverage = None
    commands = compile_commands + [
        ("checkpoint", [
            sys.executable, "-B", str(creator), "--model", args.model,
            "--component", args.component, "--output-dir", str(checkpoint),
            "--qk-pattern", args.qk_pattern,
        ]),
        ("reference", [
            sys.executable, "-B", str(importer), "--model-dir", str(checkpoint),
            "--output-dir", str(golden), "--seq-len", "32",
        ]),
        ("package", [
            sys.executable, "-B", str(packager), "--golden-dir", str(golden),
            "--executable", str(args.executable), "--output", str(package),
        ]),
        ("cmodel", [str(args.runtime_test), str(package)]),
    ]
    status = "passed"
    for phase, command in commands:
        started = time.perf_counter()
        completed = run(command, result_dir / f"{phase}.log")
        elapsed = time.perf_counter() - started
        output = completed.stdout + completed.stderr
        phases.append({
            "phase": phase,
            "returncode": completed.returncode,
            "wall_seconds": elapsed,
            "output": output.strip(),
        })
        if completed.returncode:
            status = "failed"
            break
        if phase == "reference" and args.qk_pattern == "nonzero" and args.component in ("attention", "both"):
            import numpy as np
            def read_stage(name):
                words = np.fromfile(golden / f"golden.{name}.bf16.bin", dtype="<u2")
                return (words.astype(np.uint32) << 16).view(np.float32)
            query, key = read_stage("query"), read_stage("key")
            scores = read_stage("scores")
            probability = read_stage("probability").reshape(-1, 32, 32)
            last = probability[:, -1, :]
            attention_coverage = {
                "query_nonzero_elements": int(np.count_nonzero(query)),
                "key_nonzero_elements": int(np.count_nonzero(key)),
                "score_min": float(scores.min()), "score_max": float(scores.max()),
                "last_token_probability_min": float(last.min()),
                "last_token_probability_max": float(last.max()),
                "nonuniform_heads": int(np.count_nonzero(np.ptp(last, axis=1) > 0.001)),
            }
            if not (np.isfinite(scores).all() and np.isfinite(probability).all()
                    and attention_coverage["query_nonzero_elements"]
                    and attention_coverage["key_nonzero_elements"]
                    and attention_coverage["nonuniform_heads"]):
                status = "failed"
                phases.append({"phase": "attention_coverage", "returncode": 1,
                               "output": "nonzero Q/K or nonuniform attention coverage missing"})
                break

    result = {
        "model": args.model,
        "component": args.component,
        "sequence_length": 32,
        "checkpoint_kind": "synthetic_full_shape",
        "qk_pattern": args.qk_pattern,
        "status": status,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "executable_bytes": args.executable.stat().st_size if args.executable.exists() else 0,
        "package_bytes": package.stat().st_size if package.exists() else 0,
        "phases": phases,
        "attention_coverage": attention_coverage,
    }
    (result_dir / "result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    (result_dir / "cmodel.log").write_text(
        next((str(item["output"]) for item in phases if item["phase"] == "cmodel"), ""),
        encoding="utf-8",
    )
    if args.executable.exists():
        shutil.copy2(args.executable, result_dir / "decoder.ftlpu")
    if linked.exists():
        shutil.copy2(linked, result_dir / "linked.ftlpu")
    for source, destination in (
        (golden / "metadata.json", result_dir / "metadata.json"),
        (checkpoint / "smoke_manifest.json", result_dir / "smoke_manifest.json"),
    ):
        if source.exists():
            shutil.copy2(source, destination)
    print(json.dumps(result, indent=2), flush=True)
    if status != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
