"""Recompile one mixer kernel with both packed-copy policies and verify measured binaries."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("compiler", "target-config", "baseline", "optimized", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    kernel = (args.baseline / "kernel.mlir").read_text()
    report = {"compiler_sha256": digest(args.compiler), "variants": {}}
    measured_paths = {args.baseline.resolve(), args.optimized.resolve()}
    for mode, measured in (("off", args.baseline), ("on", args.optimized)):
        folder = (args.output_dir / mode).resolve()
        if folder in measured_paths:
            raise ValueError("Use a separate directory for the recompiled control programs")
        folder.mkdir(parents=True, exist_ok=True)
        flag = "true" if mode == "on" else "false"
        ir, count = re.subn(r'ftlpu\.artifact_directory = "[^"]+"',
                            lambda _: f'ftlpu.artifact_directory = "{folder.as_posix()}"', kernel)
        if count != 1 or ir.count("module attributes {") != 1:
            raise ValueError("Expected one kernel artifact directory and one module attribute list")
        attribute = "ftlpu.elide_overwritten_packed_copies"
        setting = f"{attribute} = {flag}"
        if attribute in ir:
            ir, count = re.subn(re.escape(attribute) + r"\s*=\s*(?:true|false)\b", setting, ir)
            if count != 1:
                raise ValueError("Expected one boolean packed-copy module attribute")
        else:
            ir = ir.replace("module attributes {", f"module attributes {{{setting}, ", 1)
        source, program = folder / "kernel.mlir", folder / "program.ftlpu"
        source.write_text(ir)
        with (folder / "compile.log").open("w") as log:
            subprocess.run([args.compiler.resolve(), "--input", source, "--input-stage", "kernel",
                            "--output", program, "--target-config", args.target_config.resolve(),
                            "--icu-compression", "raw-macro", "--mxm-execution", "ws"],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        expected = digest(measured / "program.ftlpu")
        actual = digest(program)
        report["variants"][mode] = {"compiled": actual, "measured": expected, "match": actual == expected}
    out = args.output_dir / "program_comparison.json"
    out.write_text(json.dumps(report, indent=2))
    print(out.read_text())
    if not all(v["match"] for v in report["variants"].values()):
        raise SystemExit("Measured programs differ from this compiler's A/B pair")


if __name__ == "__main__":
    main()
