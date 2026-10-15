#!/usr/bin/env python3
"""Exercise layout normalization, strict coverage, specialization and ABI maps.

Each numerical fixture is serialized with the official StableHLO portable API.
Run --verify-numerics on the recorded directory in the existing Linux XLA env.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path


LAYOUT = '''module {
  func.func @main(%x: tensor<2x3xf32>, %b: tensor<3xf32>) -> tensor<2x3xf32> {
    %r1 = stablehlo.reshape %x : (tensor<2x3xf32>) -> tensor<1x2x3xf32>
    %r2 = stablehlo.reshape %r1 : (tensor<1x2x3xf32>) -> tensor<2x3xf32>
    %t1 = stablehlo.transpose %r2, dims = [1, 0] : (tensor<2x3xf32>) -> tensor<3x2xf32>
    %t2 = stablehlo.transpose %t1, dims = [1, 0] : (tensor<3x2xf32>) -> tensor<2x3xf32>
    %b1 = stablehlo.broadcast_in_dim %b, dims = [1] : (tensor<3xf32>) -> tensor<1x3xf32>
    %b2 = stablehlo.broadcast_in_dim %b1, dims = [0, 1] : (tensor<1x3xf32>) -> tensor<2x3xf32>
    %out = stablehlo.add %t2, %b2 : tensor<2x3xf32>
    return %out : tensor<2x3xf32>
  }
}
'''

POINTWISE = '''module {
  func.func @main(%x: tensor<2x3xf32>, %y: tensor<2x3xf32>) -> tensor<2x3xf32> {
    %a = stablehlo.reshape %x : (tensor<2x3xf32>) -> tensor<1x2x3xf32>
    %b = stablehlo.reshape %y : (tensor<2x3xf32>) -> tensor<1x2x3xf32>
    %sum = stablehlo.add %a, %b : tensor<1x2x3xf32>
    %out = stablehlo.reshape %sum : (tensor<1x2x3xf32>) -> tensor<2x3xf32>
    return %out : tensor<2x3xf32>
  }
}
'''

STATIC_MASK = '''module {
  func.func @main(%x: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %indices = stablehlo.iota dim = 0 : tensor<4xi64>
    %row = stablehlo.broadcast_in_dim %indices, dims = [0] : (tensor<4xi64>) -> tensor<4x4xi64>
    %column = stablehlo.broadcast_in_dim %indices, dims = [1] : (tensor<4xi64>) -> tensor<4x4xi64>
    %allowed = stablehlo.compare LE, %column, %row : (tensor<4x4xi64>, tensor<4x4xi64>) -> tensor<4x4xi1>
    %zero = stablehlo.constant dense<0.0> : tensor<4x4xf32>
    %floor = stablehlo.constant dense<-1.0e30> : tensor<4x4xf32>
    %mask = stablehlo.select %allowed, %zero, %floor : tensor<4x4xi1>, tensor<4x4xf32>
    %out = stablehlo.add %x, %mask : tensor<4x4xf32>
    return %out : tensor<4x4xf32>
  }
}
'''


def dot_source(singleton=False, precision="DEFAULT"):
    left = "1x4x3" if singleton else "3x4"
    right = "1x5x3" if singleton else "5x3"
    result = "1x4x5" if singleton else "4x5"
    batching = "[0]" if singleton else "[]"
    lhs_contract = "[2]" if singleton else "[0]"
    rhs_contract = "[2]" if singleton else "[1]"
    return f'''module {{
  func.func @main(%lhs: tensor<{left}xf32>, %rhs: tensor<{right}xf32>) -> tensor<{result}xf32> {{
    %out = "stablehlo.dot_general"(%lhs, %rhs) {{
      dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = {batching}, rhs_batching_dimensions = {batching}, lhs_contracting_dimensions = {lhs_contract}, rhs_contracting_dimensions = {rhs_contract}>,
      precision_config = [#stablehlo<precision {precision}>, #stablehlo<precision {precision}>]
    }} : (tensor<{left}xf32>, tensor<{right}xf32>) -> tensor<{result}xf32>
    return %out : tensor<{result}xf32>
  }}
}}
'''


def unsupported_source(operation):
    return f'''module {{
  func.func @main(%x: tensor<2x3xf32>) -> tensor<2x3xf32> {{
    %out = stablehlo.{operation} %x : tensor<2x3xf32>
    return %out : tensor<2x3xf32>
  }}
}}
'''


MASK = '''module {
  func.func @main(%x: tensor<2x3xf32>, %mask: tensor<2x3xi1>) -> tensor<2x3xf32> {
    %zero = stablehlo.constant dense<0.0> : tensor<2x3xf32>
    %out = stablehlo.select %mask, %x, %zero : tensor<2x3xi1>, tensor<2x3xf32>
    return %out : tensor<2x3xf32>
  }
}
'''

SPECIALIZE = '''module {
  func.func @main(%unused: tensor<7xf32>, %frequency: tensor<3xf32>, %x: tensor<3xf32>) -> tensor<3xf32> {
    %out = stablehlo.multiply %frequency, %x : tensor<3xf32>
    return %out : tensor<3xf32>
  }
}
'''


def invoke(tool, root, name, source, *arguments, succeeds=True):
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    input_path = directory / "input.mlir"
    output_path = directory / "output.mlir"
    input_path.write_text(source, encoding="utf-8")
    command = [str(tool), "--input", str(input_path), "--output", str(output_path),
               "--diagnostic-dir", str(directory / "diagnostics"), *arguments]
    result = subprocess.run(command, capture_output=True, text=True)
    (directory / "stdout.txt").write_text(result.stdout, encoding="utf-8")
    (directory / "stderr.txt").write_text(result.stderr, encoding="utf-8")
    (directory / "command.json").write_text(json.dumps(command, indent=2), encoding="utf-8")
    if (result.returncode == 0) != succeeds:
        raise AssertionError(f"{name}: unexpected exit {result.returncode}\n{result.stderr}")
    return output_path, result.stderr


def structural_tests(tool, root):
    numeric_cases = []
    cases = [
        ("layout", LAYOUT, [[2, 3], [3]]),
        ("pointwise_view", POINTWISE, [[2, 3], [2, 3]]),
        ("rank2_dimensions", dot_source(), [[3, 4], [5, 3]]),
        ("singleton_batch", dot_source(True), [[1, 4, 3], [1, 5, 3]]),
        ("precision_retained", dot_source(True, "HIGH"), [[1, 4, 3], [1, 5, 3]]),
        ("static_mask", STATIC_MASK, [[4, 4]]),
    ]
    for name, source, shapes in cases:
        normalized, _ = invoke(tool, root, name, source,
            "--pipeline", "ftlpu-normalize-stablehlo",
            "--fold-static-mask", "on" if name == "static_mask" else "off",
            "--portable-output", str(root / name / "normalized.bytecode"))
        invoke(tool, root, name + "_original", source,
            "--pipeline", "ftlpu-normalize-stablehlo", "--stablehlo-normalize", "off",
            "--portable-output", str(root / name / "original.bytecode"))
        text = normalized.read_text(encoding="utf-8")
        if "arithmetic_reassociation = false" not in text or "abi_changed = false" not in text:
            raise AssertionError(f"{name}: missing numerical/ABI scope")
        if name in ("layout", "pointwise_view"):
            if "stablehlo.reshape" in text or "stablehlo.transpose" in text:
                raise AssertionError(f"{name}: redundant layout survived")
        if name == "layout" and text.count("stablehlo.broadcast_in_dim") != 1:
            raise AssertionError("broadcast dimensions were not composed")
        if name in ("rank2_dimensions", "singleton_batch"):
            if "contracting_dims = [1] x [0]" not in text:
                raise AssertionError("actual dot contracting dimensions were not normalized")
        if name == "precision_retained" and text.count("HIGH") != 2:
            raise AssertionError("dot precision was lost during normalization")
        if name == "static_mask" and any(operation in text for operation in
                ("stablehlo.iota", "stablehlo.compare", "stablehlo.select")):
            raise AssertionError("static mask was not evaluated using the upstream interpreter")
        before_signature = next(line.strip() for line in source.splitlines() if "func.func" in line)
        after_signature = next(line.strip() for line in text.splitlines() if "func.func" in line)
        before_types = re.findall(r"tensor<[^>]+>", before_signature)
        after_types = re.findall(r"tensor<[^>]+>", after_signature)
        if before_types != after_types:
            raise AssertionError(f"{name}: function ABI changed")
        numeric_cases.append({"name": name, "input_shapes": shapes,
                              "original": f"{name}/original.bytecode",
                              "normalized": f"{name}/normalized.bytecode"})

    for name, source, operation in [
        ("unsupported_silu", unsupported_source("logistic"), "stablehlo.logistic"),
        ("unsupported_rope", unsupported_source("sine"), "stablehlo.sine"),
        ("padding_mask", MASK, "stablehlo.select"),
        ("bias_broadcast", LAYOUT, "stablehlo.broadcast_in_dim"),
    ]:
        _, stderr = invoke(tool, root, name, source,
            "--pipeline", "ftlpu-stablehlo-to-tensor", succeeds=False)
        if "incomplete Kernel conversion" not in stderr or operation not in stderr:
            raise AssertionError(f"{name}: missing strict coverage diagnostic")
        report = json.loads((root / name / "diagnostics/report.json").read_text())
        if report["failed_stage"] != "ftlpu-verify-kernel-coverage":
            raise AssertionError(f"{name}: failure was allowed to enter Tensor stage")
    partial, stderr = invoke(tool, root, "partial_debug", MASK,
        "--pipeline", "ftlpu-stablehlo-to-kernel", "--kernel-mode", "partial")
    if "ftlpu.kernel_complete = false" not in partial.read_text():
        raise AssertionError("partial output lacked an explicit incomplete marker")
    if "partial Kernel conversion" not in stderr:
        raise AssertionError("partial output lacked an explicit warning")
    invoke(tool, root, "partial_to_tensor_refused", MASK,
        "--pipeline", "ftlpu-stablehlo-to-tensor", "--kernel-mode", "partial", succeeds=False)
    _, stderr = invoke(tool, root, "rank2_precision_refused", dot_source(False, "HIGH"),
        "--pipeline", "ftlpu-stablehlo-to-kernel", succeeds=False)
    if "rank-2 matmul supports only DEFAULT precision" not in stderr:
        raise AssertionError("rank-2 numeric precision contract was silently discarded")

    bindings = root / "constants.json"
    bindings.write_text(json.dumps({"schema_version": 1, "functions": {"main": [
        {"argument_index": 1, "shape": [3], "dtype": "float32", "values": [1., .5, .25]}
    ]}}), encoding="utf-8")
    output, _ = invoke(tool, root, "specialization", SPECIALIZE,
        "--pipeline", "ftlpu-normalize-stablehlo", "--constant-bindings", str(bindings),
        "--prune-unused-inputs")
    mapping = json.loads((root / "specialization/diagnostics/abi_map.json").read_text())["functions"]["main"]
    if [entry["final_index"] for entry in mapping] != [None, None, 0]:
        raise AssertionError("pruned ABI map did not preserve original argument positions")
    if [entry["specialized_constant"] for entry in mapping] != [False, True, False]:
        raise AssertionError("specialization metadata was incorrect")
    if "tensor<7xf32>" in next(line for line in output.read_text().splitlines() if "func.func" in line):
        raise AssertionError("unused argument was not pruned")
    invalid = json.loads(bindings.read_text())
    invalid["functions"]["main"][0]["shape"] = [4]
    bad = root / "bad_constants.json"
    bad.write_text(json.dumps(invalid), encoding="utf-8")
    invoke(tool, root, "specialization_shape_refused", SPECIALIZE,
        "--pipeline", "ftlpu-normalize-stablehlo", "--constant-bindings", str(bad), succeeds=False)
    oversized = '''module {
      func.func @main(%x: tensor<1x16384xf32>, %w: tensor<16384x16384xf32>) -> tensor<1x16384xf32> {
        %out = stablehlo.dot_general %x, %w, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x16384xf32>, tensor<16384x16384xf32>) -> tensor<1x16384xf32>
        return %out : tensor<1x16384xf32>
      }
    }'''
    _, stderr = invoke(tool, root, "input_capacity", oversized,
        "--pipeline", "ftlpu-stablehlo-to-tensor", succeeds=False)
    for field in ("argument=1", "tensor<16384x16384xf32>",
                  "required_bytes=1073741824", "allocation_category=weight",
                  "role_capacity_upper_bound_bytes=", "candidate_slices="):
        if field not in stderr:
            raise AssertionError(f"allocation failure missed diagnostic {field}: {stderr}")
    (root / "numerical_cases.json").write_text(json.dumps(numeric_cases, indent=2), encoding="utf-8")
    report = {"status": "passed", "structural_cases": len(cases), "negative_cases": 8,
              "specialization_cases": 2, "numerical_verification": "not_run"}
    (root / "test_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def verify_numerics(root):
    import torch
    import torch_xla

    generator = torch.Generator().manual_seed(71)
    results = []
    for case in json.loads((root / "numerical_cases.json").read_text()):
        inputs = [torch.randn(shape, generator=generator) for shape in case["input_shapes"]]
        original = torch_xla._XLAC._run_stablehlo((root / case["original"]).read_bytes(), inputs)
        normalized = torch_xla._XLAC._run_stablehlo((root / case["normalized"]).read_bytes(), inputs)
        if len(original) != len(normalized):
            raise AssertionError("normalization output arity changed")
        errors = []
        for before, after in zip(original, normalized):
            torch.testing.assert_close(before.cpu(), after.cpu(), rtol=0, atol=0)
            errors.append(float((before.cpu() - after.cpu()).abs().max()))
        results.append({"name": case["name"], "status": "passed", "rtol": 0,
                        "atol": 0, "max_absolute_errors": errors})
    report = {"status": "passed", "torch": torch.__version__,
              "torch_xla": torch_xla.__version__, "cases": results}
    (root / "numerical_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def verify_saved_bundle(bundle, original_bytecode, normalized_bytecode, root):
    """Execute native and reserialized graphs with metadata-derived saved state."""
    import numpy as np
    import torch
    import torch_xla
    from torch_xla.stablehlo import _extract_call_parameters

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from ftlpu_frontend.bundle import validate_bundle
    from ftlpu_frontend.xla_compat import load_saved_program
    from ftlpu_frontend.tensor_io import torch_tensor

    started = time.perf_counter()
    manifest = validate_bundle(bundle)
    program = load_saved_program(bundle)
    if len(program._name_to_stablehlo) != 1:
        raise ValueError("saved numerical test currently requires a single captured entry")
    method, function = next(iter(program._name_to_stablehlo.items()))
    if "validation_inputs" in manifest:
        inputs = tuple(torch_tensor(np.load(bundle / spec["file"], allow_pickle=False), spec["dtype"])
                       for spec in manifest["validation_inputs"])
    else:
        inputs = tuple(torch.full(spec["shape"], spec["fill"], dtype=getattr(torch, spec["dtype"]))
                       for spec in manifest["configuration"]["inputs"])
    # Use the exact exporter input_locations/state lookup, never infer parameter
    # order from checkpoint files, state_dict, or similarly sized tensor shapes.
    call_args = _extract_call_parameters(inputs, function.meta, program._bundle)
    call_args = [torch_tensor(value) for value in call_args]
    artifacts = {
        "native_saved": function.bytecode,
        "compiler_original": original_bytecode.read_bytes(),
        "compiler_normalized": normalized_bytecode.read_bytes(),
    }
    values = {}
    timings = {}
    for name, artifact in artifacts.items():
        stage_started = time.perf_counter()
        values[name] = [value.cpu() for value in torch_xla._XLAC._run_stablehlo(artifact, call_args)]
        timings[name] = round(time.perf_counter() - stage_started, 6)
        print(f"XLA {name}: completed ({timings[name]} s)", flush=True)
    checks = []
    for before_name, after_name in (("native_saved", "compiler_original"),
                                   ("compiler_original", "compiler_normalized")):
        before, after = values[before_name], values[after_name]
        if len(before) != len(after):
            raise AssertionError("saved graph output arity changed")
        for index, (expected, actual) in enumerate(zip(before, after)):
            error = float((expected.to(torch.float64) - actual.to(torch.float64)).abs().max())
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)
            checks.append({"before": before_name, "after": after_name, "output": index,
                           "shape": list(actual.shape), "dtype": str(actual.dtype),
                           "max_absolute_error": error, "status": "passed"})
    report = {"status": "passed", "bundle": str(bundle.resolve()), "method": method,
              "hf_loading": False, "torch_capture": False, "checkpoint_reading": False,
              "parameter_binding": "native_exporter_metadata", "argument_count": len(call_args),
              "torch": torch.__version__, "torch_xla": torch_xla.__version__,
              "rtol": 0, "atol": 0, "checks": checks, "timings": timings,
              "elapsed_seconds": round(time.perf_counter() - started, 6)}
    root.mkdir(parents=True, exist_ok=True)
    path = root / "saved_bundle_numerical_report.json"
    if path.exists():
        raise ValueError(f"Refusing to overwrite numerical evidence: {path}")
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verify-numerics", action="store_true")
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--original-bytecode", type=Path)
    parser.add_argument("--normalized-bytecode", type=Path)
    args = parser.parse_args()
    if args.verify_numerics:
        try:
            if args.bundle is not None:
                if args.original_bytecode is None or args.normalized_bytecode is None:
                    parser.error("--bundle requires --original-bytecode and --normalized-bytecode")
                verify_saved_bundle(args.bundle, args.original_bytecode,
                                    args.normalized_bytecode, args.output_dir)
            else:
                verify_numerics(args.output_dir)
        except Exception as error:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            failure = args.output_dir / "numerical_failure_report.json"
            if failure.exists():
                failure = args.output_dir / datetime.now().strftime("numerical-failure-%Y%m%d-%H%M%S-%f.json")
            failure.write_text(json.dumps({"status": "failed", "stage": "xla_numerical_verification",
                "bundle": str(args.bundle) if args.bundle is not None else None,
                "original_bytecode": str(args.original_bytecode),
                "normalized_bytecode": str(args.normalized_bytecode),
                "exception_type": type(error).__name__, "error": str(error),
                "traceback": traceback.format_exc(), "rtol": 0, "atol": 0}, indent=2), encoding="utf-8")
            raise
    else:
        if args.tool is None:
            parser.error("--tool is required for CLI structural tests")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        if any(args.output_dir.iterdir()):
            args.output_dir /= datetime.now().strftime("run-%Y%m%d-%H%M%S-%f")
            args.output_dir.mkdir()
        print(f"Evidence directory: {args.output_dir}", flush=True)
        structural_tests(args.tool, args.output_dir)


if __name__ == "__main__":
    main()
