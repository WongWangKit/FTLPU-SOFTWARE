#!/usr/bin/env python3
"""Generates a 320x320 ONNX MatMul and lowers it to IREE flow IR."""

import argparse
import subprocess
import sys
from pathlib import Path


def create_matmul_onnx(path):
    try:
        import onnx
        from onnx import TensorProto, helper
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The ONNX Python package is required for this test. "
            "Install iree-base-compiler[onnx] in the project .venv."
        ) from exc

    a = helper.make_tensor_value_info("a", TensorProto.FLOAT, [320, 320])
    b = helper.make_tensor_value_info("b", TensorProto.FLOAT, [320, 320])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [320, 320])
    node = helper.make_node("MatMul", ["a", "b"], ["y"], name="matmul_320")
    graph = helper.make_graph([node], "matmul_320_graph", [a, b], [y])
    model = helper.make_model(
        graph,
        producer_name="ftlpu-compiler-test",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, path)


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--importer", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--iree-import-onnx", required=True)
    parser.add_argument("--iree-compile", required=True)
    args = parser.parse_args(argv)

    args.work_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = args.work_dir / "matmul_320.onnx"
    flow_ir_path = args.work_dir / "matmul_320.flow.mlir"

    create_matmul_onnx(onnx_path)

    command = [
        sys.executable,
        str(args.importer),
        "--input",
        str(onnx_path),
        "--output",
        str(flow_ir_path),
        "--input-format",
        "onnx",
        "--mode",
        "iree-compile",
        "--iree-stage",
        "flow",
        "--iree-import-onnx",
        args.iree_import_onnx,
        "--iree-compile",
        args.iree_compile,
        "--onnx-opset-version",
        "17",
    ]
    subprocess.run(command, check=True)

    flow_ir = flow_ir_path.read_text(encoding="utf-8")
    required_fragments = [
        "flow.executable",
        "flow.dispatch",
        "tensor<320x320xf32>",
    ]
    missing = [fragment for fragment in required_fragments if fragment not in flow_ir]
    if missing:
        raise AssertionError(f"missing expected IREE IR fragments: {missing}")

    if "linalg.matmul" not in flow_ir and "linalg.generic" not in flow_ir:
        raise AssertionError("expected the lowered flow IR to contain linalg compute")

    print(f"generated ONNX: {onnx_path}")
    print(f"generated IREE flow IR: {flow_ir_path}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
