#!/usr/bin/env python3
"""Exercise the HF structural boundary without checkpoint downloads.

These fixtures represent XLA's last-axis mean and singleton-layout SiLU, not
a substitute model graph. Real decoder and CModel validation are separate.
"""
import argparse
import json
from pathlib import Path
import subprocess


RMS = '''module {
  func.func @main(%x: tensor<32x64xbf16>, %weight: tensor<64xbf16>)
      -> tensor<32x64xbf16>
      attributes {ftlpu.backend_numeric_profile = "legacy_w8a16_bf16"} {
    %xf = stablehlo.convert %x : (tensor<32x64xbf16>) -> tensor<32x64xf32>
    %two = stablehlo.constant dense<2.000000e+00> : tensor<32x64xf32>
    %square = stablehlo.power %xf, %two : tensor<32x64xf32>
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %sum = stablehlo.reduce(%square init: %zero) applies stablehlo.add across dimensions = [1] : (tensor<32x64xf32>, tensor<f32>) -> tensor<32xf32>
    %reciprocal = stablehlo.constant dense<1.562500e-02> : tensor<32xf32>
    %mean = stablehlo.multiply %sum, %reciprocal : tensor<32xf32>
    %epsilon = stablehlo.constant dense<1.000000e-05> : tensor<32xf32>
    %variance = stablehlo.add %mean, %epsilon : tensor<32xf32>
    %inverse = stablehlo.rsqrt %variance : tensor<32xf32>
    %inverse_broadcast = stablehlo.broadcast_in_dim %inverse, dims = [0] : (tensor<32xf32>) -> tensor<32x64xf32>
    %normalized_f32 = stablehlo.multiply %xf, %inverse_broadcast : tensor<32x64xf32>
    %normalized = stablehlo.convert %normalized_f32 : (tensor<32x64xf32>) -> tensor<32x64xbf16>
    %weight_broadcast = stablehlo.broadcast_in_dim %weight, dims = [1] : (tensor<64xbf16>) -> tensor<32x64xbf16>
    %result = stablehlo.multiply %normalized, %weight_broadcast : tensor<32x64xbf16>
    return %result : tensor<32x64xbf16>
  }
}
'''


def ffn_source(repo):
    source = (repo / "compiler/examples/w8a16_ffn/ffn_32x64x128x64_bf16.stablehlo.mlir").read_text()
    source = source.replace("-> tensor<32x64xbf16> {", "-> tensor<32x64xbf16> attributes {ftlpu.backend_numeric_profile = \"legacy_w8a16_bf16\"} {")
    source = source.replace("    %sigmoid =", "    %gate_view = stablehlo.reshape %gate : (tensor<32x128xf32>) -> tensor<1x32x128xf32>\n    %up_view = stablehlo.reshape %up : (tensor<32x128xf32>) -> tensor<1x32x128xf32>\n    %sigmoid =")
    source = source.replace("stablehlo.logistic %gate : tensor<32x128xf32>", "stablehlo.logistic %gate_view : tensor<1x32x128xf32>")
    source = source.replace("stablehlo.multiply %gate, %sigmoid : tensor<32x128xf32>", "stablehlo.multiply %gate_view, %sigmoid : tensor<1x32x128xf32>")
    source = source.replace("stablehlo.multiply %silu, %up : tensor<32x128xf32>", "stablehlo.multiply %silu, %up_view : tensor<1x32x128xf32>\n    %hidden_matrix = stablehlo.reshape %hidden : (tensor<1x32x128xf32>) -> tensor<32x128xf32>")
    return source.replace("dot_general %hidden,", "dot_general %hidden_matrix,")


def singleton_residual_norm():
    # Keep the public ABI rank-2, as the real decoder wrapper does, while HF
    # norm/residual computation retains its singleton model batch dimension.
    source = RMS.replace("%weight: tensor<64xbf16>", "%weight: tensor<64xbf16>, %other: tensor<32x64xbf16>")
    start, body = source.split("    %xf", 1)
    body = "    %xf" + body
    body = body.replace("tensor<32x64x", "tensor<1x32x64x")
    body = body.replace("tensor<32xf32>", "tensor<1x32xf32>")
    body = body.replace("dimensions = [1]", "dimensions = [2]")
    body = body.replace("dims = [0]", "dims = [0, 1]").replace("dims = [1]", "dims = [2]")
    body = body.replace("convert %x :", "convert %residual :")
    body = body.replace("return %result : tensor<1x32x64xbf16>",
        "%matrix = stablehlo.reshape %result : (tensor<1x32x64xbf16>) -> tensor<32x64xbf16>\n    return %matrix : tensor<32x64xbf16>")
    prefix = '''    %x_view = stablehlo.reshape %x : (tensor<32x64xbf16>) -> tensor<1x32x64xbf16>
    %other_view = stablehlo.reshape %other : (tensor<32x64xbf16>) -> tensor<1x32x64xbf16>
    %residual = stablehlo.add %x_view, %other_view : tensor<1x32x64xbf16>
'''
    return start + prefix + body


def rounded_ffn(source):
    source = source.replace("    %gate_view =", '''    %gate_bf16 = stablehlo.convert %gate : (tensor<32x128xf32>) -> tensor<32x128xbf16>
    %gate_f32 = stablehlo.convert %gate_bf16 : (tensor<32x128xbf16>) -> tensor<32x128xf32>
    %up_bf16 = stablehlo.convert %up : (tensor<32x128xf32>) -> tensor<32x128xbf16>
    %up_f32 = stablehlo.convert %up_bf16 : (tensor<32x128xbf16>) -> tensor<32x128xf32>
    %gate_view =''')
    source = source.replace("reshape %gate :", "reshape %gate_f32 :").replace("reshape %up :", "reshape %up_f32 :")
    source = source.replace("    %down =", '''    %down_bf16 = stablehlo.convert %hidden_matrix : (tensor<32x128xf32>) -> tensor<32x128xbf16>
    %down_f32 = stablehlo.convert %down_bf16 : (tensor<32x128xbf16>) -> tensor<32x128xf32>
    %down =''')
    return source.replace("dot_general %hidden_matrix,", "dot_general %down_f32,")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[2]
    ffn = ffn_source(repo)
    rounding = rounded_ffn(ffn)
    cases = [
        ("hf_rms", RMS, True, "ftlpu.kernel.rms_norm"),
        ("hf_singleton_residual_norm", singleton_residual_norm(), True, "ftlpu.kernel.rms_norm"),
        ("hf_ffn_layout", ffn, True, "ftlpu.kernel.swish"),
        ("hf_ffn_profile_rounding", rounding, True, "ftlpu.kernel.swish"),
        ("wrong_mean_factor", RMS.replace("1.562500e-02", "3.125000e-02"), False, None),
        ("wrong_square_power", RMS.replace("2.000000e+00", "3.000000e+00"), False, None),
        ("nonpositive_epsilon", RMS.replace("1.000000e-05", "0.000000e+00"), False, None),
        ("silu_other_input", ffn.replace("logistic %gate_view", "logistic %up_view"), False, None),
        ("explicit_high_precision", ffn.replace("[DEFAULT, DEFAULT]", "[HIGH, HIGH]"), False, None),
        ("unsupported_fp16_rounding", rounding.replace("tensor<32x128xbf16>", "tensor<32x128xf16>"), False, None),
    ]
    report = []
    for name, source, success, marker in cases:
        input_path = args.output_dir / f"{name}.stablehlo.mlir"
        output_path = args.output_dir / f"{name}.kernel.mlir"
        input_path.write_text(source, encoding="utf-8")
        process = subprocess.run([str(args.tool), "--input", str(input_path),
            "--output", str(output_path), "--pipeline", "ftlpu-stablehlo-to-kernel",
            "--kernel-mode", "complete"], capture_output=True, text=True)
        diagnostic = process.stdout + process.stderr
        (args.output_dir / f"{name}.diagnostic.txt").write_text(diagnostic, encoding="utf-8")
        record = {"case": name, "expected_success": success, "returncode": process.returncode}
        report.append(record)
        if (process.returncode == 0) != success:
            raise AssertionError(f"{name}: unexpected result\n{diagnostic}")
        if marker and marker not in output_path.read_text(encoding="utf-8"):
            raise AssertionError(f"{name}: required primitive {marker} missing")
        if success and "stablehlo." in output_path.read_text(encoding="utf-8"):
            raise AssertionError(f"{name}: residual StableHLO computation")
        if not success and "unsupported" not in diagnostic.lower() and "precision" not in diagnostic.lower():
            raise AssertionError(f"{name}: rejection has no structural diagnostic\n{diagnostic}")
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"{len(report)} HF structural cases passed")


if __name__ == "__main__":
    main()
