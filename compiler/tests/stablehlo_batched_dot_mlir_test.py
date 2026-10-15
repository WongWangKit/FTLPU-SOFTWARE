#!/usr/bin/env python3
"""Checks static rank-3 dots preserve batches, dimensions, and transposes."""

import argparse
import subprocess
from pathlib import Path


def tensor(shape, element_type):
    return f"tensor<{'x'.join(str(size) for size in shape)}x{element_type}>"


def source(lhs, rhs, result, element_type, lhs_batch, rhs_batch,
           lhs_contract, rhs_contract, precision="DEFAULT", algorithm=None):
    lhs_type = tensor(lhs, element_type)
    rhs_type = tensor(rhs, element_type)
    result_type = tensor(result, element_type)
    dimensions = "\n".join(
        f"        {name} = [{', '.join(str(axis) for axis in axes)}],"
        for name, axes in (
            ("lhs_batching_dimensions", lhs_batch),
            ("rhs_batching_dimensions", rhs_batch),
            ("lhs_contracting_dimensions", lhs_contract),
            ("rhs_contracting_dimensions", rhs_contract),
        )
    ).rstrip(",")
    attributes = [f'''dot_dimension_numbers = #stablehlo.dot<
{dimensions}
      >''']
    if precision is not None:
        attributes.append(f"precision_config = [#stablehlo<precision {precision}>, "
                          f"#stablehlo<precision {precision}>]")
    if algorithm is not None:
        attributes.append(f"algorithm = {algorithm}")
    attribute_text = ",\n      ".join(attributes)
    return f'''module {{
  func.func @main(%lhs: {lhs_type}, %rhs: {rhs_type}) -> {result_type} {{
    %result = "stablehlo.dot_general"(%lhs, %rhs) {{
      {attribute_text}
    }} : ({lhs_type}, {rhs_type}) -> {result_type}
    return %result : {result_type}
  }}
}}
'''


def run_case(tool, directory, name, text):
    input_path = directory / f"{name}.stablehlo.mlir"
    output_path = directory / f"{name}.kernel.mlir"
    input_path.write_text(text, encoding="utf-8")
    process = subprocess.run([
        str(tool), "--input", str(input_path), "--output", str(output_path),
        "--pipeline", "ftlpu-stablehlo-to-kernel",
        # This test exercises the dot conversion rule, including generic batch
        # matmul which deliberately has no standalone complete Tensor lowering.
        "--kernel-mode", "partial", "--stablehlo-normalize", "off",
    ], text=True, capture_output=True)
    diagnostic = process.stdout + process.stderr
    (directory / f"{name}.diagnostic.txt").write_text(diagnostic, encoding="utf-8")
    return process.returncode, output_path, diagnostic


def check_lowered(path, lhs, rhs, result, element_type, m, n, k,
                  transpose_permutations=()):
    text = path.read_text(encoding="utf-8")
    if "stablehlo.dot_general" in text:
        raise AssertionError(f"{path.name}: dot_general remained")
    matmuls = [line for line in text.splitlines() if '"ftlpu.kernel.matmul"' in line]
    if len(matmuls) != 1:
        raise AssertionError(f"{path.name}: expected exactly one rank-2 matmul")
    expected = [
        f"m = {m} : i64", f"n = {n} : i64", f"k = {k} : i64",
        f"({tensor((m, k), element_type)}, {tensor((k, n), element_type)})"
        f" -> {tensor((m, n), element_type)}",
    ]
    missing = [item for item in expected if item not in matmuls[0]]
    if missing:
        raise AssertionError(f"{path.name}: missing matmul dimensions/types: {missing}")
    signature = next(line for line in text.splitlines() if "func.func @main" in line)
    for value_type in (tensor(lhs, element_type), tensor(rhs, element_type)):
        if value_type not in signature:
            raise AssertionError(f"{path.name}: input ABI changed: {signature}")
    if f"-> {tensor(result, element_type)}" not in signature:
        raise AssertionError(f"{path.name}: output ABI changed: {signature}")
    restored_type = (
        f"({tensor((m, n), element_type)}) -> {tensor(result, element_type)}"
    )
    if not any("ftlpu.kernel.reshape" in line and restored_type in line
               for line in text.splitlines()):
        raise AssertionError(f"{path.name}: result singleton-batch shape was not restored")
    for permutation in transpose_permutations:
        axes = ", ".join(str(axis) for axis in permutation)
        expected_permutation = f"permutation = array<i64: {axes}>"
        if not any("ftlpu.kernel.transpose" in line and expected_permutation in line
                   for line in text.splitlines()):
            raise AssertionError(f"{path.name}: missing transpose {permutation}")


def check_batched(path, lhs, rhs, result, element_type, transpose_rhs):
    text = path.read_text(encoding="utf-8")
    if "stablehlo.dot_general" in text or '"ftlpu.kernel.matmul"' in text:
        raise AssertionError(f"{path.name}: batch dot was flattened or left unlowered")
    batched = [line for line in text.splitlines()
               if " = ftlpu.kernel.batch_matmul " in line]
    if len(batched) != 1:
        raise AssertionError(f"{path.name}: expected exactly one batch matmul")
    expected = [
        'role = "generic"', f"transpose_rhs = {str(transpose_rhs).lower()}",
        f"({tensor(lhs, element_type)}, {tensor(rhs, element_type)})"
        f" -> {tensor(result, element_type)}",
    ]
    missing = [item for item in expected if item not in batched[0]]
    if missing:
        raise AssertionError(f"{path.name}: missing batch matmul attributes/types: {missing}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    successful = [
        # The actual HF/XLA RoPE frequency-times-position outer product.
        ("rope_outer_product", (1, 32, 1), (1, 1, 16), (1, 32, 16),
         "f32", (0,), (0,), (2,), (1,), 32, 16, 1, ()),
        # Unequal M/N/K prevents a missing RHS transpose passing by shape accident.
        ("rhs_transposed", (1, 4, 3), (1, 5, 3), (1, 4, 5),
         "bf16", (0,), (0,), (2,), (2,), 4, 5, 3, ((1, 0),)),
    ]
    for (name, lhs, rhs, result, element_type, lhs_batch, rhs_batch,
         lhs_contract, rhs_contract, m, n, k, permutations) in successful:
        returncode, output_path, diagnostic = run_case(
            args.tool, args.output_dir, name,
            source(lhs, rhs, result, element_type, lhs_batch, rhs_batch,
                   lhs_contract, rhs_contract),
        )
        if returncode != 0:
            raise AssertionError(f"{name}: singleton-batch lowering failed:\n{diagnostic}")
        check_lowered(output_path, lhs, rhs, result, element_type, m, n, k, permutations)

    returncode, output_path, diagnostic = run_case(
        args.tool, args.output_dir, "default_precision_omitted",
        source((1, 4, 3), (1, 3, 5), (1, 4, 5), "f32",
               (0,), (0,), (2,), (1,), precision=None),
    )
    if returncode != 0:
        raise AssertionError(f"default_precision_omitted: lowering failed:\n{diagnostic}")
    check_lowered(output_path, (1, 4, 3), (1, 3, 5), (1, 4, 5), "f32", 4, 5, 3)

    batched = [
        ("batch9_normal", (9, 4, 3), (9, 3, 5), (9, 4, 5), "f32", False),
        ("batch9_rhs_transposed", (9, 4, 3), (9, 5, 3), (9, 4, 5), "bf16", True),
    ]
    for name, lhs, rhs, result, element_type, transpose_rhs in batched:
        returncode, output_path, diagnostic = run_case(
            args.tool, args.output_dir, name,
            source(lhs, rhs, result, element_type, (0,), (0,),
                   (2,), (2,) if transpose_rhs else (1,)),
        )
        if returncode != 0:
            raise AssertionError(f"{name}: batch lowering failed:\n{diagnostic}")
        check_batched(output_path, lhs, rhs, result, element_type, transpose_rhs)

    rejected = [
        ("nonleading_batch", (4, 1, 3), (3, 5, 1), (1, 4, 5),
         (1,), (2,), (2,), (0,), "rank-3 matmul requires static shapes, batch dimension 0"),
        ("rank4", (1, 9, 4, 3), (1, 9, 3, 5), (1, 9, 4, 5),
         (0, 1), (0, 1), (3,), (2,), "only well-formed rank-2 matrix multiplication is supported"),
        ("dynamic_batch", ("?", 4, 3), ("?", 3, 5), ("?", 4, 5),
         (0,), (0,), (2,), (1,), "rank-3 matmul requires static shapes, batch dimension 0"),
        ("multiple_contracting_axes", (1, 4, 3), (1, 4, 3), (1,),
         (0,), (0,), (1, 2), (1, 2), "only well-formed rank-2 matrix multiplication is supported"),
        # Square dimensions previously hid wrong dimension-number interpretation.
        ("rank2_noncanonical_contract", (3, 3), (3, 3), (3, 3),
         (), (), (0,), (1,), "rank-2 matmul requires no batching and contracting dimensions [1] x [0]"),
    ]
    for (name, lhs, rhs, result, lhs_batch, rhs_batch, lhs_contract,
         rhs_contract, expected_diagnostic) in rejected:
        returncode, _, diagnostic = run_case(
            args.tool, args.output_dir, name,
            source(lhs, rhs, result, "f32", lhs_batch, rhs_batch,
                   lhs_contract, rhs_contract),
        )
        if returncode == 0:
            raise AssertionError(f"{name}: unsupported dimensions were silently accepted")
        if expected_diagnostic not in diagnostic:
            raise AssertionError(f"{name}: missing dimension-specific diagnostic:\n{diagnostic}")

    # This algorithm is a valid StableHLO f32 algorithm, copied from the upstream
    # ops_dot_general_algorithms.mlir test. Rejection must come from our guard.
    f32_algorithm = "#stablehlo.dot_algorithm<" + ", ".join([
        "lhs_precision_type = f32", "rhs_precision_type = f32",
        "accumulation_type = f32", "lhs_component_count = 1",
        "rhs_component_count = 1", "num_primitive_operations = 1",
        "allow_imprecise_accumulation = false",
    ]) + ">"
    precision_rejected = [
        ("high_precision", "HIGH", None),
        ("explicit_algorithm", "DEFAULT", f32_algorithm),
    ]
    for name, precision, algorithm in precision_rejected:
        returncode, _, diagnostic = run_case(
            args.tool, args.output_dir, name,
            source((1, 4, 3), (1, 3, 5), (1, 4, 5), "f32",
                   (0,), (0,), (2,), (1,), precision=precision, algorithm=algorithm),
        )
        if returncode == 0:
            raise AssertionError(f"{name}: unsupported precision was silently accepted")
        if "rank-3 matmul supports only DEFAULT precision and no explicit algorithm" not in diagnostic:
            raise AssertionError(f"{name}: missing precision diagnostic:\n{diagnostic}")
    print(f"Passed {len(successful) + 1} singleton-batch, {len(batched)} batch, "
          f"and {len(rejected) + len(precision_rejected)} rejection cases")


if __name__ == "__main__":
    main()
