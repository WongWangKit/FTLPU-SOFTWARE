"""Compile model-independent projection graphs through automatic weight paging.

The fixture provides equations and input bytes only. C++ decides the page
placements, lifetimes and ICU schedule; no model-specific Python planner is used.
This is a compiler regression, not a CModel numerical/performance benchmark.
"""
import argparse
import copy
import json
from pathlib import Path
import subprocess


def write_graph(directory, width):
    directory.mkdir(parents=True, exist_ok=True)
    tokens = 32
    count = tokens * width
    constants = []
    for index in range(4):
        path = directory / f"input{index}.bin"
        path.write_bytes(bytes(count * 2) if index == 0 else bytes([index]) * width**2)
        constants.append(dict(shape=[tokens, width] if index == 0 else [width, width],
                              dtype="bf16" if index == 0 else "i8", file=str(path)))

    def result(node):
        return dict(shape=[tokens, width], dtype="bf16", segments=[dict(
            node=node, output=0, indices=list(range(count)), positions=list(range(count)))])

    nodes = []
    activation = f"tensor<{tokens}x{width}xbf16>"
    weight = f"tensor<{width}x{width}xi8>"
    for index in range(3):
        source = f'''module {{ func.func @stage(%x: {activation}, %w: {weight}) -> {activation} {{
  %y = "ftlpu.kernel.matmul"(%x, %w) {{m = {tokens} : i64, n = {width} : i64,
    k = {width} : i64, unit = "MXM", rhs_scale = 1.000000e-02 : f32}}
    : ({activation}, {weight}) -> {activation}
  return %y : {activation}
}} }}'''
        input_value = (dict(constant=0, shape=[tokens, width], dtype="bf16")
                       if index == 0 else result(index - 1))
        nodes.append(dict(name=f"projection{index}", kernel=source,
                          inputs=[input_value, dict(constant=index + 1,
                                  shape=[width, width], dtype="i8")],
                          outputs=[dict(shape=[tokens, width], dtype="bf16")]))
    path = directory / "graph.json"
    # Keep all three outputs live, so placement checks exercise retained values
    # as well as the immediate producer/consumer pair.
    path.write_text(json.dumps(dict(format="ftlpu.logical-kernel-graph.v1",
        constants=constants, nodes=nodes, outputs=[result(i) for i in range(3)])), encoding="utf-8")
    return path


def write_vector_graph(directory, width):
    """A graph with arithmetic but no pageable weights needs no page slots."""
    directory.mkdir(parents=True, exist_ok=True)
    shape = [32, width]
    count = 32 * width
    path = directory / "input0.bin"
    path.write_bytes(bytes(count * 2))
    constants = [dict(shape=shape, dtype="bf16", file=str(path))]
    ty = f"tensor<32x{width}xbf16>"
    source = f'''module {{ func.func @stage(%a: {ty}) -> {ty} {{
  %o = ftlpu.kernel.vector_alu %a, %a {{kind = "add", use_immediate = true,
    immediate = 1.000000e+00 : f32, reduce = false}} : ({ty}, {ty}) -> {ty}
  return %o : {ty}
}} }}'''
    node = dict(name="vector_add", kernel=source,
                inputs=[dict(constant=0, shape=shape, dtype="bf16")],
                outputs=[dict(shape=shape, dtype="bf16")])
    result = dict(shape=shape, dtype="bf16", segments=[dict(node=0, output=0,
                  indices=list(range(count)), positions=list(range(count)))])
    path = directory / "graph.json"
    path.write_text(json.dumps(dict(format="ftlpu.logical-kernel-graph.v1",
        constants=constants, nodes=[node], outputs=[result])), encoding="utf-8")
    return path


def write_conv_graph(directory, width):
    """Conv creates a compiler view node; its BF16 kernel is resident, not paged."""
    directory.mkdir(parents=True, exist_ok=True)
    shape, weight_shape = [32, width], [width, 4]
    count = 32 * width
    constants = []
    for index, dimensions in enumerate((shape, weight_shape)):
        path = directory / f"input{index}.bin"
        path.write_bytes(bytes(dimensions[0] * dimensions[1] * 2))
        constants.append(dict(shape=dimensions, dtype="bf16", file=str(path)))
    activation = f"tensor<32x{width}xbf16>"
    weights = f"tensor<{width}x4xbf16>"
    source = f'''module {{ func.func @stage(%x: {activation}, %w: {weights}) -> {activation} {{
  %o = ftlpu.kernel.depthwise_conv1d %x, %w {{kernel_size = 4 : i64,
    causal = true, activation = "silu"}} : ({activation}, {weights}) -> {activation}
  return %o : {activation}
}} }}'''
    node = dict(name="conv1d_silu", kernel=source,
                inputs=[dict(constant=i, shape=dims, dtype="bf16")
                        for i, dims in enumerate((shape, weight_shape))],
                outputs=[dict(shape=shape, dtype="bf16")])
    result = dict(shape=shape, dtype="bf16", segments=[dict(node=0, output=0,
                  indices=list(range(count)), positions=list(range(count)))])
    path = directory / "graph.json"
    path.write_text(json.dumps(dict(format="ftlpu.logical-kernel-graph.v1",
        constants=constants, nodes=[node], outputs=[result])), encoding="utf-8")
    return path


def run(command, log, expected_error=None):
    completed = subprocess.run([str(x) for x in command], stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True)
    log.write_text(completed.stdout, encoding="utf-8")
    if expected_error is not None:
        assert completed.returncode != 0 and expected_error in completed.stdout, completed.stdout[-6000:]
        return
    if completed.returncode:
        raise AssertionError(f"Compiler failed; see {log}\n{completed.stdout[-6000:]}")


def overlaps(left, right):
    if left["bank"] != right["bank"]:
        return False
    if not set(left["slices"]) & set(right["slices"]):
        return False
    return (left["base_row"] < right["base_row"] + right["rows"] and
            right["base_row"] < left["base_row"] + left["rows"])


def validate_plan(plan, depth, bank_count, ping_pong, expected_pages=3):
    assert plan["planning"] == "tensor_subgraph_before_schedule", plan["planning"]
    assert plan["weight_page_ping_pong"] == ping_pong, plan
    assert plan["weight_paging_policy"] == ("auto" if ping_pong else "serialized"), plan
    placements = sorted(plan["weight_page_placements"], key=lambda p: p["consumer_node"])
    assert len(placements) == expected_pages, placements
    banks = [p["bank"] for p in placements]
    if not expected_pages:
        assert not ping_pong and not plan["weight_slot_slices"], plan
        return dict(banks=[], placements=[], nodes=len(plan["nodes"]))
    if ping_pong:
        assert all(bank == banks[index % 2] for index, bank in enumerate(banks)), banks
        assert banks[0] != banks[1], banks
    else:
        assert len(set(banks)) == 1, banks
    prior_by_bank = {}
    for placement in placements:
        assert 0 <= placement["bank"] < bank_count, placement
        assert placement["rows"] > 0 and placement["base_row"] >= 0, placement
        assert placement["base_row"] + placement["rows"] <= depth, placement
        assert len(placement["slices"]) == len(set(placement["slices"])), placement
        assert placement["release_before_node"] <= placement["consumer_node"], placement
        if ping_pong and placement["bank"] in prior_by_bank:
            assert placement["release_before_node"] > prior_by_bank[placement["bank"]], placement
        prior_by_bank[placement["bank"]] = placement["consumer_node"]
    if ping_pong:
        assert 0 < plan["weight_slot_rows"] <= depth // 2, plan
        assert 0 <= plan["weight_slot_base_row"] <= depth - plan["weight_slot_rows"], plan
        reserved = [dict(bank=bank, base_row=plan["weight_slot_base_row"],
                         rows=plan["weight_slot_rows"], slices=plan["weight_slot_slices"])
                    for bank in set(banks)]
        for placement in placements:
            assert placement["rows"] <= plan["weight_slot_rows"], placement
            assert placement["base_row"] == plan["weight_slot_base_row"], placement
            assert set(placement["slices"]) <= set(plan["weight_slot_slices"]), placement
        # Reserved page storage must not alias a retained graph value or a
        # native scratch buffer. This checks row overlap, not just bank names.
        checked = 0
        for node in plan["nodes"]:
            for key in ("direct_output_layouts", "retained_output_layouts", "temporary_workspaces"):
                for layout in node.get(key, []):
                    if not all(k in layout for k in ("bank", "slices", "base_row", "rows")):
                        continue
                    checked += 1
                    for placement in reserved:
                        assert not overlaps(layout, placement), (node["name"], key, layout, placement)
        assert checked, "No graph value layouts were checked"
    return dict(banks=banks, placements=placements, nodes=len(plan["nodes"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("compile", "graph-compile", "target-config", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--cases", nargs="*", help="Optional subset of regression cases")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(dict(status="pending")), encoding="utf-8")
    config = json.loads(args.target_config.read_text(encoding="utf-8-sig"))
    depth = config["mem"]["rows_per_bank"]
    bank_count = config["mem"]["banks_per_slice"]
    assert bank_count >= 2 and depth >= 4096, "Regression requires a two-bank target with >=4096 rows"
    cases = [
        ("default_auto", "", 128, depth, True, 3),
        ("explicit_auto", 'ftlpu.weight_paging_policy = "auto", ', 128, depth, True, 3),
        ("serialized", 'ftlpu.weight_paging_policy = "serialized", ', 128, depth, False, 3),
        ("legacy_enabled", "ftlpu.weight_page_ping_pong = true, ", 128, depth, True, 3),
        ("legacy_disabled", "ftlpu.weight_page_ping_pong = false, ", 128, depth, False, 3),
        ("smaller_sram", "", 128, depth // 2, True, 3),
        ("larger_pages", "", 256, depth, True, 3),
        ("bounded_slot", "ftlpu.weight_page_rows = 64 : i64, ", 256, depth, True, 6),
        ("no_weights", "ftlpu.vector_bf16_spill = true, ", 128, depth, False, 0),
        ("resident_conv", "ftlpu.vector_bf16_spill = true, ftlpu.sxm_wide = true, ",
         128, depth, False, 0),
        ("one_item_overflow", "ftlpu.weight_page_rows = 16 : i64, ", 256, depth, None, 0),
        ("invalid_policy", 'ftlpu.weight_paging_policy = "unknown", ', 128, depth, None, 0),
        ("invalid_capacity", "ftlpu.weight_page_rows = 0 : i64, ", 128, depth, None, 0),
        ("conflicting_policy", 'ftlpu.weight_paging_policy = "auto", ftlpu.weight_page_ping_pong = false, ',
         128, depth, None, 0),
    ]
    rejected = {
        "one_item_overflow": "one projection reduction item exceeds weight page capacity",
        "invalid_policy": "weight_paging_policy must be auto or serialized",
        "invalid_capacity": "weight_page_rows exceeds graph weight capacity",
        "conflicting_policy": "conflicting weight paging policies",
    }
    records = []
    for name, policy, width, rows, enabled, expected_pages in cases:
        if args.cases is not None and name not in args.cases:
            continue
        directory = out / name
        graph_writer = {"no_weights": write_vector_graph, "resident_conv": write_conv_graph}.get(name, write_graph)
        logical = graph_writer(directory / "logical", width)
        target = copy.deepcopy(config)
        target["mem"]["rows_per_bank"] = rows
        target["memory"]["attention_mask_base_row"] = rows - 256
        target_path = directory / "target.json"
        target_path.write_text(json.dumps(target, indent=2), encoding="utf-8")
        kernel = directory / "graph.kernel.mlir"
        run([args.graph_compile.resolve(), "--input", logical, "--output", kernel,
             "--emit-kernel", "--target-config", target_path], directory / "import.log")
        source = kernel.read_text(encoding="utf-8")
        assert "module {" in source
        kernel.write_text(source.replace("module {", "module attributes {" + policy +
            "ftlpu.common_barrier_protocol = true} {", 1), encoding="utf-8")
        run([args.compile.resolve(), "--input", kernel, "--input-stage", "kernel",
             "--output", directory / "program.ftlpu", "--target-config", target_path,
             "--icu-compression", "raw-macro", "--mxm-execution", "ws"], directory / "compile.log",
             expected_error=rejected.get(name))
        if name in rejected:
            records.append(dict(case=name, rejection=rejected[name]))
            print(f"{name}: rejected with capacity/policy diagnostic", flush=True)
            continue
        plan = json.loads((directory / "graph_plan.json").read_text(encoding="utf-8"))
        record = validate_plan(plan, rows, bank_count, enabled, expected_pages)
        records.append(dict(case=name, width=width, sram_rows=rows, **record))
        print(f"{name}: banks={record['banks']} SRAM rows={rows}", flush=True)
    assert records, "No regression cases selected"
    (out / "summary.json").write_text(json.dumps(dict(status="passed", cases=records), indent=2), encoding="utf-8")
    print("logical_weight_paging_test passed", flush=True)


if __name__ == "__main__":
    main()
