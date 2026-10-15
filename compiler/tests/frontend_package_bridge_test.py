"""Captured ABI to reusable ModelPackage binding and semantic rejection tests."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import ml_dtypes
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from build_decoder_layer_package import decoder_input_refs
from ftlpu_frontend.package_bridge import compiler_configuration, load_array, logical_binding_map, numerical_comparisons, role_arguments, run_stage, validate_profile


@pytest.fixture
def profile():
    config = {"model_type": "llama", "hidden_size": 8, "intermediate_size": 16,
              "num_attention_heads": 2, "num_key_value_heads": 1,
              "rope_theta": 10000.0, "rms_norm_eps": 1e-5}
    roles = {name: {"target": "entry." + name, "weight_layout": "in_out", "scale": 0.03125}
             for name in ("query", "key", "value", "output", "gate", "up", "down")}
    roles.update({name: {"target": "entry." + name} for name in ("input_norm", "post_attention_norm")})
    return {"schema_version": 1, "name": "legacy_w8a16_bf16", "architecture": config,
            "layer": 0, "sequence_length": 32, "roles": roles,
            "activation": {"input_name": "hidden_states", "file": "validation/input_0.npy"},
            "mask": "static_causal_no_padding", "position_range": [0, 31],
            "rope": {"theta": 10000.0, "inv_freq": [1.0, 0.01], "rope_scaling": None},
            "transformations": [{"operation": "symmetric_per_tensor_int8", "range": [-127, 127]}]}


def manifest(profile):
    # Deliberately different from both role insertion order and legacy ABI.
    order = ["down", "key", "post_attention_norm", "value", "gate", "input_norm", "up", "query", "output"]
    arguments = [{"position": index, "source": {"type_": "parameter", "name": profile["roles"][role]["target"]}}
                 for index, role in enumerate(order)]
    arguments.insert(4, {"position": 4, "source": {"type_": "input_arg", "position": 0}, "input_name": "hidden_states"})
    for index, argument in enumerate(arguments):
        argument["position"] = index
    return {"functions": [{"ir_entry": "forward", "arguments": arguments}]}


def command(profile, positions):
    shapes = {"activation": [32, 8], "input_layernorm": [8], "post_attention_layernorm": [8],
              "query": [8, 8], "key": [8, 4], "value": [8, 4], "output": [8, 8],
              "gate": [8, 16], "up": [8, 16], "down": [16, 8]}
    text = []
    for role, index in positions.items():
        shape = shapes[role]
        dtype = "bf16" if role in ("activation", "input_layernorm", "post_attention_layernorm") else "i8"
        size = int(np.prod(shape)) * (2 if dtype == "bf16" else 1)
        text.append(f'ftlpu.command.binding {{access = "input", bytes = {size} : i64, element_type = "{dtype}", index = {index} : i64, placement = {{slices = [0, 1]}}, shape = {shape}}}')
    text.append('ftlpu.command.binding {access = "output", bytes = 512 : i64, element_type = "bf16", index = 0 : i64, shape = [32, 8]}')
    return "\n".join(text)


def test_reordered_native_arguments_bind_without_guessing(profile):
    source = manifest(profile)
    positions = role_arguments(source, profile)
    mapping = logical_binding_map(source, profile, command(profile, positions))
    assert mapping["executable_input_bindings"]["activation"] == 4
    assert mapping["executable_input_bindings"]["query"] == 8
    refs = decoder_input_refs({"layer": 7, "attention_bias_operands": False,
                               "executable_input_bindings": mapping["executable_input_bindings"]}, 2)
    assert (8, "layers.7.query.weight") in refs
    assert (4, "hidden.2") in refs


def test_pruned_abi_explicitly_maps_original_positions(profile):
    source = manifest(profile)
    original = role_arguments(source, profile)
    # Simulate one immutable RoPE argument in the middle of captured ABI.
    for argument in source["functions"][0]["arguments"]:
        if argument["position"] >= 3:
            argument["position"] += 1
    source["functions"][0]["arguments"].insert(3, {"position": 3, "source": {"type_": "parameter", "name": "rotary.inv_freq"}})
    entries = [{"original_index": index, "final_index": None if index == 3 else index - (index > 3),
                "specialized_constant": index == 3} for index in range(11)]
    mapping = logical_binding_map(source, profile, command(profile, original), {"schema_version": 1, "functions": {"forward": entries}})
    query = next(item for item in mapping["inputs"] if item["role"] == "query")
    assert query["original_argument"] == 9
    assert query["binding_index"] == 8


@pytest.mark.parametrize("change,match", [
    (lambda p: p.update(mask="padding_mask"), "cannot represent padding"),
    (lambda p: p.update(position_range=[1, 32]), "zero-offset"),
    (lambda p: p["rope"].update(rope_scaling={"factor": 2}), "rope_scaling"),
    (lambda p: p["rope"].update(inv_freq=[1.0, 0.02]), "inv_freq"),
    (lambda p: p["architecture"].update(attention_bias=True), "preserve checkpoint attention bias"),
    (lambda p: p.update(transformations=[]), "record actual numerical"),
    (lambda p: p["roles"]["query"].update(weight_layout="guessed"), "explicit Linear storage"),
])
def test_unsupported_semantics_are_rejected(profile, change, match):
    change(profile)
    with pytest.raises(ValueError, match=match):
        validate_profile(profile)


def test_command_transpose_shape_and_dtype_are_checked(profile):
    source = manifest(profile)
    positions = role_arguments(source, profile)
    text = command(profile, positions)
    # Same element count does not make a transposed binding equivalent.
    with pytest.raises(ValueError, match="shape/dtype/bytes mismatch for key"):
        logical_binding_map(source, profile, text.replace("shape = [8, 4]", "shape = [4, 8]", 1))
    with pytest.raises(ValueError, match="shape/dtype/bytes mismatch"):
        logical_binding_map(source, profile, text.replace('element_type = "i8"', 'element_type = "bf16"', 1))


def test_extra_command_input_fails_mapping_completeness(profile):
    source = manifest(profile)
    text = command(profile, role_arguments(source, profile))
    text += '\nftlpu.command.binding {access = "input", bytes = 4 : i64, element_type = "f32", index = 100 : i64, shape = [1]}'
    with pytest.raises(ValueError, match="not fully covered"):
        logical_binding_map(source, profile, text)


def test_internal_workspace_indices_do_not_enter_package_namespace(profile):
    source = manifest(profile)
    positions = role_arguments(source, profile)
    text = command(profile, positions)
    text += '\nftlpu.command.binding {access = "internal", index = 0 : i64, name = "scratch"}'
    text += '\nftlpu.command.binding {access = "internal", index = 0 : i64, name = "mask"}'
    assert logical_binding_map(source, profile, text)["executable_input_bindings"] == positions


def test_legacy_package_default_indices_are_unchanged():
    no_bias = decoder_input_refs({"layer": 0}, 0)
    assert [index for index, _ in no_bias] == list(range(10))
    with_bias = decoder_input_refs({"layer": 0, "attention_bias_operands": True}, 0)
    assert [index for index, _ in with_bias] == list(range(13))
    assert with_bias[6] == (6, "layers.0.query.bias")


def test_duplicate_or_incomplete_package_bindings_fail_before_writing():
    with pytest.raises(ValueError, match="cover every decoder role"):
        decoder_input_refs({"layer": 0, "executable_input_bindings": {"activation": 0}}, 0)
    mapping = {"activation": 0, "input_layernorm": 1, "query": 2, "key": 3, "value": 4,
               "output": 5, "post_attention_layernorm": 6, "gate": 7, "up": 8, "down": 8}
    with pytest.raises(ValueError, match="unique non-negative"):
        decoder_input_refs({"layer": 0, "executable_input_bindings": mapping}, 0)


def test_bf16_npy_requires_declared_logical_dtype_and_preserves_bits(tmp_path):
    words = np.asarray([0x0000, 0x8000, 0x3F80, 0x7FC1, 0x0001], dtype=np.uint16)
    path = tmp_path / "bf16.npy"
    np.save(path, words.view(ml_dtypes.bfloat16))
    assert np.load(path).dtype.kind == "V"
    np.testing.assert_array_equal(load_array(path, "bfloat16").view(np.uint16), words)
    with pytest.raises(ValueError, match="differs from declared float32"):
        load_array(path, "float32")


def test_three_way_comparison_keeps_every_output(profile, tmp_path):
    from import_hf_decoder_layer import write_bf16
    bundle, payload = tmp_path / "bundle", tmp_path / "payload"
    bundle.mkdir()
    payload.mkdir()
    shape = (32, 8)
    expected = np.ones(shape, dtype=np.float32)
    actual = expected.copy()
    actual[-1, -1] = 2.0
    write_bf16(payload / "golden.bf16.bin", expected)
    actual_path = tmp_path / "actual.bin"
    write_bf16(actual_path, actual)
    np.save(bundle / "framework.npy", expected.astype(ml_dtypes.bfloat16))
    np.save(bundle / "fp32.npy", expected + np.float32(0.125))
    (bundle / "report.json").write_text(json.dumps({"numerical_verification": {
        "reference_outputs": [{"file": "framework.npy", "dtype": "bfloat16"}]}}), encoding="utf-8")
    profile["fp32_reference_file"] = "fp32.npy"
    metrics = numerical_comparisons(bundle, profile, payload, actual_path)["comparisons"]
    assert len(metrics) == 6
    assert metrics["cmodel_vs_backend_golden"]["elements"] == 256
    assert metrics["cmodel_vs_backend_golden"]["max_error_flat_index"] == 255
    assert metrics["cmodel_vs_backend_golden"]["max_absolute_error"] == 1.0
    assert metrics["converted_framework_vs_fp32_framework"]["max_absolute_error"] == 0.125


def test_stage_error_preserves_raw_stderr_traceback_and_phase(tmp_path):
    report = {"stages": []}
    with pytest.raises(RuntimeError, match="exit code 3"):
        run_stage(report, tmp_path, "command", [sys.executable, "-c", "import sys; print('original failure', file=sys.stderr); sys.exit(3)"])
    assert (tmp_path / "command.stderr.txt").read_text() == "original failure\n"
    saved = json.loads((tmp_path / "report.json").read_text())
    assert saved["failed_stage"] == "command"
    assert "RuntimeError" in saved["stages"][0]["traceback"]
    assert saved["stages"][0]["elapsed_seconds"] >= 0


def test_immutable_specialization_uses_capture_kind_and_native_sources(profile, tmp_path):
    bundle, output = tmp_path / "bundle", tmp_path / "output"
    bundle.mkdir()
    output.mkdir()
    tensors = []
    arguments = []
    for index, (target, kind, array) in enumerate([
        ("lifted_mask_min", "CONSTANT_TENSOR", np.asarray(-3.3895313892515355e38, dtype=ml_dtypes.bfloat16)),
        ("small_parameter", "PARAMETER", np.asarray(0.125, dtype=np.float32)),
        ("explicit_inv_freq", "BUFFER", np.asarray([1.0, 0.01], dtype=np.float32)),
    ]):
        np.save(bundle / (target + ".npy"), array)
        item = {"name": target, "kind": kind, "file": target + ".npy", "shape": list(array.shape), "dtype": str(array.dtype)}
        tensors.append(item)
        arguments.append({"position": index, "shape": item["shape"], "dtype": item["dtype"],
                          "source": {"type_": "parameter", "name": target}})
    np.save(bundle / "native_bool.npy", np.asarray(True))
    arguments.append({"position": 3, "shape": [], "dtype": "bool", "source": {"type_": "constant", "position": 0}})
    source = {"functions": [{"ir_entry": "main", "arguments": arguments}], "tensors": tensors,
              "constants": [{"position": 0, "file": "native_bool.npy", "shape": [], "dtype": "bool"}]}
    profile["immutable_targets"] = ["explicit_inv_freq"]
    path, _ = compiler_configuration(bundle, source, profile, output)
    entries = json.loads(path.read_text())["functions"]["main"]
    assert {item["argument_index"] for item in entries} == {0, 2, 3}
    mask = next(item for item in entries if item["argument_index"] == 0)
    assert mask["dtype"] == "bfloat16"
    assert mask["capture_kind"] == "CONSTANT_TENSOR"
    assert next(item for item in entries if item["argument_index"] == 3)["values"] == [True]
    # Same scalar shape does not justify freezing a runtime parameter.
    assert not any(item["argument_index"] == 1 for item in entries)
