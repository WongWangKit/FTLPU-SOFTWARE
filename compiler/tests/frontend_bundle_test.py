"""Logical mapping and native bundle corruption tests; no XLA runtime needed."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from ftlpu_frontend.bundle import create_manifest, ir_signature, seal_bundle, validate_bundle, write_json
from ftlpu_frontend.config import effective_config


@pytest.fixture
def native_bundle(tmp_path):
    inputs = [
        {"name": "x", "shape": [1, 2], "dtype": "float32", "fill": 0},
        {"name": "mask", "shape": [1, 2], "dtype": "float32", "fill": 1},
    ]
    config = effective_config({"entry": "example:factory", "inputs": inputs})
    capture = {
        "inputs": [
            {"kind": "PARAMETER", "name": "p_weight", "target": "model.weight",
             "shape": [2, 3], "dtype": "float32", "hf_names": ["weight", "tied_weight"]},
            {"kind": "PARAMETER", "name": "p_bias", "target": "model.bias",
             "shape": [3], "dtype": "float32"},
            {"kind": "BUFFER", "name": "b_unused", "target": "model.unused",
             "shape": [1], "dtype": "float32"},
            {"kind": "USER_INPUT", "name": "x", "target": None, "shape": [1, 2], "dtype": "float32"},
            {"kind": "USER_INPUT", "name": "mask", "target": None, "shape": [1, 2], "dtype": "float32"},
        ],
        "outputs": [{"kind": "USER_OUTPUT", "name": "add", "shape": [1, 3], "dtype": "float32"}],
    }
    for directory in ("data", "functions", "constants"):
        (tmp_path / directory).mkdir()
    for item in capture["inputs"][:3]:
        with (tmp_path / "data" / item["target"]).open("wb") as stream:
            np.save(stream, np.zeros(item["shape"], dtype=np.float32))
    with (tmp_path / "constants" / "0").open("wb") as stream:
        np.save(stream, np.asarray([2.0], dtype=np.float32))
    meta = {
        "name": "forward", "stablehlo_version": "0.0.0",
        "input_signature": [{"shape": shape, "dtype": "float32", "dynamic_dims": []}
                            for shape in ([3], [1, 2], [2, 3], [1])],
        "output_signature": [{"shape": [1, 3], "dtype": "float32"}],
        # The ABI deliberately differs from the capture and state dict order.
        "input_locations": [
            {"type_": "parameter", "name": "model.bias", "position": -1},
            {"type_": "input_arg", "name": "", "position": 0},
            {"type_": "parameter", "name": "model.weight", "position": -1},
            {"type_": "constant", "name": "", "position": 0},
        ],
        "unused_inputs": [[{"type_": "input_arg", "name": "", "position": 1},
                           {"shape": [1, 2], "dtype": "float32"}]],
    }
    write_json(tmp_path / "functions" / "forward.meta", meta)
    (tmp_path / "functions" / "forward.mlir").write_text(
        "module { func.func public @main(%b: tensor<3xf32>, %x: tensor<1x2xf32>, "
        "%w: tensor<2x3xf32>, %c: tensor<1xf32>) -> tensor<1x3xf32> {\n"
        "%0 = stablehlo.dot_general %x, %w, contracting_dims = [1] x [0], "
        "precision = [DEFAULT, DEFAULT] : (tensor<1x2xf32>, tensor<2x3xf32>) -> tensor<1x3xf32>\n"
        "return %0 : tensor<1x3xf32> } }", encoding="utf-8")
    # An opaque presence fixture, not claimed to be executable bytecode.
    (tmp_path / "functions" / "forward.bytecode").write_bytes(b"fixture")
    write_json(tmp_path / "capture_signature.json", capture)
    write_json(tmp_path / "configuration.json", config)
    (tmp_path / "capture_graph.txt").write_text("fixture", encoding="utf-8")
    create_manifest(tmp_path, config, capture, {}, {}, {"quantization": "none", "transformations": []})
    return tmp_path


def test_mapping_uses_exporter_locations(native_bundle):
    manifest = validate_bundle(native_bundle)
    states = {item["name"]: item for item in manifest["tensors"]}
    assert states["model.weight"]["function_arguments"] == [{"function": "forward", "position": 2}]
    assert states["model.bias"]["function_arguments"] == [{"function": "forward", "position": 0}]
    assert states["model.unused"]["binding"] == "not_a_function_argument"
    assert states["model.weight"]["hf_names"] == ["weight", "tied_weight"]
    assert manifest["functions"][0]["unused_input_positions"] == [1]
    assert manifest["outputs"][0]["shape"] == [1, 3]


@pytest.mark.parametrize("corruption", ["missing", "shape", "dtype", "ir_input", "ir_output",
                                        "location", "unused", "manifest", "constant", "extra_state", "bytecode"])
def test_corrupt_bundle_rejected(native_bundle, corruption):
    root = native_bundle
    if corruption == "missing":
        (root / "functions" / "forward.bytecode").unlink()
    elif corruption == "bytecode":
        (root / "functions" / "forward.bytecode").write_bytes(b"different-bytecode")
    elif corruption in ("shape", "dtype"):
        with (root / "data" / "model.weight").open("wb") as stream:
            np.save(stream, np.zeros([3, 2] if corruption == "shape" else [2, 3],
                                     dtype=np.float16 if corruption == "dtype" else np.float32))
    elif corruption.startswith("ir_"):
        path = root / "functions" / "forward.mlir"
        text = path.read_text(encoding="utf-8")
        text = text.replace("tensor<3xf32>", "tensor<4xf32>") if corruption == "ir_input" else text.replace(
            "-> tensor<1x3xf32>", "-> tensor<1x3xf16>", 1)
        path.write_text(text, encoding="utf-8")
    elif corruption in ("location", "unused"):
        path = root / "functions" / "forward.meta"
        meta = json.loads(path.read_text())
        if corruption == "location":
            meta["input_locations"][0]["name"] = "missing.bias"
        else:
            meta["unused_inputs"] = []
        write_json(path, meta)
    elif corruption == "manifest":
        path = root / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["tensors"][0]["function_arguments"][0]["position"] = 0
        write_json(path, manifest)
    elif corruption == "constant":
        (root / "constants" / "0").unlink()
    else:
        with (root / "data" / "extra").open("wb") as stream:
            np.save(stream, np.zeros([1]))
    with pytest.raises(ValueError):
        validate_bundle(root)


def test_ir_scalar_multiple_results_and_dynamic_rejection():
    signature = ir_signature("func.func @main(%x: tensor<bf16>) -> (tensor<2xi64>, tensor<f32>) {}")
    assert signature["inputs"] == [{"shape": [], "dtype": "bfloat16"}]
    assert [item["dtype"] for item in signature["outputs"]] == ["int64", "float32"]
    with pytest.raises(ValueError, match="unsupported IR"):
        ir_signature("func.func @main(%x: tensor<?xf32>) -> tensor<?xf32> {}")


@pytest.mark.parametrize("config", [
    {"quantization": "w8a16"}, {"load_options": {"load_in_8bit": True}},
    {"load_options": {"quantization_config": {}}}, {"load_options": {"torch_dtype": "float32"}},
    {"inputs": [{"name": "x", "shape": [0], "dtype": "float32", "fill": 0}]},
    {"export_dtype": "float16"},
])
def test_unsupported_configuration_rejected(config):
    with pytest.raises(ValueError):
        effective_config(config)


def test_legacy_defaults_and_config_only_model_change():
    config = effective_config({})
    assert config["load_dtype"] == "float32" and config["strict"] is False
    assert config["load_options"]["attn_implementation"] == "eager"
    other = effective_config({"model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"})
    expected = copy.deepcopy(config)
    expected["model"] = other["model"]
    assert other == expected


def test_stage_errors_and_existing_work_preserved(tmp_path):
    from ftlpu_frontend.pipeline import export_model
    with pytest.raises(ValueError, match="quantized export"):
        export_model({"quantization": "w8a16"}, tmp_path)
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["status"] == "failed" and report["failed_stage"] == "configuration"
    assert "ValueError" in (tmp_path / "error.txt").read_text()
    before = (tmp_path / "report.json").read_bytes()
    with pytest.raises(ValueError, match="overwrite"):
        export_model({}, tmp_path)
    assert (tmp_path / "report.json").read_bytes() == before


def test_seal_detects_parameter_value_change(native_bundle):
    root = native_bundle
    (root / "validation").mkdir()
    np.save(root / "validation" / "reference_0.npy", np.zeros([1, 3], dtype=np.float32))
    write_json(root / "report.json", {"numerical_verification": {
        "status": "passed", "reference_outputs": [{"file": "validation/reference_0.npy"}],
    }})
    seal_bundle(root)
    validate_bundle(root)
    # A changed value has the same NPY header and IR ABI. Content integrity
    # must still invalidate the previously verified artifact.
    with (root / "data" / "model.weight").open("wb") as stream:
        np.save(stream, np.ones([2, 3], dtype=np.float32))
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_bundle(root)


def test_unverified_bundle_cannot_be_sealed(native_bundle):
    write_json(native_bundle / "report.json", {"numerical_verification": {"status": "not_requested"}})
    with pytest.raises(ValueError, match="without serialized numerical verification"):
        seal_bundle(native_bundle)
    assert not (native_bundle / "bundle_checksums.json").exists()
