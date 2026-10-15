"""Official Safetensors reads retain the importer's float32 values bit for bit."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from ftlpu_frontend.weights import SafeTensorStore

safetensors = pytest.importorskip("safetensors")
ml_dtypes = pytest.importorskip("ml_dtypes")
from safetensors.numpy import save_file


def test_all_bf16_patterns_and_supported_storage_dtypes(tmp_path):
    words = np.arange(65536, dtype=np.uint16)
    f16 = np.asarray([0, -0.0, 0.333, -2, np.inf, np.nan], dtype=np.float16)
    f32 = np.asarray([0, -0.0, 1.5, -2, np.inf, np.nan], dtype=np.float32)
    i8 = np.asarray([-128, -127, 0, 127], dtype=np.int8)
    path = tmp_path / "model.safetensors"
    save_file({"bf16": words.view(ml_dtypes.bfloat16), "f16": f16,
               "f32": f32, "i8": i8}, path)
    store = SafeTensorStore(tmp_path)
    assert store.contains("bf16") and not store.contains("absent")
    from import_hf_decoder_layer import SafeTensorStore as LegacyStore
    legacy = LegacyStore(tmp_path)
    expected_bf16 = (words.astype(np.uint32) << 16).view(np.float32)
    for name, expected in [("bf16", expected_bf16), ("f16", f16.astype(np.float32)),
                           ("f32", f32), ("i8", i8.astype(np.float32))]:
        actual = store.read(name)
        assert actual.dtype == np.float32 and actual.shape == expected.shape
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
        np.testing.assert_array_equal(actual.view(np.uint32), legacy.read(name).view(np.uint32))
    assert store.metadata()["bf16"]["dtype"] == "BF16"
    with pytest.raises(KeyError, match="missing"):
        store.read("absent")


def test_shard_routing_and_escaping_path(tmp_path):
    for name, value in (("a", 1.25), ("b", -2.5)):
        save_file({name: np.asarray([value], dtype=np.float32)}, tmp_path / f"{name}.safetensors")
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"a": "a.safetensors", "b": "b.safetensors"}}))
    store = SafeTensorStore(tmp_path)
    assert store.read("a")[0] == 1.25 and store.read("b")[0] == -2.5
    index.write_text(json.dumps({"weight_map": {"a": "../a.safetensors"}}))
    with pytest.raises(ValueError, match="escapes"):
        SafeTensorStore(tmp_path)
