"""JSON CLI failures and a real offline checkpoint-to-StableHLO export."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[2]
CLI = REPO / "tools" / "export_hf_stablehlo.py"
sys.path.insert(0, str(REPO / "compiler" / "tools"))
from ftlpu_frontend.config import effective_config


@pytest.mark.parametrize("config, message", [
    ([], "JSON object"),
    (None, "JSON object"),
    ({"load_options": None}, "load_options"),
    ({"load_options": []}, "load_options"),
    ({"schema_version": True}, "schema_version"),
    ({"loader": None}, "AutoModel"),
    ({"revision": 12}, "revision"),
    ({"exporter": None}, "exporter"),
    ({"inputs": [None]}, "each input"),
    ({"entry": "example:factory", "inputs": [
        {"name": "x", "shape": [1], "dtype": "float32", "fill": float("nan")}
    ]}, "finite"),
])
def test_malformed_configuration_is_rejected(config, message):
    with pytest.raises(ValueError, match=message):
        effective_config(config)


def run_cli(*arguments):
    return subprocess.run([sys.executable, "-B", str(CLI), *map(str, arguments)],
                          cwd=REPO, capture_output=True, text=True, timeout=180)


@pytest.mark.parametrize("script", ["export_hf_stablehlo.py", "export_smollm2_stablehlo.py"])
def test_help_does_not_import_optional_export_dependencies(script):
    # Even on an XLA host, prove help does not import torch/Transformers.
    code = """
import runpy, sys
sys.modules['torch'] = None
sys.modules['transformers'] = None
sys.modules['torch_xla'] = None
sys.argv = [sys.argv[1], '--help']
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    result = subprocess.run([sys.executable, "-B", "-c", code, str(REPO / "tools" / script)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_cli_reports_bad_json_and_conflicting_modes(tmp_path):
    config = tmp_path / "bad.json"
    config.write_text("{", encoding="utf-8")
    result = run_cli("--config", config, "--output", tmp_path / "bundle")
    assert result.returncode == 2 and "cannot read JSON configuration" in result.stderr
    assert not (tmp_path / "bundle").exists()
    result = run_cli("--config", config, "--validate-bundle", tmp_path)
    assert result.returncode == 2 and "not allowed" in result.stderr


def test_cli_accepts_utf8_bom_and_records_configuration_failure(tmp_path):
    config = tmp_path / "bom.json"
    config.write_text(json.dumps({"load_options": None}), encoding="utf-8-sig")
    bundle = tmp_path / "bundle"
    result = run_cli("--config", config, "--output", bundle)
    assert result.returncode != 0 and "load_options must be an object" in result.stderr
    report = json.loads((bundle / "report.json").read_text())
    assert report["failed_stage"] == "configuration"
    assert [item["stage"] for item in report["stages"]] == ["configuration"]
    assert not (bundle / "manifest.json").exists()


def test_xla_cli_exports_local_checkpoint_from_json(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("torch_xla")
    checkpoint = tmp_path / "local checkpoint"
    torch.manual_seed(7)
    model = transformers.LlamaForCausalLM(transformers.LlamaConfig(
        vocab_size=32, hidden_size=32, intermediate_size=48, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2))
    model.save_pretrained(checkpoint, safe_serialization=True)
    config = tmp_path / "export.json"
    config.write_text(json.dumps({
        "model": str(checkpoint), "load_options": {"local_files_only": True},
        "inputs": [
            {"name": "input_ids", "shape": [1, 4], "dtype": "int64", "fill": 1},
            {"name": "attention_mask", "shape": [1, 4], "dtype": "int64", "fill": 1},
        ],
    }), encoding="utf-8-sig")
    bundle = tmp_path / "stablehlo bundle"
    # The caller supplies a CPU-capable XLA environment; remain offline.
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    result = run_cli("--config", config, "--output", bundle, "--verify-numerics")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "stablehlo." in (bundle / "functions" / "forward.mlir").read_text()
    report = json.loads((bundle / "report.json").read_text())
    assert report["status"] == "passed" and report["numerical_verification"]["status"] == "passed"
    assert (bundle / "bundle_checksums.json").is_file()
    assert run_cli("--validate-bundle", bundle).returncode == 0

    result = run_cli("--config", config, "--output", bundle)
    assert result.returncode != 0 and "overwrite non-empty" in result.stderr
