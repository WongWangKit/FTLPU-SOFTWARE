"""HF interface and torch.export tests; XLA conversion is tested separately."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
from ftlpu_frontend.bundle import capture_signature
from ftlpu_frontend.config import effective_config
from ftlpu_frontend.hf import checkpoint_metadata, load_model, prepare_entry
from ftlpu_frontend.pipeline import _numerics, export_model


@pytest.mark.parametrize("architecture", ["llama", "qwen2"])
def test_hf_config_only_interface_capture_and_numerics(tmp_path, architecture):
    # Small offline checkpoints from real HF implementations; random weights,
    # not claimed to be pretrained SmolLM2/Qwen numerical validation.
    config_class, model_class = (
        (transformers.LlamaConfig, transformers.LlamaForCausalLM) if architecture == "llama"
        else (transformers.Qwen2Config, transformers.Qwen2ForCausalLM))
    source = model_class(config_class(vocab_size=32, hidden_size=32, intermediate_size=48,
                                     num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                                     tie_word_embeddings=architecture == "llama"))
    subfolder = "nested" if architecture == "qwen2" else ""
    source.to(torch.bfloat16).save_pretrained(tmp_path / subfolder, safe_serialization=True)
    config = effective_config({"model": str(tmp_path), "load_options": {"local_files_only": True, "subfolder": subfolder},
                               "inputs": [
                                   {"name": "input_ids", "shape": [1, 4], "dtype": "int64", "fill": 1},
                                   {"name": "attention_mask", "shape": [1, 4], "dtype": "int64", "fill": 1}]})
    model = load_model(config)
    wrapped, inputs = prepare_entry(model, config)
    # Exercise a nontrivial mask; retain causal attention and rotary semantics.
    inputs[1][0, -1] = 0
    with torch.inference_mode():
        exported = torch.export.export(wrapped, inputs, strict=False)
        expected = wrapped(*inputs)
        actual = exported.module()(*inputs)
    torch.testing.assert_close(actual, expected)
    snapshot = capture_signature(exported, model)
    assert snapshot["outputs"][0]["shape"] == [1, 4, 32]
    assert any(item["kind"] == "BUFFER" and item["persistent"] is False
               and item["target"].endswith("rotary_emb.inv_freq") for item in snapshot["inputs"])
    assert any(item.get("hf_names") for item in snapshot["inputs"] if item["kind"] == "PARAMETER")
    if architecture == "qwen2":
        assert any(item["target"].endswith("q_proj.bias") for item in snapshot["inputs"] if item["target"])
    else:
        assert any({"model.embed_tokens.weight", "lm_head.weight"}.issubset(item.get("hf_names", []))
                   for item in snapshot["inputs"])
    numerics = _numerics(config, checkpoint_metadata(model, config), snapshot)
    assert numerics["quantization"] == "none"
    assert numerics["transformations"][0]["operation"] == "dtype_cast"
    assert all(item["from"] == "BF16" and item["to"] == "float32"
               for item in numerics["transformations"][0]["tensors"])
    sub_config = effective_config({"model": str(tmp_path),
                                  "entry": "ftlpu_frontend.hf:submodule_entry",
                                  "entry_options": {"module_path": "model.layers.0.mlp"},
                                  "inputs": [{"name": "hidden_states", "shape": [4, 32],
                                              "dtype": "float32", "fill": 0.125}]})
    child, child_inputs = prepare_entry(model, sub_config)
    assert child is model.model.layers[0].mlp
    with torch.inference_mode():
        child_exported = torch.export.export(child, child_inputs, strict=False)
        torch.testing.assert_close(child_exported.module()(*child_inputs), child(*child_inputs))


def test_implicit_buffer_mutation_rejected():
    class Stateful(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("count", torch.zeros(1))

        def forward(self, value):
            self.count.add_(1)
            return value + self.count

    exported = torch.export.export(Stateful(), (torch.ones(1),), strict=False)
    with pytest.raises(ValueError, match="state|mutation"):
        capture_signature(exported)


def test_implicit_input_mutation_rejected():
    class MutateInput(torch.nn.Module):
        def forward(self, value):
            value.add_(1)
            return value * 2

    exported = torch.export.export(MutateInput(), (torch.ones(1),), strict=False)
    with pytest.raises(ValueError, match="mutation|INPUT"):
        capture_signature(exported)


def test_explicit_tensor_state_is_retained():
    class ExplicitState(torch.nn.Module):
        def forward(self, value, count):
            updated = count + 1
            return value + updated, updated

    exported = torch.export.export(ExplicitState(), (torch.ones(1), torch.zeros(1)), strict=False)
    snapshot = capture_signature(exported)
    assert len(snapshot["inputs"]) == 2 and len(snapshot["outputs"]) == 2


def test_conversion_failure_keeps_capture_and_original_error(tmp_path, monkeypatch):
    from ftlpu_frontend import hf
    model = torch.nn.Linear(2, 3).eval()
    model.config = type("Config", (), {"model_type": "test"})()
    monkeypatch.setattr(hf, "load_model", lambda config: model)
    monkeypatch.setattr(hf, "prepare_entry", lambda model, config: (model, (torch.ones(1, 2),)))
    monkeypatch.setattr(hf, "checkpoint_metadata", lambda model, config: {"status": "unknown"})

    def fail(exported, output):
        raise RuntimeError("original conversion error")

    with pytest.raises(RuntimeError, match="original conversion error"):
        export_model({"entry": "test:factory", "exporter": "failure_fixture",
                      "inputs": [{"name": "x", "shape": [1, 2], "dtype": "float32", "fill": 1}]},
                     tmp_path, exporter=fail)
    assert (tmp_path / "capture_signature.json").is_file()
    assert (tmp_path / "capture_graph.txt").is_file()
    assert not (tmp_path / "manifest.json").exists()
    assert json.loads((tmp_path / "report.json").read_text())["failed_stage"] == "conversion"
    assert "original conversion error" in (tmp_path / "error.txt").read_text()


def test_hf_xla_bundle_and_serialized_numerical_verification(tmp_path, monkeypatch):
    pytest.importorskip("torch_xla")
    checkpoint = tmp_path / "checkpoint"
    source = transformers.LlamaForCausalLM(transformers.LlamaConfig(
        vocab_size=32, hidden_size=32, intermediate_size=48, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2))
    source.save_pretrained(checkpoint, safe_serialization=True)
    manifest = export_model({"model": str(checkpoint), "load_options": {"local_files_only": True},
                             "inputs": [
                                 {"name": "input_ids", "shape": [1, 4], "dtype": "int64", "fill": 1},
                                 {"name": "attention_mask", "shape": [1, 4], "dtype": "int64", "fill": 1}]},
                            tmp_path / "bundle", verify_numerics=True)
    assert manifest["schema_version"] == 1 and manifest["tensors"]
    assert json.loads((tmp_path / "bundle" / "report.json").read_text())["status"] == "passed"
    from ftlpu_frontend import hf
    from tools.run_saved_hf_bundle import run_saved_bundle

    def unexpected(*args, **kwargs):
        raise AssertionError("saved bundle reuse must not load HF or capture a graph")

    monkeypatch.setattr(hf, "load_model", unexpected)
    monkeypatch.setattr(torch.export, "export", unexpected)
    reuse = run_saved_bundle(tmp_path / "bundle")
    assert reuse["status"] == "passed" and reuse["hf_loading"] is False
    assert reuse["torch_capture"] is False and reuse["stablehlo_conversion"] is False
