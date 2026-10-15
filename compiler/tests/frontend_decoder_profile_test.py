"""Actual HF interfaces and explicit legacy numeric stages, without a checkpoint reread."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
from ftlpu_frontend.config import effective_config
from ftlpu_frontend.hf import prepare_entry
from ftlpu_frontend.profiles import LINEARS, decoder_entry
from ftlpu_frontend.profile_validation import check_legacy_bf16
from ftlpu_frontend.tensor_io import numpy_tensor, torch_tensor
from import_hf_decoder_layer import bf16, quantize_linear


def model(architecture):
    torch.set_num_threads(2)
    config_class, model_class = (
        (transformers.LlamaConfig, transformers.LlamaForCausalLM) if architecture == "llama"
        else (transformers.Qwen2Config, transformers.Qwen2ForCausalLM))
    torch.manual_seed(42)
    config = config_class(vocab_size=48, hidden_size=32, intermediate_size=48,
                          num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                          tie_word_embeddings=architecture == "llama")
    config._attn_implementation = "eager"
    source = model_class(config).eval()
    if architecture == "qwen2":
        with torch.no_grad():
            for path in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"):
                bias = source.model.layers[0].get_submodule(path).bias
                bias.copy_(torch.arange(bias.numel(), dtype=bias.dtype) * 0.001 - 0.03)
    return source


@pytest.mark.parametrize("architecture", ["llama", "qwen2"])
def test_real_hf_profile_capture_and_weight_compatibility(architecture):
    source = model(architecture)
    original = source.model.layers[0]
    embedding_before = source.get_input_embeddings().weight.detach().clone()
    wrapped = decoder_entry(source, {"sequence_length": 32})
    for role, path in LINEARS.items():
        old_weight = original.get_submodule(path).weight.detach().numpy()
        expected, scale = quantize_linear(old_weight)
        actual = wrapped.layer.get_submodule(path)
        np.testing.assert_array_equal(actual.quantized_weight.numpy(), expected)
        assert actual.scale == scale
        assert actual.effective_scale == float(bf16(np.array([scale], np.float32))[0])
        if original.get_submodule(path).bias is not None:
            torch.testing.assert_close(actual.bias, original.get_submodule(path).bias.to(torch.bfloat16), rtol=0, atol=0)
    config = effective_config({
        "entry": "ftlpu_frontend.profiles:decoder_entry", "quantization": "legacy_w8a16_bf16",
        "verification": "legacy_bf16_checkpoint", "entry_options": {"sequence_length": 32},
        "inputs": [{"name": "hidden_states", "shape": [32, 32], "dtype": "bfloat16", "fill": 0}],
    })
    entry, inputs = prepare_entry(source, config)
    with torch.inference_mode():
        captured = torch.export.export(entry, inputs, strict=False)
        torch.testing.assert_close(captured.module()(*inputs), entry(*inputs), rtol=0, atol=0)
    assert entry.profile["mask"] == "static_causal_no_padding"
    assert entry.profile["position_range"] == [0, 31]
    torch.testing.assert_close(source.get_input_embeddings().weight, embedding_before, rtol=0, atol=0)
    if architecture == "llama":
        assert source.get_output_embeddings().weight is source.get_input_embeddings().weight
    else:
        assert {"query_bias", "key_bias", "value_bias"} <= entry.profile["roles"].keys()


@pytest.mark.parametrize("options", [
    {"sequence_length": 32, "position_offset": 1},
    {"sequence_length": 32, "mask": "padding"},
    {"sequence_length": 31},
    {"sequence_length": 32, "use_cache": True},
])
def test_unsupported_state_or_mask_is_not_deleted(options):
    with pytest.raises(ValueError):
        decoder_entry(model("llama"), options)


def test_bf16_npy_requires_declared_dtype(tmp_path):
    value = torch.tensor([1.0, -0.25], dtype=torch.bfloat16)
    path = tmp_path / "value.npy"
    np.save(path, numpy_tensor(value), allow_pickle=False)
    array = np.load(path, allow_pickle=False)
    assert array.dtype.kind == "V"
    with pytest.raises(TypeError):
        torch_tensor(array)
    torch.testing.assert_close(torch_tensor(array, "bfloat16"), value, rtol=0, atol=0)


def test_existing_bf16_criterion_checks_every_element():
    expected = np.zeros(16, np.float32)
    actual = expected.copy()
    actual[-1] = 0.251
    with pytest.raises(AssertionError):
        check_legacy_bf16(actual, expected)
    actual[-1] = 0.25
    assert check_legacy_bf16(actual, expected)["violations"] == 0


def test_explicit_layer_input_is_used_and_validated():
    source = model("llama")
    config = effective_config({"entry": "ftlpu_frontend.profiles:decoder_entry",
        "entry_options": {"sequence_length": 32}, "quantization": "legacy_w8a16_bf16",
        "inputs": [{"name": "hidden_states", "shape": [32, 32], "dtype": "bfloat16", "fill": 0}]})
    actual = torch.full((32, 32), 0.75, dtype=torch.bfloat16)
    _, inputs = prepare_entry(source, config, (actual,))
    assert inputs[0] is actual
    with pytest.raises(ValueError, match="shape/dtype"):
        prepare_entry(source, config, (actual.float(),))


def test_failed_saved_numerics_retains_every_output(tmp_path):
    import json
    from ftlpu_frontend.pipeline import _compare

    class Identity(torch.nn.Module):
        def forward(self, x):
            return x
    wrapped = Identity()
    inputs = (torch.zeros(16, dtype=torch.bfloat16),)
    exported = torch.export.export(wrapped, inputs)
    def bad_program(x):
        result = x.clone()
        result[-1] = 0.5
        return result
    with pytest.raises(AssertionError, match="BF16 checkpoint criterion"):
        _compare(wrapped, exported, inputs, bad_program, 1e-4, 1e-5, tmp_path,
                 "legacy_bf16_checkpoint")
    report = json.loads((tmp_path / "validation/numerical_comparison.json").read_text())
    assert report["stablehlo"]["status"] == "failed"
    assert report["stablehlo"]["violations"] == 1
    assert report["stablehlo"]["maximum_error_flat_index"] == 15
    actual = torch_tensor(np.load(tmp_path / "validation/stablehlo_0.npy"), "bfloat16")
    assert actual.shape == inputs[0].shape and actual[-1] == 0.5
