"""Real HF boundary capture and bit-preserving tied host weight serialization."""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
from ftlpu_frontend.boundaries import final_norm_entry, save_model_boundaries


def model(*, tied=True, dtype=None):
    torch.set_num_threads(2)
    torch.manual_seed(17)
    config = transformers.LlamaConfig(vocab_size=40, hidden_size=32,
                                     intermediate_size=48, num_hidden_layers=1,
                                     num_attention_heads=4, num_key_value_heads=2,
                                     tie_word_embeddings=tied)
    result = transformers.LlamaForCausalLM(config).eval()
    return result if dtype is None else result.to(dtype)


def test_real_hf_final_norm_capture_and_original_unchanged():
    source = model()
    original = source.model.norm.weight
    previous = original.detach().clone()
    wrapped = final_norm_entry(source, {})
    hidden = torch.randn(32, 32).to(torch.bfloat16)
    reference = copy.deepcopy(source.model.norm).to(torch.bfloat16)
    with torch.inference_mode():
        torch.testing.assert_close(wrapped(hidden), reference(hidden), rtol=0, atol=0)
        captured = torch.export.export(wrapped, (hidden,), strict=False)
        torch.testing.assert_close(captured.module()(hidden), wrapped(hidden), rtol=0, atol=0)
    assert wrapped.frontend_hf_names == {"norm.weight": "model.norm.weight"}
    assert wrapped.frontend_backend_profile == "legacy_w8a16_bf16"
    assert wrapped.frontend_transformations[0]["from"] == "float32"
    assert wrapped.frontend_transformations[0]["to"] == "bfloat16"
    assert source.model.norm.weight is original
    assert original.dtype == torch.float32
    torch.testing.assert_close(previous, original, rtol=0, atol=0)


def test_final_norm_already_bf16_and_actual_custom_source():
    source = model(dtype=torch.bfloat16)
    source.extra_norm = copy.deepcopy(source.model.norm)
    wrapped = final_norm_entry(source, {"norm_path": "extra_norm"})
    assert wrapped.frontend_hf_names == {"norm.weight": "extra_norm.weight"}
    assert wrapped.frontend_transformations == []


@pytest.mark.parametrize("value", [torch.zeros(1, 32, 32, dtype=torch.bfloat16),
                                 torch.zeros(32, 16, dtype=torch.bfloat16),
                                 torch.zeros(32, 32, dtype=torch.float32)])
def test_final_norm_rejects_unsupported_shape_or_dtype(value):
    with pytest.raises(ValueError):
        final_norm_entry(model(), {})(value)


def test_saved_tied_boundaries_bits_shapes_and_model_unchanged(tmp_path):
    source = model()
    embedding = source.get_input_embeddings().weight
    original_embedding = embedding.detach().clone()
    original_norm = source.model.norm.weight.detach().clone()
    ids = torch.tensor([0, 39, 3], dtype=torch.int64)
    index = save_model_boundaries(source, tmp_path / "boundaries", ids)
    saved = json.loads((tmp_path / "boundaries" / "index.json").read_text())
    assert saved == index
    assert index["checkpoint_reads"] == 0
    assert index["tied_weights"]["tensor_identity_verified"]
    assert set(index["tied_weights"]["source_hf_names"]) == {"model.embed_tokens.weight", "lm_head.weight"}
    assert index["lm_head"]["last_token_only"] is False
    assert index["outputs"][1]["shape"] == [3, 40]
    assert index["token_ids"]["dtype"] == "int32"
    for name, value in (("embedding", embedding), ("final_norm", source.model.norm.weight)):
        metadata = index[name]
        observed = np.fromfile(tmp_path / "boundaries" / metadata["file"], dtype="<u2")
        expected = value.detach().to(torch.bfloat16).contiguous().view(torch.uint16).numpy()
        np.testing.assert_array_equal(observed, expected.reshape(-1))
        assert metadata["shape"] == list(value.shape)
        assert metadata["bit_roundtrip_verified"]
    np.testing.assert_array_equal(np.fromfile(tmp_path / "boundaries" / "token_ids.i32.bin", dtype="<i4"), ids.numpy())
    assert source.get_input_embeddings().weight is embedding
    assert source.get_output_embeddings().weight is embedding
    torch.testing.assert_close(embedding, original_embedding, rtol=0, atol=0)
    torch.testing.assert_close(source.model.norm.weight, original_norm, rtol=0, atol=0)


def test_already_bf16_records_no_float_boundary_cast(tmp_path):
    index = save_model_boundaries(model(dtype=torch.bfloat16), tmp_path / "boundaries",
                                  np.array([0, 1], dtype=np.int32))
    assert index["transformations"] == []


def test_bf16_negative_zero_and_nan_payload_storage(tmp_path):
    source = model(dtype=torch.bfloat16)
    bits = torch.tensor([0x8000, 0x7FC1, 0xFFC2, 0x0001], dtype=torch.uint16)
    with torch.no_grad():
        source.model.embed_tokens.weight.view(torch.uint16)[0, :4].copy_(bits)
    index = save_model_boundaries(source, tmp_path / "boundaries", [0])
    observed = np.fromfile(tmp_path / "boundaries" / index["embedding"]["file"], dtype="<u2")
    np.testing.assert_array_equal(observed[:4], bits.numpy())
    np.testing.assert_array_equal(source.model.embed_tokens.weight.view(torch.uint16)[0, :4].numpy(), bits.numpy())


def test_unknown_final_norm_option_is_rejected():
    with pytest.raises(ValueError, match="unsupported"):
        final_norm_entry(model(), {"remove_bias": True})


def test_untied_equal_weights_are_rejected_before_writing(tmp_path):
    source = model(tied=False)
    with torch.no_grad():
        source.lm_head.weight.copy_(source.model.embed_tokens.weight)
    with pytest.raises(ValueError, match="tied"):
        save_model_boundaries(source, tmp_path / "untied", [0, 1])
    assert not (tmp_path / "untied").exists()


@pytest.mark.parametrize("ids", [[], [[0, 1]], [-1], [40], [0.5], [True]])
def test_invalid_token_ids_are_rejected_before_writing(tmp_path, ids):
    with pytest.raises(ValueError, match="token_ids"):
        save_model_boundaries(model(), tmp_path / "invalid", ids)
    assert not (tmp_path / "invalid").exists()


def test_refuses_existing_artifacts(tmp_path):
    directory = tmp_path / "boundaries"
    directory.mkdir()
    existing = directory / "unrelated.txt"
    existing.write_text("preserve")
    with pytest.raises(ValueError, match="overwrite"):
        save_model_boundaries(model(), directory, [0, 1])
    assert existing.read_text() == "preserve"
