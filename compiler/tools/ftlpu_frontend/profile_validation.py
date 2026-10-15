"""Compare separate HF float reference, transformed HF, and old backend golden."""
import copy

import numpy as np

from import_hf_decoder_layer import bf16_bits, decoder_layer_reference, write_bf16
from .bundle import write_json
from .tensor_io import numpy_tensor


def metrics(actual, expected):
    difference = np.abs(np.asarray(actual, np.float64) - np.asarray(expected, np.float64))
    return {"max_absolute_error": float(difference.max()),
            "mean_absolute_error": float(difference.mean()), "elements": int(difference.size)}


def check_legacy_bf16(actual, expected, *, raise_on_error=True):
    """Exact existing hf_decoder_stack_checkpoint_test.cpp criterion, no knobs."""
    actual = np.asarray(actual, np.float32)
    expected = np.asarray(expected, np.float32)
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("BF16 checkpoint shape differs or contains non-finite values")
    magnitude = np.abs(expected)
    bits = bf16_bits(magnitude).astype(np.uint32)
    following = ((bits + 1) << 16).view(np.float32)
    tolerance = np.maximum(np.float32(0.25), np.float32(4) * (following - magnitude))
    difference = np.abs(actual - expected)
    result = {**metrics(actual, expected), "criterion": "legacy_bf16_checkpoint",
              "source": "runtime/tests/runtime/hf_decoder_stack_checkpoint_test.cpp:36",
              "minimum_absolute_bound": 0.25, "bf16_ulps": 4, "mean_bound": 0.04,
              "violations": int(np.count_nonzero(difference > tolerance)),
              "maximum_tolerance_ratio": float(np.max(difference / tolerance)),
              "maximum_error_flat_index": int(np.argmax(difference)),
              "maximum_tolerance_flat_index": int(np.argmax(difference / tolerance))}
    result["status"] = "failed" if result["violations"] or result["mean_absolute_error"] > 0.04 else "passed"
    if result["status"] == "failed" and raise_on_error:
        raise AssertionError(f"existing BF16 checkpoint criterion failed: {result}")
    return result


def save_profile_evidence(model, wrapped, inputs, output):
    import torch

    profile = wrapped.profile
    directory = output / "validation"
    directory.mkdir(exist_ok=True)
    activation = numpy_tensor(inputs[0])
    np.save(directory / "input_0.npy", activation, allow_pickle=False)
    reference = copy.deepcopy(wrapped)
    decoder_path = profile["roles"]["query"]["hf_name"].split(".self_attn.", 1)[0]
    reference.layer = copy.deepcopy(model.get_submodule(decoder_path)).float().eval()
    reference.rotary = copy.deepcopy(model.get_submodule(
        "model.rotary_emb" if "rotary_path" not in profile else profile["rotary_path"])).eval()
    with torch.inference_mode():
        fp32 = numpy_tensor(reference(inputs[0].float()))
        transformed = numpy_tensor(wrapped(*inputs))
    states = dict(wrapped.named_parameters()) | dict(wrapped.named_buffers())
    weights, scales, biases = {}, {}, {}
    for role in ("query", "key", "value", "output", "gate", "up", "down"):
        entry = profile["roles"][role]
        weights[role] = numpy_tensor(states[entry["target"]])
        scales[role] = entry["scale"]
    for role in ("query", "key", "value"):
        entry = profile["roles"].get(role + "_bias")
        if entry:
            biases[role] = numpy_tensor(states[entry["target"]]).astype(np.float32)
    stage_outputs = {}
    golden = decoder_layer_reference(
        activation.astype(np.float32),
        numpy_tensor(states[profile["roles"]["input_norm"]["target"]]).astype(np.float32),
        numpy_tensor(states[profile["roles"]["post_attention_norm"]["target"]]).astype(np.float32),
        weights, scales, profile["architecture"], biases, stage_outputs,
    )
    np.save(directory / "fp32_reference.npy", fp32, allow_pickle=False)
    np.save(directory / "profile_framework.npy", transformed, allow_pickle=False)
    np.save(directory / "backend_golden.npy", golden, allow_pickle=False)
    write_bf16(directory / "backend_golden.bf16.bin", golden)
    for name, value in stage_outputs.items():
        np.save(directory / f"backend_{name}.npy", value, allow_pickle=False)
    result = {
        "fp32_to_transformed_framework": metrics(transformed, fp32),
        "transformed_framework_to_backend_golden": metrics(golden, transformed),
        "fp32_to_backend_golden": metrics(golden, fp32),
        "bitwise_equivalence_claimed": False,
        "cmodel": "not_executed_at_frontend_export_stage",
        "golden_implementation": "import_hf_decoder_layer.decoder_layer_reference",
    }
    write_json(directory / "profile_comparison.json", result)
    profile["validation"] = result
    write_json(output / "backend_profile.json", profile)
    return result
