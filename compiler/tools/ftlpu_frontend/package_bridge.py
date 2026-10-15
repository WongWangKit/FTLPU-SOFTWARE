"""Connect a captured, explicitly converted decoder bundle to ModelPackage.

The captured StableHLO is the only compiler input. This module reuses the
existing package writer and numerical reference, not its hand-written graph.
Physical placements remain the Command binary/runtime's responsibility.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

import ml_dtypes  # Official BF16 dtype for explicitly declared NPY views.
import numpy as np

from .bundle import validate_bundle, write_json


LINEAR_ROLES = ("query", "key", "value", "output", "gate", "up", "down")
NORM_ROLES = ("input_norm", "post_attention_norm")
BIAS_ROLES = ("query_bias", "key_bias", "value_bias")
PACKAGE_ROLE = {"input_norm": "input_layernorm", "post_attention_norm": "post_attention_layernorm"}


def _artifact(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"missing artifact or path outside bundle: {name}")
    return path


def load_array(path: Path, expected_dtype: str) -> np.ndarray:
    """Restore BF16 only when its logical dtype is explicitly declared.

    NumPy serializes ml_dtypes.bfloat16 with a V2 header. Treating every V2
    payload as BF16 would be ambiguous, so callers must supply the bundle ABI.
    """
    array = np.load(path, allow_pickle=False)
    if expected_dtype == "bfloat16" and array.dtype.kind == "V" and array.dtype.itemsize == 2:
        array = array.view(ml_dtypes.bfloat16)
    if str(array.dtype) != expected_dtype:
        raise ValueError(f"NPY dtype differs from declared {expected_dtype}: {path}: {array.dtype}")
    return array


def validate_profile(profile: dict) -> dict:
    if profile.get("schema_version") != 1 or profile.get("name") != "legacy_w8a16_bf16":
        raise ValueError("bridge requires an explicit legacy_w8a16_bf16 profile v1")
    config = profile["architecture"]
    for key in ("hidden_size", "intermediate_size", "num_attention_heads", "num_key_value_heads"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"invalid architecture {key}")
    sequence = profile["sequence_length"]
    if type(sequence) is not int or sequence <= 0 or sequence % 32:
        raise ValueError("legacy decoder requires a positive sequence length divisible by 32")
    if config.get("model_type") not in ("llama", "qwen2"):
        raise ValueError("legacy numerical reference currently supports standard Llama/Qwen2")
    if profile.get("mask") != "static_causal_no_padding":
        raise ValueError("existing decoder Softmax cannot represent padding or arbitrary masks")
    position = profile.get("position_range")
    if position not in ([0, sequence - 1], list(range(sequence))):
        raise ValueError("existing RoPE initializer requires zero-offset contiguous positions")
    rope = profile["rope"]
    if rope.get("rope_scaling") is not None or config.get("rope_scaling") is not None:
        raise ValueError("legacy RoPE initializer cannot represent rope_scaling")
    hidden = config["hidden_size"]
    heads = config["num_attention_heads"]
    if hidden % heads or heads % config["num_key_value_heads"]:
        raise ValueError("invalid head dimension or GQA grouping")
    dim = hidden // heads
    if dim % 2:
        raise ValueError("RoPE head dimension must be even")
    theta = float(rope["theta"])
    if not np.isfinite(theta) or theta <= 0 or theta != float(config["rope_theta"]):
        raise ValueError("RoPE theta differs from architecture or is invalid")
    actual = np.asarray(rope["inv_freq"], dtype=np.float32)
    expected = 1.0 / (np.float32(theta) ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
    # This checks the captured values, not merely a theta annotation. HF and
    # NumPy pow may differ by a few F32 ULP; tolerance is recorded explicitly.
    if actual.shape != expected.shape or not np.allclose(actual, expected, rtol=2e-7, atol=0):
        raise ValueError("captured inv_freq is not the supported standard RoPE formula")
    roles = profile["roles"]
    required = set(LINEAR_ROLES + NORM_ROLES)
    if not required <= set(roles) or set(roles) - required - set(BIAS_ROLES):
        raise ValueError("profile roles must identify seven Linear weights and two norms")
    bias = set(roles) & set(BIAS_ROLES)
    if bias and bias != set(BIAS_ROLES):
        raise ValueError("legacy decoder package requires complete Q/K/V bias roles")
    if config.get("attention_bias", config["model_type"] == "qwen2") and not bias:
        raise ValueError("profile must preserve checkpoint attention bias")
    targets = [role["target"] for role in roles.values()]
    if len(targets) != len(set(targets)):
        raise ValueError("decoder semantic roles cannot alias one captured tensor")
    if not profile.get("transformations"):
        raise ValueError("converted profile must record actual numerical transformations")
    for role in LINEAR_ROLES:
        value = roles[role]
        if value.get("weight_layout") not in ("in_out", "out_in"):
            raise ValueError(f"explicit Linear storage layout required: {role}")
        scale = float(value["scale"])
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"invalid symmetric quantization scale: {role}")
    return config


def role_arguments(manifest: dict, profile: dict) -> dict[str, int]:
    """Join semantic roles to XLA's actual ABI, never to guessed ordering."""
    roles = profile["roles"]
    function = manifest["functions"][0]
    result = {}
    for role, description in roles.items():
        candidates = [argument["position"] for argument in function["arguments"]
                      if argument["source"].get("type_") == "parameter"
                      and argument["source"].get("name") == description["target"]]
        if len(candidates) != 1:
            raise ValueError(f"role {role} must identify exactly one captured function argument")
        result[PACKAGE_ROLE.get(role, role)] = candidates[0]
    input_name = profile["activation"]["input_name"]
    candidates = [argument["position"] for argument in function["arguments"]
                  if argument.get("input_name") == input_name]
    if len(candidates) != 1:
        raise ValueError("activation must identify exactly one captured user argument")
    result["activation"] = candidates[0]
    if len(set(result.values())) != len(result):
        raise ValueError("role mapping aliases function argument positions")
    return result


def command_bindings(text: str) -> dict[tuple[str, int], dict]:
    """Read only canonical Command binding attributes emitted by our compiler."""
    result = {}
    for match in re.finditer(r"ftlpu\.command\.binding\s*\{", text):
        start, depth = match.end(), 1
        end = start
        while end < len(text) and depth:
            if text[end] == "{": depth += 1
            if text[end] == "}": depth -= 1
            end += 1
        if depth:
            raise ValueError("unbalanced Command binding attributes")
        attributes = text[start:end - 1]
        def field(pattern: str) -> str:
            found = re.search(pattern, attributes)
            if not found:
                raise ValueError(f"missing Command binding field: {pattern}")
            return found[1]
        access = field(r'\baccess\s*=\s*"([^"]+)"')
        # Anonymous internal workspaces use independent role/name namespaces
        # and can share index 0. They are owned by the backend, not package refs.
        if access == "internal":
            continue
        if access not in ("input", "output"):
            raise ValueError(f"unknown Command binding access: {access}")
        index = int(field(r"\bindex\s*=\s*([0-9]+)\s*:"))
        item = {"access": access, "index": index,
                "dtype": field(r'\belement_type\s*=\s*"([^"]+)"'),
                "shape": [int(dim.strip()) for dim in field(r"\bshape\s*=\s*\[([^\]]*)\]").split(",") if dim.strip()],
                "bytes": int(field(r"\bbytes\s*=\s*([0-9]+)\s*:"))}
        key = (access, index)
        if key in result:
            raise ValueError(f"duplicate Command binding {key}")
        result[key] = item
    if not result:
        raise ValueError("Command IR has no typed bindings")
    return result


def logical_binding_map(manifest: dict, profile: dict, text: str, abi_map: dict | None = None) -> dict:
    original = role_arguments(manifest, profile)
    function_name = manifest["functions"][0]["ir_entry"]
    final = original.copy()
    if abi_map is not None:
        if abi_map.get("schema_version") != 1:
            raise ValueError("unsupported compiler ABI map version")
        entries = abi_map["functions"][function_name]
        positions = {item["original_index"]: item["final_index"] for item in entries}
        final_indices = [index for index in positions.values() if index is not None]
        if (len(positions) != len(entries) or set(positions) != set(range(len(manifest["functions"][0]["arguments"])))
                or any(type(index) is not int or index < 0 for index in final_indices)
                or len(set(final_indices)) != len(final_indices)):
            raise ValueError("compiler ABI map does not cover every original argument")
        final = {role: positions[index] for role, index in original.items()}
        if any(index is None for index in final.values()):
            raise ValueError("compiler pruned a required decoder activation/weight")
    bindings = command_bindings(text)
    actual_inputs = {index for access, index in bindings if access == "input"}
    if actual_inputs != set(final.values()):
        raise ValueError(f"Command inputs are not fully covered by logical roles: "
                         f"unmapped={sorted(actual_inputs - set(final.values()))}, "
                         f"missing={sorted(set(final.values()) - actual_inputs)}")
    architecture = profile["architecture"]
    hidden, intermediate = architecture["hidden_size"], architecture["intermediate_size"]
    width = architecture["num_key_value_heads"] * hidden // architecture["num_attention_heads"]
    shapes = {"activation": [profile["sequence_length"], hidden], "input_layernorm": [hidden],
              "post_attention_layernorm": [hidden], "query": [hidden, hidden], "key": [hidden, width],
              "value": [hidden, width], "output": [hidden, hidden], "gate": [hidden, intermediate],
              "up": [hidden, intermediate], "down": [intermediate, hidden],
              "query_bias": [hidden], "key_bias": [width], "value_bias": [width]}
    mapping = []
    for role, index in final.items():
        actual = bindings[("input", index)]
        dtype = "i8" if role in LINEAR_ROLES else "bf16"
        expected_bytes = int(np.prod(shapes[role])) * (1 if dtype == "i8" else 2)
        if actual["shape"] != shapes[role] or actual["dtype"] != dtype or actual["bytes"] != expected_bytes:
            raise ValueError(f"Command binding shape/dtype/bytes mismatch for {role}: {actual}")
        capture_role = next((name for name, package_role in PACKAGE_ROLE.items() if package_role == role), role)
        description = profile["roles"].get(capture_role, {})
        mapping.append({"role": role, "original_argument": original[role],
                        "logical_target": description.get("target"), "hf_name": description.get("hf_name"),
                        "input_name": profile["activation"]["input_name"] if role == "activation" else None,
                        "payload_transform": "transpose" if description.get("weight_layout") == "out_in" else "identity",
                        "binding_index": index, **actual})
    outputs = [item for (access, _), item in bindings.items() if access == "output"]
    if len(outputs) != 1 or outputs[0]["index"] != 0 or outputs[0]["shape"] != shapes["activation"] or outputs[0]["dtype"] != "bf16":
        raise ValueError("decoder Command output must be one BF16 [sequence, hidden] binding 0")
    return {"schema": "ftlpu.frontend.command-bindings", "schema_version": 1,
            "function": function_name, "inputs": mapping, "outputs": outputs,
            "executable_input_bindings": final}


def stage_layer_payload(bundle: Path, profile: dict, output: Path, bindings: dict) -> dict:
    """Materialize existing package inputs from native captured parameters."""
    config = validate_profile(profile)
    manifest = validate_bundle(bundle)
    from import_hf_decoder_layer import decoder_layer_reference, write_bf16
    output.mkdir(parents=True, exist_ok=False)
    tensors = {item["name"]: item for item in manifest["tensors"]}
    arrays = {}
    bound_roles = {item["role"]: item for item in bindings["inputs"]}
    for role, value in profile["roles"].items():
        if value["target"] not in tensors:
            raise ValueError(f"uncaptured profile target: {value['target']}")
        tensor = tensors[value["target"]]
        array = load_array(_artifact(bundle, tensor["file"]), tensor["dtype"])
        if role in LINEAR_ROLES:
            if str(array.dtype) != "int8":
                raise ValueError(f"converted Linear must be INT8: {role}")
            if value["weight_layout"] == "out_in":
                array = array.T
            array = np.ascontiguousarray(array)
        elif str(array.dtype) != "bfloat16":
            raise ValueError(f"converted norm/bias must be BF16: {role}")
        if list(array.shape) != bound_roles[PACKAGE_ROLE.get(role, role)]["shape"]:
            raise ValueError(f"explicit parameter layout disagrees with physical Command binding: {role}")
        arrays[role] = array
    activation = load_array(_artifact(bundle, profile["activation"]["file"]), "bfloat16")
    if str(activation.dtype) != "bfloat16" or list(activation.shape) != [profile["sequence_length"], config["hidden_size"]]:
        raise ValueError("package activation must be the actual BF16 capture input [sequence, hidden]")
    quantized = {role: arrays[role] for role in LINEAR_ROLES}
    scales = {role: float(profile["roles"][role]["scale"]) for role in LINEAR_ROLES}
    biases = {role.removesuffix("_bias"): arrays[role].astype(np.float32) for role in BIAS_ROLES if role in arrays}
    stage_outputs = {}
    golden = decoder_layer_reference(activation.astype(np.float32), arrays["input_norm"].astype(np.float32),
                                     arrays["post_attention_norm"].astype(np.float32), quantized, scales,
                                     config, biases, stage_outputs)
    write_bf16(output / "input.bf16.bin", activation.astype(np.float32))
    write_bf16(output / "golden.bf16.bin", golden)
    for role in NORM_ROLES:
        write_bf16(output / (PACKAGE_ROLE[role] + ".bf16.bin"), arrays[role].astype(np.float32))
    for role in BIAS_ROLES:
        if role in arrays:
            write_bf16(output / (role + ".bf16.bin"), arrays[role].astype(np.float32))
    for role, array in quantized.items():
        array.tofile(output / (role + ".i8.bin"))
    for name, value in stage_outputs.items():
        write_bf16(output / ("golden." + name + ".bf16.bin"), value)
    metadata = {"model": manifest["model_source"].get("model", manifest["model_source"].get("model_id", "captured-hf-decoder")),
                "architecture": config.get("architectures", [config["model_type"]])[0],
                "layer": profile["layer"], "seq_len": profile["sequence_length"],
                "hidden_size": config["hidden_size"], "intermediate_size": config["intermediate_size"],
                "query_heads": config["num_attention_heads"], "kv_heads": config["num_key_value_heads"],
                "head_dim": config["hidden_size"] // config["num_attention_heads"],
                "rope_theta": config["rope_theta"], "rms_norm_eps": config["rms_norm_eps"],
                "scales": scales, "attention_bias_operands": bool(biases), "attention_bias": bool(biases),
                "activation_element_type": "bf16", "executable_input_bindings": bindings["executable_input_bindings"],
                "source_bundle": str(bundle.resolve()), "source_graph_sha256": manifest["functions"][0]["artifact_sha256"]["functions/forward.mlir"]
                if "functions/forward.mlir" in manifest["functions"][0]["artifact_sha256"] else manifest["functions"][0]["artifact_sha256"][manifest["functions"][0]["mlir"]]}
    write_json(output / "metadata.json", metadata)
    return metadata


def compiler_configuration(bundle: Path, manifest: dict, profile: dict, output: Path) -> tuple[Path, Path]:
    """Bind only explicitly declared immutable capture arguments as constants."""
    function = manifest["functions"][0]
    targets = set(profile.get("immutable_targets", []))
    if profile["rope"].get("inv_freq_target"):
        targets.add(profile["rope"]["inv_freq_target"])
    for role in LINEAR_ROLES:
        if profile["roles"][role].get("scale_target"):
            targets.add(profile["roles"][role]["scale_target"])
    role_targets = {item["target"] for item in profile["roles"].values()}
    if targets & role_targets:
        raise ValueError("decoder weights/norms/biases must remain runtime-bound tensors")
    tensors = {item["name"]: item for item in manifest["tensors"]}
    # The graph signature proves lifted CONSTANT_TENSOR state immutable. Native
    # exporter constant locations are also immutable by their explicit schema.
    # BUFFER/PARAMETER state still requires an explicit profile declaration.
    targets.update(item["name"] for item in manifest["tensors"] if item["kind"] == "CONSTANT_TENSOR")
    if targets & role_targets:
        raise ValueError("decoder weights/norms/biases must remain runtime-bound tensors")
    values = []
    def binding(argument: dict, tensor: dict, provenance: dict) -> dict:
        if tensor["dtype"] not in ("float32", "bfloat16", "bool"):
            raise ValueError(f"current compiler immutable specialization does not support {tensor['dtype']}")
        array = load_array(_artifact(bundle, tensor["file"]), tensor["dtype"])
        if list(array.shape) != tensor["shape"] or any(argument[key] != tensor[key] for key in ("shape", "dtype")):
            raise ValueError("immutable artifact disagrees with its actual function argument signature")
        if array.size > 4096:
            raise ValueError(f"immutable capture argument too large for a thin specialization: {tensor['file']}")
        if not np.all(np.isfinite(array.astype(np.float64))):
            raise ValueError(f"non-finite immutable argument: {tensor['file']}")
        contents = array.reshape(-1).tolist() if tensor["dtype"] == "bool" else array.astype(np.float64).reshape(-1).tolist()
        return {"argument_index": argument["position"], "shape": tensor["shape"],
                "dtype": tensor["dtype"], "values": contents, **provenance}
    for target in sorted(targets):
        if target not in tensors:
            raise ValueError(f"immutable target was not captured: {target}")
        arguments = [item for item in function["arguments"] if item["source"].get("type_") == "parameter"
                     and item["source"].get("name") == target]
        # Explicit target can be a captured tensor already folded by XLA.
        if not arguments:
            continue
        if len(arguments) != 1:
            raise ValueError(f"immutable target has multiple argument positions: {target}")
        tensor = tensors[target]
        values.append(binding(arguments[0], tensor, {"logical_target": target, "capture_kind": tensor["kind"]}))
    native = {item["position"]: item for item in manifest["constants"]}
    for argument in function["arguments"]:
        source = argument["source"]
        if source["type_"] == "constant":
            if source["position"] not in native:
                raise ValueError("exporter immutable constant source is absent")
            values.append(binding(argument, native[source["position"]], {"native_constant": source["position"]}))
    if len({item["argument_index"] for item in values}) != len(values):
        raise ValueError("immutable specializations repeat a function argument")
    constants = output / "constant_bindings.json"
    compiler_profile = output / "compiler_profile.json"
    write_json(constants, {"schema_version": 1, "functions": {function["ir_entry"]: values}})
    write_json(compiler_profile, {"schema_version": 1, "profile": profile["name"],
                                 "rope_theta": float(profile["rope"]["theta"])})
    return constants, compiler_profile


def numerical_comparisons(bundle: Path, profile: dict, payload: Path, actual_path: Path) -> dict:
    """Measure distinct errors; acceptance thresholds stay in the old CModel test."""
    from import_hf_decoder_layer import read_bf16
    shape = (profile["sequence_length"], profile["architecture"]["hidden_size"])
    arrays = {"cmodel": read_bf16(actual_path, shape),
              "backend_golden": read_bf16(payload / "golden.bf16.bin", shape)}
    export_report = json.loads(_artifact(bundle, "report.json").read_text(encoding="utf-8"))
    references = export_report.get("numerical_verification", {}).get("reference_outputs", [])
    if len(references) != 1:
        raise ValueError("one serialized framework reference output is required for decoder comparison")
    framework_dtype = references[0].get("dtype", "bfloat16")
    arrays["converted_framework"] = load_array(_artifact(bundle, references[0]["file"]), framework_dtype).astype(np.float32)
    if profile.get("fp32_reference_file"):
        arrays["fp32_framework"] = load_array(_artifact(bundle, profile["fp32_reference_file"]), "float32")
    for name, array in arrays.items():
        if array.shape != shape or not np.all(np.isfinite(array)):
            raise ValueError(f"numerical reference/output has wrong shape or non-finite values: {name}")
    comparisons = {}
    pairs = [("cmodel", "backend_golden"), ("cmodel", "converted_framework"),
             ("backend_golden", "converted_framework")]
    if "fp32_framework" in arrays:
        pairs.extend([("converted_framework", "fp32_framework"), ("backend_golden", "fp32_framework"),
                      ("cmodel", "fp32_framework")])
    for left, right in pairs:
        actual, expected = arrays[left].astype(np.float64), arrays[right].astype(np.float64)
        error = np.abs(actual - expected)
        denominator = np.linalg.norm(expected)
        product = np.linalg.norm(actual) * denominator
        index = int(np.argmax(error))
        comparisons[left + "_vs_" + right] = {
            "elements": int(error.size), "max_absolute_error": float(error.max()), "mean_absolute_error": float(error.mean()),
            "relative_l2_error": float(np.linalg.norm(actual - expected) / denominator) if denominator else None,
            "cosine_similarity": float(np.vdot(actual, expected) / product) if product else None,
            "max_error_flat_index": index, "actual_at_max": float(actual.reshape(-1)[index]),
            "expected_at_max": float(expected.reshape(-1)[index]), "status": "measured"}
    return {"shape": list(shape), "comparisons": comparisons,
            "acceptance": "existing hf_decoder_layer_model_session_test backend golden thresholds, unchanged",
            "fp32_reference": "provided" if "fp32_framework" in arrays else "not provided"}


def run_stage(report: dict, root: Path, name: str, command: list[str], timeout: float = 900) -> None:
    print(f"{name}: {' '.join(command)}", flush=True)
    start = time.perf_counter()
    stage = {"name": name, "command": command, "status": "running"}
    report["active_stage"] = name
    report["stages"].append(stage)
    write_json(root / "report.json", report)
    try:
        completed = subprocess.run(command, capture_output=True, text=True, errors="replace", timeout=timeout)
        (root / (name + ".stdout.txt")).write_text(completed.stdout, encoding="utf-8")
        (root / (name + ".stderr.txt")).write_text(completed.stderr, encoding="utf-8")
        stage["returncode"] = completed.returncode
        if completed.returncode:
            raise RuntimeError(f"{name} failed with exit code {completed.returncode}; see {name}.stderr.txt")
        stage["status"] = "passed"
    except BaseException as error:
        stage.update(status="failed", error=str(error), traceback=traceback.format_exc())
        report["failed_stage"] = name
        if isinstance(error, subprocess.TimeoutExpired):
            for suffix, value in (("stdout", error.stdout), ("stderr", error.stderr)):
                if isinstance(value, bytes):
                    value = value.decode("utf-8", errors="replace")
                (root / (name + "." + suffix + ".txt")).write_text(value or "", encoding="utf-8")
        raise
    finally:
        stage["elapsed_seconds"] = time.perf_counter() - start
        write_json(root / "report.json", report)
        print(f"{name}: {stage['status']} ({stage['elapsed_seconds']:.3f}s)", flush=True)


def lower_decoder_bundle(*, bundle: Path, profile_path: Path, opt: Path, translate: Path,
                         target_config: Path, runtime_test: Path, output: Path,
                         constant_bindings: Path | None = None) -> dict:
    """Run explicit stages and execute the new binary through existing ModelSession."""
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema": "ftlpu.frontend.package-lowering", "schema_version": 1,
              "status": "running", "bundle": str(bundle.resolve()), "profile": str(profile_path.resolve()),
              "target_config": str(target_config.resolve()), "stages": [],
              "tool_versions": {"python": sys.version, "numpy": np.__version__}}
    write_json(output / "report.json", report)
    try:
        report["active_stage"] = "tool_validation"
        for name, path in (("opt", opt), ("translate", translate), ("runtime_test", runtime_test)):
            report["tool_versions"][name] = {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        report["active_stage"] = "bundle_validation"
        manifest = validate_bundle(bundle)
        report["active_stage"] = "profile_validation"
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        validate_profile(profile)
        report["effective_profile"] = profile
        report["numerical_stages"] = {
            "checkpoint_storage": "bundle manifest.numerics.checkpoint_storage (observed Safetensors metadata)",
            "hf_load_dtype": manifest["configuration"]["load_dtype"],
            "explicit_transformations": manifest["numerics"]["transformations"],
            "export_signature_dtypes": manifest["numerics"].get("export_signature_dtypes"),
            "backend_contract": profile.get("numerical_contract"),
            "backend_execution": "not_run",
        }
        report["profile_sha256"] = hashlib.sha256(profile_path.read_bytes()).hexdigest()
        report["target_sha256"] = hashlib.sha256(target_config.read_bytes()).hexdigest()
        write_json(output / "effective_profile.json", profile)
        ir = _artifact(bundle, manifest["functions"][0]["mlir"])
        common = ["--target-config", str(target_config), "--ffn-schedule", "tail", "--mxm-execution", "vector", "--icu-macro-schedule"]
        diagnostic = output / "diagnostics"
        diagnostic.mkdir()
        report["active_stage"] = "compiler_configuration"
        generated_constants, compiler_profile = compiler_configuration(bundle, manifest, profile, output)
        frontend = ["--prune-unused-inputs", "--fold-static-mask", "on", "--kernel-mode", "complete", "--backend-profile", str(compiler_profile),
                    "--constant-bindings", str(constant_bindings or generated_constants)]
        # Each artifact is retained. The first call consumes captured IR only;
        # subsequent reruns also start there to preserve supported CLI contracts.
        for stage in ("kernel", "tensor", "stream"):
            stage_diagnostic = diagnostic / stage
            stage_diagnostic.mkdir()
            run_stage(report, output, stage, [str(opt), "--input", str(ir), "--output", str(output / (stage + ".mlir")),
                                             "--pipeline", "ftlpu-stablehlo-to-" + stage,
                                             "--diagnostic-dir", str(stage_diagnostic)] + common + frontend)
        run_stage(report, output, "schedule", [str(opt), "--input", str(output / "stream.mlir"), "--output", str(output / "schedule.mlir"),
                                               "--pipeline", "ftlpu-stream-to-compressed-schedule",
                                               "--diagnostic-dir", str(diagnostic / "schedule")] + common)
        run_stage(report, output, "schedule_verification", [str(opt), "--input", str(output / "schedule.mlir"),
                  "--output", str(output / "schedule.verified.mlir"), "--pipeline", "ftlpu-verify-schedule",
                  "--diagnostic-dir", str(diagnostic / "schedule_verification")] + common)
        run_stage(report, output, "command", [str(opt), "--input", str(output / "schedule.verified.mlir"), "--output", str(output / "command.mlir"),
                                              "--pipeline", "ftlpu-verified-schedule-to-commands",
                                              "--diagnostic-dir", str(diagnostic / "command")] + common)
        abi_path = diagnostic / "kernel" / "abi_map.json"
        abi_map = json.loads(abi_path.read_text(encoding="utf-8")) if abi_path.exists() else None
        report["active_stage"] = "parameter_mapping_validation"
        mapping = logical_binding_map(manifest, profile, (output / "command.mlir").read_text(encoding="utf-8"), abi_map)
        write_json(output / "command_binding_map.json", mapping)
        golden_dir = output / "package_inputs"
        report["active_stage"] = "backend_golden"
        stage_layer_payload(bundle, profile, golden_dir, mapping)
        executable = output / "decoder.ftlpu"
        run_stage(report, output, "binary", [str(translate), "--input", str(output / "command.mlir"), "--output", str(executable)])
        packager = Path(__file__).resolve().parents[1] / "build_decoder_layer_package.py"
        package = output / "model.ftlpum"
        run_stage(report, output, "package", [sys.executable, str(packager), "--golden-dir", str(golden_dir),
                                              "--executable", str(executable), "--output", str(package)])
        actual_path = output / "cmodel_output.bf16.bin"
        run_stage(report, output, "cmodel", [str(runtime_test), str(package), str(actual_path)])
        report["numerical_stages"]["backend_execution"] = "passed_existing_cmodel_golden_criterion"
        report["active_stage"] = "numerical_comparisons"
        report["numerical_comparisons"] = numerical_comparisons(bundle, profile, golden_dir, actual_path)
        write_json(output / "numerical_comparisons.json", report["numerical_comparisons"])
        report.update(status="passed", package=str(package.resolve()),
                      verification_scope="captured single decoder Command + existing ModelSession/CModel golden thresholds")
    except BaseException as error:
        report.update(status="failed", error=str(error), traceback=traceback.format_exc(),
                      failed_stage=report.get("failed_stage", report.get("active_stage", "bridge")))
        write_json(output / "report.json", report)
        raise
    write_json(output / "report.json", report)
    return report
