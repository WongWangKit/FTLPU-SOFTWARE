"""Sequential frontend stages with original errors and reviewable artifacts."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import re
import subprocess
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

from .bundle import DTYPES, capture_signature, create_manifest, seal_bundle, validate_bundle, write_json
from .config import effective_config


def require_empty_output(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f"Refusing to overwrite non-empty output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def tool_versions() -> dict:
    result = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("torch", "torch-xla", "transformers", "safetensors", "numpy", "ml_dtypes"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


class Report:
    def __init__(self, output: Path, config: dict):
        self.output = output
        self.value = {"status": "running", "output": str(output.resolve()),
                      "requested_configuration": config, "tool_versions": tool_versions(), "stages": []}
        self.save()

    def save(self):
        write_json(self.output / "report.json", self.value)

    def log(self, **fields):
        line = json.dumps(fields, ensure_ascii=False)
        print(line, flush=True)
        with (self.output / "frontend.log").open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")

    @contextmanager
    def stage(self, name):
        item = {"stage": name, "status": "running"}
        self.value["stages"].append(item)
        self.save()
        self.log(stage=name, status="started")
        start = time.perf_counter()
        try:
            yield item
        except BaseException as error:
            item.update(status="failed", error_type=type(error).__name__, error=str(error))
            self.value.update(status="failed", failed_stage=name)
            if name in ("numerical_verification", "backend"):
                key = "backend_verification" if name == "backend" else name
                self.value[key] = {"status": "failed", "error": str(error)}
            trace = traceback.format_exc()
            (self.output / "error.txt").write_text(trace, encoding="utf-8")
            with (self.output / "frontend.log").open("a", encoding="utf-8") as stream:
                stream.write(trace)
            self.log(stage=name, status="failed", error=str(error), traceback="error.txt")
            raise
        else:
            item["status"] = "passed"
        finally:
            item["elapsed_seconds"] = round(time.perf_counter() - start, 6)
            self.save()
            self.log(stage=name, status=item["status"], elapsed_seconds=item["elapsed_seconds"])


def torch_xla_export(exported, output: Path):
    """Keep the original torch.export -> XLA conversion and save defaults."""
    from torch_xla.stablehlo import exported_program_to_stablehlo

    program = exported_program_to_stablehlo(exported)
    # Preserve available IR even when native serialization subsequently fails.
    text = program.get_stablehlo_text()
    if text is not None:
        (output / "conversion.mlir").write_text(text, encoding="utf-8")
    program.save(str(output))
    return program


def _numerics(config, checkpoint, capture, loaded_state_dtypes=None) -> dict:
    loaded = {item["dtype"] for item in capture["inputs"] if item["kind"] in ("PARAMETER", "BUFFER")}
    dtype_names = {"BF16": "bfloat16", "F16": "float16", "F32": "float32"}
    casts = []
    checkpoint_tensors = checkpoint.get("tensors", {})
    for item in capture["inputs"]:
        for name in item.get("hf_names", []):
            stored = checkpoint_tensors.get(name)
            observed_dtype = (loaded_state_dtypes or {}).get(name,
                item["dtype"] if config["quantization"] == "none" else config["load_dtype"])
            if stored and dtype_names.get(stored["dtype"], stored["dtype"]) != observed_dtype:
                casts.append({"parameter": item["target"], "hf_name": name,
                              "from": stored["dtype"], "to": observed_dtype})
    transformations = []
    if casts:
        transformations.append({"stage": "hf_loading", "operation": "dtype_cast", "tensors": casts})
    return {"checkpoint_storage": checkpoint,
            "requested_load_dtype": config["load_dtype"], "observed_state_dtypes": sorted(loaded),
            "quantization": config["quantization"], "transformations": transformations,
            "export_compute_policy": "torch_xla_defaults_preserve_captured_operations"}


def _compare(wrapped, exported, inputs, program, rtol, atol, output=None,
             criterion="strict_torch_close"):
    import torch
    from torch.utils._pytree import tree_flatten

    with torch.inference_mode():
        reference = wrapped(*inputs)
        captured = exported.module()(*inputs)
        converted = program(*inputs)
    reference_values, reference_tree = tree_flatten(reference)
    metrics = {}
    failures = []
    if output is not None:
        import numpy as np
        from .tensor_io import numpy_tensor
        (output / "validation").mkdir(exist_ok=True)
        metrics["reference_outputs"] = []
        metrics["actual_outputs"] = []
        for index, value in enumerate(reference_values):
            array = numpy_tensor(value)
            relative = f"validation/reference_{index}.npy"
            np.save(output / relative, array, allow_pickle=False)
            metrics["reference_outputs"].append({"file": relative, "shape": list(array.shape), "dtype": str(array.dtype)})
    for label, result in (("capture", captured), ("stablehlo", converted)):
        values, tree = tree_flatten(result)
        if tree != reference_tree or len(values) != len(reference_values):
            raise ValueError(f"{label} output structure differs from framework output")
        maximum = 0.0
        for index, (value, expected) in enumerate(zip(values, reference_values)):
            from .tensor_io import torch_tensor
            actual = torch_tensor(value).cpu()
            expected = expected.cpu()
            if output is not None:
                array = numpy_tensor(actual)
                relative = f"validation/{label}_{index}.npy"
                np.save(output / relative, array, allow_pickle=False)
                metrics["actual_outputs"].append({"file": relative, "shape": list(array.shape),
                                                   "dtype": str(array.dtype), "stage": label})
            if label == "stablehlo" and criterion == "legacy_bf16_checkpoint":
                if actual.dtype != torch.bfloat16 or expected.dtype != torch.bfloat16:
                    raise ValueError("existing BF16 criterion applies only to BF16 outputs")
                from .profile_validation import check_legacy_bf16
                observed = check_legacy_bf16(actual.float().numpy(), expected.float().numpy(), raise_on_error=False)
                observed["strict_fp32_tolerance"] = {
                    "rtol": rtol, "atol": atol,
                    "violations": int((~torch.isclose(actual.float(), expected.float(), rtol=rtol, atol=atol)).sum()),
                }
                metrics[label] = observed
                if observed["status"] == "failed":
                    failures.append(f"{label} output {index}: {observed}")
            else:
                torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
            if actual.numel():
                maximum = max(maximum, float((actual.to(torch.float64) - expected.to(torch.float64)).abs().max()))
        if label not in metrics:
            metrics[label] = {"max_absolute_error": maximum, "rtol": rtol, "atol": atol}
    if output is not None:
        write_json(output / "validation/numerical_comparison.json", metrics)
    if failures:
        raise AssertionError("existing BF16 checkpoint criterion failed: " + "; ".join(failures))
    return metrics


def export_model(config: dict, output: Path, *, exporter=None, verify_numerics=False,
                 rtol=1e-4, atol=1e-5, backend_tool=None,
                 backend_pipeline="ftlpu-stablehlo-to-kernel", loaded_model=None,
                 inputs_override=None, input_provenance=None, prepared_entry=None) -> dict:
    """Export one model. An alternate exporter is a callable(ep, output).

    It writes the same native metadata/NPY/MLIR layout and may return a callable
    for numerical verification. No registry or plugin lifecycle is required.
    """
    output = Path(output)
    require_empty_output(output)
    report = Report(output, config)
    report.log(output=str(output.resolve()), tool_versions=report.value["tool_versions"])
    with report.stage("configuration"):
        config = effective_config(config)
        if exporter is None and config["exporter"] != "torch_xla":
            raise ValueError("alternate exporters require export_model(..., exporter=callable)")
        report.value["effective_configuration"] = config
        write_json(output / "configuration.json", config)
        report.log(effective_configuration=config)
    with report.stage("loading"):
        import faulthandler
        timeout_log = (output / "loading_stack.txt").open("w", encoding="utf-8")
        faulthandler.dump_traceback_later(120, repeat=True, file=timeout_log)
        from .hf import load_model, prepare_entry, checkpoint_metadata
        try:
            model = loaded_model if loaded_model is not None else load_model(config)
            checkpoint = checkpoint_metadata(model, config)
            if prepared_entry is None:
                wrapped, inputs = (prepare_entry(model, config) if inputs_override is None
                                   else prepare_entry(model, config, inputs_override))
            else:
                from .hf import validate_inputs
                wrapped = prepared_entry
                if inputs_override is None:
                    raise ValueError("a prepared entry requires explicit validated inputs")
                inputs = validate_inputs(inputs_override, config)
            if input_provenance is not None and hasattr(wrapped, "profile"):
                wrapped.profile["activation"].update(input_provenance)
        finally:
            faulthandler.cancel_dump_traceback_later()
            timeout_log.close()
        source = {"model": config["model"], "requested_revision": config["revision"],
                  "resolved_commit": getattr(model.config, "_commit_hash", None),
                  "model_type": model.config.model_type, "loader": config["loader"],
                  "entry": config["entry"], "entry_options": config["entry_options"]}
    with report.stage("capture"):
        import torch
        with torch.inference_mode():
            exported = torch.export.export(wrapped, inputs, strict=config["strict"])
        (output / "capture_graph.txt").write_text(str(exported.graph_module), encoding="utf-8")
        if config["quantization"] != "none":
            try:
                torch.export.save(exported, output / "capture.pt2")
                report.value["capture_archive"] = {"status": "passed", "file": "capture.pt2"}
            except Exception as error:
                # HF masking utilities may emit a non-serializable, unused
                # functorch initialization node. The native XLA bundle remains
                # self-contained; preserve the actual archive error separately.
                (output / "capture.pt2").unlink(missing_ok=True)
                (output / "capture_archive_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
                report.value["capture_archive"] = {
                    "status": "unavailable", "error": str(error), "traceback": "capture_archive_error.txt"}
        capture = capture_signature(exported, model if config["quantization"] == "none" else wrapped)
        if config["quantization"] != "none":
            roles_by_target = {entry["target"]: entry for entry in wrapped.profile["roles"].values()}
            for item in capture["inputs"]:
                if item["target"] in roles_by_target:
                    item["hf_names"] = [roles_by_target[item["target"]]["hf_name"]]
        for item in capture["inputs"]:
            source_name = getattr(wrapped, "frontend_hf_names", {}).get(item["target"])
            if source_name:
                item["hf_names"] = [source_name]
        write_json(output / "capture_signature.json", capture)
        loaded_state_dtypes = {name: str(value.dtype).removeprefix("torch.")
                               for name, value in model.named_parameters(remove_duplicate=False)}
        numerics = _numerics(config, checkpoint, capture, loaded_state_dtypes)
        numerics["transformations"].extend(getattr(wrapped, "frontend_transformations", []))
        if hasattr(wrapped, "frontend_backend_profile"):
            numerics["backend_numeric_profile"] = wrapped.frontend_backend_profile
            numerics["backend_numeric_contract"] = "existing BF16 RMSNorm backend; arithmetic and rounding differ from HF, explicitly compared"
        if inputs_override is not None:
            from .tensor_io import numpy_tensor
            import numpy as np
            (output / "validation").mkdir(exist_ok=True)
            for index, value in enumerate(inputs):
                np.save(output / "validation" / f"input_{index}.npy", numpy_tensor(value), allow_pickle=False)
            write_json(output / "input_provenance.json", input_provenance or {"source": "explicit_caller_tensor"})
        if config["quantization"] != "none":
            numerics["transformations"].extend(wrapped.profile["transformations"])
            numerics["backend_numeric_contract"] = wrapped.profile["numerical_contract"]
    if config["quantization"] != "none":
        with report.stage("profile_reference"):
            from .profile_validation import save_profile_evidence
            report.value["profile_comparison"] = save_profile_evidence(model, wrapped, inputs, output)
    with report.stage("conversion"):
        has_bf16_state = any(item.get("dtype") == "bfloat16" and item["kind"] != "USER_INPUT"
                            for item in capture["inputs"])
        if exporter is None and has_bf16_state:
            from .xla_compat import export_bf16
            program = export_bf16(exported, output)
            numerics["serialization_adapter"] = {
                "scope": "torch-xla 2.9 BF16 states", "native_save": True,
                "export_weights": False, "state_source": "captured graph signature",
                "encoding": "native NumPy BF16 V2 with explicit capture/exporter dtype",
            }
        else:
            program = (exporter or torch_xla_export)(exported, output)
    with report.stage("bundle_validation"):
        manifest = create_manifest(output, config, capture, source, report.value["tool_versions"], numerics)
        if config["quantization"] != "none":
            manifest["backend_profile"] = "backend_profile.json"
            if (output / "capture.pt2").is_file():
                manifest["capture_archive"] = "capture.pt2"
            manifest["validation_inputs"] = [
                {"file": "validation/input_0.npy", "shape": list(inputs[0].shape), "dtype": "bfloat16"}]
            manifest["profile_evidence"] = [str(path.relative_to(output)).replace("\\", "/")
                                            for path in sorted((output / "validation").iterdir())]
            write_json(output / "manifest.json", manifest)
        if inputs_override is not None:
            manifest["validation_inputs"] = [
                {"file": f"validation/input_{index}.npy", "shape": list(value.shape),
                 "dtype": str(value.dtype).removeprefix("torch.")}
                for index, value in enumerate(inputs)]
            manifest["input_provenance"] = "input_provenance.json"
            write_json(output / "manifest.json", manifest)
        manifest = validate_bundle(output)
        # Describe observed IR ABI dtypes, including mixed precision. There is
        # no fictional global W8A16/export dtype or physical memory layout.
        manifest["numerics"]["export_signature_dtypes"] = sorted({
            spec["dtype"] for func in manifest["functions"]
            for spec in func["arguments"] + func["outputs"]})
        manifest["numerics"]["export_tensor_dtypes"] = sorted({
            DTYPES.get(dtype, dtype) for dtype in re.findall(
                r"tensor<(?:[0-9]+x)*([a-z][a-z0-9]*)>",
                (output / manifest["functions"][0]["mlir"]).read_text(encoding="utf-8"))})
        write_json(output / "manifest.json", manifest)
        report.log(manifest=str((output / "manifest.json").resolve()),
                   functions=manifest["functions"][0]["mlir"],
                   parameters=len(manifest["tensors"]), constants=len(manifest["constants"]))
    report.value["numerical_verification"] = {"status": "not_requested"}
    if verify_numerics:
        with report.stage("numerical_verification"):
            if exporter is None:
                from .xla_compat import load_saved_program
                program = load_saved_program(output)
            if not callable(program):
                raise ValueError("exporter did not provide an executable program for verification")
            report.value["numerical_verification"] = {"status": "passed", **_compare(
                wrapped, exported, inputs, program, rtol, atol, output, config["verification"])}
    report.value["backend_verification"] = {"status": "not_requested"}
    if backend_tool is not None:
        with report.stage("backend") as stage:
            command = [str(backend_tool), "--input", str(output / manifest["functions"][0]["mlir"]),
                       "--output", str(output / "backend.mlir"), "--pipeline", backend_pipeline]
            stage["command"] = command
            result = subprocess.run(command, capture_output=True)
            (output / "backend.stdout.txt").write_bytes(result.stdout)
            (output / "backend.stderr.txt").write_bytes(result.stderr)
            stage["returncode"] = result.returncode
            if result.returncode != 0:
                raise RuntimeError(f"backend failed with exit code {result.returncode}; see backend.stderr.txt")
            if not (output / "backend.mlir").is_file():
                raise RuntimeError("backend reported success without producing backend.mlir")
            report.value["backend_verification"] = {"status": "passed", "pipeline": backend_pipeline}
    if verify_numerics:
        with report.stage("bundle_sealing"):
            seal_bundle(output)
    report.value["status"] = "passed"
    report.save()
    report.log(status="passed", bundle=str(output.resolve()))
    return manifest
