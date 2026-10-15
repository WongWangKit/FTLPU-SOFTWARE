"""Versioned index over XLA's native save directory and logical ABI checks.

The text reader only recognizes static ranked tensor function signatures. It
does not verify the MLIR operation body; use stablehlo-opt for that check.
"""

from __future__ import annotations

import json
import hashlib
import re
from pathlib import Path

import numpy as np

DTYPES = {"f32": "float32", "f16": "float16", "bf16": "bfloat16", "f64": "float64",
          "i1": "bool", "i8": "int8", "ui8": "uint8", "i16": "int16",
          "i32": "int32", "i64": "int64"}


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def tensor_spec(tensor) -> dict:
    if not hasattr(tensor, "shape") or not hasattr(tensor, "dtype"):
        raise ValueError("entry must use only tensor inputs and outputs")
    if any(type(dim) is not int for dim in tensor.shape):
        raise ValueError("dynamic shapes are not supported by bundle v1")
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype).removeprefix("torch.")}


def capture_signature(exported, model=None) -> dict:
    # Some torch versions retain mutable ATen operations until functionalizing
    # the graph. Inspect that signature too: XLA performs functionalization and
    # then discards mutation outputs. Do not mistake an in-place local tensor
    # operation (e.g. mask construction) for a model-state mutation.
    if any(node.op == "call_function" and getattr(getattr(node.target, "_schema", None), "is_mutable", False)
           for node in exported.graph.nodes):
        functional = exported.run_decompositions({})
        if any(spec.kind.name != "USER_OUTPUT" for spec in functional.graph_signature.output_specs):
            raise ValueError("exporter cannot preserve implicit state/input mutation; "
                             "expose state as explicit tensor inputs/outputs in a thin wrapper")
    aliases = {}
    if model is not None:
        for name, value in list(model.named_parameters(remove_duplicate=False)) + list(
            model.named_buffers(remove_duplicate=False)
        ):
            aliases.setdefault(id(value), []).append(name)
    placeholders = {node.name: node for node in exported.graph.nodes if node.op == "placeholder"}
    inputs = []
    for spec in exported.graph_signature.input_specs:
        node = placeholders[spec.arg.name]
        item = {"kind": spec.kind.name, "name": spec.arg.name, "target": spec.target,
                **tensor_spec(node.meta["val"])}
        if item["kind"] not in ("USER_INPUT", "PARAMETER", "BUFFER", "CONSTANT_TENSOR"):
            raise ValueError(f"unsupported capture input kind: {item['kind']}")
        if item["kind"] != "USER_INPUT":
            if item["kind"] == "BUFFER":
                item["persistent"] = spec.persistent
            in_constants = item["kind"] == "CONSTANT_TENSOR" or (
                item["kind"] == "BUFFER" and item["persistent"] is False)
            states = exported.constants if in_constants else exported.state_dict
            value = states[spec.target]
            if tensor_spec(value) != {key: item[key] for key in ("shape", "dtype")}:
                raise ValueError(f"capture state signature mismatch: {spec.target}")
            item["hf_names"] = aliases.get(id(value), [])
        inputs.append(item)
    nodes = {node.name: node for node in exported.graph.nodes}
    outputs = []
    for spec in exported.graph_signature.output_specs:
        # XLA currently removes BUFFER_MUTATION outputs. Silently losing them
        # would change state semantics. A wrapper can expose state as tensors.
        if spec.kind.name != "USER_OUTPUT":
            raise ValueError(f"exporter cannot preserve {spec.kind.name}; expose state as "
                             "explicit tensor inputs/outputs in a thin wrapper")
        outputs.append({"kind": spec.kind.name, "name": spec.arg.name,
                        **tensor_spec(nodes[spec.arg.name].meta["val"])})
    return {"inputs": inputs, "outputs": outputs,
            "input_pytree": str(exported.call_spec.in_spec),
            "output_pytree": str(exported.call_spec.out_spec)}


def _close(text: str, start: int, left="(", right=")") -> int:
    depth = 0
    for index in range(start, len(text)):
        if text[index] == left:
            depth += 1
        elif text[index] == right:
            depth -= 1
            if depth == 0:
                return index
    raise ValueError("unbalanced IR signature")


def _split(text: str) -> list[str]:
    parts, start, depth = [], 0, 0
    for index, char in enumerate(text):
        if char in "(<{[":
            depth += 1
        elif char in ")>}]":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(text[start:index].strip())
            start = index + 1
    if text[start:].strip():
        parts.append(text[start:].strip())
    return parts


def _ir_tensor(text: str) -> dict:
    match = re.match(r"tensor<([0-9]+x)*([a-z][a-z0-9]*)>", text)
    if not match or match[2] not in DTYPES:
        raise ValueError(f"unsupported IR signature type: {text}")
    fields = match[0][7:-1].split("x")
    return {"shape": [int(dim) for dim in fields[:-1]], "dtype": DTYPES[fields[-1]]}


def ir_signature(text: str) -> dict:
    entries = list(re.finditer(r"func\.func\s+(?:(public|private)\s+)?@([\w.$-]+)\s*\(", text))
    public = [entry for entry in entries if entry[1] != "private"]
    if len(public) != 1:
        raise ValueError("expected exactly one public StableHLO entry function")
    entry = public[0]
    begin = entry.end() - 1
    end = _close(text, begin)
    arguments = []
    for part in _split(text[begin + 1:end]):
        match = re.match(r"%[\w.$-]+\s*:\s*(.*)", part, re.S)
        if not match:
            raise ValueError(f"unsupported IR argument: {part}")
        arguments.append(_ir_tensor(match[1]))
    rest = text[end + 1:].lstrip()
    results = []
    if rest.startswith("->"):
        rest = rest[2:].lstrip()
        if rest.startswith("("):
            results = [_ir_tensor(part) for part in _split(rest[1:_close(rest, 0)])]
        else:
            results = [_ir_tensor(rest)]
    return {"name": entry[2], "inputs": arguments, "outputs": results}


def _file(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"artifact path escapes bundle: {relative}")
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"missing or empty artifact: {relative}")
    return path


def _array_spec(path: Path, expected_dtype=None) -> dict:
    # np.load reads the native NPY header and mmaps payloads. No second full
    # read of the HF-loaded checkpoint, and no custom tensor format parser.
    value = np.load(path, mmap_mode="r", allow_pickle=False)
    if expected_dtype == "bfloat16" and value.dtype.kind == "V" and value.dtype.itemsize == 2:
        return {"shape": list(value.shape), "dtype": "bfloat16",
                "storage_dtype": str(value.dtype), "encoding": "bfloat16_bits"}
    return {"shape": list(value.shape), "dtype": str(value.dtype)}


def _signature(spec: dict) -> dict:
    if spec.get("dynamic_dims") or any(type(dim) is not int for dim in spec["shape"]):
        raise ValueError("dynamic exporter signatures are unsupported in bundle v1")
    return {"shape": spec["shape"], "dtype": spec["dtype"]}


def native_index(root: Path, capture: dict, configured_inputs: list[dict]) -> dict:
    """Join graph targets to actual function argument positions from .meta."""
    states = {item["target"]: item for item in capture["inputs"] if item["kind"] != "USER_INPUT"}
    users = [item for item in capture["inputs"] if item["kind"] == "USER_INPUT"]
    if len(users) != len(configured_inputs):
        raise ValueError("configured input count differs from captured tensor inputs")
    for user, configured in zip(users, configured_inputs):
        if _signature(user) != _signature(configured):
            raise ValueError(f"configured input signature mismatch: {configured['name']}")
    tensors = []
    for target, spec in states.items():
        relative = "data/" + target
        actual = _array_spec(_file(root, relative), spec["dtype"])
        if _signature(actual) != _signature(spec):
            raise ValueError(f"saved parameter shape/dtype mismatch: {target}")
        tensors.append({"name": target, "kind": spec["kind"], "hf_names": spec.get("hf_names", []),
                        "file": relative, **actual, "function_arguments": []})
        if spec["kind"] == "BUFFER":
            tensors[-1]["persistent"] = spec.get("persistent")
    state_files = {path.relative_to(root).as_posix() for path in (root / "data").rglob("*") if path.is_file()}
    if state_files != {item["file"] for item in tensors}:
        raise ValueError("native state inventory differs from captured states; unknown exporter schema")
    constants = []
    if (root / "constants").exists():
        for path in sorted((root / "constants").iterdir(), key=lambda p: p.name):
            if not path.is_file() or not path.name.isdecimal():
                raise ValueError(f"unsupported constant artifact: {path}")
            constants.append({"position": int(path.name), "file": f"constants/{path.name}",
                              **_array_spec(path)})
    if sorted(item["position"] for item in constants) != list(range(len(constants))):
        raise ValueError("native constants must have contiguous indices")
    constants_by_pos = {item["position"]: item for item in constants}
    tensors_by_name = {item["name"]: item for item in tensors}
    functions = []
    for path in sorted((root / "functions").glob("*.meta")):
        meta = json.loads(path.read_text(encoding="utf-8"))
        name = meta["name"]
        if path.name != name + ".meta":
            raise ValueError("metadata filename and function name disagree")
        mlir = f"functions/{name}.mlir"
        bytecode = f"functions/{name}.bytecode"
        _file(root, bytecode)
        abi = ir_signature(_file(root, mlir).read_text(encoding="utf-8"))
        input_specs = [_signature(spec) for spec in meta["input_signature"]]
        output_specs = [_signature(spec) for spec in meta["output_signature"]]
        if abi["inputs"] != input_specs or abi["outputs"] != output_specs:
            raise ValueError(f"exporter metadata disagrees with actual IR signature: {name}")
        if output_specs != [_signature(spec) for spec in capture["outputs"]]:
            raise ValueError("exported outputs differ from captured outputs")
        locations = meta["input_locations"]
        if len(locations) != len(input_specs):
            raise ValueError("input_locations does not cover every function argument")
        arguments, used_users = [], set()
        for position, (location, spec) in enumerate(zip(locations, input_specs)):
            argument = {"position": position, **spec, "source": location}
            kind = location["type_"]
            if kind == "parameter":
                target = location["name"]
                if target not in tensors_by_name:
                    raise ValueError(f"metadata references uncaptured state: {target}")
                tensor = tensors_by_name[target]
                if spec != _signature(tensor):
                    raise ValueError(f"function argument signature differs from parameter: {target}")
                tensor["function_arguments"].append({"function": name, "position": position})
            elif kind == "input_arg":
                index = location["position"]
                if type(index) is not int or not 0 <= index < len(users):
                    raise ValueError(f"invalid user input location: {index}")
                if spec != _signature(users[index]):
                    raise ValueError("function argument differs from captured user input")
                argument["input_name"] = configured_inputs[index]["name"]
                used_users.add(index)
            elif kind == "constant":
                constant = constants_by_pos.get(location["position"])
                if constant is not None and spec["dtype"] == "bfloat16":
                    constant.update(_array_spec(_file(root, constant["file"]), spec["dtype"]))
                if constant is None or spec != _signature(constant):
                    raise ValueError("constant location is missing or has wrong signature")
            else:
                raise ValueError(f"unknown exporter input location kind: {kind}")
            arguments.append(argument)
        unused = set()
        for location, signature in meta.get("unused_inputs", []):
            index = location["position"]
            if location["type_"] != "input_arg" or type(index) is not int or not 0 <= index < len(users):
                raise ValueError("invalid unused input metadata")
            if index in unused or index in used_users or _signature(signature) != _signature(users[index]):
                raise ValueError("inconsistent unused input metadata")
            unused.add(index)
        if used_users | unused != set(range(len(users))):
            raise ValueError("exporter metadata omits user input locations")
        functions.append({"name": name, "ir_entry": abi["name"], "mlir": mlir,
                          "bytecode": bytecode, "metadata": path.relative_to(root).as_posix(),
                          "arguments": arguments, "outputs": output_specs,
                          "unused_input_positions": sorted(unused),
                          "input_pytree": meta.get("input_pytree_spec"),
                          "output_pytree": meta.get("output_pytree_spec"),
                          "artifact_sha256": {relative: hashlib.sha256(_file(root, relative).read_bytes()).hexdigest()
                                              for relative in (mlir, bytecode)},
                          "exporter_stablehlo_version": meta.get("stablehlo_version")})
    if len(functions) != 1:
        raise ValueError("bundle v1 expects one exported function")
    for tensor in tensors:
        tensor["binding"] = "function_argument" if tensor["function_arguments"] else "not_a_function_argument"
        if not tensor["function_arguments"]:
            tensor["binding_reason"] = "not_referenced_by_exporter_metadata"
    return {"functions": functions, "tensors": tensors, "constants": constants,
            "inputs": [{"position": index, "capture_name": user["name"],
                        **{key: config[key] for key in ("name", "shape", "dtype")}}
                       for index, (user, config) in enumerate(zip(users, configured_inputs))],
            "outputs": capture["outputs"]}


def create_manifest(root: Path, config: dict, capture: dict, source: dict,
                    versions: dict, numerics: dict) -> dict:
    manifest = {"schema": "ftlpu.frontend.bundle", "schema_version": 1,
                "exporter": config["exporter"], "model_source": source,
                "configuration": config, "tool_versions": versions, "numerics": numerics,
                "validation_scope": "native tensor headers, logical mapping, text IR signature, graph artifact digests",
                "capture_signature": "capture_signature.json",
                "capture_graph": "capture_graph.txt", **native_index(root, capture, config["inputs"])}
    write_json(root / "manifest.json", manifest)
    return manifest


def validate_bundle(root: Path) -> dict:
    root = Path(root)
    manifest = json.loads(_file(root, "manifest.json").read_text(encoding="utf-8"))
    if manifest["schema"] != "ftlpu.frontend.bundle" or manifest["schema_version"] != 1:
        raise ValueError("unsupported bundle manifest version")
    capture = json.loads(_file(root, manifest["capture_signature"]).read_text(encoding="utf-8"))
    _file(root, manifest["capture_graph"])
    actual = native_index(root, capture, manifest["configuration"]["inputs"])
    for key, value in actual.items():
        if manifest[key] != value:
            raise ValueError(f"manifest {key} differs from native artifacts and capture signature")
    if "backend_profile" in manifest:
        if "capture_archive" in manifest:
            _file(root, manifest["capture_archive"])
        profile = json.loads(_file(root, manifest["backend_profile"]).read_text(encoding="utf-8"))
        captures = {item["target"]: item for item in capture["inputs"] if item["kind"] != "USER_INPUT"}
        for role, description in profile["roles"].items():
            if description["target"] not in captures:
                raise ValueError(f"profile role {role} is absent from captured state")
        for relative in manifest.get("profile_evidence", []):
            _file(root, relative)
    for item in manifest.get("validation_inputs", []):
        if _signature(_array_spec(_file(root, item["file"]), item["dtype"])) != _signature(item):
            raise ValueError(f"saved validation input shape/dtype mismatch: {item['file']}")
    if "input_provenance" in manifest:
        _file(root, manifest["input_provenance"])
    seal_path = root / "bundle_checksums.json"
    if seal_path.exists():
        seal = json.loads(seal_path.read_text(encoding="utf-8"))
        if seal.get("schema") != "ftlpu.frontend.bundle.integrity" or seal.get("schema_version") != 1:
            raise ValueError("unsupported bundle integrity record")
        required = _sealed_files(manifest, seal["verification"])
        if set(seal["files"]) != required:
            raise ValueError("bundle integrity record does not cover all artifacts")
        for relative, digest in seal["files"].items():
            if _sha256(_file(root, relative)) != digest:
                raise ValueError(f"bundle checksum mismatch: {relative}")
    return manifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sealed_files(manifest: dict, verification: dict) -> set[str]:
    paths = {"manifest.json", "configuration.json", manifest["capture_signature"], manifest["capture_graph"]}
    paths.update(item["file"] for item in manifest["tensors"] + manifest["constants"])
    for function in manifest["functions"]:
        paths.update(function[key] for key in ("mlir", "bytecode", "metadata"))
    paths.update(item["file"] for item in verification["reference_outputs"])
    paths.update(item["file"] for item in verification.get("actual_outputs", []))
    if "backend_profile" in manifest:
        paths.add(manifest["backend_profile"])
    if "capture_archive" in manifest:
        paths.add(manifest["capture_archive"])
    if "input_provenance" in manifest:
        paths.add(manifest["input_provenance"])
    paths.update(item["file"] for item in manifest.get("validation_inputs", []))
    paths.update(manifest.get("profile_evidence", []))
    return paths


def seal_bundle(root: Path) -> None:
    """Record content integrity after actual serialized numerical validation.

    This is a checksum file for one artifact, not an export cache. Saved graph,
    parameters and reference outputs can then be reused without HF/capture.
    """
    manifest = validate_bundle(root)
    report = json.loads((root / "report.json").read_text(encoding="utf-8"))
    verification = report.get("numerical_verification", {})
    if verification.get("status") != "passed" or not verification.get("reference_outputs"):
        raise ValueError("cannot seal a bundle without serialized numerical verification and references")
    paths = _sealed_files(manifest, verification)
    write_json(root / "bundle_checksums.json", {
        "schema": "ftlpu.frontend.bundle.integrity", "schema_version": 1,
        "verification": verification, "files": {name: _sha256(_file(root, name)) for name in sorted(paths)},
    })
