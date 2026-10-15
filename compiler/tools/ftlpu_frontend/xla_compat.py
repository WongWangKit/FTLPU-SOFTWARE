"""Small torch-xla 2.9 BF16 NumPy interoperability adapter.

Conversion and serialization still belong to torch-xla. Its 2.9 exporter calls
Tensor.numpy() on BF16 states, which torch 2.9 rejects; native NumPy save also
stores registered BF16 as V2. Reconstitute only where captured/native metadata
explicitly declares BF16, retaining its exact bits and native artifact layout.
"""
from pathlib import Path

import numpy as np

from .tensor_io import numpy_tensor, torch_tensor


def bf16_array(array):
    import ml_dtypes
    import torch
    if isinstance(array, torch.Tensor):
        if array.dtype != torch.bfloat16:
            raise ValueError("BF16 metadata disagrees with tensor dtype")
        return numpy_tensor(array)
    array = np.asarray(array)
    if array.dtype == np.dtype(ml_dtypes.bfloat16):
        return array
    if array.dtype.kind != "V" or array.dtype.itemsize != 2:
        raise ValueError("BF16 native metadata requires a two-byte V2 or BF16 payload")
    return array.view(ml_dtypes.bfloat16)


def export_bf16(exported, output):
    import torch_xla
    from torch_xla.stablehlo import StableHLOExportOptions, exported_program_to_stablehlo

    if not torch_xla.__version__.startswith("2.9."):
        raise ValueError("BF16 interoperability adapter is verified only for torch-xla 2.9")
    options = StableHLOExportOptions()
    options.export_weights = False
    program = exported_program_to_stablehlo(exported, options=options)
    states = {}
    for spec in exported.graph_signature.input_specs:
        if spec.kind.name == "USER_INPUT":
            continue
        constants = spec.kind.name == "CONSTANT_TENSOR" or (
            spec.kind.name == "BUFFER" and spec.persistent is False)
        value = (exported.constants if constants else exported.state_dict)[spec.target]
        states[spec.target] = numpy_tensor(value)
    # The same native save() writes arrays/metadata/bytecode. No format writer.
    program._bundle.state_dict = states
    text = program.get_stablehlo_text()
    if text is not None:
        (Path(output) / "conversion.mlir").write_text(text, encoding="utf-8")
    program.save(str(output))
    return program


def load_saved_program(root):
    from torch_xla.stablehlo import StableHLOGraphModule

    program = StableHLOGraphModule.load(str(root))
    for function in program._bundle.stablehlo_funcs:
        for location, signature in zip(function.meta.input_locations, function.meta.input_signature):
            if signature.dtype != "bfloat16":
                continue
            kind = getattr(location.type_, "value", location.type_)
            if kind == "parameter":
                value = program._bundle.state_dict[location.name]
                program._bundle.state_dict[location.name] = torch_tensor(bf16_array(value))
            elif kind == "constant":
                value = program._bundle.additional_constants[location.position]
                program._bundle.additional_constants[location.position] = torch_tensor(bf16_array(value))
    return program
