"""Model-independent Kernel IR builders for shared transformer operators.

Model adapters supply effective weights (for example 1 + checkpoint gamma).
Tensor layout, scheduling and ICU instructions belong to the existing compiler
lowerers, not to the adapter or CModel.
"""


def tensor_type(shape, dtype):
    return 'tensor<' + 'x'.join(map(str, shape)) + 'x' + dtype + '>'


def rms_norm_source(shape, epsilon):
    if len(shape) != 2:
        raise ValueError('RMSNorm expects a rows-by-hidden matrix')
    rows, hidden = shape
    if (rows != 1 and rows % 32) or hidden % 32:
        raise ValueError('shared RMSNorm lowering requires tile-aligned tensors')
    activation = tensor_type(shape, 'bf16')
    gamma = tensor_type((hidden,), 'bf16')
    return f'''module {{ func.func @stage(%a: {activation}, %g: {gamma}) -> {activation} {{
      %o = ftlpu.kernel.rms_norm %a, %g {{axis = 1 : i64,
        epsilon = {float(epsilon):.9e} : f32}} : ({activation}, {gamma}) -> {activation}
      return %o : {activation}
    }} }}'''
