"""Official NumPy/Torch tensor interchange, including registered BF16 dtype."""
import numpy as np


def numpy_tensor(value):
    import torch
    value = value.detach().cpu().contiguous()
    if value.dtype == torch.bfloat16:
        import ml_dtypes
        return value.view(torch.int16).numpy().view(ml_dtypes.bfloat16)
    return value.numpy()


def torch_tensor(value, expected_dtype=None):
    import torch
    if isinstance(value, torch.Tensor):
        return value
    array = np.asarray(value)
    if expected_dtype == "bfloat16" and array.dtype.kind == "V" and array.dtype.itemsize == 2:
        import ml_dtypes
        array = array.view(ml_dtypes.bfloat16)
    if str(array.dtype) == "bfloat16":
        return torch.from_numpy(array.view(np.uint16).copy()).view(torch.bfloat16)
    return torch.as_tensor(value)
