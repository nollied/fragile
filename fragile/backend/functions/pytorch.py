from fragile.backend.backend import torch
from fragile.backend.data_types import dtype
from fragile.backend.fragile_tensor import tensor

AVAILABLE_FUNCTIONS = [
    "argmax",
    "hash_numpy",
    "hash_tensor",
    "concatenate",
    "stack",
    "clip",
    "repeat",
    "min",
    "max",
    "norm",
    "unsqueeze",
    "where",
    "sqrt",
    "tile",
    "logical_or",
    "logical_and",
]


def concatenate(x, axis=0, out=None):
    return torch.cat(x, dim=axis, out=out)


def stack(x, axis=0, out=None):
    return torch.stack(x, dim=axis, out=out)


def clip(x, a_min, a_max, out=None):

    _tensor = min(x, other=tensor.astype(tensor(a_max), dtype=x.dtype))
    return max(_tensor, other=tensor.astype(tensor(a_min), dtype=x.dtype), out=out)

    _tensor = torch.zeros_like(x)
    if dtype.is_tensor(a_min) and not dtype.is_tensor(a_max):
        for i, x_row in enumerate(a_min):
            _tensor[:, i] = torch.clamp(x[:, i], x_row, a_max)
    elif not dtype.is_tensor(a_min) and dtype.is_tensor(a_max):
        for i, x_row in enumerate(a_max):
            _tensor[:, i] = torch.clamp(x[:, i], a_min, x_row)
    elif dtype.is_tensor(a_min) and dtype.is_tensor(a_max):
        try:  # clamp one dimensional array
            for i, (x_row, y_row) in enumerate(zip(a_min, a_max)):
                _tensor[:, i] = torch.clamp(x[:, i], x_row, y_row)
        except TypeError:  # clamp matrices
            _tensor = torch.where(x > a_min, x, a_min)
            _tensor = torch.where(_tensor < a_max, x, a_max)
    else:
        _tensor = torch.clamp(_tensor, a_min, a_max, out=out)
    return _tensor


def repeat(x, repeat, axis=None):
    return torch.repeat_interleave(x, repeat, dim=axis)


def tile(x, repeat):
    return torch.Tensor.repeat(x, *repeat if isinstance(repeat, tuple) else [repeat])


def norm(x, ord=None, axis=None, keepdims=False):
    return torch.norm(x, p=ord, dim=axis, keepdim=keepdims)


def min(x, axis=None, other=None, out=None):
    if other is None:
        axis = axis if axis is not None else 0
        val, _ = torch.min(x, dim=axis, out=out)
        return val
    return torch.min(x, other=other, out=out)


def max(x, axis=None, other=None, out=None):
    if other is None:
        axis = axis if axis is not None else 0
        val, _ = torch.max(x, dim=axis, out=out)
        return val
    return torch.max(x, other=other, out=out)


def unsqueeze(x, axis=0):
    return x.unsqueeze(axis)


def where(cond, a, b, *args, **kwargs):
    was_bool = False
    if a.dtype == dtype.bool:
        a = a.to(dtype.int32)
        was_bool = True
    if b.dtype == dtype.bool:
        b = b.to(dtype.int32)
        was_bool = True
    res = torch.where(cond, a, b)
    return res.to(dtype.bool) if was_bool else res


def argmax(x, axis=None, *args, **kwargs):
    # where not implemented for bool in
    axis = axis if axis is not None else 0
    was_bool = False
    if x.dtype == dtype.bool:
        x = x.to(dtype.int32)
        was_bool = True
    res = torch.argmax(x, dim=axis, *args, **kwargs)
    return res.to(dtype.bool) if was_bool else res


def sqrt(x, *args, **kwargs):
    # where not implemented for bool in
    was_bool = False

    if x.dtype in (dtype.int64, dtype.int32):
        _x = x.to(dtype.float32)
        was_bool = True
    else:
        _x = x
    res = torch.sqrt(_x, *args, **kwargs)
    return res.to(x.dtype) if was_bool else res


def logical_or(a, b):
    return a + b


def logical_and(a, b):
    return a & b
