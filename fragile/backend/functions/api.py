import numpy

from fragile.backend.backend import Backend, torch
from fragile.backend.data_types import dtype

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
]


def concatenate(tensors, axis=0, out=None):
    funcs = {
        "numpy": lambda x: numpy.concatenate(x, axis=axis, out=out),
        "torch": lambda x: torch.cat(x, dim=axis, out=out),
    }
    return Backend.execute(tensors, funcs)


def stack(tensors, axis=0, out=None):
    funcs = {
        "numpy": lambda x: numpy.stack(x, axis=axis, out=out),
        "torch": lambda x: torch.stack(x, dim=axis, out=out),
    }
    return Backend.execute(tensors, funcs)


def clip(tensor, a_min, a_max, out=None):
    def torch_clip(_tensor_, a_min, a_max, out=None):
        _tensor = torch.zeros_like(_tensor_)
        if dtype.is_tensor(a_min) and not dtype.is_tensor(a_max):
            for i, x in enumerate(a_min):
                _tensor[:, i] = torch.clamp(_tensor[:, i], x, a_max)
        elif not dtype.is_tensor(a_min) and dtype.is_tensor(a_max):
            for i, x in enumerate(a_max):
                _tensor[:, i] = torch.clamp(tensor[:, i], a_min, x)
        elif dtype.is_tensor(a_min) and dtype.is_tensor(a_max):
            try:  # clamp one dimensional array
                for i, (x, y) in enumerate(zip(a_min, a_max)):
                    _tensor[:, i] = torch.clamp(tensor[:, i], x, y)
            except TypeError:  # clamp matrices
                _tensor = torch.where(_tensor_ > a_min, tensor, a_min)
                _tensor = torch.where(_tensor < a_max, tensor, a_max)
        else:
            _tensor = torch.clamp(_tensor, a_min, a_max, out=out)
        return _tensor

    funcs = {
        "numpy": lambda x: numpy.clip(x, a_min, a_max, out=out),
        "torch": lambda x: torch_clip(x, a_min, a_max, out=out),
    }
    return Backend.execute(tensor, funcs)


def repeat(x, repeat, axis=None):
    funcs = {
        "numpy": lambda x: numpy.repeat(x, repeat, axis=axis),
        "torch": lambda x: torch.repeat_interleave(x, repeat, dim=axis),
    }
    return Backend.execute(x, funcs)


def tile(x, repeat):
    funcs = {
        "numpy": lambda x: numpy.tile(x, repeat),
        "torch": lambda x: torch.Tensor.repeat(
            x, *repeat if isinstance(repeat, tuple) else repeat
        ),
    }
    return Backend.execute(x, funcs)


def norm(tensor, ord=None, axis=None, keepdims=False):
    funcs = {
        "numpy": lambda x: numpy.linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims),
        "torch": lambda x: torch.norm(x, p=ord, dim=axis, keepdim=keepdims),
    }
    return Backend.execute(tensor, funcs)


def min(tensor, axis=None, other=None, out=None):
    def min_numpy(tensor, axis, other, out):
        if other is None:
            return numpy.min(tensor, axis=axis, out=out)
        else:
            return numpy.minimum(tensor, other, out=out)

    def min_torch(tensor, axis, other, out):
        if other is None:
            axis = axis if axis is not None else 0
            val, _ = torch.min(tensor, dim=axis, out=out)
            return val
        return torch.min(tensor, other=other, out=out)

    funcs = {
        "numpy": lambda x: min_numpy(x, axis=axis, other=other, out=out),
        "torch": lambda x: min_torch(x, axis=axis, other=other, out=out),
    }
    return Backend.execute(tensor, funcs)


def max(tensor, axis=None, other=None, out=None):
    def max_numpy(tensor, axis, other, out):
        if other is None:
            return numpy.max(tensor, axis=axis, out=out)
        else:
            return numpy.maximum(tensor, other, out=out)

    def max_torch(tensor, axis, other, out):
        if other is None:
            axis = axis if axis is not None else 0
            val, _ = torch.max(tensor, dim=axis, out=out)
            return val
        return torch.max(tensor, other=other, out=out)

    funcs = {
        "numpy": lambda x: max_numpy(x, axis=axis, other=other, out=out),
        "torch": lambda x: max_torch(x, axis=axis, other=other, out=out),
    }
    return Backend.execute(tensor, funcs)


def unsqueeze(x, axis=0):
    funcs = {
        "numpy": lambda x: numpy.expand_dims(x, axis=axis),
        "torch": lambda x: x.unsqueeze(axis),
    }
    return Backend.execute(x, funcs)


def where(cond, a, b, *args, **kwargs):
    def where_torch(c, _a, _b):  # where not implemented for bool in cpu
        was_bool = False
        if _a.dtype == dtype.bool:
            _a = _a.to(dtype.int32)
            was_bool = True
        if _b.dtype == dtype.bool:
            _b = _b.to(dtype.int32)
            was_bool = True
        res = torch.where(c, _a, _b)
        return res.to(dtype.bool) if was_bool else res

    funcs = {
        "numpy": lambda x: numpy.where(x, a, b, *args, **kwargs),
        "torch": lambda x: where_torch(x, a, b),
    }
    return Backend.execute(cond, funcs)


def argmax(x, axis=None, *args, **kwargs):
    def argmax_torch(_x, axis):  # where not implemented for bool in
        axis = axis if axis is not None else 0
        was_bool = False
        if _x.dtype == dtype.bool:
            _x = _x.to(dtype.int32)
            was_bool = True
        res = torch.argmax(_x, dim=axis, *args, **kwargs)
        return res.to(dtype.bool) if was_bool else res

    funcs = {
        "numpy": lambda x: numpy.argmax(x, axis=axis, *args, **kwargs),
        "torch": lambda x: argmax_torch(x, axis=axis),
    }
    return Backend.execute(x, funcs)


def sqrt(x, *args, **kwargs):
    def sqrt_torch(_x):  # where not implemented for bool in
        was_bool = False
        if _x.dtype == dtype.int64:
            _x = _x.to(dtype.float32)
            was_bool = True
        res = torch.sqrt(_x, *args, **kwargs)
        return res.to(dtype.int64) if was_bool else res

    funcs = {
        "numpy": lambda x: numpy.sqrt(x, *args, **kwargs),
        "torch": lambda x: sqrt_torch(x),
    }
    return Backend.execute(x, funcs)


def logical_or(a, b):
    funcs = {
        "numpy": lambda x: numpy.logical_or(x, b),
        "torch": lambda x: x + a,
    }
    return Backend.execute(a, funcs)
