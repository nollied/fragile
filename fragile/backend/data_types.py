from typing import Any, Callable, Dict, Union

import numpy
import torch

from fragile.backend.backend import Backend


class MetaTensor(type):
    def __getattr__(cls, item):
        if Backend.is_numpy():
            return getattr(numpy, item)
        elif Backend.is_torch():
            return getattr(torch, item)

    @property
    def type(cls):
        if Backend.is_numpy():
            return numpy.ndarray
        elif Backend.is_torch():
            return torch.Tensor


class tensor(metaclass=MetaTensor):
    def __new__(cls, *args, **kwargs):
        if Backend.is_numpy():
            return numpy.array(*args, **kwargs)
        elif Backend.is_torch():
            return torch.tensor(*args, **kwargs)

    @staticmethod
    def copy(x, requires_grad: bool=None):
        def copy_torch(x: torch.Tensor, requires_grad):
            grad = requires_grad if requires_grad is not None else Backend.use_grad()
            new_tensor = x.clone()
            if not grad:
                new_tensor = x.detach()
            return new_tensor
        funcs = {"numpy": lambda x: x.copy(),
                 "torch": lambda x: copy_torch(x, requires_grad)}
        return Backend.execute(x, funcs)


    @staticmethod
    def astype(x, dtype):
        if Backend.is_numpy():
            return x.astype(dtype)
        elif Backend.is_torch():
            return x.to(dtype)

    @staticmethod
    def as_tensor(x, *args, **kwargs):
        if Backend.is_numpy():
            return numpy.asarray(x, *args, **kwargs)
        elif Backend.is_torch():
            return torch.as_tensor(x, *args, **kwargs)


class MetaScalar(type):
    @property
    def bool(cls):
        if Backend.is_numpy():
            return numpy.bool_
        elif Backend.is_torch():
            return torch.bool

    @property
    def uint8(cls):
        if Backend.is_numpy():
            return numpy.uint8
        elif Backend.is_torch():
            return torch.uint8

    @property
    def int16(cls):
        if Backend.is_numpy():
            return numpy.int16
        elif Backend.is_torch():
            return torch.int16

    @property
    def int32(cls):
        if Backend.is_numpy():
            return numpy.int32
        elif Backend.is_torch():
            return torch.int32

    @property
    def int64(cls):
        if Backend.is_numpy():
            return numpy.int64
        elif Backend.is_torch():
            return torch.int64

    @property
    def int(cls):
        if Backend.is_numpy():
            return numpy.int64
        elif Backend.is_torch():
            return torch.int64

    @property
    def float(cls):
        if Backend.is_numpy():
            return numpy.float32
        elif Backend.is_torch():
            return torch.float32

    @property
    def float16(cls):
        if Backend.is_numpy():
            return numpy.float16
        elif Backend.is_torch():
            return torch.float16

    @property
    def float32(cls):
        if Backend.is_numpy():
            return numpy.float32
        elif Backend.is_torch():
            return torch.float32

    @property
    def float64(cls):
        if Backend.is_numpy():
            return numpy.float64
        elif Backend.is_torch():
            return torch.float64

    @property
    def hash_type(cls):
        if Backend.is_numpy():
            return numpy.dtype("<U64")
        elif Backend.is_torch():
            return cls.int64


class dtype(metaclass=MetaScalar):
    @classmethod
    def is_bool(cls, x):
        return isinstance(x, (bool, dtype.bool))

    @classmethod
    def is_float(cls, x):
        return isinstance(x, (float, cls.float64, cls.float32, cls.float16))

    @classmethod
    def is_int(cls, x):
        return isinstance(x, (int, dtype.int64, dtype.int32, dtype.int16))

    @classmethod
    def is_tensor(cls, x):
        return isinstance(x, tensor.type) # or cls.is_hash_tensor(x)


class typing:
    Tensor = Union[numpy.ndarray, torch.Tensor]
    int = Union[int, dtype.int64, dtype.int32, dtype.int16]
    float = Union[float, dtype.float64, dtype.float32, dtype.float16]
    bool = Union[bool, dtype.bool]
    StateDict = Dict[str, Dict[str, Any]]
    DistanceFunction = Callable[[numpy.ndarray, numpy.ndarray], numpy.ndarray]
    Scalar = Union[float, int]

