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


class __FragileTensor:
    def __init__(self, wrapped):
        self._wrappped = wrapped

    def __getattr__(self, item):
        return getattr(self._wrappped, item)

    def __subclasscheck__(self, subclass):
        print("checking instance", subclass, type(self._wrappped))
        if subclass == tensor:
            return dtype.is_tensor(self._wrappped)
        return self._wrappped.__subclasscheck__(subclass)

    def __instancecheck__(self, instance):
        print("checking instance", instance, type(self._wrappped))
        if instance == tensor:
            return dtype.is_tensor(self._wrappped)
        return self._wrappped.__instancecheck__(instance)


class MetaScalar(type):
    @property
    def bool(cls):
        funcs = {"numpy": lambda x: numpy.bool_, "torch": lambda x: torch.bool}
        return Backend.execute(None, funcs)

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
        funcs = {"numpy": lambda x: numpy.int64, "torch": lambda x: torch.int64}
        return Backend.execute(None, funcs)

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
        dtypes = (float, cls.float64, cls.float32, cls.float16)
        if cls.is_tensor(x):
            return x.dtype in dtypes
        return isinstance(x, dtypes)

    @classmethod
    def is_int(cls, x):
        dtypes = (int, dtype.int64, dtype.int32, dtype.int16)
        if cls.is_tensor(x):
            return x.dtype in dtypes
        return isinstance(x, dtypes)

    @classmethod
    def is_tensor(cls, x):
        return isinstance(x, tensor.type)  # or cls.is_hash_tensor(x)


class typing:
    Tensor = Union[numpy.ndarray, torch.Tensor]
    int = Union[int, dtype.int64, dtype.int32, dtype.int16]
    float = Union[float, dtype.float64, dtype.float32, dtype.float16]
    bool = Union[bool, dtype.bool]
    StateDict = Dict[str, Dict[str, Any]]
    DistanceFunction = Callable[[numpy.ndarray, numpy.ndarray], numpy.ndarray]
    Scalar = Union[float, int]


class tensor(metaclass=MetaTensor):
    def __new__(cls, x, requires_grad: bool = None, device: str = None, *args, **kwargs):
        if Backend.is_numpy():
            new_tensor = numpy.array(x, *args, **kwargs)
        elif Backend.is_torch():
            if dtype.is_tensor(x):
                return x
            try:
                use_grad = Backend.use_grad() if requires_grad is None else requires_grad
                device = Backend.get_device() if device is None else device
                try:
                    new_tensor = torch.tensor(
                        x, *args, requires_grad=use_grad, device=device, **kwargs,
                    )
                except Exception as e:
                    new_tensor = torch.tensor(
                        x, *args, requires_grad=False, device=device, **kwargs,
                    )
            except Exception as e:
                print(x, args, kwargs)
                raise e
        return new_tensor

    @staticmethod
    def copy(x, requires_grad: bool = None, device=None):
        def copy_torch(x: torch.Tensor, requires_grad, device):
            grad = requires_grad if requires_grad is not None else Backend.use_grad()
            dev = Backend.get_device() if device is None else device
            new_tensor = x.clone().to(dev)
            if not grad:
                new_tensor = new_tensor.detach()
            return new_tensor

        funcs = {
            "numpy": lambda x: x.copy(),
            "torch": lambda x: copy_torch(x, requires_grad, device),
        }
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

    @staticmethod
    def to_numpy(x):
        if isinstance(x, numpy.ndarray):
            return x
        try:
            return x.cpu().numpy()
        except Exception:
            return numpy.asarray(x)

    @classmethod
    def to_backend(cls, x: typing.Tensor, use_grad: bool = None, device: str = None):
        if Backend.is_numpy():
            return cls.to_numpy(x)
        use_grad = use_grad if use_grad is not None else Backend.use_grad()
        device = device if device is not None else Backend.get_device()
        if x.requires_grad != use_grad and dtype.is_float(x):
            x = x.requires_grad_(use_grad)
        if x.device.type != device:
            x = x.to(device=device)
        return x
