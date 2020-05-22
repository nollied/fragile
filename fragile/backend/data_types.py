from typing import Any, Callable, Dict, Union

import numpy

from fragile.backend.backend import Backend, torch


class MetaTensor(type):
    def __getattr__(cls, item):
        from fragile.backend import functions
        def wrapped(func, *args, **kwargs):  # Handle device placement automatically
            val = func(*args, **kwargs)
            #t = tensor.to_backend(val)
            if isinstance(val, torch.Tensor):
                assert not val.requires_grad
            return tensor.to_backend(val)

        if item in functions.AVAILABLE_FUNCTIONS:  # Functions available within tensor namespace
            func = getattr(functions, item)
        elif Backend.is_numpy():
            func = getattr(numpy, item)
        elif Backend.is_torch():
            func = getattr(torch, item)
        return lambda *args, **kwargs: wrapped(func, *args, **kwargs)

    @property
    def type(cls):
        funcs = {"numpy": lambda x: numpy.ndarray, "torch": lambda x: torch.Tensor}
        return Backend.execute(None, funcs)


class MetaScalar(type):
    @property
    def bool(cls):
        funcs = {"numpy": lambda x: numpy.bool_, "torch": lambda x: torch.bool}
        return Backend.execute(None, funcs)

    @property
    def uint8(cls):
        funcs = {"numpy": lambda x: numpy.uint8, "torch": lambda x: torch.uint8}
        return Backend.execute(None, funcs)

    @property
    def int16(cls):
        funcs = {"numpy": lambda x: numpy.int16, "torch": lambda x: torch.int16}
        return Backend.execute(None, funcs)

    @property
    def int32(cls):
        funcs = {"numpy": lambda x: numpy.int32, "torch": lambda x: torch.int32}
        return Backend.execute(None, funcs)

    @property
    def int64(cls):
        funcs = {"numpy": lambda x: numpy.int64, "torch": lambda x: torch.int64}
        return Backend.execute(None, funcs)

    @property
    def int(cls):
        funcs = {"numpy": lambda x: numpy.int64, "torch": lambda x: torch.int64}
        return Backend.execute(None, funcs)

    @property
    def float(cls):
        funcs = {"numpy": lambda x: numpy.float32, "torch": lambda x: torch.float32}
        return Backend.execute(None, funcs)

    @property
    def float16(cls):
        funcs = {"numpy": lambda x: numpy.float16, "torch": lambda x: torch.float16}
        return Backend.execute(None, funcs)

    @property
    def float32(cls):
        funcs = {"numpy": lambda x: numpy.float32, "torch": lambda x: torch.float32}
        return Backend.execute(None, funcs)

    @property
    def float64(cls):
        funcs = {"numpy": lambda x: numpy.float64, "torch": lambda x: torch.float64}
        return Backend.execute(None, funcs)

    @property
    def hash_type(cls):
        funcs = {"numpy": lambda x: numpy.dtype("<U64"), "torch": lambda x: torch.int64}
        return Backend.execute(None, funcs)


class dtype(metaclass=MetaScalar):
    @classmethod
    def is_bool(cls, x):
        dtypes = (bool, dtype.bool)
        if cls.is_tensor(x):
            return x.dtype in dtypes
        return isinstance(x, dtypes)

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

    @classmethod
    def to_node_id(cls, x):
        if Backend.is_numpy():
            return str(x)
        elif Backend.is_torch():
            return int(x)


class MetaTyping(type):
    @property
    def int(self):
        try:
            return Union[int, dtype.int64, dtype.int32, dtype.int16]
        except Exception as e:
            return int

    @property
    def float(self):
        try:
            return Union[float, dtype.float64, dtype.float32, dtype.float16]
        except Exception as e:
            return float

    @property
    def bool(self):
        try:
            return Union[bool, dtype.bool]
        except Exception as e:
            return bool

    @property
    def Scalar(self):
        try:
            return Union[self.float, self.int]
        except Exception as e:
            return Union[int, float]


class typing(metaclass=MetaTyping):
    Tensor = Union[numpy.ndarray, torch.Tensor]
    StateDict = Dict[str, Dict[str, Any]]
    DistanceFunction = Callable[[numpy.ndarray, numpy.ndarray], numpy.ndarray]


class tensor(metaclass=MetaTensor):
    def __new__(cls, x, requires_grad: bool = None, device: str = None, *args, **kwargs):
        if Backend.is_numpy():
            new_tensor = numpy.array(x, *args, **kwargs)
        elif Backend.is_torch():
            if dtype.is_tensor(x):
                return cls.to_backend(x, use_grad=requires_grad, device=device)
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

    @classmethod
    def copy(cls, x, requires_grad: bool = None, device=None):
        if x is None:
            return
        if not dtype.is_tensor(x):
            x = tensor(x)

        def copy_torch(x: torch.Tensor, requires_grad, device):
            grad = requires_grad if requires_grad is not None else Backend.use_grad()
            new_tensor = x.clone()
            if not grad:
                new_tensor = new_tensor.detach()
            new_tensor = cls.to_backend(new_tensor, device=device, use_grad=requires_grad)
            return new_tensor

        funcs = {
            "numpy": lambda x: x.copy(),
            "torch": lambda x: copy_torch(x, requires_grad, device),
        }
        return Backend.execute(x, funcs)

    @staticmethod
    def astype(x, dtype):
        funcs = {
            "numpy": lambda x: x.astype(dtype),
            "torch": lambda x: x.to(dtype),
        }
        return Backend.execute(x, funcs)

    @staticmethod
    def as_tensor(x, *args, **kwargs):
        funcs = {
            "numpy": lambda x: numpy.asarray(x, *args, **kwargs),
            "torch": lambda x: torch.as_tensor(x, *args, **kwargs),
        }
        return Backend.execute(x, funcs)

    @staticmethod
    def to_numpy(x):
        if isinstance(x, numpy.ndarray):
            return x
        try:
            return x.cpu().detach().numpy()
        except Exception:
            return numpy.asarray(x)

    @classmethod
    def to_torch(
        cls, x, use_grad: bool = None, device: str = None, copy: bool = False, *args, **kwargs
    ):
        def new_tensor(x, use_grad, device, *args, **kwargs):
            try:
                new_tensor = torch.tensor(
                    x, *args, requires_grad=use_grad, device=device, **kwargs,
                )
            except Exception as e:
                new_tensor = torch.tensor(x, *args, requires_grad=False, device=device, **kwargs,)
            return new_tensor
        if isinstance(x, torch.Tensor):
            return x
        use_grad = use_grad if use_grad is not None else Backend.use_grad()
        device = device if device is not None else Backend.get_device()
        if isinstance(x, numpy.ndarray):
            x = (
                torch.from_numpy(x)
                if not copy
                else new_tensor(x, use_grad, device, *args, **kwargs)
            )

        elif not isinstance(x, torch.Tensor):
            try:
                if not copy:
                    return cls.as_tensor(x)
                else:
                    x = new_tensor(x, use_grad, device, *args, **kwargs)
            except Exception:
                x = new_tensor(x, use_grad, device, *args, **kwargs)
        try:
            if x.requires_grad != use_grad and dtype.is_float(x):
                x = x.requires_grad_(use_grad)
        except RuntimeError:
            pass
        if x.device.type != device:
            x = x.to(device=device)
        return x

    @classmethod
    def to_backend(cls, x: typing.Tensor, use_grad: bool = None, device: str = None):
        if Backend.is_numpy():
            return cls.to_numpy(x)
        return cls.to_torch(x, use_grad=use_grad, device=device)
