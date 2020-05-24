import numpy

from fragile.backend.backend import Backend, torch
from fragile.backend.data_types import dtype


class MetaTensor(type):
    def __getattr__(cls, item):
        from fragile.backend.functions import api

        def wrapped(func, *args, **kwargs):  # Handle device placement automatically
            val = func(*args, **kwargs)
            return tensor.to_backend(val)

        func = None
        if item in api.AVAILABLE_FUNCTIONS:  # Functions available within tensor namespace
            func = getattr(api, item)
        elif Backend.is_numpy():
            if func is not None:
                return func
            return getattr(numpy, item)
        elif Backend.is_torch():
            func = getattr(torch, item)
        return lambda *args, **kwargs: wrapped(func, *args, **kwargs)

    @property
    def type(cls):
        funcs = {"numpy": lambda x: numpy.ndarray, "torch": lambda x: torch.Tensor}
        return Backend.execute(None, funcs)


class tensor(metaclass=MetaTensor):
    def __new__(cls, x, requires_grad: bool = None, device: str = None, *args, **kwargs):
        if Backend.is_numpy():
            new_tensor = numpy.asarray(x, *args, **kwargs)
        elif Backend.is_torch():
            if isinstance(x, torch.Tensor):
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
    def to_backend(cls, x: "typing.Tensor", use_grad: bool = None, device: str = None):
        if Backend.is_numpy():
            return cls.to_numpy(x)
        return cls.to_torch(x, use_grad=use_grad, device=device)