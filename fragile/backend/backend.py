from contextlib import contextmanager

import torch


@contextmanager
def _use_backend(cls, name, device=None, use_grad=None):
    if name is not None:
        cls._checkvalid_backend(name)
    curr_state = cls.get_backend_state()
    cls.set_backend(name=name, device=device, use_grad=use_grad)
    try:
        yield
    finally:
        cls.set_backend(**curr_state)


class Backend:
    AVAILABLE_BACKENDS = ["numpy", "torch"]
    _backend = "numpy"
    _use_grad = False
    _device = "gpu" if torch.cuda.is_available() else "cpu"

    @classmethod
    def _checkvalid_backend(cls, name):
        if name not in cls.AVAILABLE_BACKENDS:
            raise ValueError(
                "%s not supported. Available backends: %s" % (name, cls.AVAILABLE_BACKENDS)
            )

    @classmethod
    def get_backend_state(cls):
        state = {
            "name": str(cls._backend),
            "device": str(cls._device),
            "use_grad": bool(cls._use_grad),
        }
        return state

    @classmethod
    def get_current_backend(cls):
        return cls._backend

    @classmethod
    def get_device(cls):
        return cls._device

    @classmethod
    def use_grad(cls):
        return cls._use_grad

    @classmethod
    def set_backend(cls, name=None, device=None, use_grad: bool = None):
        if name is not None:
            cls._checkvalid_backend(name)
            cls._backend = name
        if device is not None:
            cls._device = device
        if use_grad is not None:
            cls._use_grad = use_grad

    @classmethod
    def is_numpy(cls):
        return cls._backend == "numpy"

    @classmethod
    def is_torch(cls):
        return cls._backend == "torch"

    @classmethod
    def execute(cls, value, funcs):
        backend = cls.get_current_backend()
        return funcs[backend](value)

    @classmethod
    def use_backend(cls, name=None, device=None, use_grad=None):
        return _use_backend(cls, name=name, device=device, use_grad=use_grad)
