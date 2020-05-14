import torch
AVAILABLE_BACKENDS = ["numpy", "torch"]


class Backend:
    _backend = "numpy"
    _use_grad = False
    _device = "gpu" if torch.cuda.is_available() else "cpu"

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
    def set_backend(cls, name=None, device=None, use_grad: bool=None):
        if name is not None:
            if name not in AVAILABLE_BACKENDS:
                raise ValueError(
                    "%s not supported. Available backends: %s" % (name, AVAILABLE_BACKENDS)
                )
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
    def execute(cls, tensor, funcs):
        backend = cls.get_current_backend()
        try:
            return funcs[backend](tensor)
        except Exception as e:
            print(backend, tensor.dtype, type(tensor.dtype))
            raise e
