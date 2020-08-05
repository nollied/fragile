import numpy


from fragile.backend.backend import Backend, torch
from fragile.backend.data_types import typing
from fragile.backend.fragile_tensor import tensor


class MetaTorchRandomState(type):
    @property
    def seed(cls):
        return torch.random.seed()


class TorchRandomState(metaclass=MetaTorchRandomState):
    def __init__(self, seed: typing.Scalar):
        numpy.random.seed(seed)
        torch.random.manual_seed(int(seed))

    def __getattr__(self, item):
        return getattr(torch, item)

    @staticmethod
    def permutation(x):
        idx = torch.randperm(x.shape[0])
        sample = x[idx].to(Backend.get_device()).detach()
        return tensor.to_backend(sample)

    @staticmethod
    def random_sample(*args, **kwargs):
        sample = torch.rand(*args, **kwargs)
        return tensor.to_backend(sample)

    @staticmethod
    def choice(a, size=None, replace=True, p=None):
        a = tensor.to_backend(a)
        size = size if size is not None else 1
        if replace:
            size = size if isinstance(size, tuple) else (size,)
            indices = tensor.to_backend(torch.randint(len(a), size))
            samples = a[indices]
        else:
            indices = tensor.to_backend(torch.randperm(len(a)))[:size]
            samples = a[indices]
        return tensor.to_backend(samples)

    @staticmethod
    def uniform(
        low=0.0, high=1.0, size=None, dtype=None,
    ):
        uniform = torch.distributions.uniform.Uniform(low, high)
        if size is not None:
            size = size if isinstance(size, tuple) else (size,)
            sample = uniform.sample(size)
        else:
            sample = uniform.sample()
        if dtype is not None:
            sample = sample.to(dtype)
        return tensor.to_backend(sample)

    @staticmethod
    def randint(low, high, size=None, dtype=None):
        size = size if size is not None else (1,)
        size = size if isinstance(size, tuple) else (size,)
        data = torch.randint(low, high, size)
        if dtype is not None:
            data = data.to(dtype)
        return tensor.to_backend(data)

    @staticmethod
    def normal(loc=0, scale=1.0, size=None):
        size = size if size is not None else (1,)
        size = size if isinstance(size, tuple) else (size,)
        return tensor.to_backend(torch.normal(mean=loc, std=scale, size=size))

    @classmethod
    def random(cls, size=None):
        return cls.uniform(size=size)


class MetaRandomState(type):
    def __getattr__(cls, item):
        funcs = {
            "numpy": lambda x: getattr(cls._numpy_random_state, x),
            "torch": lambda x: getattr(cls._torch_random_state, x),
        }
        return Backend.execute(item, funcs)


class RandomState(metaclass=MetaRandomState):
    _random_seed = 160290
    _numpy_random_state = numpy.random.RandomState(seed=_random_seed)
    _torch_random_state = TorchRandomState(seed=_random_seed)

    @classmethod
    def seed(cls, seed: typing.Scalar = _random_seed):
        cls._numpy_random_state = numpy.random.RandomState(seed=seed)
        cls._torch_random_state = TorchRandomState(seed=seed)


random_state = RandomState
