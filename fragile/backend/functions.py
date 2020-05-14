import numpy
import torch
import xxhash

from fragile.backend.backend import Backend


def hash_tensor(tensor):
    def hash_numpy(x: numpy.ndarray) -> int:
        """Return a value that uniquely identifies a numpy array."""
        return xxhash.xxh64_hexdigest(x.tobytes())

    funcs = {"numpy": hash_numpy,
             "torch": lambda x: hash(x),
             }
    return Backend.execute(tensor, funcs)


def concatenate(tensors, axis=0, out=None):
    funcs = {"numpy": lambda x: numpy.concatenate(x, axis=axis, out=out),
             "torch": lambda x: torch.cat(x, dim=axis, out=out),
             }
    return Backend.execute(tensors, funcs)


def stack(tensors, axis=0, out=None):
    funcs = {"numpy": lambda x: numpy.stack(x, axis=axis, out=out),
             "torch": lambda x: torch.stack(x, dim=axis, out=out),
             }
    return Backend.execute(tensors, funcs)


def clip(tensor, a_min, a_max, out=None):
    funcs = {"numpy": lambda x: numpy.clip(x, a_min, a_max, out=out),
             "torch": lambda x: torch.clamp(x, a_min, a_max, out=out),
             }
    return Backend.execute(tensor, funcs)


def repeat(tensor, repeat, axis=None):
    funcs = {"numpy": lambda x: numpy.repeat(x, repeat, axis=axis),
             "torch": lambda x: torch.repeat_interleave(x, repeat, dim=axis),
             }
    return Backend.execute(tensor, funcs)
