from fragile.backend.backend import Backend
from fragile.backend.functions import pytorch, numpy

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


class MetaAPI(type):

    def __getattr__(self, item):
        return self.get_function(name=item)

    @staticmethod
    def get_function(name):
        if Backend.is_numpy():
            backend = numpy
        else:
            backend = pytorch
        return getattr(backend, name)


class API(metaclass=MetaAPI):
    pass

