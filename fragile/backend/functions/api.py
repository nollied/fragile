from fragile.backend import functions
from fragile.backend.backend import Backend


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
] + list(functions.fractalai.AVAILABLE_FUNCTIONS)


class MetaAPI(type):
    def __getattr__(self, item):
        return self.get_function(name=item)

    @staticmethod
    def get_function(name):
        if name in functions.fractalai.AVAILABLE_FUNCTIONS:
            return getattr(functions.fractalai, name)
        elif Backend.is_numpy():
            backend = functions.numpy
        else:
            backend = functions.pytorch
        return getattr(backend, name)


class API(metaclass=MetaAPI):
    pass
