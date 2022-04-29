import copy
from typing import Any, Dict, Generator, List, Optional, Set, Tuple, Union

import judo
from judo import tensor
from judo.functions.api import API
from judo.functions.hashing import hasher

from fragile.core.typing import StateDict, Tensor


_BASE_SLOTS = (
    "_param_dict",
    "_n_walkers",
    "_clone_names",
    "_list_names",
    "_tensor_names",
    "_names",
)


class State:
    """
    Data structure that handles the data defining a population of agents.

    Each population attribute will be stored as a tensor with its first dimension \
    (batch size) representing each agent.

    In order to define a tensor attribute, a `param_dict` dictionary needs to be \
    specified using the following structure::

        param_dict = {"name_1": {"shape": 1,
                                 "dtype": numpy.float32,
                                },
                     }

    Where tuple is a tuple indicating the shape of the desired tensor. The \
    created arrays will accessible the ``name_1`` attribute of the class, or \
    indexing the class with ``states["name_1"]``.

    If ``size`` is not defined the attribute will be considered a vector of \
    length `n_walkers`.


    Args:
        n_walkers: The number of items in the first dimension of the tensors.
        param_dict: Dictionary defining the attributes of the tensors.

    """

    def __init__(self, n_walkers: int, param_dict: StateDict):
        """
        Initialize a :class:`SwarmState`.

        Args:
             n_walkers: The number of items in the first dimension of the tensors.
             param_dict: Dictionary defining the attributes of the tensors.

        """

        def shape_is_vector(v):
            shape = v.get("shape", ())
            return not (shape is None or (isinstance(shape, tuple) and len(shape) > 0))

        self._param_dict = param_dict
        self._n_walkers = n_walkers
        self._names = tuple(param_dict.keys())
        self._list_names = set(k for k, v in param_dict.items() if v.get("shape", 1) is None)
        self._tensor_names = set(k for k in self.names if k not in self._list_names)
        self._vector_names = set(k for k, v in param_dict.items() if shape_is_vector(v))

    @property
    def n_walkers(self) -> int:
        return self._n_walkers

    @property
    def param_dict(self) -> StateDict:
        return self._param_dict

    @property
    def names(self) -> Tuple[str]:
        return self._names

    @property
    def tensor_names(self) -> Set[str]:
        return self._tensor_names

    @property
    def list_names(self) -> Set[str]:
        return self._list_names

    @property
    def vector_names(self) -> Set[str]:
        return self._vector_names

    def __len__(self) -> int:
        """Length is equal to n_walkers."""
        return self._n_walkers

    def __setitem__(self, key, value: Union[Tuple, List, Tensor]):
        """
        Allow the class to set its attributes as if it was a dict.

        Args:
            key: Attribute to be set.
            value: Value of the target attribute.

        Returns:
            None.

        """
        setattr(self, key, value)

    def __getitem__(self, item: str) -> Union[Tensor, List[Tensor], "SwarmState"]:
        """
        Query an attribute of the class as if it was a dictionary.

        Args:
            item: Name of the attribute to be selected.

        Returns:
            The corresponding item.

        """
        return getattr(self, item)

    def __repr__(self) -> str:
        string = f"{self.__class__.__name__} with {self._n_walkers} walkers\n"
        for k, v in self.items():
            shape = v.shape if hasattr(v, "shape") else None
            new_str = "{}: {} {}\n".format(k, v.__class__.__name__, shape)
            string += new_str
        return string

    def __hash__(self) -> int:
        return hasher.hash_state(self)

    def hash_attribute(self, name: str) -> int:
        """Return a unique id for a given attribute."""
        return hasher.hash_tensor(self[name])

    def hash_batch(self, name: str) -> List[int]:
        """Return a unique id for each walker attribute."""
        return hasher.hash_iterable(self[name])

    def get(self, name: str, default=None, raise_error: bool = True):
        """
        Get an attribute by key and return the default value if it does not exist.

        Args:
            name: Attribute to be recovered.
            default: Value returned in case the attribute is not part of state.
            raise_error: If True, raise AttributeError if name is not present in states.

        Returns:
            Target attribute if found in the instance, otherwise returns the
            default value.

        """
        if name not in self.names:
            if raise_error:
                raise AttributeError(f"{name} not present in states.")
            return default
        return self[name]

    def keys(self) -> "_dict_keys[str, Dict[str, Any]]":  # pyflakes: disable=F821
        """Return a generator for the values of the stored data."""
        return self.param_dict.keys()

    def values(self) -> Generator:
        """Return a generator for the values of the stored data."""
        return (self[name] for name in self.keys())

    def items(self) -> Generator:
        """Return a generator for the attribute names and the values of the stored data."""
        return ((name, self[name]) for name in self.keys())

    def itervals(self):
        """
        Iterate the states attributes by walker.

        Returns:
            Tuple containing all the names of the attributes, and the values that
            correspond to a given walker.

        """
        if len(self) <= 1:
            yield self.values()
            raise StopIteration
        for i in range(self.n_walkers):
            yield tuple(v[i] for v in self.values())

    def iteritems(self):
        """
        Iterate the states attributes by walker.

        Returns:
            Tuple containing all the names of the attributes, and the values that
            correspond to a given walker.

        """
        if self.n_walkers < 1:
            return self.values()
        for i in range(self.n_walkers):
            yield tuple(self.names), tuple(v[i] for v in self.values())

    def update(self, other: Union["SwarmState", Dict[str, Tensor]] = None, **kwargs):
        """
        Modify the data stored in the SwarmState instance.

        Existing attributes will be updated, and new attributes will be created if needed.

        Args:
            other: State class that will be copied upon update.
            **kwargs: It is possible to specify the update as name value attributes,
                where name is the name of the attribute to be updated, and value
                is the new value for the attribute.
        """
        new_values = other.to_dict() if isinstance(other, SwarmState) else (other or {})
        new_values.update(kwargs)
        for name, val in new_values.items():
            val = self._parse_value(name, val)
            setattr(self, name, val)

    def to_dict(self) -> Dict[str, Union[Tensor, list]]:
        """Return the stored data as a dictionary of arrays."""
        return {name: value for name, value in self.items()}

    def copy(self) -> "SwarmState":
        """Crete a copy of the current instance."""
        new_states = self.__class__(n_walkers=self.n_walkers, param_dict=self.param_dict)
        new_states.update(**dict(self))
        return new_states

    def reset(self) -> None:
        """Reset the values of the class"""
        data = self.params_to_arrays(self.param_dict, self.n_walkers)
        self.update(data)

    def params_to_arrays(self, param_dict: StateDict, n_walkers: int) -> Dict[str, Tensor]:
        """
        Create a dictionary containing the arrays specified by param_dict.

        Args:
            param_dict: Dictionary defining the attributes of the tensors.
            n_walkers: Number items in the first dimension of the data tensors.

        Returns:
            Dictionary with the same names as param_dict, containing arrays specified
            by `param_dict` values.

        """
        tensor_dict = {}
        for key, val in param_dict.items():
            val = copy.deepcopy(val)
            shape = val.pop("shape", -1)  # If shape is not specified assume it's a scalar vector.
            if (
                shape is None or key in self.list_names
            ):  # If shape is None assume it's not a tensor but a list.
                value = [None] * n_walkers
            else:  # Initialize the tensor to zeros. Assumes dtype is a valid argument.
                if shape == -1:
                    shape = (n_walkers,)
                else:
                    shape = (
                        (n_walkers, shape) if isinstance(shape, int) else ((n_walkers,) + shape)
                    )
                value = API.zeros(shape, **val)
            tensor_dict[key] = value
        return tensor_dict

    def _parse_value(self, name, value) -> Any:
        if name in self.list_names:
            assert isinstance(value, list), (name, value)
            return value
        tensor_val = tensor(value, dtype=self._param_dict[name].get("dtype", type(value[0])))
        return tensor_val


class SwarmState(State):
    def __init__(self, n_walkers: int, param_dict: StateDict):
        """
        Initialize a :class:`SwarmState`.

        Args:
             n_walkers: The number of items in the first dimension of the tensors.
             param_dict: Dictionary defining the attributes of the tensors.

        """
        self._clone_names = set(k for k, v in param_dict.items() if v.get("clone"))
        super(SwarmState, self).__init__(n_walkers=n_walkers, param_dict=param_dict)

    @property
    def clone_names(self) -> Set[str]:
        return self._clone_names

    def clone(self, will_clone, compas_clone, clone_names):
        """Clone all the stored data according to the provided index."""
        for name in clone_names:
            values = self[name]
            if name in self.tensor_names:
                self[name][will_clone] = values[compas_clone][will_clone]
            else:
                self[name] = [
                    values[comp] if wc else val
                    for val, comp, wc in zip(values, compas_clone, will_clone)
                ]

    def export_walker(self, index: int, names=None):
        names = names if names is not None else self.keys()
        return {
            k: (v[[index]] if k in self.tensor_names else [v[index]])
            for k, v in self.items()
            if k in names
        }

    def import_walker(self, data: Dict[str, Tensor], index: int = 0):
        for name, tensor_ in data.items():
            if name in self.tensor_names:
                self[name][[index]] = judo.copy(tensor_)
            else:
                self[name][index] = judo.copy(tensor_)

    def reset(self, root_walker: Optional[Dict[str, Tensor]] = None) -> None:
        """Reset the values of the class"""
        super(SwarmState, self).reset()
        if root_walker:
            for name, array in root_walker.items():
                if name in self.tensor_names:
                    self[name][:] = judo.copy(array)
                else:
                    self[name] = copy.deepcopy([array[0] for _ in range(self.n_walkers)])
