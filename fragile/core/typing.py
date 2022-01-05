from typing import Any, Callable, Dict, Generator, List, Set, Tuple, Union

from judo.judo_backend import torch
import numpy


StateDict = Dict[str, Dict[str, Any]]

Tensor = Union[numpy.ndarray, torch.Tensor]
Vector = Union[numpy.ndarray, torch.Tensor]
Matrix = Union[numpy.ndarray, torch.Tensor]
Scalar = Union[int, float]

NodeId = Union[str, int]
NodeData = Union[Tuple[dict, dict], Tuple[dict, dict, dict]]
NodeDataGenerator = Generator[NodeData, None, None]
NamesData = Union[Tuple[str], Set[str], List[str]]

DistanceFunction = Callable[[Tensor, Tensor], Tensor]

InputDict = StateDict
StateData = Dict[str, Union[Tensor, list]]
