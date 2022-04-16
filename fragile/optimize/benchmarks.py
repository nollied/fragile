import math
from typing import Callable

import judo
from judo import random_state, tensor
from judo.data_structures.bounds import Bounds
from numba import jit
import numpy as np

from fragile.core.env import Function


"""
This file includes several test functions for optimization described here:
https://en.wikipedia.org/wiki/Test_functions_for_optimization
"""


def sphere(x: np.ndarray) -> np.ndarray:
    return judo.sum(x**2, 1).flatten()


def rastrigin(x: np.ndarray) -> np.ndarray:
    dims = x.shape[1]
    A = 10
    result = A * dims + judo.sum(x**2 - A * judo.cos(2 * math.pi * x), 1)
    return result.flatten()


def eggholder(x: np.ndarray) -> np.ndarray:
    x, y = x[:, 0], x[:, 1]
    first_root = judo.sqrt(judo.abs(x / 2.0 + (y + 47)))
    second_root = judo.sqrt(judo.abs(x - (y + 47)))
    result = -1 * (y + 47) * judo.sin(first_root) - x * judo.sin(second_root)
    return result


def styblinski_tang(x) -> np.ndarray:
    return judo.sum(x**4 - 16 * x**2 + 5 * x, 1) / 2.0


def rosenbrock(x) -> np.ndarray:
    return 100 * judo.sum((x[:, :-2] ** 2 - x[:, 1:-1]) ** 2, 1) + judo.sum(
        (x[:, :-2] - 1) ** 2,
        1,
    )


def easom(x) -> np.ndarray:
    exp_term = (x[:, 0] - np.pi) ** 2 + (x[:, 1] - np.pi) ** 2
    return -judo.cos(x[:, 0]) * judo.cos(x[:, 1]) * judo.exp(-exp_term)


def holder_table(_x) -> np.ndarray:
    x, y = _x[:, 0], _x[:, 1]
    exp = np.abs(1 - (np.sqrt(x * x + y * y) / np.pi))
    return -np.abs(np.sin(x) * np.cos(y) * np.exp(exp))


@jit(nopython=True)
def _lennard_fast(state):
    state = state.reshape(-1, 3)
    npart = len(state)
    epot = 0.0
    for i in range(npart):
        for j in range(npart):
            if i > j:
                r2 = np.sum((state[j, :] - state[i, :]) ** 2)
                r2i = 1.0 / r2
                r6i = r2i * r2i * r2i
                epot = epot + r6i * (r6i - 1.0)
    epot = epot * 4
    return epot


def lennard_jones(x: np.ndarray) -> np.ndarray:
    result = np.zeros(x.shape[0])
    x = judo.to_numpy(x)
    assert isinstance(x, np.ndarray)
    for i in range(x.shape[0]):
        try:
            result[i] = _lennard_fast(x[i])
        except ZeroDivisionError:
            result[i] = np.inf
    result = judo.as_tensor(result)
    return result


class OptimBenchmark(Function):

    benchmark = None
    best_state = None

    def __init__(self, dims: int, function: Callable, **kwargs):
        bounds = self.get_bounds(dims=dims)
        super(OptimBenchmark, self).__init__(bounds=bounds, function=function, **kwargs)

    @staticmethod
    def get_bounds(dims: int) -> Bounds:
        raise NotImplementedError


class Sphere(OptimBenchmark):
    benchmark = tensor(0.0)

    def __init__(self, dims: int, **kwargs):
        super(Sphere, self).__init__(dims=dims, function=sphere, **kwargs)

    @staticmethod
    def get_bounds(dims):
        bounds = [(-1000, 1000) for _ in range(dims)]
        return Bounds.from_tuples(bounds)

    @property
    def best_state(self):
        return judo.zeros(self.shape)


class Rastrigin(OptimBenchmark):
    benchmark = tensor(0.0)

    def __init__(self, dims: int, **kwargs):
        super(Rastrigin, self).__init__(dims=dims, function=rastrigin, **kwargs)

    @staticmethod
    def get_bounds(dims):
        bounds = [(-5.12, 5.12) for _ in range(dims)]
        return Bounds.from_tuples(bounds)

    @property
    def best_state(self):
        return judo.zeros(self.shape)


class EggHolder(OptimBenchmark):
    benchmark = tensor(-959.64066271)

    def __init__(self, dims: int = None, **kwargs):
        super(EggHolder, self).__init__(dims=2, function=eggholder, **kwargs)

    @staticmethod
    def get_bounds(dims=None):
        bounds = [(-512.0, 512.0), (-512.0, 512.0)]
        # bounds = [(1, 512.0), (1, 512.0)]
        return Bounds.from_tuples(bounds)

    @property
    def best_state(self):
        return tensor([512.0, 404.2319])


class StyblinskiTang(OptimBenchmark):
    def __init__(self, dims: int, **kwargs):
        super(StyblinskiTang, self).__init__(dims=dims, function=styblinski_tang, **kwargs)

    @staticmethod
    def get_bounds(dims):
        bounds = [(-5.0, 5.0) for _ in range(dims)]
        return Bounds.from_tuples(bounds)

    @property
    def best_state(self):
        return judo.ones(self.shape) * -2.903534

    @property
    def benchmark(self):
        return tensor(-39.16617 * self.shape[0])


class Rosenbrock(OptimBenchmark):
    def __init__(self, dims: int, **kwargs):
        super(Rosenbrock, self).__init__(dims=dims, function=rosenbrock, **kwargs)

    @staticmethod
    def get_bounds(dims):
        bounds = [(-10.0, 10.0) for _ in range(dims)]
        return Bounds.from_tuples(bounds)

    @property
    def best_state(self):
        return judo.ones(self.shape)

    @property
    def benchmark(self):
        return tensor(0.0)


class Easom(OptimBenchmark):
    def __init__(self, dims: int = None, **kwargs):
        super(Easom, self).__init__(dims=2, function=easom, **kwargs)

    @staticmethod
    def get_bounds(dims):
        bounds = [(-100.0, 100.0) for _ in range(dims)]
        return Bounds.from_tuples(bounds)

    @property
    def best_state(self):
        return judo.ones(self.shape) * np.pi

    @property
    def benchmark(self):
        return tensor(-1)


class HolderTable(OptimBenchmark):
    def __init__(self, dims: int = None, *args, **kwargs):
        super(HolderTable, self).__init__(dims=2, function=holder_table, *args, **kwargs)

    @staticmethod
    def get_bounds(dims):
        bounds = [(-10.0, 10.0) for _ in range(dims)]
        return Bounds.from_tuples(bounds)

    @property
    def best_state(self):
        return tensor([8.05502, 9.66459])

    @property
    def benchmark(self):
        return tensor(-19.2085)


class LennardJones(OptimBenchmark):
    # http://doye.chem.ox.ac.uk/jon/structures/LJ/tables.150.html
    minima = {
        "2": -1,
        "3": -3,
        "4": -6,
        "5": -9.103852,
        "6": -12.712062,
        "7": -16.505384,
        "8": -19.821489,
        "9": -24.113360,
        "10": -28.422532,
        "11": -32.765970,
        "12": -37.967600,
        "13": -44.326801,
        "14": -47.845157,
        "15": -52.322627,
        "20": -77.177043,
        "25": -102.372663,
        "30": -128.286571,
        "38": -173.928427,
        "50": -244.549926,
        "100": -557.039820,
        "104": -582.038429,
    }

    benchmark = None

    def __init__(self, n_atoms: int = 10, dims=None, **kwargs):
        self.n_atoms = n_atoms
        self.dims = 3 * n_atoms
        self.benchmark = [np.zeros(self.n_atoms * 3), self.minima.get(str(int(n_atoms)), 0)]
        super(LennardJones, self).__init__(dims=self.dims, function=lennard_jones, **kwargs)

    @staticmethod
    def get_bounds(dims):
        bounds = [(-15, 15) for _ in range(dims)]
        return Bounds.from_tuples(bounds)

    def __reset(self, **kwargs):
        states = super(LennardJones, self).reset()
        new_states = random_state.normal(0, scale=1.0, size=states.states.shape)
        states.update(observs=new_states, states=judo.copy(new_states))
        return states


ALL_BENCHMARKS = [Sphere, Rastrigin, EggHolder, StyblinskiTang, HolderTable, Easom]
