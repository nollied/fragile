from typing import Callable, Tuple

import judo
from judo import Backend, Bounds, tensor, typing
import numpy
from scipy.optimize import Bounds as ScipyBounds, minimize

from fragile.core.env import Function


class Minimizer:
    """Apply ``scipy.optimize.minimize`` to a :class:`Function`."""

    def __init__(self, function: Function, bounds=None, **kwargs):
        """
        Initialize a :class:`Minimizer`.

        Args:
            function: :class:`Function` that will be minimized.
            bounds: :class:`Bounds` defining the domain of the minimization \
                    process. If it is ``None`` the :class:`Function` :class:`Bounds` \
                    will be used.
            *args: Passed to ``scipy.optimize.minimize``.
            **kwargs: Passed to ``scipy.optimize.minimize``.

        """
        self.env = function
        self.function = function.function
        self.bounds = self.env.bounds if bounds is None else bounds
        self.kwargs = kwargs

    def minimize(self, x: typing.Tensor):
        """
        Apply ``scipy.optimize.minimize`` to a single point.

        Args:
            x: Array representing a single point of the function to be minimized.

        Returns:
            Optimization result object returned by ``scipy.optimize.minimize``.

        """

        def _optimize(_x):
            try:
                _x = _x.reshape((1,) + _x.shape)
                y = self.function(_x)
            except (ZeroDivisionError, RuntimeError):
                y = numpy.inf
            return y

        bounds = ScipyBounds(
            ub=judo.to_numpy(self.bounds.high) if self.bounds is not None else None,
            lb=judo.to_numpy(self.bounds.low) if self.bounds is not None else None,
        )
        return minimize(_optimize, x, bounds=bounds, **self.kwargs)

    def minimize_point(self, x: typing.Tensor) -> Tuple[typing.Tensor, typing.Scalar]:
        """
        Minimize the target function passing one starting point.

        Args:
            x: Array representing a single point of the function to be minimized.

        Returns:
            Tuple containing a numpy array representing the best solution found, \
            and the numerical value of the function at that point.

        """
        optim_result = self.minimize(x)
        point = tensor(optim_result["x"])
        reward = tensor(float(optim_result["fun"]))
        return point, reward

    def minimize_batch(self, x: typing.Tensor) -> Tuple[typing.Tensor, typing.Tensor]:
        """
        Minimize a batch of points.

        Args:
            x: Array representing a batch of points to be optimized, stacked \
               across the first dimension.

        Returns:
            Tuple of arrays containing the local optimum found for each point, \
            and an array with the values assigned to each of the points found.

        """
        x = judo.to_numpy(judo.copy(x))
        with Backend.use_backend("numpy"):
            result = judo.zeros_like(x)
            rewards = judo.zeros((x.shape[0], 1))
            for i in range(x.shape[0]):
                new_x, reward = self.minimize_point(x[i, :])
                result[i, :] = new_x
                rewards[i, :] = float(reward)
        self.bounds.high = tensor(self.bounds.high)
        self.bounds.low = tensor(self.bounds.low)
        result, rewards = tensor(result), tensor(rewards)
        return result, rewards


class MinimizerWrapper(Function):
    """
    Wrapper that applies a local minimization process to the observations \
    returned by a :class:`Function`.
    """

    def __init__(self, function: Function, **kwargs):
        """
        Initialize a :class:`MinimizerWrapper`.

        Args:
            function: :class:`Function` to be minimized after each step.
            *args: Passed to the internal :class:`Optimizer`.
            **kwargs: Passed to the internal :class:`Optimizer`.

        """
        self.unwrapped = function
        self.minimizer = Minimizer(function=self.unwrapped, **kwargs)

    @property
    def shape(self) -> Tuple[int, ...]:
        """Return the shape of the wrapped environment."""
        return self.unwrapped.shape

    @property
    def function(self) -> Callable:
        """Return the function of the wrapped environment."""
        return self.unwrapped.function

    @property
    def bounds(self) -> Bounds:
        """Return the bounds of the wrapped environment."""
        return self.unwrapped.bounds

    @property
    def custom_domain_check(self) -> Callable:
        """Return the custom_domain_check of the wrapped environment."""
        return self.unwrapped.custom_domain_check

    def __getattr__(self, item):
        return getattr(self.unwrapped, item)

    def __repr__(self):
        return self.unwrapped.__repr__()

    def setup(self, swarm):
        self.unwrapped.setup(swarm)

    def make_transitions(self, inplace: bool = True, **kwargs):
        """
        Perform a local optimization process to the observations returned after \
        calling ``make_transitions`` on the wrapped :class:`Function`.
        """
        function_data = self.unwrapped.make_transitions(inplace=False, **kwargs)
        new_points, rewards = self.minimizer.minimize_batch(function_data.get("observs"))
        # new_points, rewards = tensor(new_points), tensor(rewards)
        oobs = self.calculate_oobs(new_points, rewards)
        new_data = dict(observs=new_points, rewards=rewards.flatten(), oobs=oobs)
        if inplace:
            self.update(**new_data)
        else:
            return new_data
