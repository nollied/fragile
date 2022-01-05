from typing import Callable, Tuple, Union

import judo
from judo import Backend, Bounds, dtype, random_state, tensor, typing
import numpy
from scipy.optimize import Bounds as ScipyBounds, minimize

from fragile.new_core.api_classes import EnvironmentAPI
from fragile.new_core.typing import StateData, StateDict


class Function(EnvironmentAPI):
    """
    Environment that represents an arbitrary mathematical function bounded in a \
    given interval.
    """

    default_inputs = {"actions": {}, "observs": {"clone": True}}

    def __init__(
        self,
        function: Callable[[typing.Tensor], typing.Tensor],
        bounds: Union[Bounds, "gym.spaces.box.Box"],
        custom_domain_check: Callable[[typing.Tensor, typing.Tensor, int], typing.Tensor] = None,
        actions_as_perturbations: bool = True,
    ):
        """
        Initialize a :class:`Function`.

        Args:
            function: Callable that takes a batch of vectors (batched across \
                      the first dimension of the array) and returns a vector of \
                      typing.Scalar. This function is applied to a batch of walker \
                      observations.
            bounds: :class:`Bounds` that defines the domain of the function.
            custom_domain_check: Callable that checks points inside the bounds \
                    to know if they are in a custom domain when it is not just \
                    a set of rectangular bounds. It takes a batch of points as \
                    input and returns an array of booleans. Each ``True`` value \
                    indicates that the corresponding point is **outside**  the \
                    ``custom_domain_check``.
            actions_as_perturbations: If ``True`` the actions are interpreted as \
                    perturbations that will be applied to the past states. \
                    If ``False`` the actions are interpreted as the new states to \
                    be evaluated.

        """

        if not isinstance(bounds, (Bounds)) and bounds.__class__.__name__ != "Box":
            raise TypeError(f"Bounds needs to be an instance of Bounds or Box, found {bounds}")
        self.function = function
        self.bounds = bounds if isinstance(bounds, Bounds) else Bounds.from_space(bounds)
        self._action_space = self.bounds.to_space()
        self.custom_domain_check = custom_domain_check
        self._actions_as_perturbations = actions_as_perturbations
        super(Function, self).__init__(
            observs_shape=self.shape,
            observs_dtype=dtype.float32,
            action_dtype=dtype.float32,
            action_shape=self.shape,
        )

    @property
    def n_dims(self) -> int:
        """Return the number of dimensions of the function to be optimized."""
        return len(self.bounds)

    @property
    def shape(self) -> Tuple[int, ...]:
        """Return the shape of the environment."""
        return self.bounds.shape

    @property
    def action_space(self) -> "gym.spaces.box.Box":
        """Action space with the same characteristics as self.bounds."""
        return self._action_space

    @classmethod
    def from_bounds_params(
        cls,
        function: Callable,
        shape: tuple = None,
        high: Union[int, float, typing.Tensor] = numpy.inf,
        low: Union[int, float, typing.Tensor] = numpy.NINF,
        custom_domain_check: Callable[[typing.Tensor], typing.Tensor] = None,
    ) -> "Function":
        """
        Initialize a function defining its shape and bounds without using a :class:`Bounds`.

        Args:
            function: Callable that takes a batch of vectors (batched across \
                      the first dimension of the array) and returns a vector of \
                      typing.Scalar. This function is applied to a batch of walker \
                      observations.
            shape: Input shape of the solution vector without taking into account \
                    the batch dimension. For example, a two dimensional function \
                    applied to a batch of 5 walkers will have shape=(2,), even though
                    the observations will have shape (5, 2)
            high: Upper bound of the function domain. If it's an typing.Scalar it will \
                  be the same for all dimensions. If its a numpy array it will \
                  be the upper bound for each dimension.
            low: Lower bound of the function domain. If it's an typing.Scalar it will \
                  be the same for all dimensions. If its a numpy array it will \
                  be the lower bound for each dimension.
            custom_domain_check: Callable that checks points inside the bounds \
                    to know if they are in a custom domain when it is not just \
                    a set of rectangular bounds.

        Returns:
            :class:`Function` with its :class:`Bounds` created from the provided arguments.

        """
        if (
            not (judo.is_tensor(high) or isinstance(high, (list, tuple)))
            and not (judo.is_tensor(low) or isinstance(low, (list, tuple)))
            and shape is None
        ):
            raise TypeError("Need to specify shape or high or low must be a numpy array.")
        bounds = Bounds(high=high, low=low, shape=shape)
        return Function(function=function, bounds=bounds, custom_domain_check=custom_domain_check)

    def __repr__(self):
        text = "{} with function {}, obs shape {},".format(
            self.__class__.__name__,
            self.function.__name__,
            self.shape,
        )
        return text

    def step(self, actions, observs, **kwargs) -> StateData:
        """

        Sum the target action to the observations to obtain the new points, and \
        evaluate the reward and boundary conditions.

        Returns:
            Dictionary containing the information of the new points evaluated.

             ``{"states": new_points, "observs": new_points, "rewards": typing.Scalar array, \
             "oobs": boolean array}``

        """
        new_points = actions + observs if self._actions_as_perturbations else actions
        rewards = self.function(new_points).flatten()
        oobs = self.calculate_oobs(points=new_points, rewards=rewards)
        return dict(observs=new_points, rewards=rewards, oobs=oobs)

    def reset(self, inplace: bool = True, **kwargs) -> Union[None, StateData]:
        """
        Reset the :class:`Function` to the start of a new episode and returns \
        an :class:`StatesEnv` instance describing its internal state.
        """
        n_walkers = self.swarm.n_walkers
        oobs = judo.zeros(n_walkers, dtype=judo.bool)
        new_points = self.sample_action(batch_size=n_walkers)
        actions = judo.zeros_like(new_points) if self._actions_as_perturbations else new_points
        rewards = self.function(new_points).flatten()
        return dict(observs=new_points, rewards=rewards, oobs=oobs, actions=actions)

    def calculate_oobs(self, points: typing.Tensor, rewards: typing.Tensor) -> typing.Tensor:
        """
        Determine if a given batch of vectors lie inside the function domain.

        Args:
            points: Array of batched vectors that will be checked to lie inside \
                    the :class:`Function` bounds.
            rewards: Array containing the rewards of the current walkers.

        Returns:
            Array of booleans of length batch_size (points.shape[0]) that will \
            be ``True`` if a given point of the batch lies outside the bounds, \
            and ``False`` otherwise.

        """
        oobs = judo.logical_not(self.bounds.contains(points)).flatten()
        if self.custom_domain_check is not None:
            points_in_bounds = judo.logical_not(oobs)
            oobs[points_in_bounds] = self.custom_domain_check(
                points[points_in_bounds],
                rewards[points_in_bounds],
                len(rewards),
            )
        return oobs

    def sample_action(self, batch_size: int) -> typing.Tensor:
        """
        Return a matrix of points sampled uniformly from the :class:`Function` \
        domain.

        Args:
            batch_size: Number of points that will be sampled.

        Returns:
            Array containing ``batch_size`` points that lie inside the \
            :class:`Function` domain, stacked across the first dimension.

        """
        shape = tuple([batch_size]) + self.shape
        new_points = random_state.uniform(
            low=judo.astype(self.bounds.low, judo.float),
            high=judo.astype(self.bounds.high, judo.float32),
            size=shape,
        )
        return new_points


class Minimizer:
    """Apply ``scipy.optimize.minimize`` to a :class:`Function`."""

    def __init__(self, function: Function, bounds=None, *args, **kwargs):
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
        self.args = args
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
        return minimize(_optimize, x, bounds=bounds, *self.args, **self.kwargs)

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

    def __init__(self, function: Function, *args, **kwargs):
        """
        Initialize a :class:`MinimizerWrapper`.

        Args:
            function: :class:`Function` to be minimized after each step.
            *args: Passed to the internal :class:`Optimizer`.
            **kwargs: Passed to the internal :class:`Optimizer`.

        """
        self.unwrapped = function
        self.minimizer = Minimizer(function=self.unwrapped, *args, **kwargs)

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

    def make_transitions(self):
        """
        Perform a local optimization process to the observations returned after \
        calling ``make_transitions`` on the wrapped :class:`Function`.
        """
        self.unwrapped.make_transitions()
        new_points, rewards = self.minimizer.minimize_batch(self.get("observs"))
        # new_points, rewards = tensor(new_points), tensor(rewards)
        oobs = self.calculate_oobs(new_points, rewards)
        self.update(observs=new_points, rewards=rewards.flatten(), oobs=oobs)
