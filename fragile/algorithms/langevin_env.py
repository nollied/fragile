from typing import Callable, Dict

import numpy

from fragile.core.bounds import Bounds
from fragile.core.states import StatesEnv
from fragile.optimize.env import Function


class Langevin(Function):
    """
    Environment that represents an arbitrary mathematical function bounded in a \
    given interval.
    """

    def __init__(
        self,
        function: Callable[[numpy.ndarray], numpy.ndarray],
        derivative: Callable[[numpy.ndarray], numpy.ndarray],
        bounds: Bounds,
        custom_domain_check: Callable[[numpy.ndarray], numpy.ndarray] = None,
        brownian: bool = False,
        dt: float = 1.0,
        temperature: float = 0.5,
    ):
        """
        Initialize a :class:`Function`.

        Args:
            function: Callable that takes a batch of vectors (batched across \
                      the first dimension of the array) and returns a vector of \
                      scalar. This function is applied to a batch of walker \
                      observations.
            bounds: :class:`Bounds` that defines the domain of the function.
            custom_domain_check: Callable that checks points inside the bounds \
                    to know if they are in a custom domain when it is not just \
                    a set of rectangular bounds. It takes a batch of points as \
                    input and returns an array of booleans. Each ``True`` value \
                    indicates that the corresponding point is **outside**  the \
                    ``custom_domain_check``.
            brownian: Ignore the force term when updating the velocities.
            dt: Step size of the integration.
            temperature: Temperature parameter of the Langevin dynamics.
            derivative: Derivative of function.

        """
        super(Langevin, self).__init__(
            function=function, bounds=bounds, custom_domain_check=custom_domain_check
        )
        self.brownian = brownian
        self.derivative = derivative
        self.dt = dt
        self.temperature = temperature

    def make_transitions(
        self, observs: numpy.ndarray, actions: numpy.ndarray
    ) -> Dict[str, numpy.ndarray]:
        """

        Sum the target action to the observations to obtain the new points, and \
        evaluate the reward and boundary conditions.

        Args:
            observs: Batch of points returned in the last step.
            actions: Perturbation that will be applied to ``observs``.

        Returns:
            Dictionary containing the information of the new points evaluated.

             ``{"states": new_points, "observs": new_points, "rewards": scalar array, \
             "oobs": boolean array}``

        """
        sep = int(numpy.ceil(self.n_dims / 2))
        new_points = numpy.zeros_like(observs)
        positions = observs[:, :sep]
        force = -self.derivative(positions)
        velocities = force + numpy.sqrt(2 * self.temperature) * actions
        new_positions = positions + velocities * self.dt
        new_points[:, :sep] = new_positions
        new_points[:, sep:] = velocities
        oobs = self.calculate_oobs(points=new_points)
        rewards = self.function(new_positions).flatten()
        data = {"states": new_points, "observs": new_points, "rewards": rewards, "oobs": oobs}
        return data

    def reset(self, batch_size: int = 1, **kwargs) -> StatesEnv:
        """
        Reset the :class:`Function` to the start of a new episode and returns \
        an :class:`StatesEnv` instance describing its internal state.

        Args:
            batch_size: Number of walkers that the returned state will have.
            **kwargs: Ignored. This environment resets without using any external data.

        Returns:
            :class:`EnvStates` instance describing the state of the :class:`Function`. \
            The first dimension of the data tensors (number of walkers) will be \
            equal to batch_size.

        """
        oobs = numpy.zeros(batch_size, dtype=numpy.bool_)
        new_points = self.sample_bounds(batch_size=batch_size)
        potential = self.function(new_points[:, self.n_dims]).flatten()  # Take only positions.
        new_states = self.states_from_data(
            states=new_points,
            observs=new_points,
            rewards=potential,
            oobs=oobs,
            batch_size=batch_size,
        )
        return new_states

    def calculate_oobs(self, points: numpy.ndarray) -> numpy.ndarray:
        """
        Determine if a given batch of vectors lie inside the function domain.

        Args:
            points: Array of batched vectors that will be checked to lie inside \
                    the :class:`Function` bounds.

        Returns:
            Array of booleans of length batch_size (points.shape[0]) that will \
            be ``True`` if a given point of the batch lies outside the bounds, \
            and ``False`` otherwise.

        """
        oobs = numpy.logical_not(self.bounds.points_in_bounds(points)).flatten()
        if self.custom_domain_check is not None:
            points_in_bounds = numpy.logical_not(oobs)
            oobs[points_in_bounds] = self.custom_domain_check(points[points_in_bounds])
        return oobs

    def sample_bounds(self, batch_size: int) -> numpy.ndarray:
        """
        Return a matrix of points sampled uniformly from the :class:`Function` \
        domain.

        Args:
            batch_size: Number of points that will be sampled.

        Returns:
            Array containing ``batch_size`` points that lie inside the \
            :class:`Function` domain, stacked across the first dimension.

        """
        size = self.n_dims * 2
        new_points = numpy.zeros(tuple([batch_size]) + tuple([size]), dtype=numpy.float32)
        for i in range(batch_size):
            new_points[i, :] = self.random_state.uniform(
                low=self.bounds.low, high=self.bounds.high, size=size
            )
        return new_points
