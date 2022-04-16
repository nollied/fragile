from typing import Callable, Tuple

import judo
from judo import Backend, Bounds, tensor, typing
import numpy
import numpy as np
from scipy.optimize import Bounds as ScipyBounds, minimize

from fragile.core.env import Function
from fragile.core.fractalai import relativize
from fragile.core.typing import StateData, StateDict
from fragile.optimize.utils import GradFuncWrapper


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


class ParticleSimulation(Function):
    default_inputs = {
        "actions": {},
        "observs": {"clone": True},
        "velocities": {"clone": True},
        "oobs": {"optional": True},
        "time": {"clone": True},
    }
    default_outputs = "observs", "rewards", "oobs", "velocities"

    def __init__(self, step_size=1.0, steps=10, **kwargs):
        self.step_size = step_size
        self.steps = steps
        super(ParticleSimulation, self).__init__(**kwargs)
        # self.grad_func = GradFuncWrapper(lambda theta: self.function(numpy.array([theta]))[0])
        self.grad_func = GradFuncWrapper(self.function)

    @classmethod
    def from_instance(
        cls,
        function: Function,
        step_size: float = 1,
        steps=1,
        x0=None,
        start_same_pos: bool = False,
    ):
        return cls(
            function=function.function,
            bounds=function.bounds,
            custom_domain_check=function.custom_domain_check,
            actions_as_perturbations=True,
            step_size=step_size,
            steps=steps,
            x0=x0,
            start_same_pos=start_same_pos,
        )

    @property
    def param_dict(self) -> StateDict:
        param_dict = super(ParticleSimulation, self).param_dict
        param_dict["velocities"] = dict(param_dict["observs"])
        param_dict["kinetic_energy"] = dict(param_dict["rewards"])
        param_dict["total_energy"] = dict(param_dict["rewards"])
        param_dict["time"] = dict(param_dict["rewards"])
        return param_dict

    def calculate_forces(self, observs, **kwargs):

        compas = observs[np.random.permutation(np.arange(len(observs)))]
        deltas = self.bounds.pbc_distance(observs, compas)
        distances = np.linalg.norm(deltas, axis=1)
        vectors = (deltas) / np.clip(distances.reshape(-1, 1), 1e-4, np.inf)
        r2 = relativize(distances)
        r2i = 1.0 / (r2**2)
        r6i = r2i * r2i * r2i
        epot = r6i * (r6i - 1.0) * 4
        forces = epot.reshape(-1, 1) * vectors
        return forces

    def fai_leapfrog(self, init_pos, init_momentum, grad, step_size, function, force=None):
        """Perfom a leapfrog jump in the Hamiltonian space
        INPUTS
        ------
        init_pos: ndarray[float, ndim=1]
            initial parameter position
        init_momentum: ndarray[float, ndim=1]
            initial momentum
        grad: float
            initial gradient value
        step_size: float
            step size
        function: callable
            it should return the log probability and gradient evaluated at theta
            logp, grad = function(theta)
        OUTPUTS
        -------
        new_position: ndarray[float, ndim=1]
            new parameter position
        new_momentum: ndarray[float, ndim=1]
            new momentum
        gradprime: float
            new gradient
        new_potential: float
            new lnp
        """
        potential, grad = function(init_pos)
        lj_force = 0  # self.calculate_forces(init_pos)
        pot = relativize(potential).reshape(-1, 1)
        grad = (grad / numpy.linalg.norm(grad, axis=1).reshape(-1, 1)) * pot
        # make half step in init_momentum
        new_momentum = init_momentum + 0.5 * step_size * (lj_force - grad + force)
        # make new step in theta
        new_position = init_pos + step_size * new_momentum
        new_position = self.bounds.pbc(new_position)
        # compute new gradient
        new_potential, gradprime = function(new_position)
        new_pot = relativize(new_potential).reshape(-1, 1)
        gradprime = (gradprime / numpy.linalg.norm(gradprime, axis=1).reshape(-1, 1)) * new_pot

        # make half step in init_momentum again
        new_force = 0  # self.calculate_forces(new_position)
        new_momentum = new_momentum + 0.5 * step_size * (new_force - gradprime + force)
        return new_position, new_momentum, gradprime, new_potential

    def step(self, actions, observs, velocities, **kwargs) -> StateData:
        """

        Sum the target action to the observations to obtain the new points, and \
        evaluate the reward and boundary conditions.

        Returns:
            Dictionary containing the information of the new points evaluated.

             ``{"states": new_points, "observs": new_points, "rewards": typing.Scalar array, \
             "oobs": boolean array}``

        """
        for _ in range(self.steps):
            _, grad = self.grad_func(observs)
            data = self.fai_leapfrog(
                init_pos=observs,
                init_momentum=velocities,
                grad=grad,
                step_size=self.step_size,
                function=self.grad_func,
                force=0,
            )
            new_position, velocities, gradprime, new_potential = data
            observs = self.bounds.pbc(judo.tensor(new_position))
        return dict(
            observs=observs,
            velocities=judo.tensor(velocities),
            rewards=judo.tensor(new_potential),
            oobs=judo.zeros(len(new_potential), dtype=judo.bool),
        )
