from typing import Optional, Union

import judo
from judo import Backend, Bounds, dtype, random_state, tensor
import numpy.linalg

from fragile.core.api_classes import PolicyAPI, SwarmAPI
from fragile.core.typing import StateData, Tensor


class RandomPlangym(PolicyAPI):
    def setup(self, swarm):
        super(RandomPlangym, self).setup(swarm)
        if hasattr(self.swarm.env, "sample_action"):
            sample = self.swarm.env.sample_action
        elif hasattr(self.swarm.env, "action_space"):
            sample = self.swarm.env.action_space.select_dt
        elif hasattr(self.swarm.env, "plangym_env"):
            penv = self.swarm.env.plangym_env
            if hasattr(penv, "sample_action"):
                sample = penv.sample_action
            elif hasattr(penv, "action_space"):
                sample = penv.action_space.select_dt
        else:
            raise TypeError("Environment does not have a sample_action method or an action_space")
        self._sample_func = sample

    def select_actions(self, **kwargs):
        a = [self._sample_func() for _ in range(len(self.swarm))]
        return a


class Discrete(PolicyAPI):
    def __init__(self, actions: int = None, probs: Optional[Tensor] = None, **kwargs):
        super(Discrete, self).__init__(**kwargs)
        # TODO: parse all possible ways to infer actions, probs, and n_actions
        self.probs = probs
        self._n_actions = None
        self._actions = actions
        self._setup_params(actions)

    @property
    def n_actions(self):
        return self._n_actions

    @property
    def actions(self):
        return self._actions

    def select_actions(self, **kwargs) -> Tensor:
        return random_state.choice(self.actions, p=self.probs, size=self.swarm.n_walkers)

    def setup(self, swarm: SwarmAPI):
        self._swarm = swarm
        if self.n_actions is None:
            if hasattr(self.swarm.env, "n_actions"):
                self._n_actions = self.swarm.env.n_actions
            elif hasattr(self.swarm.env, "action_space"):
                self._n_actions = self.swarm.env.action_space.n
            else:
                raise TypeError("n_actions cannot be inferred.")
        self._setup_params(self.actions)

    def _setup_params(self, actions):
        if actions is None and self.n_actions is None and self.actions is not None:
            self._n_actions = len(self.actions)  # Try to set up n_actions using actions
        elif actions is None and self.n_actions is not None:  # setup actions with n_actions
            self._actions = judo.arange(self.n_actions) if self.actions is None else self.actions
        elif isinstance(actions, (list, tuple)) or judo.is_tensor(actions):
            self._actions = tensor(actions)
            self._n_actions = len(self.actions)
        elif actions is not None:  # Actions is an integer-like value
            self._n_actions = self.n_actions if self.n_actions is not None else int(actions)
            self._actions = judo.arange(self._n_actions)
        elif self.probs is not None:  # Try to infer values from probs
            self._n_actions = len(self.probs)
            self._actions = judo.arange(self._n_actions)


class BinarySwap(PolicyAPI):
    def __init__(self, n_swaps: int, n_actions: int = None, **kwargs):
        self._n_actions = n_actions
        self._n_swaps = n_swaps
        super(BinarySwap, self).__init__(**kwargs)

    @property
    def n_actions(self):
        return self._n_actions

    @property
    def n_swaps(self):
        return self._n_swaps

    def select_actions(self, **kwargs):
        from numba import jit
        import numpy

        @jit(nopython=True)
        def flip_values(actions: numpy.ndarray, flips: numpy.ndarray):
            for i in range(flips.shape[0]):
                for j in range(flips.shape[1]):
                    actions[i, flips[i, j]] = numpy.logical_not(actions[i, flips[i, j]])
            return actions

        observs = judo.to_numpy(self.get("observs"))
        with Backend.use_backend("numpy"):
            actions = judo.astype(observs, dtype.bool)
            flips = random_state.randint(0, self.n_actions, size=(observs.shape[0], self.n_swaps))
            actions = judo.astype(flip_values(actions, flips), dtype.int64)
        actions = tensor(actions)
        return actions

    def setup(self, swarm: SwarmAPI):
        self._swarm = swarm
        if self.n_actions is None:
            if hasattr(self.swarm.env, "n_actions"):
                self._n_actions = self.swarm.env.n_actions
            elif hasattr(self.swarm.env, "action_space"):
                self._n_actions = self.swarm.env.action_space.n
            else:
                raise TypeError("n_actions cannot be inferred.")


class ContinuousPolicy(PolicyAPI):
    def __init__(self, bounds=None, second_order: bool = False, step: float = 1.0, **kwargs):
        self.second_order = second_order
        self.step = step
        self.bounds = bounds
        self._env_bounds = None
        if second_order:
            kwargs["inputs"] = {"actions": {"clone": True}, **kwargs.get("inputs", {})}
        super(ContinuousPolicy, self).__init__(**kwargs)

    @property
    def env_bounds(self) -> Bounds:
        return self._env_bounds

    def select_actions(self, **kwargs):
        raise NotImplementedError()

    def act(self, inplace: bool = True, **kwargs) -> Union[None, StateData]:
        """Calculate SwarmState containing the data needed to interact with the environment."""
        action_input = self._prepare_tensors(**kwargs)
        actions_data = self.select_actions(**action_input)
        if not isinstance(actions_data, dict):
            actions_data = {"actions": actions_data}
        actions = actions_data["actions"]
        if self.second_order:
            prev_actions = action_input["actions"]
            actions = prev_actions + actions * self.step
        actions_data["actions"] = self.env_bounds.clip(actions)
        if inplace:
            self.update(**actions_data)
        else:
            return actions_data

    def setup(self, swarm):
        super(ContinuousPolicy, self).setup(swarm)
        if self.bounds is None:
            if hasattr(self.swarm.env, "bounds"):
                self.bounds = self.swarm.env.bounds
            elif hasattr(self.swarm.env, "action_space"):
                self.bounds = Bounds.from_space(self.swarm.env.action_space)
            else:
                raise ValueError("Bounds is not defined and not present in the Environment.")
        if hasattr(self.swarm.env, "bounds"):
            self._env_bounds = self.swarm.env.bounds
        elif hasattr(self.swarm.env, "action_space"):
            self._env_bounds = Bounds.from_space(self.swarm.env.action_space)
        else:
            self._env_bounds = self.bounds


class Uniform(ContinuousPolicy):
    def select_actions(self, **kwargs) -> Tensor:
        shape = tuple([self.swarm.n_walkers]) + self.bounds.shape
        new_points = random_state.uniform(
            low=self.bounds.low,
            high=self.bounds.high,
            size=shape,
        )
        return new_points


class Gaussian(ContinuousPolicy):
    def __init__(self, loc: float = 0.0, scale: float = 1.0, **kwargs):
        super(Gaussian, self).__init__(**kwargs)
        self.loc = loc
        self.scale = scale

    def select_actions(self, **kwargs) -> Tensor:
        shape = tuple([self.swarm.n_walkers]) + self.bounds.shape
        new_points = random_state.normal(
            loc=self.loc,
            scale=self.scale,
            size=shape,
        )
        return new_points


class GaussianModulus(ContinuousPolicy):
    def __init__(self, loc: float = 0.0, scale: float = 1.0, **kwargs):
        super(GaussianModulus, self).__init__(**kwargs)
        self.loc = loc
        self.scale = scale

    def select_actions(self, **kwargs) -> Tensor:
        shape = tuple([self.swarm.n_walkers]) + self.bounds.shape
        new_points = random_state.uniform(
            low=-1.0,
            high=1.0,
            size=shape,
        )
        new_points = new_points / numpy.linalg.norm(new_points, axis=1).reshape(-1, 1)

        modulus = random_state.normal(
            loc=self.loc,
            scale=self.scale,
            size=(self.swarm.n_walkers, 1),
        )

        return new_points * modulus
