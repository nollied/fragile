import copy
from typing import Any, Dict, Iterable, Optional, Set, Tuple, Union

import judo
from judo import dtype, random_state

from fragile.core.state import SwarmState
from fragile.core.typing import InputDict, StateData, StateDict, Tensor


class SwarmComponent:
    """Every class that stores its data in :class:`SwarmState` inherits  from this class."""

    default_inputs = {}
    default_outputs = tuple()
    default_param_dict = {}

    def __init__(
        self,
        swarm: Optional["SwarmAPI"] = None,
        param_dict: Optional[StateDict] = None,
        inputs: Optional[InputDict] = None,
        outputs: Optional[Tuple[str]] = None,
    ):
        param_dict = param_dict or {}
        param_dict = {**self.default_param_dict, **param_dict}
        inputs = inputs or {}
        inputs = {**self.default_inputs, **inputs}
        outputs = tuple(outputs) if outputs is not None else tuple()
        outputs = tuple(set(self.default_outputs + outputs))
        self._swarm = None
        self._param_dict = param_dict
        self._inputs = inputs
        self._outputs = outputs
        if swarm is not None:  # This way you can run side effects on child classes
            self.setup(swarm)

    @property
    def swarm(self) -> "SwarmAPI":
        return self._swarm

    @property
    def n_walkers(self) -> int:
        return self.swarm.n_walkers

    @property
    def inputs(self) -> InputDict:
        return dict(self._inputs)

    @property
    def outputs(self) -> Tuple[str, ...]:
        return tuple(self._outputs)

    @property
    def param_dict(self) -> StateDict:
        return dict(self._param_dict)

    def setup(self, swarm):
        self._swarm = swarm

    def get(self, name: str, default: Any = None, raise_error: bool = False) -> Any:
        """Access attributes of the :class:`Swarm` and its children."""
        return self.swarm.state.get(name=name, default=default, raise_error=raise_error)

    def get_input_data(self) -> StateData:
        def get_one_input(name, values):
            return self.get(name, values.get("default"), not values.get("optional", False))

        return {k: get_one_input(k, v) for k, v in self.inputs.items()}

    def update(self, other: Union["SwarmState", Dict[str, Tensor]] = None, **kwargs) -> None:
        """
        Modify the data stored in the SwarmState instance.

        Existing attributes will be updated, and new attributes will be created if needed.

        Args:
            other: State class that will be copied upon update.
            **kwargs: It is possible to specify the update as name value attributes,
                where name is the name of the attribute to be updated, and value
                is the new value for the attribute.
        Returns:
            None
        """
        return self.swarm.state.update(other=other, **kwargs)

    def _prepare_tensors(self, **kwargs):
        if kwargs:
            step_data = kwargs
        else:
            step_data = self.get_input_data()
        return step_data

    def reset(
        self,
        inplace: bool = True,
        root_walker: Optional[StateData] = None,
        states: Optional[StateData] = None,
        **kwargs,
    ):
        pass


class EnvironmentAPI(SwarmComponent):
    """
    The Environment is in charge of stepping the walkers, acting as a state \
    transition function.

    For every different problem, a new Environment needs to be implemented
    following the :class:`EnvironmentAPI` interface.

    """

    default_inputs = {"actions": {}}
    default_outputs = "observs", "rewards", "oobs"

    def __init__(
        self,
        action_shape,
        action_dtype,
        observs_shape,
        observs_dtype,
        swarm: "SwarmAPI" = None,
    ):
        self._action_shape = action_shape
        self._action_dtype = action_dtype
        self._observs_shape = observs_shape
        self._observs_dtype = observs_dtype
        super(EnvironmentAPI, self).__init__(
            swarm=swarm,
            param_dict=self.param_dict,
        )

    @property
    def observs_shape(self) -> tuple:
        return self._observs_dtype

    @property
    def observs_dtype(self):
        return self._observs_dtype

    @property
    def action_shape(self):
        return self._action_shape

    @property
    def action_dtype(self):
        return self._action_dtype

    @property
    def param_dict(self) -> StateDict:
        param_dict = {
            "observs": {"shape": self._observs_shape, "dtype": self._observs_dtype},
            "rewards": {"dtype": dtype.float32},
            "oobs": {"dtype": dtype.bool},
            "actions": {"shape": self._action_shape, "dtype": self._action_dtype},
        }
        return param_dict

    def step(self, **kwargs) -> StateData:
        raise NotImplementedError()

    def make_transitions(self, inplace: bool = True, **kwargs) -> Union[None, StateData]:
        """
        Return the data corresponding to the new state of the environment after \
        using the input data to make the corresponding state transition.

        Args:
            inplace: If False return the new data. If True, update the state of the Swarm.
            **kwargs: Keyword arguments passed if the returned value from the
                ``states_to_data`` function of the class was a dictionary.

        Returns:
            Dictionary containing the data representing the state of the environment
            after the state transition. The keys of the dictionary are the names of
            the data attributes and its values are arrays representing a batch of
            new values for that attribute.

            The :class:`StatesEnv` returned by ``step`` will contain the returned
            data.

        """
        input_data = self._prepare_tensors(**kwargs)
        out_data = self.step(**input_data)
        if inplace:
            self.update(**out_data)
        return out_data

    def reset(
        self,
        inplace: bool = True,
        root_walker: Optional[StateData] = None,
        states: Optional[StateData] = None,
        **kwargs,
    ) -> Union[None, StateData]:
        return self.make_transitions(inplace=inplace, **kwargs)


class PolicyAPI(SwarmComponent):
    """The policy is in charge of calculating the interactions with the :class:`Environment`."""

    default_outputs = tuple(["actions"])

    def select_actions(self, **kwargs) -> Union[Tensor, StateData]:
        raise NotImplementedError

    def act(self, inplace: bool = True, **kwargs) -> Union[None, StateData]:
        """Calculate SwarmState containing the data needed to interact with the environment."""
        action_input = self._prepare_tensors(**kwargs)
        actions_data = self.select_actions(**action_input)
        if not isinstance(actions_data, dict):
            actions_data = {"actions": actions_data}
        if inplace:
            self.update(**actions_data)
        else:
            return actions_data

    def reset(
        self,
        inplace: bool = True,
        root_walker: Optional[StateData] = None,
        states: Optional[StateData] = None,
        **kwargs,
    ) -> Union[None, StateData]:
        # TODO: only run act when inputs are not present in root_walker/states
        # if root_walker is None and states is None:
        return self.act(inplace=inplace, **kwargs)


class Callback(SwarmComponent):
    """
    The :class:`Walkers` is a data structure that takes care of all the data involved \
    in making a Swarm evolve.
    """

    name = None

    def before_reset(self):
        pass

    def after_reset(self):
        pass

    def before_evolve(self):
        pass

    def after_evolve(self):
        pass

    def before_policy(self):
        pass

    def after_policy(self):
        pass

    def before_env(self):
        pass

    def after_env(self):
        pass

    def before_walkers(self):
        pass

    def after_walkers(self):
        pass

    def reset(
        self,
        inplace: bool = True,
        root_walker: Optional[StateData] = None,
        states: Optional[StateData] = None,
        **kwargs,
    ):
        pass

    def evolution_end(self) -> bool:
        return False

    def run_end(self):
        pass


class WalkersMetric(SwarmComponent):
    def __call__(self, inplace: bool = True, **kwargs) -> Tensor:
        input_data = self._prepare_tensors(**kwargs)
        out_data = self.calculate(**input_data)
        if inplace:
            self.update(**out_data)
        return out_data

    def calculate(self, **kwargs):
        raise NotImplementedError()

    def reset(
        self,
        inplace: bool = True,
        root_walker: Optional[StateData] = None,
        states: Optional[StateData] = None,
        **kwargs,
    ):
        pass


class WalkersAPI(SwarmComponent):
    def balance(self, inplace: bool = True, **kwargs) -> Union[None, StateData]:
        input_data = self._prepare_tensors(**kwargs)
        out_data = self.run_epoch(inplace=inplace, **input_data)
        if inplace:
            self.update(**out_data)
        return out_data

    def run_epoch(self, inplace: bool = True, **kwargs) -> StateData:
        raise NotImplementedError()

    def reset(self, inplace: bool = True, **kwargs):
        pass

    def get_in_bounds_compas(self, oobs=None) -> Tensor:
        """
        Return an array of indexes corresponding to an alive walker chosen \
        at random.
        """
        n_walkers = len(oobs) if oobs is not None else self.swarm.n_walkers
        indexes = judo.arange(n_walkers, dtype=int)
        if oobs is None or oobs.all():  # No need to sample if all walkers are dead.
            return indexes
        alive_indexes = indexes[~oobs]
        compas_ix = random_state.permutation(alive_indexes)
        compas = random_state.choice(compas_ix, len(oobs), replace=True)
        compas[: len(compas_ix)] = compas_ix
        return compas

    def clone_walkers(self, will_clone=None, compas_clone=None, **kwargs):
        """Sample the clone probability distribution and clone the walkers accordingly."""
        self.swarm.state.clone(
            will_clone=will_clone,
            compas_clone=compas_clone,
            clone_names=self.swarm.clone_names,
        )


class SwarmAPI:
    """
    The Swarm implements the iteration logic to make the :class:`Walkers` evolve.

    It contains the necessary logic to use an Environment, a Model, and a \
    Walkers instance to create the algorithm execution loop.
    """

    walkers_last = True

    def __init__(
        self,
        n_walkers: int,
        env: EnvironmentAPI,
        policy: PolicyAPI,
        walkers: WalkersAPI,
        callbacks: Optional[Iterable[Callback]] = None,
        minimize: bool = False,
        max_epochs: int = 1e100,
    ):
        """Initialize a :class:`SwarmAPI`."""
        self.minimize = minimize
        self.max_epochs = int(max_epochs)
        self._n_walkers = n_walkers
        self._epoch = 0
        self._env = env
        self._policy = policy
        self._walkers = walkers
        self._state = None
        self._inputs = {}
        self._clone_names = set()
        self._callbacks = {}
        callbacks = callbacks if callbacks is not None else []
        for callback in callbacks:
            self.register_callback(callback, setup=False)
        self.setup()

    @property
    def n_walkers(self) -> int:
        return self._n_walkers

    @property
    def epoch(self) -> int:
        """Return the current epoch of the search algorithm."""
        return self._epoch

    @property
    def state(self) -> SwarmState:
        return self._state

    @property
    def env(self) -> EnvironmentAPI:
        """All the simulation code (problem specific) will be handled here."""
        return self._env

    @property
    def policy(self) -> PolicyAPI:
        """
        All the policy and random perturbation code (problem specific) will \
        be handled here.
        """
        return self._policy

    @property
    def walkers(self) -> WalkersAPI:
        """Access the :class:`Walkers` in charge of implementing the FAI evolution process."""
        return self._walkers

    @property
    def callbacks(self) -> Dict[str, Callback]:
        return self._callbacks

    @property
    def param_dict(self) -> StateDict:
        return copy.deepcopy(self.state.param_dict)

    @property
    def clone_names(self) -> Set[str]:
        return self._clone_names

    @property
    def inputs(self) -> dict:
        return self._inputs

    def __len__(self) -> int:
        return self.n_walkers

    def __getattr__(self, item):
        if item in self.callbacks:
            return self.callbacks[item]
        return super(SwarmAPI, self).__getattribute__(item)

    def setup_state(self, param_dict: StateDict, n_walkers: Optional[int] = None):
        if n_walkers is not None:
            self._n_walkers = n_walkers
        self._state = SwarmState(n_walkers=self.n_walkers, param_dict=param_dict)
        self._state.reset()

    def setup(self) -> None:
        self._setup_components()
        self._setup_clone_names()
        self._setup_inputs()

    def register_callback(self, callback, setup: bool = True):
        """Increment the current epoch counter."""
        if setup:
            callback.setup(self)
            new_param_dict = {**self.param_dict, **callback.param_dict}
            self.setup_state(n_walkers=self.n_walkers, param_dict=new_param_dict)
        self.callbacks[callback.name] = callback
        clone_names = [k for k, v in callback.inputs.items() if v.get("clone")]
        self._clone_names = set(list(self.clone_names) + clone_names)

    def get(self, name: str, default: Any = None, raise_error: bool = False) -> Any:
        """Access attributes of the :class:`Swarm` and its children."""
        return self.state.get(name=name, default=default, raise_error=raise_error)

    def reset(
        self,
        root_walker: Optional["OneWalker"] = None,
        state: Optional[SwarmState] = None,
    ):
        """
        Reset a :class:`fragile.Swarm` and clear the internal data to start a \
        new search process.

        Args:
            root_walker: Walker representing the initial state of the search.
                The walkers will be reset to this walker, and it will
                be added to the root of the :class:`StateTree` if any.
            state: :class:`SwarmState` that define the initial state of the Swarm.
        """
        self.state.reset(root_walker=root_walker)
        if not self.walkers_last:
            self.walkers.reset(root_walker=root_walker)
        self.env.reset(root_walker=root_walker)
        self.policy.reset(root_walker=root_walker)
        if self.walkers_last:
            self.walkers.reset(root_walker=root_walker)
        for callback in self.callbacks.values():
            callback.reset(root_walker=root_walker)
        self._epoch = 0

    def run(
        self,
        root_walker: Optional[StateData] = None,
        state: Optional[StateData] = None,
    ) -> None:
        """
        Run a new search process until the stop condition is met.

        Args:
            root_walker: Walker representing the initial state of the search.
                The walkers will be reset to this walker, and it will
                be added to the root of the :class:`StateTree` if any.
            state: StateData dictionary that define the initial state of the Swarm.

        Returns:
            None.

        """
        self.before_reset()
        self.reset(root_walker=root_walker)
        self.after_reset()
        while not self.evolution_end():
            self.before_evolve()
            self.evolve()
            self.after_evolve()
        self.run_end()

    def evolution_end(self) -> bool:
        return (
            (self.epoch >= self.max_epochs)
            or self.get("oobs").all()
            or any(c.evolution_end() for c in self.callbacks.values())
        )

    def evolve(self):
        """
        Make the walkers undergo a perturbation process in the swarm \
        :class:`Environment`.

        This function updates the :class:`StatesEnv` and the :class:`StatesModel`.
        """
        if not self.walkers_last:
            self.before_walkers()
            self.walkers.balance()
            self.after_walkers()
        self.before_policy()
        self.policy.act()
        self.after_policy()
        self.before_env()
        self.env.make_transitions()
        self.after_env()
        if self.walkers_last:
            self.before_walkers()
            self.walkers.balance()
            self.after_walkers()
        self._epoch += 1

    def before_reset(self):
        for callback in self.callbacks.values():
            callback.before_reset()

    def after_reset(self):
        for callback in self.callbacks.values():
            callback.after_reset()

    def before_policy(self):
        for callback in self.callbacks.values():
            callback.before_policy()

    def after_policy(self):
        for callback in self.callbacks.values():
            callback.after_policy()

    def before_env(self):
        for callback in self.callbacks.values():
            callback.before_env()

    def after_env(self):
        for callback in self.callbacks.values():
            callback.after_env()

    def before_walkers(self):
        for callback in self.callbacks.values():
            callback.before_walkers()

    def after_walkers(self):
        for callback in self.callbacks.values():
            callback.after_walkers()

    def before_evolve(self):
        for callback in self.callbacks.values():
            callback.before_evolve()

    def after_evolve(self):
        for callback in self.callbacks.values():
            callback.after_evolve()

    def run_end(self):
        for callback in self.callbacks.values():
            callback.run_end()
