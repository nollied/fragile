import copy
from typing import Callable, Optional

from judo.data_structures.bounds import Bounds
import numpy
import pandas

from fragile.core.api_classes import SwarmAPI
from fragile.core.env import Function
from fragile.core.state import SwarmState


class Swarm(SwarmAPI):
    def __repr__(self):
        text = f"{self.__class__.__name__}: walkers {self.n_walkers} iteration {self.epoch}\n"
        with numpy.printoptions(linewidth=100, threshold=200, edgeitems=9):
            if hasattr(self, "root"):
                text += self.root.__repr__()
            text += f"\nState statistics:\n{self._state_stats_df().round(3)}"
        return text

    def to_html(self):
        line_break = '<br style="line-height:1px; content: "  ";>'
        text = (
            f"<strong>{self.__class__.__name__}</strong>: "
            f"walkers {self.n_walkers} iteration {self.epoch}\n"
        )
        with numpy.printoptions(linewidth=100, threshold=200, edgeitems=9):
            if hasattr(self, "root"):
                text += self.root.to_html()
            if hasattr(self, "tree"):
                text += self.tree.to_html()
            text += "\n<strong>State statistics</strong>:\n"
            text = text.replace("\n\n", "\n").replace("\n", line_break)
            text += f"\n{self._state_stats_df().round(3).to_html()}"
        return text

    def _state_stats_df(self):
        skip_print = {"observs", "states", "id_walkers", "parent_ids", "infos", "compas_clone"}
        data = {
            k: v
            for k, v in self.state.items()
            if k in self.state.vector_names and k not in skip_print
        }
        return pandas.DataFrame(data).astype(float).describe().T.drop(columns=["count"])

    def _setup_inputs(self):
        # TODO: Make it so it does not override params
        inputs = {
            **copy.deepcopy(self.env.inputs),
            **copy.deepcopy(self.policy.inputs),
            **copy.deepcopy(self.walkers.inputs),
        }
        for callback in self.callbacks.values():
            inputs.update(dict(callback.inputs))
        self._inputs = inputs

    def _setup_clone_names(self):
        clone_env = [k for k, v in self.env.inputs.items() if v.get("clone")]
        clone_policy = [k for k, v in self.policy.inputs.items() if v.get("clone")]
        clone_walkers = [k for k, v in self.walkers.inputs.items() if v.get("clone")]
        clone_names = clone_env + clone_policy + clone_walkers
        for callback in self.callbacks.values():
            clone_callback = [k for k, v in callback.inputs.items() if v.get("clone")]
            clone_names = clone_names + clone_callback
        self._clone_names = set(list(self.clone_names) + clone_names)

    def _setup_components(self):
        self.env.setup(self)
        self.setup_state(n_walkers=self.n_walkers, param_dict=self.env.param_dict)
        self.policy.setup(self)
        acc_params = {**self.env.param_dict, **self.policy.param_dict, **self.state.param_dict}
        self.setup_state(n_walkers=self.n_walkers, param_dict=acc_params)
        self.walkers.setup(self)
        acc_params = {
            **self.env.param_dict,
            **self.policy.param_dict,
            **self.walkers.param_dict,
            **self.state.param_dict,
        }
        self.setup_state(n_walkers=self.n_walkers, param_dict=acc_params)
        for callback in self.callbacks.values():
            callback.setup(self)
            acc_params.update(callback.param_dict)
            self.setup_state(n_walkers=self.n_walkers, param_dict=acc_params)


class FunctionMapper(Swarm):
    """It is a swarm adapted to minimize mathematical functions."""

    def __init__(
        self,
        minimize: bool = True,
        start_same_pos: bool = False,
        **kwargs,
    ):
        """
        Initialize a :class:`FunctionMapper`.

        Args:
            accumulate_rewards: If `True` the rewards obtained after transitioning
                to a new state will accumulate. If `False` only the last reward
                will be taken into account.
            minimize: If `True` the algorithm will perform a minimization process.
                If `False` it will be a maximization process.
            start_same_pos: If `True` all the walkers will have the same starting position.
            **kwargs: Passed :class:`Swarm` __init__.
        """
        super(FunctionMapper, self).__init__(
            minimize=minimize,
            **kwargs,
        )
        self.start_same_pos = start_same_pos

    @classmethod
    def from_function(
        cls,
        function: Callable,
        bounds: Bounds,
        **kwargs,
    ) -> "FunctionMapper":
        """
        Initialize a :class:`FunctionMapper` using a python callable and a \
        :class:`Bounds` instance.

        Args:
            function: Callable representing an arbitrary function to be optimized.
            bounds: Represents the domain of the function to be optimized.
            **kwargs: Passed to :class:`FunctionMapper` __init__.

        Returns:
            Instance of :class:`FunctionMapper` that optimizes the target function.

        """
        env = Function(function=function, bounds=bounds)
        return FunctionMapper(env=env, **kwargs)

    def __repr__(self):
        return "{}\n{}".format(self.env.__repr__(), super(FunctionMapper, self).__repr__())

    def reset(
        self,
        root_walker: Optional["OneWalker"] = None,
        state: Optional[SwarmState] = None,
    ):
        """
        Reset the :class:`fragile.Walkers`, the :class:`Function` environment, the \
        :class:`Model` and clear the internal data to start a new search process.

        Args:
            root_walker: Walker representing the initial state of the search. \
                         The walkers will be reset to this walker, and it will \
                         be added to the root of the :class:`StateTree` if any.
            state: StateData dictionary that define the initial state of the Swarm.

        """
        super(FunctionMapper, self).reset(root_walker=root_walker, state=state)
        if self.start_same_pos:
            observs = self.get("observs")
            observs[:] = observs[0]
            self.update(observs=observs)
            states = self.get("states", None, raise_error=False)
            if states is not None:
                states[:] = states[0]
                self.update(states=states)
