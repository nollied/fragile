from typing import Iterable, Optional, Tuple, Union

import einops
import judo
from judo.typing import Tensor
import numpy
import numpy as np

from fragile.new_core.api_classes import Callback, EnvironmentAPI, PolicyAPI, WalkersAPI
from fragile.new_core.policy import ContinuousPolicy
from fragile.new_core.states import SwarmState
from fragile.new_core.swarm import Swarm
from fragile.new_core.typing import InputDict, StateData, StateDict


class StoreInitAction(Callback):
    name = "store_init_action"
    default_inputs = {"init_actions": {"clone": True}}
    default_outputs = ("init_actions",)

    @property
    def param_dict(self) -> StateDict:
        return {"init_actions": dict(self.swarm.param_dict["actions"])}

    def before_env(self):
        if self.swarm.epoch == 0:
            self.update(init_actions=self.get("actions").copy())


class BoundaryInitAction(Callback):
    name = "boundary_action"
    default_outputs = ("actions",)

    def __init__(self, only_first_epoch: bool = True, **kwargs):
        self.only_first_epoch = only_first_epoch
        super(BoundaryInitAction, self).__init__(**kwargs)

    def after_policy(self):
        assert hasattr(self.swarm.policy, "bounds")
        if self.only_first_epoch and self.swarm.epoch > 0:
            return
        low_val = einops.repeat(self.swarm.policy.bounds.low, "n -> b n", b=self.swarm.n_walkers)
        high_val = einops.repeat(self.swarm.policy.bounds.high, "n -> b n", b=self.swarm.n_walkers)
        condition = judo.random_state.random(low_val.shape) < 0.5
        actions = numpy.where(condition, low_val, high_val)
        self.update(actions=actions)


class FMCPolicy(PolicyAPI):

    default_inputs = {"init_actions": {}, "oobs": {}}

    def __init__(self, inner_swarm, **kwargs):
        self.inner_swarm = inner_swarm
        super(FMCPolicy, self).__init__(**kwargs)

    @property
    def param_dict(self) -> StateDict:
        pdict = self.inner_swarm.env.param_dict
        return {**{"init_actions": dict(pdict["actions"])}, **pdict}

    def select_actions(self, **kwargs) -> Union[Tensor, StateData]:
        if hasattr(self.inner_swarm.env.action_space, "n"):
            return self._choose_majority()
        init_actions = self.inner_swarm.get("init_actions")
        return init_actions.mean(0)[np.newaxis, :]

    def _choose_majority(self):
        init_actions = judo.to_numpy(self.inner_swarm.get("init_actions"))
        y = numpy.bincount(init_actions, minlength=self.inner_swarm.env.action_space.n)
        most_used_action = judo.tensor([y.argmax()])
        return most_used_action


class EnvSwarm(EnvironmentAPI):
    def __init__(self, swarm):
        self.inner_swarm = swarm
        super(EnvSwarm, self).__init__(
            swarm=None,
            action_shape=swarm.env.action_shape,
            action_dtype=swarm.env.action_dtype,
            observs_shape=swarm.env.observs_shape,
            observs_dtype=swarm.env.observs_dtype,
        )

    def __getattr__(self, item):
        return getattr(self.inner_swarm.env, item)

    @property
    def inputs(self) -> InputDict:
        return self.inner_swarm.env.inputs

    @property
    def outputs(self) -> Tuple[str, ...]:
        return self.inner_swarm.env.outputs + ("scores",)

    @property
    def param_dict(self) -> StateDict:
        pdict = self.inner_swarm.env.param_dict
        return {**{"scores": pdict["rewards"]}, **pdict}

    def make_transitions(self, inplace: bool = True, **kwargs) -> Union[None, StateData]:
        def get_env_names():
            names = []
            for k, v in self.inner_swarm.env.inputs.items():
                is_optional = v.get("optional", False)
                if not is_optional or (is_optional and k in self.inner_swarm.param_dict.keys()):
                    names.append(k)
            return names

        env_data = {name: self.swarm.state[name] for name in get_env_names()}
        env_data = self.inner_swarm.env.make_transitions(inplace=False, **env_data)
        env_data["scores"] = env_data["rewards"]
        if self.inner_swarm.walkers.accumulate_reward:
            env_data["scores"] += self.get("scores")
        if inplace:
            self.swarm.state.update(**env_data)
        else:
            return env_data


class WalkersSwarm(WalkersAPI):
    def __init__(self, inner_swarm, **kwargs):
        self.inner_swarm = inner_swarm
        self.accumulate_reward = self.inner_swarm.walkers.accumulate_reward
        super(WalkersSwarm, self).__init__(**kwargs)

    @property
    def param_dict(self) -> StateDict:
        return self.inner_swarm.param_dict

    def __getattr__(self, item):
        return getattr(self.inner_swarm, item)

    def run_epoch(self, inplace: bool = True, **kwargs) -> StateData:
        self.inner_swarm.run(root_walker=dict(self.swarm.state))
        return {}

    def reset(self, inplace: bool = True, **kwargs):
        super(WalkersSwarm, self).reset(inplace=inplace, **kwargs)
        self.inner_swarm.reset()
        walker = self.inner_swarm.state.export_walker(0)
        self.swarm.state.update(**walker)


class FMCSwarm(Swarm):
    walkers_last = False

    def __init__(
        self,
        swarm: Swarm,
        policy: PolicyAPI = None,
        env: EnvironmentAPI = None,
        callbacks: Optional[Iterable[Callback]] = None,
        minimize: bool = False,
        max_epochs: int = int(1e100),
    ):
        swarm.register_callback(StoreInitAction())
        # if hasattr(swarm.policy, "bounds"):
        #    swarm.register_callback(BoundaryInitAction())
        self._swarm = swarm
        env = EnvSwarm(swarm) if env is None else env
        super(FMCSwarm, self).__init__(
            n_walkers=swarm.n_walkers,
            policy=FMCPolicy(swarm),
            env=env,
            callbacks=callbacks,
            minimize=minimize,
            max_epochs=max_epochs,
            walkers=WalkersSwarm(swarm),
        )

    @property
    def swarm(self) -> Swarm:
        return self._swarm

    @property
    def n_walkers(self) -> int:
        return 1

    """def before_policy(self):
        if self.epoch > 0:
            super(FMCSwarm, self).before_policy()

    def after_policy(self):
        if self.epoch > 0:
            super(FMCSwarm, self).after_policy()

    def before_env(self):
        if self.epoch > 0:
            super(FMCSwarm, self).before_env()

    def after_env(self):
        if self.epoch > 0:
            super(FMCSwarm, self).after_env()

    def before_walkers(self):
        if self.epoch > 0:
            super(FMCSwarm, self).before_walkers()

    def after_walkers(self):
        if self.epoch > 0:
            super(FMCSwarm, self).after_walkers()"""
