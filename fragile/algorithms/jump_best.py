from typing import Iterable, Optional, Tuple, Union

import networkx as nx

from fragile.callbacks.data_tracking import StoreInitAction
from fragile.callbacks.tree import HistoryTree
from fragile.core.api_classes import Callback, EnvironmentAPI, PolicyAPI, WalkersAPI
from fragile.core.swarm import Swarm
from fragile.core.typing import InputDict, StateData, StateDict, Tensor


class DummyFMCPolicy(PolicyAPI):
    def __init__(self, inner_swarm, **kwargs):
        self.inner_swarm = inner_swarm
        super(DummyFMCPolicy, self).__init__(**kwargs)

    def select_actions(self, **kwargs) -> Union[Tensor, StateData]:
        return {}

    def act(self, inplace: bool = True, **kwargs) -> Union[None, StateData]:
        if not inplace:
            return {}


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
        env_data = self.swarm.inputs_from_swarm("env")
        env_data = self.inner_swarm.env.make_transitions(inplace=False, **env_data)
        env_data["scores"] = env_data["rewards"]
        if self.inner_swarm.walkers.accumulate_reward:
            env_data["scores"] += self.get("scores")
        if inplace:
            self.swarm.state.update(**env_data)
        else:
            return env_data

    def step(self, **kwargs) -> StateData:
        pass


class JumpToBestTree(HistoryTree):
    def __len__(self):
        return len(self.graph.nodes)

    def after_evolve(self):
        pass

    def after_env(self):
        pass

    def update_tree(self):
        pass


class JumpToBestEnv(EnvSwarm):
    def make_transitions(self, inplace: bool = True, **kwargs) -> Union[None, StateData]:
        return
        self.swarm.state.import_walker(self.inner_swarm.root.data)
        if hasattr(self.swarm, "root"):
            self.swarm.root.update_root()

        if not inplace:
            return {}

    def reset(
        self,
        inplace: bool = True,
        root_walker: Optional[StateData] = None,
        states: Optional[StateData] = None,
        **kwargs,
    ) -> Union[None, StateData]:
        return super(JumpToBestEnv, self).make_transitions()


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


class JumpToBest(WalkersSwarm):
    def run_epoch(self, inplace: bool = True, **kwargs) -> StateData:

        super(JumpToBest, self).run_epoch(inplace=inplace, **kwargs)
        print(
            self.swarm.root.id_walker,
            self.inner_swarm.root.id_walker,
            self.inner_swarm.tree.data_tree.root_id,
        )
        root_graph = self.inner_swarm.tree.get_root_graph()
        self.swarm.tree.data_tree.compose(root_graph)
        self.swarm.state.update(**self.inner_swarm.root.data)
        self.swarm.root.update_root()
        print(
            "self tree",
            nx.is_tree(self.tree.graph),
            "inner",
            nx.is_tree(self.inner_swarm.tree.graph),
            "root",
            nx.is_tree(root_graph),
        )
        return {}

    def reset(self, inplace: bool = True, **kwargs):
        super(JumpToBest, self).reset(inplace=inplace, **kwargs)
        root_graph = self.inner_swarm.tree.get_root_graph()
        print(
            "self tree",
            nx.is_tree(self.tree.graph),
            "inner",
            nx.is_tree(self.inner_swarm.tree.graph),
            "root",
            nx.is_tree(root_graph),
        )
        self.swarm.tree.data_tree._data = root_graph
        walker = self.inner_swarm.state.export_walker(0)
        self.swarm.state.update(**walker)
        self.swarm.root.reset()
        self.swarm.root.update_root()


class FMCSwarm(Swarm):
    walkers_last = False

    def __init__(
        self,
        swarm: Swarm,
        policy: PolicyAPI = None,
        env: EnvironmentAPI = None,
        callbacks: Optional[Iterable[Callback]] = None,
        minimize: bool = False,
        max_epochs: int = 1e100,
    ):
        swarm.register_callback(StoreInitAction())
        # if hasattr(swarm.policy, "bounds"):
        #    swarm.register_callback(BoundaryInitAction())
        self._swarm = swarm
        policy = DummyFMCPolicy(swarm) if policy is None else policy
        env = EnvSwarm(swarm) if env is None else env
        super(FMCSwarm, self).__init__(
            n_walkers=swarm.n_walkers,
            policy=policy,
            env=env,
            callbacks=callbacks,
            minimize=minimize,
            max_epochs=int(max_epochs),
            walkers=JumpToBest(swarm),
        )

    @property
    def swarm(self) -> Swarm:
        return self._swarm

    @property
    def n_walkers(self) -> int:
        return 1

    def inputs_from_swarm(self, name):
        def get_names():
            names = []
            inputs = getattr(self.swarm, name).inputs
            for k, v in inputs.items():
                is_optional = v.get("optional", False)
                if not is_optional or (is_optional and k in self.swarm.param_dict.keys()):
                    names.append(k)
            return names

        return {name: self.state[name] for name in get_names()}
