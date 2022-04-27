from typing import Optional, Tuple, Union

import judo
from judo import dtype, random_state, tensor
import numpy
import numpy as np

from fragile.core.api_classes import WalkersAPI, WalkersMetric
from fragile.core.fractalai import fai_iteration, relativize
from fragile.core.typing import InputDict, StateData, StateDict, Tensor


class SimpleWalkers(WalkersAPI):
    default_inputs = {
        "observs": {},
        "oobs": {"optional": True},
        "rewards": {},
        "scores": {"clone": True},
    }
    default_param_dict = {"scores": {"dtype": dtype.float32}}
    default_outputs = tuple(["scores"])

    def __init__(
        self,
        accumulate_reward: bool = True,
        score_scale: float = 1.0,
        diversity_scale: float = 1.0,
        minimize: bool = False,
        **kwargs,
    ):
        self.score_scale = score_scale
        self.diversity_scale = diversity_scale
        self.accumulate_reward = accumulate_reward
        self.minimize = minimize
        super(SimpleWalkers, self).__init__(**kwargs)

    def run_epoch(
        self,
        observs,
        rewards,
        scores,
        oobs=None,
        inplace: bool = True,
        **kwargs,
    ) -> StateData:
        scores = rewards + scores if self.accumulate_reward else rewards
        sign_scores = -1.0 * scores if self.minimize else scores
        compas_ix, will_clone = fai_iteration(
            observs=observs,
            rewards=sign_scores,
            oobs=oobs,
            dist_coef=self.diversity_scale,
            reward_coef=self.score_scale,
        )
        if inplace:
            self.clone_walkers(compas_clone=compas_ix, will_clone=will_clone)
        return {"scores": scores}


def l2_norm(x: Tensor, y: Tensor) -> Tensor:
    return judo.sqrt(judo.sum((x - y) ** 2, 1))


class ScoreMetric(WalkersMetric):
    default_param_dict = {"scores": {"dtype": dtype.float32}}
    default_outputs = tuple(["scores"])


class RewardScore(ScoreMetric):
    default_inputs = {"rewards": {}}

    def __init__(self, accumulate_reward: bool = True, keep_max_reward: bool = False, **kwargs):
        self.accumulate_reward = accumulate_reward
        self.keep_max_reward = keep_max_reward
        super(RewardScore, self).__init__(**kwargs)

    @property
    def inputs(self) -> InputDict:
        inputs = super(RewardScore, self).inputs
        if self.accumulate_reward:
            inputs["scores"] = {"clone": True}
        return inputs

    def calculate(self, rewards, scores=None, **kwargs):
        values = rewards + scores if self.accumulate_reward else rewards
        if self.keep_max_reward and scores is not None:
            values = np.maximum(values, scores)
        return {"scores": values}

    def reset(
        self,
        inplace: bool = True,
        root_walker: Optional[StateData] = None,
        states: Optional[StateData] = None,
        **kwargs,
    ):
        if root_walker is None and not self.accumulate_reward:
            self.update(scores=self.get("rewards"))


class SonicScore(ScoreMetric):
    accumulate_reward = False
    default_inputs = {"rewards": {}, "infos": {}}

    @staticmethod
    def score_from_info(info):
        try:
            score = info["score"] + 10000 * info["x"] / info["screen_x_end"] + 100 * info["rings"]
        except Exception:
            return info["score"] + 10 * info["x"] + 100 * info["rings"]

        return score

    def calculate(self, rewards, scores=None, **kwargs):
        scores = tensor([self.score_from_info(info) for info in self.get("infos")])
        return {"scores": scores}


class DiversityMetric(WalkersMetric):
    default_param_dict = {"diversities": {"dtype": dtype.float32}}
    default_outputs = tuple(["diversities"])


class RandomDistance(DiversityMetric):
    default_inputs = {"observs": {}, "oobs": {}}

    def calculate(self, observs, oobs, **kwargs):
        n_walkers = self.swarm.n_walkers
        compas = self.swarm.walkers.get_in_bounds_compas(oobs=oobs)
        obs = judo.astype(observs.reshape(n_walkers, -1), dtype.float32)
        if hasattr(self.swarm.env, "bounds"):
            deltas = self.swarm.env.bounds.pbc_distance(obs, obs[compas])
            return {"diversities": numpy.linalg.norm(deltas, axis=1).flatten()}
        return {"diversities": l2_norm(obs, obs[compas]).flatten()}


class Walkers(WalkersAPI):
    default_param_dict = {
        "compas_clone": {"dtype": dtype.int64},
        "virtual_rewards": {"dtype": dtype.float32},
        "clone_probs": {"dtype": dtype.float32},
        "will_clone": {"dtype": dtype.bool},
    }
    default_outputs = (
        "compas_clone",
        "virtual_rewards",
        "clone_probs",
        "will_clone",
    )

    default_inputs = {"oobs": {}, "terminals": {"optional": True}}

    def __init__(
        self,
        score: ScoreMetric = None,
        diversity: DiversityMetric = None,
        minimize: bool = False,
        score_scale: float = 1.0,
        diversity_scale: float = 1.0,
        track_data=None,
        accumulate_reward: bool = True,
        keep_max_reward: bool = False,
        clone_period: int = 1,
        **kwargs,
    ):

        self.minimize = minimize
        self.score_scale = score_scale
        self.diversity_scale = diversity_scale
        self.clone_period = clone_period
        self.score = (
            score
            if score is not None
            else RewardScore(accumulate_reward=accumulate_reward, keep_max_reward=keep_max_reward)
        )
        self.accumulate_reward = self.score.accumulate_reward
        self.diversity = diversity if diversity is not None else RandomDistance()
        self.track_data = set(track_data) if track_data is not None else set()
        super(WalkersAPI, self).__init__(**kwargs)

    @property
    def param_dict(self) -> StateDict:
        return {
            **super(WalkersAPI, self).param_dict,
            **self.diversity.param_dict,
            **self.score.param_dict,
        }

    @property
    def inputs(self) -> InputDict:
        return {**super(WalkersAPI, self).inputs, **self.diversity.inputs, **self.score.inputs}

    @property
    def outputs(self) -> Tuple[str, ...]:
        return super(WalkersAPI, self).outputs + self.score.outputs + self.diversity.outputs

    def setup(self, swarm):
        super(Walkers, self).setup(swarm)
        self.diversity.setup(swarm)
        self.score.setup(swarm)
        self.minimize = swarm.minimize

    def balance(self, inplace: bool = True, **kwargs) -> Union[None, StateData]:
        if self.swarm.epoch % self.clone_period == 0 or self.swarm.epoch == 0:
            return super(Walkers, self).balance(inplace=inplace, **kwargs)

    def run_epoch(self, inplace: bool = True, oobs=None, **kwargs):
        scores = self.score(**kwargs)
        diversities = self.diversity(oobs=oobs, **kwargs)
        virtual_rewards = self.calculate_virtual_reward(**{**scores, **diversities})
        clone_data = self.calculate_clones(oobs=oobs, **virtual_rewards)
        if inplace:
            self.clone_walkers(**clone_data)
        return {**scores, **diversities, **virtual_rewards, **clone_data}

    def calculate_virtual_reward(self, scores, diversities, **kwargs):
        """Apply the virtual reward formula to account for all the different goal scores."""
        scores = -1.0 * scores if self.minimize else scores
        norm_scores = relativize(scores)
        norm_diver = relativize(diversities)
        virtual_rewards = norm_scores**self.score_scale * norm_diver**self.diversity_scale
        return {"virtual_rewards": virtual_rewards}

    def calculate_clones(self, virtual_rewards, oobs=None):
        """Calculate the walkers that will clone and their target companions."""
        n_walkers = len(virtual_rewards)
        all_virtual_rewards_are_equal = (virtual_rewards == virtual_rewards[0]).all()
        if all_virtual_rewards_are_equal:
            clone_probs = judo.zeros(n_walkers, dtype=dtype.float)
            compas_clone = judo.arange(n_walkers)
        else:
            compas_clone = self.get_in_bounds_compas(oobs)
            # This value can be negative!!
            clone_probs = (virtual_rewards[compas_clone] - virtual_rewards) / virtual_rewards
        will_clone = clone_probs > random_state.random_sample(n_walkers)
        if oobs is not None:
            will_clone[oobs] = True  # Out of bounds walkers always clone
        return dict(
            clone_probs=clone_probs,
            will_clone=will_clone,
            compas_clone=compas_clone,
        )

    def reset(self, inplace: bool = True, **kwargs):
        super(Walkers, self).reset(inplace=inplace, **kwargs)
        self.score.reset(**kwargs)
        self.diversity.reset(**kwargs)


class NoBalance(Walkers):
    @property
    def param_dict(self) -> StateDict:
        return self.score.param_dict

    @property
    def inputs(self) -> InputDict:
        return self.score.inputs

    @property
    def outputs(self) -> Tuple[str, ...]:
        return self.score.outputs

    def run_epoch(self, inplace: bool = True, oobs=None, **kwargs):
        scores = self.score(**kwargs)
        return scores
