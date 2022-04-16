from judo.functions.random import random_state
import plangym
import pytest

from fragile.algorithms.fmc import FMCSwarm
from fragile.callbacks.root_walker import BestWalker
from fragile.callbacks.time_steps import GaussianDt
from fragile.callbacks.tree import HistoryTree
from fragile.core.env import PlangymEnv
from fragile.core.policy import Discrete, Gaussian, RandomPlangym
from fragile.core.swarm import Swarm
from fragile.core.walkers import Walkers


def create_pacman():
    plangym_env = plangym.make(name="ALE/MsPacman-v5", obs_type="ram", frameskip=5)
    walkers = Walkers(score_scale=2, accumulate_reward=True)
    env = PlangymEnv(plangym_env)
    callbacks = [BestWalker(), GaussianDt(low=2, high=15, loc=5, scale=5)]
    n_walkers = 40
    swarm = Swarm(
        policy=Discrete(),
        walkers=walkers,
        env=env,
        callbacks=callbacks,
        n_walkers=n_walkers,
        max_epochs=50,
    )
    return swarm


def create_cartpole(max_epochs=150):
    plangym_env = plangym.make(name="CartPole-v0")
    walkers = Walkers(score_scale=2, accumulate_reward=True)
    env = PlangymEnv(plangym_env)
    callbacks = [BestWalker(), HistoryTree(prune=True)]
    n_walkers = 64
    swarm = Swarm(
        policy=RandomPlangym(),
        walkers=walkers,
        env=env,
        callbacks=callbacks,
        n_walkers=n_walkers,
        max_epochs=max_epochs,
    )
    return swarm


def create_dm_cartpole():
    plangym_env = plangym.make(name="cartpole-balance", frameskip=2)
    walkers = Walkers(score_scale=2, accumulate_reward=True)
    env = PlangymEnv(plangym_env)
    policy = Gaussian(scale=2, second_order=True)
    callbacks = [BestWalker(), GaussianDt(low=2, high=15, loc=5, scale=5)]
    n_walkers = 32
    swarm = Swarm(
        policy=policy,
        walkers=walkers,
        env=env,
        callbacks=callbacks,
        n_walkers=n_walkers,
        max_epochs=50,
    )
    return swarm


def cartpole_fmc():
    swarm = create_cartpole(max_epochs=10)
    fmc = FMCSwarm(
        swarm=swarm,
        max_epochs=50,
        callbacks=[BestWalker(always_update=True), HistoryTree(names=["states", "actions"])],
    )
    return fmc


swarms = [
    (create_pacman(), 1800),
    (create_cartpole(), 148),
    (create_dm_cartpole(), 50),
    (cartpole_fmc(), 48),
]


@pytest.mark.parametrize("swarm_and_target", swarms)
def test_score_gets_higher(swarm_and_target):
    swarm, target_score = swarm_and_target
    random_state.seed(160290)
    swarm.run()
    root_score = float(swarm.root.score)
    beats_target = root_score <= target_score if swarm.minimize else root_score >= target_score
    assert beats_target, (root_score, target_score)
