from judo.functions.random import random_state
import pytest

from fragile.algorithms.fmc import FMCSwarm
from fragile.callbacks.root_walker import BestWalker
from fragile.callbacks.tree import HistoryTree
from fragile.core.policy import Gaussian
from fragile.core.swarm import FunctionMapper
from fragile.core.walkers import Walkers
from fragile.optimize.benchmarks import Rastrigin
from fragile.optimize.env import MinimizerWrapper


def create_rastrigin(max_epochs=500):
    walkers = Walkers(score_scale=2, accumulate_reward=False)
    env = Rastrigin(2)
    policy = Gaussian(scale=0.25, second_order=False)
    callbacks = [BestWalker()]
    n_walkers = 32
    swarm = FunctionMapper(
        policy=policy,
        walkers=walkers,
        env=env,
        callbacks=callbacks,
        n_walkers=n_walkers,
        max_epochs=max_epochs,
    )
    return swarm


def create_minimzer_rastrigin(max_epochs=500):
    walkers = Walkers(score_scale=2, accumulate_reward=False)
    env = MinimizerWrapper(Rastrigin(2))
    policy = Gaussian(scale=0.25, second_order=False)
    callbacks = [BestWalker()]
    n_walkers = 32
    swarm = FunctionMapper(
        policy=policy,
        walkers=walkers,
        env=env,
        callbacks=callbacks,
        n_walkers=n_walkers,
        max_epochs=max_epochs,
    )
    return swarm


def rastrigin_fmc():
    swarm = create_rastrigin(max_epochs=10)
    fmc = FMCSwarm(
        minimize=True,
        swarm=swarm,
        max_epochs=150,
        callbacks=[BestWalker(always_update=True), HistoryTree(names=["states", "actions"])],
    )
    return fmc


swarms = [(create_rastrigin(), 1), (rastrigin_fmc(), 18), (create_minimzer_rastrigin(), 1)]


@pytest.mark.parametrize("swarm_and_target", swarms)
def test_score_gets_higher(swarm_and_target):
    random_state.seed(160290)
    swarm, target_score = swarm_and_target
    swarm.run()
    root_score = float(swarm.root.score)
    beats_target = root_score <= target_score if swarm.minimize else root_score >= target_score
    assert beats_target, (root_score, target_score)
