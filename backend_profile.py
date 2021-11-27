import sys

from fragile.backend import Backend
from fragile.backend.slogging import setup as setup_logging
from plangym import AtariEnvironment
from fragile.core import DiscreteEnv, NormalContinuous, HistoryTree, GaussianDt, Swarm
from fragile.optimize.benchmarks import EggHolder
from fragile.distributed.env import ParallelEnv

# Backend.set_backend("torch")


def main():

    setup_logging(level="INFO", structured=False)

    def atari_environment():
        game_name = "MsPacman-v0"
        plangym_env = AtariEnvironment(
            name=game_name,
            clone_seeds=True,
            autoreset=True,
        )
        return DiscreteEnv(env=plangym_env)

    # env_callable = lambda: ParallelEnv(atari_environment, n_workers=8)
    model_callable = lambda env: NormalContinuous(env=env)
    # tree_callable = lambda: HistoryTree(names=["states", "actions", "dt"], prune=True)

    n_walkers = 10000  # A bigger number will increase the quality of the trajectories sampled.
    max_epochs = 1000  # Increase to sample longer games.
    reward_scale = 2  # Rewards are more important than diversity.
    distance_scale = 1
    minimize = True  # We want to get the maximum score possible.

    swarm = Swarm(
        model=model_callable,
        env=EggHolder,
        # tree=tree_callable,
        n_walkers=n_walkers,
        max_epochs=max_epochs,
        reward_scale=reward_scale,
        distance_scale=distance_scale,
        minimize=minimize,
    )
    swarm.run()


if __name__ == "__main__":
    sys.exit(main())
