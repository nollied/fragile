import warnings

from hypothesis import given
from hypothesis.errors import HypothesisDeprecationWarning
from hypothesis.extra.numpy import arrays
import numpy
import pytest

from fragile.core.functions import relativize
from fragile.core.states import StatesEnv, StatesModel, StatesWalkers
from fragile.core.utils import float_type, NUMPY_IGNORE_WARNINGS_PARAMS
from fragile.core.walkers import Walkers


warnings.filterwarnings("ignore", category=HypothesisDeprecationWarning)


@pytest.fixture()
def states_walkers():
    return StatesWalkers(10)


N_WALKERS = 13


def get_walkers_discrete_gym():
    env_params = {
        "states": {"size": (128,), "dtype": numpy.int64},
        "observs": {"size": (64, 64, 3), "dtype": numpy.float32},
        "rewards": {"dtype": numpy.float32},
        "oobs": {"dtype": numpy.bool_},
    }
    model_params = {
        "actions": {"size": (10,), "dtype": numpy.int64},
        "dt": {"size": None, "dtype": numpy.float32},
    }
    return Walkers(
        n_walkers=N_WALKERS, env_state_params=env_params, model_state_params=model_params
    )


def get_function_walkers():
    env_params = {
        "states": {"size": (3,), "dtype": numpy.int64},
        "observs": {"size": (3,), "dtype": numpy.float32},
        "rewards": {"dtype": numpy.float32},
        "oobs": {"dtype": numpy.bool_},
    }
    model_params = {
        "actions": {"size": (3,), "dtype": numpy.int64},
        "dt": {"size": None, "dtype": numpy.float32},
    }
    return Walkers(
        n_walkers=N_WALKERS,
        env_state_params=env_params,
        model_state_params=model_params,
        minimize=True,
    )


walkers_config = {"discrete-gym": get_walkers_discrete_gym, "function": get_function_walkers}
walkers_fixture_params = ["discrete-gym", "function"]


class TestStatesWalkers:
    def test_reset(self, states_walkers):
        for name in states_walkers.keys():
            assert states_walkers[name] is not None, name
            assert len(states_walkers[name]) == states_walkers.n, name

        states_walkers.reset()
        for name in states_walkers.keys():
            assert states_walkers[name] is not None, name
            assert len(states_walkers[name]) == states_walkers.n, name

    def test_update(self, states_walkers):
        states_walkers = StatesWalkers(10)
        states_walkers.reset()
        test_vals = numpy.arange(states_walkers.n)
        states_walkers.update(virtual_rewards=test_vals, distances=test_vals)
        assert (states_walkers.virtual_rewards == test_vals).all()
        assert (states_walkers.distances == test_vals).all()


@pytest.fixture(params=walkers_fixture_params, scope="class")
def walkers(request):
    return walkers_config.get(request.param, get_walkers_discrete_gym)()


class TestWalkers:
    def test_repr_not_crashes(self, walkers):
        assert isinstance(walkers.__repr__(), str)

    def test_getattr(self, walkers):
        assert isinstance(walkers.states.will_clone, numpy.ndarray)
        assert isinstance(walkers.env_states.observs, numpy.ndarray)
        assert isinstance(walkers.env_states, StatesEnv)
        assert isinstance(walkers.model_states, StatesModel)
        with pytest.raises(AttributeError):
            assert isinstance(walkers.moco, numpy.ndarray)

    def test_get_best_index(self, walkers):
        # Rewards = [1,1,...] InBounds = [0,0,...]
        walkers.states.update(cum_rewards=numpy.ones(walkers.n), in_bounds=numpy.zeros(walkers.n))
        best_idx = walkers.get_best_index()
        # If there are no in_bound rewards, the last walker is returned
        assert best_idx == walkers.n - 1

        # Some OOB rewards
        #
        # Rewards = [0,1,0,...] InBounds = [0,1,...]
        oobs_best_idx = 1
        oobs_rewards = numpy.zeros(walkers.n)
        oobs_rewards[oobs_best_idx] = 1
        some_oobs = numpy.zeros(walkers.n)
        some_oobs[oobs_best_idx] = 1
        walkers.states.update(cum_rewards=oobs_rewards, in_bounds=some_oobs)
        best_idx = walkers.get_best_index()
        assert best_idx == oobs_best_idx

        # If the walkers are minimizing, set all but one reward to 1.0
        # If the walkers are maximizing, set all but one reward to 0.0
        positive_val = 0.0 if walkers.minimize else 1.0
        negative_val = 1.0 if walkers.minimize else 0.0
        # Rewards = [-,+,-,-,-,...] InBounds = [1,...]
        mixed_rewards = numpy.full(walkers.n, fill_value=negative_val)
        mixed_best = 1  # could be any index
        mixed_rewards[mixed_best] = positive_val
        walkers.states.update(cum_rewards=mixed_rewards, in_bounds=numpy.ones(walkers.n))
        best_idx = walkers.get_best_index()
        assert best_idx == mixed_best

    def test_calculate_end_condition(self, walkers):
        walkers.reset()
        walkers.env_states.update(oobs=numpy.ones(walkers.n))
        assert walkers.calculate_end_condition()
        walkers.env_states.update(oobs=numpy.zeros(walkers.n))
        assert not walkers.calculate_end_condition()
        walkers.max_epochs = 10
        walkers._epoch = 8
        assert not walkers.calculate_end_condition()
        walkers._epoch = 11
        assert walkers.calculate_end_condition()

    def test_alive_compas(self, walkers):
        end_cond = numpy.zeros_like(walkers.env_states.oobs).astype(bool).copy()
        end_cond[3] = True
        walkers.states.in_bounds = end_cond
        compas = walkers.get_in_bounds_compas()
        assert numpy.all(compas == 3), "Type of end_cond: {} end_cond: {}: alive ix: {}".format(
            type(end_cond), end_cond, walkers.states.in_bounds
        )
        assert len(compas.shape) == 1

    def test_update_clone_probs(self, walkers):
        walkers.reset()
        walkers.states.update(virtual_rewards=relativize(numpy.arange(walkers.n)))
        walkers.update_clone_probs()
        assert 0 < numpy.sum(walkers.states.clone_probs == walkers.states.clone_probs[0]), (
            walkers.states.virtual_rewards,
            walkers.states.clone_probs,
        )
        walkers.reset()
        walkers.update_clone_probs()
        assert numpy.sum(walkers.states.clone_probs == walkers.states.clone_probs[0]) == walkers.n
        assert walkers.states.clone_probs.shape[0] == walkers.n
        assert len(walkers.states.clone_probs.shape) == 1

    def test_balance_not_crashes(self, walkers):
        with numpy.errstate(**NUMPY_IGNORE_WARNINGS_PARAMS):
            walkers.reset()
            walkers.balance()
            assert walkers.env_states.oobs.sum() == 0

    def test_accumulate_rewards(self, walkers):
        walkers.reset()
        walkers._accumulate_rewards = True
        walkers.states.update(cum_rewards={None, 3})  # Override array of Floats and set to None
        walkers.states.update(cum_rewards=None)
        rewards = numpy.arange(len(walkers))
        walkers._accumulate_and_update_rewards(rewards)
        assert (walkers.states.cum_rewards == rewards).all()
        walkers._accumulate_rewards = False
        walkers.states.update(cum_rewards=numpy.zeros(len(walkers)))
        rewards = numpy.arange(len(walkers))
        walkers._accumulate_and_update_rewards(rewards)
        assert (walkers.states.cum_rewards == rewards).all()
        walkers._accumulate_rewards = True
        walkers.states.update(cum_rewards=numpy.ones(len(walkers)))
        rewards = numpy.arange(len(walkers))
        walkers._accumulate_and_update_rewards(rewards)
        assert (walkers.states.cum_rewards == rewards + 1).all()

    @given(observs=arrays(numpy.float32, shape=(N_WALKERS, 64, 64, 3)))
    def test_distances_not_crashes(self, walkers, observs):
        with numpy.errstate(**NUMPY_IGNORE_WARNINGS_PARAMS):
            walkers.env_states.update(observs=observs)
            walkers.calculate_distances()
            assert isinstance(walkers.states.distances[0], float_type)
            assert len(walkers.states.distances.shape) == 1
            assert walkers.states.distances.shape[0] == walkers.n
