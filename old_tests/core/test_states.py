import judo
from judo import Backend, dtype, tensor
import pytest

from fragile.core.states import States, StatesEnv, StatesModel, StatesWalkers
from tests.core.test_swarm import create_atari_swarm


state_classes = [States, StatesEnv, StatesModel, StatesWalkers]


@pytest.fixture(scope="class", params=state_classes)
def states_class(request):
    return request.param


class TestStates:
    def test_init_dict(self, states_class):
        state_dict = {"name_1": {"size": tuple([1]), "dtype": dtype.float32}}
        new_states = states_class(state_dict=state_dict, batch_size=2)
        assert new_states.n == 2

    def test_init_kwargs(self, states_class):
        name = "miau"
        new_states = states_class(batch_size=2, miau=name)
        assert new_states._batch_size == 2
        assert name in new_states.keys()
        assert getattr(new_states, name) == name, type(new_states)

    def test_getitem(self, states_class):
        name = "miau"
        new_states = states_class(batch_size=2, miau=name)
        assert new_states[name] == name, type(new_states)

    def test_setitem(self, states_class):
        name_1 = "miau"
        val_1 = name_1
        name_2 = "elephant"
        val_2 = judo.arange(10)
        new_states = states_class(batch_size=2)
        new_states[name_1] = val_1
        new_states[name_2] = val_2
        assert new_states[name_1] == val_1, type(new_states)
        assert (new_states[name_2] == val_2).all(), type(new_states)

    def test_repr(self, states_class):
        name = "miau"
        new_states = states_class(batch_size=2, miau=name)
        assert isinstance(new_states.__repr__(), str)

    def test_n(self, states_class):
        new_states = states_class(batch_size=2)
        assert new_states.n == new_states._batch_size == 2

    def test_get(self, states_class):
        new_states = states_class(batch_size=2, test="test")
        assert new_states.get("test") == "test"
        assert new_states.get("AKSJDFKG") is None
        assert new_states.get("ASSKADFKA", 5) == 5

    def test_split_states(self, states_class):
        batch_size = 20
        new_states = states_class(batch_size=batch_size, test="test")
        for s in new_states.split_states(batch_size):
            assert len(s) == 1
            assert s.test == "test"
        data = judo.repeat(judo.arange(5).reshape(1, -1), batch_size, 0)
        new_states = states_class(batch_size=batch_size, test="test", data=data)
        for s in new_states.split_states(batch_size):
            assert len(s) == 1
            assert s.test == "test"
            assert bool((s.data == judo.arange(5)).all()), s.data
        chunk_len = 4
        test_data = judo.repeat(judo.arange(5).reshape(1, -1), chunk_len, 0)
        for s in new_states.split_states(5):
            assert len(s) == chunk_len
            assert s.test == "test"
            assert (s.data == test_data).all(), (s.data.shape, test_data.shape)

        batch_size = 21
        data = judo.repeat(judo.arange(5).reshape(1, -1), batch_size, 0)
        new_states = states_class(batch_size=batch_size, test="test", data=data)
        chunk_len = 5
        test_data = judo.repeat(judo.arange(5).reshape(1, -1), chunk_len, 0)
        split_states = list(new_states.split_states(5))
        for s in split_states[:-1]:
            assert len(s) == chunk_len
            assert s.test == "test"
            assert (s.data == test_data).all(), (s.data.shape, test_data.shape)

        assert len(split_states[-1]) == 1
        assert split_states[-1].test == "test"
        assert (split_states[-1].data == judo.arange(5)).all(), (s.data.shape, test_data.shape)

    def test_get_params_dir(self, states_class):
        state_dict = {"name_1": {"size": tuple([1]), "dtype": dtype.float32}}
        new_states = states_class(state_dict=state_dict, batch_size=2)
        params_dict = new_states.get_params_dict()
        assert isinstance(params_dict, dict)
        for k, v in params_dict.items():
            assert isinstance(k, str)
            assert isinstance(v, dict)
            for ki, _ in v.items():
                assert isinstance(ki, str)

    def test_merge_states(self, states_class):
        batch_size = 21
        data = judo.repeat(judo.arange(5).reshape(1, -1), batch_size, 0)
        new_states = states_class(batch_size=batch_size, test="test", data=data)
        split_states = tuple(new_states.split_states(batch_size))
        merged = new_states.merge_states(split_states)
        assert len(merged) == batch_size
        assert merged.test == "test"
        assert (merged.data == data).all()

        split_states = tuple(new_states.split_states(5))
        merged = new_states.merge_states(split_states)
        assert len(merged) == batch_size
        assert merged.test == "test"
        assert (merged.data == data).all()


class TestFragileStates:
    def test_clone(self, states_class):
        batch_size = 10
        states = states_class(batch_size=batch_size)
        states.miau = judo.arange(states.n)
        states.miau_2 = judo.arange(states.n)

        will_clone = judo.zeros(states.n, dtype=judo.bool)
        will_clone[3:6] = True
        compas_ix = tensor(list(range(states.n))[::-1])

        states.clone(will_clone=will_clone, compas_ix=compas_ix)
        target_1 = judo.arange(10)

        assert bool(judo.all(target_1 == states.miau)), (target_1 - states.miau, states_class)

    def test_merge_states(self, states_class):
        batch_size = 21
        data = judo.repeat(judo.arange(5).reshape(1, -1), batch_size, 0)
        new_states = states_class(batch_size=batch_size, test="test", data=data)
        split_states = tuple(new_states.split_states(batch_size))
        merged = new_states.merge_states(split_states)
        assert len(merged) == batch_size
        assert merged.test == "test"
        assert (merged.data == data).all()

        split_states = tuple(new_states.split_states(5))
        merged = new_states.merge_states(split_states)
        assert len(merged) == batch_size
        assert merged.test == "test"
        assert (merged.data == data).all()

    def test_merge_states_with_atari(self):
        swarm = create_atari_swarm()
        for states in (swarm.walkers.states, swarm.walkers.env_states, swarm.walkers.model_states):
            split_states = tuple(states.split_states(states.n))
            merged = states.merge_states(split_states)
            assert len(merged) == states.n
            if (
                Backend.is_numpy() and Backend.use_true_hash()
            ):  # Pytorch hashes are not real hashes
                assert hash(merged) == hash(states)
