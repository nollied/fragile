# import numpy
import pytest

from fragile.backend import dtype, random_state, tensor
from fragile.core.utils import get_plangym_env, remove_notebook_margin, resize_frame


class TestUtils:
    def test_remove_notebook_margin_not_crashes(self):
        remove_notebook_margin()

    @pytest.mark.parametrize("size", [(64, 64, 3), (21, 73, 3), (70, 20, 3)])
    def test_resize_frame_rgb(self, size):
        random_state.seed(160290)
        frame = random_state.randint(0, 255, size=size, dtype=dtype.uint8)
        resized = resize_frame(frame, width=size[1], height=size[0], mode="RGB")
        assert size == resized.shape

    @pytest.mark.parametrize("size", [(64, 64), (21, 73), (70, 2)])
    def test_resize_frame_grayscale(self, size):
        random_state.seed(160290)
        frame = random_state.randint(0, 255, size=size + (3,), dtype=dtype.uint8)
        resized = resize_frame(frame, size[1], size[0], "L")
        assert size == resized.shape

    def test_get_plangym_env(self):
        class dummy_shape:
            shape = (2, 2)

        class DummyEnv:
            def __init__(self):
                class dummy_n:
                    n = 1

                self.action_space = dummy_n
                self.observation_space = dummy_shape

            def get_state(self):
                return dummy_shape

        class DummySwarm:
            @property
            def env(self):
                class dummy_env:
                    _env = DummyEnv

                return dummy_env

        swarm = DummySwarm()
        with pytest.raises(TypeError):
            get_plangym_env(swarm)

        class DummySwarm:
            @property
            def env(self):
                from fragile.core.env import DiscreteEnv

                return DiscreteEnv(DummyEnv())

        swarm = DummySwarm()
        with pytest.raises(TypeError):
            get_plangym_env(swarm)
