from typing import Union

import judo
from judo.data_types import dtype
from judo.functions.random import random_state
import numpy

from fragile.core.api_classes import Callback
from fragile.core.typing import StateData, Tensor


class TimeStepAPI(Callback):
    name = "time_step"
    default_param_dict = {"dt": {"dtype": dtype.int32}}
    default_outputs = tuple(["dt"])

    def select(self, inplace: bool = True, **kwargs) -> Union[None, StateData]:
        """Calculate SwarmState containing the data needed to interact with the environment."""
        dt_input = self._prepare_tensors(**kwargs)
        dt_data = self.calculate(**dt_input)
        if not isinstance(dt_data, dict):
            dt_data = {"dt": dt_data}
        if inplace:
            self.update(**dt_data)
        else:
            return dt_data

    def calculate(self, **kwargs) -> Tensor:
        raise NotImplementedError

    def reset(self, inplace: bool = True, **kwargs) -> Union[None, StateData]:
        return self.select(inplace=inplace)

    def before_policy(self):
        self.select()


class ConstantDt(TimeStepAPI):
    def __init__(self, value: int, **kwargs):
        super(ConstantDt, self).__init__(**kwargs)
        self.value = value

    def calculate(self) -> Tensor:
        return self.value * judo.ones(self.swarm.n_walkers, dtype=dtype.int32)


class UniformDt(TimeStepAPI):
    def __init__(self, high: int, low: int = 1, **kwargs):
        super(UniformDt, self).__init__(**kwargs)
        self.low = low
        self.high = high

    def calculate(self) -> Tensor:
        return random_state.randint(
            low=self.low,
            high=self.high,
            size=self.swarm.n_walkers,
            dtype=dtype.int32,
        )

    def reset(self, inplace: bool = True, **kwargs) -> Union[None, StateData]:
        dt_data = {"dt": self.low * judo.ones(self.swarm.n_walkers, dtype=dtype.int32)}
        if inplace:
            self.update(**dt_data)
        else:
            return dt_data


class GaussianDt(UniformDt):
    def __init__(self, loc: float, scale: float, low: int = 1, high=numpy.inf, **kwargs):
        self.loc = loc
        self.scale = scale
        super(GaussianDt, self).__init__(high=high, low=low, **kwargs)

    def calculate(self) -> Tensor:
        dts = random_state.normal(loc=self.loc, scale=self.scale, size=self.n_walkers)
        return judo.astype(judo.clip(dts, self.low, self.high), dtype=dtype.int32)
