import numpy

from fragile.core.api_classes import Callback


class KillWorstWalkers(Callback):
    name = "oobs_kill_worst"
    default_inputs = {"oobs": {}, "scores": {}}

    def __init__(self, stds: float = 1.0, **kwargs):
        super(KillWorstWalkers, self).__init__(**kwargs)
        self._stds = stds

    def oob_condition(self):
        scores = self.get("scores")
        return scores < (scores.max() - scores.std() * self._stds)

    def after_env(self):
        oobs = numpy.logical_or(self.get("oobs"), self.oob_condition())
        self.update(oobs=oobs)
