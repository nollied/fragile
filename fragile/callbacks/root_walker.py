import copy

import judo
import numpy

from fragile.core.api_classes import Callback


class RootWalker(Callback):
    name = "root"

    def __init__(self, **kwargs):
        self._data = {}
        self.minimize = False
        super(RootWalker, self).__init__(**kwargs)

    def __getattr__(self, item):
        plural = item + "s"
        if plural in self._data:
            return self._data[plural][0]
        elif item in self._data:
            return self._data[item][0]
        return self.__getattribute__(item)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}: score: {self.data.get('scores', [numpy.nan])[0]}"

    def to_html(self):
        return (
            f"<strong>{self.__class__.__name__}</strong>: "
            f"Score: {self.data.get('scores', [numpy.nan])[0]}\n"
        )

    @property
    def data(self):
        return self._data

    def setup(self, swarm):
        super(RootWalker, self).setup(swarm)
        self.minimize = swarm.minimize

    def reset(self, root_walker=None, state=None, **kwargs):
        if root_walker is None:
            value = [numpy.inf if self.minimize else -numpy.inf]
            self._data = {"scores": value, "rewards": value}
            self.update_root()
        else:
            self._data = {k: copy.deepcopy(v) for k, v in root_walker.items()}

    def before_walkers(self):
        self.update_root()

    def update_root(self):
        raise NotImplementedError()


class BestWalker(RootWalker):
    default_inputs = {"scores": {}, "oobs": {"optional": True}}

    def __init__(self, always_update: bool = False, fix_root=True, **kwargs):
        super(BestWalker, self).__init__(**kwargs)
        self.minimize = None
        self.always_update = always_update
        self._fix_root = fix_root

    def get_best_index(self):
        scores, oobs = self.get("scores"), self.get("oobs")
        index = judo.arange(len(scores))
        ix = scores[~oobs].argmin() if self.minimize else scores[~oobs].argmax()
        return index[~oobs][ix]

    def get_best_walker(self):
        return self.swarm.state.export_walker(self.get_best_index())

    def update_root(self):
        best = self.get_best_walker()
        score_improves = (
            (best["scores"][0] < self.score) if self.minimize else (best["scores"][0] > self.score)
        )
        if self.always_update or score_improves or numpy.isinf(self.score):
            # new_best = {k: copy.deepcopy(v) for k, v in best.items()}
            self._data = copy.deepcopy(best)

    def fix_root(self):
        if self._fix_root:
            self.swarm.state.import_walker(self.data)

    def after_env(self):
        self.fix_root()


class TrackWalker(RootWalker):
    default_inputs = {"scores": {}, "oobs": {"optional": True}}

    def __init__(self, walker_index=0, **kwargs):
        super(TrackWalker, self).__init__(**kwargs)
        self.walker_index = walker_index

    def update_root(self):
        walker = self.swarm.state.export_walker(self.walker_index)
        self._data = copy.deepcopy({k: copy.deepcopy(v) for k, v in walker.items()})
        # print(self.swarm.root.observs, self.swarm.root.data)
