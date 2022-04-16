import numpy
import pandas as pd
import panel

from fragile.dataviz.stream_plots import Curve, PlotCallback, RGB, Table


class PlotRootWalker(PlotCallback):
    name = "plot_root"

    def __init__(self, **kwargs):
        super(PlotRootWalker, self).__init__(**kwargs)
        self._image_available = False
        self.table = Table(title="Run Summary", height=50)
        self.curve = Curve(
            data_names=["epoch", "score"],
            title="Score",
            xlabel="Epoch",
            ylabel="Score",
        )
        self.image = None
        self._last_score = -numpy.inf

    @property
    def root(self):
        return self.swarm.root

    def setup(self, swarm):
        super(PlotRootWalker, self).setup(swarm)
        self._image_available = (
            hasattr(self.swarm.env, "plangym_env") or "rgb" in self.swarm.param_dict
        )
        if self._image_available:
            first_img = (
                self.root.rgb
                if "rgb" in self.root.data
                else self.swarm.env.plangym_env.get_image().astype(numpy.uint8)
            )
            self.image = RGB(data=first_img, title="Root Image")

    def send(self):
        current_score = self.root.score
        summary_table = pd.DataFrame(
            columns=["epoch", "best_score", "pct_oobs"],
            data=[[self.swarm.epoch, current_score, self.get("oobs").mean()]],
        )
        self.table.send(summary_table)
        score_data = pd.DataFrame(
            columns=["epoch", "score"],
            data=[[self.swarm.epoch, current_score]],
        )
        self.curve.send(score_data)
        if self._image_available and current_score != self._last_score:
            if "rgb" in self.root.data:
                img_data = self.root.rgb
            else:
                img_data = self.image_from_state(self.root.state)
            self.image.send(img_data)
        self._last_score = float(current_score)

    def panel(self):
        summary = panel.Row(self.table.plot, self.curve.plot)
        if self._image_available:
            return panel.Row(summary, self.image.plot)
        return summary

    def image_from_state(self, state):
        self.swarm.env.plangym_env.set_state(state)
        self.swarm.env.plangym_env.step(0)
        return self.swarm.env.plangym_env.get_image().astype(numpy.uint8)

    def reset(self, root_walker=None, state=None, **kwargs):
        self.curve.data_stream.clear()
        return super(PlotRootWalker, self).reset(root_walker=root_walker, state=state, **kwargs)
