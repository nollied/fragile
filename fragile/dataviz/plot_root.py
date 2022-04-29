from typing import Tuple, Union

import numpy
import pandas as pd
import panel

from fragile.dataviz.stream_plots import Curve, Div, PlotCallback, RGB, Table


class PlotRootWalker(PlotCallback):
    name = "plot_root"

    def __init__(self, image_shape=None, step_after_set_state: bool = False, **kwargs):
        super(PlotRootWalker, self).__init__(**kwargs)
        self._image_available = False
        self._image_shape = image_shape
        self.table = Table(title="Run Summary", height=50)
        self.curve = Curve(
            data_names=["epoch", "score"],
            title="Score",
            xlabel="Epoch",
            ylabel="Score",
        )
        self.info = Div(title="Root info")
        self.image = None
        self._last_score = -numpy.inf
        self._step_after_set_state = step_after_set_state

    @property
    def root(self):
        return self.swarm.root

    @property
    def image_shape(self) -> Union[None, Tuple[int, ...]]:
        return self._image_shape

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
            if self.image_shape is None:
                h, w, *_ = first_img.shape
                self._image_shape = (h, w)
            self.image = RGB(data=first_img, title="Root Image")
            self.image.opts(height=self.image_shape[0], width=self.image_shape[1])

    def send(self):
        if hasattr(self.root, "info"):
            self.info.send(repr(self.root.info))
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
            return panel.Column(summary, panel.Row(self.info.plot, self.image.plot))
        return summary

    def image_from_state(self, state):
        self.swarm.env.plangym_env.set_state(state)
        if self._step_after_set_state:
            old_frameskip = int(self.swarm.env.plangym_env.frameskip)
            self.swarm.env.plangym_env.frameskip = 1
            self.swarm.env.plangym_env.step(self.swarm.env.plangym_env.sample_action())
            self.swarm.env.plangym_env.frameskip = old_frameskip
        return self.swarm.env.plangym_env.get_image().astype(numpy.uint8)

    def reset(self, root_walker=None, state=None, **kwargs):
        self.curve.data_stream.clear()
        return super(PlotRootWalker, self).reset(root_walker=root_walker, state=state, **kwargs)


class PlotMario(PlotRootWalker):
    def __init__(self, **kwargs):
        super(PlotRootWalker, self).__init__(**kwargs)
        self._image_available = False
        self.run_table = Table(title="Run Summary", height=50)
        self.info_table = Table(title="State info", height=50, width=600)
        self.curve = Curve(
            data_names=["epoch", "score"],
            title="Score",
            xlabel="Epoch",
            ylabel="Score",
        )
        self.image = None
        self._last_score = -numpy.inf

    def send(self):
        current_score = self.root.score
        summary_table = pd.DataFrame(
            columns=["epoch", "best_score", "pct_oobs"],
            data=[[self.swarm.epoch, current_score, self.get("oobs").mean()]],
        )
        info_df = pd.DataFrame(
            columns=list(self.root.info.keys()),
            data=[list(self.root.info.values())],
        )
        self.info_table.send(info_df)
        self.run_table.send(summary_table)
        score_data = pd.DataFrame(
            columns=["epoch", "score"],
            data=[[self.swarm.epoch, current_score]],
        )
        self.curve.send(score_data)
        if self._image_available and (
            current_score != self._last_score or self.root.always_update
        ):
            if "rgb" in self.root.data:
                img_data = self.root.rgb
            else:
                img_data = self.image_from_state(self.root.state)
            self.image.send(img_data)
        self._last_score = float(current_score)

    def panel(self):
        summary = panel.Row(self.run_table.plot, self.info_table.plot)
        plot_row = (
            panel.Row(self.curve.plot, self.image.plot)
            if self._image_available
            else self.curve.plot
        )
        return panel.Column(summary, plot_row)

    def image_from_state(self, state):
        self.swarm.env.plangym_env.set_state(state)
        # self.swarm.env.plangym_env.step(0)
        return self.swarm.env.plangym_env.get_image().astype(numpy.uint8)
