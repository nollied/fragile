import panel

from fragile.dataviz.stream_plots import Histogram, Landscape2D, PlotCallback


class Plot2DSwarm(PlotCallback):
    name = "plot_2d"

    def __init__(self, **kwargs):
        super(Plot2DSwarm, self).__init__(**kwargs)
        self.score_landscape_sp = Landscape2D(
            title="Score Landscape", plot_scatter=False, n_points=50
        )

    def send(self):
        observs = self.swarm.memory.observs  # self.get("observs")
        scores = self.swarm.memory.scores.flatten()  # self.get("scores")
        self.score_landscape_sp.send((observs[:, 0], observs[:, 1], scores))

    def panel(self):
        return panel.Column(self.score_landscape_sp.plot)
