# import holoviews
from holoviews.streams import Pipe  # Buffer
import pandas
import panel

from fragile.dataviz.stream_plots import (  # Histogram,; Landscape2D,; QuadMeshContours,
    PlotCallback,
    Scatter,
)


class Plot2DSwarm(PlotCallback):
    name = "plot_2d"

    def __init__(self, **kwargs):
        super(Plot2DSwarm, self).__init__(**kwargs)
        stream = Pipe(data=pandas.DataFrame({"x": [0], "y": [0], "c": [0]}))

        self.walkers_scatter_sp = Scatter(
            title="Score Landscape",
            stream=stream,
            color="c",
            cmap="viridis",
            colorbar=True,
            # plot=lambda data: holoviews.Scatter(data).opts(color="c")
        )
        # self.landscape_2d_sp = QuadMeshContours()

    def setup(self, swarm):
        super(Plot2DSwarm, self).setup(swarm)
        # bounds = swarm.env.bounds
        # xlim, ylim = (bounds.low[0], bounds.high[0]), (bounds.low[1], bounds.high[1])
        # self.landscape_2d_sp.plot = self.landscape_2d_sp.plot.redim(
        #    x=holoviews.Dimension("x", range=xlim),
        #    y=holoviews.Dimension("y", range=ylim),
        # )

    def send(self):
        observs = self.get("observs")
        scores = self.get("scores")
        df = pandas.DataFrame({"x": observs[:, 0], "y": observs[:, 1], "c": scores})
        # data = (observs[:, 0], observs[:, 1], scores)
        self.walkers_scatter_sp.send(df)
        # self.landscape_2d_sp.send(data)

    def panel(self):
        return panel.Column(self.walkers_scatter_sp.plot)  # + self.landscape_2d_sp.plot)
