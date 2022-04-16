from typing import Callable, Tuple

import holoviews
from holoviews import Store
from holoviews.streams import Buffer, Pipe
import judo
import numpy
import pandas
import pandas as pd
from scipy.interpolate import griddata

from fragile.core.api_classes import Callback


class StreamingPlot:
    """Represents a holoviews plot updated with streamed data."""

    name = ""
    default_opts = {"framewise": True, "axiswise": True, "normalize": True, "shared_axes": False}
    stream_class = Pipe
    default_bokeh_opts = {}
    default_mpl_opts = {}

    def __init__(
        self,
        plot: Callable,
        stream=None,
        data=None,
        bokeh_opts: dict = None,
        mpl_opts: dict = None,
        **kwargs,
    ):
        """
        Initialize a :class:`StreamingPlot`.

        Args:
            plot: Callable that returns a holoviews plot.
            stream: Class used to stream data to the plot.
            data: Passed to :class:`Plot`.``get_plot_data``. Contains the necessary data to
                initialize the plot.
            args: Passed to ``opts``.
            kwargs: Passed to ``opts``.
        """
        self.data_stream = None
        self.plot = None
        bokeh_opts = bokeh_opts if bokeh_opts is not None else {}
        mpl_opts = mpl_opts if mpl_opts is not None else {}
        self.bokeh_opts = {**self.default_bokeh_opts, **bokeh_opts}
        self.mpl_opts = {**self.default_mpl_opts, **mpl_opts}
        self.common_kwargs = {**self.default_opts, **kwargs}
        self.init_stream(stream, data)
        self.init_plot(plot)

    @property
    def opts_kwargs(self):
        if Store.current_backend == "bokeh":
            backend_kwargs = self.bokeh_opts
        elif Store.current_backend == "matplotlib":
            backend_kwargs = self.mpl_opts
        else:
            backend_kwargs = {}
        return {**self.common_kwargs, **backend_kwargs}

    @opts_kwargs.setter
    def opts_kwargs(self, kwargs):
        self.common_kwargs = {**self.common_kwargs, **kwargs}

    def get_default_data(self):
        raise NotImplementedError()

    def get_default_stream(self, data):
        return self.stream_class(data=data)

    def preprocess_data(self, data):
        """Perform the necessary data wrangling for plotting the data."""
        return data

    def send(self, data) -> None:
        """Stream data to the plot and keep track of how many times the data has been streamed."""
        data = self.preprocess_data(data)
        self.data_stream.send(data)

    def init_plot(self, plot: Callable) -> None:
        """
        Initialize the holoviews plot to accept streaming data.

        Args:
            plot: Callable that returns a holoviews plot.

        """

        def plot_func(data):
            _plot = plot(data)
            return self.opts(plot=_plot)

        self.plot = holoviews.DynamicMap(plot_func, streams=[self.data_stream])
        # self.opts()

    def init_stream(self, stream=None, data=None):
        """Initialize the data stream that will be used to stream data to the plot."""
        if stream is None:
            data = self.preprocess_data(data) if data is not None else self.get_default_data()
            stream = self.get_default_stream(data=data)
        self.data_stream = stream

    def update_kwargs(self, **kwargs):
        """Update the supplied options kwargs with backend specific parameters."""
        if Store.current_backend == "bokeh":
            opt_dict = dict(self.bokeh_opts)
        elif Store.current_backend == "matplotlib":
            opt_dict = dict(self.mpl_opts)
        else:
            opt_dict = {}
        opt_dict.update(kwargs)
        return opt_dict

    def opts(self, plot=None, **kwargs):
        """Update the plot parameters. Same as `holoviews` `opts`."""
        if self.plot is None:
            return
        self.common_kwargs.update(kwargs)
        if plot is None:
            self.plot = self.plot.opts(**self.opts_kwargs)
        else:
            return plot.opts(**self.opts_kwargs)


class Table(StreamingPlot):
    """``holoviews.Table`` with data streaming capabilities."""

    default_opts = {
        "framewise": True,
        "axiswise": True,
        "normalize": True,
    }

    name = "table"
    default_bokeh_opts = {
        "height": 350,
        "width": 350,
    }

    def __init__(
        self,
        data=None,
        plot=holoviews.Table,
        **kwargs,
    ):
        """
        Initialize a :class:`Table`.

        Args:
            data: Data to initialize the stream.
            stream: :class:`holoviews.stream` type. Defaults to :class:`Pipe`.
            bokeh_opts: Default options for the plot when rendered using the "bokeh" backend.
            mpl_opts: Default options for the plot when rendered using the "matplotlib" backend.
            **kwargs: Passed to :class:`StreamingPlot`.

        """
        super(Table, self).__init__(
            plot=plot,
            data=data,
            **kwargs,
        )

    def get_default_data(self):
        return pandas.DataFrame()

    def opts(self, plot=None, **kwargs):
        """Update the plot parameters. Same as `holoviews` `opts`."""
        if self.plot is None:
            return
        self.common_kwargs.update(kwargs)
        if plot is None:
            self.plot = self.plot.opts(holoviews.opts.Table(**self.opts_kwargs))
        else:
            return plot.opts(holoviews.opts.Table(**self.opts_kwargs))


class RGB(StreamingPlot):
    """``holoviews.RGB`` with data streaming capabilities."""

    name = "rgb"

    def __init__(self, data=None, plot=holoviews.RGB, **kwargs):
        """Initialize a :class:`RGB`."""
        super(RGB, self).__init__(plot=plot, data=data, **kwargs)

    def get_default_data(self):
        return []

    def opts(self, plot=None, **kwargs):
        """Update the plot parameters. Same as `holoviews` `opts`."""
        if self.plot is None:
            return
        self.common_kwargs.update(kwargs)
        if plot is None:
            self.plot = self.plot.opts(holoviews.opts.RGB(**self.opts_kwargs))
        else:
            return plot.opts(holoviews.opts.RGB(**self.opts_kwargs))


class Curve(StreamingPlot):
    """
    Create a ``holoviews.Curve`` plot that plots steaming data.

    The streaming process is handled using a :class:`Buffer`.
    """

    name = "curve"
    default_bokeh_opts = {
        "height": 350,
        "width": 400,
        "shared_axes": False,
        "tools": ["hover"],
    }
    stream_class = Buffer

    def __init__(
        self,
        data=None,
        plot=holoviews.Curve,
        buffer_length: int = 10000,
        index: bool = False,
        data_names=("x", "y"),
        **kwargs,
    ):
        """
        Initialize a :class:`Curve`.

        Args:
            buffer_length: Maximum number of data points that will be displayed in the plot.
            index: Passed to the :class:`Buffer` that streams data to the plot.
            data: Passed to :class:`Plot`.``get_plot_data``. Contains the necessary data to \
                  initialize the plot.
            bokeh_opts: Default options for the plot when rendered using the \
                       "bokeh" backend.
            mpl_opts: Default options for the plot when rendered using the \
                    "matplotlib" backend.
        """

        self._buffer_length = buffer_length
        self._index = index

        self._data_names = data.columns.values if data is not None else data_names
        super(Curve, self).__init__(
            plot=plot,
            data=data,
            **kwargs,
        )

    def get_default_stream(self, data):
        return self.stream_class(data=data, length=self._buffer_length, index=self._index)

    def get_default_data(self):
        return pandas.DataFrame(columns=self._data_names)

    def opts(self, plot=None, **kwargs):
        """Update the plot parameters. Same as `holoviews` `opts`."""
        if self.plot is None:
            return
        self.common_kwargs.update(kwargs)
        if plot is None:
            self.plot = self.plot.opts(holoviews.opts.Curve(**self.opts_kwargs))
        else:
            return plot.opts(holoviews.opts.Curve(**self.opts_kwargs))


class Histogram(StreamingPlot):
    """
    Create a ``holoviews.Histogram`` plot that plots steaming data.

    The streaming process is handled using a :class:`Pipe`.
    """

    name = "histogram"
    default_opts = {
        "ylabel": "count",
        "framewise": True,
        "axiswise": True,
        "normalize": True,
        "shared_axes": False,
    }
    default_bokeh_opts = {"tools": ["hover"]}

    def __init__(
        self,
        data=None,
        plot=None,
        n_bins: int = 20,
        **kwargs,
    ):
        """
        Initialize a :class:`Histogram`.

        Args:
            n_bins: Number of bins of the histogram that will be plotted.
            data: Used to initialize the plot.
            bokeh_opts: Default options for the plot when rendered using the \
                       "bokeh" backend.
            mpl_opts: Default options for the plot when rendered using the \
                    "matplotlib" backend.
        """
        self.n_bins = n_bins
        self.xlim = (None, None)
        super(Histogram, self).__init__(
            plot=self.plot_histogram if plot is None else plot,
            data=data,
            **kwargs,
        )

    @staticmethod
    def plot_histogram(data):
        """
        Plot the histogram.

        Args:
            data: Tuple containing (values, bins), xlim. xlim is a tuple \
                  containing two typing_.Scalars that represent the limits of the x \
                  axis of the histogram.

        Returns:
            Histogram plot.

        """
        plot_data, xlim = data
        return holoviews.Histogram(plot_data).redim(x=holoviews.Dimension("x", range=xlim))

    def opts(self, plot=None, **kwargs):
        """Update the plot parameters. Same as `holoviews` `opts`."""
        if self.plot is None:
            return
        self.common_kwargs.update(kwargs)
        if plot is None:
            self.plot = self.plot.opts(holoviews.opts.Histogram(**self.opts_kwargs))
        else:
            return plot.opts(holoviews.opts.Histogram(**self.opts_kwargs))

    def preprocess_data(self, data):
        data = judo.to_numpy(data)
        # print(data)
        data[numpy.isnan(data)] = 0.0
        return numpy.histogram(data, self.n_bins), self.xlim

    def get_default_data(self):
        data = numpy.arange(10)
        return self.preprocess_data(data)


class Bivariate(StreamingPlot):
    """
    Create a ``holoviews.Bivariate`` plot that plots steaming data.

    The streaming process is handled using a :class:`Pipe`.
    """

    name = "bivariate"
    default_bokeh_opts = {
        "height": 350,
        "width": 400,
        "tools": ["hover"],
    }

    def __init__(self, data=None, plot=holoviews.Bivariate, **kwargs):
        """
        Initialize a :class:`Bivariate`.

        Args:
            data: Passed to ``holoviews.Bivariate``.
            bokeh_opts: Default options for the plot when rendered using the \
                       "bokeh" backend.
            mpl_opts: Default options for the plot when rendered using the \
                    "matplotlib" backend.
            *args: Passed to ``holoviews.Bivariate``.
            **kwargs: Passed to ``holoviews.Bivariate``.
        """
        super(Bivariate, self).__init__(
            plot=plot,
            data=data,
            **kwargs,
        )

    def opts(self, plot=None, **kwargs):
        """Update the plot parameters. Same as `holoviews` `opts`."""
        if self.plot is None:
            return
        self.common_kwargs.update(kwargs)
        if plot is None:
            self.plot = self.plot.opts(holoviews.opts.Bivariate(**self.opts_kwargs))
        else:
            return plot.opts(holoviews.opts.Bivariate(**self.opts_kwargs))

    def __opts(
        self,
        title="",
        xlabel: str = "x",
        ylabel: str = "y",
        framewise: bool = True,
        axiswise: bool = True,
        normalize: bool = True,
        *args,
        **kwargs,
    ):
        """
        Update the plot parameters. Same as ``holoviews`` ``opts``.

        The default values updates the plot axes independently when being \
        displayed in a :class:`Holomap`.
        """
        kwargs = self.update_kwargs(**kwargs)
        # Add specific defaults to Scatter
        scatter_kwargs = dict(kwargs)
        if Store.current_backend == "bokeh":
            scatter_kwargs["size"] = scatter_kwargs.get("size", 3.5)
        elif Store.current_backend == "matplotlib":
            scatter_kwargs["s"] = scatter_kwargs.get("s", 15)
        self.plot = self.plot.opts(
            holoviews.opts.Bivariate(
                title=title,
                xlabel=xlabel,
                ylabel=ylabel,
                framewise=framewise,
                axiswise=axiswise,
                normalize=normalize,
                *args,
                **kwargs,
            ),
            holoviews.opts.Scatter(
                alpha=0.7,
                xlabel=xlabel,
                ylabel=ylabel,
                framewise=framewise,
                axiswise=axiswise,
                normalize=normalize,
                *args,
                **scatter_kwargs,
            ),
            holoviews.opts.NdOverlay(
                normalize=normalize,
                framewise=framewise,
                axiswise=axiswise,
            ),
        )

    def get_default_data(self):
        return []


class QuadMesh(StreamingPlot):
    """
    Create a ``holoviews.Histogram`` plot that plots steaming data.

    The streaming process is handled using a :class:`Pipe`.
    """

    name = "quadmesh"
    default_bokeh_opts = {
        "tools": ["hover"],
        "bgcolor": "lightgray",
        "colorbar": True,
        "height": 350,
        "line_width": 1.0,
        "width": 400,
        "shared_axes": False,
        "cmap": "viridis",
    }

    # default_mpl_opts = {"linewidth": 1.0}

    def __init__(
        self,
        data=None,
        plot=None,
        n_points: int = 20,
        **kwargs,
    ):
        """
        Initialize a :class:`Histogram`.

        Args:
            n_bins: Number of bins of the histogram that will be plotted.
            data: Used to initialize the plot.
        """
        self.n_points = n_points
        super(QuadMesh, self).__init__(
            plot=self.plot_quadmesh if plot is None else plot,
            data=data,
            **kwargs,
        )

    def send(self, data, xx=None, yy=None, zz=None) -> None:
        """Stream data to the plot and keep track of how many times the data has been streamed."""
        data = self.preprocess_data(data, xx=xx, yy=yy, zz=zz)
        self.data_stream.send(data)

    def plot_quadmesh(self, data):
        """
        Plot the data as an energy landscape.

        Args:
            data: (x, y, xx, yy, z, xlim, ylim). x, y, z represent the \
                  coordinates of the points that will be interpolated. xx, yy \
                  represent the meshgrid used to interpolate the points. xlim, \
                  ylim are tuples containing the limits of the x and y axes.

        Returns:
            Plot representing the interpolated energy landscape of the target points.

        """
        xx, yy, zz, xlim, ylim, zlim = data
        plot = holoviews.QuadMesh((xx, yy, zz))
        return plot
        return plot.redim(
            x=holoviews.Dimension("x", range=xlim),
            y=holoviews.Dimension("y", range=ylim),
            z=holoviews.Dimension("z", range=zlim),
        )

    def preprocess_data(
        self,
        data: Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray],
        xx=None,
        yy=None,
        zz=None,
    ):
        """Create the meshgrid needed to interpolate the target data points."""
        x, y, z = (data[:, 0], data[:, 1], data[:, 2]) if isinstance(data, numpy.ndarray) else data
        x, y, z = judo.to_numpy(x), judo.to_numpy(y), judo.to_numpy(z)
        # target grid to interpolate to
        if zz is None:
            xi = numpy.linspace(x.min(), x.max(), self.n_points)
            yi = numpy.linspace(y.min(), y.max(), self.n_points)
            xx, yy = numpy.meshgrid(xi, yi)
            zz = griddata((x, y), z, (xx, yy), method="linear")
        xlim, ylim, zlim = (x.min(), x.max()), (y.min(), y.max()), (z.min(), z.max())
        return xx, yy, zz, xlim, ylim, zlim

    def get_default_data(self):
        X = numpy.random.standard_normal((10, 2))
        z = numpy.random.standard_normal(10)
        data = X[:, 0], X[:, 1], z
        return self.preprocess_data(data)

    def opts(self, plot=None, **kwargs):
        """Update the plot parameters. Same as `holoviews` `opts`."""
        if self.plot is None:
            return
        self.common_kwargs.update(kwargs)
        if plot is None:
            self.plot = self.plot.opts(holoviews.opts.QuadMesh(**self.opts_kwargs))
        else:
            return plot.opts(holoviews.opts.QuadMesh(**self.opts_kwargs))


class QuadMeshContours(QuadMesh):
    default_opts = {"shared_axes": True}
    default_bokeh_opts = {
        "tools": ["hover"],
        "bgcolor": "lightgray",
        "alpha": 0.9,
        "show_legend": False,
        "height": 350,
        "width": 400,
        "line_color": "black",
        # "shared_axes": False,
        "colorbar": True,
        "cmap": "viridis",  # ["black"],
    }

    def __init__(self, data=None, levels: int = 16, **kwargs):
        self.levels = levels
        super(QuadMeshContours, self).__init__(data=data, **kwargs)

    def plot_quadmesh(self, data):
        """
        Plot the data as an energy landscape.

        Args:
            data: (x, y, xx, yy, z, xlim, ylim). x, y, z represent the \
                  coordinates of the points that will be interpolated. xx, yy \
                  represent the meshgrid used to interpolate the points. xlim, \
                  ylim are tuples containing the limits of the x and y axes.

        Returns:
            Plot representing the interpolated energy landscape of the target points.

        """
        xx, yy, zz, xlim, ylim, zlim = data
        mesh = holoviews.QuadMesh((xx, yy, zz))
        plot = holoviews.operation.contours(mesh, levels=self.levels)
        return (mesh * plot).redim(
            x=holoviews.Dimension("x", range=xlim),
            y=holoviews.Dimension("y", range=ylim),
            z=holoviews.Dimension("z", range=zlim),
        )

    def send(self, data, xx=None, yy=None, zz=None) -> None:
        x, y, z = data
        self.common_kwargs["zlim"] = (z.min(), z.max())
        super(QuadMeshContours, self).send(data=data, xx=xx, yy=yy, zz=zz)

    def opts(self, plot=None, **kwargs):
        """Update the plot parameters. Same as `holoviews` `opts`."""
        if self.plot is None:
            return
        self.common_kwargs.update(kwargs)
        if plot is None:
            self.plot = self.plot.opts(
                holoviews.opts.Contours(**{**self.opts_kwargs, **{"cmap": ["black"]}}),
                holoviews.opts.QuadMesh(**self.opts_kwargs),
            )
        else:
            return plot.opts(
                holoviews.opts.Contours(**{**self.opts_kwargs, **{"cmap": ["black"]}}),
                holoviews.opts.QuadMesh(**self.opts_kwargs),
            )


class Scatter(StreamingPlot):
    default_bokeh_opts = {
        "tools": ["hover"],
        "bgcolor": "lightgray",
        "line_color": "black",
        # "fill_color": "red",
        "height": 350,
        "size": 3.5,
        "width": 400,
        "shared_axes": False,
    }
    default_mpl_opts = {"s": 1}

    def __init__(
        self,
        data=None,
        plot=holoviews.Scatter,
        n_points: int = 20,
        **kwargs,
    ):
        """
        Initialize a :class:`Histogram`.

        Args:
            n_bins: Number of bins of the histogram that will be plotted.
            data: Used to initialize the plot.
        """
        self.n_points = n_points
        super(Scatter, self).__init__(
            plot=plot,
            data=data,
            **kwargs,
        )

    def get_default_data(self):
        return pd.DataFrame({"x": [], "y": []})

    def opts(self, plot=None, **kwargs):
        """Update the plot parameters. Same as `holoviews` `opts`."""
        if self.plot is None:
            return
        self.common_kwargs.update(kwargs)
        if plot is None:
            self.plot = self.plot.opts(holoviews.opts.Scatter(**self.opts_kwargs))
        else:
            return plot.opts(holoviews.opts.Scatter(**self.opts_kwargs))


class Landscape2D(StreamingPlot):
    """
    Plots the interpolated landscaped of values of a set of points.

    The data is visualized creating a :class:`holoviews.QuadMesh` with a \
    :class:`holoviews.Contours` plot with the original data points displayed as \
    a :class:`holoviews.Scatter`.
    """

    name = "landscape"
    default_bokeh_opts = {
        "height": 350,
        "width": 400,
        "tools": ["hover"],
        "shared_axes": False,
    }

    def __init__(
        self,
        data=None,
        contours: bool = True,
        **kwargs,
    ):
        """
        Initialize a :class:`Landscape2d`.

        Args:
            n_points: Number of points per dimension used to create the \
                      mesh-grid grid that will be used to interpolate the data.
            data: Initial data for the plot.
            bokeh_opts: Default options for the plot when rendered using the \
                       "bokeh" backend.
            mpl_opts: Default options for the plot when rendered using the \
                    "matplotlib" backend.
            invert_cmap: If ``True``, invert the colormap to assign high value \
                         colors to the lowest values.

        """
        self.quadmesh_ps = QuadMesh(data=data, **kwargs)
        self.contours_ps = (
            None
            if not contours
            else QuadMeshContours(data=data, stream=self.quadmesh_ps.data_stream, **kwargs)
        )
        self.scatter_ps = Scatter(data=data)
        super(Landscape2D, self).__init__(
            stream=self.quadmesh_ps.data_stream,
            plot=self.plot_landscape,
            data=data,
            **kwargs,
        )

    def init_plot(self, plot: Callable) -> None:
        """
        Initialize the holoviews plot to accept streaming data.

        Args:
            plot: Callable that returns a holoviews plot.

        """
        # self.plot = self.quadmesh_ps.plot
        if True:  # self.contours_ps is not None:
            self.plot = self.contours_ps.plot

        # self.plot = self.plot * self.scatter_ps.plot

    def plot_landscape(self, data):
        """
        Plot the data as an energy landscape.

        Args:
            data: (x, y, xx, yy, z, xlim, ylim). x, y, z represent the \
                  coordinates of the points that will be interpolated. xx, yy \
                  represent the meshgrid used to interpolate the points. xlim, \
                  ylim are tuples containing the limits of the x and y axes.

        Returns:
            Plot representing the interpolated energy landscape of the target points.

        """
        plot = self.quadmesh_ps.plot_quadmesh(data)
        if self.contours_ps:
            plot = plot * self.contours_ps.plot_quadmesh(data)
        return plot.opts(
            holoviews.opts.QuadMesh(**self.quadmesh_ps.opts_kwargs),
            holoviews.opts.Contours(**self.contours_ps.opts_kwargs),
        )

    def preprocess_data(self, data):
        return self.quadmesh_ps.preprocess_data(data)

    def send(self, data) -> None:
        """Stream data to the plot and keep track of how many times the data has been streamed."""
        self.quadmesh_ps.send(data)
        df = pandas.DataFrame({"x": data[0], "y": data[1]})
        self.scatter_ps.send(df)
        # self.plot = self.plot.opts(framewise=True)
        xlim, ylim = (data[0].min(), data[0].max()), (data[1].min(), data[1].max())
        zlim = (data[2].min(), data[2].max())
        self.quadmesh_ps.plot = self.quadmesh_ps.plot.redim(
            x=holoviews.Dimension("x", range=xlim),
            y=holoviews.Dimension("y", range=ylim),
            z=holoviews.Dimension("z", range=zlim),
        )
        self.contours_ps.plot = self.contours_ps.plot.redim(
            x=holoviews.Dimension("x", range=xlim),
            y=holoviews.Dimension("y", range=ylim),
            z=holoviews.Dimension("z", range=zlim),
        )
        # self.plot = self.plot

    def get_default_data(self):
        return self.quadmesh_ps.get_default_data()

    def opts(self, **kwargs):
        self.quadmesh_ps.opts(**kwargs)
        if self.contours_ps is not None:
            self.contours_ps.opts(**kwargs)

    def __opts(
        self,
        title="Distribution landscape",
        xlabel: str = "x",
        ylabel: str = "y",
        framewise: bool = True,
        axiswise: bool = True,
        normalize: bool = True,
        cmap: str = "default",
        **kwargs,
    ):
        """
        Update the plot parameters. Same as ``holoviews`` ``opts``.

        The default values updates the plot axes independently when being \
        displayed in a :class:`Holomap`.
        """
        kwargs = self.update_kwargs(**kwargs)
        cmap = cmap if cmap != "default" else ("viridis_r" if self.invert_cmap else "viridis")
        # Add specific defaults to Contours
        contours_kwargs = dict(kwargs)
        if Store.current_backend == "bokeh":
            contours_kwargs["line_width"] = contours_kwargs.get("line_width", 1)
        elif Store.current_backend == "matplotlib":
            contours_kwargs["linewidth"] = contours_kwargs.get("linewidth", 1)

        # Add specific defaults to Scatter
        scatter_kwargs = dict(kwargs)
        if Store.current_backend == "bokeh":
            scatter_kwargs["fill_color"] = scatter_kwargs.get("fill_color", "red")
            scatter_kwargs["size"] = scatter_kwargs.get("size", 3.5)
        elif Store.current_backend == "matplotlib":
            scatter_kwargs["color"] = scatter_kwargs.get("color", "red")
            scatter_kwargs["s"] = scatter_kwargs.get("s", 15)

        self.plot = self.plot.opts(
            holoviews.opts.QuadMesh(
                cmap=cmap,
                colorbar=True,
                title=title,
                bgcolor="lightgray",
                xlabel=xlabel,
                ylabel=ylabel,
                framewise=framewise,
                axiswise=axiswise,
                normalize=normalize,
                **kwargs,
            ),
            holoviews.opts.Contours(
                cmap=["black"],
                alpha=0.9,
                title=title,
                xlabel=xlabel,
                ylabel=ylabel,
                show_legend=False,
                framewise=framewise,
                axiswise=axiswise,
                normalize=normalize,
                **contours_kwargs,
            ),
            holoviews.opts.Scatter(
                alpha=0.7,
                xlabel=xlabel,
                ylabel=ylabel,
                framewise=framewise,
                axiswise=axiswise,
                normalize=normalize,
                **scatter_kwargs,
            ),
            holoviews.opts.NdOverlay(
                normalize=normalize,
                framewise=framewise,
                axiswise=axiswise,
            ),
        )


class PlotCallback(Callback):
    def __init__(self, report_interval: int = 1, **kwargs):
        self.report_interval = report_interval
        super(PlotCallback, self).__init__(**kwargs)

    def after_evolve(self):
        super(PlotCallback, self).before_walkers()
        if self.swarm.epoch % self.report_interval == 0 or self.swarm.epoch == 0:
            self.send()

    def send(self):
        raise NotImplementedError()

    def panel(self):
        raise NotImplementedError()

    def run_end(self):
        self.send()
