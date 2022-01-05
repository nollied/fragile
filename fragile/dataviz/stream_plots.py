from typing import Callable, Tuple, Union

import holoviews
from holoviews import Store
from holoviews.streams import Buffer, Pipe
import judo
from judo.typing import Scalar
import numpy
import pandas
from scipy.interpolate import griddata

from fragile.core.api_classes import Callback


class StreamingPlot:
    """Represents a holoviews plot updated with streamed data."""

    name = ""

    def __init__(
        self,
        plot: Callable,
        stream=Pipe,
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
        self.bokeh_opts = bokeh_opts if bokeh_opts is not None else {}
        self.mpl_opts = mpl_opts if mpl_opts is not None else {}
        self.init_stream(stream, data)
        self.init_plot(plot, **kwargs)

    def get_default_data(self):
        raise NotImplementedError()

    def preprocess_data(self, data):
        """Perform the necessary data wrangling for plotting the data."""
        return data

    def send(self, data) -> None:
        """Stream data to the plot and keep track of how many times the data has been streamed."""
        data = self.preprocess_data(data)
        self.data_stream.send(data)

    def init_plot(self, plot: Callable, **kwargs) -> None:
        """
        Initialize the holoviews plot to accept streaming data.

        Args:
            plot: Callable that returns a holoviews plot.
            kwargs: Passed to ``opts``.

        """
        self.plot = holoviews.DynamicMap(plot, streams=[self.data_stream])
        self.opts(**kwargs)

    def init_stream(self, stream, data=None):
        """Initialize the data stream that will be used to stream data to the plot."""
        data = self.preprocess_data(data) if data is not None else self.get_default_data()
        self.data_stream = stream(data=data)

    @staticmethod
    def update_default_opts(mpl_default, passed_mpl, bokeh_default, passed_bokeh):
        """Update the backend specific parameter default values with external supplied defaults."""
        if passed_bokeh is None:
            bokeh_opts = bokeh_default
        else:
            bokeh_default.update(passed_bokeh)
            bokeh_opts = bokeh_default

        if passed_mpl is None:
            mpl_opts = mpl_default
        else:
            mpl_default.update(passed_mpl)
            mpl_opts = mpl_default
        return mpl_opts, bokeh_opts

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

    def opts(self, **kwargs):
        """Update the plot parameters. Same as `holoviews` `opts`."""
        if self.plot is None:
            return
        opt_dict = self.update_kwargs(**kwargs)
        self.plot = self.plot.opts(**opt_dict)


class Table(StreamingPlot):
    """``holoviews.Table`` with data streaming capabilities."""

    name = "table"

    def __init__(
        self,
        data=None,
        stream=Pipe,
        bokeh_opts: dict = None,
        mpl_opts: dict = None,
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
        default_bokeh_opts = {
            "height": 350,
            "width": 350,
        }
        default_mpl_opts = {}
        mpl_opts, bokeh_opts = self.update_default_opts(
            default_mpl_opts,
            mpl_opts,
            default_bokeh_opts,
            bokeh_opts,
        )
        super(Table, self).__init__(
            stream=stream,
            plot=holoviews.Table,
            data=data,
            mpl_opts=mpl_opts,
            bokeh_opts=bokeh_opts,
            **kwargs,
        )

    def get_default_data(self):
        return pandas.DataFrame()

    def opts(self, *args, **kwargs):
        """
        Update the plot parameters. Same as ``holoviews`` ``opts``.

        The default values updates the plot axes independently when being \
        displayed in a :class:`Holomap`.
        """
        plot_opts = self.update_kwargs(**kwargs)
        self.plot = self.plot.opts(holoviews.opts.Table(*args, **plot_opts))


class RGB(StreamingPlot):
    """``holoviews.RGB`` with data streaming capabilities."""

    name = "rgb"

    def __init__(self, data=None, **kwargs):
        """Initialize a :class:`RGB`."""
        super(RGB, self).__init__(stream=Pipe, plot=holoviews.RGB, data=data, **kwargs)

    def get_default_data(self):
        return []

    def opts(
        self,
        xaxis=None,
        yaxis=None,
        *args,
        **kwargs,
    ):
        """
        Update the plot parameters. Same as ``holoviews`` ``opts``.

        The default values updates the plot axes independently when being \
        displayed in a :class:`Holomap`.
        """
        plot_opts = self.update_kwargs(**kwargs)
        self.plot = self.plot.opts(
            holoviews.opts.RGB(xaxis=xaxis, yaxis=yaxis, *args, **plot_opts),
        )


class Curve(StreamingPlot):
    """
    Create a ``holoviews.Curve`` plot that plots steaming data.

    The streaming process is handled using a :class:`Buffer`.
    """

    name = "curve"

    def __init__(
        self,
        buffer_length: int = 10000,
        index: bool = False,
        data=None,
        bokeh_opts: dict = None,
        mpl_opts: dict = None,
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

        def get_stream(data):
            return Buffer(data, length=buffer_length, index=index)

        default_bokeh_opts = {
            "height": 350,
            "width": 400,
            "shared_axes": False,
            "tools": ["hover"],
        }
        default_mpl_opts = {}
        mpl_opts, bokeh_opts = self.update_default_opts(
            default_mpl_opts,
            mpl_opts,
            default_bokeh_opts,
            bokeh_opts,
        )
        self._data_names = data.columns.values if data is not None else data_names
        super(Curve, self).__init__(
            stream=get_stream,
            plot=holoviews.Curve,
            data=data,
            mpl_opts=mpl_opts,
            bokeh_opts=bokeh_opts,
            **kwargs,
        )

    def get_default_data(self):
        return pandas.DataFrame(columns=self._data_names)

    def opts(
        self,
        title="curve",
        xlabel: str = "x",
        ylabel: str = "y",
        framewise: bool = True,
        axiswise: bool = True,
        normalize: bool = True,
        **kwargs,
    ):
        """
        Update the plot parameters. Same as ``holoviews`` ``opts``.

        The default values updates the plot axes independently when being \
        displayed in a :class:`Holomap`.
        """
        kwargs = self.update_kwargs(**kwargs)
        self.plot = self.plot.opts(
            holoviews.opts.Curve(
                title=title,
                xlabel=xlabel,
                ylabel=ylabel,
                framewise=framewise,
                axiswise=axiswise,
                normalize=normalize,
                **kwargs,
            ),
            holoviews.opts.NdOverlay(
                normalize=normalize,
                framewise=framewise,
                axiswise=axiswise,
            ),
        )


class Histogram(StreamingPlot):
    """
    Create a ``holoviews.Histogram`` plot that plots steaming data.

    The streaming process is handled using a :class:`Pipe`.
    """

    name = "histogram"

    def __init__(
        self,
        n_bins: int = 20,
        data=None,
        bokeh_opts: dict = None,
        mpl_opts: dict = None,
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
        default_bokeh_opts = {"shared_axes": False, "tools": ["hover"]}
        default_mpl_opts = {}
        mpl_opts, bokeh_opts = self.update_default_opts(
            default_mpl_opts,
            mpl_opts,
            default_bokeh_opts,
            bokeh_opts,
        )
        super(Histogram, self).__init__(
            stream=Pipe,
            plot=self.plot_histogram,
            data=data,
            mpl_opts=mpl_opts,
            bokeh_opts=bokeh_opts,
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

    def opts(
        self,
        title="",
        xlabel: str = "x",
        ylabel: str = "count",
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
        self.plot = self.plot.opts(
            holoviews.opts.Histogram(
                title=title,
                xlabel=xlabel,
                ylabel=ylabel,
                framewise=framewise,
                axiswise=axiswise,
                normalize=normalize,
                *args,
                **kwargs,
            ),
            holoviews.opts.NdOverlay(
                normalize=normalize,
                framewise=framewise,
                axiswise=axiswise,
            ),
        )

    def preprocess_data(self, data):
        return self.get_plot_data(data)

    def get_plot_data(
        self,
        data: numpy.ndarray,
    ) -> Tuple[
        Tuple[numpy.ndarray, numpy.ndarray],
        Tuple[Union[Scalar, None], Union[Scalar, None]],
    ]:
        """
        Calculate the histogram of the streamed data.

        Args:
            data: Values used to calculate the histogram.

        Returns:
            Tuple containing (values, bins), xlim. xlim is a tuple \
                  containing two typing_.Scalars that represent the limits of the x \
                  axis of the histogram.

        """
        if data is None:
            data = numpy.arange(10)
        data = judo.to_numpy(data)
        data[numpy.isnan(data)] = 0.0
        return numpy.histogram(data, self.n_bins), self.xlim

    def get_default_data(self):
        return self.get_plot_data(None)


class Bivariate(StreamingPlot):
    """
    Create a ``holoviews.Bivariate`` plot that plots steaming data.

    The streaming process is handled using a :class:`Pipe`.
    """

    name = "bivariate"

    def __init__(self, data=None, bokeh_opts=None, mpl_opts=None, **kwargs):
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

        def bivariate(data):
            return holoviews.Bivariate(data, **kwargs)

        default_bokeh_opts = {
            "height": 350,
            "width": 400,
            "tools": ["hover"],
            "shared_axes": False,
        }
        default_mpl_opts = {}
        mpl_opts, bokeh_opts = self.update_default_opts(
            default_mpl_opts,
            mpl_opts,
            default_bokeh_opts,
            bokeh_opts,
        )
        super(Bivariate, self).__init__(
            stream=Pipe,
            plot=bivariate,
            data=data,
            bokeh_opts=bokeh_opts,
            mpl_opts=mpl_opts,
        )

    def opts(
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


class Landscape2D(StreamingPlot):
    """
    Plots the interpolated landscaped of values of a set of points.

    The data is visualized creating a :class:`holoviews.QuadMesh` with a \
    :class:`holoviews.Contours` plot with the original data points displayed as \
    a :class:`holoviews.Scatter`.
    """

    name = "landscape"

    def __init__(
        self,
        n_points: int = 50,
        data=None,
        invert_cmap: bool = False,
        mpl_opts: dict = None,
        bokeh_opts: dict = None,
        plot_scatter: bool = True,
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
        self.n_points = n_points
        self._plot_scatter = plot_scatter
        self.invert_cmap = invert_cmap
        self.xlim = (None, None)
        self.ylim = (None, None)
        default_bokeh_opts = {
            "height": 350,
            "width": 400,
            "tools": ["hover"],
            "shared_axes": False,
        }
        default_mpl_opts = {}

        mpl_opts, bokeh_opts = self.update_default_opts(
            default_mpl_opts,
            mpl_opts,
            default_bokeh_opts,
            bokeh_opts,
        )
        super(Landscape2D, self).__init__(
            stream=Pipe,
            plot=self.plot_landscape,
            data=data,
            mpl_opts=mpl_opts,
            bokeh_opts=bokeh_opts,
            **kwargs,
        )

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
        x, y, xx, yy, z, xlim, ylim = data
        zz = griddata((x, y), z, (xx, yy), method="linear")
        mesh = holoviews.QuadMesh((xx, yy, zz))
        contour = holoviews.operation.contours(mesh, levels=16)

        contour_mesh = mesh * contour
        if self._plot_scatter:
            scatter = holoviews.Scatter((x, y))
            contour_mesh = contour_mesh * scatter
        return contour_mesh.redim(
            x=holoviews.Dimension("x", range=xlim),
            y=holoviews.Dimension("y", range=ylim),
        )

    def preprocess_data(self, data):
        return self.get_plot_data(data)

    def get_plot_data(self, data: Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]):
        """Create the meshgrid needed to interpolate the target data points."""
        x, y, z = (data[:, 0], data[:, 1], data[:, 2]) if isinstance(data, numpy.ndarray) else data
        x, y, z = judo.to_numpy(x), judo.to_numpy(y), judo.to_numpy(z)
        # target grid to interpolate to
        xi = numpy.linspace(x.min(), x.max(), self.n_points)
        yi = numpy.linspace(y.min(), y.max(), self.n_points)
        xx, yy = numpy.meshgrid(xi, yi)
        return x, y, xx, yy, z, self.xlim, self.ylim

    def get_default_data(self):
        X = numpy.random.standard_normal((10, 2))
        z = numpy.random.standard_normal(10)
        data = X[:, 0], X[:, 1], z
        return self.get_plot_data(data)

    def opts(
        self,
        title="Distribution landscape",
        xlabel: str = "x",
        ylabel: str = "y",
        framewise: bool = True,
        axiswise: bool = True,
        normalize: bool = True,
        cmap: str = "default",
        *args,
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
                *args,
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
                *args,
                **contours_kwargs,
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


class PlotCallback(Callback):
    def __init__(self, report_interval: int = 1, **kwargs):
        self.report_interval = report_interval
        super(PlotCallback, self).__init__(**kwargs)

    def after_evolve(self):
        super(PlotCallback, self).before_walkers()
        if self.swarm.epoch % self.report_interval == 0:
            self.send()

    def send(self):
        raise NotImplementedError()

    def panel(self):
        raise NotImplementedError()

    def run_end(self):
        self.send()
