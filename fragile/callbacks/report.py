import logging
from typing import Optional

import einops
from judo.functions.notebook import running_in_ipython
import numpy
from tqdm.autonotebook import tqdm

from fragile.core.api_classes import Callback


def statistics_from_array(x: numpy.ndarray):
    """Return the (mean, std, max, min) of an array."""
    try:
        return (
            einops.asnumpy(x).mean(),
            einops.asnumpy(x).std(),
            einops.asnumpy(x).max(),
            einops.asnumpy(x).min(),
        )
    except (AttributeError, TypeError, ValueError):
        return numpy.nan, numpy.nan, numpy.nan, numpy.nan


class Report(Callback):
    name = "report"
    _log = logging.getLogger("Swarm")

    def __init__(
        self,
        report_interval: Optional[int] = None,
        progress_bar: bool = True,
        notebook_widget: bool = True,
        **kwargs,
    ):
        self._use_progress_bar = progress_bar
        self._use_notebook_widget = notebook_widget
        self._ipython_mode = None
        self._notebook_container = None
        self.report_interval = report_interval
        self.tqdm = None
        self.html_output = None
        super(Report, self).__init__(**kwargs)

    def before_reset(self):
        self._ipython_mode = running_in_ipython()
        if self._use_progress_bar:
            self.tqdm = tqdm(total=self.swarm.max_epochs)
        if self._use_notebook_widget:
            self.setup_notebook_container()

    def after_evolve(self):
        if self._use_progress_bar:
            self.tqdm.update(1)
        if self.report_interval is not None and self.swarm.epoch % self.report_interval == 0:
            self.report_progress()

    def run_end(self):
        if self._use_progress_bar:
            self.tqdm.close()
        if self.report_interval is not None:
            self.report_progress()

    def report_progress(self):
        """Report information of the current run."""
        if self._ipython_mode and self._use_notebook_widget:
            html = self.swarm.to_html()
            self._notebook_container.value = "%s" % html
        elif not self._ipython_mode:
            self._log.info(repr(self))

    def setup_notebook_container(self):
        """Display the display widgets if the Swarm is running in an IPython kernel."""
        if self._ipython_mode and self._use_notebook_widget:
            from IPython.core.display import display, HTML as cell_html
            from ipywidgets import HTML

            # Set font weight of tqdm progressbar
            display(cell_html("<style> .widget-label {font-weight: bold !important;} </style>"))
            self._notebook_container = HTML()
            display(self._notebook_container)
