from itertools import product
import os
import tempfile
import warnings

import holoviews
import pytest

from fragile.dataviz.plot_saver import PlotSaver
from tests import IN_CI
from tests.dataviz.test_swarm_viz import swarm_dict, swarm_names, backends


@pytest.fixture(params=tuple(product(swarm_names, backends)), scope="class")
def plot_saver(request):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        swarm_name, backend = request.param
        holoviews.extension(backend)
        swarm_viz = swarm_dict.get(swarm_name)()
        swarm_viz.stream_interval = 1
        plot_saver = PlotSaver(swarm_viz, output_path="Miau_db", fmt="png")
        return plot_saver


class TestPlotSaver:
    def test_get_file_name(self, plot_saver):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            filename = plot_saver._get_file_name()
            assert isinstance(filename, str)
            name, extension = filename.split(".")
            assert extension == plot_saver._fmt
            class_name, epoch = name.split("_")
            assert class_name == plot_saver.unwrapped.__class__.__name__.lower()
            assert len(epoch) == 5
            assert int(epoch) == plot_saver.epoch

    @pytest.mark.skipif(IN_CI, reason="Need geckodriver. Skipping until fixed.")
    def test_run_step(self, plot_saver):
        with tempfile.TemporaryDirectory() as tmpdirname:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                plot_saver.output_path = tmpdirname
                plot_saver.reset()
                plot_saver.run_step()
                file_name = plot_saver._get_file_name()
                assert file_name in list(os.listdir(tmpdirname))
