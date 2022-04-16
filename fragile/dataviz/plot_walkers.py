import panel

from fragile.core.fractalai import relativize
from fragile.dataviz.stream_plots import Histogram, PlotCallback


class PlotWalkers(PlotCallback):
    name = "plot_walkers"

    def __init__(
        self,
        plot_scores: bool = True,
        plot_diversities: bool = True,
        plot_virtual_rewards: bool = True,
        plot_norm_scores: bool = True,
        plot_norm_diversities: bool = True,
        **kwargs,
    ):
        super(PlotWalkers, self).__init__(**kwargs)
        self._plot_scores = plot_scores
        self._plot_diversities = plot_diversities
        self._plot_virtual_rewards = plot_virtual_rewards
        self._plot_norm_scores = plot_norm_scores
        self._plot_norm_diversities = plot_norm_diversities
        self.virtual_reward_sp = None
        self.diversities_sp = None
        self.scores_sp = None
        self.norm_scores_sp = None
        self.norm_diversities_sp = None
        if self._plot_scores:
            self.scores_sp = Histogram(title="Scores")
        if self._plot_virtual_rewards:
            self.virtual_reward_sp = Histogram(title="Virtual Rewards")
        if self._plot_diversities:
            self.diversities_sp = Histogram(title="Diversities")
        if self._plot_norm_scores:
            self.norm_scores_sp = Histogram(title="Normalized Scores")
        if self._plot_norm_diversities:
            self.norm_diversities_sp = Histogram(title="Normalized Diversities")

    def send(self):
        if self._plot_scores:
            self.scores_sp.send(self.get("scores"))
        if self._plot_diversities:
            self.diversities_sp.send(self.get("diversities"))
        if self._plot_virtual_rewards:
            self.virtual_reward_sp.send(self.get("virtual_rewards"))
        if self._plot_norm_scores:
            scores = self.get("scores")
            scores = -1.0 * scores if self.swarm.minimize else scores
            self.norm_scores_sp.send(relativize(scores))
        if self._plot_norm_diversities:
            self.norm_diversities_sp.send(relativize(self.get("diversities")))

    def panel(self):
        histograms = []
        if self._plot_scores:
            histograms.append(self.scores_sp.plot)
        if self._plot_diversities:
            histograms.append(self.diversities_sp.plot)
        if self._plot_virtual_rewards:
            histograms.append(self.virtual_reward_sp.plot)
        norm_histograms = []
        if self._plot_norm_scores:
            norm_histograms.append(self.norm_scores_sp.plot)
        if self._plot_norm_diversities:
            norm_histograms.append(self.norm_diversities_sp.plot)

        return panel.Column(panel.Row(*histograms), panel.Row(*norm_histograms))
