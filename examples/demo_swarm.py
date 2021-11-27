#!/usr/bin/env python
# coding: utf-8
from functools import partial
from itertools import product

import holoviews as hv
import numpy as np
import pandas as pd
import ray
from scipy import stats
from sklearn.preprocessing import KBinsDiscretizer

from fragile.core import NormalContinuous
from fragile.core.tree import HistoryTree
from fragile.optimize import FunctionMapper
from fragile.optimize.benchmarks import ALL_BENCHMARKS


def tree_callable():
    return HistoryTree(names=["observs", "rewards"], prune=False)


def sample_one_landscape(landscape, n_particles=1000, scale_pct=10, max_epochs=1000):
    env = landscape(2)

    def gaussian_model(env, scale_pct=scale_pct):
        # Gaussian of mean 0 and std of 5% of the search space width, adapted to the environment bounds
        scale = (env.bounds.high[0] - env.bounds.low[0]) * (scale_pct / 100.0)
        return NormalContinuous(scale=scale, loc=0.0, bounds=env.bounds)

    swarm = FunctionMapper(
        env=env,
        model=gaussian_model,
        n_walkers=n_particles,
        max_epochs=max_epochs,
        tree=tree_callable,
        use_notebook_widget=False,
        show_pbar=False,
    )
    swarm.run(report_interval=1000000)
    df = extract_data_from_swarm(swarm)
    exp_id = (env.__class__.__name__, n_particles, scale_pct, max_epochs)
    return exp_id, df


@ray.remote
def sample_landscape(landscape, n_particles=50, scale_pct=10, max_epochs=10000):
    return sample_one_landscape(
        landscape, n_particles=n_particles, max_epochs=max_epochs, scale_pct=scale_pct
    )


def extract_data_from_swarm(swarm):
    # Extract coordinates and energy from swarm data structures
    obs = np.vstack([swarm.tree.data.nodes[x]["observs"] for x in swarm.tree.data.nodes])
    z = np.array([swarm.tree.data.nodes[x]["rewards"] for x in swarm.tree.data.nodes])
    x, y = obs[:, 0], obs[:, 1]
    df = pd.DataFrame(columns=["x", "y", "z"], data=np.stack([x, y, -z], 1))
    return df


def count_bins(df, n_bins=10):
    # Discretize state space so we can count how many points lie in each bin
    est = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="uniform")
    bins = est.fit_transform(df[["x", "y"]].values)
    df["bin_x"] = bins[:, 0]
    df["bin_y"] = bins[:, 1]
    energy_df = df.groupby(["bin_x", "bin_y"])["z"].mean().reset_index()
    counts = df.groupby(["bin_x", "bin_y"])["z"].count().reset_index()
    # Normalize counts and energy so we can compare them
    pos_energy = energy_df["z"] - energy_df["z"].min() + 1e-30  # np.abs(energy_df["z"]).min()
    reg_energy = np.exp(pos_energy / pos_energy.sum())
    log_probs = np.log(counts["z"] / counts["z"].sum()).values
    return log_probs, reg_energy


def fit_linreg(xs, ys):
    slope, intercep, rval, pval, _ = stats.linregress(xs, ys)
    return slope, intercep, rval, pval


def plot_data(xs, ys, slope, intercep, rval, pval, name):
    linreg = hv.Slope(slope, intercep).opts(color="red")
    scatter = hv.Scatter((xs, ys)).opts(xlabel="log probability", ylabel="energy")
    reg_plot = (scatter * linreg).relabel("%s | r2: %.3f | pval: %.2e" % (name, rval, pval))
    return reg_plot.opts(fontsize={"title": 8}, framewise=True, xlim=(None, None))


def run_simulations(samples_run, stds):
    results = ray.get(
        [
            sample_landscape.remote(
                function, scale_pct=std, n_particles=particles, max_epochs=epochs
            )
            for function, (particles, epochs), std in product(ALL_BENCHMARKS, samples_run, stds)
        ]
    )
    samples = {k: v for k, v in results}
    return samples


def plot_df(df, name, n_bins=10):
    log_probs, energy = count_bins(df, n_bins=n_bins)
    slope, intercep, rval, pval = fit_linreg(log_probs, energy)
    return plot_data(log_probs, energy, slope, intercep, rval, pval, name)


def selected_run(n, e, s, n_parts, scale, epochs):
    return n == n_parts and s == scale and e == epochs


def regression_dmap(samples, samples_run, stds, n_bins=20, run=0, temperature=0):
    plots = [
        plot_df(v, k, n_bins)
        for (k, n, e, s), v in samples.items()
        if selected_run(n, e, s, samples_run[run][0], samples_run[run][1], stds[temperature])
    ]
    return hv.Layout(plots).opts(shared_axes=False).cols(3)


def analyze_data(samples, samples_run, stds):
    regression = partial(regression_dmap, samples_run=samples_run, stds=stds, samples=samples)
    dmap = hv.DynamicMap(regression, kdims=["n_bins", "run", "temperature"]).redim(
        n_bins=dict(range=(5, 50), step=1, default=25),
        run=dict(range=(0, len(samples_run) - 1), step=1),
        temperature=dict(range=(0, len(stds) - 1), step=1),
    )
    return dmap
