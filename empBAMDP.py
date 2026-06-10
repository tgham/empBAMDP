import marimo

__generated_with = "0.23.6"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # imports
    """)
    return


@app.cell
def _():
    # Imports
    import numpy as np
    from numpy import nan
    import pandas as pd
    import matplotlib.pyplot as plt
    import scipy.stats as stats
    from scipy.spatial.distance import cdist
    from tqdm.auto import tqdm
    import seaborn as sns
    import importlib
    from scipy.stats import bernoulli
    import warnings
    from scipy.special import softmax
    from scipy.spatial.distance import cdist
    import gymnasium as gym
    from gymnasium.envs.registration import register, registry, make, spec
    import pickle
    import copy
    from itertools import product
    import json
    from functools import partial
    from scipy.optimize import Bounds, minimize, differential_evolution
    import multiprocess as mp
    from pybads import BADS
    import os
    import IPython

    import pingouin as pg
    from scipy.special import expit

    # from emp_utils import 
    from emp_runners import enumerate_curves
    from plotter import plot_curves, plot_heatmap


    # warnings.filterwarnings('ignore')


    # magic command not supported in marimo; please file an issue to add support
    # %load_ext autoreload
    # '%autoreload 2' command supported automatically in marimo
    return np, pd, plot_curves, plot_heatmap, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Expts
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## $\ell$ and $\alpha$ sweep
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Load
    """)
    return


@app.cell
def _(pd):
    ## load
    n_arms = 2
    n_outcomes = 4
    n_trials = 6
    termination_arm = True
    # df_curves = pd.read_csv('useful_saves/sweep/{}arms_{}outcomes_{}trials_{}_ksweep.csv'.format(n_arms, n_outcomes, n_trials, ['noTermination', 'Termination'][termination_arm]))
    df_curves = pd.read_csv('useful_saves/sweep/{}arms_{}outcomes_{}trials_{}_unknown_contexts_0.0k.csv'.format(n_arms, n_outcomes, n_trials, ['noTermination', 'Termination'][termination_arm]))
    print('alphas:', df_curves['alpha'].unique())
    df_curves
    return df_curves, n_arms, n_outcomes, n_trials, termination_arm


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Plot curves per history
    """)
    return


@app.cell
def _(df_curves, n_arms, n_trials, plot_curves, termination_arm):
    ## plots
    alpha_to_plot = 0.1
    horizon_to_plot = n_trials
    k_to_plot = 0.00
    plot_curves(df_curves.loc[(df_curves['alpha'] == alpha_to_plot) 
    # & (df_curves['horizon'] == horizon_to_plot)
    & (df_curves['k'] == k_to_plot)
    ], n_arms=n_arms, y='p', eps_tie=1e-06, termination_arm=termination_arm, info_seeker=True, ncols=7)  #    df_tip=df_tip
    return


@app.cell
def _(df_curves):
    df_curves.loc[df_curves['alpha'] == 'unknown']
    return


@app.cell
def _(df_curves, n_arms, plot_curves, termination_arm):
    ## plots
    alpha_to_plot = 0.1
    plot_curves(df_curves.loc[(df_curves['alpha'] == alpha_to_plot) 
    ], n_arms=n_arms, y='p', eps_tie=1e-06, termination_arm=termination_arm, info_seeker=True, ncols=7)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Plot heatmaps per history
    """)
    return


@app.cell
def _(df_curves, plot_heatmap):
    # metric = 'p_terminate'
    metric = 'p_diff'

    # y_axis = 'k'
    # fixed_alpha = 0.0125
    # fixed_k = None

    y_axis = 'alpha'
    fixed_alpha = None
    fixed_k = 0.01


    plot_heatmap(df_curves, panel_size=(3, 3), plot_info_seeker=False, shared_colorbar=True, max_n_cols=7,
    y_axis=y_axis, metric=metric, fixed_alpha=fixed_alpha, fixed_k=fixed_k
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cost analysis
    """)
    return


@app.cell
def _(df_curves, n_arms, n_outcomes, n_trials, termination_arm):
    ## find history with max emp for each alpha and ell
    max_emps = df_curves.groupby(['alpha', 'ell']).apply(lambda x: x.loc[x['current_emp'].idxmax()])[[
        # 't',
        'ell','alpha',
        'current_emp','history_str',]]

    ## save csv
    max_emps.to_csv('useful_saves/sweep/{}arms_{}outcomes_{}trials_{}_max_emps.csv'.format(n_arms, n_outcomes, n_trials, ['noTermination', 'Termination'][termination_arm]), index=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Misc
    """)
    return


@app.cell
def _(np, plt):
    ells_3 = [0.01, 0.1, 0.5, 1.0, 10, 100]
    x = np.linspace(1e-16, 1, 1000)
    plt.figure(figsize=(8, 6))
    for ell_5 in ells_3:
        plt.plot(x, x ** ell_5, label=f'ell={ell_5}')
    plt.xlabel('max_p')
    plt.ylabel('emp')
    plt.legend()
    return


if __name__ == "__main__":
    app.run()
