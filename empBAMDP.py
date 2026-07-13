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
    # import addcopyfighandler

    # from emp_utils import 
    from emp_runners import enumerate_curves
    from plotter import plot_curves, plot_heatmap


    # warnings.filterwarnings('ignore')
    return np, pd, plot_curves, plot_heatmap, plt, sns, softmax


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
    df_curves = pd.read_csv('useful_saves/sweep/{}arms_{}outcomes_{}trials_{}.csv'.format(n_arms, n_outcomes, n_trials, ['noTermination', 'Termination'][termination_arm]))
    # df_curves = pd.read_csv('useful_saves/sweep/{}arms_{}outcomes_{}trials_{}_ksweep.csv'.format(n_arms, n_outcomes, n_trials, ['noTermination', 'Termination'][termination_arm]))
    # df_curves = pd.read_csv('useful_saves/sweep/{}arms_{}outcomes_{}trials_{}_unknown_contexts_0.0k.csv'.format(n_arms, n_outcomes, n_trials, ['noTermination', 'Termination'][termination_arm]))

    print('alphas:', df_curves['alpha'].unique())
    df_curves
    return df_curves, n_arms, n_trials, termination_arm


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
    & (df_curves['horizon'] == horizon_to_plot)
    # & (df_curves['k'] == k_to_plot)
    ], n_arms=n_arms, y='p', eps_tie=1e-06, termination_arm=termination_arm, info_seeker=True, ncols=7)  #    df_tip=df_tip
    return alpha_to_plot, horizon_to_plot


@app.cell
def _(df_curves):
    df_curves.loc[df_curves['alpha'] == 'unknown']
    return


@app.cell
def _(
    alpha_to_plot,
    df_curves,
    horizon_to_plot,
    n_arms,
    plot_curves,
    termination_arm,
):
    ## plots
    # alpha_to_plot = 0.1
    # alpha_to_plot = 'unknown'
    # horizon_to_plot = 1
    plot_curves(df_curves.loc[(df_curves['alpha'] == alpha_to_plot) & (df_curves['horizon'] == horizon_to_plot)
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
    ## Info-seeker
    """)
    return


@app.cell
def _(df_curves):
    ## find cases where info-seeker prefers most-sampled arm
    df_most = df_curves.loc[df_curves['info_p_0']>df_curves['info_p_1']]
    print(df_curves.loc[df_curves['info_p_0']>df_curves['info_p_1'],'history_str'].unique())
    print(df_curves.loc[df_curves['info_p_0']>df_curves['info_p_1'],'info_p_0'].max())
    df_most
    # df_most.loc[df_most['alpha']==0.1]
    df_most.loc[df_most['history_str']=='a0o0:2-a0o1:1-a0o2:1-a1o1:1']
    return


@app.cell
def _(np):
    def diri_var(a_z, counts):
        a = a_z + counts
        a0 = a.sum(axis=1, keepdims=True)
        var = a * (a0 - a) / (a0 ** 2 * (a0 + 1))
        mean = a / a0
        return mean, var

    a_z = 0.0125
    counts = np.array([[2, 1, 2, 0], [0,1,0,0]])
    mean, var = diri_var(a_z, counts)
    print('mean',mean)
    print('var',var)
    print('var sum', var.sum(1))
    print('var sum', var.sum())

    ## for each action, let's calculate the expected sum MSE that could arise from any of the four outcomes
    exMSE = []
    for a in range(2):
        print()
        exMSE.append(0)
        init_mean, init_var = diri_var(a_z, counts)
        for outc in range(4):
            counts_tmp = counts.copy()
            counts_tmp[a][outc] += 1
            mean, var = diri_var(a_z, counts_tmp)
            sumMSE = var.sum()
            p = init_mean[a][outc]
            exMSE[a] += p*sumMSE
            print(p, sumMSE)
    print()
    print(exMSE)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cost analysis
    """)
    return


@app.cell
def _(df_curves):
    ## find history with max emp for each alpha and ell
    max_emps = df_curves.groupby(['alpha', 'ell']).apply(lambda x: x.loc[x['current_emp'].idxmax()])[[
        # 't',
        'ell','alpha',
        'current_emp','history_str',]]
    max_emps.loc[max_emps['ell']==10.069386314760273]
    # max_emps


    ## save csv
    # max_emps.to_csv('useful_saves/sweep/{}arms_{}outcomes_{}trials_{}_max_emps.csv'.format(n_arms, n_outcomes, n_trials, ['noTermination', 'Termination'][termination_arm]), index=False)
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Expt design
    """)
    return


@app.cell
def _(np):
    ## runtime checks: for a given set of task params, how long does the computation of a single history take?
    from emp_utils import EmpowermentAgent
    na = 4
    nk = 8
    nt = 6
    alph = 0.1
    contexts = [(float(alph), 1.0)]
    agent = EmpowermentAgent(n_arms=na, n_outcomes=nk, termination_arm=True, contexts=contexts, ell = 1)
    init_counts = np.zeros((na, nk))
    agent.bellman_Q(init_counts, nt-1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Define rooms
    """)
    return


@app.cell
def _(np, plt, sns, softmax):
    ## import rooms
    from rooms import rooms_2a_4k, rooms_4a_4k, rooms_3a_6k

    ## convert these to transition matrices
    room_Ts = []
    # for room in rooms_2a_4k:
    for room in rooms_4a_4k:
    # for room in rooms_3a_6k:
        T_tmp = np.array(room) / np.array(room).sum(axis=1, keepdims=True)
        room_Ts.append(T_tmp)

    ## calculate empowerment for each room
    def emp(T, ell=0.1):
        return np.sum(np.max(T, axis=0) ** ell)
    ells = [0.1, 1.0, 10.0]
    prefs = np.zeros((len(ells), len(room_Ts)))
    for e, ell in enumerate(ells):
        Vs = []
        for i, T in enumerate(room_Ts):
            V = emp(T, ell=ell)
            Vs.append(V)
        prefs[e] = softmax(Vs)

    #heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(prefs, annot=True, fmt=".2f", cmap="YlGnBu", xticklabels=[f'Room {i+1}' for i in range(len(room_Ts))], yticklabels=[f'ell={ell}' for ell in ells])
    plt.show()


    return (rooms_2a_4k,)


@app.cell
def _(np, plt, rooms_2a_4k):
    ## plot each room as a 3x3 grid. rooms_2a_4k contains 6 rooms, where each room contains 2 lists of 4 binary values, i.e. 2 actions. these binary values refer to which of the cardinal directions (up, down, left, right) are reachable from that action. For example, if the first list is [1, 0, 1, 0], this means that from the central cell, the first action can take the agent up or left but not down or right
    roi = rooms_2a_4k
    # roi = rooms_3a_6k
    fig, axs = plt.subplots(len(roi[0]), len(roi), figsize=(len(roi)*3, len(roi[0])*3))
    for ri, r in enumerate(roi):
        for a in range(len(r)):
            action = r[a]
            action_grid = np.zeros((3, 3))
            for d in range(len(action)):
                if action[d] == 1:
                    if len(roi[0]) == 2:  # 2a_4k, i.e. cardinal directions
                        if d == 0:  # up
                            action_grid[0, 1] = 1
                        elif d == 1:  # down
                            action_grid[2, 1] = 1
                        elif d == 2:  # left
                            action_grid[1, 0] = 1
                        elif d == 3:  # right
                            action_grid[1, 2] = 1
                    else:  # 3a_6k, i.e. two columns
                        if d == 0:  
                            action_grid[0, 0] = 1
                        elif d == 1:  
                            action_grid[1, 0] = 1
                        elif d == 2:  
                            action_grid[2, 0] = 1
                        elif d == 3:  
                            action_grid[0, 2] = 1
                        elif d == 4:  
                            action_grid[1, 2] = 1
                        elif d == 5:  
                            action_grid[2, 2] = 1
            axs[a, ri].imshow(action_grid, 
            ## red cmap
            cmap='Reds',
             vmin=0, vmax=1)
            axs[a, ri].set_xticks([])
            axs[a, ri].set_yticks([])
        axs[0, ri].set_title(f'Room {ri+1}')
        # plt.suptitle(f'Room {ri+1}')
    plt.show()
    return


@app.cell
def _(np, plt):
    from scipy.stats import dirichlet
    alpha = np.ones(4) * 0.2
    true_T = np.random.dirichlet(alpha=alpha, size=2)
    trials = 6
    counts = np.zeros((2, 4))
    cmaps = ['Reds', 'Blues']
    for t in range(trials):
        ac = np.random.choice(2)
        outcome = np.random.choice(4, p=true_T[ac])
        counts[ac][outcome] += 1
        post_mean = (counts + alpha) / (counts.sum(axis=1, keepdims=True) + alpha.sum())

        ## plot the outcome observed
        grid = np.zeros((3,3)) + np.nan
        if outcome == 0:
            grid[0,1] = 1
        elif outcome == 1:
            grid[2,1] = 1
        elif outcome == 2:
            grid[1,0] = 1
        elif outcome == 3:
            grid[1,2] = 1
        plt.figure(figsize=(3, 3))
        plt.imshow(grid,
        ## cmap should be from white to red
        cmap = 'Greys',
        vmin=0, vmax=1)
        plt.xticks([])
        plt.yticks([])
        plt.show()

        ## plot heatmap for the two actions
        for ac in range(2):
            grid = np.zeros((3,3)) + np.nan
            ## cardinal directions
            grid[0,1] = post_mean[ac][0]
            grid[2,1] = post_mean[ac][1]
            grid[1,0] = post_mean[ac][2]
            grid[1,2] = post_mean[ac][3]

            plt.subplot(1, 2, ac+1)
            plt.imshow(grid, 
            ## cmap should be from white to red
            cmap = cmaps[ac],
            vmin=0, vmax=1)
            plt.title(f'Action {ac+1}')
            plt.xticks([])
            plt.yticks([])
        plt.suptitle(f'Trial {t+1}')
        plt.tight_layout()
        plt.show()

    return


if __name__ == "__main__":
    app.run()
