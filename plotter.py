import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory

def _split_counts_str(h):
    if h in ('', 'init'):
        return (h, '')
    parts = h.split('-', 1)
    return (parts[0], parts[1] if len(parts) > 1 else '')


def _add_group_brackets(ax, hs, line_y=-0.5, label_y=-0.55):
    if not hs:
        return
    groups = []
    cur_top, _ = _split_counts_str(hs[0])
    cur_start = 0
    for i in range(1, len(hs)):
        top_i, _ = _split_counts_str(hs[i])
        if top_i != cur_top:
            groups.append((cur_top, cur_start, i - 1))
            cur_top, cur_start = top_i, i
    groups.append((cur_top, cur_start, len(hs) - 1))
    trans = ax.get_xaxis_transform()
    for top, s, e in groups:
        ax.plot([s - 0.35, e + 0.35], [line_y, line_y],
                color='k', lw=1, transform=trans, clip_on=False)
        ax.text((s + e) / 2, label_y, top + '-',
                ha='center', va='top', transform=trans, rotation=45,
                fontsize=9, fontweight='bold')


def plot_history_panels(
    df,
    ells,
    n_arms,
    canonicalize=True,
    n_plot_trials=None,
    disagree_only=True,
    group_brackets=True,
    skip_t0=True,
    legend=False,
    ell_colors=None,
    figsize=(13, 12),
    suptitle=None,
):
    """Plot Q, P(choose arm), P(repeat or arm 0), and Δemp / P(terminate)
    panels across ells. Rows = metric, columns = history length t.

    Expects df to have been preprocessed by emp_utils.filter_histories so
    it has 'history_canon_counts_str' (if canonicalize=True) and
    'disagree' columns. Row filtering (t, disagreement) and per-panel
    layout (hist_by_t, x_local, axis sharing, brackets) all happen here;
    filter_histories itself only adds derived columns.
    """

    max_trials = int(df['t'].max()) + 1
    n_plot_trials = np.min([n_plot_trials, max_trials]) if n_plot_trials is not None else max_trials
    x_col = 'history_canon_counts_str' if canonicalize else 'history_counts_str'
    is_counts = x_col.endswith('counts_str')

    if ell_colors is None:
        default_cycle = ['tab:blue', 'tab:orange', 'tab:red',
                         'tab:green', 'tab:purple', 'tab:brown']
        ell_colors = {e: default_cycle[i % len(default_cycle)]
                      for i, e in enumerate(ells)}

    sub = df.copy()
    if n_plot_trials is not None:
        sub = sub[sub['t'] < int(n_plot_trials)]
    sub = sub[sub[x_col].astype(str) != '']
    if disagree_only:
        sub = sub[sub['disagree']]
    sub = sub.drop_duplicates(subset=['t', x_col, 'ell'])

    hist_by_t = {}
    x_local = {}
    for t in sorted(sub['t'].unique()):
        hs = (
            sub[sub['t'] == t][[x_col]]
            .drop_duplicates()
            .sort_values(x_col)[x_col]
            .tolist()
        )
        hist_by_t[t] = hs
        for i, h in enumerate(hs):
            x_local[(t, h)] = i
    sub = sub.copy()
    sub['x_local'] = sub.apply(lambda r: x_local[(r['t'], r[x_col])], axis=1)

    max_t = int(n_plot_trials) if n_plot_trials is not None else (int(sub['t'].max()) + 1)
    ts = list(range(1 if skip_t0 else 0, max_t))
    n_cols = len(ts)
    width_ratios = [max(1, len(hist_by_t.get(t, []))) for t in ts]

    n_rows = 4
    ## sharex='col' keeps the axes box and tick positions identical across
    ## rows in each column; otherwise tight_layout pads the bottom row
    ## differently for the rotated tick labels.
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=figsize,
        sharey='row',
        sharex='col',
        gridspec_kw={'width_ratios': width_ratios},
    )
    if n_cols == 1:
        axes = axes.reshape(n_rows, 1)

    styles = ['o-', 's--', 'd:']
    has_terminate = 'p_terminate' in df.columns

    for j, t in enumerate(ts):
        sub_t = sub[sub['t'] == t]
        hs = hist_by_t.get(t, [])
        xs_t = list(range(len(hs)))

        ## (1) Q-values
        ax = axes[0, j]
        for ell in ells:
            s = sub_t[sub_t['ell'] == ell].sort_values('x_local')
            c = ell_colors[ell]
            mfcs = ['white', str(c), str(c)]
            for a in range(n_arms):
                if canonicalize:
                    y_axis = f'canon_Q_{a}'
                else:
                    y_axis = f'Q_{a}'
                ax.plot(s['x_local'], s[y_axis], styles[a], color=c, alpha=0.85,
                        label=f'ell={ell}, arm {a}', markerfacecolor=mfcs[a])
            if has_terminate:
                ax.plot(s['x_local'], s['Q_terminate'], 'X-', color=c, alpha=0.85,
                        label=f'ell={ell}, terminate', markerfacecolor='none')
        ax.set_title(f't = {t}', y=1.02 + len(ells) * 0.07 + 0.02)
        if j == 0:
            ax.set_ylabel('Q')
        ax.grid(alpha=0.2)
        ax.set_xticks(xs_t)
        ax.tick_params(labelbottom=False, labelleft=True)

        best_a_col = 'canon_best_a' if canonicalize else 'best_a'
        if best_a_col in sub_t.columns:
            ## recompute disagreement scoped to the plotted ells, so that
            ## a (t, history) is only "disagree" if the ells we're actually
            ## drawing differ on best_a here. (sub_t['disagree'] reflects
            ## the full ell set in df and is trivially True everywhere when
            ## disagree_only=True pre-filtered the frame.)
            sub_t_plot = sub_t[sub_t['ell'].isin(ells)]
            disagree_per_x = sub_t_plot.groupby('x_local')[best_a_col].nunique() > 1
            trans_top = blended_transform_factory(ax.transData, ax.transAxes)
            for xi, h in zip(xs_t, hs):
                disagree = bool(disagree_per_x.get(xi, False))
                for ei, ell in enumerate(ells):
                    row = sub_t_plot[(sub_t_plot['ell'] == ell) & (sub_t_plot['x_local'] == xi)]
                    if row.empty:
                        continue
                    best_a = int(row[best_a_col].iloc[0])
                    label = 'T' if best_a == n_arms else str(best_a)
                    ax.text(xi, 1.02 + ei * 0.07, label,
                            color=ell_colors[ell], ha='center', va='bottom',
                            transform=trans_top, clip_on=False,
                            fontsize=10, fontweight='bold' if disagree else 'normal')

        if legend and j == n_cols - 1:
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=3)

        ## (2) Choice probabilities P(arm a)
        ax = axes[1, j]
        for ell in ells:
            s = sub_t[sub_t['ell'] == ell].sort_values('x_local')
            c = ell_colors[ell]
            mfcs = ['white', str(c), str(c)]
            for a in range(n_arms):
                if canonicalize:
                    y_axis = f'canon_p_{a}'
                else:
                    y_axis = f'p_{a}'
                ax.plot(s['x_local'], s[y_axis], styles[a], color=c, alpha=0.85,
                        label=f'ell={ell}, arm {a}', markerfacecolor=mfcs[a])
            if has_terminate:
                ax.plot(s['x_local'], s['p_terminate'], 'X-', color=c, alpha=0.85,
                        label=f'ell={ell}, terminate', markerfacecolor='none')
        chance = 1 / (n_arms + (1 if has_terminate else 0))
        ax.axhline(chance, color='k', linewidth=2, linestyle='--', alpha=0.5)
        if j == 0:
            ax.set_ylabel('P(choose arm)')
        ax.grid(alpha=0.2)
        ax.set_xticks(xs_t)
        ax.tick_params(labelbottom=False, labelleft=True)

        ## (3) P(repeat) (or P(choose arm 0) when grouped by counts)
        if is_counts:
            if canonicalize:
                y_ax = 'canon_p_0'
            else:
                y_ax = 'p_0'
            y_label = 'P(choose arm 0)'
        else:
            y_ax = 'p_repeat'
            y_label = 'P(repeat previous action)'
        ax = axes[2, j]
        if t == 0:
            ax.text(0.5, 0.5, 'no previous action',
                    ha='center', va='center', transform=ax.transAxes,
                    color='grey', fontsize=10, rotation=90)
            ax.tick_params(bottom=False, labelbottom=False)
        else:
            for ell in ells:
                s = sub_t[sub_t['ell'] == ell].sort_values('x_local')
                ax.plot(s['x_local'], s[y_ax], 'o-',
                        color=ell_colors[ell], alpha=0.85, label=f'ell={ell}')
            chance = 1 / (n_arms + (1 if has_terminate else 0))
            ax.axhline(chance, color='k', linewidth=2, linestyle='--', alpha=0.5)
            ax.set_xticks(xs_t)
            ax.tick_params(labelbottom=False)
        ax.grid(alpha=0.2)
        ax.tick_params(labelleft=True)
        if j == 0:
            ax.set_ylabel(y_label)

        ## (4) 1-step expected empowerment gain OR P(terminate)
        ax = axes[-1, j]
        if has_terminate:
            y_baseline = 1 / (n_arms + 1)
            y_label = 'P(terminate)'
        else:
            y_baseline = 0
            y_label = r'$\Delta$ Emp (one-step)'
        for ell in ells:
            s = sub_t[sub_t['ell'] == ell].sort_values('x_local')
            c = ell_colors[ell]
            if has_terminate:
                ax.plot(s['x_local'], s['p_terminate'], 'o-', color=c, alpha=0.85,
                        label=f'ell={ell}')
            else:
                if canonicalize:
                    y_axis_0 = f'canon_delta_emp_0'
                    y_axis_1 = f'canon_delta_emp_1'
                else:
                    y_axis_0 = f'delta_emp_0'
                    y_axis_1 = f'delta_emp_1'
                ax.plot(s['x_local'], s[y_axis_0], 'o-', color=c, alpha=0.85,
                        label=f'ell={ell}, arm 0')
                ax.plot(s['x_local'], s[y_axis_1], 's--', color=c, alpha=0.85,
                        label=f'ell={ell}, arm 1', markerfacecolor='none')
        ax.set_xticks(xs_t)
        if is_counts:
            leaf_labels = [_split_counts_str(h)[1] for h in hs]
            ax.set_xticklabels(leaf_labels, rotation=70, ha='right', fontsize=9)
            if group_brackets:
                _add_group_brackets(ax, hs)
        else:
            ax.set_xticklabels(hs, rotation=70, ha='right', fontsize=9)
        ax.grid(alpha=0.2)
        ax.tick_params(labelleft=True)
        ax.axhline(y_baseline, color='k', linewidth=2, linestyle='--', alpha=0.5)
        if j == 0:
            ax.set_ylabel(y_label)

    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=11)
    plt.tight_layout()
    return fig, axes