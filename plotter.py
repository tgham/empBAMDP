import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory, offset_copy
from matplotlib.ticker import LogLocator
from matplotlib.patches import Rectangle, Circle
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerBase
from scipy.special import softmax

def _split_counts_str(h):
    if h in ('', 'init'):
        return (h, '')
    parts = h.split('-', 1)
    return (parts[0], parts[1] if len(parts) > 1 else '')


def _add_group_brackets(ax, hs, line_offset_pts=None, label_offset_pts=None,
                        char_drop_pts=5.0, padding_pts=22):
    """Draw a labelled bracket beneath each contiguous run of histories
    sharing the same first count-pair (their "top"). Anchored at a fixed
    pixel offset below the axis rather than in axes-fraction, so the
    position is stable across rows of different heights.

    When line_offset_pts is None, each group's offset is sized to the
    longest leaf label in that group (drop = max_chars * char_drop_pts +
    padding_pts), so short-labelled groups sit close to the axis while
    long-labelled groups get pushed further down.
    """
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
    base = ax.get_xaxis_transform()
    ## ax.plot updates data limits even via a blended transform, which breaks
    ## log-scale axes when y is negative. Save/restore ylim to neutralise.
    ylim = ax.get_ylim()
    for top, s, e in groups:
        if line_offset_pts is None:
            grp_max = max(len(_split_counts_str(hs[k])[1])
                          for k in range(s, e + 1))
            drop = grp_max * char_drop_pts + padding_pts
            l_off = -drop
            lbl_off = -(drop + 3)
        else:
            l_off = line_offset_pts
            lbl_off = (label_offset_pts if label_offset_pts is not None
                       else line_offset_pts - 3)
        line_trans = offset_copy(base, fig=ax.figure, y=l_off, units='points')
        label_trans = offset_copy(base, fig=ax.figure, y=lbl_off, units='points')
        ax.plot([s - 0.35, e + 0.35], [0, 0],
                color='k', lw=1, transform=line_trans, clip_on=False)
        ax.text((s + e) / 2, 0, top + '-',
                ha='center', va='top', transform=label_trans, rotation=45,
                fontsize=9, fontweight='bold')
    ax.set_ylim(ylim)


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


def plot_tipping_points(
    tip_df,
    n_arms,
    n_plot_trials=None,
    skip_t0=True,
    group_brackets=True,
    arm_colors=None,
    log_scale=True,
    y_range=None,
    figsize=(14, 9),
    suptitle=None,
    bar_width=0.85,
    legend=True,
    last_trial_own_row=True,
):
    """Plot per-(canonical history, arm) preferred-ell intervals from
    `enumerate_tipping_intervals`. Columns are trial indices t; within each
    column the x-axis is the canonical history (lex-sorted, matching
    `plot_history_panels`), and the y-axis is ell. Each row in `tip_df` is
    drawn as a coloured vertical bar from `ell_lo` to `ell_hi`, coloured by
    `arm`. When `last_trial_own_row=True` the largest-t panel gets its own
    full-width row beneath the rest so its histories aren't squeezed.
    """

    df = tip_df.copy()
    df = df[df['history_str'].astype(str) != '']

    max_trials = int(df['t'].max()) + 1
    n_plot_trials = min(n_plot_trials, max_trials) if n_plot_trials is not None else max_trials

    ts_all = sorted(int(t) for t in df['t'].unique() if int(t) < n_plot_trials)
    if skip_t0:
        ts_all = [t for t in ts_all if t > 0]

    hist_by_t = {}
    x_local = {}
    for t in ts_all:
        hs = sorted(df[df['t'] == t]['history_str'].unique().tolist())
        hist_by_t[t] = hs
        for i, h in enumerate(hs):
            x_local[(t, h)] = i

    arms_in_df = sorted({int(a) for a in df['arm'].unique()})
    if arm_colors is None:
        default_cycle = ['tab:blue', 'tab:orange', 'tab:red',
                         'tab:green', 'tab:purple', 'tab:brown']
        arm_colors = {a: default_cycle[a % len(default_cycle)] for a in range(n_arms)}
        arm_colors[n_arms] = 'tab:grey'

    ## fixed slot layout: every history reserves one sub-column per arm that
    ## ever appears, so arm k sits at the same horizontal offset everywhere.
    n_slots = max(1, len(arms_in_df))
    slot_w = bar_width / n_slots
    arm_slot = {a: i for i, a in enumerate(arms_in_df)}

    split_rows = last_trial_own_row and len(ts_all) >= 2
    if split_rows:
        top_ts = ts_all[:-1]
        bot_t = ts_all[-1]
    else:
        top_ts = ts_all
        bot_t = None

    n_top = max(1, len(top_ts))
    top_widths = [max(1, len(hist_by_t.get(t, []))) for t in top_ts] or [1]

    fig = plt.figure(figsize=figsize)
    if split_rows:
        gs = fig.add_gridspec(
            2, n_top, width_ratios=top_widths,
            height_ratios=[1, 1.1], hspace=1.4,
        )
        ax0 = fig.add_subplot(gs[0, 0])
        axes_top = [ax0] + [fig.add_subplot(gs[0, j], sharey=ax0)
                            for j in range(1, n_top)]
        ax_bot = fig.add_subplot(gs[1, :], sharey=ax0)
        plotted = list(zip(top_ts, axes_top)) + [(bot_t, ax_bot)]
        all_axes = axes_top + [ax_bot]
    else:
        gs = fig.add_gridspec(1, n_top, width_ratios=top_widths)
        ax0 = fig.add_subplot(gs[0, 0])
        axes_top = [ax0] + [fig.add_subplot(gs[0, j], sharey=ax0)
                            for j in range(1, n_top)]
        plotted = list(zip(top_ts, axes_top))
        all_axes = axes_top

    for t, ax in plotted:
        hs = hist_by_t[t]
        sub_t = df[df['t'] == t]
        for _, row in sub_t.iterrows():
            h = row['history_str']
            if (t, h) not in x_local:
                continue
            x = x_local[(t, h)]
            arm = int(row['arm'])
            lo = float(row['ell_lo'])
            hi = float(row['ell_hi'])
            color = arm_colors.get(arm, 'tab:grey')
            slot = arm_slot.get(arm, 0)
            x_pos = x + (slot - (n_slots - 1) / 2) * slot_w
            ax.bar(
                x_pos, hi - lo, bottom=lo, width=slot_w * 0.95,
                color=color, alpha=0.9,
                edgecolor='k', linewidth=0.4,
            )

        if log_scale:
            ax.set_yscale('log')
        if y_range is not None:
            ax.set_ylim(y_range)
        ax.axhline(1.0, color='k', linewidth=1.3, alpha=0.7, zorder=1.5)

        ax.set_title(f't = {t}')
        ax.set_xticks(list(range(len(hs))))
        leaf_labels = [_split_counts_str(h)[1] for h in hs]
        ax.set_xticklabels(leaf_labels, rotation=80, ha='right', fontsize=9)
        ax.grid(alpha=0.2, which='both', axis='y')
        if group_brackets:
            _add_group_brackets(ax, hs)

    axes_top[0].set_ylabel(r'$\ell$')
    if split_rows:
        ax_bot.set_ylabel(r'$\ell$')

    if legend:
        handles, labels = [], []
        for a in arms_in_df:
            lbl = 'terminate' if a == n_arms else f'arm {a}'
            handles.append(plt.Rectangle((0, 0), 1, 1,
                                         color=arm_colors.get(a, 'tab:grey')))
            labels.append(lbl)
        fig.legend(handles, labels, loc='upper right',
                   ncol=len(labels), framealpha=0.9)

    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=11, y=1.0)
    plt.tight_layout()
    return fig, all_axes


def plot_arm_curves(
    df_curves,
    n_arms,
    df_tip=None,
    termination_arm=True,
    y='p',
    arm_colors=None,
    log_x=True,
    ncols=4,
    panel_size=(3.2, 2.6),
    eps_tie=1e-8,
    suptitle=None,
    info_seeker=False,
    ML=False,
    save = True,
):
    """Plot Q-value (or softmax-prob) curves across ell for each (history, t)
    in `df_curves` (output of `enumerate_curves`). Returns one figure per
    trial t; each figure has a grid of panels for that trial's histories.

    `y='Q'` plots Q_a vs ell; `y='p'` plots softmax choice probabilities.
    For all histories on a single axis (including empowerment), see
    `plot_emp_curves`.

    Each panel gets a short strip beneath the curve plot showing which
    action(s) is the argmax across ell. When `df_tip` is provided, the strip
    is built from `df_tip`'s per-arm preferred intervals — horizontal bars
    coloured by arm and stacked by arm slot. When `df_tip` is not provided,
    the strip is derived directly from the curves in `df_curves`: at each
    sampled ell, plot a `|` marker at the arm slot of every action whose Q
    is within `eps_tie` of the max for that ell. Match `eps_tie` to the
    value used when building `df_tip` to make the two strips agree.

    Returns a dict mapping t -> (fig, main_axes) where main_axes is a 2D
    numpy array of main axes for that trial's grid.
    """
    if 'history_str' not in df_curves.columns or len(df_curves) == 0:
        raise ValueError("df_curves is empty or missing 'history_str'")
    if y not in ('Q', 'p'):
        raise ValueError(f"y must be 'Q' or 'p', got {y!r}")

    panels_by_t = {}
    for t, history_str in (df_curves[['t', 'history_str']]
                           .drop_duplicates()
                           .sort_values(['t', 'history_str'])
                           .itertuples(index=False, name=None)):
        panels_by_t.setdefault(int(t), []).append(history_str)

    ts = sorted(panels_by_t.keys())

    if arm_colors is None:
        default_cycle = [
            'tab:blue', 'tab:red', 'tab:green',
            # 'tab:blue', 'tab:orange', 'tab:red',
            #              'tab:green', 'tab:purple', 'tab:brown',
                         ]
        arm_colors = {a: default_cycle[a % len(default_cycle)] for a in range(n_arms)}
        if termination_arm:
            arm_colors[n_arms] = 'tab:grey'

    ell_lo = df_curves['ell'].min()
    ell_hi = df_curves['ell'].max()

    panel_h = panel_size[1] * 1.25
    figs_by_t = {}

    for t in ts:
        histories = panels_by_t[t]
        n_p = len(histories)
        nr = (n_p + ncols - 1) // ncols
        figsize = (panel_size[0] * ncols, nr * panel_h + 0.6)

        fig = plt.figure(figsize=figsize, constrained_layout=False)
        outer = fig.add_gridspec(nr, ncols, hspace=0.55, wspace=0.3)
        main_axes = np.empty((nr, ncols), dtype=object)
        strip_axes = np.empty((nr, ncols), dtype=object)
        for i in range(nr):
            for j in range(ncols):
                inner = outer[i, j].subgridspec(
                    2, 1, height_ratios=[4, 1.2], hspace=0.08)
                ax_main = fig.add_subplot(inner[0])
                ax_strip = fig.add_subplot(inner[1], sharex=ax_main)
                main_axes[i, j] = ax_main
                strip_axes[i, j] = ax_strip

        ax_flat = main_axes.flat
        strip_flat = strip_axes.flat

        for i, history_str in enumerate(histories):
            ax = ax_flat[i]
            sub = (df_curves[(df_curves['history_str'] == history_str)
                           & (df_curves['t'] == t)]
                   .sort_values('ell'))
            for a in range(n_arms):
                ax.plot(sub['ell'], sub[f'{y}_{a}'], '-',
                        color=arm_colors[a], label=f'arm {a}', alpha=0.9)
            if termination_arm and f'{y}_terminate' in sub.columns:
                ax.plot(sub['ell'], sub[f'{y}_terminate'], '-',
                        color=arm_colors[n_arms], label='terminate', alpha=0.9)
            if log_x:
                ax.set_xscale('log')
                decades = np.log10(ell_hi) - np.log10(ell_lo)
                log_ticks = np.logspace(np.ceil(np.log10(ell_lo)), np.floor(np.log10(ell_hi)), num=int(decades) + 1)
                ax.set_xticks(log_ticks)
            if ML:
                ax.set_title(f'{history_str}\nML = {sub["ML"].iloc[0]:.3f}', fontsize=9)
            else:
                ax.set_title(history_str, fontsize=9)
            ax.grid(alpha=0.25, which='both')
            if i % ncols == 0:
                ax.set_ylabel(y)

            chance = 1 / (n_arms + (1 if termination_arm else 0))
            if y == 'p':
                ax.axhline(chance, color='k', 
                           linestyle='--', 
                           linewidth=1, 
                        #    alpha=0.7,
                        zorder=1.5)
            
            ## let's also plot info_p_a and info_p_terminate as coloured dotted lines
            if info_seeker:
                for a in range(n_arms):
                    ax.plot(sub['ell'], sub[f'info_p_{a}'], ':',
                            color=arm_colors[a], label=f'Info-seeker: arm {a}', 
                            linewidth=2,
                            # alpha=0.7
                            )
                if termination_arm and 'info_p_terminate' in sub.columns:
                    ax.plot(sub['ell'], sub['info_p_terminate'], ':',
                            color=arm_colors[n_arms], label='Info-seeker: terminate', 
                            linewidth=2,
                            # alpha=0.7
                            )

            strip_ax = strip_flat[i]
            arms_present = []
            if df_tip is not None:
                sub_tip = df_tip[(df_tip['history_str'] == history_str)
                                 & (df_tip['t'] == t)]
                arms_present = sorted({int(a) for a in sub_tip['arm'].unique()})
                for slot, arm in enumerate(arms_present):
                    arm_rows = sub_tip[sub_tip['arm'] == arm]
                    for _, r in arm_rows.iterrows():
                        strip_ax.barh(
                            slot, r['ell_hi'] - r['ell_lo'], left=r['ell_lo'],
                            height=0.8, color=arm_colors.get(arm, 'tab:grey'),
                            alpha=0.9, edgecolor='k', linewidth=0.4,
                        )
            else:
                arm_labels = list(range(n_arms))
                Q_cols = [f'Q_{a}' for a in range(n_arms)]
                p_cols = [f'p_{a}' for a in range(n_arms)]
                if termination_arm and 'Q_terminate' in sub.columns:
                    Q_cols.append('Q_terminate')
                    p_cols.append('p_terminate')
                    arm_labels.append(n_arms)
                Q_matrix = sub[Q_cols].values
                p_matrix = sub[p_cols].values
                ells_arr = sub['ell'].values
                # max_per_row = Q_matrix.max(axis=1, keepdims=True)
                max_per_row = p_matrix.max(axis=1, keepdims=True)
                # co_argmax = np.abs(Q_matrix - max_per_row) < eps_tie
                co_argmax = np.abs(p_matrix - max_per_row) < eps_tie
                arms_present = [arm_labels[k] for k in range(len(arm_labels))
                                if co_argmax[:, k].any()]
                for slot, arm in enumerate(arms_present):
                    k = arm_labels.index(arm)
                    ells_for_arm = ells_arr[co_argmax[:, k]]
                    strip_ax.scatter(
                        ells_for_arm, np.full(len(ells_for_arm), slot),
                        color=arm_colors.get(arm, 'tab:grey'),
                        marker='|', s=80, linewidths=1.2,
                    )

            if arms_present:
                strip_ax.set_yticks(range(len(arms_present)))
                strip_ax.set_yticklabels(
                    ['T' if a == n_arms else str(a) for a in arms_present],
                    fontsize=7)
                strip_ax.set_ylim(-0.5, len(arms_present) - 0.5)
            strip_ax.grid(alpha=0.2, axis='x', which='both')
            ax.tick_params(labelbottom=False)
            if i // ncols == nr - 1:
                strip_ax.set_xlabel(r'$\ell$')

        for j in range(n_p, nr * ncols):
            ax_flat[j].set_visible(False)
            strip_flat[j].set_visible(False)

        handles, labels = main_axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right',
                   ncol=len(labels), framealpha=0.9)
        title = f't = {t}' if suptitle is None else f'{suptitle}  |  t = {t}'
        fig.suptitle(title, fontsize=12, fontweight='bold')
        figs_by_t[t] = (fig, main_axes)
    

    return figs_by_t


## ---------------------------------------------------------------------------
## history glyphs: a visual stand-in for history strings like 'a0o0:2-a1o1:1'.
## A plus-shaped grid of 5 cells; outcome o gets the cell in direction
## HISTORY_OUTCOME_DIRS[o] from the (empty) centre, and each (a, o) observation
## is a token in that cell, coloured by action.
## ---------------------------------------------------------------------------

HISTORY_OUTCOME_DIRS = {0: (0, 1), 1: (1, 0), 2: (0, -1), 3: (-1, 0)}  # up, right, down, left
HISTORY_ACTION_COLORS = ['tab:blue', 'tab:red', 'tab:green', 'tab:orange',
                         'tab:purple', 'tab:brown']


def parse_history_str(history):
    """'a0o0:2-a1o1:1' -> {(0, 0): 2, (1, 1): 1}; '' / 'init' -> {}."""
    counts = {}
    if history in ('', 'init', None):
        return counts
    for tok in str(history).split('-'):
        m = re.fullmatch(r'a(\d+)o(\d+):(\d+)', tok.strip())
        if m is None:
            raise ValueError(f'cannot parse history token {tok!r} in {history!r}')
        a, o, c = map(int, m.groups())
        counts[(a, o)] = counts.get((a, o), 0) + c
    return counts


def _token_offsets(n, min_grid=2):
    """Centres (in units of cell size, relative to cell centre) and radius
    for n tokens packed on a near-square grid inside one cell. The grid is
    at least `min_grid` wide so token size only shrinks once a cell gets
    crowded, rather than a lone token filling its cell."""
    if n == 0:
        return [], 0.
    ncol = int(np.ceil(np.sqrt(n)))
    nrow = int(np.ceil(n / ncol))
    pitch = 0.84 / max(ncol, nrow, min_grid)
    offsets = []
    for i in range(n):
        r, c = divmod(i, ncol)
        n_in_row = min(ncol, n - r * ncol)  # centre a partial last row
        x = (c - (n_in_row - 1) / 2) * pitch
        y = ((nrow - 1) / 2 - r) * pitch
        offsets.append((x, y))
    return offsets, 0.42 * pitch


def history_glyph_artists(history, center=(0., 0.), size=1., action_colors=None,
                          outcome_dirs=None, transform=None,
                          cell_kw=None, token_kw=None):
    """Build (but don't add) the patches for one history glyph.

    `history` is a history string or a {(a, o): count} dict. `size` is the
    side of one cell, so the glyph spans 3*size. `transform` is applied to
    every patch (e.g. a legend handlebox transform); None leaves them in
    the data coords of whatever axes they get added to.
    """
    counts = parse_history_str(history) if not isinstance(history, dict) else history
    action_colors = HISTORY_ACTION_COLORS if action_colors is None else action_colors
    outcome_dirs = HISTORY_OUTCOME_DIRS if outcome_dirs is None else outcome_dirs
    cell_kw = {'facecolor': '0.96', 'edgecolor': '0.55', 'linewidth': 0.6, **(cell_kw or {})}
    token_kw = {'edgecolor': 'white', 'linewidth': 0.3, **(token_kw or {})}
    cx, cy = center

    by_outcome = {}
    for (a, o), c in sorted(counts.items()):
        if o not in outcome_dirs:
            raise ValueError(f'outcome {o} has no glyph position (outcome_dirs has '
                             f'{sorted(outcome_dirs)})')
        by_outcome.setdefault(o, []).extend([a] * c)

    artists = []
    for dx, dy in [(0, 0)] + [outcome_dirs[o] for o in sorted(outcome_dirs)]:
        artists.append(Rectangle((cx + (dx - 0.5) * size, cy + (dy - 0.5) * size),
                                 size, size, **cell_kw))
    for o, actions in by_outcome.items():
        dx, dy = outcome_dirs[o]
        offsets, r = _token_offsets(len(actions))
        for a, (ox, oy) in zip(actions, offsets):
            color = (action_colors[a] if isinstance(action_colors, dict)
                     else action_colors[a % len(action_colors)])
            artists.append(Circle((cx + (dx + ox) * size, cy + (dy + oy) * size),
                                  r * size, facecolor=color, **token_kw))
    if transform is not None:
        for art in artists:
            art.set_transform(transform)
    return artists


def draw_history_glyph(ax, history, center=(0., 0.), size=1., **kwargs):
    """Add a history glyph to `ax` in data coords (see history_glyph_artists).
    Tokens are circles, so use an equal-aspect axis to keep them round."""
    artists = history_glyph_artists(history, center=center, size=size, **kwargs)
    for art in artists:
        ax.add_patch(art)
    return artists


def plot_history_glyph(history, ax=None, size=1., title=None, **kwargs):
    """Standalone glyph on its own (possibly new) axis. Returns ax."""
    if ax is None:
        _, ax = plt.subplots(figsize=(1.2, 1.2))
    draw_history_glyph(ax, history, size=size, **kwargs)
    lim = 1.55 * size
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.axis('off')
    if title is not None:
        ax.set_title(title, fontsize=6)
    return ax


class _HistoryHandle:
    """Legend proxy: a line colour plus a history to draw as a glyph."""
    def __init__(self, history, color, lw=2.0):
        self.history, self.color, self.lw = history, color, lw


class _HistoryGlyphHandler(HandlerBase):
    """Legend handler drawing a short line sample followed by the glyph."""
    def __init__(self, action_colors=None, outcome_dirs=None, **kw):
        super().__init__(**kw)
        self.action_colors, self.outcome_dirs = action_colors, outcome_dirs

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width,
                       height, fontsize, trans):
        x0, y0 = -xdescent, -ydescent
        size = height / 3
        glyph_w = 3 * size
        line_end = x0 + max(width - glyph_w - 0.3 * fontsize, 0.3 * width)
        line = Line2D([x0, line_end], [y0 + height / 2] * 2,
                      color=orig_handle.color, lw=orig_handle.lw,
                      solid_capstyle='butt', transform=trans)
        glyph = history_glyph_artists(
            orig_handle.history,
            center=(x0 + width - glyph_w / 2, y0 + height / 2), size=size,
            action_colors=self.action_colors, outcome_dirs=self.outcome_dirs,
            transform=trans, cell_kw={'linewidth': 0.4})
        return [line] + glyph


def history_glyph_legend(ax, histories, colors, action_colors=None,
                         outcome_dirs=None, show_text=False, glyph_height=5,
                         fontsize=7, **legend_kw):
    """Legend on `ax` with one (line colour, glyph) entry per history.
    `glyph_height` is the glyph height in units of `fontsize`. With
    `show_text` the history string is kept as the entry's label."""
    handles = [_HistoryHandle(h, c) for h, c in zip(histories, colors)]
    labels = [h if show_text else '' for h in histories]
    legend_kw = {'handleheight': glyph_height, 'handlelength': glyph_height + 1.5,
                 'handletextpad': 0.4 if show_text else 0., 'labelspacing': 0.3,
                 'columnspacing': 1.0, 'framealpha': 0.9, **legend_kw}
    handler = _HistoryGlyphHandler(action_colors=action_colors, outcome_dirs=outcome_dirs)
    return ax.legend(handles, labels, fontsize=fontsize,
                     handler_map={_HistoryHandle: handler}, **legend_kw)


def plot_emp_curves(df_curves, y='emp', temp=1.0, log_x=True, figsize=(7, 4.5),
                    cmap='viridis', suptitle=None, history_repr='text',
                    action_colors=None, glyph_labels=False):
    """Plot a per-history quantity vs ell for every history in `df_curves`
    on one axis, one line per history, coloured by history (ordered by t,
    then history_str, along `cmap`). Returns (fig, ax).

    `y='emp'` plots the empowerment of the current belief state
    (`current_emp`); `y='p'` plots, at each ell, the softmax of `current_emp`
    across the plotted histories, softmax(emp / temp) -- i.e. the probability
    of choosing each history if its empowerment were its value. The softmax
    runs over exactly the histories in `df_curves`, so filter it to the set
    being compared (and to a single alpha / horizon / cost) first.

    `history_repr` sets how histories are identified: 'text' (history strings
    in the legend), 'glyph' (legend of line sample + history glyph) or 'inset'
    (a side panel with one glyph per history, underlined in its curve colour).
    `action_colors` colours glyph tokens by action; `glyph_labels` also
    prints the history strings next to the glyphs.
    """
    if y not in ('emp', 'p'):
        raise ValueError(f"y must be 'emp' or 'p', got {y!r}")
    if history_repr not in ('text', 'glyph', 'inset'):
        raise ValueError(f"history_repr must be 'text', 'glyph' or 'inset', got {history_repr!r}")
    histories = (df_curves[['t', 'history_str']]
                 .drop_duplicates()
                 .sort_values(['t', 'history_str'])['history_str']
                 .tolist())
    colors = plt.get_cmap(cmap)(np.linspace(0, 1, max(len(histories), 2)))

    ## (ell x history) grid of empowerments
    if df_curves.duplicated(['ell', 'history_str']).any():
        raise ValueError('df_curves has several rows per (ell, history_str); '
                         'filter it to a single alpha / horizon / cost first')
    emp =(df_curves.pivot(index='ell', columns='history_str', values='current_emp')
           .sort_index()[histories])
    if y == 'p':
        vals = pd.DataFrame(softmax(emp.values / temp, axis=1),
                            index=emp.index, columns=emp.columns)
        y_label = 'p'
    else:
        vals, y_label = emp, 'emp'

    if history_repr == 'inset':
        ## side panel of glyphs, roughly as tall as the main axis
        n = len(histories)
        g_nrows = max(1, min(n, int(np.ceil(np.sqrt(2 * n)))))
        g_ncols = int(np.ceil(n / g_nrows))
        cell_in = figsize[1] / g_nrows
        fig = plt.figure(figsize=(figsize[0] + g_ncols * cell_in, figsize[1]))
        gs = fig.add_gridspec(1, 2, width_ratios=[figsize[0], g_ncols * cell_in], wspace=0.05)
        ax = fig.add_subplot(gs[0])
        ggs = gs[1].subgridspec(g_nrows, g_ncols, wspace=0.05, hspace=0.25 if glyph_labels else 0.05)
        for i, (color, h) in enumerate(zip(colors, histories)):
            r, c = i % g_nrows, i // g_nrows  # fill column-major, like a legend
            gax = fig.add_subplot(ggs[r, c])
            plot_history_glyph(h, ax=gax, action_colors=action_colors,
                               title=h if glyph_labels else None)
            gax.set_ylim(-1.9, 1.55)
            gax.plot([-1.2, 1.2], [-1.75, -1.75], color=color, lw=2.5,
                     solid_capstyle='butt')
    else:
        fig, ax = plt.subplots(figsize=figsize)

    for color, history_str in zip(colors, histories):
        ax.plot(vals.index, vals[history_str], '-', color=color,
                label=history_str, alpha=0.9)
    if y == 'p':
        ax.axhline(1 / len(histories), color='k', linestyle='--',
                   linewidth=1, zorder=1.5)
    if log_x:
        ax.set_xscale('log')
    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.25, which='both')
    if history_repr == 'text':
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=7,
                  ncol=max(1, len(histories) // 25), framealpha=0.9)
    elif history_repr == 'glyph':
        glyph_height, spacing, fontsize = 5, 0.3, 7
        rows_fit = max(1, int(figsize[1] * 72 / ((glyph_height + spacing) * fontsize)))
        history_glyph_legend(ax, histories, colors, action_colors=action_colors,
                             show_text=glyph_labels, glyph_height=glyph_height,
                             fontsize=fontsize, labelspacing=spacing,
                             loc='center left', bbox_to_anchor=(1.02, 0.5),
                             ncol=int(np.ceil(len(histories) / rows_fit)))
    if suptitle is not None:
        ax.set_title(suptitle, fontsize=12, fontweight='bold')
    if history_repr != 'inset':  # tight_layout fights the glyph subgridspec
        fig.tight_layout()
    return fig, ax


def _heatmap_metric_spec(metric, cmap):
    """Resolve a heat `metric` into the data columns to pivot, the columns
    used to build the heat value, the info-seeker columns, a colorbar label,
    whether the scale is symmetric about zero, and a default cmap.

    Returns (value_cols, info_cols, build_heat, cbar_label, symmetric, cmap).
    `build_heat(pivot, cols)` turns a pivot (with a column level matching
    `cols`) into the 2D heat array.
    """
    if metric == 'p_diff':
        value_cols = ['p_0', 'p_1']
        info_cols = ['info_p_0', 'info_p_1']
        cbar_label = 'p₀ − p₁'
        symmetric = True
        cmap = 'RdBu' if cmap is None else cmap

        def build_heat(pivot, cols):
            return (pivot[cols[0]] - pivot[cols[1]]).values
    elif metric == 'p_terminate':
        value_cols = ['p_terminate']
        info_cols = ['info_p_terminate']
        cbar_label = 'P(terminate)'
        symmetric = False
        cmap = 'viridis' if cmap is None else cmap

        def build_heat(pivot, cols):
            return pivot[cols[0]].values
    else:
        raise ValueError(f"unknown metric {metric!r}; expected 'p_diff' or 'p_terminate'")
    return value_cols, info_cols, build_heat, cbar_label, symmetric, cmap


def _heatmap_limits(heat, symmetric):
    """Per-array (vmin, vmax) for a heat panel. Symmetric metrics centre on
    zero with a small floor so a near-flat panel doesn't blow up the scale;
    asymmetric metrics use the data range (with a small floor)."""
    if symmetric:
        vmax_abs = max(abs(np.nanmin(heat)), abs(np.nanmax(heat)))
        if vmax_abs < 1e-3:
            vmax_abs = 1e-3
        return -vmax_abs, vmax_abs
    vmin = float(np.nanmin(heat))
    vmax = float(np.nanmax(heat))
    if vmax - vmin < 1e-3:
        vmax = vmin + 1e-3
    return vmin, vmax


def plot_heatmap(
    df_curves,
    y_axis='alpha',
    metric='p_diff',
    fixed_k=None,
    fixed_alpha=None,
    max_n_cols=4,
    cmap=None,
    panel_size=(3.0, 3.0),
    suptitle=None,
    shared_colorbar=True,
    plot_info_seeker=True,
):
    """Plot 2D heatmaps of a per-(history, t) metric over an (ell, y_axis) grid.
    Returns one figure per trial t; each figure has a grid of panels for that
    trial's histories.

    `y_axis` chooses what varies down the y-axis:
      - 'alpha' (default): one row per Dirichlet concentration alpha, at the
        single sampling cost `fixed_k`.
      - 'k': one row per sampling cost k, at the single concentration
        `fixed_alpha` — i.e. how a fixed-ell agent's preference shifts as the
        cost of sampling changes.
    The variable not on the y-axis is sliced to a single value (`fixed_k` /
    `fixed_alpha`); if left None it defaults to 0.0 when present, else the
    smallest value, so the pivot never silently averages over it.

    `metric` chooses the heat:
      - 'p_diff' (default): p_0 - p_1, symmetric diverging scale.
      - 'p_terminate': P(terminate), sequential scale.

    Expected input: df_curves from enumerate_curves with ell, alpha and (for
    the k-sweep / cost slicing) k columns.

    Returns a dict mapping t -> (fig, axes_array) where axes_array is a 2D numpy
    array of axes for that trial's grid.
    """
    if 'history_str' not in df_curves.columns or len(df_curves) == 0:
        raise ValueError("df_curves is empty or missing 'history_str'")
    if y_axis not in ('alpha', 'k'):
        raise ValueError(f"unknown y_axis {y_axis!r}; expected 'alpha' or 'k'")
    if y_axis not in df_curves.columns:
        raise ValueError(f"df_curves missing {y_axis!r} column")

    value_cols, info_cols, build_heat, cbar_label, symmetric, cmap = \
        _heatmap_metric_spec(metric, cmap)
    if plot_info_seeker:
        plot_info_seeker = all(c in df_curves.columns for c in info_cols)

    ## the variable NOT on the y-axis is fixed to a single slice so the pivot
    ## (which aggregates with 'first') never silently collapses it.
    slice_col = 'k' if y_axis == 'alpha' else 'alpha'
    requested = fixed_k if y_axis == 'alpha' else fixed_alpha
    if slice_col in df_curves.columns:
        slice_vals = sorted(v for v in df_curves[slice_col].dropna().unique())
        ## default to the lowest value present; otherwise use the requested one.
        slice_val = requested if requested is not None else (
            slice_vals[0] if slice_vals else None)
        if slice_val is not None:
            df_curves = df_curves[df_curves[slice_col] == slice_val]
            if len(df_curves) == 0:
                raise ValueError(f"no rows with {slice_col} == {slice_val!r}")
        slice_note = f'{slice_col} = {slice_val}'
    else:
        slice_note = None

    y_label = 'α' if y_axis == 'alpha' else 'k'

    panels_by_t = {}
    for t, history_str in (df_curves[['t', 'history_str']]
                           .drop_duplicates()
                           .sort_values(['t', 'history_str'])
                           .itertuples(index=False, name=None)):
        panels_by_t.setdefault(int(t), []).append(history_str)

    ts = sorted(panels_by_t.keys())

    panel_h = panel_size[1]
    figs_by_t = {}

    for t in ts:
        ncols = min(max_n_cols, len(panels_by_t[t]))

        # Calculate vmin/vmax based on shared_colorbar setting
        if shared_colorbar:
            # Shared colorbar: normalize across all histories in this trial.
            # build_heat works on the raw trial frame too (plain column access).
            df_t = df_curves[df_curves['t'] == t]
            vmin, vmax = _heatmap_limits(build_heat(df_t, value_cols), symmetric)
        else:
            # Per-history colorbars: will compute for each history in the loop
            vmin = None
            vmax = None

        histories = panels_by_t[t]
        n_p = len(histories)
        nr = (n_p + ncols - 1) // ncols
        figsize = (panel_size[0] * ncols, nr * panel_h)

        fig = plt.figure(figsize=figsize, constrained_layout=False)
        outer = fig.add_gridspec(nr, ncols)
        axes = np.empty((nr, ncols), dtype=object)
        info_axes = np.empty((nr, ncols), dtype=object)
        for i in range(nr):
            for j in range(ncols):
                inner = outer[i, j].subgridspec(1, 2, width_ratios=[20, 1], wspace=0)
                ax_main = fig.add_subplot(inner[0])
                ax_info = fig.add_subplot(inner[1], sharey=ax_main)
                axes[i, j] = ax_main
                info_axes[i, j] = ax_info

        ax_flat = axes.flat
        info_flat = info_axes.flat

        for i, history_str in enumerate(histories):
            ax = ax_flat[i]
            ax_info = info_flat[i]
            sub = (df_curves[(df_curves['history_str'] == history_str)
                           & (df_curves['t'] == t)])

            if len(sub) == 0:
                ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                        transform=ax.transAxes, color='grey', fontsize=10)
                ax.set_title(history_str, fontsize=8)
                ax_info.set_visible(False)
                continue

            pivot_cols = value_cols + (info_cols if plot_info_seeker else [])
            pivot = sub.pivot_table(index=y_axis, columns='ell',
                                    values=pivot_cols, aggfunc='first')
            if pivot.empty or value_cols[0] not in pivot.columns:
                ax.text(0.5, 0.5, 'pivot error', ha='center', va='center',
                        transform=ax.transAxes, color='grey', fontsize=10)
                ax.set_title(history_str, fontsize=8)
                ax_info.set_visible(False)
                continue

            heat = build_heat(pivot, value_cols)
            ells = pivot[value_cols[0]].columns.values
            ys = pivot[value_cols[0]].index.values

            # Compute per-history vmin/vmax if not using shared colorbar
            if not shared_colorbar:
                hist_vmin, hist_vmax_final = _heatmap_limits(heat, symmetric)
            else:
                hist_vmin = vmin
                hist_vmax_final = vmax

            pm = ax.pcolormesh(ells, ys, heat, cmap=cmap, vmin=hist_vmin, vmax=hist_vmax_final,
                               shading='auto')
            ax.set_xscale('log')
            # Set x-axis ticks to show each decade
            ax.xaxis.set_major_locator(LogLocator(base=10, numticks=15))
            ax.set_title(history_str, fontsize=8)

            if i % ncols == 0:
                ax.set_ylabel(y_label)
            if i // ncols == nr - 1:
                ax.set_xlabel(r'$\ell$')
            ax.grid(alpha=0.25, which='both')

            # Plot info-seeker panel (vertical bar)
            if plot_info_seeker and all(c in pivot.columns for c in info_cols):
                info_heat = build_heat(pivot, info_cols)
                # Average across ell (should be constant for info_p)
                info_col = info_heat.mean(axis=1, keepdims=True)
                im = ax_info.imshow(info_col, aspect='auto', cmap=cmap, vmin=hist_vmin, vmax=hist_vmax_final,
                                    extent=[0, 1, ys[0], ys[-1]], origin='lower', interpolation='none')
                ax_info.tick_params(labelleft=False, labelbottom=False)
                if i // ncols == nr - 1:
                    ax_info.set_xlabel('info-seeker', fontsize=7)

                # Add per-panel colorbar if not using shared colorbar
                if not shared_colorbar:
                    fig.colorbar(im, ax=ax_info, label=cbar_label, pad=0.02, aspect=20)
            else:
                ax_info.set_visible(False)
                # If not plotting info-seeker but using per-panel colorbars, add colorbar to main axis
                if not shared_colorbar:
                    fig.colorbar(pm, ax=ax, label=cbar_label, pad=0.02)

        for j in range(n_p, nr * ncols):
            ax_flat[j].set_visible(False)
            info_flat[j].set_visible(False)

        # Apply tight layout before adding colorbar to prevent width adjustments
        # Reserve space at the top for the suptitle
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Add a single shared colorbar for the entire trial (only when shared_colorbar=True)
        if shared_colorbar:
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            # Attach colorbar to info axis if plotting info-seeker, otherwise to main axis
            cbar_ax = info_axes[0, -1] if plot_info_seeker else axes[0, -1]
            cbar_kwargs = {'label': cbar_label, 'pad': 0.08}
            if plot_info_seeker:
                cbar_kwargs['aspect'] = 20  # Make it taller when attached to narrow info axis
            fig.colorbar(sm, ax=cbar_ax, **cbar_kwargs)

        title_bits = [f't = {t}']
        if slice_note is not None:
            title_bits.append(slice_note)
        title = '  |  '.join(title_bits)
        if suptitle is not None:
            title = f'{suptitle}  |  {title}'
        fig.suptitle(title, fontsize=12, fontweight='bold')
        figs_by_t[t] = (fig, axes)

    return figs_by_t