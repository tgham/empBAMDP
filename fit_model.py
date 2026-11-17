"""Fit the empowerment / info-seeking agents to participants' behavioural data.

Lifted out of the `Fitting > Fit emp agent` cells of empBAMDP.ipynb so the fits
can be run headlessly (e.g. under run.sh) rather than in the notebook.

    ## Experiment 4 (arms): sequential presses within each room
    python fit_model.py --expt arms --n_arms 3 --horizon 3 --init_t 1

    ## Experiment 6 (rooms): one 2AFC choice between preset rooms per trial
    python fit_model.py --expt rooms --n_arms 3 --optimizer de --log_ell_bounds -4 4

    ## Bayesian fit by grid approximation (priors: log ell ~ N(0, 2), beta ~ N+(0, 10))
    python fit_model.py --expt rooms --n_arms 3 --optimizer grid --log_ell_bounds -4 4

Writes one row per (id, agent_type) to
  arms:  useful_saves/fits/{n_arms}a{n_outcomes}k_fits.csv
  rooms: useful_saves/fits/rooms/{data_dir name}_fits.csv
which is what the downstream notebook cells (BIC comparison, ell histograms,
simulating agents under the fitted ells) read back in. With --optimizer grid,
the csv also has each fit's log_evidence and posterior summaries, and each
(id, agent_type)'s full log posterior grid goes to `{stem}_posteriors.npz`.

With --mode loocv (or both), also runs leave-one-block-out cross-validation:
one fold per value of the data's `block` column, fitting to every other block
and scoring the held-out block's trials under the fitted parameters. Writes one
row per (id, agent_type) to the fits path with `_fits.csv` swapped for
`_loocv.csv`: each fold's fitted parameters, train_nll and test_nll as
`{col}_block{b}` columns (b = held-out block), plus `test_nll` and
`n_test_trials` totalled over folds.
"""

from emp_runners import fit_emp, nll_emp
import pandas as pd
import numpy as np
from scipy.special import logsumexp
import load_data
import load_data_rooms
import argparse
import os


## per-experiment defaults: where the data lives, which loader / dataframe to
## use, and the sampling cost (as in each experiment's config.js)
EXPTS = {
    'arms': dict(
        loader=load_data.load_directory,
        df_key='sample',
        data_dir='expt/Experiment4 - arms/data/{n_arms}a{n_outcomes}k',
        cost=lambda n_trials: 1 / (n_trials + 1),
    ),
    'rooms': dict(
        loader=load_data_rooms.load_directory,
        df_key='choice',
        data_dir='expt/Experiment6 - rooms/data/{n_arms}a{n_outcomes}k_gold',
        cost=lambda n_trials: 0.0,  ## SAMPLE_COST = 0
    ),
}


def loocv_emp(df, fit_kwargs, init_t=0, verbose=True):
    """Leave-one-block-out cross-validation of `fit_emp`.

    For each block b: fit every agent type to all trials outside b, then score
    each subject's block-b trials with `nll_emp` under that subject's fitted
    (ell, beta, agent_alpha). Returns one row per (id, agent_type, block), with
    the training fit's parameters and `test_nll`.

    Grid fits (optimizer='grid') score block b by the posterior predictive
    instead, averaging the held-out likelihood over the training posterior:
        p(D_b | D_-b) = sum_g p(theta_g | D_-b) p(D_b | theta_g)
        test_nll = -logsumexp(log_post_train - NLL_test)
    and keep the plug-in score (at the posterior mean) as `test_nll_plugin`.
    """
    blocks = np.sort(df['block'].unique())
    rows = []
    for b in blocks:
        if verbose:
            print(f'--- fold: holding out block {b} ({len(blocks)} blocks) ---')
        df_train = df.loc[df['block'] != b]
        df_test = df.loc[df['block'] == b]
        df_fits = fit_emp(df_ppt=df_train, init_t=init_t, verbose=verbose, **fit_kwargs)
        for fit in df_fits.to_dict('records'):
            post = fit.pop('_post', None)
            df_test_ppt = df_test.loc[df_test['id'] == fit['id']]
            if df_test_ppt.empty:
                continue
            score_kwargs = dict(alpha=fit['agent_alpha'], horizon=fit_kwargs.get('horizon'), init_t=init_t)

            ## plug-in score at the fitted point; info agent has no ell (NaN once in a DataFrame) -> None
            ell = None if pd.isna(fit['ell']) else fit['ell']
            test_nll = nll_emp(df_test_ppt, ell=ell, beta=fit['beta'], **score_kwargs)

            ## grid fit: posterior predictive, averaging the held-out likelihood over the training posterior
            extra = {}
            if post is not None:
                ells = None if post['log_ells'] is None else np.exp(post['log_ells'])
                nll_test_grid = nll_emp(df_test_ppt, ell=ells, beta=post['betas'], **score_kwargs)
                extra['test_nll_plugin'] = test_nll
                test_nll = float(-logsumexp(post['log_post'] - nll_test_grid))
            ## scored trials, as counted in _fit_ppt
            if df_test_ppt.get('expt', pd.Series(['arms'])).iloc[0] == 'rooms':
                n_test_trials = len(df_test_ppt)
            else:
                n_test_trials = int((df_test_ppt['trial'] >= init_t).sum())
            rows.append({**fit,
                         'block': b,
                         'train_nll': fit['nll'],
                         'test_nll': test_nll,
                         **extra,
                         'n_test_trials': n_test_trials})
    df_cv = pd.DataFrame(rows).drop(columns=['nll', 'BIC', 'n_trials'])
    return df_cv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--expt', choices=list(EXPTS), default='arms',
                        help="'arms': Experiment 4, sequential presses; "
                             "'rooms': Experiment 6, 2AFC choices between preset rooms")
    parser.add_argument('--data_dir', type=str, default=None,
                        help='directory of participant .json files; defaults to '
                             "'expt/Experiment4 - arms/data/{n_arms}a{n_outcomes}k' (arms) or "
                             "'expt/Experiment6 - rooms/data/{n_arms}a{n_outcomes}k_gold' (rooms)")
    parser.add_argument('--n_arms', type=int, default=3,
                        help='arms: sets the design; rooms: read from the data, so only '
                             'used for the default --data_dir')
    parser.add_argument('--n_outcomes', type=int, default=4, help='as --n_arms')
    parser.add_argument('--n_trials', type=int, default=8)
    parser.add_argument('--cost', type=float, default=None,
                        help='sampling cost; defaults to 1/(n_trials+1) for arms and 0 for '
                             "rooms, as in each experiment's config.js")
    parser.add_argument('--alpha', type=float, default=None,
                        help='override the Dirichlet alpha logged per participant')
    parser.add_argument('--termination_arm', action='store_true')
    parser.add_argument('--log_ell_bounds', type=float, default=None, nargs=2,
                        help='bounds on the fitted log ell (default: unbounded)')
    parser.add_argument('--optimizer', choices=['lbfgsb', 'de', 'grid'], default='lbfgsb',
                        help="'lbfgsb': multi-start L-BFGS-B, allows open --log_ell_bounds; "
                             "'de': differential evolution, needs finite bounds; "
                             "'grid': Bayesian grid approximation over (log ell, beta).")
    parser.add_argument('--n_ell_samples', type=int, default= 200, help='log-ell grid size for --optimizer grid')
    parser.add_argument('--n_beta_samples', type=int, default=200, help='beta grid size for --optimizer grid')
    parser.add_argument('--n_starts', type=int, default=10,
                        help='starting points for --optimizer lbfgsb')
    parser.add_argument('--inv_temp_bounds', type=float, default=(0.001, 100), nargs=2)
    parser.add_argument('--alpha_bounds', type=float, default=None, nargs=2,
                        help="if given, also fit the agent's Dirichlet alpha within these "
                             "bounds (default: fixed to the design's alpha)")
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--horizon', type=int, default=1,
                        help='planning horizon (arms only; ignored for rooms)')
    parser.add_argument('--init_t', type=int, default=1,
                        help='leading trials used to warm the belief without being scored '
                             '(arms only; ignored for rooms)')
    parser.add_argument('--agent_types', nargs='+', default=['emp', 'emp_lo', 'emp_1', 'emp_hi', 'info'],
                        choices=['emp', 'emp_lo', 'emp_1', 'emp_hi', 'info'],
                        help="models to fit, one row per subject each. Pass "
                             "'emp_lo emp_1 emp_hi' in place of 'emp' to split "
                             "the empowerment agent by ell<1, ell=1 and ell>1.")
    parser.add_argument('--mode', choices=['fit', 'loocv', 'both'], default='fit',
                        help="'fit': fit to all trials; 'loocv': leave-one-block-out "
                             "cross-validation over the data's `block` column; 'both'")
    parser.add_argument('--n_jobs', type=int, default=-1)
    parser.add_argument('--out', type=str, default=None,
                        help='output csv; defaults to '
                             "'useful_saves/fits/{n_arms}a{n_outcomes}k_fits.csv' (arms) or "
                             "'useful_saves/fits/rooms/{data_dir name}_fits.csv' (rooms)")

    args = parser.parse_args()
    expt = EXPTS[args.expt]

    data_dir = args.data_dir or expt['data_dir'].format(n_arms=args.n_arms, n_outcomes=args.n_outcomes)
    cost = args.cost if args.cost is not None else expt['cost'](args.n_trials)
    if args.out:
        out = args.out
    elif args.expt == 'rooms':
        out = f'useful_saves/fits/rooms/{os.path.basename(os.path.normpath(data_dir))}_fits.csv'
    else:
        out = f'useful_saves/fits/{args.n_arms}a{args.n_outcomes}k_fits.csv'

    print('FITTING EMP AGENTS')
    print(f'  data_dir: {data_dir}')
    print(f'  cost: {cost}')
    print(f'  out: {out}')
    for k, v in vars(args).items():
        if k not in ('data_dir', 'cost', 'out'):
            print(f'  {k}: {v}')

    ## load participants
    data = expt['loader'](data_dir)
    df = data[expt['df_key']].copy()

    ## design constants fit_emp reads off each participant's first row. The
    ## rooms loader already sets n_arms, n_outcomes, n_AFC and expt from the data.
    if args.expt == 'arms':
        df['n_arms'] = args.n_arms
        df['n_outcomes'] = args.n_outcomes
        df['n_rooms'] = df.groupby('id')['room'].transform('nunique')
    df['n_trials'] = args.n_trials
    df['termination_arm'] = args.termination_arm
    df['cost'] = cost
    if args.alpha is not None:
        df['alpha'] = args.alpha

    print(f'Loaded {len(df)} trials from '
          f'{df["id"].nunique()} participants')

    fit_kwargs = dict(
        agent_types=args.agent_types,
        param_bounds=[args.log_ell_bounds and tuple(args.log_ell_bounds), tuple(args.inv_temp_bounds)],
        horizon=args.horizon,
        n_jobs=args.n_jobs,
        optimizer=args.optimizer,
        n_starts=args.n_starts,
        alpha_bounds=args.alpha_bounds and tuple(args.alpha_bounds),
        seed=args.seed,
        n_ell_samples=args.n_ell_samples,
        n_beta_samples=args.n_beta_samples,
    )
    if os.path.dirname(out):
        os.makedirs(os.path.dirname(out), exist_ok=True)
    stem = out[:-len('_fits.csv')] if out.endswith('_fits.csv') else os.path.splitext(out)[0]

    ## fit to all trials
    if args.mode in ('fit', 'both'):
        df_fits = fit_emp(df_ppt=df, init_t=args.init_t, verbose=True, **fit_kwargs)

        ## grid fits: full posterior grids to an npz, one set of arrays per (id, agent_type)
        if '_post' in df_fits.columns:
            posts = {}
            for pid, agent_type, post in df_fits[['id', 'agent_type', '_post']].itertuples(index=False):
                for k, v in post.items():
                    if v is not None:
                        posts[f'{pid}__{agent_type}__{k}'] = v
            np.savez_compressed(f'{stem}_posteriors.npz', **posts)
            print(f'Saved posterior grids to {stem}_posteriors.npz')
            df_fits = df_fits.drop(columns='_post')

        df_fits.to_csv(out, index=False)
        print(f'Saved {len(df_fits)} fits to {out}')

        ## quick look at how the models compare
        cols = ['nll', 'BIC'] + (['log_evidence'] if 'log_evidence' in df_fits.columns else [])
        print(df_fits.groupby('agent_type')[cols].mean())

    ## leave-one-block-out cross-validation
    if args.mode in ('loocv', 'both'):
        if 'block' not in df.columns:
            raise ValueError("--mode loocv needs a 'block' column in the data")
        df_cv = loocv_emp(df, fit_kwargs, init_t=args.init_t, verbose=True)

        ## one row per (id, agent_type): each fold's fit and held-out NLL as
        ## `{col}_block{b}` columns (b = held-out block), then the totals over folds
        fold_cols = ['log_ell', 'ell', 'beta', 'agent_alpha', 'success',
                     'train_nll', 'test_nll', 'n_test_trials']
        ## grid fits: training evidence, and the plug-in score next to the posterior predictive
        grid_cols = [c for c in ('log_evidence', 'test_nll_plugin') if c in df_cv.columns]
        fold_cols += grid_cols
        keys = ['id', 'agent_type']
        df_wide = df_cv.pivot(index=keys, columns='block', values=fold_cols)
        df_wide.columns = [f'{col}_block{b}' for col, b in df_wide.columns]
        total_cols = ['test_nll', 'n_test_trials'] + [c for c in grid_cols if c == 'test_nll_plugin']
        totals = df_cv.groupby(keys)[total_cols].sum()
        df_wide = df_wide.join(totals).reset_index()
        df_wide.to_csv(f'{stem}_loocv.csv', index=False)
        print(f'Saved {len(df_wide)} cross-validated fits to {stem}_loocv.csv')

        ## quick look: summed held-out NLL, averaged over subjects
        print(df_wide.groupby('agent_type')[['test_nll'] + total_cols[2:]].mean()
              .add_prefix('mean summed '))


if __name__ == '__main__':
    main()
