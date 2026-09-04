"""Fit the empowerment / info-seeking agents to participants' sampling data.

Lifted out of the `Fitting > Fit emp agent` cell of empBAMDP.ipynb so the fits
can be run headlessly (e.g. under run.sh) rather than in the notebook.

    python fit_model.py --data_dir expt/Experiment4/data/3a4k --n_arms 3

Writes one row per (subject_id, agent_type) to
`useful_saves/fits/{n_arms}a{n_outcomes}k_fits.csv`, which is what the
downstream notebook cells (BIC comparison, ell histograms, simulating agents
under the fitted ells) read back in.
"""

from emp_runners import fit_emp
from load_data import load_directory
import pandas as pd
import numpy as np
import argparse
import os


def fit_data(df, agent_types, ell_bounds, temp_bounds, horizon=3, init_t=1,
             n_jobs=-1, verbose=True):
    """Fit `agent_types` to one dataframe of participant trials.

    `df` must already carry the design columns fit_emp reads off the first row
    of each participant (n_arms, n_outcomes, n_trials, termination_arm, cost,
    alpha); see `prepare_df`.
    """
    df_fits = fit_emp(
        df_ppt=df,
        agent_types=agent_types,
        param_bounds=[ell_bounds, temp_bounds],
        horizon=horizon,
        init_t=init_t,
        n_jobs=n_jobs,
        verbose=verbose,
    )

    ## the info-seeking agent has no ell, so don't leave the optimiser's
    ## placeholder in the column
    df_fits.loc[df_fits['agent_type'] == 'info', 'ell'] = None

    return df_fits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=None,
                        help='directory of participant .json files; defaults to '
                             'expt/Experiment4/data/{n_arms}a{n_outcomes}k')
    parser.add_argument('--n_arms', type=int, default=3)
    parser.add_argument('--n_outcomes', type=int, default=4)
    parser.add_argument('--n_trials', type=int, default=8)
    parser.add_argument('--cost', type=float, default=None,
                        help='sampling cost; defaults to 1/(n_trials+1), as in config.js')
    parser.add_argument('--alpha', type=float, default=None,
                        help='override the Dirichlet alpha logged per participant')
    parser.add_argument('--no_termination_arm', dest='termination_arm',
                        action='store_false',
                        help='the task has no terminate action (it does by default)')
    parser.add_argument('--ell_bounds', type=float, default=(0.01, 10), nargs=2)
    parser.add_argument('--temp_bounds', type=float, default=(0.001, 0.1), nargs=2)
    parser.add_argument('--horizon', type=int, default=3)
    parser.add_argument('--init_t', type=int, default=1,
                        help='leading trials used to warm the belief without being scored')
    parser.add_argument('--agent_types', nargs='+', default=['emp', 'info'],
                        choices=['emp', 'emp_lo', 'emp_1', 'emp_hi', 'info'],
                        help="models to fit, one row per subject each. Pass "
                             "'emp_lo emp_1 emp_hi' in place of 'emp' to split "
                             "the empowerment agent by ell<1, ell=1 and ell>1.")
    parser.add_argument('--n_jobs', type=int, default=-1)
    parser.add_argument('--out', type=str, default=None,
                        help='output csv; defaults to '
                             'useful_saves/fits/{n_arms}a{n_outcomes}k_fits.csv')

    args = parser.parse_args()

    data_dir = args.data_dir or f'expt/Experiment4/data/{args.n_arms}a{args.n_outcomes}k'
    cost = args.cost if args.cost is not None else 1 / (args.n_trials + 1)
    out = args.out or (f'useful_saves/fits/{args.n_arms}a{args.n_outcomes}k_fits.csv')

    print('FITTING EMP AGENTS')
    print(f'  data_dir: {data_dir}')
    print(f'  cost: {cost}')
    print(f'  out: {out}')
    for k, v in vars(args).items():
        if k not in ('data_dir', 'cost', 'out'):
            print(f'  {k}: {v}')

    ## load participants
    data = load_directory(data_dir)
    
    df = data['sample'].copy()
    df['n_arms'] = args.n_arms
    df['n_outcomes'] = args.n_outcomes
    df['n_trials'] = args.n_trials
    df['termination_arm'] = args.termination_arm
    df['cost'] = cost
    if args.alpha is not None:
        df['alpha'] = args.alpha
    df['n_rooms'] = df.groupby('subject_id')['room'].transform('nunique')

    print(f'Loaded {len(df)} trials from '
          f'{df["subject_id"].nunique()} participants')

    ## fit
    df_fits = fit_emp(
        df_ppt=df,
        agent_types=args.agent_types,
        param_bounds=[tuple(args.ell_bounds), tuple(args.temp_bounds)],
        horizon=args.horizon,
        init_t=args.init_t,
        n_jobs=args.n_jobs,
        verbose=args.verbose,
    )
    df_fits.loc[df_fits['agent_type'] == 'info', 'ell'] = None

    ## save
    if os.path.dirname(out):
        os.makedirs(os.path.dirname(out), exist_ok=True)
    df_fits.to_csv(out, index=False)
    print(f'Saved {len(df_fits)} fits to {out}')

    ## quick look at how the models compare
    print(df_fits.groupby('agent_type')[['nll', 'BIC']].mean())


if __name__ == '__main__':
    main()
