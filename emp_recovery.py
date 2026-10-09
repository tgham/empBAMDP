from emp_runners import gen_arms, gen_rooms, fit_emp, emp_ell_bounds, ELL_TRUNCNORM
from emp_utils import canonical_states, canonical_count_matrix, array_to_hist, canonicalise_histories, top_n_pareto, pareto_front_idx
import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib
from scipy.stats import truncnorm
import argparse
import ast
import os

def load_diag_histories(args, term):
    """The `args.n_rooms` most diagnostic histories for this design.

    The diagnosticity tables sweep alpha / horizon / cost, so they must be
    filtered down to the condition being recovered before ranking -- otherwise
    the top of the table is just whichever alpha happens to score highest, and
    the same history reappears once per condition.

    `--diag_target model` ranks by how well a history separates the empowerment
    agent from the info-seeking one (model recovery); `--diag_target ell` ranks
    by how well it separates ells (parameter recovery).

    For `--expt rooms` each row is a tuple of `args.n_AFC` histories to choose
    between, ranked and de-duplicated by `pair_str`.

    Returns a list of histories, each `(((arm, outcome), count), ...)`, or for
    rooms a list of tuples of them.
    """
    rooms = args.expt == 'rooms'
    path = (f'useful_saves/diag/{args.expt}/{args.n_arms}arms_{args.n_outcomes}outcomes_'
            f'{args.n_trials}trials_{term}_{args.n_AFC}AFC_{args.diag_target}_{args.ell_prior}_diag.csv')
    df_diag = pd.read_csv(path)

    ## keep only the rows generated under this run's design (rooms are valued at their leaf, so no horizon)
    sel = df_diag
    conds = (('alpha', args.alpha), ('cost', args.cost)) if rooms else \
            (('alpha', args.alpha), ('horizon', args.horizon), ('cost', args.cost))
    for col, val in conds:
        if col in sel.columns:
            match = sel.loc[np.isclose(sel[col].astype(float), float(val))]
            if match.empty:
                raise ValueError(
                    f'{path} has no rows with {col}={val} '
                    f'(available: {sorted(sel[col].unique())})'
                )
            sel = match

    ## one row per history (or tuple of histories), keeping its best score on the first objective
    key = 'pair_str' if rooms else 'history'
    missing = [c for c in args.diag_cols if c not in sel.columns]
    if missing:
        raise ValueError(f'{path} has no column(s) {missing} (available: {list(sel.columns)})')
    sel = sel.sort_values(args.diag_cols[0], ascending=False).drop_duplicates(key)

    ## the top_n most diagnostic by Pareto rank over diag_cols (or, by default, just the
    ## first Pareto layer), cycled until n_rooms is reached
    if args.top_n is None:
        if len(args.diag_cols) == 1:
            raise ValueError('--top_n is required with a single --diag_cols objective '
                             '(its first Pareto layer is just the tied maxima)')
        sel = sel.dropna(subset=args.diag_cols)
        sel = sel.iloc[pareto_front_idx(*(sel[c].to_numpy() for c in args.diag_cols))]
    else:
        if len(sel) < args.top_n:
            raise ValueError(
                f'{path} only has {len(sel)} distinct histories for this condition, '
                f'but --top_n={args.top_n} were requested'
            )
        sel = top_n_pareto(sel, n=args.top_n, cols=args.diag_cols)
    sel = sel.iloc[np.arange(args.n_rooms) % len(sel)].reset_index(drop=True)

    if rooms:
        n_AFC = sum(c.startswith('history_') and c[len('history_'):].isdigit() for c in sel.columns)
        if n_AFC != args.n_AFC:
            raise ValueError(f'{path} has {n_AFC}-AFC tuples, but --n_AFC={args.n_AFC} was requested')
        return [tuple(ast.literal_eval(row[f'history_{r}']) for r in range(n_AFC))
                for _, row in sel.iterrows()]

    return [ast.literal_eval(h) for h in sel['history']]


def main():
    
    ## init expt
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_arms', type=int, default=2)
    parser.add_argument('--n_outcomes', type=int, default=4)
    parser.add_argument('--n_trials', type=int, default=8)
    parser.add_argument('--n_rooms', type=int, default=25)
    parser.add_argument('--alpha', type=float, default=0.4)
    parser.add_argument('--ell_bounds', type=float, default=(0.01, 10), nargs=2)
    parser.add_argument('--ell_prior', choices=['loguniform', 'truncnorm'], default='truncnorm',
                        help="Distribution to sample generative ells from, within --ell_bounds.")
    parser.add_argument('--temp_bounds', type=float, default=(0.001, 0.2), nargs=2)
    parser.add_argument('--cost', type=float, default=0.0)
    parser.add_argument('--horizon', type=int, default=1)
    parser.add_argument('--init_t', type=int, default=1)
    parser.add_argument('--n_sims', type=int, default=100)
    parser.add_argument('--n_jobs', type=int, default=-1)
    parser.add_argument('--agent_types', nargs='+', default=[
        'emp',
        'info'
                                                              ],
                        choices=['emp', 'emp_lo', 'emp_1', 'emp_hi', 'info'],
                        help="Models to generate from and fit. Pass "
                             "'emp_lo emp_1 emp_hi' in place of 'emp' to split "
                             "the empowerment agent by ell<1, ell=1 and ell>1.")
    parser.add_argument('--gen_data', action='store_true')
    parser.add_argument('--seed', type=int, default=None,
                        help='Seed for the generative parameters (default: fresh entropy).')
    parser.add_argument('--skip_recovery', action='store_true')
    parser.add_argument('--termination_arm', action='store_true')

    
    ### expt type
    parser.add_argument('--expt', type=str, default='arms', choices=['rooms', 'arms'])
    
    ## horizons task
    parser.add_argument('--preset_histories', action='store_true')
    parser.add_argument('--n_subseq_trials', type=int, default=1)

    ## room task
    parser.add_argument('--n_AFC', type=int, default=2)

    parser.add_argument('--diag_target', choices=['model', 'ell'], default='model')
    parser.add_argument('--diag_cols', nargs='+', default=['mi_emp'],
                        help='Diagnosticity columns to maximise jointly (Pareto layers) when picking preset histories.')
    parser.add_argument('--top_n', type=int, default=None,
                        help='Number of distinct preset histories to pick, by Pareto rank; repeated cyclically '
                             'up to --n_rooms. Required for a single objective; with several, '
                             'defaults to the whole first Pareto layer.')

    args = parser.parse_args()

    term = ["noTermination", "Termination"][args.termination_arm]

    ## rooms task: every choice is between preset histories, and all of it is scored
    rooms = args.expt == 'rooms'
    if rooms:
        args.preset_histories = True
    init_t = 0 if args.preset_histories else args.init_t

    ## pathname for saving
    if rooms:
        stem = (f'useful_saves/recovery/{args.expt}/{args.n_arms}arms_{args.n_outcomes}outcomes_'
                f'{args.n_trials}trials_{args.n_sims}sims_'
                f'{args.alpha}alpha_{args.cost}cost_{term}_{args.n_AFC}AFC_{args.n_rooms}rooms_{args.diag_target}target')
    else:
        stem = (f'useful_saves/recovery/{args.expt}/{args.n_arms}arms_{args.n_outcomes}outcomes_'
                f'{args.n_trials}trials_{args.n_sims}sims_{args.horizon}h_'
                f'{args.alpha}alpha_{args.cost}cost_{term}')
        if args.preset_histories:
            stem += f'_preset_{args.n_rooms}rooms_{args.n_subseq_trials}subseq_{args.diag_target}target'
    if args.preset_histories:
        stem += f'_top{args.top_n or "front"}_{"-".join(args.diag_cols)}'
    stem += f'_{args.ell_prior}'

    if args.gen_data:
        print('EMP RECOVERY')
        print(f'Generating {args.n_sims} datasets with following settings:')
        print(f'  - Number of arms: {args.n_arms}')
        print(f'  - Number of outcomes: {args.n_outcomes}')
        print(f'  - Number of trials: {args.n_trials}')
        print(f'  - Expt type: {args.expt}')
        print(f'  - Number of rooms: {args.n_rooms}')
        if rooms:
            print(f'  - N AFC: {args.n_AFC}')
        print(f'  - Alpha: {args.alpha}')
        print(f'  - Horizon: {args.horizon}')
        print(f'  - Cost: {args.cost}')
        print(f'  - Initial trial: {init_t}')
        print(f'  - Termination arm: {args.termination_arm}')
        print(f'  - Agent types: {args.agent_types}')
        print(f'  - Preset histories: {args.preset_histories}')
        print(f'  - N subsequent trials: {args.n_subseq_trials}')

        ## load preset histories with the largest diagnosticity scores
        if args.preset_histories:
            diag_histories = load_diag_histories(args, term)
            print(f'  - Diagnosticity target: {args.diag_target}')
            print(f'  - Pareto objectives: {args.diag_cols}, '
                  f'{f"top {args.top_n}" if args.top_n else "first layer"} '
                  f'({len(set(diag_histories))} distinct) cycled to {args.n_rooms} rooms')
            if rooms:
                print(f'  - Loaded {len(diag_histories)} diagnostic {args.n_AFC}-AFC tuples '
                      f'(lengths {sorted({sum(c for _, c in h) for hs in diag_histories for h in hs})})')
            else:
                print(f'  - Loaded {len(diag_histories)} diagnostic histories '
                      f'(lengths {sorted({sum(c for _, c in h) for h in diag_histories})})')
        else:
            diag_histories = None

        # Define the worker function
        def _gen_single_sim(sim_id, args, agent_type, rng):
            
            ## Sample ell over this type's slice of the bounds (the whole
            ## range for 'emp', ell=1 for 'emp_1')
            if agent_type == 'info':
                ell = None
            else:
                lo, hi = emp_ell_bounds(agent_type, args.ell_bounds)
                if lo == hi:
                    ell = lo
                elif args.ell_prior == 'loguniform':
                    ell = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
                else:
                    ## truncated normal, (loc, scale) per agent type
                    loc, scale = ELL_TRUNCNORM[agent_type]
                    ell = float(truncnorm.rvs((lo - loc) / scale, (hi - loc) / scale, loc=loc, scale=scale,
                                              random_state=rng))
            temp = rng.uniform(*args.temp_bounds)


            # Generate data
            if rooms:
                sim_tmp = gen_rooms(
                    n_arms=args.n_arms,
                    n_outcomes=args.n_outcomes,
                    n_trials=args.n_trials,
                    n_rooms=args.n_rooms,
                    alpha=args.alpha,
                    ell=ell,
                    cost=args.cost,
                    temp=temp,
                    termination_arm=args.termination_arm,
                    diag_histories=diag_histories,
                )
            else:
                sim_tmp = gen_arms(
                    n_arms=args.n_arms,
                    n_outcomes=args.n_outcomes,
                    n_trials=args.n_trials,
                    n_rooms=args.n_rooms,
                    alpha=args.alpha,
                    ell=ell,
                    horizon=args.horizon,
                    cost = args.cost,
                    temp=temp,
                    termination_arm=args.termination_arm,

                    ## horizons task
                    diag_histories=diag_histories,
                    n_subseq_trials=args.n_subseq_trials
                )
            sim_tmp['subject_id'] = [sim_id] * len(sim_tmp['room'])
            sim_tmp['agent_type'] = [agent_type] * len(sim_tmp['room'])

            return sim_tmp

        ## parallellise
        agent_type_ids = []
        for agent_type in args.agent_types:
            agent_type_ids += [agent_type] * args.n_sims
        n_sims_total = len(agent_type_ids)

        ## an independent generator per sim, passed in explicitly: a pickled
        ## scipy distribution carries a frozen copy of its random state, so
        ## without random_state every joblib task would draw the same ell
        rngs = [np.random.default_rng(ss) for ss in np.random.SeedSequence(args.seed).spawn(n_sims_total)]
        with tqdm_joblib(tqdm(desc="Generating datasets", total=n_sims_total, ncols=100, unit='sim', mininterval=1)):
            results = Parallel(n_jobs=args.n_jobs)(
                delayed(_gen_single_sim)(sim_id, args, agent_type, rngs[sim_id])
                for sim_id, agent_type in enumerate(agent_type_ids)
            )

        ## each results is a dictionary. we now need to convert each to a DataFrame and concatenate them into a single DataFrame.
        df_sim = pd.concat([pd.DataFrame.from_dict(res) for res in results], ignore_index=True)
        print(f"Generated {len(df_sim)} rows of data.")

        ## add other useful info
        df_sim['n_arms'] = args.n_arms
        df_sim['n_outcomes'] = args.n_outcomes
        df_sim['n_trials'] = args.n_trials
        df_sim['n_rooms'] = args.n_rooms
        df_sim['alpha'] = args.alpha
        df_sim['termination_arm'] = args.termination_arm
        df_sim['cost'] = args.cost
        df_sim['expt'] = args.expt
        if rooms:
            df_sim['n_AFC'] = args.n_AFC

        # Reorder columns so that 'subject_id' is first
        cols = df_sim.columns.tolist()
        cols = ['subject_id'] + [c for c in cols if c != 'subject_id']
        df_sim = df_sim[cols]

        ## canonicalise histories (rooms histories are already canonical, from the diag table)
        if not rooms:
            df_sim = canonicalise_histories(df_sim, args.n_arms, args.n_outcomes)

        ## Save
        print('saving...')
        os.makedirs(os.path.dirname(stem), exist_ok=True)
        path = f'{stem}.csv'
        df_sim.to_csv(path, index=False)

        print(f"Saved {len(df_sim)} rows to {path}")

    
    ## or preload existing data
    else:
        path = f'{stem}.csv'
        df_sim = pd.read_csv(path)


    ## fit data
    if not args.skip_recovery:
        param_bounds = [
            args.ell_bounds,
            args.temp_bounds
        ]
        print('fitting')
        df_fits = fit_emp(
            df_ppt=df_sim,
            agent_types=args.agent_types,
            param_bounds=param_bounds,
            horizon=args.horizon,
            init_t=init_t,
            n_jobs=args.n_jobs,
            verbose=True
        )

        ## add the generative params back in 
        for sim in range(len(df_sim['subject_id'].unique())):
            df_fits.loc[df_fits['subject_id']==sim, 'gen_agent_type'] = df_sim.loc[df_sim['subject_id']==sim, 'agent_type'].iloc[0]
            df_fits.loc[df_fits['subject_id']==sim, 'gen_ell'] = df_sim.loc[df_sim['subject_id']==sim, 'gen_ell'].iloc[0]
            df_fits.loc[df_fits['subject_id']==sim, 'gen_temp'] = df_sim.loc[df_sim['subject_id']==sim, 'gen_temp'].iloc[0]

        ## save fits
        path = f'{stem}_fits.csv'
        df_fits.to_csv(path, index=False)
        print(f"Saved fits to {path}")

if __name__ == '__main__':
    main()