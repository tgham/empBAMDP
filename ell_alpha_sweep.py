from emp_utils import *
from emp_runners import *
import numpy as np
import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_arms', type=int, default=2)
    parser.add_argument('--n_outcomes', type=int, default=4)
    parser.add_argument('--n_trials', type=int, default=6)
    parser.add_argument('--n_ell_samples', type=int, default=100)
    parser.add_argument('--ell_lo', type=float, default=0.01)
    parser.add_argument('--ell_hi', type=float, default=100)
    parser.add_argument('--horizon', type=int, default=None)
    parser.add_argument('--alphas', type=float, nargs='+', default=[0.1, 0.5, 1])
    parser.add_argument('--termination_arm', action='store_true')
    parser.add_argument('--contexts', type=float, nargs='+', default=None)
    parser.add_argument('--context_prior', type=float, nargs='+', default=None)
    parser.add_argument('--ks', type=float, nargs='+',
                        default=[round(x, 2) for x in np.arange(0.01, 0.101, 0.01)])

    args = parser.parse_args()

    tag = ["noTermination", "Termination"][args.termination_arm]
    stem = f'useful_saves/sweep/{args.n_arms}arms_{args.n_outcomes}outcomes_{args.n_trials}trials_{tag}'
    max_emps_path = f'{stem}_max_emps.csv'

    ## run smallest k first so a k==0 run can build df_max before the costed ks need it
    ks = sorted(args.ks)

    ## df_max: load it now unless we'll compute it ourselves during a k==0 run
    if 0 in ks or 0.0 in ks:
        df_max = None
    else:
        print(f'Loading df_max from {max_emps_path}')
        df_max = pd.read_csv(max_emps_path)

    ## run expt
    print('Running experiment with parameters:')
    print(f'  n_arms: {args.n_arms}')
    print(f'  n_outcomes: {args.n_outcomes}')
    print(f'  n_trials: {args.n_trials}')
    print(f'  n_ell_samples: {args.n_ell_samples}')
    print(f'  ell_lo: {args.ell_lo}')
    print(f'  ell_hi: {args.ell_hi}')
    print(f'  alphas: {args.alphas}')
    print(f'  termination_arm: {args.termination_arm}')
    print(f'  contexts: {args.contexts}')
    print(f'  context_prior: {args.context_prior}')
    print(f'  ks: {ks}')

    for k in ks:
        print(f'\n=== k = {k} ===')
        df_curves = enumerate_curves(n_arms=args.n_arms, n_outcomes=args.n_outcomes, n_trials=args.n_trials, alphas=args.alphas,
                                     ell_hi=args.ell_hi, ell_lo=args.ell_lo,
                                    context_prior=args.context_prior, contexts=args.contexts,
                                     termination_arm=args.termination_arm,
                                     n_jobs=-1, n_ell_samples=args.n_ell_samples,
                                     df_max=df_max, k=k
                                     )

        ## save
        df_curves.to_csv(f'{stem}_{k}k.csv', index=False)

        ## the cost-free run is the source of truth for max empowerment: build
        ## df_max from it (per the empBAMDP.py recipe) and reuse for costed ks
        if k == 0:
            df_max = (df_curves.groupby(['alpha', 'ell'])
                      .apply(lambda x: x.loc[x['current_emp'].idxmax()])
                      [['ell', 'alpha', 'current_emp', 'history_str']]
                      .reset_index(drop=True))
            df_max.to_csv(max_emps_path, index=False)
            print(f'Saved df_max to {max_emps_path}')

if __name__ == '__main__':
    main()