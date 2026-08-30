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
    parser.add_argument('--n_jobs', type=int, default=-1)
    parser.add_argument('--n_ell_samples', type=int, default=100)
    parser.add_argument('--ell_lo', type=float, default=0.01)
    parser.add_argument('--ell_hi', type=float, default=100)
    parser.add_argument('--horizons', type=int, nargs='+', default=None)
    parser.add_argument('--alphas', type=float, nargs='+', default=[0.1, 0.5, 1])
    parser.add_argument('--termination_arm', action='store_true')
    parser.add_argument('--contexts', type=float, nargs='+', default=None)
    parser.add_argument('--context_prior', type=float, nargs='+', default=None)
    parser.add_argument('--init_t', type=int, default=0)
    parser.add_argument('--independent_contexts', action='store_true')
    parser.add_argument('--costs', type=float, nargs='+',
                        default=[0])

    args = parser.parse_args()

    tag = ["noTermination", "Termination"][args.termination_arm]
    stem = f'useful_saves/sweep/{args.n_arms}arms_{args.n_outcomes}outcomes_{args.n_trials}trials_{tag}'
    max_emps_path = f'{stem}_max_emps.csv'

    if args.contexts is not None:
        stem += f'_unknown_contexts'

    ## run expt
    print('Running experiment with parameters:')
    print(f'  n_arms: {args.n_arms}')
    print(f'  n_outcomes: {args.n_outcomes}')
    print(f'  n_trials: {args.n_trials}')
    print(f'  n_ell_samples: {args.n_ell_samples}')
    print(f'  horizons: {args.horizons}')
    print(f'  ell_lo: {args.ell_lo}')
    print(f'  ell_hi: {args.ell_hi}')
    print(f'  alphas: {args.alphas}')
    print(f'  termination_arm: {args.termination_arm}')
    print(f'  contexts: {args.contexts}')
    print(f'  context_prior: {args.context_prior}')
    print(f'  independent_contexts: {args.independent_contexts}')
    print(f'  costs: {args.costs}')
    print(f'  init_t: {args.init_t}')
    print(f'  n_jobs: {args.n_jobs}')


    df_curves = enumerate_curves(n_arms=args.n_arms, n_outcomes=args.n_outcomes, n_trials=args.n_trials, alphas=args.alphas,
                                 ell_hi=args.ell_hi, ell_lo=args.ell_lo,
                                 horizons=args.horizons, independent_contexts=args.independent_contexts,
                                context_prior=args.context_prior, contexts=args.contexts,
                                 termination_arm=args.termination_arm,
                                 n_jobs=args.n_jobs, n_ell_samples=args.n_ell_samples,
                                 costs=args.costs,
                                    init_t=args.init_t
                                 )

    ## save
    print(f'Saving df_curves to {stem}_ksweep.csv')
    if len(args.costs) > 1:
        df_curves.to_csv(f'{stem}_ksweep.csv', index=False)
    else:
        df_curves.to_csv(f'{stem}_{args.costs[0]}k.csv', index=False)


if __name__ == '__main__':
    main()