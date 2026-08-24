from emp_utils import *
from emp_runners import *
import numpy as np
import pandas as pd
import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_arms', type=int, default=2)
    parser.add_argument('--n_outcomes', type=int, default=4)
    parser.add_argument('--n_trials', type=int, default=6)
    parser.add_argument('--n_jobs', type=int, default=-1)
    parser.add_argument('--horizons', type=int, nargs='+', default=None)
    parser.add_argument('--alphas', type=float, nargs='+', default=[0.1, 0.5, 1])
    parser.add_argument('--termination_arm', action='store_true')
    parser.add_argument('--contexts', type=float, nargs='+', default=None)
    parser.add_argument('--context_prior', type=float, nargs='+', default=None)
    parser.add_argument('--independent_contexts', action='store_true')
    parser.add_argument('--init_t', type=int, default=0)
    parser.add_argument('--costs', type=float, nargs='+', default=[0])

    ## diagnosticity-specific: the ell prior and the choice policy
    parser.add_argument('--n_samples', type=int, default=200)
    parser.add_argument('--temp', type=float, default=1.0)
    parser.add_argument('--prior_mu', type=float, default=0.0)
    parser.add_argument('--prior_sigma', type=float, default=1.0)
    parser.add_argument('--sampling', type=str, default='quantile',
                        choices=['quantile', 'random'])
    parser.add_argument('--seed', type=int, default=None)

    args = parser.parse_args()

    tag = ["noTermination", "Termination"][args.termination_arm]
    stem = (f'useful_saves/diag/{args.n_arms}arms_{args.n_outcomes}outcomes_'
            f'{args.n_trials}trials_{tag}')
    if args.contexts is not None:
        stem += '_unknown_contexts'
    os.makedirs('useful_saves/diag', exist_ok=True)

    ## run expt
    print('Running diagnosticity sweep with parameters:')
    for k, v in vars(args).items():
        print(f'  {k}: {v}')

    df_diag = enumerate_diagnosticity(
        n_arms=args.n_arms, n_outcomes=args.n_outcomes, n_trials=args.n_trials,
        alphas=args.alphas, contexts=args.contexts, context_prior=args.context_prior,
        independent_contexts=args.independent_contexts,
        termination_arm=args.termination_arm, temp=args.temp,
        horizons=args.horizons, costs=args.costs,
        n_samples=args.n_samples, prior_mu=args.prior_mu,
        prior_sigma=args.prior_sigma, sampling=args.sampling, seed=args.seed,
        init_t=args.init_t, n_jobs=args.n_jobs,
    )

    ## save
    out = f'{stem}_diag.csv'
    print(f'Saving df_diag to {out}')
    df_diag.to_csv(out, index=False)


if __name__ == '__main__':
    main()
