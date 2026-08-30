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
    parser.add_argument('--alphas', type=float, nargs='+', default=[0.25])
    parser.add_argument('--termination_arm', action='store_true')
    parser.add_argument('--contexts', type=float, nargs='+', default=None)
    parser.add_argument('--context_prior', type=float, nargs='+', default=None)
    parser.add_argument('--independent_contexts', action='store_true')
    parser.add_argument('--init_t', type=int, default=0)

    args = parser.parse_args()

    tag = ["noTermination", "Termination"][args.termination_arm]
    stem = (f'useful_saves/pareto/{args.n_arms}arms_{args.n_outcomes}outcomes_'
            f'{args.n_trials}trials_{tag}')
    if args.contexts is not None:
        stem += '_unknown_contexts'
    os.makedirs('useful_saves/pareto', exist_ok=True)

    ## run expt
    print('Running pareto sweep with parameters:')
    for k, v in vars(args).items():
        print(f'  {k}: {v}')

    df_pareto = pareto_run(
        n_arms=args.n_arms, n_outcomes=args.n_outcomes, n_trials=args.n_trials,
        alphas=args.alphas, contexts=args.contexts, context_prior=args.context_prior,
        independent_contexts=args.independent_contexts,
        termination_arm=args.termination_arm,
        init_t=args.init_t, n_jobs=args.n_jobs,
    )

    ## save
    out = f'{stem}_pareto.csv'
    print(f'Saving df_pareto to {out}')
    df_pareto.to_csv(out, index=False)


if __name__ == '__main__':
    main()
