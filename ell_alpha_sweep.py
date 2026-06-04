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
    args = parser.parse_args()

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
    df_curves = enumerate_curves(n_arms=args.n_arms, n_outcomes=args.n_outcomes, n_trials=args.n_trials, alphas=args.alphas,
                                 ell_hi=args.ell_hi, ell_lo=args.ell_lo,
                                 termination_arm=args.termination_arm,
                                 n_jobs=-1, n_ell_samples=args.n_ell_samples
                                 )
    
    ## save
    df_curves.to_csv(f'useful_saves/sweep/{args.n_arms}arms_{args.n_outcomes}outcomes_{args.n_trials}trials.csv', index=False)

if __name__ == '__main__':
    main()