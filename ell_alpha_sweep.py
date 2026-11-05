from emp_utils import *
from emp_runners import *
from emp_runners import _diag_ell_grid
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
    parser.add_argument('--init_t', type=int, default=0)
    parser.add_argument('--costs', type=float, nargs='+',
                        default=[0])

    ## the diag_sweep.py prior grid of ells, instead of --ell_lo / --ell_hi
    parser.add_argument('--diag_grid', action='store_true',
                        help="Sweep the ells of diag_sweep.py's prior grid, built from the args below "
                             "and --n_ell_samples, instead of --ell_lo / --ell_hi.")
    parser.add_argument('--target', type=str, default='model', choices=['ell', 'model'])
    parser.add_argument('--ell_prior', type=str, default='lognormal', choices=['lognormal', 'loguniform'])
    parser.add_argument('--emp_type', type=str, nargs='+', default=['emp'],
                        choices=['emp', 'emp_lo', 'emp_1', 'emp_hi'])
    parser.add_argument('--log_ell_bounds', type=float, nargs=2, default=None)

    parser.add_argument('--beta', type=float, default=1.0,
                        help="Softmax inverse temperature, shared by the emp and info agents.")
    parser.add_argument('--normalise_q', nargs='?', const='prior', default=None, choices=['prior', 'analytic', 'analytic_max'],
                        help="Normalise emp values (see `leaf_q_bounds`): 'prior' (the default when given bare) divides by the "
                             "ell-prior-weighted mean of each ell's best leaf value over the task's histories; 'analytic' maps "
                             "each ell onto [0, 1] with its analytic min and max; 'analytic_max' divides every ell by "
                             "the largest analytic max, K (the number of outcomes). Info values are already normalised by the empty prior's variance.")

    args = parser.parse_args()

    tag = ["noTermination", "Termination"][args.termination_arm]
    stem = f'useful_saves/sweep/{args.n_arms}arms_{args.n_outcomes}outcomes_{args.n_trials}trials_{tag}'
    ells, ell_w0s = None, None
    if args.diag_grid:
        emp_type = args.emp_type[0] if len(args.emp_type) == 1 else tuple(args.emp_type)
        ells, ell_w0s = _diag_ell_grid(args.target, emp_type, args.n_ell_samples,
                                 args.log_ell_bounds and tuple(args.log_ell_bounds), args.ell_prior)
        stem += f'_diagGrid_{args.target}_{args.ell_prior}'
        if args.emp_type != ['emp']:
            stem += '_' + '_'.join(args.emp_type)
        if args.log_ell_bounds is not None:
            stem += f'_logEll{args.log_ell_bounds[0]}to{args.log_ell_bounds[1]}'
    if args.normalise_q:
        stem += '_normQ' + ('' if args.normalise_q == 'prior' else args.normalise_q)
    if args.beta != 1:
        stem += f'_beta{args.beta}'
    max_emps_path = f'{stem}_max_emps.csv'

    ## run expt
    print('Running experiment with parameters:')
    print(f'  n_arms: {args.n_arms}')
    print(f'  n_outcomes: {args.n_outcomes}')
    print(f'  n_trials: {args.n_trials}')
    print(f'  n_ell_samples: {args.n_ell_samples}')
    print(f'  horizons: {args.horizons}')
    print(f'  ell_lo: {args.ell_lo}')
    print(f'  ell_hi: {args.ell_hi}')
    if args.diag_grid:
        print(f'  diag grid: target={args.target}, ell_prior={args.ell_prior}, emp_type={args.emp_type}, '
              f'log_ell_bounds={args.log_ell_bounds} -> {len(ells)} ells in [{ells.min():.4g}, {ells.max():.4g}]')
    print(f'  alphas: {args.alphas}')
    print(f'  termination_arm: {args.termination_arm}')
    print(f'  costs: {args.costs}')
    print(f'  beta: {args.beta}')
    print(f'  normalise_q: {args.normalise_q}')
    print(f'  init_t: {args.init_t}')
    print(f'  n_jobs: {args.n_jobs}')


    df_curves = enumerate_curves(n_arms=args.n_arms, n_outcomes=args.n_outcomes, n_trials=args.n_trials, alphas=args.alphas,
                                 ell_hi=args.ell_hi, ell_lo=args.ell_lo,
                                 horizons=args.horizons,
                                 termination_arm=args.termination_arm,
                                 n_ell_samples=args.n_ell_samples,
                                 costs=args.costs,
                                    init_t=args.init_t,
                                 beta=args.beta,
                                 normalise_q=args.normalise_q,
                                 ells=ells,
                                 ell_w0s=ell_w0s
                                 )

    ## save
    print(f'Saving df_curves to {stem}_ksweep.csv')
    if len(args.costs) > 1:
        df_curves.to_csv(f'{stem}_ksweep.csv', index=False)
    else:
        df_curves.to_csv(f'{stem}_{args.costs[0]}k.csv', index=False)


if __name__ == '__main__':
    main()