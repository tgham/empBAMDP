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
    parser.add_argument('--alphas', type=float, nargs='+', default=[0.25])
    parser.add_argument('--termination_arm', action='store_true')
    parser.add_argument('--init_t', type=int, default=0)
    parser.add_argument('--costs', type=float, nargs='+', default=[0])

    ## diagnosticity-specific: the ell prior and the choice policy
    parser.add_argument('--n_ell_samples', type=int, default=200)
    parser.add_argument('--temp_emp', type=float, default=1.0)
    parser.add_argument('--temp_info', type=float, default=1.0)
    parser.add_argument('--prior_mu', type=float, default=0.0)
    parser.add_argument('--prior_sigma', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--ell_prior', type=str, default='lognormal',
                        choices=['lognormal', 'truncnorm'],
                        help="'truncnorm' uses the parameter-recovery prior, log ell ~ N(0, 2) over "
                             "--emp_type's slice of --log_ell_bounds, ignoring --prior_mu/--prior_sigma.")
    parser.add_argument('--emp_type', type=str, nargs='+', default=['emp'],
                        choices=['emp', 'emp_lo', 'emp_1', 'emp_hi'],
                        help="Several types (e.g. emp_lo emp_1 emp_hi) split emp into "
                             "one model per type; --target model only.")
    parser.add_argument('--log_ell_bounds', type=float, nargs=2, default=None,
                        help='truncation of the truncnorm prior (default: none; the grid stops at +/- 4 sigma)')
    parser.add_argument('--target', type=str, default='ell',
                        choices=['ell', 'model'])
    parser.add_argument('--p_model', type=float, nargs='+', default=None,
                        help="Prior over the models: one per --emp_type, then info. "
                             "Default uniform.")
    
    parser.add_argument('--expt', type=str, default='arms', choices=['arms','rooms'])
    if parser.parse_known_args()[0].expt == 'rooms':
        parser.add_argument('--n_AFC', type=int, default=2)
        parser.add_argument('--n_room_samples', type=int, default=None)

    args = parser.parse_args()

    tag = ["noTermination", "Termination"][args.termination_arm]
    stem = (f'useful_saves/diag/{args.expt}/{args.n_arms}arms_{args.n_outcomes}outcomes_'
            f'{args.n_trials}trials_{tag}')
    if args.expt=='rooms':
        stem += f'_{args.n_AFC}AFC'
    stem += f'_{args.target}'
    stem += f'_{args.ell_prior}'
    if args.emp_type != ['emp']:
        stem += '_' + '_'.join(args.emp_type)
    os.makedirs('useful_saves/diag', exist_ok=True)

    ## run expt
    print('Running diagnosticity sweep with parameters:')
    for k, v in vars(args).items():
        print(f'  {k}: {v}')

    df_diag = enumerate_diagnosticity(
        n_arms=args.n_arms, n_outcomes=args.n_outcomes, n_trials=args.n_trials,
        alphas=args.alphas,
        termination_arm=args.termination_arm, temp_emp=args.temp_emp,
        horizons=args.horizons, costs=args.costs,
        n_ell_samples=args.n_ell_samples,
        prior_mu=args.prior_mu, prior_sigma=args.prior_sigma, seed=args.seed,
        ell_prior=args.ell_prior,
        emp_type=args.emp_type[0] if len(args.emp_type) == 1 else tuple(args.emp_type),
        log_ell_bounds=args.log_ell_bounds and tuple(args.log_ell_bounds),
        init_t=args.init_t, n_jobs=args.n_jobs,
        target=args.target, temp_info=args.temp_info,
        p_model=None if args.p_model is None else tuple(args.p_model),
        expt=args.expt, n_AFC=getattr(args, 'n_AFC', 2),
        n_room_samples=getattr(args, 'n_room_samples', None)
    )

    ## save
    out = f'{stem}_diag.csv'
    print(f'Saving df_diag to {out}')
    df_diag.to_csv(out, index=False)


if __name__ == '__main__':
    main()
