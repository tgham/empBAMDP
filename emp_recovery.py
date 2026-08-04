from emp_runners import gen_emp, fit_emp
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse

def main():
    
    ## init expt
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_arms', type=int, default=2)
    parser.add_argument('--n_outcomes', type=int, default=4)
    parser.add_argument('--n_trials', type=int, default=8)
    parser.add_argument('--n_rooms', type=int, default=25)
    parser.add_argument('--alpha', type=float, default=0.4)
    parser.add_argument('--ell_bounds', type=float, default=(0.01, 5), nargs=2)
    parser.add_argument('--temp_bounds', type=float, default=(0.01, 5), nargs=2)
    parser.add_argument('--h', type=int, default=1)
    parser.add_argument('--init_t', type=int, default=1)
    parser.add_argument('--n_sims', type=int, default=100)
    parser.add_argument('--n_jobs', type=int, default=-1)
    parser.add_argument('--agent_types', nargs='+', default=['emp', 'info'])
    parser.add_argument('--gen_data', action='store_true')
    parser.add_argument('--termination_arm', action='store_true')

    args = parser.parse_args()

    if args.gen_data:
        df_sim = pd.DataFrame()
        print('generating '+str(args.n_sims)+' datasets')
        for sim in tqdm(range(args.n_sims)):

            ## sample new params from uniform
            ell = np.random.uniform(*args.ell_bounds)
            temp = np.random.uniform(*args.temp_bounds)

            ## generate data
            df_sim_tmp = gen_emp(n_arms=args.n_arms, n_outcomes=args.n_outcomes, n_trials=args.n_trials, n_rooms=args.n_rooms, alpha=args.alpha, ell=ell, h=args.h, temp=temp, termination_arm=args.termination_arm)
            df_sim_tmp['subject_id'] = sim 
            cols = df_sim_tmp.columns.tolist()
            cols = ['subject_id'] + [c for c in cols if c != 'subject_id']
            df_sim_tmp = df_sim_tmp[cols]
            df_sim = pd.concat([df_sim, df_sim_tmp], ignore_index=True)

        ## save
        path = f'useful_saves/recovery/{args.n_arms}arms_{args.n_outcomes}outcomes_{args.n_trials}trials_{args.n_sims}sims_{args.h}h_{args.alpha}alpha_{args.termination_arm}termination.csv'
        df_sim.to_csv(path, index=False)
    else:
        path = f'useful_saves/recovery/{args.n_arms}arms_{args.n_outcomes}outcomes_{args.n_trials}trials_{args.n_sims}sims_{args.h}h_{args.alpha}alpha_{args.termination_arm}termination.csv'
        df_sim = pd.read_csv(path)


    ## fit data
    param_bounds = [
        args.ell_bounds,
        args.temp_bounds
    ]
    print('fitting')
    df_fits = fit_emp(
        df_ppt=df_sim,
        agent_types=args.agent_types,
        param_bounds=param_bounds,

        horizon=args.h,
        init_t=args.init_t,
        n_jobs=args.n_jobs,
        verbose=True
    )

    ## add the generative params back in 
    for sim in range(n_sims):
        df_fits.loc[df_fits['subject_id']==sim, 'gen_ell'] = df_sim.loc[df_sim['subject_id']==sim, 'gen_ell'].iloc[0]
        df_fits.loc[df_fits['subject_id']==sim, 'gen_temp'] = df_sim.loc[df_sim['subject_id']==sim, 'gen_temp'].iloc[0]


if __name__ == '__main__':
    main()