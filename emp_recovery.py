from emp_runners import gen_emp, fit_emp
from emp_utils import canonical_states, canonical_count_matrix, array_to_hist, canonicalise_histories
import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib
import argparse

def main():
    
    ## init expt
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_arms', type=int, default=2)
    parser.add_argument('--n_outcomes', type=int, default=4)
    parser.add_argument('--n_trials', type=int, default=8)
    parser.add_argument('--n_rooms', type=int, default=25)
    parser.add_argument('--alpha', type=float, default=0.4)
    parser.add_argument('--ell_bounds', type=float, default=(0.01, 10), nargs=2)
    parser.add_argument('--temp_bounds', type=float, default=(0.01, 0.4), nargs=2)
    parser.add_argument('--cost', type=float, default=0.0)
    parser.add_argument('--horizon', type=int, default=1)
    parser.add_argument('--init_t', type=int, default=1)
    parser.add_argument('--n_sims', type=int, default=100)
    parser.add_argument('--n_jobs', type=int, default=-1)
    parser.add_argument('--agent_types', nargs='+', default=[
        'emp',
        'info'
                                                              ])
    parser.add_argument('--gen_data', action='store_true')
    parser.add_argument('--termination_arm', action='store_true')

    args = parser.parse_args()

    if args.gen_data:
        print('EMP RECOVERY')
        print(f'Generating {args.n_sims} datasets with following settings:')
        print(f'  - Number of arms: {args.n_arms}')
        print(f'  - Number of outcomes: {args.n_outcomes}')
        print(f'  - Number of trials: {args.n_trials}')
        print(f'  - Number of rooms: {args.n_rooms}')
        print(f'  - Alpha: {args.alpha}')
        print(f'  - Horizon: {args.horizon}')
        print(f'  - Cost: {args.cost}')
        print(f'  - Initial trial: {args.init_t}')
        print(f'  - Termination arm: {args.termination_arm}')
        print(f'  - Agent types: {args.agent_types}')

        # Define the worker function
        def _gen_single_sim(sim_id, args, agent_type):
            
            ## Sample parameters 
            if agent_type == 'emp':
                # ell = np.random.uniform(*args.ell_bounds)
                ell = np.exp(np.random.uniform(np.log(args.ell_bounds[0]), np.log(args.ell_bounds[1]))) ## sample ell from a log-uniform distribution
            elif agent_type == 'info':
                ell = None
            temp = np.random.uniform(*args.temp_bounds)

            # Generate data
            sim_tmp = gen_emp(
                n_arms=args.n_arms,
                n_outcomes=args.n_outcomes,
                n_trials=args.n_trials,
                n_rooms=args.n_rooms,
                alpha=args.alpha,
                ell=ell,
                horizon=args.horizon,
                cost = args.cost,
                temp=temp,
                termination_arm=args.termination_arm
            )
            sim_tmp['subject_id'] = [sim_id] * len(sim_tmp['room'])
            sim_tmp['agent_type'] = [agent_type] * len(sim_tmp['room'])

            return sim_tmp

        ## parallellise
        agent_type_ids = []
        for agent_type in args.agent_types:
            agent_type_ids += [agent_type] * args.n_sims
        n_sims_total = len(agent_type_ids)
        with tqdm_joblib(tqdm(desc="Generating datasets", total=n_sims_total, ncols=100, unit='sim', mininterval=1)):
            results = Parallel(n_jobs=args.n_jobs)(
                delayed(_gen_single_sim)(sim_id, args, agent_type)
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

        # Reorder columns so that 'subject_id' is first
        cols = df_sim.columns.tolist()
        cols = ['subject_id'] + [c for c in cols if c != 'subject_id']
        df_sim = df_sim[cols]

        ## canonicalise histories (memoised on the count matrix -- see emp_utils)
        df_sim = canonicalise_histories(df_sim, args.n_arms, args.n_outcomes)

        ## Save
        print('saving...')
        term = ["noTermination", "Termination"][args.termination_arm]
        path = f'useful_saves/recovery/{args.n_arms}arms_{args.n_outcomes}outcomes_{args.n_trials}trials_{args.n_sims}sims_{args.horizon}h_{args.alpha}alpha_{term}.csv'
        df_sim.to_csv(path, index=False)

        print(f"Saved {len(df_sim)} rows to {path}")

    
    ## or preload existing data
    else:
        term = ["noTermination", "Termination"][args.termination_arm]
        path = f'useful_saves/recovery/{args.n_arms}arms_{args.n_outcomes}outcomes_{args.n_trials}trials_{args.n_sims}sims_{args.horizon}h_{args.alpha}alpha_{term}.csv'
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
        horizon=args.horizon,
        init_t=args.init_t,
        n_jobs=args.n_jobs,
        verbose=True
    )

    ## add the generative params back in 
    for sim in range(len(df_sim['subject_id'].unique())):
        df_fits.loc[df_fits['subject_id']==sim, 'gen_agent_type'] = df_sim.loc[df_sim['subject_id']==sim, 'agent_type'].iloc[0]
        df_fits.loc[df_fits['subject_id']==sim, 'gen_ell'] = df_sim.loc[df_sim['subject_id']==sim, 'gen_ell'].iloc[0]
        df_fits.loc[df_fits['subject_id']==sim, 'gen_temp'] = df_sim.loc[df_sim['subject_id']==sim, 'gen_temp'].iloc[0]

    ## save fits
    term = ["noTermination", "Termination"][args.termination_arm]
    path = f'useful_saves/recovery/{args.n_arms}arms_{args.n_outcomes}outcomes_{args.n_trials}trials_{args.n_sims}sims_{args.horizon}h_{args.alpha}alpha_{term}_fits.csv'
    df_fits.to_csv(path, index=False)

if __name__ == '__main__':
    main()