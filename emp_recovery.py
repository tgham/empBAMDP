from emp_runners import gen_emp, fit_emp
from emp_utils import canonical_states, canonical_count_matrix, array_to_hist, canon_to_concrete
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
    parser.add_argument('--ell_bounds', type=float, default=(0.01, 3), nargs=2)
    parser.add_argument('--temp_bounds', type=float, default=(0.01, 0.4), nargs=2)
    parser.add_argument('--h', type=int, default=1)
    parser.add_argument('--init_t', type=int, default=1)
    parser.add_argument('--n_sims', type=int, default=100)
    parser.add_argument('--n_jobs', type=int, default=-1)
    parser.add_argument('--agent_types', nargs='+', default=['emp', 'info'])
    parser.add_argument('--gen_data', action='store_true')
    parser.add_argument('--termination_arm', action='store_true')

    args = parser.parse_args()

    if args.gen_data:
        print(f'Generating {args.n_sims} datasets...')

        # Define the worker function
        def _gen_single_sim(sim_id, args):
            
            ## Sample parameters 
            ell = np.random.uniform(*args.ell_bounds)
            temp = np.random.uniform(*args.temp_bounds)

            # Generate data
            sim_tmp = gen_emp(
                n_arms=args.n_arms,
                n_outcomes=args.n_outcomes,
                n_trials=args.n_trials,
                n_rooms=args.n_rooms,
                alpha=args.alpha,
                ell=ell,
                h=args.h,
                temp=temp,
                termination_arm=args.termination_arm
            )
            sim_tmp['subject_id'] = [sim_id] * len(sim_tmp['room'])

            return sim_tmp

        ## parallellise
        with tqdm_joblib(tqdm(desc="Generating datasets", total=args.n_sims, ncols=100, unit='sim', mininterval=1)):
            results = Parallel(n_jobs=args.n_jobs)(
                delayed(_gen_single_sim)(sim_id, args)
                for sim_id in range(args.n_sims)
            )

        ## each results is a dictionary. we now need to convert each to a DataFrame and concatenate them into a single DataFrame.
        df_sim = pd.concat([pd.DataFrame.from_dict(res) for res in results], ignore_index=True)
        print(f"Generated {len(df_sim)} rows of data.")

        # Reorder columns so that 'subject_id' is first
        cols = df_sim.columns.tolist()
        cols = ['subject_id'] + [c for c in cols if c != 'subject_id']
        df_sim = df_sim[cols]

        ## canonicalise histories
        df_sim['canonical_counts_array'] = df_sim['counts_array'].apply(lambda x: canonical_count_matrix(x)[0])
        df_sim['history_str'] = df_sim['canonical_counts_array'].apply(lambda x: array_to_hist(x, args.n_arms, args.n_outcomes)[1])
        df_sim = df_sim.apply(lambda x: canon_to_concrete(x), axis=1)

        ## Save
        print('saving...')
        term = ["noTermination", "Termination"][args.termination_arm]
        path = f'useful_saves/recovery/{args.n_arms}arms_{args.n_outcomes}outcomes_{args.n_trials}trials_{args.n_sims}sims_{args.h}h_{args.alpha}alpha_{term}.csv'
        df_sim.to_csv(path, index=False)

        print(f"Saved {len(df_sim)} rows to {path}")

    
    ## or preload existing data
    else:
        term = ["noTermination", "Termination"][args.termination_arm]
        path = f'useful_saves/recovery/{args.n_arms}arms_{args.n_outcomes}outcomes_{args.n_trials}trials_{args.n_sims}sims_{args.h}h_{args.alpha}alpha_{term}.csv'
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
    for sim in range(args.n_sims):
        df_fits.loc[df_fits['subject_id']==sim, 'gen_ell'] = df_sim.loc[df_sim['subject_id']==sim, 'gen_ell'].iloc[0]
        df_fits.loc[df_fits['subject_id']==sim, 'gen_temp'] = df_sim.loc[df_sim['subject_id']==sim, 'gen_temp'].iloc[0]

    ## save fits
    path = f'useful_saves/recovery/{args.n_arms}arms_{args.n_outcomes}outcomes_{args.n_trials}trials_{args.n_sims}sims_{args.h}h_{args.alpha}alpha_{["noTermination" if not args.termination_arm else "Termination"]}_fits.csv'
    df_fits.to_csv(path, index=False)

if __name__ == '__main__':
    main()