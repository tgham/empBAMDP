import ast
import itertools
import numpy as np
import pandas as pd
from emp_utils import *
from scipy.optimize import bisect, brentq, minimize, differential_evolution
from scipy.special import softmax as _softmax
from scipy.special import logsumexp
from joblib import Parallel, delayed, effective_n_jobs
import warnings
from tqdm_joblib import tqdm_joblib
from scipy.stats import lognorm, truncnorm

from emp_utils import canonical_states, canonical_count_matrix, array_to_hist, canon_to_concrete
from emp_models import make_agent
warnings.filterwarnings('ignore')


def run_emp(df_ppt, ell=1, horizon = None, init_t = 0, temp = 1, verbose=False):
    """Simulate an empowerment-bandit agent yoked to participants' actual trial
    sequences. Returns a tidy DataFrame, one row per (subject_id, room, trial),
    tagged with `ell`, so results from several ell-agents can be pd.concat'd.
    """
    ## extract info from df_ppt
    n_trials = int(df_ppt['n_trials'].values[0])
    n_outcomes = int(df_ppt['n_outcomes'].values[0])
    n_arms = int(df_ppt['n_arms'].values[0])
    n_rooms = int(df_ppt['n_rooms'].values[0])  
    alpha = float(df_ppt['alpha'].values[0])
    termination_arm = bool(df_ppt['termination_arm'].values[0])
    cost = float(df_ppt['cost'].values[0]) if 'cost' in df_ppt.columns else 0.0
    n_actions = n_arms + int(termination_arm)
    terminate_idx = n_arms if termination_arm else None

    records = []

    ## emp agent at this ell, or info-seeking agent if ell is None; memo shared across subjects and rooms
    agent = make_agent(n_arms, n_outcomes, alpha, ell, termination_arm, cost)

    # for pid in df_ppt['subject_id'].unique():
    if verbose:
        print(f'Running run_emp for ell={ell}, horizon={horizon}, cost={cost}, termination_arm={termination_arm}, init_t={init_t}, temp={temp}')
        pbar = tqdm(range(len(df_ppt['subject_id'].unique())), desc='Subjects')
    for p in range(len(df_ppt['subject_id'].unique())):
        pid = df_ppt['subject_id'].unique()[p]
        df_p = df_ppt.loc[df_ppt['subject_id'] == pid]

        for r in range(n_rooms):
            df_pr = df_p.loc[df_p['room'] == r]
            counts = np.zeros((n_arms, n_outcomes), dtype=int)

            ## fill in counts with the actual counts from the participant's history up to init_t
            for t in range(init_t):
                row_df = df_pr.loc[df_pr['trial'] == t]
                if not row_df['terminated'].values[0]:
                    actual_action = row_df['action'].values[0]
                    actual_outcome =row_df['outcome'].values[0]
                    counts[actual_action, actual_outcome] += 1
                
                    ## since we're not simulating the agent for these trials, fill with nans
                    actual_action = n_arms
                    actual_outcome = np.nan
                    Q_a0 = np.nan
                    p_a0 = np.nan
                    chose_a0 = np.nan
                    Q_a1 = np.nan
                    p_a1 = np.nan
                    chose_a1 = np.nan
                    if n_arms > 2:
                        Q_a2 = np.nan
                        p_a2 = np.nan
                        chose_a2 = np.nan
                    current_emp = np.nan
                    row = {
                        'subject_id': pid, 'room': r, 'trial': t, 'ell': ell,
                        'chose_a0': chose_a0, 'chose_a1': chose_a1,
                        'p_choice_a0': p_a0, 'p_choice_a1': p_a1,
                        'current_emp': current_emp,
                        'Q_a0': Q_a0, 'Q_a1': Q_a1,
                        'Q_a2': Q_a2 if n_arms > 2 else np.nan,
                        'p_a2': p_a2 if n_arms > 2 else np.nan,
                        'chose_a2': chose_a2 if n_arms > 2 else np.nan,
                        'Q_terminate': np.nan if not termination_arm else np.nan,
                        'p_terminate': np.nan if not termination_arm else np.nan,
                        # 'p_repeat_choice': np.nan if prev_action is None else probs[prev_action],
                    }
                    records.append(row)
                else:
                    break

            last_action = np.nan
            for t in range(init_t, n_trials):
                row_df = df_pr.loc[df_pr['trial'] == t]
                if not row_df.empty:
                    h = (n_trials - t) if horizon is None else min(horizon, n_trials - t)

                    Q = agent.Q(counts, h)[0]

                    max_Q = np.nanmax(Q)
                    best_arms = np.where(Q == max_Q)[0]
                    action = int(np.random.choice(best_arms)) if len(best_arms) > 1 else int(best_arms[0])
                    probs = _softmax(Q/temp)

                    ## calculate current emp
                    current_emp = agent.leaf_value(counts)[0]

                    if row_df['terminated'].values[0]:
                        terminated = True
                        actual_action = n_arms
                        actual_outcome = np.nan
                        Q_a0 = np.nan
                        p_a0 = np.nan
                        chose_a0 = np.nan
                        Q_a1 = np.nan
                        p_a1 = np.nan
                        chose_a1 = np.nan
                        chose_least_sampled = np.nan
                        p_chose_least_sampled = np.nan
                        repeat_choice = np.nan
                        p_repeat_choice = np.nan
                        if n_arms > 2:
                            Q_a2 = np.nan
                            p_a2 = np.nan
                            chose_a2 = np.nan
                    else:
                        terminated=False

                        ## yoked
                        # actual_action = row_df['action'].values[0]
                        # actual_outcome = row_df['outcome'].values[0]

                        ## unyoked
                        actual_action = action
                        trueT = row_df['trueT'].values[0]
                        if actual_action < n_arms:
                            actual_outcome = int(np.random.choice(n_outcomes, p=trueT[action]))
                        else:
                            actual_outcome = np.nan ## terminated


                        ## some useful measures for comparing with humans
                        least_sampled_counts = np.min(counts.sum(axis=1))
                        most_sampled_counts = np.max(counts.sum(axis=1))
                        chose_least_sampled = action in np.where(counts.sum(axis=1) == least_sampled_counts)[0]
                        p_chose_least_sampled = probs[np.where(counts.sum(axis=1) == least_sampled_counts)[0]].max()
                        if t>0:
                            repeat_choice = action == last_action
                            p_repeat_choice = probs[last_action] if not np.isnan(last_action) else np.nan
                        else:
                            repeat_choice = np.nan
                            p_repeat_choice = np.nan
                        last_action = actual_action ## i.e. see if the agent repeats what the participant did

                        ## update counts for next trial
                        if actual_outcome is not np.nan:
                            counts[actual_action, actual_outcome] += 1

                        ## counts post diff - i.e. diff between the two arms
                        if n_arms == 2:
                            counts_post_diff = np.abs(counts[0].sum() - counts[1].sum())

                        ### map onto each of a0, a1 etc.

                        ## a0
                        if df_ppt.loc[(df_ppt['subject_id'] == pid) & (df_ppt['room'] == r) & (df_ppt['trial'] == t), 'a0'].values[0] == 'blue':
                            Q_a0 = Q[0]
                            p_a0 = probs[0]
                            chose_a0 = action == 0
                        elif df_ppt.loc[(df_ppt['subject_id'] == pid) & (df_ppt['room'] == r) & (df_ppt['trial'] == t), 'a0'].values[0] == 'red':
                            Q_a0 = Q[1]
                            p_a0 = probs[1]
                            chose_a0 = action == 1
                        elif df_ppt.loc[(df_ppt['subject_id'] == pid) & (df_ppt['room'] == r) & (df_ppt['trial'] == t), 'a0'].values[0] == 'green':
                            Q_a0 = Q[2]
                            p_a0 = probs[2]
                            chose_a0 = action == 2

                        ## a1
                        if df_ppt.loc[(df_ppt['subject_id'] == pid) & (df_ppt['room'] == r) & (df_ppt['trial'] == t), 'a1'].values[0] == 'blue':
                            Q_a1 = Q[0]
                            p_a1 = probs[0]
                            chose_a1 = action == 0
                        elif df_ppt.loc[(df_ppt['subject_id'] == pid) & (df_ppt['room'] == r) & (df_ppt['trial'] == t), 'a1'].values[0] == 'red':
                            Q_a1 = Q[1]
                            p_a1 = probs[1]
                            chose_a1 = action == 1
                        elif df_ppt.loc[(df_ppt['subject_id'] == pid) & (df_ppt['room'] == r) & (df_ppt['trial'] == t), 'a1'].values[0] == 'green':
                            Q_a1 = Q[2]
                            p_a1 = probs[2]
                            chose_a1 = action == 2

                        ## a2
                        if n_arms ==3:
                            if df_ppt.loc[(df_ppt['subject_id'] == pid) & (df_ppt['room'] == r) & (df_ppt['trial'] == t), 'a2'].values[0] == 'blue':
                                Q_a2 = Q[0]
                                p_a2 = probs[0]
                                chose_a2 = action == 0
                            elif df_ppt.loc[(df_ppt['subject_id'] == pid) & (df_ppt['room'] == r) & (df_ppt['trial'] == t), 'a2'].values[0] == 'red':
                                Q_a2 = Q[1]
                                p_a2 = probs[1]
                                chose_a2 = action == 1
                            elif df_ppt.loc[(df_ppt['subject_id'] == pid) & (df_ppt['room'] == r) & (df_ppt['trial'] == t), 'a2'].values[0] == 'green':
                                Q_a2 = Q[2]
                                p_a2 = probs[2]
                                chose_a2 = action == 2

                        
                    
                    ## record the row
                    row = {
                        'subject_id': pid, 'room': r, 'trial': t, 'ell': ell,
                        'chose_a0': chose_a0, 'chose_a1': chose_a1,
                        'p_choice_a0': p_a0, 'p_choice_a1': p_a1,
                        'current_emp': current_emp,
                        'Q_a0': Q_a0, 'Q_a1': Q_a1,
                        'chose_a2': chose_a2 if n_arms > 2 else np.nan,
                        'p_choice_a2': p_a2 if n_arms > 2 else np.nan,
                        'Q_a2': Q_a2 if n_arms > 2 else np.nan,
                        'chose_least_sampled': chose_least_sampled,
                        'p_chose_least_sampled': p_chose_least_sampled,
                        'repeat_choice': repeat_choice,
                        'p_repeat_choice': p_repeat_choice,
                        'emp_counts_post_diff': counts_post_diff if n_arms == 2 else np.nan,
                    }
                    if termination_arm:
                        row['Q_terminate'] = Q[-1]
                        row['p_terminate'] = probs[-1]
                    records.append(row)

                    if terminated:
                        break
                else:
                    break
        pbar.update(1) if verbose else None

    df_ell = pd.DataFrame.from_records(records)


    ## merge

        # ### concat

    # ## ensure all columns are present
    for col in df_ell.columns:
        if col not in df_ppt.columns:
            df_ppt[col] = np.nan
    df_ppt['ell'] = 'human'
    df_ppt['agent'] = 'human'
    df_ell.loc[df_ell['ell'].isnull(), 'agent'] = 'info'
    df_ell.loc[~df_ell['ell'].isnull(), 'agent'] = 'emp'
    df_full = pd.concat([df_ppt, df_ell], ignore_index=True, sort=False)


    
    # df_ell = df_ell.rename(columns={'chose_a0': 'ell'+str(ell)+'_chose_a0', 'chose_a1': 'ell'+str(ell)+'_chose_a1', 
    #                                 'chose_a2': 'ell'+str(ell)+'_chose_a2' if n_arms > 2 else np.nan,
    #                                 'p_choice_a0': 'ell'+str(ell)+'_p_choice_a0', 'p_choice_a1': 'ell'+str(ell)+'_p_choice_a1',
    #                                 'p_choice_a2': 'ell'+str(ell)+'_p_choice_a2' if n_arms > 2 else np.nan,
    #                                 'Q_a0': 'ell'+str(ell)+'_Q_a0', 'Q_a1': 'ell'+str(ell)+'_Q_a1', 
    #                                 'Q_a2': 'ell'+str(ell)+'_Q_a2' if n_arms > 2 else np.nan,
    #                                 'current_emp': 'ell'+str(ell)+'_current_emp',
    #                                 'Q_terminate': 'ell'+str(ell)+'_Q_terminate', 'p_terminate': 'ell'+str(ell)+'_p_terminate'})
    # df_ell = df_ell.rename(columns={'chose_a0': 'ell_chose_a0', 'chose_a1': 'ell_chose_a1', 
    #                                 'chose_a2': 'ell_chose_a2' if n_arms > 2 else np.nan,
    #                                 'p_choice_a0': 'ell_p_choice_a0', 'p_choice_a1': 'ell_p_choice_a1',
    #                                 'p_choice_a2': 'ell_p_choice_a2' if n_arms > 2 else np.nan,
    #                                 'Q_a0': 'ell_Q_a0', 'Q_a1': 'ell_Q_a1', 
    #                                 'Q_a2': 'ell_Q_a2' if n_arms > 2 else np.nan,
    #                                 'current_emp': 'ell_current_emp',
    #                                 'Q_terminate': 'ell_Q_terminate', 'p_terminate': 'ell_p_terminate'})
    # df_full = df_ppt.merge(df_ell, on=['subject_id', 'room', 'trial'], how='left')
    return df_full
    

## generate a single synthetic dataset, i.e. an ell agent acting in its own emp bandit env
def gen_arms(n_arms, n_outcomes, n_trials, n_rooms, alpha, ell, cost, horizon, termination_arm=True, diag_histories=None, n_subseq_trials=1, temp=1.0, greedy =False, seed=None):
    """Generate synthetic data from an agent in its own emp bandit env."""
    ## emp agent at this ell, or info-seeking agent if ell is None (NB: the info agent here is cost-free)
    agent = make_agent(n_arms, n_outcomes, alpha, ell, termination_arm,
                       cost=cost if ell is not None else 0.0)

    ## define ell_1 agent for scoring expected p(reward)
    ell_1_agent = make_agent(n_arms, n_outcomes, alpha, 1.0, termination_arm, cost)
    
    ## init data
    sim_out = defaultdict(list)
    
    ## loop through bandit envs
    for r in range(n_rooms):

        
        ## if full expt, initialise counts to 0 and play the whole room
        counts = np.zeros((n_arms, n_outcomes), dtype=int)
        if diag_histories is None:
            history_0 = ()
            t0 = 0
            n_trials_in_room = n_trials
            p_matrix = None

        ## if preset histories, seed the belief with one of the diagnostic histories
        else:
            history_0 = diag_histories[r % len(diag_histories)]
            for (a_obs, o_obs), c in history_0:
                counts[a_obs, o_obs] += c
            t0 = int(counts.sum())
            n_trials_in_room = min(n_subseq_trials, n_trials - t0)
            if n_trials_in_room < 1:
                raise ValueError(
                    f'preset history of length {t0} leaves no room for '
                    f'{n_subseq_trials} subsequent trials within n_trials={n_trials}'
                )
            
            ## true env needs to be consistent with the preset history - i.e. posterior draw from the prior, conditioned on the preset history
            prior = np.full((n_arms, n_outcomes), float(alpha))
            prior += counts
            p_matrix = np.zeros((n_arms, n_outcomes))
            for a in range(n_arms):
                p_matrix[a] = np.random.dirichlet(prior[a])

        ## fresh initialisation of env
        env = make_emp_env(n_arms=n_arms, n_outcomes=n_outcomes, n_trials=n_trials,
                        alpha=alpha, ell=ell, termination_arm=termination_arm, p_matrix=p_matrix,
                        seed=seed)
        env.reset()

        ## loop through trials
        for i in range(n_trials_in_room):
            t = t0 + i

            ## compute Q 
            h = (n_trials - t) if horizon is None else min(horizon, n_trials - t)
            Q = agent.Q(counts, h)[0]
            probs = _softmax(Q/temp)

            ## select action
            if greedy:
                max_Q = np.nanmax(Q)
                best_arms = np.where(Q == max_Q)[0]
                if len(best_arms) > 1:
                    action = int(np.random.choice(best_arms))
                else:
                    action = int(best_arms[0])
            else: #prob matching
                action = int(np.random.choice(len(probs), p=probs))
            (_, outcome), _, terminated, truncated, _ = env.step(action)

            ## calculate max counts fraction - i.e. the action that has been sampled the most, divided by total samples
            max_counts_fraction = np.max(counts.sum(axis=1)) / np.sum(counts) if np.sum(counts) > 0 else 0.0

            ## some useful measures for comparing with humans
            least_sampled_counts = np.min(counts.sum(axis=1))
            most_sampled_counts = np.max(counts.sum(axis=1))
            chose_least_sampled = action in np.where(counts.sum(axis=1) == least_sampled_counts)[0]
            p_chose_least_sampled = probs[np.where(counts.sum(axis=1) == least_sampled_counts)[0]].max()
            if i>0:
                repeat_choice = action == last_action
                p_repeat_choice = probs[last_action] if not np.isnan(last_action) else np.nan
            else:
                repeat_choice = np.nan
                p_repeat_choice = np.nan
            last_action = action


            ## save
            sim_out['room'].append(r)
            sim_out['trial'].append(t)
            sim_out['history_0'].append(repr(history_0))
            sim_out['action'].append(action)
            sim_out['outcome'].append(outcome)
            sim_out['terminated'].append(action == n_arms if termination_arm else False)
            sim_out['counts_array'].append(counts.copy())
            sim_out['max_counts_fraction'].append(max_counts_fraction)
            sim_out['chose_least_sampled'].append(chose_least_sampled)
            sim_out['p_chose_least_sampled'].append(p_chose_least_sampled)
            sim_out['repeat_choice'].append(repeat_choice)
            sim_out['p_repeat_choice'].append(p_repeat_choice)
            for a in range(n_arms):
                sim_out[f'Q_{a}'].append(Q[a])
                sim_out[f'p_{a}'].append(probs[a])
            if termination_arm:
                sim_out['Q_terminate'].append(Q[-1])
                sim_out['p_terminate'].append(probs[-1])

            ## update counts if no termination
            if action != n_arms:
                counts[action, outcome] += 1

            ## counts post diff - i.e. diff between the two arms
            if n_arms == 2:
                counts_post_diff = np.abs(counts[0].sum() - counts[1].sum())
            else:
                counts_post_diff = np.nan
            sim_out['counts_post_diff'].append(counts_post_diff)

            ## score on current probability of reward - i.e. emp_1
            ell_1 = ell_1_agent.leaf_value(counts)[0]
            sim_out['ell_1'].append(ell_1)

            ## but, terminate if the agent chose the termination arm
            if terminated or truncated:
                break

    
    ## add info to dict about params
    sim_out['gen_ell'] = [ell] * len(sim_out['room'])
    sim_out['gen_horizon'] = [horizon] * len(sim_out['room'])
    sim_out['gen_temp'] = [temp] * len(sim_out['room'])

    return sim_out


## generate a single synthetic dataset for the rooms task, i.e. an ell agent choosing between n_AFC belief states
def gen_rooms(n_arms, n_outcomes, n_trials, n_rooms, alpha, ell, cost, horizon=None, termination_arm=True, diag_histories=None, temp=1.0, greedy=False, seed=None):
    """Generate synthetic data from an agent choosing between preset belief states.

    Each of the `n_rooms` choices is between one tuple of `diag_histories`, with
    p(r|ell) = softmax_r(Emp_ell(h_r) / temp) -- the leaf value of each room's
    belief state, as in `_diag_model_row`. `n_trials` and `horizon` are unused,
    and kept only so the signature mirrors `gen_arms`.
    """
    ## emp agent at this ell, or info-seeking agent if ell is None 
    agent = make_agent(n_arms, n_outcomes, alpha, ell, termination_arm,
                       cost=cost if ell is not None else 0.0) #(NB: the info agent here is cost-free)

    ## define ell_1 agent for scoring expected p(reward)
    ell_1_agent = make_agent(n_arms, n_outcomes, alpha, 1.0, termination_arm, cost)

    ## init data
    sim_out = defaultdict(list)

    ## loop through choices, each between one diagnostic tuple of histories
    for r in range(n_rooms):

        ## seed each room's belief with its preset history
        histories = diag_histories[r % len(diag_histories)]
        counts_array = np.zeros((len(histories), n_arms, n_outcomes), dtype=int)
        for k, history_k in enumerate(histories):
            for (a_obs, o_obs), c in history_k:
                counts_array[k, a_obs, o_obs] += c

        ## compute Q = emp of each room's belief state
        Q = np.array([agent.leaf_value(C)[0] for C in counts_array])
        probs = _softmax(Q/temp)

        ## select room
        if greedy:
            max_Q = np.nanmax(Q)
            best_rooms = np.where(Q == max_Q)[0]
            if len(best_rooms) > 1:
                action = int(np.random.choice(best_rooms))
            else:
                action = int(best_rooms[0])
        else: #prob matching
            action = int(np.random.choice(len(probs), p=probs))

        ## save
        sim_out['room'].append(r)
        sim_out['trial'].append(0)
        sim_out['action'].append(action)
        for k, history_k in enumerate(histories):
            sim_out[f'room_history_{k}'].append(repr(history_k))
            sim_out[f't_{k}'].append(int(counts_array[k].sum()))
            sim_out[f'Q_{k}'].append(Q[k])
            sim_out[f'p_{k}'].append(probs[k])

        ## score chosen room on current probability of reward - i.e. emp_1
        ell_1 = ell_1_agent.leaf_value(counts_array[action])[0]
        sim_out['ell_1'].append(ell_1)

    ## add info to dict about params
    sim_out['gen_ell'] = [ell] * len(sim_out['room'])
    sim_out['gen_horizon'] = [horizon] * len(sim_out['room'])
    sim_out['gen_temp'] = [temp] * len(sim_out['room'])

    return sim_out


def enumerate_curves(n_arms, n_outcomes, n_trials, alphas = [0.1],
                     termination_arm=True, temp=1,
                     horizons = None,
                     ell_lo=0.001, ell_hi=100,
                     n_ell_samples=50,
                     costs=(0.0,),init_t=0):
    """Q / softmax-prob curves over ell for canonical histories.

    - enumerate ALL canonical histories at all trials,
      sampling `n_ell_samples` log-spaced ells in `[ell_lo, ell_hi]` for each.
      Coarse but exhaustive picture of how Q/p vary with ell.

    Each curve is produced by one belief agent (`EmpAgent`) per value in
    `alphas` (the agent knows its Dirichlet concentration); the `alpha` column
    holds the value. Each also emits the info-seeking columns `info_*`, computed
    by an `InfoSeekingAgent` with the same alpha.

    `skip_t0` (default True): drop the t=0 (empty `init`) history. There the
    agent has observed nothing and so has equal preference over the actions --
    uninteresting, and the costliest to sweep since its remaining horizon is
    largest. Set False to include it.

    SAMPLING COST: `ks` is a list of cost fractions to sweep; the whole curve
    enumeration is repeated for each `k` and stacked into one DataFrame with `k`
    (and the resulting per-row `cost`) as columns. For a given `k`, each arm pull
    is penalised by `c = k * (max achievable emp for this alpha, ell)`, paid
    recursively on every pull over the horizon (see `EmpAgent.Q`); the
    terminate action is free and the info-seeking columns stay cost-free.

    Returns a long-format DataFrame with one row per (history_str, t, ell,
    agent), columns: alpha, Q_0, Q_1, ..., Q_terminate (if
    applicable), p_0, p_1, ..., p_terminate, and matching info_* columns.
    Every ell is solved in one traversal. Each alpha's agents (one pair per cost)
    share their memo across histories and horizons, and are freed before the
    next alpha, so peak memory is one alpha's memo rather than the whole sweep's.
    """

    ## generate all canonical histories for the given (n_arms, n_outcomes, n_trials)
    states = canonical_states(n_arms, n_outcomes, n_trials)
    states_by_t_and_h = {(int(t), hs): C for (t, C, _, hs, _) in states}
    
    ## if no horizon, set to full horizon
    if horizons is None:
        horizons = [n_trials]

    ## define tasks: i.e. sweep all canonical histories with the predefined ell range
    sweep_tasks = [(int(t), history_str, ell_lo, ell_hi)
                    for (t, _, _, history_str, _) in states]
    # if skip_t0: ## skip first trial (uninteresting + costly)
    #     sweep_tasks = [task for task in sweep_tasks if task[0] != 0]
    sweep_tasks = [task for task in sweep_tasks if task[0] >= init_t] ## skip first init_t trials (uninteresting + costly)

    ### cost info

    ## sampling costs to sweep
    costs = [costs] if np.isscalar(costs) else list(costs)

    ## sampled ells, shared by every history
    sample_ells = np.logspace(np.log10(ell_lo), np.log10(ell_hi), n_ell_samples)

    ## big loop
    rows = []
    for alpha in alphas:

        ## create agents for this alpha: (emp for every sampled ell at once; info-seeker), for each cost
        agents = {c: (make_agent(n_arms, n_outcomes, alpha, None, termination_arm, cost=c), # info-seeker 
                      make_agent(n_arms, n_outcomes, alpha, sample_ells, termination_arm, cost=c)) # emp
                  for c in {float(k) for k in costs} | {0.0}}
        
        ## plus cost-free ones for the *_cost_free columns
        info_agent_cost_free, emp_agent_cost_free = agents[0.0]

        for horizon in horizons:
            for i in tqdm(range(len(sweep_tasks)), desc=f"Enumerating curves (alpha={alpha})"):
                t, history_str, _, _ = sweep_tasks[i]
                counts_array = states_by_t_and_h[(t, history_str)]
                h_remaining = int(np.min([horizon, n_trials - t]))

                ## get LML of history
                LML = info_agent_cost_free.marginal_likelihood(counts_array)

                ## get info-seeker's current MSE, and each ell agent's current emp (no cost)
                current_info_cost_free = info_agent_cost_free.leaf_value(counts_array)[0]
                current_emps_cost_free = emp_agent_cost_free.leaf_value(counts_array)

                ## loop through costs
                for cost in costs:
                    info_agent, emp_agent = agents[float(cost)]

                    ## emp of current belief state for each ell agent
                    current_emps = emp_agent.leaf_value(counts_array)
                    
                    ## get info-seeker's current MSE
                    current_info = info_agent.leaf_value(counts_array)[0]


                    ### Q values

                    ## empowerment agents: one row of Q per ell
                    Qs = emp_agent.Q(counts_array, h_remaining)

                    ## info-seeking agent (not parameterised by ell)
                    Q_info = info_agent.Q(counts_array, h_remaining)[0]
                    info_best_a = int(np.argmax(Q_info))
                    info_probs = _softmax(Q_info / temp)

                    ## save data
                    for ei in range(len(sample_ells)):
                        e = sample_ells[ei]
                        Q = Qs[ei]
                        probs = _softmax(Q / temp)
                        row = {'alpha': alpha,
                            'horizon': horizon, 'history_str': history_str, 't': t, 'ell': e, 
                            'current_emp': current_emps[ei], 'current_info': current_info,
                            'current_emp_cost_free': current_emps_cost_free[ei], 'current_info_cost_free': current_info_cost_free,
                            'cost': cost,
                            'LML': LML,
                            'info_best_a': info_best_a}
                        for a in range(n_arms):
                            row[f'Q_{a}'] = Q[a]
                            row[f'p_{a}'] = probs[a]
                            row[f'Q_info_{a}'] = Q_info[a]
                            row[f'info_p_{a}'] = info_probs[a]
                        if termination_arm:
                            row['Q_terminate'] = Q[-1]
                            row['p_terminate'] = probs[-1]
                            row['Q_info_terminate'] = Q_info[-1]
                            row['info_p_terminate'] = info_probs[-1]
                        rows.append(row)

    return pd.DataFrame(rows)


### Diagnosticity of an observation history: I(A; ell | h) - How much does observing the chosen action tell us about the agent's ell?

## truncated-normal prior on ell, (loc, scale) per empowerment agent type --
## the same prior parameter recovery samples generative ells from
ELL_TRUNCNORM = {'emp': (0.0, 5.0), 'emp_lo': (0.5, 1.0), 'emp_hi': (1.0, 2.0)}


def _ell_prior_dist(prior, mu, sigma, agent_type, ell_bounds):
    """The frozen scipy prior over ell, and its (lo, hi) support."""
    if prior == 'lognormal':
        return lognorm(sigma, scale=np.exp(mu)), (np.exp(mu - 4 * sigma), np.exp(mu + 4 * sigma))
    if prior == 'truncnorm':
        lo, hi = emp_ell_bounds(agent_type, ell_bounds)
        loc, scale = ELL_TRUNCNORM[agent_type]
        return truncnorm((lo - loc) / scale, (hi - loc) / scale, loc=loc, scale=scale), (lo, hi)
    raise ValueError(f"prior must be 'lognormal' or 'truncnorm', got {prior!r}")


## generate samples
def ell_prior_samples(n_samples=200, mu=0.0, sigma=1.0, seed=None,
                      prior='lognormal', agent_type='emp', ell_bounds=(0.01, 10.0)):
    """Sample of ell from the prior, with the weight each sample carries.

    `prior='lognormal'` is LN(mu, sigma): median exp(mu), heavy right tail.
    `prior='truncnorm'` is the truncated normal over `agent_type`'s slice of
    `ell_bounds` (see `emp_ell_bounds`), with (loc, scale) from `ELL_TRUNCNORM`.

    `sampling='grid'` (default) spaces ells evenly in log ell -- over
    mu +/- 4 sigma for the lognormal, over the support for the truncnorm -- and
    weights each by the prior mass of its log-ell cell (see `ell_weights`).
    `sampling='quantile'` returns the stratified midpoint quantiles
    ppf((m + 0.5)/M): deterministic and low-variance because p(a|h,ell) is
    smooth in ell. `sampling='random'` draws i.i.d. (use `seed`), kept for
    MC-error checks. Quantile and random samples are already distributed as
    the prior, so they are EQUAL WEIGHT.

    Returns (ells, weights), weights normalised to sum to 1.
    """
    n_samples = int(n_samples)

    ## get the prior distr
    dist, (lo, hi) = _ell_prior_dist(prior, mu, sigma, agent_type, ell_bounds)

    ## evenly spaced in log ell, weighted by prior mass of each cell
    ells = np.geomspace(lo, hi, n_samples)
    return ells, ell_weights(ells, dist)


## (normalised) prior weights for a grid of ells evenly spaced in log ell
def ell_weights(ells, dist):
    """Prior mass of each ell's cell on a log-spaced grid.

    A cell of fixed width dz in z = log ell covers d(ell) = ell * dz, so its
    mass is p_ell(ell) * ell * dz -- i.e. the prior density over log ell. The
    dz is common to every cell and cancels on normalising.
    """
    log_pi = dist.logpdf(ells) + np.log(ells)
    log_pi -= scipy.special.logsumexp(log_pi)
    return np.exp(log_pi)

## quick helper for computing H(p) = -sum p log p
def _neg_p_log_p(p):
    p = np.asarray(p, dtype=float)
    return -np.sum(np.where(p > 0, p * np.log(np.where(p > 0, p, 1.0)), 0.0), axis=-1)


def _top2_gap(V):
    """Top-two gap along the last axis of an (..., A) array, as a flat array.

    The margin behind every hard argmax in this module. In Q units it is what
    `temp` divides to set the choice odds; in probability units log(top1/top2)
    plays the same role. Zero when there is only one action.
    """
    V = np.atleast_2d(np.asarray(V, dtype=float))
    if V.shape[-1] < 2:
        return np.zeros(V.shape[0])
    S = np.sort(V, axis=-1)
    return S[..., -1] - S[..., -2]



def _mi_from_step(P, ell_w0s):
    P = np.asarray(P, dtype=float)
    
    ## dummy if info
    if ell_w0s is None:
        ell_w0s = np.array([1.0], dtype=float)
        info_seeker = True
    else:
        info_seeker = False
    H_ell = float(-np.sum(ell_w0s * np.log(ell_w0s + 1e-12)))
    
    ## or, H(ell|h) = log M for M equal-weight samples
    # H_ell = float(np.log(P.shape[0]))
    
    ## p(ell_m|h,a) = p(a|h,ell_m) p(ell_m) / sum_m' p(a|h,ell_m')p(ell_m') NB no need for p(ell_m) because the samples are equal weight
    col = (P * ell_w0s[:, None]).sum(axis=0, keepdims=True) # normalise each column
    P_post = np.divide(P * ell_w0s[:, None], col, out=np.zeros_like(P), where=col > 0)

    ## p(a|h) = sum_m p(a|h,ell_m) p(ell_m|h)
    p_marg = np.sum(P * ell_w0s[:, None], axis=0)

    ## E_a[H(ell|h,a)] = sum_a p(a|h) H(ell|h,a) 
    H_ell_cond = float(p_marg @ _neg_p_log_p(P_post.T)) # transpose puts ell on the last axis for `_neg_p_log_p`. See docstring.

    mi = max(H_ell - H_ell_cond, 0.0)

    return H_ell, H_ell_cond, mi, p_marg


def _mi_from_sequences(agent, counts_array, t, n_trials, horizon, temp, ell_w0s=None):
    """(H_ell, H_ell_cond, mi) from every (action, outcome) sequence that can follow h.

    The multi-step counterpart to `_mi_from_step`: how much the remaining
    h_remaining = min(horizon, n_trials - t) choices, and their outcomes, reveal about ell,

        I(seq; ell | h) = H(ell|h) - sum_seq p(seq|h) H(ell|h,seq).

    `agent` is the emp agent over the ell grid whose prior weights are `ell_w0s`.
    Step k of a sequence plans with the receding horizon min(horizon, n_trials - t - k),
    as in `gen_arms`. Every sequence is enumerated, so time and memory grow as
    (n_arms * n_outcomes) ** h_remaining.
    """
    n_arms, n_outcomes = agent.n_arms, agent.n_outcomes
    h_remaining = int(min(horizon, n_trials - t))

    ## emp agent 
    if ell_w0s is not None:
        n_ell = len(ell_w0s)
        ell_log_w0s = np.log(ell_w0s + 1e-12)
    ## info agent (no ell, so just single dummy ell with weight 1)
    else:
        n_ell = 1
        ell_log_w0s = np.log([1.0])

    ## every sequence of h_remaining (action, outcome) steps, or fewer if it terminates
    seqs = ao_sequences(n_arms, n_outcomes, h_remaining, termination_arm=agent.termination_arm)
    n_seqs = len(seqs)

    ## log p(seq|h,ell) for every sequence, all ells at once: (n_seqs, n_ell)
    log_seq_Ps = np.zeros((n_seqs, n_ell))
    for s, seq in enumerate(seqs):

        ## start from current history counts
        seq_counts = np.array(counts_array, dtype=np.int64)

        ## get choice probs for the rest of the sequence
        for subseq_t, (at, ot) in enumerate(seq):

            ## log p(a|h,ell) for every ell (retrieved from the agent's memo)
            logP = agent.log_policy(seq_counts, min(horizon, n_trials - t - subseq_t), temp)
            log_seq_Ps[s] += logP[:, at]

            ## outcome only relevant if no termination
            if at < n_arms:

                ## get probability of outcome under this arm (same for every ell)
                log_seq_Ps[s] += np.log(agent.predictive(seq_counts)[at, ot])

                ## update counts with the next action/outcome
                seq_counts[at, ot] += 1

            else:
                ## terminate action has no outcome, so this should be the end of the sequence
                pass

    ## entropy of prior weights: H(ell|h) = -sum_g w_0^g log w_0^g
    H_ell = float(_neg_p_log_p(ell_w0s))

    ## marginal over sequences: p(seq|h) = \sum_g w_0^g p(seq|h,ell_g)
    log_seq_marg = logsumexp(log_seq_Ps + ell_log_w0s[None, :], axis=1)

    ## posterior weights: p(ell|h,seq) \propto w_0^g p(seq|h,ell_g)
    w_post_log = log_seq_Ps + ell_log_w0s[None, :]
    w_post_log -= log_seq_marg[:, None]  #normalised
    w_post = np.exp(w_post_log)

    ## entropy of posterior weights: H(ell|h,seq) = -sum_g p(ell_g|h,seq) log p(ell_g|h,seq)
    H_ell_post = _neg_p_log_p(w_post)

    ## expected posterior entropy: E_seq[H(ell|h,seq)] = sum_seq p(seq|h) H(ell|h,seq)
    H_ell_cond = float(np.exp(log_seq_marg) @ H_ell_post)

    mi = max(H_ell - H_ell_cond, 0.0)
    return H_ell, H_ell_cond, mi, np.exp(log_seq_marg)


def _diag_emp_row(t, counts_array, canon_counts, history_str,
                          ell_samples, n_arms, n_outcomes, n_trials, alpha,
                          termination_arm, horizon, cost, temp,
                          tie_tol=None, agent=None, ell_w0s=None,
                          ):
    """Per-canonical-history diagnosticity row.

    `agent` is the emp agent over `ell_samples`: pass one to share its memo
    across histories (see `_diag_rows`), else a fresh one is built for this row.
    `ell_w0s` are the prior weights on `ell_samples` (from `ell_prior_samples`);
    None means equal weight.

    TIES: a hard argmax makes an ell whose top two Q's differ by 1e-9 look fully
    committed to the winner, and on a symmetric history (`init`, `a0o0:1-a1o0:1`)
    the arms tie to machine precision, so np.argmax hands the whole mass to the
    lowest index -- (1, 0, 0) for a history with no preference at all. So the
    membership is what is counted, not the argmax: an action counts for an ell
    when it is (one of) the BEST actions, i.e. within `tie_tol` of the top Q.
    A permissive and a strict count bracket the truth:

      - `best_a_frac_{a}`     : fraction of ells for which a is AMONG the best,
                                Q_a >= max_Q - tie_tol*temp. Ties credit every
                                tied action, so this does NOT sum to 1 -- a sum
                                above 1 is precisely the signature of ties, and
                                two actions both reading ~1 means the ells are
                                globally indifferent, not split.
      - `best_a_frac_dec_{a}` : fraction for which a is UNIQUELY best, i.e. no
                                other action within tie_tol. `tie_frac` collects
                                the rest, so sum_a best_a_frac_dec_{a} + tie_frac
                                = 1. Use this to select genuine ell-splits.
      - `gap_*`               : the top-two Q gap per ell, raw and in temp units.
                                gap/temp is the behaviourally meaningful scale --
                                the winner beats the runner-up exp(gap/temp):1 in
                                a two-way softmax.
      - `p_best_mean`         : E_ell[max_a p(a|h,ell)], a soft decisiveness
                                scalar (1/n_actions when every ell is
                                indifferent, 1 when all are decisive).

    `tie_tol` is in temp units; default log(3) ~ 1.0986, i.e. the winner must be
    at least 3:1 over the runner-up to count as uniquely best. `tie_tol=0` gives
    exact ties only, recovering the old argmax count except that exactly-tied
    actions now share credit instead of going to the lowest index. Note `mi` is
    ALREADY tie-robust -- built from the softmax policies, near-indifferent ells
    barely move it.

    ROOM SELECTION (`room_selection=True`): the choice is between rooms rather
    than arms. `counts_array` is then an (n_rooms, n_arms, n_outcomes) stack and
    `t`, `canon_counts`, `history_str` are per-room tuples. Each room is valued
    by its leaf empowerment, p(r|h,ell) = softmax_r(Emp_ell(h_r) / temp), so the
    choice is always one step and `horizon` / `n_trials` are unused. The row
    carries per-room `t_{r}`, `history_str_{r}`, `LML_{r}`, `p_marg_{r}` and
    `best_room_frac[_dec]_{r}` (the `best_a_frac` diagnostics, over rooms),
    plus the joined `pair_str`.
    """

    ## determine expt type
    if np.ndim(counts_array) == 3: # a list or stack of (n_arms, n_outcomes) count matrices, one per room
        room_selection = True # choice between rooms
    else:
        room_selection = False # choice between arms

    ### I(A; ell | h) = H(A|h) - E_ell[H(A|h,ell)]

    ## init
    tie_tol = float(np.log(2.0)) if tie_tol is None else float(tie_tol)

    ## room selection is always a one-step choice, so horizon/n_trials don't enter (and t is per-room)
    h_remaining = 1 if room_selection else int(np.min([horizon, n_trials - t]))

    ## grid sampling of ells
    n_ell = len(ell_samples)
    ell_w0s = np.full(n_ell, 1.0 / n_ell) if ell_w0s is None else np.asarray(ell_w0s, dtype=float)

    ## memoised agent over every sampled ell
    if agent is None:
        agent = make_agent(n_arms, n_outcomes, alpha, ell_samples, termination_arm, cost)

    ## Q values for single history - i.e. counts_array is a single (n_arms, n_outcomes) array, not a list of two arrays for a pair of histories
    if not room_selection:
        Qs = agent.Q(counts_array, h_remaining)             # (n_ell, n_actions)

    ## else, compare empowerment of the rooms' histories
    elif room_selection:
        counts_array = np.asarray(counts_array)
        n_AFC = counts_array.shape[0]
        Qs = np.zeros((n_ell, n_AFC))
        for r in range(n_AFC):
            Qs[:, r] = agent.leaf_value(counts_array[r]) ## i.e. Q is given by the empowerment afforded by the room's belief state, for every ell

    ## if just a single action, MI is taken over single step
    if h_remaining == 1:

        ## I(A;ell|h), from p(a|h,ell) for each sampled ell, where a is either the next sample or the chosen room
        if not room_selection:
            P = np.exp(agent.log_policy(counts_array, h_remaining, temp))
        else:
            ## softmax over the rooms, for each ell
            P = _softmax(Qs / temp, axis=1)                 # (n_ell, n_AFC)

        H_ell, H_ell_cond, mi, p_marg = _mi_from_step(P, ell_w0s)

    ## else, need to marginalise over sequences resulting from h
    else:

        ## I(seq;ell|h), exactly over every sequence of the remaining choices
        H_ell, H_ell_cond, mi, p_marg = _mi_from_sequences(agent, counts_array, t, n_trials, horizon, temp, ell_w0s)

    ### which action(s) does the diagnosticity comes from? 
    # near_best = Qs >= Qs.max(axis=1, keepdims=True) - tie_tol * temp
    tol = Qs.max() * 1e-6 
    near_best = Qs >= Qs.max(axis=1, keepdims=True) - tol ## i.e. which actions are among the best
    decisive = near_best.sum(axis=1) == 1 ## i.e. is there just one best action
    frac = near_best.mean(axis=0)
    frac_dec = (near_best & decisive[:, None]).mean(axis=0)

    ## how decisive is that choice? top-two Q gap per ell, in temp units.
    gap = _top2_gap(Qs)                                  # >= 0
    gap_temp = gap / temp

    row = {
        'alpha': alpha,
        'horizon': horizon, 'cost': cost, 'temp': temp,
        **({} if room_selection else {'t': t, 'history_str': history_str, 'history': canon_counts}),
        'H_ell': H_ell,
        'H_ell_cond': H_ell_cond,
        'mi': mi,
        'mi_bits': mi / np.log(2.0),
        'n_ell_samples': n_ell,

        ## tie diagnostics
        'tie_tol': tie_tol,
        'tie_frac': float(1.0 - decisive.mean()),
        'gap_mean': float(gap.mean()),
        'gap_median': float(np.median(gap)),
        'gap_min': float(gap.min()),
        'gap_mean_temp': float(gap_temp.mean()),
        'gap_median_temp': float(np.median(gap_temp)),
    }

    ## room selection: t, history and LML are per room, and the "actions" are the rooms
    if room_selection:
        row['pair_str'] = ' | '.join(history_str)
        for r in range(n_AFC):
            row[f't_{r}'] = t[r]
            row[f'history_str_{r}'] = history_str[r]
            row[f'history_{r}'] = canon_counts[r]
            row[f'LML_{r}'] = agent.marginal_likelihood(counts_array[r])
            row[f'emp_mean_{r}'] = float(ell_w0s @ Qs[:, r]) # prior-weighted E_ell[emp of room r]
            row[f'p_marg_{r}'] = p_marg[r]
            row[f'best_room_frac_{r}'] = frac[r]
            row[f'best_room_frac_dec_{r}'] = frac_dec[r]
        return row

    row['LML'] = agent.marginal_likelihood(counts_array)
    for a in range(n_arms):
        # row[f'p_marg_{a}'] = p_marg[a]
        row[f'best_a_frac_{a}'] = frac[a]
        row[f'best_a_frac_dec_{a}'] = frac_dec[a]
    if termination_arm:
        # row['p_marg_terminate'] = p_marg[-1]
        row['best_a_frac_terminate'] = frac[-1]
        row['best_a_frac_dec_terminate'] = frac_dec[-1]
    return row

def _diag_model_row(t, counts_array, canon_counts, history_str,
                    ell_samples, n_arms, n_outcomes, n_trials, alpha,
                    termination_arm, horizon, cost, temp_emp,
                    temp_info, p_model=(0.5, 0.5), tie_tol=None,
                    emp_agent=None, info_agent=None, ell_w0s=None):
    """Per-canonical-history MODEL diagnosticity I(A;M|h), M in {emp, info}.

    The counterpart to `_diag_emp_row`: where that asks how much the next action
    reveals about ell WITHIN the empowerment model, this asks how much it reveals
    about WHICH MODEL is generating the choices.

        I(A;M|h) = H(A|h) - sum_m p(m) H(A|h,m)

    with p(a|h,emp) = E_ell[p(a|h,ell)] the ell-MARGINALISED emp policy (the
    nuisance parameter is integrated out, not conditioned on) and p(a|h,info) the
    single info-seeking policy. Using E_ell[H(A|h,ell)] here instead would give
    I(A;M,ell|h), which double-counts the ell-diagnosticity; that quantity is
    still reported as `mi_joint` for reference, and satisfies

        mi_joint = mi + p(emp) * mi_ell.

    PRIOR vs POSTERIOR: both p(ell) and p(m) are PRIORS, not beliefs updated on h.
    This is a design-time score -- "if I showed a participant this history, how
    much would their next choice tell me?" -- not an observer's running belief.

    TIES: `mi` itself is tie-robust (it is a Jensen-Shannon divergence between
    the two softmax policies -- exactly JSD when p_model = (0.5, 0.5)), but any
    argmax READING of this row is not, so the margins are reported on all three
    fronts. `tie_tol` is a log-odds threshold, default log(2) ~ 0.693 (winner
    at least 2:1 over the runner-up), and applies to each:

      - WITHIN emp, across ell: `gap_*_emp`, `tie_frac_emp`, `p_best_mean_emp`
        and `best_a_frac[_dec]_emp_{a}` -- the `_diag_emp_row` diagnostics for
        the ell-split that `mi_ell` scores. Q gaps here are divided by `temp`.
        As there, `best_a_frac_emp_{a}` counts ells where a is AMONG the best
        (it does not sum to 1) and `_dec_` where it is uniquely best.
      - WITHIN info: `gap_info` (raw) and `gap_info_temp` = gap/`temp_info`,
        which IS the log-odds of the info agent's top two actions; `info_tie`
        flags gap_info_temp <= tie_tol, i.e. an info policy with no real
        preference. `p_best_info` is its max probability.
      - BETWEEN models: `best_a_emp` / `best_a_info` / `models_agree` is the
        hard read -- "the two models want different actions" -- and it is the
        one to distrust. The emp marginal has no single Q, so its margin is in
        probability units: `logodds_emp_marg` = log(p_top1/p_top2), with
        `emp_marg_tie` flagging it. `model_tie` is True when EITHER side is
        indifferent, in which case `models_agree` is reading argmax noise.
        `tvd_emp_info` = 0.5*sum_a |p(a|h,emp) - p(a|h,info)| is the scale-free
        0-1 companion: how far apart the two policies actually are.

    The Q scales of the two agents are unrelated (see TEMPERATURE), so there is
    no meaningful cross-model gap in Q units -- every between-model quantity
    here is in probability space, the only common currency.
    """

    ## determine expt type
    if np.ndim(counts_array) == 3: # a list or stack of (n_arms, n_outcomes) count matrices, one per room
        room_selection = True # choice between rooms
    else:
        room_selection = False # choice between arms

    ## init
    if room_selection:
        counts_array = np.asarray(counts_array)
    h_remaining = 1 if room_selection else int(np.min([horizon, n_trials - t]))
    n_actions = n_arms + int(termination_arm) if not room_selection else counts_array.shape[0]
    tie_tol = float(np.log(2.0)) if tie_tol is None else float(tie_tol)

    ## grid sampling of ells
    n_ell = len(ell_samples)
    ell_w0s = np.full(n_ell, 1.0 / n_ell) if ell_w0s is None else np.asarray(ell_w0s, dtype=float)


    ## memoised agents
    if emp_agent is None:
        emp_agent = make_agent(n_arms, n_outcomes, alpha, ell_samples, termination_arm, cost)
    if info_agent is None:
        info_agent = make_agent(n_arms, n_outcomes, alpha, None, termination_arm, cost)
    
    ## Q-values in sampling task
    if not room_selection:
        Qs_emp = emp_agent.Q(counts_array, h_remaining)             # (n_ell, n_actions)
        Q_info = info_agent.Q(counts_array, h_remaining)[0]         # (n_actions,)

    ## Q-values = emp of each room
    else:
        n_AFC = counts_array.shape[0]
        Qs_emp = np.zeros((n_ell, n_AFC))
        for r in range(n_AFC):
            Qs_emp[:, r] = emp_agent.leaf_value(counts_array[r]) # (n_ell, n_AFC)
        Q_info = np.zeros(n_AFC)
        for r in range(n_AFC):
            Q_info[r] = info_agent.leaf_value(counts_array[r])[0] # (n_AFC,)

    ### p(a|h,m) for each model
    
    ## if just single action, MI is taken over single step
    if h_remaining == 1:

        ## empowerment agent: marginalise over ell
        if not room_selection:
            P_emp = np.exp(emp_agent.log_policy(counts_array, h_remaining, temp_emp))
        else:
            ## softmax over the rooms, for each ell
            P_emp = _softmax(Qs_emp / temp_emp, axis=1)
        H_ell, H_cond_ell, mi_emp, p_marg_emp = _mi_from_step(P_emp, ell_w0s)

        ## info-seeking agent: not parameterised by ell, so a single policy
        if not room_selection:
            P_info = np.exp(info_agent.log_policy(counts_array, h_remaining, temp_info))
        else:
            P_info = _softmax(Q_info / temp_info)
            P_info = P_info[None, :] ## make it 2D for `_mi_from_step` to accept

        _, _, _, p_marg_info = _mi_from_step(P_info, None) ## don't need mi_info - trivially 0, because no ell-dependence

    
    ## else need to marginalise over sequences resulting from h
    else:

        ## emp terms
        _, _, mi_emp, p_marg_emp = _mi_from_sequences(emp_agent, counts_array, t, n_trials, horizon, temp_emp, ell_w0s)

        ## info-seeker terms
        _, _, _, p_marg_info = _mi_from_sequences(info_agent, counts_array, t, n_trials, horizon, temp_info, None)


    ### margins -- see TIES. tie_tol is a log-odds threshold throughout.

    ## within emp, across ell: the same diagnostics `_diag_emp_row` reports --
    ## `best_a_frac_emp_{a}` counts ells where a is AMONG the best (so it does
    ## not sum to 1), `_dec_` where it is uniquely best.
    near_best_emp = Qs_emp >= Qs_emp.max(axis=1, keepdims=True) - tie_tol * temp_emp
    decisive_emp = near_best_emp.sum(axis=1) == 1
    frac_emp = near_best_emp.mean(axis=0)
    frac_dec_emp = (near_best_emp & decisive_emp[:, None]).mean(axis=0)
    gap_emp = _top2_gap(Qs_emp)
    gap_emp_temp = gap_emp / temp_emp

    ## within info: one policy, so gap/temp_info IS its top-two log-odds
    gap_info = float(_top2_gap(Q_info)[0])
    gap_info_temp = gap_info / temp_info

    ## between models: probability space, the only common currency
    best_a_emp = int(np.argmax(p_marg_emp))
    best_a_info = int(np.argmax(p_marg_info))
    p_emp_sorted = np.sort(p_marg_emp)
    logodds_emp_marg = float(np.log(p_emp_sorted[-1] / p_emp_sorted[-2])
                             if n_actions > 1 and p_emp_sorted[-2] > 0 else 0.0)
    emp_marg_tie = bool(logodds_emp_marg <= tie_tol)
    info_tie = bool(gap_info_temp <= tie_tol)
    tvd = float(0.5 * np.abs(p_marg_emp - p_marg_info).sum())


    ### I(A;M|h) = H(M|h) - E_a[H(M|h,a)]

    ## H(M|h) = -sum_m p(m|h) log p(m|h) = H(M) because the prior is independent of h
    p_m = np.asarray(p_model, dtype=float)
    p_m = p_m / p_m.sum()
    H_M = float(_neg_p_log_p(p_m))

    ## H(M|h,a) = -sum_m p(m|h,a) log p(m|h,a),
    ## p(m|h,a)  = p(a|h,m) p(m) / sum_m' p(a|h,m') p(m')
    p_marg_model = p_m[0] * p_marg_emp + p_m[1] * p_marg_info # p(a|h)
    H_marg_model = float(_neg_p_log_p(p_marg_model))
    p_m_emp = p_marg_emp * p_m[0] / p_marg_model
    p_m_info = p_marg_info * p_m[1] / p_marg_model
    p_m_post = np.stack([p_m_emp, p_m_info], axis=0) # (M, A)
    H_M_cond = _neg_p_log_p(p_m_post.T) # transpose puts a on the last axis

    ## E_a[H(M|h,a)] = sum_a p(a|h) H(M|h,a)
    E_a_H_M_cond = float(p_marg_model @ H_M_cond) 
    mi_model = max(H_M - E_a_H_M_cond, 0.0)

    ## normalise by H(M): I(A;M|h) <= min(H(A), H(M))
    mi_norm = mi_model / H_M if H_M > 0 else 0.0


    row = {
        'alpha': alpha,
        'horizon': horizon, 'cost': cost, 'temp_emp': temp_emp, 'temp_info': temp_info,
        **({} if room_selection else {'t': t, 'history_str': history_str, 'history': canon_counts}),
        'target': 'model',
        'p_model_emp': p_m[0],
        'H_A_h': H_marg_model,
        'mi': mi_model,
        'mi_bits': mi_model / np.log(2.0),
        'mi_norm': mi_norm,
        'mi_emp': mi_emp,        # I(A;ell|h,emp) -- the _diag_emp_row quantity
        'n_ell_samples': n_ell,
        ## tie diagnostics -- within emp (across ell)
        'tie_tol': tie_tol,
        'tie_frac_emp': float(1.0 - decisive_emp.mean()),
        'gap_mean_emp': float(gap_emp.mean()),
        'gap_median_emp': float(np.median(gap_emp)),
        'gap_min_emp': float(gap_emp.min()),
        'gap_mean_temp_emp': float(gap_emp_temp.mean()),
        'gap_median_temp_emp': float(np.median(gap_emp_temp)),
        ## -- within info
        'gap_info': gap_info,
        'gap_info_temp': gap_info_temp,
        'info_tie': info_tie,
        ## -- between models
        'best_a_emp': best_a_emp,
        'best_a_info': best_a_info,
        'models_agree': bool(best_a_emp == best_a_info),
        'logodds_emp_marg': logodds_emp_marg,
        'emp_marg_tie': emp_marg_tie,
        'model_tie': bool(emp_marg_tie or info_tie),
        'tvd_emp_info': tvd,
    }
    ## room selection: t, history and LML are per room, and the "actions" are the rooms
    if room_selection:
        row['pair_str'] = ' | '.join(history_str)
        for r in range(n_AFC):
            row[f't_{r}'] = t[r]
            row[f'history_str_{r}'] = history_str[r]
            row[f'history_{r}'] = canon_counts[r]
            row[f'LML_{r}'] = emp_agent.marginal_likelihood(counts_array[r]) # belief-model property, shared by both agents
            row[f'p_marg_{r}'] = p_marg_model[r]
            row[f'p_marg_emp_{r}'] = p_marg_emp[r]
            row[f'p_marg_info_{r}'] = p_marg_info[r]
            row[f'best_room_frac_emp_{r}'] = frac_emp[r]
            row[f'best_room_frac_dec_emp_{r}'] = frac_dec_emp[r]
    else:
        row['LML'] = emp_agent.marginal_likelihood(counts_array) # belief-model property, shared by both agents
        for a in range(n_arms):
            row[f'p_marg_{a}'] = p_marg_model[a]
            row[f'p_marg_emp_{a}'] = p_marg_emp[a]
            row[f'p_marg_info_{a}'] = p_marg_info[a]
            row[f'best_a_frac_emp_{a}'] = frac_emp[a]
            row[f'best_a_frac_dec_emp_{a}'] = frac_dec_emp[a]
        if termination_arm:
            row['p_marg_terminate'] = p_marg_model[-1]
            row['p_marg_emp_terminate'] = p_marg_emp[-1]
            row['p_marg_info_terminate'] = p_marg_info[-1]
            row['best_a_frac_emp_terminate'] = frac_emp[-1]
            row['best_a_frac_dec_emp_terminate'] = frac_dec_emp[-1]
    return row


def _diag_rows(target, states, args, ell_w0s=None):
    """Diagnosticity rows for a batch of canonical histories, sharing one set of agents.

    The task unit of `enumerate_diagnosticity`, module-level so joblib can pickle
    it. `args` are the row function's arguments after the history. The agents are
    built here and dropped on return, so their memo is shared by every history in
    the batch but never outlives it, in the parent process or in a worker.
    """
    ell_samples, n_arms, n_outcomes, _, alpha, termination_arm, _, cost = args[:8]
    emp_agent = make_agent(n_arms, n_outcomes, alpha, ell_samples, termination_arm, cost)
    if target == 'ell':
        return [_diag_emp_row(t, C, cc, hs, *args, agent=emp_agent, ell_w0s=ell_w0s)
                for (t, C, cc, hs, _) in states]
    info_agent = make_agent(n_arms, n_outcomes, alpha, None, termination_arm, cost)
    return [_diag_model_row(t, C, cc, hs, *args, emp_agent=emp_agent, info_agent=info_agent,
                            ell_w0s=ell_w0s)
            for (t, C, cc, hs, _) in states]


def enumerate_diagnosticity(n_arms=2, n_outcomes=4, n_trials=6, alphas=(0.1,),
                            termination_arm=True, temp_emp=1.0, temp_info=1.0,
                            horizons=None, costs=(0.0,),
                            n_ell_samples=200, prior_mu=0.0, prior_sigma=1.0,
                            seed=None,
                            ell_prior='lognormal', agent_type='emp', ell_bounds=(0.01, 10.0),
                            init_t=0, n_jobs=1,
                            target='ell', p_model=(0.5, 0.5), tie_tol=None,
                            expt='arms', n_AFC=2, n_room_samples=None):
    """Diagnosticity of every canonical history, for one of two targets.

    Mirrors `enumerate_curves`: the same canonical-history enumeration, the same
    agent specs (one agent per value in `alphas`), and
    the same horizon / cost sweeps. Where `enumerate_curves` reports the Q/p curve
    at each ell, this reports the single scalar that summarises how much the
    action reveals about ell.

    `expt` selects WHAT the agent chooses between:
      - 'arms'  (default): the next action within one history, as above.
      - 'rooms': a room, from `n_AFC` distinct canonical histories, with
                p(r|h,ell) = softmax_r(Emp_ell(h_r) / temp_emp) -- the leaf
                empowerment of each room's belief state (see `_diag_emp_row`,
                ROOM SELECTION). One step by construction, so `horizons` is
                ignored (the `horizon` column reads 0). For target='model' the
                info agent likewise values each room by its leaf, with
                softmax_r(Info(h_r) / temp_info). `n_room_samples=None` (default) enumerates every
                unordered tuple of histories once -- quadratic in the number of
                histories for pairs, so keep n_trials small; an int instead
                draws that many tuples uniformly, with replacement across
                tuples (so one can recur), seeded by `seed`. Rows carry a
                `pair` index and per-room `t_{r}`, `history_str_{r}`,
                `orbit_size_{r}` etc; `history_str_{r}` joins against the
                'arms' output.

    `target` selects WHAT the action is diagnostic OF:
      - 'ell'   (default): I(A;ell|h) -- which ell, within the empowerment model
                (`_diag_emp_row`). Takes the extra `tie_tol` (in temp units,
                default log(3)): how far below the top Q an action may sit and
                still count as one of the best. `best_a_frac_{a}` counts ells
                where a is among the best (ties credit every tied action, so it
                does not sum to 1); `best_a_frac_dec_{a}` counts only ells where
                a is uniquely best, with `tie_frac` taking the remainder. Select
                genuine ell-splits on the `_dec_` columns -- a plain argmax
                would score a 1e-9 Q difference as a decisive win.
      - 'model':           I(A;M|h) with M in {emp, info} -- which model, with ell
                marginalised out of the emp policy (`_diag_model_row`). Takes the
                extra `temp_info` (info agent's softmax temperature, defaults to
                `temp`) and `p_model` (prior over the two models). `tie_tol`
                applies here too, as a log-odds threshold on each model's own
                top-two margin: `model_tie` flags the histories where
                `models_agree` is reading argmax noise.
    Both emit `target` and `expt` columns and a comparable `mi`, so the frames concat.

    The ell prior is `ell_prior`: 'lognormal' is LN(prior_mu, prior_sigma);
    'truncnorm' is the truncated normal over `agent_type`'s slice of
    `ell_bounds`, as in parameter recovery (see `ell_prior_samples`).

    The ell sample is drawn ONCE and reused across every history, alpha, horizon
    and cost -- common random numbers, so the resulting mi values are directly
    comparable between histories, which is the point of the score.

    Returns a long DataFrame, one row per (alpha, horizon, cost, t,
    history_str) -- or per (alpha, cost, pair) for 'rooms'. Column names match
    `enumerate_curves` so the two merge on
    ['alpha', 'horizon', 'cost', 't', 'history_str'].

    COST: n_samples Bayes-adaptive Bellman solves per (history, alpha, horizon,
    cost) -- the same shape of cost as `enumerate_curves` with
    n_ell_samples = n_samples. `n_jobs` parallelises over interleaved batches of
    histories; each batch builds its own agents and frees them on return, so at
    most `n_jobs` memos are alive at once.
    """
    if target not in ('ell', 'model'):
        raise ValueError(f"target must be 'ell' or 'model', got {target!r}")
    if expt not in ('arms', 'rooms'):
        raise ValueError(f"expt must be 'arms' or 'rooms', got {expt!r}")

    ## shared ell sample from the prior, with its weights
    ell_samples, ell_w0s = ell_prior_samples(n_ell_samples, mu=prior_mu, sigma=prior_sigma,
                                             seed=seed, prior=ell_prior,
                                             agent_type=agent_type, ell_bounds=ell_bounds)

    ## canonical histories, optionally skipping the first init_t trials
    states = canonical_states(n_arms, n_outcomes, n_trials)
    states = [s for s in states if int(s[0]) >= init_t]

    if expt == 'rooms':
        ## rooms are valued at their leaf, so a single dummy horizon of 0
        horizons = [0]
        states = _room_tuples(states, n_AFC, n_room_samples, seed)
    elif horizons is None:
        # horizons = [n_trials]
        horizons = [1]


    costs = [costs] if np.isscalar(costs) else list(costs)

    rows = []
    for alpha in alphas:
        for horizon in horizons:
            for cost in costs:
                desc = (f"Diagnosticity[{target}, {expt}] (alpha={alpha}, "
                        f"h={horizon}, cost={cost})")
                args = (ell_samples, n_arms, n_outcomes, n_trials, alpha,
                        termination_arm, horizon, cost, temp_emp)
                args = args + ((temp_info, p_model, tie_tol) if target == 'model'
                               else (tie_tol,))
                block = _run_diag_rows(target, states, args, n_jobs, desc, ell_w0s)

                ## rooms: index the tuples, and carry each room's orbit size
                if expt == 'rooms':
                    for p, (row, pair) in enumerate(zip(block, states)):
                        row['pair'] = p
                        for r, o in enumerate(pair[4]):
                            row[f'orbit_size_{r}'] = o
                rows.extend(block)

    df = pd.DataFrame(rows)
    df['target'] = target
    df['expt'] = expt
    df['ell_prior'] = ell_prior
    if ell_prior == 'truncnorm':
        df['agent_type'] = agent_type
        df['ell_lo'], df['ell_hi'] = emp_ell_bounds(agent_type, ell_bounds)
        df['prior_loc'], df['prior_scale'] = ELL_TRUNCNORM[agent_type]
    else:
        df['prior_mu'] = prior_mu
        df['prior_sigma'] = prior_sigma
    return df


def _room_tuples(states, n_AFC, n_room_samples=None, seed=None):
    """Tuples of `n_AFC` distinct canonical histories, packed for `_diag_rows`.

    `n_room_samples=None` gives every unordered tuple once; an int draws that many
    uniformly, with replacement across tuples. Each tuple comes back in the
    per-history (t, C, canon_counts, history_str, orbit_size) layout, per room:
    ((t_0, t_1, ...), stacked C, (cc_0, ...), (hs_0, ...), (orbit_0, ...)).
    """
    n_states = len(states)
    if n_states < n_AFC:
        raise ValueError(f'only {n_states} canonical histories, cannot fill {n_AFC} distinct rooms')

    if n_room_samples is None:
        idx = list(itertools.combinations(range(n_states), n_AFC))
    else:
        rng = np.random.default_rng(seed)
        idx = [rng.choice(n_states, size=n_AFC, replace=False) for _ in range(int(n_room_samples))]
    tuples = [tuple(zip(*(states[i] for i in ii))) for ii in idx]
    return [(ts, np.stack(Cs), ccs, hss, os_) for (ts, Cs, ccs, hss, os_) in tuples]


def _run_diag_rows(target, states, args, n_jobs, desc, ell_w0s=None):
    """`_diag_rows` over `states`, serially or in `n_jobs` parallel batches, in the order of `states`."""
    if n_jobs == 1:
        return _diag_rows(target, tqdm(states, desc=desc), args, ell_w0s)

    ## several batches per worker, interleaved so each mixes early (deep) and late histories
    n_batches = min(len(states), 4 * effective_n_jobs(n_jobs))
    with tqdm_joblib(tqdm(total=n_batches, desc=desc)):
        batch_rows = Parallel(n_jobs=n_jobs)(
            delayed(_diag_rows)(target, states[b::n_batches], args, ell_w0s)
            for b in range(n_batches)
        )

    ## undo the interleaving so rows keep the order of `states`
    block = [None] * len(states)
    for b, r in enumerate(batch_rows):
        block[b::n_batches] = r
    return block


def diagnosticity_for_counts(C, n_arms=None, n_outcomes=None, n_trials=None,
                             alpha=0.1,
                             termination_arm=True, temp_emp=1.0, temp_info=1.0, horizon=None, cost=0.0,
                             n_samples=200, prior_mu=0.0, prior_sigma=1.0,
                             seed=None,
                             ell_prior='lognormal', agent_type='emp', ell_bounds=(0.01, 10.0),
                             target='ell', p_model=(0.5, 0.5), tie_tol=None):
    """Diagnosticity for ONE arbitrary (non-canonical) count matrix.

    For scoring a real participant's history (`run_emp`) or a simulated one
    (`gen_arms`'s `counts_array`). `C` is canonicalised first: diagnosticity is
    constant on arm/outcome-relabelling orbits, so the canonical value is the
    right one, and the returned `history_str` is the canonical label that joins
    against `enumerate_diagnosticity` / `enumerate_curves` output.

    `n_trials` and `horizon` both default to "t pulls already taken, t more to
    come"; pass them explicitly to match a particular task design.

    `target` ('ell' or 'model'), `temp_info`, `p_model` and the ell prior
    (`ell_prior`, `agent_type`, `ell_bounds`) behave exactly as in
    `enumerate_diagnosticity`.

    Returns the row dict.
    """
    if target not in ('ell', 'model'):
        raise ValueError(f"target must be 'ell' or 'model', got {target!r}")
    C = np.asarray(C, dtype=int)
    if n_arms is None or n_outcomes is None:
        n_arms, n_outcomes = C.shape
    counts_array, _ = canonical_count_matrix(C)
    canon_counts, history_str = array_to_hist(counts_array, n_arms, n_outcomes)
    t = int(counts_array.sum())
    if n_trials is None:
        n_trials = t + (t if horizon is None else horizon)
    if horizon is None:
        horizon = n_trials


    ell_samples, ell_w0s = ell_prior_samples(n_samples, mu=prior_mu, sigma=prior_sigma,
                                             seed=seed, prior=ell_prior,
                                             agent_type=agent_type, ell_bounds=ell_bounds)
    args = (ell_samples, n_arms, n_outcomes, n_trials, alpha,
            termination_arm, horizon, cost, temp_emp)
    if target == 'ell':
        row = _diag_emp_row(t, counts_array, canon_counts, history_str, *args,
                            tie_tol=tie_tol, ell_w0s=ell_w0s)
    else:
        row = _diag_model_row(t, counts_array, canon_counts, history_str, *args,
                              temp_info=temp_info, p_model=p_model,
                              tie_tol=tie_tol, ell_w0s=ell_w0s)
    row['target'] = target
    row['orbit_size'] = orbit_sequence_count(counts_array)
    row['ell_prior'] = ell_prior
    if ell_prior == 'truncnorm':
        row['agent_type'] = agent_type
        row['ell_lo'], row['ell_hi'] = emp_ell_bounds(agent_type, ell_bounds)
        row['prior_loc'], row['prior_scale'] = ELL_TRUNCNORM[agent_type]
    else:
        row['prior_mu'] = prior_mu
        row['prior_sigma'] = prior_sigma
    return row

    







### fitting functions

## the empowerment agent, optionally split by where ell sits relative to 1
EMP_AGENT_TYPES = ('emp', 'emp_lo', 'emp_1', 'emp_hi')


def emp_ell_bounds(agent_type, ell_bounds, eps=1e-3):
    """Restrict `ell_bounds` to the slice of ell that `agent_type` occupies.

    'emp' keeps the full range, as before. The split types carve it at ell=1:
    'emp_lo' takes the compressive region (ell<1, breadth of reachable
    outcomes), 'emp_hi' the expansive one (ell>1, control over individual
    outcomes), and 'emp_1' pins ell=1 so only temp is free. `eps` keeps the two
    open types strictly off 1, so a fit that runs to the boundary stays
    distinguishable from 'emp_1'.

    Returns (lo, hi); lo == hi for the pinned type.
    """
    lo, hi = float(ell_bounds[0]), float(ell_bounds[1])
    if agent_type == 'emp':
        return (lo, hi)
    if agent_type == 'emp_1':
        return (1.0, 1.0)
    if agent_type == 'emp_lo':
        bounds = (lo, min(hi, 1.0 - eps))
    elif agent_type == 'emp_hi':
        bounds = (max(lo, 1.0 + eps), hi)
    else:
        raise ValueError(f"unknown empowerment agent type {agent_type!r}; "
                         f"expected one of {EMP_AGENT_TYPES}")
    if bounds[0] >= bounds[1]:
        raise ValueError(f"ell_bounds {tuple(ell_bounds)} leave no room for "
                         f"'{agent_type}'")
    return bounds


## parallelised fitting
def fit_emp(df_ppt,
            # ell_bounds=(0.1, 5.0), temp_bounds=(0.1, 10.0),
            param_bounds, agent_types=['emp', 'info'],
                           horizon=None, init_t=0,
                           maxiter=200, tol=1e-6, n_jobs=-1,
                           ell_eps=1e-3, verbose=True):
    """
    Parallelized version of fit_emp_model using joblib.

    Parameters:
    -----------
    df_ppt : pd.DataFrame
        Participant data
    param_bounds : (ell_bounds, temp_bounds)
        Parameter bounds, each a (lo, hi) tuple.
    agent_types : list of str
        Models to fit, one row per subject each. 'info' is the info-seeking
        agent (temp only); 'emp' the empowerment agent over the full ell range.
        Passing 'emp_lo' / 'emp_1' / 'emp_hi' instead of 'emp' splits it into
        three types by ell<1, ell=1 (a one-parameter model) and ell>1; see
        `emp_ell_bounds`.
    horizon : int or None
    k : float
    init_t : int
        Leading trials used to warm the belief without being scored. Preset
        ("horizons") rooms need init_t=0: their warm-up is the instructed
        history in `preset_history`, which is already in the belief.
    maxiter : int
    tol : float
    n_jobs : int
        Number of parallel jobs (-1 = all cores)
    ell_eps : float
        Margin keeping 'emp_lo'/'emp_hi' off ell=1.
    verbose : bool

    Returns:
    --------
    pd.DataFrame with fitted parameters per subject
    """
    pids = df_ppt['subject_id'].unique()
    df_fits = pd.DataFrame()
    for agent_type in agent_types:
        if agent_type == 'info':
            ell_bounds, temp_bounds = (None, None), param_bounds[1]
        else:
            ell_bounds = emp_ell_bounds(agent_type, param_bounds[0], eps=ell_eps)
            temp_bounds = param_bounds[1]
        with tqdm_joblib(tqdm(total=len(pids), desc=f"Fitting {agent_type}",
                              disable=not verbose)):
            results = Parallel(n_jobs=n_jobs)(
                delayed(_fit_ppt)(
                    pid=pid,
                    df_ppt=df_ppt.loc[df_ppt['subject_id'] == pid],
                    ell_bounds=ell_bounds,
                    temp_bounds=temp_bounds,
                    horizon=horizon,
                    init_t=init_t,
                    maxiter=maxiter,
                    tol=tol,
                    verbose=False,
                    agent_type=agent_type
                )
                for pid in pids
            )
        if df_fits.empty:
            df_fits = pd.DataFrame(results)
        else:
            df_fits = pd.concat([df_fits, pd.DataFrame(results)], ignore_index=True)
    return df_fits
def _fit_ppt(pid, df_ppt, ell_bounds, temp_bounds, horizon,
             init_t, maxiter, tol, verbose, agent_type=None,
             popsize=15, mutation=(0.5, 1), recombination=0.7, seed=None):
    """
    Fit model for a single subject using differential evolution.
    """
    
    ## hoist the data out of the DataFrame
    design = _design_from_df(df_ppt)
    if design['expt'] == 'rooms':
        rooms = _rooms_from_df(df_ppt)
    else:
        arms = _arms_from_df(df_ppt)

    ## emp vs info agent, and which parameters the optimiser actually searches
    if agent_type is None:
        agent_type = 'info' if ell_bounds[0] is None else 'emp'
    fixed_ell = None
    if agent_type == 'info':
        bounds = [temp_bounds]
    elif ell_bounds[0] == ell_bounds[1]: ## ell pinned, e.g. 'emp_1'
        fixed_ell = float(ell_bounds[0])
        bounds = [temp_bounds]
    else:
        bounds = [ell_bounds, temp_bounds]

    def compute_nll(params):
        if len(params) == 1: ## info-seeking agent, or emp with ell pinned
            ell = fixed_ell
            temp = params[0]
        else: ## empowerment agent
            ell, temp = params

        if design['expt'] == 'rooms':
            return _nll_from_rooms(rooms, design, ell, temp)
        elif design['expt'] == 'arms':
            return _nll_from_arms(arms, design, ell, temp, horizon, init_t)

    res = differential_evolution(
        func=compute_nll,
        bounds=bounds,
        maxiter=maxiter,
        tol=tol,
        popsize=popsize,
        mutation=mutation,
        recombination=recombination,
        seed=seed,
        polish=True,    
        workers=1,      
        updating='deferred',
        disp=False
    )
    if len(res.x) == 1:
        ell_hat = fixed_ell
        temp_hat = res.x[0]
    else:
        ell_hat, temp_hat = res.x
    nll = res.fun
    success = res.success

    ## calculate n_trials from the scored count
    if design['expt'] == 'rooms':
        n_fit_trials = len(rooms)
    else:
        n_fit_trials = len(df_ppt.loc[df_ppt['trial'] >= init_t])
    n_free_params = len(bounds) ## temp, plus ell wherever it is free
    BIC = n_free_params*np.log(n_fit_trials) + 2*nll

    return {
        'subject_id': pid,
        'ell': ell_hat,
        'temp': temp_hat,
        'nll': nll,
        'BIC': BIC,
        'success': success,
        'n_trials': n_fit_trials,
        'agent_type': agent_type
    }

## efficient hoisting of important info for fitting
def _design_from_df(df_ppt):

    ## get task params
    return {
        'n_arms': int(df_ppt['n_arms'].values[0]),
        'n_outcomes': int(df_ppt['n_outcomes'].values[0]),
        'n_trials': int(df_ppt['n_trials'].values[0]),
        'termination_arm': bool(df_ppt['termination_arm'].values[0]),
        'cost': float(df_ppt['cost'].values[0]),
        'alpha': float(df_ppt['alpha'].values[0]),
        'expt': str(df_ppt['expt'].values[0]) if 'expt' in df_ppt.columns else 'arms',
    }


def _counts_from_preset(history, n_arms, n_outcomes):
    """Turn a preset history -- `(((a, o), count), ...)`, or its repr as read
    back from CSV -- into the count matrix the agent starts the room with.

    Returns None for an absent/empty history, i.e. "start from a flat prior".
    This is the same encoding as the `history` column of the diagnosticity
    tables, so a preset history can be dropped straight into `gen_arms`.
    """
    if history is None or (isinstance(history, float) and np.isnan(history)):
        return None
    if isinstance(history, str):
        history = history.strip()
        if not history or history in ('()', 'nan'):
            return None
        history = ast.literal_eval(history)
    if len(history) == 0:
        return None
    counts = np.zeros((n_arms, n_outcomes), dtype=int)
    for (a_obs, o_obs), c in history:
        counts[int(a_obs), int(o_obs)] += int(c)
    return counts


def _arms_from_df(df_ppt):
    """Hoist each room's choice sequence out of the DataFrame, together with the
    belief it starts from.

    In the full task that starting belief is flat; in the preset-history
    ("horizons") task each room opens with an instructed history, carried in the
    `preset_history` column, and only the choices that follow it are scored.
    """
    cols = ['subject_id', 'room', 'trial', 'action', 'outcome', 'terminated']
    has_history_0 = 'history_0' in df_ppt.columns
    if has_history_0:
        cols = cols + ['history_0']
        n_arms = int(df_ppt['n_arms'].values[0])
        n_outcomes = int(df_ppt['n_outcomes'].values[0])

    ## get trial info
    df = df_ppt[cols]
    df = df.sort_values(['subject_id', 'room', 'trial'])
    rooms = []
    for _, d in df.groupby(['subject_id', 'room'], sort=True):
        init_counts = (_counts_from_preset(d['history_0'].iloc[0], n_arms, n_outcomes)
                       if has_history_0 else None)
        rooms.append((
            d['trial'].to_numpy(dtype=int),
            d['action'].fillna(-1).to_numpy(dtype=int),
            d['outcome'].fillna(-1).to_numpy(dtype=int),
            d['terminated'].astype(bool).to_numpy(),
            init_counts,
        ))
    return rooms


# NLL of the choices under, given parameterised model
def _nll_from_arms(arms, design, ell, temp, horizon, init_t):

    n_arms = design['n_arms']
    n_outcomes = design['n_outcomes']
    n_trials = design['n_trials']
    termination_arm = design['termination_arm']
    cost = design['cost']
    alpha = design['alpha']

    ## fresh agent per evaluation: the optimiser moves ell on every call, so a shared cache would never be reused
    agent = make_agent(n_arms, n_outcomes, alpha, ell, termination_arm, cost)

    NLL = 0.0
    for trials, actions, outcomes, terminated, init_counts in arms:

        ## flat prior for the full task
        if init_counts is None:
            counts = np.zeros((n_arms, n_outcomes), dtype=int)
        
        ## or, instructed history for the horizons task
        else:
            counts = init_counts.copy()
        for i in range(len(trials)):
            t = int(trials[i])

            ## before init_t: fill the belief from the actual history - i.e. doesn't contribute to NLL
            if t < init_t:
                if terminated[i]:
                    break
                counts[actions[i], outcomes[i]] += 1
                continue

            h = (n_trials - t) if horizon is None else min(horizon, n_trials - t)
            Q = agent.Q(counts, h)[0]
            probs = _softmax(Q / temp)

            if terminated[i]:
                NLL -= np.log(probs[n_arms])
                break

            NLL -= np.log(probs[actions[i]])
            counts[actions[i], outcomes[i]] += 1
    return NLL


def _rooms_from_df(df_ppt):
    """Hoist each room's choice out of the DataFrame, together with the belief
    states on offer.

    In the rooms task each room is a single choice between the `n_AFC` belief
    states in its `room_history_{k}` columns.
    """
    n_arms = int(df_ppt['n_arms'].values[0])
    n_outcomes = int(df_ppt['n_outcomes'].values[0])
    n_AFC = int(df_ppt['n_AFC'].values[0])

    ## get choice info
    df = df_ppt.sort_values(['subject_id', 'room'])
    rooms = []
    for _, row in df.iterrows():
        counts_array = np.zeros((n_AFC, n_arms, n_outcomes), dtype=int)
        for k in range(n_AFC):
            counts = _counts_from_preset(row[f'room_history_{k}'], n_arms, n_outcomes)
            if counts is not None:
                counts_array[k] = counts
        rooms.append((
            counts_array,
            int(row['action']),
        ))
    return rooms


# NLL of the room choices, given parameterised model
def _nll_from_rooms(rooms, design, ell, temp):

    n_arms = design['n_arms']
    n_outcomes = design['n_outcomes']
    termination_arm = design['termination_arm']
    cost = design['cost']
    alpha = design['alpha']

    ## fresh agent per evaluation, as in _nll_from_arms
    agent = make_agent(n_arms, n_outcomes, alpha, ell, termination_arm, cost)

    NLL = 0.0
    for counts_array, action in rooms:

        ## Q = emp of each room's belief state
        Q = np.array([agent.leaf_value(counts)[0] for counts in counts_array])
        probs = _softmax(Q / temp)

        NLL -= np.log(probs[action])
    return NLL


def nll_emp(df_ppt, ell=1, horizon=None, init_t=0, temp=1):
    """NLL of the yoked choices in `df_ppt` under (ell, temp). Returns a float.

    The scoring counterpart to `run_emp`. Extracts the sequence and the task
    constants, then scores; `_fit_ppt` skips this wrapper and reuses a single
    extraction across every evaluation of the optimiser. Rooms-task data
    (`expt == 'rooms'`) is scored with `_nll_from_rooms`.
    """
    design = _design_from_df(df_ppt)
    if design['expt'] == 'rooms':
        return _nll_from_rooms(_rooms_from_df(df_ppt), design, ell, temp)
    elif design['expt'] == 'arms':
        return _nll_from_arms(_arms_from_df(df_ppt), design,
                            ell, temp, horizon, init_t)

### calculate PFs for info-seeking agent
def pareto_run(n_arms=2, n_outcomes=4, n_trials=6, alphas=(0.1,),
                        termination_arm=True, 
                        init_t=0, n_jobs=1,):

    rows = []
    for alpha in alphas:
        desc = (f"PF: alpha={alpha}")
        
        ## calculate value of root state (i.e. flat prior) with different horizons, sharing one memo
        agent = make_agent(n_arms, n_outcomes, alpha, None, termination_arm, cost=0.0)
        counts = np.zeros((n_arms, n_outcomes), dtype=int)
        for h_remaining in range(n_trials, -1, -1):
            V = agent.V(counts, h_remaining)[0]
            row = {
                'alpha': alpha,
                'V': V,
                'h_remaining': h_remaining,
            }
            rows.append(row)
    df = pd.DataFrame(rows)

    ## calculate ∆V for each horizon
    df['delta_V'] = df['V'].diff(-1)

    ## calculate c* = \frac{V(h+1) - V(h)}{(h+1)V(h+1)-hV(h)} for each horizon
    df['c_star'] = df['delta_V'] / ((df['h_remaining'] + 1) * df['V'] - df['h_remaining'] * df['V'].shift(-1))

    return df
    





