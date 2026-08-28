import ast
import importlib.util as _ilu
import numpy as np
import pandas as pd
from emp_utils import *
from scipy.optimize import bisect, brentq, minimize, differential_evolution
from scipy.special import softmax as _softmax
from joblib import Parallel, delayed
import warnings
from tqdm_joblib import tqdm_joblib
from scipy.stats import lognorm

from emp_utils import canonical_states, canonical_count_matrix, array_to_hist, canon_to_concrete
warnings.filterwarnings('ignore')

## EmpBandit lives in a sibling repo and is loaded dynamically. Done once at
## import time so worker processes don't re-import per task.
_spec = _ilu.spec_from_file_location("bandit", "../context_exploration/gym_bandits/bandit.py")
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
EmpBandit = _mod.EmpBandit


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
    contexts = [(float(alpha), 1.0)]

    records = []

    ## determine whether fitting emp or info-seeking agent
    if ell is not None:
        agent_type = 'emp'
        Q_func = _emp_bellman_Q

        ## current empowerment for one belief context at one ell
        def _leaf_emp(ctx, e, canon_C):
            agent = EmpowermentAgent(n_arms, n_outcomes, ctx, ell=e,
                                        termination_arm=termination_arm, cost=cost
                                        )
            return agent.leaf_value(canon_C)
        
    else:
        agent_type = 'info'
        Q_func = _info_bellman_Q
        def _leaf_emp(ctx, e, canon_C):
            agent = InfoSeekingAgent(n_arms, n_outcomes, ctx,
                                        termination_arm=termination_arm, cost=cost
                                        )
            return agent.leaf_value(canon_C)

    # for pid in df_ppt['subject_id'].unique():
    if verbose:
        print(f'Running run_emp for ell={ell}, horizon={horizon}, cost={cost}, termination_arm={termination_arm}, init_t={init_t}, temp={temp}')
        pbar = tqdm(range(len(df_ppt['subject_id'].unique())), desc='Subjects')
    for p in range(len(df_ppt['subject_id'].unique())):
        pid = df_ppt['subject_id'].unique()[p]
        df_p = df_ppt.loc[df_ppt['subject_id'] == pid]

        for r in range(n_rooms):
            df_pr = df_p.loc[df_p['room'] == r]
            canon_C = np.zeros((n_arms, n_outcomes), dtype=int)

            ## fill in canon_C with the actual counts from the participant's history up to init_t
            for t in range(init_t):
                row_df = df_pr.loc[df_pr['trial'] == t]
                if not row_df['terminated'].values[0]:
                    actual_action = row_df['action'].values[0]
                    actual_outcome =row_df['outcome'].values[0]
                    canon_C[actual_action, actual_outcome] += 1
                
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

            for t in range(init_t, n_trials):
                row_df = df_pr.loc[df_pr['trial'] == t]
                if not row_df.empty:
                    h = (n_trials - t) if horizon is None else min(horizon, n_trials - t)

                    Q = Q_func(n_arms, n_outcomes, contexts, ell,
                                        termination_arm, canon_C, h, cost = cost,
                                        )

                    max_Q = np.nanmax(Q)
                    best_arms = np.where(Q == max_Q)[0]
                    action = int(np.random.choice(best_arms)) if len(best_arms) > 1 else int(best_arms[0])
                    probs = _softmax(Q/temp)

                    ## calculate current emp
                    current_emp = _leaf_emp(contexts, ell, canon_C)

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
                        if n_arms > 2:
                            Q_a2 = np.nan
                            p_a2 = np.nan
                            chose_a2 = np.nan
                    else:
                        terminated=False
                        actual_action = row_df['action'].values[0]
                        actual_outcome = row_df['outcome'].values[0]

                        ## update canon_C for next trial
                        canon_C[actual_action, actual_outcome] += 1

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



    ## merge

        # ### concat

    # ## ensure all columns are present
    # for col in df_ell.columns:
    #     if col not in df_ppt.columns:
    #         df_ppt[col] = np.nan
    # df_ppt['ell'] = 'human'
    # df_full = pd.concat([df_ppt, df_ell], ignore_index=True, sort=False)
    df_ell = pd.DataFrame.from_records(records)
    df_ell = df_ell.rename(columns={'chose_a0': 'ell'+str(ell)+'_chose_a0', 'chose_a1': 'ell'+str(ell)+'_chose_a1', 
                                    'chose_a2': 'ell'+str(ell)+'_chose_a2' if n_arms > 2 else np.nan,
                                    'p_choice_a0': 'ell'+str(ell)+'_p_choice_a0', 'p_choice_a1': 'ell'+str(ell)+'_p_choice_a1',
                                    'p_choice_a2': 'ell'+str(ell)+'_p_choice_a2' if n_arms > 2 else np.nan,
                                    'Q_a0': 'ell'+str(ell)+'_Q_a0', 'Q_a1': 'ell'+str(ell)+'_Q_a1', 
                                    'Q_a2': 'ell'+str(ell)+'_Q_a2' if n_arms > 2 else np.nan,
                                    'current_emp': 'ell'+str(ell)+'_current_emp',
                                    'Q_terminate': 'ell'+str(ell)+'_Q_terminate', 'p_terminate': 'ell'+str(ell)+'_p_terminate'})
    df_full = df_ppt.merge(df_ell, on=['subject_id', 'room', 'trial'], how='left')
    return df_full
    

## generate a single synthetic dataset, i.e. an ell agent acting in its own emp bandit env
def gen_emp(n_arms, n_outcomes, n_trials, n_rooms, alpha, ell, cost, horizon, termination_arm=True, diag_histories=None, n_subseq_trials=1, temp=1.0, greedy =False, seed=None):
    """Generate synthetic data from an agent in its own emp bandit env."""
    if ell is not None:
        agent = EmpowermentAgent(n_arms=n_arms, n_outcomes=n_outcomes,
                                contexts=[(float(alpha), 1.0)], ell=ell,
                                termination_arm=termination_arm)
        info_agent = False
    else:
        agent = InfoSeekingAgent(n_arms=n_arms, n_outcomes=n_outcomes,
                                contexts=[(float(alpha), 1.0)],
                                termination_arm=termination_arm)
        info_agent=True
        
    ## define ell_1 agent for scoring expected p(reward)
    ell_1_agent = EmpowermentAgent(n_arms=n_arms, n_outcomes=n_outcomes, cost=cost,
                                contexts=[(float(alpha), 1.0)], ell=1,
                                termination_arm=termination_arm)
    
    ## init data
    sim_out = defaultdict(list)
    
    ## loop through bandit envs
    for r in range(n_rooms):

        ## fresh initialisation of env
        env = make_emp_env(n_arms=n_arms, n_outcomes=n_outcomes, n_trials=n_trials,
                        alpha=alpha, ell=ell, termination_arm=termination_arm,
                        seed=seed)
        
        ## if full expt, initialise counts to 0 and play the whole room
        counts = np.zeros((n_arms, n_outcomes), dtype=int)
        if diag_histories is None:
            history_0 = ()
            t0 = 0
            n_trials_in_room = n_trials

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

        ## loop through trials
        env.reset() ## NEED TO UPDATE: if preset, need to update env to reflect this
        for i in range(n_trials_in_room):
            t = t0 + i

            ## compute Q 
            h = (n_trials - t) if horizon is None else min(horizon, n_trials - t)
            Q = agent.bellman_Q(counts, h)
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


            ## convert termination idx
            if termination_arm and action == n_arms:
                action = -1


            ## calculate max counts fraction - i.e. the action that has been sampled the most, divided by total samples
            max_counts_fraction = np.max(counts.sum(axis=1)) / np.sum(counts) if np.sum(counts) > 0 else 0.0


            ## save
            sim_out['room'].append(r)
            sim_out['trial'].append(t)
            sim_out['history_0'].append(repr(history_0))
            sim_out['action'].append(action)
            sim_out['outcome'].append(outcome)
            sim_out['terminated'].append(action == -1 if termination_arm else False)
            sim_out['counts_array'].append(counts.copy())
            sim_out['max_counts_fraction'].append(max_counts_fraction)
            for a in range(n_arms):
                sim_out[f'Q_{a}'].append(Q[a])
                sim_out[f'p_{a}'].append(probs[a])
            if termination_arm:
                sim_out['Q_terminate'].append(Q[-1])
                sim_out['p_terminate'].append(probs[-1])

            ## update counts
            counts[action, outcome] += 1

            ## score on current probability of reward - i.e. emp_1
            ell_1 = ell_1_agent.leaf_value(counts)
            sim_out['ell_1'].append(ell_1)

            ## but, terminate if the agent chose the termination arm
            if terminated or truncated:
                break

    
    ## add info to dict about params
    sim_out['gen_ell'] = [ell] * len(sim_out['room'])
    sim_out['gen_horizon'] = [horizon] * len(sim_out['room'])
    sim_out['gen_temp'] = [temp] * len(sim_out['room'])

    return sim_out


def _emp_rows_for_history(t, canon_C, canon_counts, history_str, orbit_size, horizon,
                          n_arms, n_outcomes, n_trials, alpha, termination_arm, ells, temp):
    """Per-(canonical history, ell) empowerment / Q / probs / deltas rows."""
    init_alphas = np.full((n_arms, n_outcomes), float(alpha))
    alphas = init_alphas + canon_C
    h_remaining = np.min([horizon, n_trials - t])

    ## info-seeking agent: Bayes-adaptive minimisation of end-state posterior variance (ell-free)
    a0 = alphas.sum(axis=1, keepdims=True)
    current_var = float(np.sum(alphas * (a0 - alphas) / (a0**2 * (a0 + 1))))
    info_Q = bellman_info_Q(alphas.copy(), n_arms, n_outcomes,
                            h_remaining, termination_arm)        
    info_best_a = int(np.argmax(info_Q))
    info_probs = _softmax(info_Q / temp)                        

    rows = []
    for ell in ells:
        current_p = alphas / alphas.sum(axis=1, keepdims=True)
        current_emp = EmpBandit.empowerment(current_p, ell)
        max_reach = np.max(current_p, axis=0)

        Q = bellman_emp_Q(alphas.copy(), n_arms, n_outcomes,
                          h_remaining, termination_arm, ell, verbose=False)
        best_a = np.argmax(Q)

        probs = _softmax(Q / temp)
        policy_entropy = -np.sum(probs * np.log(probs + 1e-12))

        delta_emp = np.zeros(n_arms)
        entropy = np.zeros(n_arms)
        for a in range(n_arms):
            denom = alphas[a].sum()
            expected = 0.0
            for o in range(n_outcomes):
                p_o = alphas[a, o] / denom
                next_alphas = alphas.copy()
                next_alphas[a, o] += 1
                next_p = next_alphas / next_alphas.sum(axis=1, keepdims=True)
                expected += p_o * EmpBandit.empowerment(next_p, ell)
            delta_emp[a] = expected - current_emp
            entropy[a] = EmpBandit.entropy(alphas[a])

        chosen_entropy = entropy[best_a] if best_a < n_arms else np.nan
        chosen_prob = probs[best_a]

        n_untried_arms = np.sum(alphas.sum(axis=1) == init_alphas.sum(axis=1).min())
        n_unobserved_outcomes = np.sum(alphas.sum(axis=0) == init_alphas.sum(axis=0).min())

        least_sampled = np.where(alphas.sum(axis=1) == alphas.sum(axis=1).min())[0]
        if len(least_sampled) > 1:
            p_choose_least_sampled = probs[least_sampled].max()
        else:
            p_choose_least_sampled = probs[least_sampled[0]]

        row = {
            'ell': ell,
            't': t,
            'history': canon_counts,
            'history_str': history_str,
            'orbit_size': orbit_size,
            'current_emp': current_emp,
            'current_var': current_var,
            'p_choose_least_sampled': p_choose_least_sampled,
            'best_a': best_a,
            'info_best_a': info_best_a,
            'policy_entropy': policy_entropy,
            'chosen_prob': chosen_prob,
            'chosen_entropy': chosen_entropy,
            'total_entropy': np.sum(entropy),
            'n_untried_arms': n_untried_arms,
            'n_unobserved_outcomes': n_unobserved_outcomes,
        }
        for a in range(n_arms):
            row[f'Q_{a}'] = Q[a]
            row[f'p_{a}'] = probs[a]
            row[f'delta_emp_{a}'] = delta_emp[a]
            row[f'entropy_{a}'] = entropy[a]
            row[f'info_Q_{a}'] = info_Q[a]
            row[f'info_p_{a}'] = info_probs[a]
        for o in range(n_outcomes):
            row[f'max_reach__{o}'] = max_reach[o]
        if termination_arm:
            row['Q_terminate'] = Q[-1]
            row['p_terminate'] = probs[-1]
            row['info_Q_terminate'] = info_Q[-1]
            row['info_p_terminate'] = info_probs[-1]
        rows.append(row)
    return rows



def enumerate_emp_rows(n_arms=2, n_outcomes=2, n_trials=3, alpha=1.0, termination_arm=True,
                       ells=(0.33, 1.0, 3.0), temp=1.0, n_jobs=1):
    """Enumerate per-(canonical history, ell) empowerment / Q / probs / deltas.

    One row per (ell, canonical history). `n_jobs` controls parallel evaluation
    across canonical histories (joblib); n_jobs=1 runs serially.
    """
    tasks = canonical_states(n_arms, n_outcomes, n_trials)
    if n_jobs == 1:
        print("Running serially...")
        batches = [_emp_rows_for_history(t, C, cc, hs, os,
                                         n_arms, n_outcomes, n_trials, alpha,
                                         termination_arm, ells, temp)
                   for (t, C, cc, hs, os) in tasks]
    else:
        print(f"Running in parallel with n_jobs={n_jobs}...")
        batches = Parallel(n_jobs=n_jobs)(
            delayed(_emp_rows_for_history)(t, C, cc, hs, os,
                                           n_arms, n_outcomes, n_trials, alpha,
                                           termination_arm, ells, temp)
            for (t, C, cc, hs, os) in tasks
        )
    df = pd.DataFrame([r for batch in batches for r in batch])
    df['history_counts'] = df['history']
    df['history_counts_str'] = df['history_str']
    return df



def _emp_bellman_V(n_arms, n_outcomes, ctx, ell, termination_arm, counts, h, cost=0.0,
                   independent_contexts=False):
    """Module-level (picklable) helper: build an EmpowermentAgent for one ell
    and return its horizon-h V over the given counts. Used by the joblib path."""
    agent = EmpowermentAgent(n_arms, n_outcomes, ctx, ell=ell,
                             termination_arm=termination_arm, cost=cost,
                             independent_contexts=independent_contexts)
    return agent.bellman_V(counts, h)

def _info_bellman_V(n_arms, n_outcomes, ctx, ell=None, termination_arm=False, counts=None, h=0, cost=0.0):
    """Module-level (picklable) helper: build an InfoSeekingAgent and return its
    horizon-h V over the given counts. Used by the joblib path.
    """
    agent = InfoSeekingAgent(n_arms, n_outcomes, ctx, termination_arm=termination_arm, cost=cost)
    # return - agent.bellman_Q(counts, h) ## negate because minimising posterior variance
    return agent.bellman_V(counts, h)

def _emp_bellman_Q(n_arms, n_outcomes, ctx, ell, termination_arm, counts, h, cost=0.0,
                   independent_contexts=False):
    """Module-level (picklable) helper: build an EmpowermentAgent for one ell
    and return its horizon-h Q over the given counts. Used by the joblib path.
    `cost` is the per-pull sampling cost (subtracted from arm Q's in the recursion)."""
    agent = EmpowermentAgent(n_arms, n_outcomes, ctx, ell=ell,
                             termination_arm=termination_arm, cost=cost,
                             independent_contexts=independent_contexts)
    return agent.bellman_Q(counts, h)

def _info_bellman_Q(n_arms, n_outcomes, ctx, ell=None, termination_arm=False, counts=None, h=0, cost=0.0):
    """Module-level (picklable) helper: build an InfoSeekingAgent and return its
    horizon-h Q over the given counts. Used by the joblib path.
    (takes ell and cost to ensure compatibility with the _emp_bellman_Q signature, but ignores it)
    """
    agent = InfoSeekingAgent(n_arms, n_outcomes, ctx, termination_arm=termination_arm, cost=cost)
    # return - agent.bellman_Q(counts, h) ## negate because minimising posterior variance
    return agent.bellman_Q(counts, h)


def _get_LML(n_arms, n_outcomes, ctx, ell, termination_arm, counts, h, cost=0.0,
                   independent_contexts=False):
    agent = EmpowermentAgent(n_arms, n_outcomes, ctx, ell=ell,
                             termination_arm=termination_arm, cost=cost,
                             independent_contexts=independent_contexts)
    return agent.marginal_likelihood(counts)


def enumerate_curves(n_arms, n_outcomes, n_trials, alphas = [0.1],
                     contexts=None, context_prior=None,
                     independent_contexts=False,
                     termination_arm=True, temp=1,
                     horizons = None,
                     ell_lo=0.001, ell_hi=100,
                     n_ell_samples=50,
                     df_max=None, costs=(0.0,),
                     tied_only=False, init_t=0, n_jobs=1):
    """Q / softmax-prob curves over ell for canonical histories.

    - enumerate ALL canonical histories at all trials,
      sampling `n_ell_samples` log-spaced ells in `[ell_lo, ell_hi]` for each.
      Coarse but exhaustive picture of how Q/p vary with ell.

    Each curve is produced by one belief agent (`EmpAgent`). Two kinds are
    swept into the same DataFrame for comparison:

    - KNOWN context: one agent per value in `alphas` (the agent knows its
      Dirichlet concentration). `alpha` column holds the numeric value.
    - UNKNOWN context: if `contexts` is given (e.g. [0.1, 1.0]), ONE extra
      agent that infers p(z|h) over that context set and acts on the mixture
      posterior predictive. `context_prior` defaults to uniform. Its rows are
      labelled `alpha='unknown'`.

    `independent_contexts` (default False): controls how the UNKNOWN-context
    agent infers z. When False, a single GLOBAL posterior p(z|h) is shared
    across all arms (all arms drawn from the same prior). When True, each arm
    infers its OWN posterior p(z_a|n_a) from only that arm's counts (arms may be
    drawn from the same or different priors); rows then carry per-arm
    `p_ctx_{arm}_{context}` columns instead of the global `p_ctx_{context}`.

    Both kinds also emit the (now context-aware) info-seeking columns `info_*`,
    computed by an `InfoSeekingAgent` over the same context set.

    `skip_t0` (default True): drop the t=0 (empty `init`) history. There the
    agent has observed nothing and so has equal preference over the actions --
    uninteresting, and the costliest to sweep since its remaining horizon is
    largest. Set False to include it.

    SAMPLING COST: `ks` is a list of cost fractions to sweep; the whole curve
    enumeration is repeated for each `k` and stacked into one DataFrame with `k`
    (and the resulting per-row `cost`) as columns. For a given `k`, each arm pull
    is penalised by `c = k * (max achievable emp for this alpha, ell)`, paid
    recursively on every pull over the horizon (see `EmpAgent.bellman_Q`); the
    terminate action is free and the info-seeking columns stay cost-free.

    The per-(alpha, ell) max empowerment comes from `df_max` (columns `ell`,
    `alpha`, `current_emp`). If `df_max` is None and any `k != 0` is requested,
    it is derived internally as the max `current_emp` over all swept histories
    for each (alpha, ell) -- i.e. the `k=0` pass seeds the costed ones, so a
    single call is self-contained. 

    Returns a long-format DataFrame with one row per (history_str, t, ell,
    agent), columns: alpha, context_set, Q_0, Q_1, ..., Q_terminate (if
    applicable), p_0, p_1, ..., p_terminate, and matching info_* columns.
    `n_jobs` parallelises the inner ell sweep.
    """

    ## generate all canonical histories for the given (n_arms, n_outcomes, n_trials)
    states = canonical_states(n_arms, n_outcomes, n_trials)
    states_by_t_and_h = {(int(t), hs): C for (t, C, _, hs, _) in states}
    if n_jobs != 1:
        print(f"Running in parallel with n_cores = {n_jobs}")
    

    ## if no horizon, set to full horizon
    if horizons is None:
        horizons = [n_trials]

    
    
    ## agents to sweep: 
    
    # known context, i.e. one agent per tested alpha
    agent_specs = [(alpha_val, str(alpha_val), [(float(alpha_val), 1.0)]) # (alpha_label, context_set_str, contexts=[(alpha, prior), ...])
                   for alpha_val in alphas] 
    
    # unknown-context agent
    if contexts is not None:
        if context_prior is None:
            context_prior = [1.0 / len(contexts)] * len(contexts) ## flat prior over contexts 
        ctx_unknown = [(float(a), float(p)) for a, p in zip(contexts, context_prior)]
        context_set_str = 'ctx' + str(tuple(float(a) for a in contexts))
        agent_specs.append(('unknown', context_set_str, ctx_unknown))


    ## define tasks: i.e. sweep all canonical histories with the predefined ell range
    sweep_tasks = [(int(t), history_str, ell_lo, ell_hi)
                    for (t, _, _, history_str, _) in states]
    # if skip_t0: ## skip first trial (uninteresting + costly)
    #     sweep_tasks = [task for task in sweep_tasks if task[0] != 0]
    sweep_tasks = [task for task in sweep_tasks if task[0] >= init_t] ## skip first init_t trials (uninteresting + costly)

    ## function for quickly getting current empowerment for one belief state at one ell
    def _leaf_emp(ctx, e, canon_C, cost):
        agent = EmpowermentAgent(n_arms, n_outcomes, ctx, ell=e, 
                                 termination_arm=termination_arm, cost=cost,
                                 independent_contexts=independent_contexts)
        return agent.leaf_value(canon_C)
    
    ## function for quickly getting current MSE for one belief state
    def _leaf_mse(ctx, canon_C, cost):
        agent = InfoSeekingAgent(n_arms, n_outcomes, ctx, termination_arm=termination_arm, cost=cost)
        return agent.leaf_value(canon_C)

    
    ### cost info
    
    ## sampling costs to sweep
    costs = [costs] if np.isscalar(costs) else list(costs)
    need_cost = any(float(k) != 0 for k in costs)

    ## big loop
    rows = []
    for alpha_label, context_set, ctx in agent_specs:
        for horizon in horizons:
            for i in tqdm(range(len(sweep_tasks)), desc=f"Enumerating curves (alpha={alpha_label})"):
                t, history_str, e_lo, e_hi = sweep_tasks[i]
                canon_C = states_by_t_and_h[(t, history_str)]
                h_remaining = int(np.min([horizon, n_trials - t]))
                sample_ells = np.logspace(np.log10(e_lo), np.log10(e_hi), n_ell_samples)

                ## get LML of history
                LML = _get_LML(n_arms, n_outcomes, ctx, None, termination_arm, canon_C, h_remaining, cost=0.0, independent_contexts=independent_contexts)

                ## posterior prob of context if unknown
                if context_set.startswith('ctx'):
                    agent_tmp = EmpowermentAgent(n_arms, n_outcomes, ctx, ell=sample_ells[0],
                                                 termination_arm=termination_arm,
                                                 independent_contexts=independent_contexts)
                    p_ctx = agent_tmp.context_posterior(canon_C)
                else: # known context, so no posterior needed
                    p_ctx = np.array([1.0])

                ## get info-seeker's current MSE (no cost)
                current_info_cost_free = _leaf_mse(ctx, canon_C, cost=0.0)

                ## loop through costs
                for cost in costs:

                    ## get info-seeker's current MSE
                    current_info = _leaf_mse(ctx, canon_C, cost=cost)

                    ## emp of current belief state for each ell agent, with and without cost
                    current_emps = [_leaf_emp(ctx, e, canon_C, cost=cost) for e in sample_ells]
                    current_emps_cost_free = [_leaf_emp(ctx, e, canon_C, cost=0.0) for e in sample_ells]


                    ### Q values

                    ## info-seeking agent (not parameterised by ell)
                    info_Q = _info_bellman_Q(n_arms, n_outcomes, ctx, None, termination_arm, canon_C, h_remaining, cost=cost)
                    info_best_a = int(np.argmax(info_Q))
                    info_probs = _softmax(info_Q / temp)

                    ## empowerment agents (for each ell)
                    if n_jobs == 1:
                        Qs = [_emp_bellman_Q(n_arms, n_outcomes, ctx, e,
                                             termination_arm, canon_C, h_remaining, cost=cost, 
                                             independent_contexts=independent_contexts)
                            for e in sample_ells]
                    else:
                        Qs = Parallel(n_jobs=n_jobs)(
                            delayed(_emp_bellman_Q)(n_arms, n_outcomes, ctx, e,
                                                    termination_arm, canon_C, h_remaining, cost=cost,
                                                    independent_contexts=independent_contexts)
                            for e in sample_ells
                        )

                    ## save data
                    for ei in range(len(sample_ells)):
                        e = sample_ells[ei]
                        Q = Qs[ei]
                        probs = _softmax(Q / temp)
                        row = {'alpha': alpha_label, 'context_set': context_set,
                            'horizon': horizon, 'history_str': history_str, 't': t, 'ell': e, 
                            'current_emp': current_emps[ei], 'current_info': current_info,
                            'current_emp_cost_free': current_emps_cost_free[ei], 'current_info_cost_free': current_info_cost_free,
                            'cost': cost,
                            'LML': LML,
                            'info_best_a': info_best_a}
                        for a in range(n_arms):
                            row[f'Q_{a}'] = Q[a]
                            row[f'p_{a}'] = probs[a]
                            row[f'info_Q_{a}'] = info_Q[a]
                            row[f'info_p_{a}'] = info_probs[a]
                        if independent_contexts and p_ctx.ndim == 2:
                            ## per-arm context posteriors: p_ctx_{arm}_{context}
                            for a in range(p_ctx.shape[0]):
                                for ctx_i in range(p_ctx.shape[1]):
                                    row[f'p_ctx_{a}_{ctx_i}'] = p_ctx[a, ctx_i]
                        else:
                            for ctx_i in range(len(ctx)):
                                row[f'p_ctx_{ctx_i}'] = p_ctx[ctx_i]
                        if termination_arm:
                            row['Q_terminate'] = Q[-1]
                            row['p_terminate'] = probs[-1]
                            row['info_Q_terminate'] = info_Q[-1]
                            row['info_p_terminate'] = info_probs[-1]
                        rows.append(row)

    return pd.DataFrame(rows)


## Diagnosticity of an observation history: I(A; ell | h) - How much does observing the chosen action tell us about the agent's ell?

##   I(A;ell|h) = H(A|h) - E_ell[ H(A|h,ell) ]
##   H(A|h)     = -sum_a p(a|h) log p(a|h),   p(a|h) = int p(a|h,ell) p(ell) dell
##   H(A|h,ell) = -sum_a p(a|h,ell) log p(a|h,ell)

def ell_prior_samples(n_samples=200, mu=0.0, sigma=1.0, sampling='quantile', seed=None):
    """Equal-weight sample of ell from the lognormal LN(mu, sigma).

    Support is (0, inf) already, so no truncation is needed. LN(0,1) has median
    1, mean exp(0.5) = 1.649, and a heavy right tail (99.5th pct ~ 13).

    `sampling='quantile'` (default) returns the stratified midpoint quantiles
    ppf((m + 0.5)/M): deterministic, reproducible, and far lower variance than
    i.i.d. draws at the same M because p(a|h,ell) is smooth in ell -- a couple of
    hundred quantiles buy what many thousands of random draws would.
    `sampling='random'` draws i.i.d. (use `seed`), kept for MC-error checks.

    Both are EQUAL WEIGHT, so every downstream estimator is a plain mean.
    """
    n_samples = int(n_samples)
    if sampling == 'quantile':
        q = (np.arange(n_samples) + 0.5) / n_samples
        return lognorm.ppf(q, sigma, scale=np.exp(mu))
    elif sampling == 'random':
        return lognorm.rvs(sigma, scale=np.exp(mu), size=n_samples,
                           random_state=seed)


def _neg_p_log_p(p):
    """-sum p log p along the last axis, treating 0 log 0 as 0 (nats)."""
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


def _mi_from_policies(P):
    """(H_marg, H_cond, mi) from an (M, A) array of equal-weight policies.

    Row m is p(.|h, ell_m). `mi` is clipped at 0: it is non-negative in exact
    arithmetic, so any negative value is float noise.
    """
    P = np.asarray(P, dtype=float)
    p_marg = P.mean(axis=0)                       # p(a|h), the ell-marginal
    H_marg = float(_neg_p_log_p(p_marg))          # H(A|h)
    H_cond = float(np.mean(_neg_p_log_p(P)))      # E_ell[H(A|h,ell)]
    return H_marg, H_cond, max(H_marg - H_cond, 0.0)


def _diag_emp_row(t, canon_C, canon_counts, history_str,
                          ell_samples, n_arms, n_outcomes, n_trials, ctx,
                          alpha_label, context_set, independent_contexts,
                          termination_arm, horizon, cost, temp, tie_tol=None):
    """Per-canonical-history diagnosticity row. Module-level so joblib can pickle it.

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
    """
    h_remaining = int(np.min([horizon, n_trials - t]))
    n_actions = n_arms + int(termination_arm)
    n_ell = len(ell_samples)
    tie_tol = float(np.log(3.0)) if tie_tol is None else float(tie_tol)

    ## p(a|h,ell) for each sampled ell
    P = np.empty((n_ell, n_actions))
    Qs = np.empty((n_ell, n_actions))
    for m, ell in enumerate(ell_samples):
        Q = _emp_bellman_Q(n_arms, n_outcomes, ctx, ell, termination_arm,
                           canon_C, h_remaining, cost=cost,
                           independent_contexts=independent_contexts)
        Qs[m] = Q
        P[m] = _softmax(Q / temp)
    H_marg, H_cond, mi = _mi_from_policies(P)
    p_marg = P.mean(axis=0)

    ## which action(s) the diagnosticity comes from. `near_best[m, a]` is "a is
    ## among the best actions at ell_m"; an ell is decisive when that set is a
    ## singleton, which is exactly gap/temp > tie_tol. See TIES.
    near_best = Qs >= Qs.max(axis=1, keepdims=True) - tie_tol * temp
    decisive = near_best.sum(axis=1) == 1
    frac = near_best.mean(axis=0)                        # does NOT sum to 1
    frac_dec = (near_best & decisive[:, None]).mean(axis=0)

    ## how decisive is that choice? top-two Q gap per ell, in temp units.
    gap = _top2_gap(Qs)                                  # >= 0
    gap_temp = gap / temp

    ## get LML
    LML = _get_LML(n_arms, n_outcomes, ctx, None, termination_arm, canon_C, h_remaining, cost=cost,
                   independent_contexts=independent_contexts)

    row = {
        'alpha': alpha_label, 'context_set': context_set,
        'horizon': horizon, 'cost': cost, 'temp': temp,
        't': t, 'history_str': history_str, 'history': canon_counts,
        'H_A_h': H_marg,
        'E_H_A_h_ell': H_cond,
        'mi': mi,
        'mi_bits': mi / np.log(2.0),
        'mi_norm': mi / np.log(n_actions),
        'n_ell_samples': n_ell,
        'LML': LML,
        ## tie diagnostics
        'tie_tol': tie_tol,
        'tie_frac': float(1.0 - decisive.mean()),
        'gap_mean': float(gap.mean()),
        'gap_median': float(np.median(gap)),
        'gap_min': float(gap.min()),
        'gap_mean_temp': float(gap_temp.mean()),
        'gap_median_temp': float(np.median(gap_temp)),
        'p_best_mean': float(P.max(axis=1).mean()),
    }
    for a in range(n_arms):
        row[f'p_marg_{a}'] = p_marg[a]
        row[f'best_a_frac_{a}'] = frac[a]
        row[f'best_a_frac_dec_{a}'] = frac_dec[a]
    if termination_arm:
        row['p_marg_terminate'] = p_marg[-1]
        row['best_a_frac_terminate'] = frac[-1]
        row['best_a_frac_dec_terminate'] = frac_dec[-1]
    return row

def _diag_model_row(t, canon_C, canon_counts, history_str,
                    ell_samples, n_arms, n_outcomes, n_trials, ctx,
                    alpha_label, context_set, independent_contexts,
                    termination_arm, horizon, cost, temp_emp,
                    temp_info, p_model=(0.5, 0.5), tie_tol=None):
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
    fronts. `tie_tol` is a log-odds threshold, default log(3) ~ 1.0986 (winner
    at least 3:1 over the runner-up), and applies to each:

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
    h_remaining = int(np.min([horizon, n_trials - t]))
    n_actions = n_arms + int(termination_arm)

    ### p(a|h,m) for each model

    ## emp agent: one policy per sampled ell, then marginalise over ell
    n_ell = len(ell_samples)
    P_emp = np.empty((n_ell, n_actions))
    Qs_emp = np.empty((n_ell, n_actions))
    for m, ell in enumerate(ell_samples):
        Q = _emp_bellman_Q(n_arms, n_outcomes, ctx, ell, termination_arm,
                           canon_C, h_remaining, cost=cost,
                           independent_contexts=independent_contexts)
        Qs_emp[m] = Q
        P_emp[m] = _softmax(Q / temp_emp)
    H_marg_emp, H_cond_emp, mi_emp = _mi_from_policies(P_emp)
    p_marg_emp = P_emp.mean(axis=0)

    ## info-seeking agent: not parameterised by ell, so a single policy.
    info_Q = _info_bellman_Q(n_arms, n_outcomes, ctx, None, termination_arm,
                             canon_C, h_remaining, cost=cost)
    p_marg_info = _softmax(info_Q / temp_info)
    H_marg_info = float(_neg_p_log_p(p_marg_info))   # H(A|h,info); I(A;ell|h,info) = 0

    ### margins -- see TIES. tie_tol is a log-odds threshold throughout.
    tie_tol = float(np.log(3.0)) if tie_tol is None else float(tie_tol)

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
    gap_info = float(_top2_gap(info_Q)[0])
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

    ## E_m[H(A|h,m)] = sum_m p(m) H(A|h,m)
    p_m = np.asarray(p_model, dtype=float)
    p_m = p_m / p_m.sum()
    H_cond_model = p_m[0] * H_marg_emp + p_m[1] * H_marg_info

    ## p(a|h) = sum_m p(m) p(a|h,m)
    p_marg_model = p_m[0] * p_marg_emp + p_m[1] * p_marg_info
    H_marg_model = float(_neg_p_log_p(p_marg_model))

    ## I(A;M|h) = H(A|h) - E_m[H(A|h,m)]
    mi_model = max(H_marg_model - H_cond_model, 0.0)

    ## I(A;M,ell|h) = H(A|h) - E_m[E_ell[H(A|h,ell,m)]] -- kept for the identity check
    mi_joint = max(H_marg_model - (p_m[0] * H_cond_emp + p_m[1] * H_marg_info), 0.0)

    ## normalise by H(M): I(A;M|h) <= min(H(A), H(M))
    H_M = float(_neg_p_log_p(p_m))
    mi_norm = mi_model / H_M if H_M > 0 else 0.0

    ## get LML (belief-model property, shared by both agents)
    LML = _get_LML(n_arms, n_outcomes, ctx, None, termination_arm, canon_C, h_remaining,
                   cost=cost, independent_contexts=independent_contexts)

    row = {
        'alpha': alpha_label, 'context_set': context_set,
        'horizon': horizon, 'cost': cost, 'temp_emp': temp_emp, 'temp_info': temp_info,
        't': t, 'history_str': history_str, 'history': canon_counts,
        'target': 'model',
        'p_model_emp': p_m[0],
        'H_A_h': H_marg_model,
        'E_m_H_A_h': H_cond_model,
        'H_A_h_emp': H_marg_emp,
        'H_A_h_info': H_marg_info,
        'mi': mi_model,
        'mi_bits': mi_model / np.log(2.0),
        'mi_norm': mi_norm,
        'mi_ell': mi_emp,        # I(A;ell|h,emp) -- the _diag_emp_row quantity
        'mi_joint': mi_joint,    # I(A;M,ell|h) = mi + p(emp)*mi_ell
        'n_ell_samples': n_ell,
        'LML': LML,
        ## tie diagnostics -- within emp (across ell)
        'tie_tol': tie_tol,
        'tie_frac_emp': float(1.0 - decisive_emp.mean()),
        'gap_mean_emp': float(gap_emp.mean()),
        'gap_median_emp': float(np.median(gap_emp)),
        'gap_min_emp': float(gap_emp.min()),
        'gap_mean_temp_emp': float(gap_emp_temp.mean()),
        'gap_median_temp_emp': float(np.median(gap_emp_temp)),
        'p_best_mean_emp': float(P_emp.max(axis=1).mean()),
        ## -- within info
        'gap_info': gap_info,
        'gap_info_temp': gap_info_temp,
        'p_best_info': float(p_marg_info.max()),
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


def enumerate_diagnosticity(n_arms=2, n_outcomes=4, n_trials=6, alphas=(0.1,),
                            contexts=None, context_prior=None,
                            independent_contexts=False,
                            termination_arm=True, temp_emp=1.0, temp_info=1.0,
                            horizons=None, costs=(0.0,),
                            n_samples=200, prior_mu=0.0, prior_sigma=1.0,
                            sampling='quantile', seed=None,
                            init_t=0, n_jobs=1,
                            target='ell', p_model=(0.5, 0.5), tie_tol=None):
    """Diagnosticity of every canonical history, for one of two targets.

    Mirrors `enumerate_curves`: the same canonical-history enumeration, the same
    agent specs (one KNOWN-context agent per value in `alphas`, plus one
    UNKNOWN-context agent labelled `alpha='unknown'` if `contexts` is given), and
    the same horizon / cost sweeps. Where `enumerate_curves` reports the Q/p curve
    at each ell, this reports the single scalar that summarises how much the
    action reveals about ell.

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
    Both emit a `target` column and a comparable `mi`, so the two frames concat.

    The ell sample is drawn ONCE and reused across every history, alpha, horizon
    and cost -- common random numbers, so the resulting mi values are directly
    comparable between histories, which is the point of the score.

    Returns a long DataFrame, one row per (alpha, context_set, horizon, cost, t,
    history_str). Column names match `enumerate_curves` so the two merge on
    ['alpha', 'context_set', 'horizon', 'cost', 't', 'history_str'].

    COST: n_samples Bayes-adaptive Bellman solves per (history, alpha, horizon,
    cost) -- the same shape of cost as `enumerate_curves` with
    n_ell_samples = n_samples. `n_jobs` parallelises over histories.
    """
    if target not in ('ell', 'model'):
        raise ValueError(f"target must be 'ell' or 'model', got {target!r}")
    row_fn = _diag_emp_row if target == 'ell' else _diag_model_row

    ## shared ell sample from the truncated-normal prior
    ell_samples = ell_prior_samples(n_samples, mu=prior_mu, sigma=prior_sigma,
                                    sampling=sampling, seed=seed)

    ## canonical histories, optionally skipping the first init_t trials
    states = canonical_states(n_arms, n_outcomes, n_trials)
    states = [s for s in states if int(s[0]) >= init_t]

    if horizons is None:
        # horizons = [n_trials]
        horizons = [1]

    ## agents to sweep: one per known alpha, plus the unknown-context agent
    agent_specs = [(alpha_val, str(alpha_val), [(float(alpha_val), 1.0)])
                   for alpha_val in alphas]
    if contexts is not None:
        if context_prior is None:
            context_prior = [1.0 / len(contexts)] * len(contexts)   ## flat prior
        ctx_unknown = [(float(a), float(p)) for a, p in zip(contexts, context_prior)]
        context_set_str = 'ctx' + str(tuple(float(a) for a in contexts))
        agent_specs.append(('unknown', context_set_str, ctx_unknown))

    costs = [costs] if np.isscalar(costs) else list(costs)

    rows = []
    for alpha_label, context_set, ctx in agent_specs:
        for horizon in horizons:
            for cost in costs:
                desc = (f"Diagnosticity[{target}] (alpha={alpha_label}, "
                        f"h={horizon}, cost={cost})")
                args = (ell_samples, n_arms, n_outcomes, n_trials, ctx,
                        alpha_label, context_set, independent_contexts,
                        termination_arm, horizon, cost, temp_emp)
                args = args + ((temp_info, p_model, tie_tol) if target == 'model'
                               else (tie_tol,))
                if n_jobs == 1:
                    rows.extend(
                        row_fn(t, C, cc, hs, *args)
                        for (t, C, cc, hs, os) in tqdm(states, desc=desc)
                    )
                else:
                    with tqdm_joblib(tqdm(total=len(states), desc=desc)):
                        rows.extend(Parallel(n_jobs=n_jobs)(
                            delayed(row_fn)(t, C, cc, hs, *args)
                            for (t, C, cc, hs, os) in states
                        ))

    df = pd.DataFrame(rows)
    df['target'] = target
    df['prior_mu'] = prior_mu
    df['prior_sigma'] = prior_sigma
    return df


def diagnosticity_for_counts(C, n_arms=None, n_outcomes=None, n_trials=None,
                             alpha=0.1, contexts=None, context_prior=None,
                             independent_contexts=False,
                             termination_arm=True, temp_emp=1.0, temp_info=1.0, horizon=None, cost=0.0,
                             n_samples=200, prior_mu=0.0, prior_sigma=1.0,
                             sampling='quantile', seed=None,
                             target='ell', p_model=(0.5, 0.5), tie_tol=None):
    """Diagnosticity for ONE arbitrary (non-canonical) count matrix.

    For scoring a real participant's history (`run_emp`) or a simulated one
    (`gen_emp`'s `counts_array`). `C` is canonicalised first: diagnosticity is
    constant on arm/outcome-relabelling orbits, so the canonical value is the
    right one, and the returned `history_str` is the canonical label that joins
    against `enumerate_diagnosticity` / `enumerate_curves` output.

    `n_trials` and `horizon` both default to "t pulls already taken, t more to
    come"; pass them explicitly to match a particular task design.

    `target` ('ell' or 'model'), `temp_info` and `p_model` behave exactly as in
    `enumerate_diagnosticity`.

    Returns the row dict.
    """
    if target not in ('ell', 'model'):
        raise ValueError(f"target must be 'ell' or 'model', got {target!r}")
    C = np.asarray(C, dtype=int)
    if n_arms is None or n_outcomes is None:
        n_arms, n_outcomes = C.shape
    canon_C, _ = canonical_count_matrix(C)
    canon_counts, history_str = array_to_hist(canon_C, n_arms, n_outcomes)
    t = int(canon_C.sum())
    if n_trials is None:
        n_trials = t + (t if horizon is None else horizon)
    if horizon is None:
        horizon = n_trials

    if contexts is None:
        ctx = [(float(alpha), 1.0)]
        alpha_label, context_set = alpha, str(alpha)
    else:
        if context_prior is None:
            context_prior = [1.0 / len(contexts)] * len(contexts)
        ctx = [(float(a), float(p)) for a, p in zip(contexts, context_prior)]
        alpha_label = 'unknown'
        context_set = 'ctx' + str(tuple(float(a) for a in contexts))

    ell_samples = ell_prior_samples(n_samples, mu=prior_mu, sigma=prior_sigma,
                                    sampling=sampling, seed=seed)
    args = (ell_samples, n_arms, n_outcomes, n_trials, ctx,
            alpha_label, context_set, independent_contexts,
            termination_arm, horizon, cost, temp_emp)
    if target == 'ell':
        row = _diag_emp_row(t, canon_C, canon_counts, history_str, *args,
                            tie_tol=tie_tol)
    else:
        row = _diag_model_row(t, canon_C, canon_counts, history_str, *args,
                              temp_info=temp_info, p_model=p_model,
                              tie_tol=tie_tol)
    row['target'] = target
    row['orbit_size'] = orbit_sequence_count(canon_C)
    row['prior_mu'] = prior_mu
    row['prior_sigma'] = prior_sigma
    return row

    







### fitting functions

## parallelised fitting 
def fit_emp(df_ppt, 
            # ell_bounds=(0.1, 5.0), temp_bounds=(0.1, 10.0),
            param_bounds, agent_types=['emp', 'info'],
                           horizon=None, init_t=0,
                           maxiter=200, tol=1e-6, n_jobs=-1, 
                           verbose=True):
    """
    Parallelized version of fit_emp_model using joblib.

    Parameters:
    -----------
    df_ppt : pd.DataFrame
        Participant data
    ell_bounds, temp_bounds : tuples
        Parameter bounds
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
    verbose : bool

    Returns:
    --------
    pd.DataFrame with fitted parameters per subject
    """
    pids = df_ppt['subject_id'].unique()
    df_fits = pd.DataFrame()
    for agent_type in agent_types:
        if agent_type == 'emp':
            ell_bounds, temp_bounds = param_bounds
        elif agent_type == 'info':
            ell_bounds, temp_bounds = (None, None), param_bounds[1]
        with tqdm_joblib(tqdm(total=len(pids), desc="Fitting", disable=not verbose)):
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
                    verbose=False  # avoid duplicate/interleaved worker prints
                )
                for pid in pids
            )
        if df_fits.empty:
            df_fits = pd.DataFrame(results)
        else:
            df_fits = pd.concat([df_fits, pd.DataFrame(results)], ignore_index=True)
    return df_fits
def _fit_ppt(pid, df_ppt, ell_bounds, temp_bounds, horizon,
             init_t, maxiter, tol, verbose,
             popsize=15, mutation=(0.5, 1), recombination=0.7, seed=None):
    """
    Fit model for a single subject using differential evolution.
    """
    
    ## hoist the data out of the DataFrame
    design = _design_from_df(df_ppt)
    rooms = _rooms_from_df(df_ppt)

    def compute_nll(params):
        if len(params) == 1: ## info-seeking agent
            ell = None
            temp = params[0]
        else: ## empowerment agent
            ell, temp = params

        return _nll_from_rooms(rooms, design, ell, temp, horizon, init_t)

    ## emp vs info agent
    if ell_bounds[0] is not None:
        agent_type = 'emp'
        bounds = [ell_bounds, temp_bounds]
    elif ell_bounds[0] is None:
        agent_type = 'info'
        bounds = [temp_bounds]
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
        ell_hat = None
        temp_hat = res.x[0]
    else:
        ell_hat, temp_hat = res.x
    nll = res.fun
    success = res.success

    ## calculate n_trials from the scored count
    n_fit_trials = len(df_ppt.loc[df_ppt['trial'] >= init_t])

    return {
        'subject_id': pid,
        'ell': ell_hat,
        'temp': temp_hat,
        'nll': nll,
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
        'contexts': [(float(df_ppt['alpha'].values[0]), 1.0)],
    }


def _counts_from_preset(history, n_arms, n_outcomes):
    """Turn a preset history -- `(((a, o), count), ...)`, or its repr as read
    back from CSV -- into the count matrix the agent starts the room with.

    Returns None for an absent/empty history, i.e. "start from a flat prior".
    This is the same encoding as the `history` column of the diagnosticity
    tables, so a preset history can be dropped straight into `gen_emp`.
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


def _rooms_from_df(df_ppt):
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
def _nll_from_rooms(rooms, design, ell, temp, horizon, init_t):

    n_arms = design['n_arms']
    n_outcomes = design['n_outcomes']
    n_trials = design['n_trials']
    termination_arm = design['termination_arm']
    cost = design['cost']
    contexts = design['contexts']
    Q_func = _emp_bellman_Q if ell is not None else _info_bellman_Q

    NLL = 0.0
    for trials, actions, outcomes, terminated, init_counts in rooms:

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
            Q = Q_func(n_arms, n_outcomes, contexts, ell,
                       termination_arm, counts, h, cost=cost)
            probs = _softmax(Q / temp)

            if terminated[i]:
                NLL -= np.log(probs[n_arms])
                break

            NLL -= np.log(probs[actions[i]])
            counts[actions[i], outcomes[i]] += 1
    return NLL


def nll_emp(df_ppt, ell=1, horizon=None, init_t=0, temp=1):
    """NLL of the yoked choices in `df_ppt` under (ell, temp). Returns a float.

    The scoring counterpart to `run_emp`. Extracts the sequence and the task
    constants, then scores; `_fit_ppt` skips this wrapper and reuses a single
    extraction across every evaluation of the optimiser.
    """
    return _nll_from_rooms(_rooms_from_df(df_ppt), _design_from_df(df_ppt),
                           ell, temp, horizon, init_t)

### calculate PFs for info-seeking agent
def pareto_run(n_arms=2, n_outcomes=4, n_trials=6, alphas=(0.1,),
                        contexts=None, context_prior=None,
                        independent_contexts=False,
                        termination_arm=True, 
                        init_t=0, n_jobs=1,):

    ## agents to sweep: one per known alpha, plus the unknown-context agent
    agent_specs = [(alpha_val, str(alpha_val), [(float(alpha_val), 1.0)])
                   for alpha_val in alphas]
    if contexts is not None:
        if context_prior is None:
            context_prior = [1.0 / len(contexts)] * len(contexts)   ## flat prior
        ctx_unknown = [(float(a), float(p)) for a, p in zip(contexts, context_prior)]
        context_set_str = 'ctx' + str(tuple(float(a) for a in contexts))
        agent_specs.append(('unknown', context_set_str, ctx_unknown))
    rows = []
    for alpha_label, context_set, ctx in agent_specs:
        desc = (f"PF: alpha={alpha_label}")
        
        ## calculate value of root state (i.e. flat prior) with different horizons
        counts = np.zeros((n_arms, n_outcomes), dtype=int)
        for h_remaining in range(n_trials, -1, -1):
            V = _info_bellman_V(n_arms, n_outcomes, ctx, None,
                                termination_arm, counts, h_remaining, cost=0)
            row = {
                'alpha': alpha_label,
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
    





