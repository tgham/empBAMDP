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

def run_emp(df_ppt, ell=1, horizon = None, init_t = 0, temp = 1, fitting=False, verbose=False):
    """Run an empowerment-bandit agent yoked to participants' actual trial
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

    if not fitting:
        records = []
    elif fitting:
        ppt_choices = []
        p_ppt_choices = []

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
                    if not fitting:
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

                        
                    
                    ## record the row if simulating agents
                    if not fitting:
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

                    ## record choice probs if fitting
                    elif fitting:
                        ppt_choices.append(actual_action)
                        p_ppt_choices.append(probs[actual_action])

                    if terminated:
                        break
                else:
                    break
        pbar.update(1) if verbose else None



    ## merge (if not fitting)
    if not fitting:

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
    else: 
        # NLL = np.sum([-(np.log(p) if c else np.log(1-p)) for p, c in zip(p_a0s, chose_a0s)])
        NLL = np.sum([-(np.log(p)) for p in p_ppt_choices])
        return NLL
    

## generate a single synthetic dataset, i.e. an ell agent acting in its own emp bandit env
def gen_emp(n_arms, n_outcomes, n_trials, n_rooms, alpha, ell, horizon, termination_arm=True, temp=1.0, greedy =False, seed=None):
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
    ell_1_agent = EmpowermentAgent(n_arms=n_arms, n_outcomes=n_outcomes,
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
        counts = np.zeros((n_arms, n_outcomes), dtype=int)
        env.reset()

        ## loop through trials
        for t in range(n_trials):

            ## compute Q 
            h = (n_trials - t) if horizon is None else min(horizon, n_trials - t)
            Q = agent.bellman_Q(counts, h)
            if info_agent:
                Q = -Q  # negate: info-seeking agent minimises expected posterior variance
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
            (_, outcome), reward, terminated, truncated, _ = env.step(action)


            ## convert termination idx
            if termination_arm and action == n_arms:
                action = -1


            ## calculate max counts fraction - i.e. the action that has been sampled the most, divided by total samples
            max_counts_fraction = np.max(counts.sum(axis=1)) / np.sum(counts) if np.sum(counts) > 0 else 0.0


            ## save
            sim_out['room'].append(r)
            sim_out['trial'].append(t)
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


            ## terminate if the agent chose the termination arm
            if terminated or truncated:
                break

            ## else, update counts
            else:
                counts[action, outcome] += 1

                ## score on current probability of reward - i.e. emp_1
                ell_1 = ell_1_agent.leaf_value(counts)
                sim_out['ell_1'].append(ell_1)
    
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
                            h_remaining, termination_arm)        # lower = better
    info_best_a = int(np.argmin(info_Q))
    info_probs = _softmax(-info_Q / temp)                        # negate: minimisation

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



def _emp_bellman_Q(n_arms, n_outcomes, ctx, ell, termination_arm, counts, h, cost=0.0,
                   independent_contexts=False):
    """Module-level (picklable) helper: build an EmpowermentAgent for one ell
    and return its horizon-h Q over the given counts. Used by the joblib path.
    `cost` is the per-pull sampling cost (subtracted from arm Q's in the recursion)."""
    agent = EmpowermentAgent(n_arms, n_outcomes, ctx, ell=ell,
                             termination_arm=termination_arm, cost=cost,
                             independent_contexts=independent_contexts)
    return agent.bellman_Q(counts, h)

def _info_bellman_Q(n_arms, n_outcomes, ctx, ell, termination_arm, counts, h, cost =None):
    """Module-level (picklable) helper: build an InfoSeekingAgent and return its
    horizon-h Q over the given counts. Used by the joblib path.
    (takes ell and cost to ensure compatibility with the _emp_bellman_Q signature, but ignores it)
    """
    agent = InfoSeekingAgent(n_arms, n_outcomes, ctx, termination_arm=termination_arm)
    return - agent.bellman_Q(counts, h) ## negate because minimising posterior variance


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
    def _leaf_emp(ctx, e, canon_C):
        agent = EmpowermentAgent(n_arms, n_outcomes, ctx, ell=e,
                                 termination_arm=termination_arm,
                                 independent_contexts=independent_contexts)
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

                ## info-seeking agent (not parameterised by ell)
                info_Q = _info_bellman_Q(n_arms, n_outcomes, ctx, None, termination_arm, canon_C, h_remaining)
                info_best_a = int(np.argmax(info_Q))
                info_probs = _softmax(info_Q / temp)

                ## emp of current belief state for each ell agent
                current_emps = [_leaf_emp(ctx, e, canon_C) for e in sample_ells]

                ## posterior prob of context if unknown
                if context_set.startswith('ctx'):
                    agent_tmp = EmpowermentAgent(n_arms, n_outcomes, ctx, ell=sample_ells[0],
                                                 termination_arm=termination_arm,
                                                 independent_contexts=independent_contexts)
                    p_ctx = agent_tmp.context_posterior(canon_C)
                else: # known context, so no posterior needed
                    p_ctx = np.array([1.0])
                    
                ## evaluate Q at each cost
                for cost in costs:
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
                            'horizon': horizon, 'history_str': history_str, 't': t, 'ell': e, 'current_emp': current_emps[ei],
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


def _diag_row_for_history(t, canon_C, canon_counts, history_str, orbit_size,
                          ell_samples, n_arms, n_outcomes, n_trials, ctx,
                          alpha_label, context_set, independent_contexts,
                          termination_arm, horizon, cost, temp):
    """Per-canonical-history diagnosticity row. Module-level so joblib can pickle it."""
    h_remaining = int(np.min([horizon, n_trials - t]))
    n_actions = n_arms + int(termination_arm)

    ## p(a|h,ell) for each sampled ell
    P = np.empty((len(ell_samples), n_actions))
    best_a = np.empty(len(ell_samples), dtype=int)
    for m, ell in enumerate(ell_samples):
        Q = _emp_bellman_Q(n_arms, n_outcomes, ctx, ell, termination_arm,
                           canon_C, h_remaining, cost=cost,
                           independent_contexts=independent_contexts)
        P[m] = _softmax(Q / temp)
        best_a[m] = int(np.argmax(Q))

    H_marg, H_cond, mi = _mi_from_policies(P)
    p_marg = P.mean(axis=0)

    ## fraction of sampled ells for which each action is the greedy choice --
    ## shows WHICH action the diagnosticity comes from.
    frac = np.bincount(best_a, minlength=n_actions) / len(ell_samples)

    ## get LML
    LML = _get_LML(n_arms, n_outcomes, ctx, None, termination_arm, canon_C, h_remaining, cost=cost,
                   independent_contexts=independent_contexts)

    row = {
        'alpha': alpha_label, 'context_set': context_set,
        'horizon': horizon, 'cost': cost, 'temp': temp,
        't': t, 'history_str': history_str, 'history': canon_counts,
        'orbit_size': orbit_size,
        'H_A_h': H_marg,
        'E_H_A_h_ell': H_cond,
        'mi': mi,
        'mi_bits': mi / np.log(2.0),
        'mi_norm': mi / np.log(n_actions),
        'n_ell_samples': len(ell_samples),
        'LML': LML,
    }
    for a in range(n_arms):
        row[f'p_marg_{a}'] = p_marg[a]
        row[f'best_a_frac_{a}'] = frac[a]
    if termination_arm:
        row['p_marg_terminate'] = p_marg[-1]
        row['best_a_frac_terminate'] = frac[-1]
    return row


def enumerate_diagnosticity(n_arms=2, n_outcomes=4, n_trials=6, alphas=(0.1,),
                            contexts=None, context_prior=None,
                            independent_contexts=False,
                            termination_arm=True, temp=1.0,
                            horizons=None, costs=(0.0,),
                            n_samples=200, prior_mu=0.0, prior_sigma=1.0,
                            sampling='quantile', seed=None,
                            init_t=0, n_jobs=1):
    """Diagnosticity I(A;ell|h) for every canonical history.

    Mirrors `enumerate_curves`: the same canonical-history enumeration, the same
    agent specs (one KNOWN-context agent per value in `alphas`, plus one
    UNKNOWN-context agent labelled `alpha='unknown'` if `contexts` is given), and
    the same horizon / cost sweeps. Where `enumerate_curves` reports the Q/p curve
    at each ell, this reports the single scalar that summarises how much the
    action reveals about ell.

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
    ## shared ell sample from the truncated-normal prior
    ell_samples = ell_prior_samples(n_samples, mu=prior_mu, sigma=prior_sigma,
                                    sampling=sampling, seed=seed)

    ## canonical histories, optionally skipping the first init_t trials
    states = canonical_states(n_arms, n_outcomes, n_trials)
    states = [s for s in states if int(s[0]) >= init_t]

    if horizons is None:
        horizons = [n_trials]

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
                desc = f"Diagnosticity (alpha={alpha_label}, h={horizon}, cost={cost})"
                args = (ell_samples, n_arms, n_outcomes, n_trials, ctx,
                        alpha_label, context_set, independent_contexts,
                        termination_arm, horizon, cost, temp)
                if n_jobs == 1:
                    rows.extend(
                        _diag_row_for_history(t, C, cc, hs, os, *args)
                        for (t, C, cc, hs, os) in tqdm(states, desc=desc)
                    )
                else:
                    with tqdm_joblib(tqdm(total=len(states), desc=desc)):
                        rows.extend(Parallel(n_jobs=n_jobs)(
                            delayed(_diag_row_for_history)(t, C, cc, hs, os, *args)
                            for (t, C, cc, hs, os) in states
                        ))

    df = pd.DataFrame(rows)
    df['prior_mu'] = prior_mu
    df['prior_sigma'] = prior_sigma
    return df


def diagnosticity_for_counts(C, n_arms=None, n_outcomes=None, n_trials=None,
                             alpha=0.1, contexts=None, context_prior=None,
                             independent_contexts=False,
                             termination_arm=True, temp=1.0, horizon=None, cost=0.0,
                             n_samples=200, prior_mu=0.0, prior_sigma=1.0,
                             sampling='quantile', seed=None):
    """Diagnosticity for ONE arbitrary (non-canonical) count matrix.

    For scoring a real participant's history (`run_emp`) or a simulated one
    (`gen_emp`'s `counts_array`). `C` is canonicalised first: diagnosticity is
    constant on arm/outcome-relabelling orbits, so the canonical value is the
    right one, and the returned `history_str` is the canonical label that joins
    against `enumerate_diagnosticity` / `enumerate_curves` output.

    `n_trials` and `horizon` both default to "t pulls already taken, t more to
    come"; pass them explicitly to match a particular task design.

    Returns the row dict.
    """
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
    row = _diag_row_for_history(t, canon_C, canon_counts, history_str,
                                orbit_sequence_count(canon_C), ell_samples,
                                n_arms, n_outcomes, n_trials, ctx,
                                alpha_label, context_set, independent_contexts,
                                termination_arm, horizon, cost, temp)
    row['prior_mu'] = prior_mu
    row['prior_sigma'] = prior_sigma
    return row

    







### fitting functions

## parallelised fitting 
def fit_emp(df_ppt, 
            # ell_bounds=(0.1, 5.0), temp_bounds=(0.1, 10.0),
            param_bounds, agent_types=['emp', 'info'],
                           horizon=None, init_t=0,
                           maxiter=200, tol=1e-6, n_jobs=-1, verbose=True):
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
    def compute_nll(params):
        if len(params) == 1: ## info-seeking agent
            ell = None
            temp = params[0]
        else: ## empowerment agent
            ell, temp = params

        NLL = run_emp(
            df_ppt=df_ppt,
            ell=ell,
            horizon=horizon,
            init_t=init_t,
            temp=temp,
            fitting=True,
            verbose=False
        )
        return NLL

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
