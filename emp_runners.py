import importlib.util as _ilu
import numpy as np
import pandas as pd
from emp_utils import *
from scipy.optimize import bisect, brentq
from scipy.special import softmax as _softmax
from joblib import Parallel, delayed

## EmpBandit lives in a sibling repo and is loaded dynamically. Done once at
## import time so worker processes don't re-import per task.
_spec = _ilu.spec_from_file_location("bandit", "../context_exploration/gym_bandits/bandit.py")
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
EmpBandit = _mod.EmpBandit


def run_emp(df_ppt, ell=1, horizon = None, k=0.0, termination_arm=True, init_t = 0, temp = 1):
    """Run an empowerment-bandit agent yoked to participants' actual trial
    sequences. Returns a tidy DataFrame, one row per (subject_id, room, trial),
    tagged with `ell`, so results from several ell-agents can be pd.concat'd.
    """

    ## extract info from df_ppt ## hack for now
    n_trials = 8
    n_outcomes = 4 
    n_arms = 2
    n_rooms = df_ppt['room'].max()  
    n_actions = n_arms + int(termination_arm)
    terminate_idx = n_arms if termination_arm else None
    alpha = 0.4
    contexts = [(float(alpha), 1.0)]

    cost = 0.0
    button_map = {'blue': 0, 'red': 1}
    outcome_map = {'up': 0, 'left': 1, 'down': 2, 'right': 3}

    records = []

    ## current empowerment (cost-free leaf) for one belief context at one ell
    def _leaf_emp(ctx, e, canon_C):
        agent = EmpowermentAgent(n_arms, n_outcomes, ctx, ell=e,
                                 termination_arm=termination_arm,
                                 )
        ## canon_C is the RAW count matrix; the agent adds the prior alpha
        ## internally (predictive: alpha + counts), so do NOT pre-offset here.
        return agent.leaf_value(canon_C)

    # for pid in df_ppt['subject_id'].unique():
    for p in tqdm(range(len(df_ppt['subject_id'].unique()))):
        pid = df_ppt['subject_id'].unique()[p]
        df_p = df_ppt.loc[df_ppt['subject_id'] == pid]

        for r in range(n_rooms):
            df_pr = df_p.loc[df_p['room'] == r+1]

            prev_action = None

            canon_C = np.zeros((n_arms, n_outcomes), dtype=int)

            ## fill in canon_C with the actual counts from the participant's history up to init_t
            for t in range(init_t):
                row_df = df_pr.loc[df_pr['trial'] == t+1]
                if not row_df['ended_early'].values[0]:
                    actual_action = button_map[row_df['chosen_button'].values[0]]
                    actual_outcome = outcome_map[row_df['outcome'].values[0]]
                    canon_C[actual_action, actual_outcome] += 1
                
                    ## since we're not simulating the agent, fill with nans
                    actual_action = n_arms + 1
                    actual_outcome = np.nan
                    Q_a0 = np.nan
                    Q_a1 = np.nan
                    p_a0 = np.nan
                    p_a1 = np.nan
                    chose_a0 = np.nan
                    chose_a1 = np.nan
                    current_emp = np.nan
                    row = {
                        'subject_id': pid, 'room': r+1, 'trial': t+1, 'ell': ell,
                        'chose_a0': chose_a0, 'chose_a1': chose_a1,
                        'p_choice_a0': p_a0, 'p_choice_a1': p_a1,
                        'current_emp': current_emp,
                        'Q_a0': Q_a0, 'Q_a1': Q_a1,
                        'Q_terminate': np.nan if not termination_arm else np.nan,
                        'p_terminate': np.nan if not termination_arm else np.nan,
                        # 'p_repeat_choice': np.nan if prev_action is None else probs[prev_action],
                    }
                    records.append(row)
                else:
                    # for tt in range(t, init_t):
                    #     row['trial'] = tt + 1
                    #     records.append(row)
                    break

            for t in range(init_t, n_trials):
                row_df = df_pr.loc[df_pr['trial'] == t+1]
                if not row_df.empty:
                    h = (n_trials - t) if horizon is None else min(horizon, n_trials - t)

                    Q = _emp_bellman_Q(n_arms, n_outcomes, contexts, ell,
                                        termination_arm, canon_C, h, cost = cost,
                                        )

                    max_Q = np.nanmax(Q)
                    best_arms = np.where(Q == max_Q)[0]
                    action = int(np.random.choice(best_arms)) if len(best_arms) > 1 else int(best_arms[0])
                    probs = _softmax(Q/temp)

                    ## calculate current emp
                    current_emp = _leaf_emp(contexts, ell, canon_C)

                    if row_df['ended_early'].values[0]:
                        terminated = True
                        actual_action = n_arms+1
                        actual_outcome = np.nan
                        Q_a0 = np.nan
                        Q_a1 = np.nan
                        p_a0 = np.nan
                        p_a1 = np.nan
                        chose_a0 = np.nan
                        chose_a1 = np.nan
                    else:
                        terminated=False
                        actual_action = button_map[row_df['chosen_button'].values[0]]
                        actual_outcome = outcome_map[row_df['outcome'].values[0]]

                        ## update canon_C for next trial
                        canon_C[actual_action, actual_outcome] += 1

                        ## map onto a0 and a1 etc.
                        if df_ppt.loc[(df_ppt['subject_id'] == pid) & (df_ppt['room'] == r+1) & (df_ppt['trial'] == t+1), 'a0'].values[0] == 'blue':
                            Q_a0 = Q[0]
                            Q_a1 = Q[1]
                            p_a0 = probs[0]
                            p_a1 = probs[1]
                            agent_action = 0 if action == 0 else 1
                            chose_a0 = action == 0
                            chose_a1 = action == 1
                        elif df_ppt.loc[(df_ppt['subject_id'] == pid) & (df_ppt['room'] == r+1) & (df_ppt['trial'] == t+1), 'a0'].values[0] == 'red':
                            Q_a0 = Q[1]
                            Q_a1 = Q[0]
                            p_a0 = probs[1]
                            p_a1 = probs[0]
                            agent_action = 0 if action == 1 else 1
                            chose_a0 = action == 1
                            chose_a1 = action == 0
                        row = {
                            'subject_id': pid, 'room': r+1, 'trial': t+1, 'ell': ell,
                            'chose_a0': chose_a0, 'chose_a1': chose_a1,
                            'p_choice_a0': p_a0, 'p_choice_a1': p_a1,
                            'current_emp': current_emp,
                            'Q_a0': Q_a0, 'Q_a1': Q_a1,
                            # 'p_repeat_choice': np.nan if prev_action is None else probs[prev_action],
                        }

                    # row = {
                    #     'subject_id': pid, 'room': r, 'trial': t, 'ell': ell,
                    #     'agent_action': action, 'actual_action': actual_action,
                    #     'actual_outcome': actual_outcome,
                    #     'current_emp': current_emp,
                    #     'agent_matches_ppt': action == actual_action,
                    #     'p_repeat_choice': np.nan if prev_action is None else probs[prev_action],
                    # }

                    if termination_arm:
                        row['Q_terminate'] = Q[-1]
                        row['p_terminate'] = probs[-1]
                    records.append(row)

                    prev_action = actual_action

                    if terminated:
                        break
                else:
                    break

    df_ell = pd.DataFrame.from_records(records)

    # ### concat

    # ## ensure all columns are present
    # for col in df_ell.columns:
    #     if col not in df_ppt.columns:
    #         df_ppt[col] = np.nan
    # df_ppt['ell'] = 'human'
    # df_full = pd.concat([df_ppt, df_ell], ignore_index=True, sort=False)


    ### or, merge
    df_ell = df_ell.rename(columns={'chose_a0': 'ell'+str(ell)+'_chose_a0', 'chose_a1': 'ell'+str(ell)+'_chose_a1', 
                                    'p_choice_a0': 'ell'+str(ell)+'_p_choice_a0', 'p_choice_a1': 'ell'+str(ell)+'_p_choice_a1',
                                    'Q_a0': 'ell'+str(ell)+'_Q_a0', 'Q_a1': 'ell'+str(ell)+'_Q_a1', 
                                    'current_emp': 'ell'+str(ell)+'_current_emp',
                                    'Q_terminate': 'ell'+str(ell)+'_Q_terminate', 'p_terminate': 'ell'+str(ell)+'_p_terminate'})
    df_full = df_ppt.merge(df_ell, on=['subject_id', 'room', 'trial'], how='left')

    return df_full

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


def _tipping_rows_for_history(t, canon_C, history_str,
                              n_arms, n_outcomes, n_trials, alpha, termination_arm,
                              horizon=None,
                              ell_lo=0.001, ell_hi=100, n_ell_samples=200, n_check_samples = 50, eps_tie=1e-8,
                              n_jobs=1):
    """Per-(canonical history, arm, interval) preferred ell-range rows.

    `n_jobs` parallelises the inner n_ell_samples-wide bellman_emp_Q sweep (the
    dominant cost when n_trials is large), not the outer history loop.
    """
    init_alphas = np.full((n_arms, n_outcomes), float(alpha))
    alphas = init_alphas + canon_C
    if horizon is None:
        horizon = n_trials
    h_remaining = np.min([horizon, n_trials - t])

    ## info-seeking agent: ell-free verdict for this history, stamped on every tip row
    info_Q = bellman_info_Q(alphas.copy(), n_arms, n_outcomes, h_remaining, termination_arm)
    info_best_a = int(np.argmin(info_Q))

    ### 1. detect argmax transitions by coarse sampling + bisect within each bracket
    sample_ells = np.logspace(np.log10(ell_lo), np.log10(ell_hi), n_ell_samples)
    if n_jobs == 1:
        sample_Qs = [bellman_emp_Q(alphas.copy(), n_arms, n_outcomes,
                                   h_remaining, termination_arm, e, verbose=False)
                     for e in sample_ells]
    else:
        sample_Qs = Parallel(n_jobs=n_jobs)(
            delayed(bellman_emp_Q)(alphas.copy(), n_arms, n_outcomes,
                                   h_remaining, termination_arm, e, verbose=False)
            for e in sample_ells
        )
    ## per-sample co-argmax SET: every arm within eps_tie of the row max. Using
    ## the set (rather than the integer argmax) catches transitions where a
    ## tied arm joins/leaves the co-best set without flipping the argmax index.
    sample_coargmax = [frozenset(int(a) for a in
                                 np.flatnonzero(np.abs(Q - Q.max()) < eps_tie))
                       for Q in sample_Qs]

    transitions = set()
    for i in range(len(sample_ells) - 1):
        s_lo = sample_coargmax[i]
        s_hi = sample_coargmax[i + 1]
        if s_lo == s_hi:
            continue
        union = s_lo | s_hi
        sym_diff = s_lo ^ s_hi
        for a in sorted(union):
            for b in sorted(union):
                if a >= b:
                    continue
                if a not in sym_diff and b not in sym_diff:
                    ## both arms stay co-best across the bracket; any internal
                    ## crossing wouldn't change the set on either side.
                    continue
                def pref_diff(ell_, _a1=a, _a2=b):
                    Q = bellman_emp_Q(alphas.copy(), n_arms, n_outcomes,
                                      h_remaining, termination_arm, ell_, verbose=False)
                    return Q[_a1] - Q[_a2]
                try:
                    trans = bisect(pref_diff, sample_ells[i], sample_ells[i + 1])

                    ## validity check: only keep if (a, b) actually swap co-best
                    ## status across trans (rejects roots between two suboptimal arms)
                    Q_lo = bellman_emp_Q(alphas.copy(), n_arms, n_outcomes,
                                         h_remaining, termination_arm, trans - 1e-5, verbose=False)
                    Q_hi = bellman_emp_Q(alphas.copy(), n_arms, n_outcomes,
                                         h_remaining, termination_arm, trans + 1e-5, verbose=False)
                    pref_lo = np.flatnonzero(np.abs(Q_lo - Q_lo.max()) < eps_tie)
                    pref_hi = np.flatnonzero(np.abs(Q_hi - Q_hi.max()) < eps_tie)
                    if not (a in pref_lo and b in pref_hi) and not (b in pref_lo and a in pref_hi):
                        continue
                    transitions.add(trans)
                except ValueError:
                    continue

    ### 2. partition [ell_lo, ell_hi] into segments and label each by its co-argmax set
    breakpoints = sorted({ell_lo, ell_hi, *transitions})
    segments = []
    for lo, hi in zip(breakpoints[:-1], breakpoints[1:]):
        mid = np.sqrt(lo * hi)
        Q_mid = bellman_emp_Q(alphas.copy(), n_arms, n_outcomes,
                              h_remaining, termination_arm, mid, verbose=False)
        co_best = frozenset(int(a) for a in
                            np.flatnonzero(np.abs(Q_mid - Q_mid.max()) < eps_tie))
        segments.append((lo, hi, co_best))

    ### 3. merge adjacent segments with the same co-argmax set
    merged = []
    for lo, hi, arms in segments:
        if merged and merged[-1][2] == arms:
            merged[-1] = (merged[-1][0], hi, arms)
        else:
            merged.append((lo, hi, arms))

    ### 4. per arm, collect every merged segment in which it is co-argmax (with tie flag)
    per_arm = {}
    for lo, hi, arms in merged:
        is_tied = len(arms) > 1
        for arm in arms:
            per_arm.setdefault(arm, []).append((lo, hi, is_tied))

    ### 5. per arm, fuse contiguous segments into a single preferred interval;
    ###    has_ties is True if ANY sub-segment of the fused interval had a tie.
    ###    Truly disjoint intervals (gap between them) remain separate rows.
    tip_rows = []
    for arm, intervals in per_arm.items():
        fused = []
        for lo, hi, is_tied in intervals:
            if fused and fused[-1][1] == lo:
                prev_lo, _, prev_tied = fused[-1]
                fused[-1] = (prev_lo, hi, prev_tied or is_tied)
            else:
                fused.append((lo, hi, is_tied))

        for idx, (lo, hi, has_ties) in enumerate(fused):
            tip_row = {
                'history_str': history_str,
                't': t,
                'arm': arm,
                'ell_lo': lo,
                'ell_hi': hi,
                'interval_idx': idx,
                'has_ties': has_ties,
                'info_best_a': info_best_a,
            }
            for a in range(n_arms):
                tip_row[f'info_Q_{a}'] = info_Q[a]
            if termination_arm:
                tip_row['info_Q_terminate'] = info_Q[-1]
            tip_rows.append(tip_row)

            ## debug: scan ells inside the saved interval and flag any where this arm
            ## is no longer in the co-argmax set (signals a missed transition).
            for ell_ in np.geomspace(lo, hi, n_check_samples):
                Q_ = bellman_emp_Q(alphas.copy(), n_arms, n_outcomes,
                                   h_remaining, termination_arm, ell_, verbose=False)
                if np.abs(Q_[arm] - Q_.max()) >= eps_tie:
                    print(f"*** TIE VIOLATION: history={history_str}, lo={lo}, hi={hi}, ell={ell_}, arm={arm}, Q={Q_}")
    return tip_rows


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


def enumerate_tipping_intervals(n_arms=2, n_outcomes=2, n_trials=3, alpha=1.0, termination_arm=True,
                                ell_lo=0.001, ell_hi=100, n_ell_samples=200, n_check_samples=50, eps_tie=1e-8, n_jobs=1):
    """Enumerate per-(canonical history, arm) preferred ell intervals.

    One row per (canonical history, arm, contiguous preferred interval), with
    `has_ties=True` if any sub-segment of the fused interval has multiple
    co-argmax arms.

    `n_jobs` parallelises the inner n_samples-wide bellman_emp_Q sweep within
    each history, not the outer history loop. This targets the t=0 / small-t
    bottleneck where one history's deep-horizon sweep dominates wall time.
    """
    tasks = canonical_states(n_arms, n_outcomes, n_trials)
    batches = [_tipping_rows_for_history(t, C, hs,
                                         n_arms, n_outcomes, n_trials, alpha,
                                         termination_arm, ell_lo, ell_hi, n_ell_samples, n_check_samples, eps_tie,
                                         n_jobs=n_jobs)
               for (t, C, _, hs, _) in tasks]
    return pd.DataFrame([r for batch in batches for r in batch])


def _emp_bellman_Q(n_arms, n_outcomes, ctx, ell, termination_arm, counts, h, cost=0.0,
                   independent_contexts=False):
    """Module-level (picklable) helper: build an EmpowermentAgent for one ell
    and return its horizon-h Q over the given counts. Used by the joblib path.
    `cost` is the per-pull sampling cost (subtracted from arm Q's in the recursion)."""
    agent = EmpowermentAgent(n_arms, n_outcomes, ctx, ell=ell,
                             termination_arm=termination_arm, cost=cost,
                             independent_contexts=independent_contexts)
    return agent.bellman_Q(counts, h)


def enumerate_curves(n_arms, n_outcomes, n_trials, alphas = [0.1],
                     contexts=None, context_prior=None,
                     independent_contexts=False,
                     termination_arm=True, temp=1,
                     horizons = None,
                     ell_lo=0.001, ell_hi=100,
                     n_ell_samples=50,
                     df_max=None, ks=(0.0,),
                     tied_only=False, skip_t0=True, n_jobs=1):
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
    if skip_t0: ## skip first trial (uninteresting + costly)
        sweep_tasks = [task for task in sweep_tasks if task[0] != 0]

    ## function for quickly getting current empowerment for one belief state at one ell
    def _leaf_emp(ctx, e, canon_C):
        agent = EmpowermentAgent(n_arms, n_outcomes, ctx, ell=e,
                                 termination_arm=termination_arm,
                                 independent_contexts=independent_contexts)
        return agent.leaf_value(canon_C)

    
    ### cost info
    
    ## sampling costs to sweep
    ks = [ks] if np.isscalar(ks) else list(ks)
    need_cost = any(float(k) != 0 for k in ks)


    ## get the max achievable empowerment over all histories
    if df_max is None and need_cost:
        sample_ells = np.logspace(np.log10(ell_lo), np.log10(ell_hi), n_ell_samples)
        max_rows = []
        for alpha_label, _, ctx in agent_specs:
            max_emp = np.full(n_ell_samples, -np.inf)
            for (t, _, _, history_str, _) in states:
                canon_C = states_by_t_and_h[(int(t), history_str)]
                for ei, e in enumerate(sample_ells):
                    max_emp[ei] = max(max_emp[ei], _leaf_emp(ctx, e, canon_C))
            for ei, e in enumerate(sample_ells):
                max_rows.append({'alpha': alpha_label, 'ell': e,
                                 'current_emp': max_emp[ei]})
        df_max = pd.DataFrame(max_rows)

    ## convert to table for fast lookup
    cost_tables = None
    if df_max is not None:
        cost_tables = {}
        for key, grp in df_max.groupby(df_max['alpha'].astype(str)):
            cost_tables[key] = (grp['ell'].to_numpy(dtype=float),
                                grp['current_emp'].to_numpy(dtype=float))

    ## actual cost = k * (max achievable emp for this alpha, ell)
    def cost_for(alpha_label, e, k):
        if k == 0 or cost_tables is None:
            return 0.0
        key = str(alpha_label)
        if key not in cost_tables:
            raise KeyError(f"df_max has no rows for alpha={alpha_label!r}")
        ells_arr, emps_arr = cost_tables[key]
        hits = np.flatnonzero(np.isclose(ells_arr, e))
        if len(hits) == 0:
            raise KeyError(f"df_max has no current_emp for alpha={alpha_label!r}, ell={e}")
        return k * float(emps_arr[hits[0]])

    ## big loop
    rows = []
    for alpha_label, context_set, ctx in agent_specs:
        for horizon in horizons:
            
            ## also define the info-seeking agent (ell-free) over this context set
            info_agent = InfoSeekingAgent(n_arms, n_outcomes, ctx, termination_arm,
                                          independent_contexts=independent_contexts)
            for i in tqdm(range(len(sweep_tasks)), desc=f"Enumerating curves (alpha={alpha_label})"):
                t, history_str, e_lo, e_hi = sweep_tasks[i]
                canon_C = states_by_t_and_h[(t, history_str)]
                h_remaining = int(np.min([horizon, n_trials - t]))
                sample_ells = np.logspace(np.log10(e_lo), np.log10(e_hi), n_ell_samples)

                ## info-seeking agent (not parameterised by ell)
                info_Q = info_agent.bellman_Q(canon_C, h_remaining)
                info_best_a = int(np.argmin(info_Q)) # NB seeks to minimise posterior var
                info_probs = _softmax(-info_Q / temp)

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
                    
                ## evaluate Q at each k
                for k in ks:
                    costs = [cost_for(alpha_label, e, k) for e in sample_ells]

                    if n_jobs == 1:
                        Qs = [_emp_bellman_Q(n_arms, n_outcomes, ctx, e,
                                             termination_arm, canon_C, h_remaining, cost=c,
                                             independent_contexts=independent_contexts)
                            for e, c in zip(sample_ells, costs)]
                    else:
                        Qs = Parallel(n_jobs=n_jobs)(
                            delayed(_emp_bellman_Q)(n_arms, n_outcomes, ctx, e,
                                                    termination_arm, canon_C, h_remaining, cost=c,
                                                    independent_contexts=independent_contexts)
                            for e, c in zip(sample_ells, costs)
                        )

                    ## save data
                    for ei in range(len(sample_ells)):
                        e = sample_ells[ei]
                        Q = Qs[ei]
                        probs = _softmax(Q / temp)
                        row = {'alpha': alpha_label, 'context_set': context_set,
                            'horizon': horizon, 'history_str': history_str, 't': t, 'ell': e, 'current_emp': current_emps[ei],
                            'cost': costs[ei], 'k': k, 
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
