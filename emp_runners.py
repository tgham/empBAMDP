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

def run_emp_bamcp(agent, env, verbose=True):
    """Run an agent on the empowerment bandit task for n_trials."""
    n_trials = env.n_trials
    n_arms = env.n_afc
    n_outcomes = env.n_outcomes

    p_choice_history = np.zeros((n_trials, n_arms))
    actions = np.zeros(n_trials, dtype=int)
    outcomes = np.zeros(n_trials, dtype=int)
    rewards = np.zeros(n_trials)

    env.reset()

    for t in range(n_trials):
        env_copy = copy.deepcopy(env)
        env_copy.set_sim(True)
        

        Q = agent.compute_Q(env_copy)
        probs = agent.softmax(Q)
        mean_probs = agent.sampler.mean_probs()
        max_Q = np.nanmax(Q)
        best_arms = np.where(Q == max_Q)[0]
        if len(best_arms) > 1:
            action = int(np.random.choice(best_arms))
        else:            
            action = int(best_arms[0])
        
        p_choice_history[t] = probs

        env.set_sim(False)
        (_, outcome), reward, terminated, truncated, _ = env.step(action)

        actions[t] = action
        outcomes[t] = outcome
        rewards[t] = reward

        if verbose:
            print(f"  trial {t+1:>3}/{n_trials}  Q-values {np.round(Q, 3)}  pulled arm {action}, outcome {outcome}, "
                  f"empowerment reward {reward:.4f}")

        if terminated or truncated:
            break

    return {
        'p_choice': p_choice_history,
        'actions': actions,
        'outcomes': outcomes,
        'rewards': rewards,
        'cumulative_reward': np.cumsum(rewards),
        'true_p_matrix': env.p_matrix.copy(),
        'posterior_p_matrix': env.posterior_p_matrix.copy(),
        'ell': env.ell,
    }


def run_emp(agent, env, horizon=None, policy='bellman', termination_arm=None,
            df_max=None, k=0.0, verbose=True):
    """Run an empowerment-bandit agent with exact Q estimates.

    `policy='bellman'` (default): Bayes-adaptive optimal Q via the recursion
        V(h, 0)   = emp_l(h)
        V(h, d>0) = max( emp_l(h),                                       # terminate
                         max_a sum_o p(o|a,h) V(h u (a,o), d-1) )        # pull arm a
        Q(h, a_1) = sum_o p(o|a_1, h) V(h u (a_1, o), H-1)
        Q(h, terminate) = emp_l(h)
    Subsequent actions are assumed Bayes-optimal -- this is the value BAMCP
    approximates.

    `policy='uniform_tail'`: Q under a uniform random follow-up policy --
    exhaustive enumeration of all (a, o) sequences with posterior-predictive
    weights, averaged uniformly over the action tail. Lower bound on the
    Bellman Q; useful as a comparison baseline.

    `termination_arm`: if True (or auto-detected from `env.termination_arm`),
    the agent has an extra action that immediately collects the current
    empowerment and ends the episode. `uniform_tail` does not currently
    support termination.

    `df_max`/`k`: optional per-pull sampling cost. If `df_max` (columns `ell`,
    `alpha`, `current_emp`) is given, every arm pull is penalised by
    `c = k * current_emp` for this env's (env.alpha, env.ell); the cost is paid
    recursively over the horizon and the terminate action is free. Only the
    `bellman` policy supports a cost.

    H = min(horizon, n_trials - t) is the remaining horizon at each trial,
    p(o|a, h) is the posterior predictive of the env's Dirichlet posterior.
    """
    if termination_arm is None:
        termination_arm = bool(getattr(env, 'termination_arm', False))

    ## per-pull sampling cost for this env's (alpha, ell): c = k * max emp
    cost = 0.0
    if df_max is not None and k != 0:
        m = df_max.loc[(df_max['alpha'].astype(str) == str(env.alpha))
                       & np.isclose(df_max['ell'].astype(float), float(env.ell)),
                       'current_emp']
        if len(m) == 0:
            raise KeyError(f"df_max has no current_emp for alpha={env.alpha!r}, ell={env.ell!r}")
        cost = k * float(m.iloc[0])

    if policy == 'bellman':
        Q_fn = lambda alphas, n_a, n_o, h_, e: bellman_emp_Q(
            alphas, n_a, n_o, h_, termination_arm, e, verbose=verbose, cost=cost)
    elif policy == 'uniform_tail':
        if termination_arm:
            raise NotImplementedError("uniform_tail policy does not support termination_arm")
        if k != 0:
            raise NotImplementedError("uniform_tail policy does not support a sampling cost")
        Q_fn = uniform_tail_emp_Q
    else:
        raise ValueError(f"unknown policy {policy!r}; expected 'bellman' or 'uniform_tail'")

    n_trials = env.n_trials
    n_arms = getattr(env, 'n_arms', env.n_afc - int(termination_arm))
    n_outcomes = env.n_outcomes
    ell = env.ell
    n_actions = n_arms + int(termination_arm)
    terminate_idx = n_arms if termination_arm else None

    Q_history = np.zeros((n_trials, n_actions))
    p_choice_history = np.zeros((n_trials, n_actions))
    p_repeat_choice = np.zeros(n_trials)
    emp_improvement = np.zeros((n_trials, n_actions))
    actions = np.zeros(n_trials, dtype=int)
    outcomes = np.zeros(n_trials, dtype=int)
    rewards = np.zeros(n_trials)

    env.reset()

    ## calculate initial empowerment under flat prior
    flat_prior_p = np.ones((n_arms, n_outcomes)) / n_outcomes
    prev_emp = env.empowerment(flat_prior_p, ell)
    print('initial emp:', prev_emp)

    last_t = n_trials - 1
    for t in range(n_trials):
        h = (n_trials - t) if horizon is None else min(horizon, n_trials - t)

        Q = Q_fn(env.alphas.copy(), n_arms, n_outcomes, h, ell)
        probs = agent.softmax(Q)

        max_Q = np.nanmax(Q)
        best_arms = np.where(Q == max_Q)[0]
        if len(best_arms) > 1:
            action = int(np.random.choice(best_arms))
        else:
            action = int(best_arms[0])

        Q_history[t] = Q
        p_choice_history[t] = probs

        env.set_sim(False)
        (_, outcome), reward, terminated, truncated, _ = env.step(action)

        actions[t] = action
        outcomes[t] = outcome
        rewards[t] = reward
        last_t = t

        emp_improvement[t] = Q / prev_emp
        prev_emp = reward

        if t == 0:
            p_repeat_choice[t] = np.nan
        else:
            last_action = actions[t-1]
            p_repeat_choice[t] = probs[last_action]

        if verbose:
            action_str = 'terminate' if action == terminate_idx else f'arm {action}'
            print(f"  trial {t+1:>3}/{n_trials}  Q={np.round(Q, 4)}  "
                  f"chose {action_str}, outcome {outcome}, "
                  f"empowerment reward {reward:.4f}")

        if terminated or truncated:
            break

    ## trim trailing zeros if the agent terminated early
    keep = last_t + 1
    return {
        'Q': Q_history[:keep],
        'p_choice': p_choice_history[:keep],
        'p_repeat_choice': p_repeat_choice[:keep],
        'actions': actions[:keep],
        'outcomes': outcomes[:keep],
        'emp_improvement': emp_improvement[:keep],
        'rewards': rewards[:keep],
        'cumulative_reward': np.cumsum(rewards[:keep]),
        'true_p_matrix': env.p_matrix.copy(),
        'posterior_p_matrix': env.posterior_p_matrix.copy(),
        'ell': ell,
        'termination_arm': termination_arm,
        'terminated_early': terminated and (action == terminate_idx) if termination_arm else False,
    }

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


def _emp_bellman_Q(n_arms, n_outcomes, ctx, ell, termination_arm, counts, h, cost=0.0):
    """Module-level (picklable) helper: build an EmpowermentAgent for one ell
    and return its horizon-h Q over the given counts. Used by the joblib path.
    `cost` is the per-pull sampling cost (subtracted from arm Q's in the recursion)."""
    agent = EmpowermentAgent(n_arms, n_outcomes, ctx, ell=ell,
                             termination_arm=termination_arm, cost=cost)
    return agent.bellman_Q(counts, h)


def enumerate_curves(n_arms, n_outcomes, n_trials, alphas = [0.1],
                     contexts=None, context_prior=None,
                     df_tip=None, termination_arm=True, temp=1,
                     horizons = None,
                     ell_lo=0.001, ell_hi=100,
                     n_ell_samples=50,
                     df_max=None, ks=(0.0,),
                     tied_only=False, n_jobs=1):
    """Q / softmax-prob curves over ell for canonical histories.

    Two modes, switched by whether `df_tip` is provided:

    - `df_tip is None`: enumerate ALL canonical histories at all trials,
      sampling `n_ell_samples` log-spaced ells in `[ell_lo, ell_hi]` for each.
      Coarse but exhaustive picture of how Q/p vary with ell.

    - `df_tip is not None`: iterate only the (history, t) pairs present in
      `df_tip`, and for each, sample `n_ell_samples` log-spaced ells between
      `min(ell_lo)` and `max(ell_hi)` of that history's df_tip rows. Focuses
      the sweep on each history's interesting range. With `tied_only=True`,
      restrict to histories that have at least one `has_ties=True` row.

    Each curve is produced by one belief agent (`EmpAgent`). Two kinds are
    swept into the same DataFrame for comparison:

    - KNOWN context: one agent per value in `alphas` (the agent knows its
      Dirichlet concentration). `alpha` column holds the numeric value.
    - UNKNOWN context: if `contexts` is given (e.g. [0.1, 1.0]), ONE extra
      agent that infers p(z|h) over that context set and acts on the mixture
      posterior predictive. `context_prior` defaults to uniform. Its rows are
      labelled `alpha='unknown'`.

    Both kinds also emit the (now context-aware) info-seeking columns `info_*`,
    computed by an `InfoSeekingAgent` over the same context set.

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
    single call is self-contained. (Internal derivation requires `df_tip is
    None`, since the per-history ell grids otherwise differ; pass `df_max`
    explicitly in that case.)

    Returns a long-format DataFrame with one row per (history_str, t, ell,
    agent), columns: alpha, context_set, Q_0, Q_1, ..., Q_terminate (if
    applicable), p_0, p_1, ..., p_terminate, and matching info_* columns.
    `n_jobs` parallelises the inner ell sweep.
    """
    states = canonical_states(n_arms, n_outcomes, n_trials)
    states_by_th = {(int(t), hs): C for (t, C, _, hs, _) in states}
    if n_jobs != 1:
        print(f"Running in parallel with n_cores = {n_jobs}")

    ## set horizon
    if horizons is None:
        horizons = [n_trials]

    ## agents to sweep: known (one per alpha) + optional one unknown-context agent.
    ## entries: (alpha_label, context_set_str, contexts=[(alpha, prior), ...])
    agent_specs = [(alpha_val, str(alpha_val), [(float(alpha_val), 1.0)])
                   for alpha_val in alphas]
    if contexts is not None:
        if context_prior is None:
            context_prior = [1.0 / len(contexts)] * len(contexts)
        ctx_unknown = [(float(a), float(p)) for a, p in zip(contexts, context_prior)]
        context_set_str = 'ctx' + str(tuple(float(a) for a in contexts))
        agent_specs.append(('unknown', context_set_str, ctx_unknown))

    if df_tip is None:
        ## sweep all canonical histories with the predefined range
        sweep_tasks = [(int(t), history_str, ell_lo, ell_hi)
                       for (t, _, _, history_str, _) in states]
    else:
        df_tmp = df_tip[df_tip['has_ties']] if tied_only else df_tip
        if len(df_tmp) == 0:
            return pd.DataFrame()
        ranges = (df_tmp.groupby(['history_str', 't'])
                  .agg(ell_min=('ell_lo', 'min'),
                       ell_max=('ell_hi', 'max'))
                  .reset_index())
        sweep_tasks = [(int(r['t']), r['history_str'],
                        float(r['ell_min']), float(r['ell_max']))
                       for _, r in ranges.iterrows()]

    ## current empowerment (cost-free leaf) for one belief context at one ell
    def _leaf_emp(ctx, e, canon_C):
        agent = EmpowermentAgent(n_arms, n_outcomes, ctx, ell=e,
                                 termination_arm=termination_arm)
        counts = canon_C + np.array([a for a, _ in ctx]).reshape(-1, 1)
        return agent.leaf_value(counts)

    ## sampling costs to sweep
    ks = [ks] if np.isscalar(ks) else list(ks)
    need_cost = any(float(k) != 0 for k in ks)

    ## per-pull sampling cost: c = k * (max achievable emp for this alpha, ell),
    ## looked up from df_max. If df_max is absent, derive it from the cost-free
    ## leaf empowerment, taking the max over all swept histories per (alpha, ell).
    if df_max is None and need_cost:
        if df_tip is not None:
            raise ValueError("enumerate_curves: pass df_max explicitly when df_tip "
                             "is set (per-history ell grids prevent internal derivation)")
        sample_ells = np.logspace(np.log10(ell_lo), np.log10(ell_hi), n_ell_samples)
        max_rows = []
        for alpha_label, _, ctx in agent_specs:
            max_emp = np.full(n_ell_samples, -np.inf)
            for (t, _, _, history_str, _) in states:
                canon_C = states_by_th[(int(t), history_str)]
                for ei, e in enumerate(sample_ells):
                    max_emp[ei] = max(max_emp[ei], _leaf_emp(ctx, e, canon_C))
            for ei, e in enumerate(sample_ells):
                max_rows.append({'alpha': alpha_label, 'ell': e,
                                 'current_emp': max_emp[ei]})
        df_max = pd.DataFrame(max_rows)

    ## build per-alpha (ell, current_emp) arrays once for fast lookup
    cost_tables = None
    if df_max is not None:
        cost_tables = {}
        for key, grp in df_max.groupby(df_max['alpha'].astype(str)):
            cost_tables[key] = (grp['ell'].to_numpy(dtype=float),
                                grp['current_emp'].to_numpy(dtype=float))

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

    rows = []
    for alpha_label, context_set, ctx in agent_specs:
        for horizon in horizons:
            ## info-seeking agent (ell-free) over this context set
            info_agent = InfoSeekingAgent(n_arms, n_outcomes, ctx, termination_arm)
            for i in tqdm(range(len(sweep_tasks)), desc=f"Enumerating curves (alpha={alpha_label})"):
                t, history_str, e_lo, e_hi = sweep_tasks[i]
                canon_C = states_by_th[(t, history_str)]
                # h_remaining = horizon - t
                h_remaining = int(np.min([horizon, n_trials - t]))
                sample_ells = np.logspace(np.log10(e_lo), np.log10(e_hi), n_ell_samples)

                ## info-seeking agent: ell-free, same across all sampled ells for this history
                info_Q = info_agent.bellman_Q(canon_C, h_remaining)
                info_best_a = int(np.argmin(info_Q))
                info_probs = _softmax(-info_Q / temp)

                ## current emp (cost-free leaf): k-independent, computed once
                current_emps = [_leaf_emp(ctx, e, canon_C) for e in sample_ells]

                ## sweep the sampling costs: re-evaluate Q at each k, stack rows
                for k in ks:
                    costs = [cost_for(alpha_label, e, k) for e in sample_ells]

                    if n_jobs == 1:
                        Qs = [_emp_bellman_Q(n_arms, n_outcomes, ctx, e,
                                             termination_arm, canon_C, h_remaining, cost=c)
                            for e, c in zip(sample_ells, costs)]
                    else:
                        Qs = Parallel(n_jobs=n_jobs)(
                            delayed(_emp_bellman_Q)(n_arms, n_outcomes, ctx, e,
                                                    termination_arm, canon_C, h_remaining, cost=c)
                            for e, c in zip(sample_ells, costs)
                        )

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
                        if termination_arm:
                            row['Q_terminate'] = Q[-1]
                            row['p_terminate'] = probs[-1]
                            row['info_Q_terminate'] = info_Q[-1]
                            row['info_p_terminate'] = info_probs[-1]
                        rows.append(row)

    return pd.DataFrame(rows)


