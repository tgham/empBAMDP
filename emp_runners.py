import itertools
import numpy as np
import pandas as pd
from emp_utils import *
from scipy.optimize import bisect, brentq

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


def run_emp(agent, env, horizon=None, policy='bellman', termination_arm=None, verbose=True):
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

    H = min(horizon, n_trials - t) is the remaining horizon at each trial,
    p(o|a, h) is the posterior predictive of the env's Dirichlet posterior.
    """
    if termination_arm is None:
        termination_arm = bool(getattr(env, 'termination_arm', False))

    if policy == 'bellman':
        Q_fn = lambda alphas, n_a, n_o, h_, e: bellman_emp_Q(
            alphas, n_a, n_o, h_, termination_arm, e, verbose=verbose)
    elif policy == 'uniform_tail':
        if termination_arm:
            raise NotImplementedError("uniform_tail policy does not support termination_arm")
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

def enumerate_emp_histories(n_arms=2, n_outcomes=2, n_trials=3, alpha=1.0, termination_arm = True,
                            ells=(0.33, 1.0, 3.0), temp=1.0):
    """Enumerate canonical (a, o) histories of length 0..n_trials-1.

    Histories are canonicalised under arm-relabel x outcome-relabel
    (S_{n_arms} x S_{n_outcomes} acting on the count matrix). One row per
    equivalence class per ell — the dominant cost (`_bellman_emp_Q`) runs
    once per orbit instead of once per raw sequence.

    For each canonical history h and each ell, computes:
      - current_emp: empowerment of the posterior implied by h
      - Q[a]: Bayes-adaptive optimal value of taking arm a from h with the
        remaining horizon n_trials - len(h)
      - probs[a]: softmax(Q / temp) — the agent's choice probabilities
      - delta_emp[a]: 1-step expected empowerment gain
            E_o~p(o|a,h)[Emp(h u (a,o))] - Emp(h)
      - orbit_size: number of raw (a, o) sequences that canonicalise to h.
        Sum over rows at trial t equals (n_arms * n_outcomes) ** t.

    Note: `prev_action` and `p_repeat` are not emitted — they depend on
    sequence order, which is undefined for a canonical count matrix.

    Returns a long-format DataFrame with one row per (ell, canonical history).
    """
    import importlib.util as _ilu
    from scipy.special import softmax as _softmax

    _spec = _ilu.spec_from_file_location("bandit", "../context_exploration/gym_bandits/bandit.py")
    _mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    EmpBandit = _mod.EmpBandit

    rows = []
    init_alphas = np.full((n_arms, n_outcomes), float(alpha))

    ## Build canonical states incrementally: at each trial t, extend every
    ## canonical history of length t-1 by every (a, o) pair, canonicalise the
    ## resulting count matrix, and dedupe. canon_states[t] maps canonical key
    ## -> canonical count matrix.
    zero_C = np.zeros((n_arms, n_outcomes), dtype=int)
    canon_states = [{tuple(zero_C.flatten().tolist()): zero_C}]
    for t in range(1, n_trials):
        next_states = {}
        for prev_C in canon_states[t - 1].values():
            for a in range(n_arms):
                for o in range(n_outcomes):
                    new_C = prev_C.copy()
                    new_C[a, o] += 1
                    canon_C, key = canonical_count_matrix(new_C)
                    if key not in next_states:
                        next_states[key] = canon_C
        canon_states.append(next_states)

    for ell in ells:
        for t in range(n_trials):
            for key, canon_C in canon_states[t].items():
                alphas = init_alphas + canon_C
                orbit_size = orbit_sequence_count(canon_C)

                current_p = alphas / alphas.sum(axis=1, keepdims=True)
                current_emp = EmpBandit.empowerment(current_p, ell)

                ## calculate max reachabilities
                max_reach = np.max(current_p, axis=0)

                h_remaining = n_trials - t
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

                if best_a < n_arms:
                    chosen_entropy = entropy[best_a]
                else:
                    chosen_entropy = np.nan
                chosen_prob = probs[best_a]

                ## get number of untried arms or unobserved outcomes
                n_untried_arms = np.sum(alphas.sum(axis=1) == init_alphas.sum(axis=1).min())
                n_unobserved_outcomes = np.sum(alphas.sum(axis=0) == init_alphas.sum(axis=0).min())

                ## prob of choose least sampled (if there are ties, choose the max)
                least_sampled = np.where(alphas.sum(axis=1) == alphas.sum(axis=1).min())[0]
                if len(least_sampled) > 1:
                    p_choose_least_sampled = probs[least_sampled].max()
                else:
                    p_choose_least_sampled = probs[least_sampled[0]]

                ## canonical "history" representation: list of (a, o, count)
                ## tuples sorted by (a, o), and a stringified version matching
                ## the format used by get_history_counts / filter_histories.
                canon_counts = tuple(
                    ((int(a), int(o)), int(canon_C[a, o]))
                    for a in range(n_arms) for o in range(n_outcomes)
                    if canon_C[a, o] > 0
                )
                history_str = '-'.join(
                    f'a{a}o{o}:{c}' for ((a, o), c) in canon_counts
                ) or 'init'

                ## find tipping point - i.e. ell at which argmax changes 
                action_pairs = list(itertools.combinations(range(n_arms +1), 2))
                tipping_points = []
                for ai, (a1, a2) in enumerate(action_pairs):
                    if a1 != a2:
                        def pref_diff(ell_):
                            Q = bellman_emp_Q(alphas.copy(), n_arms, n_outcomes,
                                               h_remaining, termination_arm, ell_, verbose=False)
                            diff = Q[a1] - Q[a2]
                            return diff
                        try:
                            ell_switch = bisect(pref_diff, 0.01, 100)
                        except ValueError:
                            continue

                        ## only keep if this is a switch between best actions
                        _eps = 1e-6
                        argmax_lo = np.argmax(bellman_emp_Q(alphas.copy(), n_arms, n_outcomes,
                                                             h_remaining, termination_arm,
                                                             ell_switch - _eps, verbose=False))
                        argmax_hi = np.argmax(bellman_emp_Q(alphas.copy(), n_arms, n_outcomes,
                                                             h_remaining, termination_arm,
                                                             ell_switch + _eps, verbose=False))
                        if argmax_lo == argmax_hi:
                            continue

                        print(f"Tipping point for history {history_str}: arms {a1} and {a2}: {ell_switch}")
                        tipping_points.append((ell_switch, a1, a2))
                        
                row = {
                    'ell': ell,
                    't': t,
                    'history': canon_counts,
                    'history_str': history_str,
                    'orbit_size': orbit_size,
                    'current_emp': current_emp,
                    'p_choose_least_sampled': p_choose_least_sampled,
                    'best_a': best_a,
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
                for o in range(n_outcomes):
                    row[f'max_reach__{o}'] = max_reach[o]
                if termination_arm:
                    row['Q_terminate'] = Q[-1]
                    row['p_terminate'] = probs[-1]
                rows.append(row)
        print()
            

    df = pd.DataFrame(rows)
    ## history_counts / history_counts_str: already canonical here, but kept
    ## for compatibility with filter_histories and downstream notebooks.
    df['history_counts'] = df['history']
    df['history_counts_str'] = df['history_str']

    return df


