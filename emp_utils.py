import itertools
import numpy as np
from plotter import *
import matplotlib.pyplot as plt
from enum import Enum
import seaborn as sns
import scipy
from scipy.spatial.distance import cdist
from scipy.special import softmax
from collections import defaultdict
from IPython.display import display, clear_output
from numba import jit, njit
import pickle 
import pandas as pd
import json
import os
from tqdm.auto import tqdm
import copy
import ast
from itertools import permutations
from scipy.optimize import brentq, bisect


## create empowerment env
def make_emp_env(n_arms=3, n_outcomes=5, n_trials=20, alpha=1.0, ell=1.0,
                 termination_arm=False, seed=None):
    """
    Create an EmpBanditWrapper (MCTS-compatible empowerment bandit).

    Args:
        n_arms:     Number of arms.
        n_outcomes: Number of possible outcomes per arm.
        n_trials:   Number of trials the agent will play.
        alpha:      Dirichlet concentration for the prior over each arm's outcome distribution.
        ell:        Empowerment exponent (agent-side free parameter).
        seed:       Optional random seed (set before the P matrix is sampled).

    Returns:
        An EmpBanditWrapper instance ready for use with BAMCP / MCTS.
    """
    import importlib.util as _ilu

    _spec = _ilu.spec_from_file_location(
        "bandit", "gym_bandits/bandit.py")
    _mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)

    if seed is not None:
        np.random.seed(seed)

    env = _mod.EmpBanditWrapper(
        n_arms=n_arms, n_outcomes=n_outcomes, alpha=alpha, ell=ell, n_trials=n_trials,
        termination_arm=termination_arm, seed=seed,
    )
    return env




def bellman_emp_V(alphas, n_arms, n_outcomes, depth, termination_arm, ell):
    """Bayes-adaptive optimal value with `depth` future pulls remaining.

    V(h, 0)   = emp_l(h)
    V(h, d>0) = max_a sum_o p(o|a, h) * V(h u (a,o), d-1)

    `alphas` is the running Dirichlet posterior; mutated in place and restored.
    """
    posterior_p = alphas / alphas.sum(axis=1, keepdims=True)
    if depth == 0:
        return float(np.sum(np.max(posterior_p, axis=0) ** ell))
    if termination_arm:
        best = float(np.sum(np.max(posterior_p, axis=0) ** ell)) ## value of terminating immediately, i.e. current empowerment, without any more samples
    else:
        best = -np.inf
    for a in range(n_arms):
        denom = alphas[a].sum()
        ev = 0.0
        for o in range(n_outcomes):
            p_o = alphas[a, o] / denom
            alphas[a, o] += 1
            ev += p_o * bellman_emp_V(alphas, n_arms, n_outcomes, depth - 1, termination_arm, ell)
            alphas[a, o] -= 1
        if ev > best:
            best = ev
    return best


def bellman_emp_Q(current_alphas, n_arms, n_outcomes, h, termination_arm, ell, verbose=False):
    """Per-first-action Bayes-adaptive optimal Q with horizon h.

    Q[a_1] = sum_o p(o|a_1, h) * V(h u (a_1, o), depth = h-1).
    Subsequent actions are taken to maximise expected end-state empowerment
    given the resulting belief, i.e. argmax_a inside bellman_emp_V.
    """
    Q = np.zeros(n_arms + termination_arm)
    work = current_alphas.astype(float).copy()
    for a in range(n_arms):
        denom = work[a].sum()
        for o in range(n_outcomes):
            p_o = work[a, o] / denom
            work[a, o] += 1
            V = bellman_emp_V(work, n_arms, n_outcomes, h - 1, termination_arm, ell)
            Q[a] += p_o * V
            if verbose:
                print(f"action {a}, outcome {o}, p(o|a,h)={p_o:.4f}, V(h u (a,o), h-1)={V:.4f}")
            work[a, o] -= 1

    ## Q(terminate) is just the immediate empowerment under the current belief, no future pulls
    if termination_arm:
        posterior_p = current_alphas / current_alphas.sum(axis=1, keepdims=True)
        Q[-1] = float(np.sum(np.max(posterior_p, axis=0) ** ell))
    return Q

def bellman_info_V(alphas, n_arms, n_outcomes, depth, termination_arm):
    """Bayes-adaptive *minimal* end-state posterior variance with `depth` pulls remaining.

    The info-seeking agent wants to reduce uncertainty, so it MINIMISES the final
    posterior variance:

    V(h, 0)   = MSE(h), i.e. the total variance of the agent's posterior
    V(h, d>0) = min_a sum_o p(o|a, h) * V(h u (a,o), d-1)

    `alphas` is the running Dirichlet posterior; mutated in place and restored.
    """
    a0 = alphas.sum(axis=1, keepdims=True)
    if depth == 0:
        return float(np.sum(alphas * (a0 - alphas) / (a0**2 * (a0 + 1))))
    if termination_arm:
        ## value of terminating now: keep the current variance, take no more samples
        best = float(np.sum(alphas * (a0 - alphas) / (a0**2 * (a0 + 1))))
    else:
        best = np.inf
    for a in range(n_arms):
        denom = alphas[a].sum()
        ev = 0.0
        for o in range(n_outcomes):
            p_o = alphas[a, o] / denom
            alphas[a, o] += 1
            ev += p_o * bellman_info_V(alphas, n_arms, n_outcomes, depth - 1, termination_arm)
            alphas[a, o] -= 1
        if ev < best:
            best = ev
    return best


def bellman_info_Q(current_alphas, n_arms, n_outcomes, h, termination_arm, verbose=False):
    """Per-first-action Bayes-adaptive info-seeking Q with horizon h.

    Q[a_1] is the expected end-state posterior variance after pulling arm a_1 now and
    acting variance-minimally thereafter (via bellman_info_V). LOWER IS BETTER -- the
    info-seeking agent prefers the action that drives the final posterior variance down.
    Not parameterised by ell.
    """
    Q = np.zeros(n_arms + termination_arm)
    work = current_alphas.astype(float).copy()
    for a in range(n_arms):
        denom = work[a].sum()
        for o in range(n_outcomes):
            p_o = work[a, o] / denom
            work[a, o] += 1
            V = bellman_info_V(work, n_arms, n_outcomes, h - 1, termination_arm)
            Q[a] += p_o * V
            if verbose:
                print(f"action {a}, outcome {o}, p(o|a,h)={p_o:.4f}, V(h u (a,o), h-1)={V:.4f}")
            work[a, o] -= 1

    ## Q(terminate) is just the current posterior variance, no future pulls
    if termination_arm:
        a0 = current_alphas.sum(axis=1, keepdims=True)
        Q[-1] = float(np.sum(current_alphas * (a0 - current_alphas) / (a0**2 * (a0 + 1))))
    return Q



def canonical_count_matrix(C, _perm_cache={}):
    """Return the lex-MAX count matrix in the orbit of C under
    S_{n_arms} x S_{n_outcomes} acting by row and column permutation,
    along with a hashable key.

    Lex-max in row-major order packs counts towards low arm/outcome indices,
    so the canonical form always uses a0, o0 first and increments only when a
    genuinely new arm/outcome is required. Equivalent histories therefore
    share the same canonical key, and the canonical labels are the smallest
    that the multiset of counts admits.

    Two count matrices in the same orbit induce identical posteriors up to
    arm- and outcome-relabelling, so the per-history quantities computed by
    enumerate_emp_histories (current_emp, Q, probs, delta_emp, entropy) are
    constant on orbits.
    """
    n_arms, n_outcomes = C.shape
    cache_key = (n_arms, n_outcomes)
    if cache_key not in _perm_cache:
        _perm_cache[cache_key] = (
            [np.array(p) for p in itertools.permutations(range(n_arms))],
            [np.array(p) for p in itertools.permutations(range(n_outcomes))],
        )
    row_perms, col_perms = _perm_cache[cache_key]

    best_flat = None
    best_C = None
    for rp in row_perms:
        Cr = C[rp]
        for cp in col_perms:
            cand = Cr[:, cp]
            flat = tuple(cand.flatten().tolist())
            if best_flat is None or flat > best_flat:
                best_flat = flat
                best_C = cand
    return best_C, best_flat


def orbit_size(canon_C):
    """Number of distinct count matrices in the orbit of canon_C under
    S_{n_arms} x S_{n_outcomes}, i.e. (n_arms! * n_outcomes!) / |Stab(C)|."""
    n_arms, n_outcomes = canon_C.shape
    from math import factorial
    stab = 0
    for rp in itertools.permutations(range(n_arms)):
        Cr = canon_C[np.array(rp)]
        for cp in itertools.permutations(range(n_outcomes)):
            if np.array_equal(Cr[:, np.array(cp)], canon_C):
                stab += 1
    return (factorial(n_arms) * factorial(n_outcomes)) // stab


def orbit_sequence_count(canon_C):
    """Number of raw (a, o) sequences (orderings of the count matrix entries)
    that canonicalise to canon_C. Equals
        (orbit_size of matrix) * (multinomial t! / Π C[a,o]!)
    where t = canon_C.sum(). This is the multiplicity to record per
    canonical row so that summing orbit_size at trial t recovers
    (n_arms * n_outcomes) ** t.
    """
    from math import factorial
    t = int(canon_C.sum())
    multinom = factorial(t)
    for c in canon_C.flatten():
        multinom //= factorial(int(c))
    return orbit_size(canon_C) * multinom


def canonical_states(n_arms, n_outcomes, n_trials):
    """Enumerate canonical (a, o) histories of length 0..n_trials-1.

    Histories are canonicalised under arm-relabel x outcome-relabel
    (S_{n_arms} x S_{n_outcomes} acting on the count matrix). Returns a flat
    list of (t, canon_C, canon_counts, history_str, orbit_size) tuples — one
    entry per canonical history at each trial, ready to feed into a parallel map.
    """
    zero_C = np.zeros((n_arms, n_outcomes), dtype=int)
    states = [{tuple(zero_C.flatten().tolist()): zero_C}]
    for t in range(1, n_trials):
        next_states = {}
        for prev_C in states[t - 1].values():
            for a in range(n_arms):
                for o in range(n_outcomes):
                    new_C = prev_C.copy()
                    new_C[a, o] += 1
                    canon_C, key = canonical_count_matrix(new_C)
                    if key not in next_states:
                        next_states[key] = canon_C
        states.append(next_states)

    tasks = []
    for t, st in enumerate(states):
        for canon_C in st.values():
            canon_counts = tuple(
                ((int(a), int(o)), int(canon_C[a, o]))
                for a in range(n_arms) for o in range(n_outcomes)
                if canon_C[a, o] > 0
            )
            history_str = '-'.join(f'a{a}o{o}:{c}' for ((a, o), c) in canon_counts) or 'init'
            tasks.append((t, canon_C, canon_counts, history_str, orbit_sequence_count(canon_C)))
    return tasks




## define unordered 'history_counts', i.e. sufficient statistic for belief state
def get_history_counts(history):
    if history == 'init':
        return 'init'
    pairs = history
    counts = {}
    for pair in pairs:
        counts[pair] = counts.get(pair, 0) + 1
    sorted_counts = tuple(sorted(counts.items()))
    str_counts = '-'.join(f'a{a}o{o}:{count}' for ((a, o), count) in sorted_counts)
    return sorted_counts, str_counts



def filter_histories(df, canonicalize=True):
    """Add cleaning columns to df — does not drop rows or build any
    plotting state.

    Adds:
      - 'history_canon_counts_str' (when canonicalize=True): each history's
        count multiset reduced to its lex-min representative under
        (arm-label permutation) x (global outcome-label permutation).
        Histories sharing a belief state collapse to the same key.
      - 'disagree': True when the ells disagree on `best_a` within the
        row's (t, canonical-history) group (or (t, history_counts_str)
        if canonicalize=False).

    't' is coerced to int; rows where the coercion fails (e.g. a stray
    corrupted CSV row) are dropped.
    """

    def _parse_counts(hc):
        if isinstance(hc, str):
            try:
                return ast.literal_eval(hc)
            except Exception:
                return ()
        return hc or ()

    out = df.copy()
    out = out[pd.to_numeric(out['t'], errors='coerce').notna()]
    out['t'] = out['t'].astype(int)

    p_cols = [c for c in out.columns if str(c).startswith('p_') and str(c)[2:].isdigit()]
    n_arms = len(p_cols)

    canon_cache = {}

    def _canon_counts_str(counts_tuple):
        key = tuple(sorted(counts_tuple)) if counts_tuple else ()
        if key in canon_cache:
            return canon_cache[key]
        counts = dict(counts_tuple) if counts_tuple else {}
        if not counts:
            canon_cache[key] = ('', {a: a for a in range(n_arms)})
            return canon_cache[key]
        arms_used = sorted({a for a, _ in counts})
        outcomes_used = sorted({o for _, o in counts})
        best = None
        best_arm_map = None
        for arm_perm in permutations(range(len(arms_used))):
            arm_map = dict(zip(arms_used, arm_perm))
            for out_perm in permutations(range(len(outcomes_used))):
                out_map = dict(zip(outcomes_used, out_perm))
                s = '-'.join(
                    f'a{a}o{o}:{c}'
                    for (a, o), c in sorted(
                        ((arm_map[a], out_map[o]), c)
                        for (a, o), c in counts.items()
                    )
                )
                if best is None or s < best:
                    best = s
                    best_arm_map = arm_map
                    
        used_orig = set(best_arm_map.keys())
        unused_orig = [a for a in range(n_arms) if a not in used_orig]
        used_canon = set(best_arm_map.values())
        unused_canon = [c for c in range(n_arms) if c not in used_canon]
        
        full_arm_map = best_arm_map.copy()
        for o, c in zip(unused_orig, unused_canon):
            full_arm_map[o] = c
            
        canon_to_orig = {c: o for o, c in full_arm_map.items()}
        
        canon_cache[key] = (best, canon_to_orig)
        return canon_cache[key]

    if canonicalize:
        results = out['history_counts'].apply(
            lambda hc: _canon_counts_str(_parse_counts(hc))
        )
        out['history_canon_counts_str'] = results.apply(lambda r: r[0])
        arm_maps = results.apply(lambda r: r[1]).tolist()
        
        if n_arms > 0:
            p_matrix = out[[f'p_{i}' for i in range(n_arms)]].values
            q_matrix = out[[f'Q_{i}' for i in range(n_arms)]].values
            delta_matrix = out[[f'delta_emp_{i}' for i in range(n_arms)]].values
            entropy_matrix = out[[f'entropy_{i}' for i in range(n_arms)]].values
            
            for a in range(n_arms):
                orig_indices = np.array([m.get(a, a) for m in arm_maps])
                out[f'canon_p_{a}'] = p_matrix[np.arange(len(out)), orig_indices]
                out[f'canon_Q_{a}'] = q_matrix[np.arange(len(out)), orig_indices]
                out[f'canon_delta_emp_{a}'] = delta_matrix[np.arange(len(out)), orig_indices]
                out[f'canon_entropy_{a}'] = entropy_matrix[np.arange(len(out)), orig_indices]

            ## check which Q is the best out of the canon_Qs and Q_terminate
            if f'Q_terminate' in out.columns:
                termination_idx = n_arms 
                out[f'canon_best_a'] = out[[f'canon_Q_{i}' for i in range(n_arms)] + [f'Q_terminate']
                                        ].idxmax(axis=1).apply(lambda s: int(s.split('_')[2]) if s.startswith('canon_Q') else termination_idx)
            else:
                 out[f'canon_best_a'] = out[[f'canon_Q_{i}' for i in range(n_arms)]].idxmax(axis=1).apply(lambda s: int(s.split('_')[2]))

            ## check which action has the best delta_emp
            out[f'canon_best_delta_emp'] = out[[f'canon_delta_emp_{i}' for i in range(n_arms)]].idxmax(axis=1).apply(lambda s: int(s.split('_')[3]))
            out[f'max_delta_emp'] = out[[f'canon_delta_emp_{i}' for i in range(n_arms)]].max(axis=1)
            

                
        group_col = 'history_canon_counts_str'
        disagree = out.groupby(['t', group_col])['canon_best_a'].nunique() > 1
    else:
        group_col = 'history_counts_str'
        disagree = out.groupby(['t', group_col])['best_a'].nunique() > 1


    out = out.merge(
        disagree.rename('disagree').reset_index(), on=['t', group_col]
    )

    ## sanity check: for each history_canon_counts_str, canon_p_a.nunique() should be 1 for each a, and canon_Q_a.unique() should be 1 for each a
    for a in range(n_arms):
        for ell in out['ell'].unique():
            subset = out[out['ell'] == ell]

            ## round to 5 dp to avoid floating point issues 
            subset[f'canon_p_{a}'] = subset[f'canon_p_{a}'].round(5)
            subset[f'canon_Q_{a}'] = subset[f'canon_Q_{a}'].round(5)
            p_nunique = subset.groupby(['t', 'history_canon_counts_str'])[f'canon_p_{a}'].nunique()
            q_nunique = subset.groupby(['t', 'history_canon_counts_str'])[f'canon_Q_{a}'].nunique()

        assert (p_nunique <= 1).all(), f"p_{a} not unique within canon groups: {p_nunique[p_nunique > 1]}"
        assert (q_nunique <= 1).all(), f"Q_{a} not unique within canon groups: {q_nunique[q_nunique > 1]}"

    ## define choice probabilities wrt/ canonicalised actions, e.g. p0 is p(choose a0), where a0 is the first action in the history_counts
    return out

## find ell at which the preference between two options switches
def ell_tip(agent, env, a1, a2):
    def pref_diff(ell):
        env.ell = ell
        Q = agent.compute_Q(env)
        return Q[a1] - Q[a2]

    try:
        ell_switch = brentq(pref_diff, 0.01, 100)
    except ValueError:
        ell_switch = None

    return ell_switch



