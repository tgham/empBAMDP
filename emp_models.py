"""Bayes-adaptive belief agents for the empowerment bandit.

The one place Q and V values are computed. Each agent holds a Dirichlet belief
over every arm's outcome distribution and solves the finite-horizon
Bayes-adaptive Bellman recursion over count matrices, memoised on
(counts, depth). Leaves return one value per parameter setting, so a single
traversal serves a whole grid of ells.

Agents own their memo, and callers own the agents: keep one alive for as long
as its cache should be reused (e.g. one per alpha in a sweep, one per NLL
evaluation while fitting), then let it go, which frees the memory.
"""
import numpy as np
from scipy.special import gammaln, logsumexp


class EmpAgent:
    """Bayes-adaptive belief agent over a symmetric Dirichlet prior.

    Each arm's outcome distribution has its own Dir(alpha * 1_K) prior, so after
    counts n the posterior predictive is
    p(o|a,h) = (alpha + n_{a,o}) / sum_o'(alpha + n_{a,o'}).

    State carried through the recursion is the integer count matrix `counts`
    (n_arms x n_outcomes). The recursion scaffold (expectation over outcomes
    weighted by the predictive, max over arms, termination arm) is shared;
    subclasses supply the objective via `leaf_value`, which returns an
    (n_params,) vector -- one entry per parameter setting (e.g. per ell).

    MEMOISATION: V, Q and log_policy are cached per agent. V only depends on the
    count matrix and the remaining depth, so the subtree below a history is
    reused by the next trial, by other histories and by other horizons.
    Returned arrays are cached and read-only. The cache lives as long as the
    agent: build a fresh one (or call `clear_cache`) when it shouldn't outlive
    a computation.
    """

    n_params = 1

    def __init__(self, n_arms, n_outcomes, alpha, termination_arm=False, cost=0.0):
        self.n_arms = n_arms
        self.n_outcomes = n_outcomes
        self.termination_arm = bool(termination_arm)
        self.n_actions = n_arms + int(self.termination_arm)

        ## per-pull sampling cost, paid on every arm pull throughout the horizon.
        self.cost = float(cost)

        ## symmetric Dirichlet concentration
        self.alpha = float(alpha)

        self.clear_cache()

    def clear_cache(self):
        self._V, self._Q, self._logP = {}, {}, {}

    def leaf_value(self, counts):
        """(n_params,) value of ending in belief state `counts`."""
        raise NotImplementedError

    ## ---- belief model -------------------------------------------------
    def marginal_likelihood(self, counts):
        """log p(h) = sum_a [log B(alpha + n_a) - log B(alpha * 1_K)]."""
        K, a = self.n_outcomes, self.alpha
        num = gammaln(a + counts).sum()                        # sum_{a,o} gammaln(alpha + n_{a,o})
        den = gammaln(K * a + counts.sum(axis=1)).sum()        # sum_a gammaln(K*alpha + n_a)
        logB0 = K * gammaln(a) - gammaln(K * a)                # log B(alpha * 1_K), per arm
        return float(num - den - self.n_arms * logB0)

    def predictive(self, counts):
        """Posterior predictive matrix p(o|a,h), shape (A, O)."""
        a = self.alpha + counts
        return a / a.sum(axis=1, keepdims=True)

    ## ---- memoised Bellman recursion -----------------------------------
    def V(self, counts, depth):
        """(n_params,) value with `depth` future pulls remaining."""
        return self._value(np.array(counts, dtype=np.int64), int(depth))

    def _value(self, counts, depth):
        ## `counts` is an int64 working array, mutated and restored in place
        k = (counts.tobytes(), depth)
        v = self._V.get(k)
        if v is not None:
            return v

        ## depth=0 is just the value of the current belief
        if depth == 0:
            v = self.leaf_value(counts)
        else:
            p = self.predictive(counts)

            ## floor is the value of terminating now
            v = self._value(counts, 0) if self.termination_arm else np.full(self.n_params, -np.inf)
            for a in range(self.n_arms):
                ev = np.zeros(self.n_params)
                for o in range(self.n_outcomes):
                    counts[a, o] += 1
                    ev += p[a, o] * self._value(counts, depth - 1)
                    counts[a, o] -= 1
                v = np.maximum(v, ev)
        v.flags.writeable = False
        self._V[k] = v
        return v

    def Q(self, counts, h):
        """(n_params, n_actions) per-first-action Q with horizon h >= 1.

        Q[:, a] = sum_o p(o|a,h) V(counts u (a,o), h-1); Q[:, terminate] = leaf(counts).
        The caller's `counts` is left untouched.
        """
        counts = np.asarray(counts, dtype=np.int64)
        k = (counts.tobytes(), int(h))
        Q = self._Q.get(k)
        if Q is None:
            if h < 1:
                raise ValueError(f"Q needs a horizon of at least 1, got h={h}")
            work = counts.copy()
            p = self.predictive(work)
            Q = np.zeros((self.n_params, self.n_actions))
            for a in range(self.n_arms):
                for o in range(self.n_outcomes):
                    work[a, o] += 1
                    Q[:, a] += p[a, o] * self._value(work, h - 1)
                    work[a, o] -= 1
            if self.termination_arm:
                Q[:, -1] = self._value(work, 0)
            Q.flags.writeable = False
            self._Q[k] = Q
        return Q

    def log_policy(self, counts, h, temp):
        """(n_params, n_actions) log softmax(Q / temp)."""
        counts = np.asarray(counts, dtype=np.int64)
        k = (counts.tobytes(), int(h), float(temp))
        logP = self._logP.get(k)
        if logP is None:
            z = self.Q(counts, h) / temp
            logP = z - logsumexp(z, axis=1, keepdims=True)
            logP.flags.writeable = False
            self._logP[k] = logP
        return logP


class EmpowermentAgent(EmpAgent):
    """Maximises end-state empowerment Emp_ell = sum_o (max_a p(o|a))^ell.

    `ell` may be a scalar or an array; values come back with one row per ell.
    The predictive does not depend on ell -- only the leaf's `** ell` and the
    max over actions do -- so the whole grid is solved in one traversal.
    """

    def __init__(self, n_arms, n_outcomes, alpha, ell, termination_arm=False, cost=0.0):
        super().__init__(n_arms, n_outcomes, alpha, termination_arm, cost=cost)
        self.ells = np.atleast_1d(np.asarray(ell, dtype=float)) # ensures 1D array, even if scalar
        self.n_params = len(self.ells)

        ## define emp normaliser
        # if self.ell>1:
        #     self._emp_norm = self.n_outcomes ** (1-self.ell) * np.min([self.n_arms, self.n_outcomes])**self.ell
        # else:
        #     self._emp_norm = self.n_arms
        self._emp_norm = self.n_outcomes

    def leaf_value(self, counts):

        ## calculate skewed expectation for all ells at once
        max_p = self.predictive(counts).max(axis=0)
        emp = (max_p[None, :] ** self.ells[:, None]).sum(axis=1)

        ## normalise
        # emp /= self._emp_norm

        ## cost is determined by number of pulls taken already - i.e. reachable reward enters into expectation calculation
        return emp * (1 - counts.sum() * self.cost)


class InfoSeekingAgent(EmpAgent):
    """Maximises end-state posterior variance reduction.

    Scores a belief state by 1 - Var[p|h] / Var[p|h_0]: the Dirichlet posterior
    variance summed over all (a, o) cells, normalised by its value at the
    flat-prior root. HIGHER IS BETTER. Not parameterised by ell, so n_params = 1.
    """

    def __init__(self, n_arms, n_outcomes, alpha, termination_arm=False, cost=0.0):
        super().__init__(n_arms, n_outcomes, alpha, termination_arm, cost=cost)

        ## define MSE(h_0) - i.e. the posterior variance at root
        self._var_norm = self._var(np.zeros((self.n_arms, self.n_outcomes))).sum()

    def _var(self, counts):
        """Dirichlet posterior variance of each p_{a,o}, shape (A, O)."""
        a = self.alpha + counts
        a0 = a.sum(axis=1, keepdims=True)
        return a * (a0 - a) / (a0 ** 2 * (a0 + 1))

    def leaf_value(self, counts):

        ## normalise var
        var = 1 - (self._var(counts).sum() / self._var_norm)

        ## apply cost
        return np.array([var * (1 - counts.sum() * self.cost)])


def make_agent(n_arms, n_outcomes, alpha, ell=None, termination_arm=False, cost=0.0):
    """A fresh EmpowermentAgent for `ell` (scalar or grid), or an InfoSeekingAgent if ell is None."""
    if ell is None:
        return InfoSeekingAgent(n_arms, n_outcomes, alpha, termination_arm=termination_arm, cost=cost)
    return EmpowermentAgent(n_arms, n_outcomes, alpha, ell=ell, termination_arm=termination_arm, cost=cost)
