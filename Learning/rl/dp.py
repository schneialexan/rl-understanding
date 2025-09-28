# rl/dp.py
from __future__ import annotations
import math
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
import numpy as np

State = Any
Action = Any
Number = Union[float, int]

# Types compatible with your top-level code:
TransitionDict = Mapping[Tuple[State, Action], Mapping[State, float]]
RewardDictSA = Mapping[Tuple[State, Action], Number]
RewardDictS = Mapping[State, Number]
RewardDictSAS = Mapping[Tuple[State, Action], Mapping[State, Number]]

def compute_q_from_v(
    s: State,
    actions: Sequence[Action],
    V: Mapping[State, Number],
    P: Union[TransitionDict, Callable[[State, Action], Mapping[State, float]]],
    r: Union[RewardDictSA, RewardDictS, RewardDictSAS, Callable[..., Number]],
    gamma: float,
) -> Mapping[Action, float]:
    """
    Compute one-step Q(s,a) = r(s,a) + gamma * sum_{s'} P(s'|s,a) V(s').
    This helper is used for greedy-from-V policies and policy extraction.
    """
    q = {}
    for a in actions:
        # expected reward r(s,a) may be a function or table
        # reuse expected_reward-like logic but simpler here:
        # support r(s,a), r(s), r(s,a,s_next) maps, etc.
        # we will assume expected reward r(s,a) available via provided callable or mapping
        if callable(r):
            try:
                r_sa = r(s, a)
            except TypeError:
                # expecting s,a,s'
                # compute expected reward under transition
                dist = P(s, a) if callable(P) else P[(s, a)]
                r_sa = sum(dist[s2] * r(s, a, s2) for s2 in dist.keys())
        else:
            key = (s, a)
            if key in r:
                rv = r[key]
                if isinstance(rv, Mapping):
                    # r depends on next state; compute expectation
                    dist = P(s, a) if callable(P) else P[(s, a)]
                    r_sa = sum(dist[s2] * rv[s2] for s2 in dist.keys())
                else:
                    r_sa = rv
            elif s in r:
                r_sa = r[s]
            else:
                raise KeyError(f"Reward undefined for {(s,a)}")
        # transition expectation
        dist = P(s, a) if callable(P) else P[(s, a)]
        exp_next = 0.0
        for s2, p in dist.items():
            exp_next += p * V[s2]
        q[a] = r_sa + gamma * exp_next
    return q

def policy_evaluation_iterative(
    policy: Callable[[State], Action] | Any,
    S: Sequence[State],
    A: Sequence[Action],
    P: Union[TransitionDict, Callable[[State, Action], Mapping[State, float]]],
    r: Union[RewardDictSA, RewardDictS, RewardDictSAS, Callable[..., Number]],
    gamma: float,
    tol: float = 1e-8,
    max_iters: int = 10000,
    initial_V: Optional[Mapping[State, Number]] = None,
) -> Dict[State, float]:
    """
    Iterative policy evaluation: V_{n+1} = T^pi V_n.
    Implements the fixed-policy Bellman operator and iterates until tol.
    """
    V = {s: 0.0 for s in S} if initial_V is None else dict(initial_V)
    # if policy provides distribution, we use expectation; else policy(s) returns an action
    for it in range(max_iters):
        delta = 0.0
        V_new = {}
        for s in S:
            # build expected one-step
            # if policy has .action_distribution, use that
            if hasattr(policy, "action_distribution"):
                dist = policy.action_distribution(s)
                val = 0.0
                for a, pa in dist.items():
                    # reward expectation r(s,a)
                    # re-use compute_q_from_v but with V placeholder to compute r + gamma P V? cheaper to compute manually:
                    # expected reward r(s,a)
                    if callable(r):
                        try:
                            r_sa = r(s, a)
                        except TypeError:
                            # r(s,a,s')
                            dist_sa = P(s, a) if callable(P) else P[(s, a)]
                            r_sa = sum(dist_sa[s2] * r(s, a, s2) for s2 in dist_sa)
                    else:
                        key = (s, a)
                        if key in r:
                            rv = r[key]
                            if isinstance(rv, Mapping):
                                dist_sa = P(s, a) if callable(P) else P[(s, a)]
                                r_sa = sum(dist_sa[s2] * rv[s2] for s2 in dist_sa)
                            else:
                                r_sa = rv
                        elif s in r:
                            r_sa = r[s]
                        else:
                            raise KeyError(f"Reward undefined for {(s,a)}")
                    # expectation over next states
                    dist_next = P(s, a) if callable(P) else P[(s, a)]
                    exp_next = sum(p * V[s2] for s2, p in dist_next.items())
                    val += pa * (r_sa + gamma * exp_next)
                V_new[s] = val
            else:
                # deterministic policy(s) -> a
                a = policy(s)
                # compute r_sa
                if callable(r):
                    try:
                        r_sa = r(s, a)
                    except TypeError:
                        dist_sa = P(s, a) if callable(P) else P[(s, a)]
                        r_sa = sum(dist_sa[s2] * r(s, a, s2) for s2 in dist_sa)
                else:
                    key = (s, a)
                    if key in r:
                        rv = r[key]
                        if isinstance(rv, Mapping):
                            dist_sa = P(s, a) if callable(P) else P[(s, a)]
                            r_sa = sum(dist_sa[s2] * rv[s2] for s2 in dist_sa)
                        else:
                            r_sa = rv
                    elif s in r:
                        r_sa = r[s]
                    else:
                        raise KeyError(f"Reward undefined for {(s,a)}")
                dist_next = P(s, a) if callable(P) else P[(s, a)]
                exp_next = sum(p * V[s2] for s2, p in dist_next.items())
                V_new[s] = r_sa + gamma * exp_next
            delta = max(delta, abs(V_new[s] - V[s]))
        V = V_new
        if delta < tol:
            break
    return V

def policy_evaluation_exact_linear(
    policy: Callable[[State], Action] | Any,
    S: Sequence[State],
    A: Sequence[Action],
    P: Union[TransitionDict, Callable[[State, Action], Mapping[State, float]]],
    r: Union[RewardDictSA, RewardDictS, RewardDictSAS, Callable[..., Number]],
    gamma: float,
) -> Dict[State, float]:
    """
    Exact solution: solve linear system (I - gamma P_pi) V = r_pi when policy is stationary.
    Useful when |S| small; lecture notes show this closed-form. See Week 2 slides. :contentReference[oaicite:2]{index=2}
    """
    n = len(S)
    s_idx = {s: i for i, s in enumerate(S)}
    P_pi = np.zeros((n, n), dtype=float)
    r_vec = np.zeros(n, dtype=float)
    for i, s in enumerate(S):
        # get action distribution under policy
        if hasattr(policy, "action_distribution"):
            dist = policy.action_distribution(s)
        else:
            a = policy(s)
            dist = {a: 1.0}
        # r_pi(s) = E_{a~pi}[ r(s,a) ]
        r_s = 0.0
        for a, pa in dist.items():
            # expected immediate reward for (s,a) as in compute_q_from_v
            if callable(r):
                try:
                    r_sa = r(s, a)
                except TypeError:
                    dist_sa = P(s, a) if callable(P) else P[(s, a)]
                    r_sa = sum(dist_sa[s2] * r(s, a, s2) for s2 in dist_sa)
            else:
                key = (s, a)
                if key in r:
                    rv = r[key]
                    if isinstance(rv, Mapping):
                        dist_sa = P(s, a) if callable(P) else P[(s, a)]
                        r_sa = sum(dist_sa[s2] * rv[s2] for s2 in dist_sa)
                    else:
                        r_sa = rv
                elif s in r:
                    r_sa = r[s]
                else:
                    raise KeyError(f"Reward undefined for {(s,a)}")
            r_s += pa * r_sa
            # transition mass to next states
            dist_sa = P(s, a) if callable(P) else P[(s, a)]
            for s2, p in dist_sa.items():
                j = s_idx[s2]
                P_pi[i, j] += pa * p
        r_vec[i] = r_s
    # solve (I - gamma * P_pi) V = r_vec
    I = np.eye(n)
    V_vec = np.linalg.solve(I - gamma * P_pi, r_vec)
    return {s: float(V_vec[s_idx[s]]) for s in S}

def extract_greedy_policy_from_V(
    V: Mapping[State, Number],
    S: Sequence[State],
    A: Sequence[Action],
    P: Union[TransitionDict, Callable[[State, Action], Mapping[State, float]]],
    r: Union[RewardDictSA, RewardDictS, RewardDictSAS, Callable[..., Number]],
    gamma: float,
) -> Mapping[State, Action]:
    """
    Returns a deterministic greedy policy mapping s->argmax_a [ r(s,a) + gamma sum P(s'|s,a) V(s') ].
    This is the policy-extraction step from finite/infinite-horizon VI (lecture slides). :contentReference[oaicite:3]{index=3}
    """
    policy_map = {}
    for s in S:
        q = compute_q_from_v(s, A, V, P, r, gamma)
        # argmax tie-breaking: choose first; if you want uniform-over-argmax use policy implementations above
        best_a = max(q.items(), key=lambda kv: kv[1])[0]
        policy_map[s] = best_a
    return policy_map

def value_iteration(
    S: Sequence[State],
    A: Sequence[Action],
    P: Union[TransitionDict, Callable[[State, Action], Mapping[State, float]]],
    r: Union[RewardDictSA, RewardDictS, RewardDictSAS, Callable[..., Number]],
    gamma: float,
    tol: float = 1e-8,
    max_iters: int = 10000,
) -> Tuple[Dict[State, float], Dict[State, Action]]:
    """
    Standard value iteration for infinite-horizon discounted MDP.
    Implements Bellman optimality operator T* repeatedly until convergence.
    Returns (V, greedy_policy).
    See lecture slides Week 2 for convergence and Banach fixed point theory references. :contentReference[oaicite:4]{index=4}
    """
    V = {s: 0.0 for s in S}
    for it in range(max_iters):
        delta = 0.0
        V_new = {}
        for s in S:
            best = -float("inf")
            for a in A:
                # compute r + gamma * sum P V
                if callable(r):
                    try:
                        r_sa = r(s, a)
                    except TypeError:
                        dist_sa = P(s, a) if callable(P) else P[(s, a)]
                        r_sa = sum(dist_sa[s2] * r(s, a, s2) for s2 in dist_sa)
                else:
                    key = (s, a)
                    if key in r:
                        rv = r[key]
                        if isinstance(rv, Mapping):
                            dist_sa = P(s, a) if callable(P) else P[(s, a)]
                            r_sa = sum(dist_sa[s2] * rv[s2] for s2 in dist_sa)
                        else:
                            r_sa = rv
                    elif s in r:
                        r_sa = r[s]
                    else:
                        raise KeyError(f"Reward undefined for {(s,a)}")
                dist_sa = P(s, a) if callable(P) else P[(s, a)]
                exp_next = sum(p * V[s2] for s2, p in dist_sa.items())
                val = r_sa + gamma * exp_next
                if val > best:
                    best = val
            V_new[s] = best
            delta = max(delta, abs(V_new[s] - V[s]))
        V = V_new
        if delta < tol:
            break
    policy_map = extract_greedy_policy_from_V(V, S, A, P, r, gamma)
    return V, policy_map

def policy_iteration(
    S: Sequence[State],
    A: Sequence[Action],
    P: Union[TransitionDict, Callable[[State, Action], Mapping[State, float]]],
    r: Union[RewardDictSA, RewardDictS, RewardDictSAS, Callable[..., Number]],
    gamma: float,
    max_policy_iters: int = 1000,
    eval_tol: float = 1e-8,
) -> Tuple[Dict[State, float], Dict[State, Action]]:
    """
    Classic Howard Policy Iteration: evaluate policy then improve greedily.
    See Week 2 slides for finite convergence properties. :contentReference[oaicite:5]{index=5}
    """
    # initialize arbitrary deterministic policy (choose first action)
    policy_map = {s: A[0] for s in S}
    for it in range(max_policy_iters):
        # evaluate current policy (exact)
        from functools import partial
        class DummyPolicy:
            def __init__(self, mapping): self.mapping = mapping
            def __call__(self, s): return self.mapping[s]
            def action_distribution(self, s):
                return {self.mapping[s]: 1.0}
        pol = DummyPolicy(policy_map)
        V = policy_evaluation_exact_linear(pol, S, A, P, r, gamma)
        # policy improvement
        improved = False
        for s in S:
            q = compute_q_from_v(s, A, V, P, r, gamma)
            best_a = max(q.items(), key=lambda kv: kv[1])[0]
            if best_a != policy_map[s]:
                improved = True
                policy_map[s] = best_a
        if not improved:
            break
    return V, policy_map

def finite_horizon_value_iteration(
    S: Sequence[State],
    A: Sequence[Action],
    P: Union[TransitionDict, Callable[[State, Action], Mapping[State, float]]],
    r: Union[RewardDictSA, RewardDictS, RewardDictSAS, Callable[..., Number]],
    H: int,
) -> Tuple[List[Dict[State, float]], List[Dict[State, Action]]]:
    """
    Finite-horizon backward recursion producing time-indexed value functions V_t and non-stationary greedy policies pi_t.
    Returns lists: V_list[H] ... V_0? We'll return V[0..H] where V[H] is terminal value.
    The lecture describes backward induction (Week 1). :contentReference[oaicite:6]{index=6}
    """
    V_list: List[Dict[State, float]] = [{s: 0.0 for s in S} for _ in range(H + 1)]
    # terminal value: V_H(s) = max_a r(s,a) according to lecture (some variants use terminal reward)
    V_list[H] = {s: max(( (r[(s,a)] if (s,a) in r else (r[s] if s in r else 0.0)) for a in A )) for s in S} if not callable(r) else {s: max(( (r(s,a) if (callable(r) and (lambda: True)) else 0.0) for a in A )) for s in S}
    # policies per time t
    policy_list: List[Dict[State, Action]] = [{s: A[0] for s in S} for _ in range(H)]
    # backward induction
    for t in reversed(range(H)):
        Vt = {}
        for s in S:
            best_val = -float("inf")
            best_a = None
            for a in A:
                # compute r(s,a) + sum p(s'|s,a) V_{t+1}(s')
                # handle r as mapping or callable (simpler approach similar to compute_q_from_v)
                if callable(r):
                    try:
                        r_sa = r(s, a)
                    except TypeError:
                        # r(s,a,s')
                        dist_sa = P(s, a) if callable(P) else P[(s, a)]
                        r_sa = sum(dist_sa[s2] * r(s, a, s2) for s2 in dist_sa)
                else:
                    key = (s, a)
                    if key in r:
                        rv = r[key]
                        if isinstance(rv, Mapping):
                            dist_sa = P(s, a) if callable(P) else P[(s, a)]
                            r_sa = sum(dist_sa[s2] * rv[s2] for s2 in dist_sa)
                        else:
                            r_sa = rv
                    elif s in r:
                        r_sa = r[s]
                    else:
                        r_sa = 0.0
                dist_sa = P(s, a) if callable(P) else P[(s, a)]
                exp_next = sum(dist_sa[s2] * V_list[t+1][s2] for s2, p in dist_sa.items())
                val = r_sa + exp_next
                if val > best_val:
                    best_val = val
                    best_a = a
            Vt[s] = best_val
            policy_list[t][s] = best_a
        V_list[t] = Vt
    return V_list, policy_list
