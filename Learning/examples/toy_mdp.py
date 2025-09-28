from __future__ import annotations
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import random
from typing import Any, Dict, Tuple
from policy.policy import (
    DeterministicPolicy,
    RandomPolicy,
    epsilon_greedy_from_q_fn,
    softmax_from_q_fn,
    greedy_policy_from_q_fn,
    GreedyFromVPolicy,
)
from rl.dp import value_iteration, policy_iteration, compute_q_from_v
from utils.mdp_utils import simulate_episode, set_random_seed


# Toy MDP: 3 states, 2 actions
S = ["s0", "s1", "s2"]
A = ["left", "right"]

# deterministic transitions (toy)
# P[(s,a)] -> dict next state -> prob
P = {
    ("s0", "left"): {"s1": 1.0},
    ("s0", "right"): {"s0": 1.0},
    ("s1", "left"): {"s2": 1.0},
    ("s1", "right"): {"s0": 1.0},
    ("s2", "left"): {"s2": 1.0},
    ("s2", "right"): {"s0": 1.0},
}

# rewards: prefer reaching s2 with left from s1
r = {
    ("s0", "left"): 0.0,
    ("s0", "right"): 0.0,
    ("s1", "left"): 1.0,
    ("s1", "right"): 0.0,
    ("s2", "left"): 0.0,
    ("s2", "right"): 0.0,
}

gamma = 0.9


def q_fn_from_table(V):
    # closure to produce q_fn used by policy implementations
    return lambda s: compute_q_from_v(s, A, V, P, r, gamma)


def main():
    # value iteration
    V_vi, policy_vi = value_iteration(S, A, P, r, gamma)
    print("Value iteration V:", V_vi)
    print("Value iteration greedy policy:", policy_vi)

    # policy iteration
    V_pi, policy_pi = policy_iteration(S, A, P, r, gamma)
    print("Policy iteration V:", V_pi)
    print("Policy iteration policy:", policy_pi)

    # convert greedy from V into policy objects
    greedy_pol = DeterministicPolicy(policy_vi)
    random_pol = RandomPolicy(A)

    # demonstrate epsilon-greedy derived from Q of VI
    q_fn = q_fn_from_table(V_vi)
    eps_pol = epsilon_greedy_from_q_fn(q_fn, A, epsilon=0.2)
    soft_pol = softmax_from_q_fn(q_fn, tau=0.5, A=A)

    seed = set_random_seed(42)
    # simulate episodes
    for pol_name, pol in [
        ("greedy", greedy_pol),
        ("random", random_pol),
        ("eps", eps_pol),
        ("soft", soft_pol),
    ]:
        traj, ret = simulate_episode("s0", pol, P, r, H=10, gamma=gamma, rng=seed)
        print(f"{pol_name}: return={ret}, trajectory={traj}")


if __name__ == "__main__":
    main()
