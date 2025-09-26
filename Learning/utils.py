from typing import Optional, Dict, Tuple
import random
import numpy as np

def is_valid_mdp(
    S: set,
    A: set,
    H: int,
    gamma: float,
    P0: Optional[Dict] = None,
    P: Optional[Dict[Tuple, Dict]] = None,
    r: Optional[Dict[Tuple, float]] = None
) -> Tuple[bool, str]:
    """
    Validates whether the given components form a proper MDP.
    
    Mandatory:
        S     : set of states
        A     : set of actions
        H     : horizon (positive integer) -> number of moves until stop of episode
        gamma : discount factor in [0,1)
    
    Optional:
        P0    : initial state distribution (dict mapping state -> probability)
        P     : transition kernel (dict mapping (s,a) -> {s': prob})
        r     : reward function (dict mapping (s,a) -> reward)
    """
    # 1. Check mandatory sets
    if not S or not A:
        return False, "State set S or action set A is empty."

    # 2. Check horizon
    if not isinstance(H, int) or H <= 0:
        return False, "Horizon H must be a positive integer."

    # 3. Check discount factor
    if not (0 <= gamma < 1):
        return False, "Discount factor gamma must be in [0,1)."

    # 4. Check initial state distribution (optional)
    if P0 is not None:
        if not np.isclose(sum(P0.values()), 1):
            return False, "P0 does not sum to 1."
        if not all(s in S for s in P0):
            return False, "P0 contains states not in S."

    # 5. Check transition kernel (optional)
    if P is not None:
        for (s, a), dist in P.items():
            if s not in S or a not in A:
                return False, f"Transition key {(s,a)} has invalid state/action."
            if not np.isclose(sum(dist.values()), 1):
                return False, f"Transition probabilities for {(s,a)} do not sum to 1."
            if not all(s_prime in S for s_prime in dist):
                return False, f"Transition probabilities for {(s,a)} contain invalid state."

    # 6. Check reward function (optional)
    if r is not None:
        for (s, a), reward in r.items():
            if s not in S or a not in A:
                return False, f"Reward key {(s,a)} has invalid state/action."
            if not isinstance(reward, (int, float)):
                return False, f"Reward for {(s,a)} is not numeric."

    return True, "This is a valid MDP (with optional components considered)."


def sample_next_state(
    s: int,
    a: str,
    P: Optional[Dict[Tuple[int, str], Dict[int, float]]]
) -> int:
    """
    Sample the next state given current state `s` and action `a` using transition kernel `P`.
    """
    if P is None:
        raise ValueError("Transition kernel P is not provided.")
    if (s, a) not in P:
        raise ValueError(f"Transition kernel undefined for state {s} and action {a}.")
    
    next_states = list(P[(s, a)].keys())
    probabilities = list(P[(s, a)].values())
    return random.choices(next_states, weights=probabilities, k=1)[0]

def expected_reward(
    s: int,
    a: str,
    r: Dict[Tuple[int, str], float]
) -> float:
    """
    Return the expected reward for taking action `a` in state `s`.

    Args:
        s: current state
        a: action taken
        r: dictionary mapping (state, action) -> expected reward

    Returns:
        Expected immediate reward (float)
    """
    if (s, a) not in r:
        raise ValueError(f"Reward not defined for state {s} and action {a}")
    return r[(s, a)]