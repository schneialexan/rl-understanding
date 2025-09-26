from __future__ import annotations

import random
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
from policy.policy import Policy

Number = Union[int, float]
State = Any
Action = Any

# Type aliases
TransitionDict = Mapping[Tuple[State, Action], Mapping[State, float]]
RewardDictSA = Mapping[Tuple[State, Action], Number]
RewardDictS = Mapping[State, Number]
RewardDictSAS = Mapping[Tuple[State, Action], Mapping[State, Number]]


def is_distribution_valid(dist: Mapping[Any, float], tolerance: float = 1e-8) -> bool:
    """
    Check if a mapping of outcomes to probabilities is a valid distribution.

    Parameters
    ----------
    dist : Mapping[Any, float]
        Mapping of outcomes to probabilities.
    tolerance : float
        Allowed deviation from sum 1.

    Returns
    -------
    bool
        True if all probabilities >= 0 and sum to 1 within tolerance.
    """
    if not dist:
        return False
    probs = list(dist.values())
    if any(p < -tolerance for p in probs):
        return False
    return abs(sum(probs) - 1.0) <= tolerance


def is_valid_mdp(
    S: Iterable[State],
    A: Iterable[Action],
    H: Optional[int],
    gamma: float,
    P0: Optional[Mapping[State, float]] = None,
    P: Optional[Union[TransitionDict, Callable[[State, Action], Mapping[State, float]]]] = None,
    r: Optional[Union[RewardDictSA, RewardDictS, RewardDictSAS, Callable[..., Number]]] = None,
) -> Tuple[bool, str]:
    """
    Validate a Markov Decision Process (MDP) definition.

    Parameters
    ----------
    S : Iterable[State]
        Set of states.
    A : Iterable[Action]
        Set of actions.
    H : Optional[int]
        Horizon (None = infinite).
    gamma : float
        Discount factor in [0,1).
    P0 : Optional[Mapping[State, float]]
        Optional initial state distribution.
    P : Optional[Union[TransitionDict, Callable]]
        Optional transition kernel.
    r : Optional[Union[RewardDictSA, RewardDictS, RewardDictSAS, Callable]]
        Optional reward function.

    Returns
    -------
    Tuple[bool, str]
        (is_valid, message)
    """
    S_set = set(S)
    A_set = set(A)

    if not S_set:
        return False, "State set S is empty."
    if not A_set:
        return False, "Action set A is empty."
    if H is not None and (not isinstance(H, int) or H <= 0):
        return False, "Horizon H must be a positive integer or None (infinite)."
    if not isinstance(gamma, (int, float)) or not (0 <= gamma < 1):
        return False, "Discount factor gamma must be in [0, 1)."

    if P0 is not None:
        if not set(P0.keys()).issubset(S_set):
            return False, "P0 contains states not in S."
        if not is_distribution_valid(P0):
            return False, "P0 is not a valid probability distribution."

    if P is not None:
        if callable(P):
            try:
                sample_s = next(iter(S_set))
                sample_a = next(iter(A_set))
                dist = P(sample_s, sample_a)
                if not isinstance(dist, Mapping):
                    return False, "Callable P must return a mapping of next states to probabilities."
            except Exception as e:
                return False, f"Callable P raised an error when sampled: {e}"
        else:
            for (s, a), dist in P.items():
                if s not in S_set:
                    return False, f"Transition key state {s} not in S."
                if a not in A_set:
                    return False, f"Transition key action {a} not in A."
                if not set(dist.keys()).issubset(S_set):
                    return False, f"Transition distribution for {(s,a)} contains states not in S."
                if not is_distribution_valid(dist):
                    return False, f"Transition distribution for {(s,a)} is not valid."

    if r is not None:
        if callable(r):
            try:
                sample_s = next(iter(S_set))
                sample_a = next(iter(A_set))
                _ = r(sample_s, sample_a)
            except TypeError:
                try:
                    sample_s_next = next(iter(S_set))
                    _ = r(sample_s, sample_a, sample_s_next)
                except Exception as e:
                    return False, f"Callable r has unexpected signature or raised error: {e}"
            except Exception as e:
                return False, f"Callable r raised an error when sampled: {e}"
        else:
            if all(isinstance(k, tuple) and len(k) == 2 for k in r.keys()):
                for (s, a), val in r.items():
                    if s not in S_set or a not in A_set:
                        return False, f"Reward key {(s,a)} contains invalid state/action."
                    if not isinstance(val, (int, float)):
                        return False, f"Reward for {(s,a)} is not numeric."
            else:
                sample_key = next(iter(r.keys()))
                if sample_key in S_set:
                    for s, val in r.items():
                        if s not in S_set:
                            return False, "Reward mapping contains state not in S."
                        if not isinstance(val, (int, float)):
                            return False, f"Reward for state {s} is not numeric."
                else:
                    for (s, a), mapping in r.items():
                        if s not in S_set or a not in A_set:
                            return False, f"Reward key {(s,a)} contains invalid state/action."
                        if not set(mapping.keys()).issubset(S_set):
                            return False, f"Reward mapping for {(s,a)} contains invalid next-states."

    return True, "This is a valid MDP (optional components checked)."


def sample_next_state(
    s: State,
    a: Action,
    P: Union[TransitionDict, Callable[[State, Action], Mapping[State, float]]],
    rng: Optional[random.Random] = None,
) -> State:
    """
    Sample the next state given the current state, action, and transition kernel.

    Parameters
    ----------
    s : State
        Current state.
    a : Action
        Action taken.
    P : Union[TransitionDict, Callable]
        Transition kernel.
    rng : Optional[random.Random]
        Random number generator.

    Returns
    -------
    State
        Sampled next state.
    """
    if rng is None:
        rng = random
    if P is None:
        raise ValueError("Transition kernel P must be provided to sample next state.")
    dist = P(s, a) if callable(P) else P[(s, a)]
    choices = list(dist.keys())
    weights = list(dist.values())
    try:
        return rng.choices(choices, weights=weights, k=1)[0]
    except AttributeError:
        cum = []
        ssum = 0.0
        for w in weights:
            ssum += w
            cum.append(ssum)
        r = rng.random()
        for idx, c in enumerate(cum):
            if r <= c:
                return choices[idx]
        return choices[-1]


def expected_reward(
    s: State,
    a: Action,
    r: Union[RewardDictSA, RewardDictS, RewardDictSAS, Callable[..., Number]],
    s_next: Optional[State] = None,
) -> Number:
    """
    Compute the expected immediate reward for a state-action pair.

    Parameters
    ----------
    s : State
        Current state.
    a : Action
        Action taken.
    r : Union[RewardDictSA, RewardDictS, RewardDictSAS, Callable]
        Reward specification.
    s_next : Optional[State]
        Optional next state (if reward depends on it).

    Returns
    -------
    Number
        Expected reward.
    """
    if callable(r):
        try:
            return r(s, a)
        except TypeError:
            if s_next is None:
                raise ValueError("r appears to expect (s,a,s_next). Provide s_next.")
            return r(s, a, s_next)
    key = (s, a)
    if key in r:
        val = r[key]
        if isinstance(val, Mapping):
            if s_next is None:
                raise ValueError("Reward mapping depends on next-state; provide s_next.")
            return val[s_next]
        return val
    if s in r:
        return r[s]
    raise KeyError(f"Reward not defined for state {s} and action {a}.")


def sample_noisy_reward(
    s: State,
    a: Action,
    r: Union[RewardDictSA, RewardDictS, RewardDictSAS, Callable[..., Number]],
    s_next: Optional[State] = None,
    noise_std: float = 0.0,
    rng: Optional[random.Random] = None,
) -> float:
    """
    Sample a noisy reward around the expected reward.

    Parameters
    ----------
    s : State
        Current state.
    a : Action
        Action taken.
    r : Union[RewardDictSA, RewardDictS, RewardDictSAS, Callable]
        Reward specification.
    s_next : Optional[State]
        Optional next state.
    noise_std : float
        Standard deviation for Gaussian noise.
    rng : Optional[random.Random]
        Random number generator.

    Returns
    -------
    float
        Noisy reward sample.
    """
    if rng is None:
        rng = random
    mu = expected_reward(s, a, r, s_next)
    return float(mu) if noise_std <= 0 else float(rng.gauss(mu, noise_std))


def simulate_episode(
    s0: State,
    policy: Policy,
    P: Union[TransitionDict, Callable[[State, Action], Mapping[State, float]]],
    r: Union[RewardDictSA, RewardDictS, RewardDictSAS, Callable[..., Number]],
    H: Optional[int] = None,
    gamma: Optional[float] = None,
    is_terminal: Optional[Callable[[State], bool]] = None,
    noise_std: float = 0.0,
    rng: Optional[random.Random] = None,
) -> Tuple[List[Tuple[State, Action, float, State]], float]:
    """
    Simulate an episode using the given policy.

    Parameters
    ----------
    s0 : State
        Initial state.
    policy : Policy
        Policy to follow.
    P : Union[TransitionDict, Callable]
        Transition kernel.
    r : Union[RewardDictSA, RewardDictS, RewardDictSAS, Callable]
        Reward specification.
    H : Optional[int]
        Horizon.
    gamma : Optional[float]
        Discount factor.
    is_terminal : Optional[Callable[[State], bool]]
        Function to check terminal states.
    noise_std : float
        Noise standard deviation for reward sampling.
    rng : Optional[random.Random]
        Random number generator.

    Returns
    -------
    Tuple[List[Tuple[State, Action, float, State]], float]
        (trajectory, total return)
    """
    if rng is None:
        rng = random
    state = s0
    trajectory: List[Tuple[State, Action, float, State]] = []
    t = 0
    total_return = 0.0
    discount = 1.0

    while True:
        if H is not None and t >= H:
            break
        if is_terminal is not None and is_terminal(state):
            break

        action = policy(state)
        next_state = sample_next_state(state, action, P, rng=rng)
        reward = sample_noisy_reward(state, action, r, s_next=next_state, noise_std=noise_std, rng=rng)

        trajectory.append((state, action, reward, next_state))
        total_return += reward if gamma is None else discount * reward
        discount *= gamma if gamma is not None else 1.0
        state = next_state
        t += 1

    return trajectory, total_return


def set_random_seed(seed: Optional[int]) -> random.Random:
    """
    Create a random number generator with the given seed.

    Parameters
    ----------
    seed : Optional[int]
        Seed for RNG.

    Returns
    -------
    random.Random
        Random number generator instance.
    """
    return random.Random(seed)
