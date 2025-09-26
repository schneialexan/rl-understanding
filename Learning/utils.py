from __future__ import annotations

import random
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

Number = Union[int, float]
State = Any
Action = Any

# Type aliases
TransitionDict = Mapping[Tuple[State, Action], Mapping[State, float]]
RewardDictSA = Mapping[Tuple[State, Action], Number]
RewardDictS = Mapping[State, Number]
RewardDictSAS = Mapping[Tuple[State, Action], Mapping[State, Number]]


def is_distribution_valid(dist: Mapping[Any, float], tolerance: float = 1e-8) -> bool:
    """Check that a mapping of outcomes -> probabilities is a valid distribution.

    - All probabilities >= 0
    - Probabilities sum to 1 (within tolerance)
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
    """Validate an MDP tuple where P0, P, r are optional.

    Parameters
    ----------
    S, A
        Iterables describing finite states and actions (converted to sets internally).
    H
        Horizon. If None, it can be considered infinite for validation purposes.
    gamma
        Discount factor in [0, 1).
    P0
        Optional initial-state distribution (mapping state -> prob).
    P
        Optional transition kernel. Either a dict mapping (s,a) -> {s': prob}
        or a callable P(s, a) -> mapping of next states to probabilities.
    r
        Optional reward function. Can be:
          - mapping (s,a) -> number
          - mapping s -> number
          - mapping (s,a) -> {s': number}
          - callable (s, a) or (s, a, s_next) -> number

    Returns
    -------
    (bool, message)
        Whether the MDP is valid and a short message.
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

    # P0
    if P0 is not None:
        if not set(P0.keys()).issubset(S_set):
            return False, "P0 contains states not in S."
        if not is_distribution_valid(P0):
            return False, "P0 is not a valid probability distribution."

    # P
    if P is not None:
        if callable(P):
            # Can't fully validate callable P w/o querying; check one sample for shape
            try:
                sample_s = next(iter(S_set))
                sample_a = next(iter(A_set))
                dist = P(sample_s, sample_a)
                if not isinstance(dist, Mapping):
                    return False, "Callable P must return a mapping of next states to probabilities."
            except Exception as e:
                return False, f"Callable P raised an error when sampled: {e}"
        else:
            # dict-based validation
            for (s, a), dist in P.items():
                if s not in S_set:
                    return False, f"Transition key state {s} not in S."
                if a not in A_set:
                    return False, f"Transition key action {a} not in A."
                if not set(dist.keys()).issubset(S_set):
                    return False, f"Transition distribution for {(s,a)} contains states not in S."
                if not is_distribution_valid(dist):
                    return False, f"Transition distribution for {(s,a)} is not valid."

    # r
    if r is not None:
        if callable(r):
            # Can't validate arbitrary callable; try to call with (s,a)
            try:
                sample_s = next(iter(S_set))
                sample_a = next(iter(A_set))
                _ = r(sample_s, sample_a)
            except TypeError:
                # maybe takes (s,a,s_next)? That's acceptable too
                try:
                    sample_s_next = next(iter(S_set))
                    _ = r(sample_s, sample_a, sample_s_next)
                except Exception as e:
                    return False, f"Callable r has unexpected signature or raised error: {e}"
            except Exception as e:
                return False, f"Callable r raised an error when sampled: {e}"
        else:
            # mapping-based: verify keys
            # (s,a) -> number
            if all(isinstance(k, tuple) and len(k) == 2 for k in r.keys()):
                for (s, a), val in r.items():
                    if s not in S_set or a not in A_set:
                        return False, f"Reward key {(s,a)} contains invalid state/action."
                    if not isinstance(val, (int, float)):
                        return False, f"Reward for {(s,a)} is not numeric."
            else:
                # maybe s -> number or (s,a) -> dict of s' -> number
                sample_key = next(iter(r.keys()))
                if sample_key in S_set:
                    # s -> number mapping
                    for s, val in r.items():
                        if s not in S_set:
                            return False, "Reward mapping contains state not in S."
                        if not isinstance(val, (int, float)):
                            return False, f"Reward for state {s} is not numeric."
                else:
                    # (s,a) -> {s': number}
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
    """Sample a next state given current state `s`, action `a`, and transition kernel `P`.

    `P` can be either:
      - a mapping (s,a) -> {s': prob}
      - a callable P(s, a) -> mapping of next states to probabilities

    Returns a sampled next-state (respecting the distribution).
    """
    if rng is None:
        rng = random

    if P is None:
        raise ValueError("Transition kernel P must be provided to sample next state.")

    if callable(P):
        dist = P(s, a)
    else:
        if (s, a) not in P:
            raise ValueError(f"Transition kernel undefined for state {s} and action {a}.")
        dist = P[(s, a)]

    if not dist:
        raise ValueError(f"Empty distribution for state {s} and action {a}.")

    choices = list(dist.keys())
    weights = list(dist.values())
    # use rng.choices for sampling when rng is random.Random
    try:
        return rng.choices(choices, weights=weights, k=1)[0]
    except AttributeError:
        # fallback for numpy or other RNG objects
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
    """Return expected immediate reward r(s,a).

    Supports:
    - mapping (s,a) -> number
    - mapping s -> number
    - mapping (s,a) -> {s': number} (returns value for s_next)
    - callable r(s, a) or r(s, a, s_next)
    """
    if callable(r):
        # try calling with (s,a) then (s,a,s_next)
        try:
            return r(s, a)
        except TypeError:
            if s_next is None:
                raise ValueError("r appears to expect (s,a,s_next). Provide s_next.")
            return r(s, a, s_next)

    # mapping-based
    # 1) (s,a) -> number
    key = (s, a)
    if key in r:  # type: ignore[arg-type]
        val = r[key]  # type: ignore[index]
        if isinstance(val, Mapping):
            # (s,a) -> {s': number}
            if s_next is None:
                raise ValueError("Reward mapping depends on next-state; provide s_next.")
            if s_next not in val:
                raise KeyError(f"No reward defined for next-state {s_next} under {(s,a)}.")
            return val[s_next]
        return val  # number

    # 2) s -> number
    if s in r:  # type: ignore[arg-type]
        return r[s]  # type: ignore[index]

    raise KeyError(f"Reward not defined for state {s} and action {a}.")


def sample_noisy_reward(
    s: State,
    a: Action,
    r: Union[RewardDictSA, RewardDictS, RewardDictSAS, Callable[..., Number]],
    s_next: Optional[State] = None,
    noise_std: float = 0.0,
    rng: Optional[random.Random] = None,
) -> float:
    """Return a noisy sample around the expected reward r(s,a).

    If noise_std == 0.0 returns the deterministic expected reward.
    """
    if rng is None:
        rng = random
    mu = expected_reward(s, a, r, s_next)
    if noise_std <= 0:
        return float(mu)
    return float(rng.gauss(mu, noise_std))


def simulate_episode(
    s0: State,
    policy: Callable[[State], Action],
    P: Union[TransitionDict, Callable[[State, Action], Mapping[State, float]]],
    r: Union[RewardDictSA, RewardDictS, RewardDictSAS, Callable[..., Number]],
    H: Optional[int] = None,
    gamma: Optional[float] = None,
    is_terminal: Optional[Callable[[State], bool]] = None,
    noise_std: float = 0.0,
    rng: Optional[random.Random] = None,
) -> Tuple[List[Tuple[State, Action, float, State]], float]:
    """Simulate one episode and return (trajectory, total_return).

    trajectory is a list of tuples: (state, action, reward, next_state)
    total_return is optionally discounted sum if gamma is provided; otherwise
    it's the undiscounted sum.

    Parameters
    ----------
    s0: initial state
    policy: function mapping state -> action
    P, r: transition kernel and reward (dict or callable)
    H: horizon (None means no fixed horizon; use is_terminal or you must provide H)
    gamma: if provided, compute discounted return
    is_terminal: optional predicate(state) -> bool to stop episode early
    noise_std: standard deviation for sampling noisy rewards
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

        if gamma is None:
            total_return += reward
        else:
            total_return += discount * reward
            discount *= gamma

        state = next_state
        t += 1

    return trajectory, total_return


def simulate_random_policy(
    s0: State,
    S: Iterable[State],
    A: Sequence[Action],
    P: Union[TransitionDict, Callable[[State, Action], Mapping[State, float]]],
    r: Union[RewardDictSA, RewardDictS, RewardDictSAS, Callable[..., Number]],
    H: Optional[int] = None,
    gamma: Optional[float] = None,
    is_terminal: Optional[Callable[[State], bool]] = None,
    noise_std: float = 0.0,
    rng: Optional[random.Random] = None,
) -> Tuple[List[Tuple[State, Action, float, State]], float]:
    """Helper: simulate an episode with a uniformly random policy over actions A."""
    if rng is None:
        rng = random

    def rand_policy(s: State) -> Action:
        return rng.choice(list(A))

    return simulate_episode(
        s0, rand_policy, P, r, H=H, gamma=gamma, is_terminal=is_terminal, noise_std=noise_std, rng=rng
    )


def set_random_seed(seed: Optional[int]) -> random.Random:
    """Create and return a Random instance seeded with `seed` (or system random if None)."""
    rng = random.Random(seed)
    return rng
