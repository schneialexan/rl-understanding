import random
from typing import Any, Callable, List, Tuple, Mapping, Union

Number = Union[int, float]
State = Any
Action = Any

TransitionDict = Mapping[Tuple[State, Action], Mapping[State, float]]
RewardDictSA = Mapping[Tuple[State, Action], Number]
RewardDictS = Mapping[State, Number]
RewardDictSAS = Mapping[Tuple[State, Action], Mapping[State, Number]]

def simulate_episode(
    s0: State,
    policy,
    P,
    r,
    H: int = None,
    gamma: float = None,
    is_terminal: Callable[[State], bool] = None,
    noise_std: float = 0.0,
    rng: random.Random = None,
) -> Tuple[List[Tuple[State, Action, float, State]], float]:
    if rng is None:
        rng = random
    state = s0
    trajectory = []
    t = 0
    total_return = 0.0
    discount = 1.0

    while True:
        if H is not None and t >= H:
            break
        if is_terminal is not None and is_terminal(state):
            break

        action = policy(state)
        dist = P[(state, action)]
        next_state = rng.choices(list(dist.keys()), weights=list(dist.values()), k=1)[0]

        reward = r.get((state, action), 0.0)

        trajectory.append((state, action, reward, next_state))
        total_return += reward if gamma is None else discount * reward
        discount *= gamma if gamma is not None else 1.0
        state = next_state
        t += 1

    return trajectory, total_return

def set_random_seed(seed: int = None) -> random.Random:
    return random.Random(seed)
