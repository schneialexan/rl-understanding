from random import Random
from typing import Any, Callable, Dict, Hashable, Sequence, Union, Mapping

State = Hashable
Action = Hashable
Number = float

class Policy:
    """Class that handles deterministic or stochastic policies."""
    def __init__(
        self,
        deterministic: Union[Callable[[State], Action], None] = None,
        stochastic: Union[Mapping[tuple[State, Action], float], None] = None,
        rng: Random | None = None,
    ):
        if deterministic is None and stochastic is None:
            raise ValueError("Must provide either deterministic or stochastic policy.")
        if deterministic is not None and stochastic is not None:
            raise ValueError("Cannot provide both deterministic and stochastic at once.")
        self.deterministic = deterministic
        self.stochastic = stochastic
        self.rng = rng or random

    def __call__(self, s: State) -> Action:
        if self.deterministic:
            return self.deterministic(s)
        if self.stochastic:
            actions_probs = {a: p for (state, a), p in self.stochastic.items() if state == s}
            if not actions_probs:
                raise ValueError(f"No stochastic policy defined for state {s}.")
            actions = list(actions_probs.keys())
            probs = list(actions_probs.values())
            total = sum(probs)
            probs = [p / total for p in probs]
            return self.rng.choices(actions, weights=probs, k=1)[0]

    # --- helper constructors for common policies ---
    @classmethod
    def random(cls, A: Sequence[Action], rng: Random | None = None):
        """Return a deterministic wrapper around uniform random sampling."""
        def det(s): 
            return (rng or random).choice(list(A))
        return cls(deterministic=det)

    @classmethod
    def greedy(cls, Q: dict[tuple[State, Action], Number], A: Sequence[Action], rng: Random | None = None):
        """Return a deterministic greedy policy."""
        rng = rng or random
        def det(s: State) -> Action:
            vals = [Q.get((s, a), float("-inf")) for a in A]
            max_val = max(vals)
            candidates = [a for a, v in zip(A, vals) if v == max_val]
            return rng.choice(candidates)
        return cls(deterministic=det)

    @classmethod
    def epsilon_greedy(cls, Q: dict[tuple[State, Action], Number], A: Sequence[Action], epsilon: float = 0.1, rng: Random | None = None):
        """Return a stochastic epsilon-greedy policy."""
        rng = rng or random
        stochastic_map: Dict[tuple[State, Action], float] = {}
        for s, _ in set((k[0], None) for k in Q.keys()):
            vals = [Q.get((s, a), float("-inf")) for a in A]
            max_val = max(vals)
            max_actions = [a for a, v in zip(A, vals) if v == max_val]
            for a in A:
                if a in max_actions:
                    stochastic_map[(s, a)] = (1 - epsilon) / len(max_actions)
                else:
                    stochastic_map[(s, a)] = epsilon / (len(A) - len(max_actions))
        return cls(stochastic=stochastic_map, rng=rng)
