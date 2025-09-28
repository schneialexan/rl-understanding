# policy/policy.py
from __future__ import annotations
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
import math
import random
from dataclasses import dataclass

State = Any
Action = Any
Number = Union[int, float]

class Policy:
    """
    Abstract policy interface.
    A policy is any callable policy(state) -> action (for deterministic policies),
    or policy.action_distribution(state) -> mapping action->prob (for stochastic ones).
    We implement __call__ to always return a sampled action (deterministic for deterministic policies).
    """

    def action_distribution(self, s: State) -> Mapping[Action, float]:
        """
        Return a distribution mapping actions -> probability for state s.
        Default: raise NotImplementedError for deterministic policies that don't expose distributions.
        """
        raise NotImplementedError

    def __call__(self, s: State) -> Action:
        """
        Sample an action according to action_distribution(s).
        """
        dist = self.action_distribution(s)
        # simple sampling (works with python's random)
        choices, probs = zip(*dist.items())
        r = random.random()
        cum = 0.0
        for a, p in zip(choices, probs):
            cum += p
            if r <= cum:
                return a
        return choices[-1]

@dataclass
class DeterministicPolicy(Policy):
    """
    Deterministic policy backed by a mapping s -> a.
    __call__ returns that action.
    action_distribution returns a Dirac distribution.
    """
    policy_map: Dict[State, Action]

    def action_distribution(self, s: State) -> Mapping[Action, float]:
        a = self.policy_map[s]
        return {a: 1.0}

    def __call__(self, s: State) -> Action:
        return self.policy_map[s]

@dataclass
class RandomPolicy(Policy):
    """
    Uniform random over a provided action set A (same distribution for every state).
    Useful to replicate the lecture's "random policy".
    """
    A: Sequence[Action]

    def action_distribution(self, s: State) -> Mapping[Action, float]:
        if not self.A:
            raise ValueError("No actions provided")
        p = 1.0 / len(self.A)
        return {a: p for a in self.A}

@dataclass
class UniformOverArgmaxPolicy(Policy):
    """
    When there are multiple maximizing actions, pick uniformly over the argmax set.
    Input: a function q(s) -> Mapping[action, value] or a Q-table mapping.
    """
    q_fn: Callable[[State], Mapping[Action, float]]

    def action_distribution(self, s: State) -> Mapping[Action, float]:
        qvals = self.q_fn(s)
        maxv = max(qvals.values())
        argmax = [a for a, v in qvals.items() if abs(v - maxv) < 1e-12]
        p = 1.0 / len(argmax)
        return {a: (p if a in argmax else 0.0) for a in qvals.keys()}

@dataclass
class EpsilonGreedyPolicy(Policy):
    """
    Epsilon-greedy wrt q_fn(s) mapping or list of actions.
    - With probability epsilon: uniform random over A
    - Else: greedy (uniform over argmax if multiple)
    """
    q_fn: Callable[[State], Mapping[Action, float]]
    A: Sequence[Action]
    epsilon: float = 0.1

    def action_distribution(self, s: State) -> Mapping[Action, float]:
        qvals = self.q_fn(s)
        # argmax set
        maxv = max(qvals.values())
        argmax = [a for a, v in qvals.items() if abs(v - maxv) < 1e-12]
        nA = len(self.A)
        # exploration prob per action:
        base = self.epsilon / nA
        # greedy mass:
        greedy_mass = (1.0 - self.epsilon) / len(argmax)
        dist = {}
        for a in self.A:
            dist[a] = base + (greedy_mass if a in argmax else 0.0)
        return dist

@dataclass
class SoftmaxPolicy(Policy):
    """
    Boltzmann (softmax) policy over Q-values: pi(a|s) propto exp(Q(s,a)/tau)
    Small tau -> close to greedy; large tau -> near-uniform.
    """
    q_fn: Callable[[State], Mapping[Action, float]]
    tau: float = 1.0
    A: Optional[Sequence[Action]] = None

    def action_distribution(self, s: State) -> Mapping[Action, float]:
        qvals = self.q_fn(s)
        actions = list(qvals.keys()) if self.A is None else list(self.A)
        exps = []
        for a in actions:
            exps.append(math.exp(qvals[a] / (self.tau + 1e-12)))
        Z = sum(exps)
        if Z == 0:
            p = 1.0 / len(actions)
            return {a: p for a in actions}
        return {a: e / Z for a, e in zip(actions, exps)}

@dataclass
class GreedyFromVPolicy(Policy):
    """
    Greedy policy derived from a value function V and the MDP (requires P and r to compute Q).
    We pass a helper q_from_v(s) -> mapping(action->Q(s,a)) which computes the one-step Q from V.
    """
    q_from_v_fn: Callable[[State], Mapping[Action, float]]

    def action_distribution(self, s: State) -> Mapping[Action, float]:
        qvals = self.q_from_v_fn(s)
        maxv = max(qvals.values())
        argmax = [a for a, v in qvals.items() if abs(v - maxv) < 1e-12]
        p = 1.0 / len(argmax)
        return {a: (p if a in argmax else 0.0) for a in qvals.keys()}

# convenience factory functions
def greedy_policy_from_q_fn(q_fn: Callable[[State], Mapping[Action, float]]):
    return UniformOverArgmaxPolicy(q_fn=q_fn)

def epsilon_greedy_from_q_fn(q_fn: Callable[[State], Mapping[Action, float]], A: Sequence[Action], epsilon: float = 0.1):
    return EpsilonGreedyPolicy(q_fn=q_fn, A=A, epsilon=epsilon)

def softmax_from_q_fn(q_fn: Callable[[State], Mapping[Action, float]], tau: float = 1.0, A: Optional[Sequence[Action]] = None):
    return SoftmaxPolicy(q_fn=q_fn, tau=tau, A=A)
