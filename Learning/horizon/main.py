import sys
import os
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import sample_next_state, expected_reward


# Grid setup
S = [(i, j) for i in range(3) for j in range(3)]
A = ["up", "down", "left", "right"]
H_finite = 4  # finite horizon
gamma = 0.9  # discount factor for infinite horizon

# Transition kernel: deterministic moves
P = {}
for i, j in S:
    for a in A:
        ni, nj = i, j
        if a == "up" and i > 0:
            ni -= 1
        if a == "down" and i < 2:
            ni += 1
        if a == "left" and j > 0:
            nj -= 1
        if a == "right" and j < 2:
            nj += 1
        P[((i, j), a)] = {(ni, nj): 1.0}

# Reward: +10 at goal, 0 elsewhere
r = {}
goal = (2, 2)
for s in S:
    for a in A:
        next_s = list(P[(s, a)].keys())[0]
        r[(s, a)] = 10 if next_s == goal else 0


# Simple "perfect" policy: always move right if possible, else down
def policy(state):
    i, j = state
    if j < 2:
        return "right"
    if i < 2:
        return "down"
    return "up"


def random_policy(state):
    return random.choice(["up", "down", "left", "right"])


# --- Finite horizon simulation ---
print("Finite horizon simulation:")
state = (0, 0)
total_reward = 0
for t in range(H_finite):
    a = policy(state)
    next_state = sample_next_state(state, a, P)
    reward = expected_reward(state, a, r)
    print(
        f"Step {t+1}: state={state}, action={a}, reward={reward}, next_state={next_state}"
    )
    total_reward += reward
    state = next_state
print("Total reward (finite horizon):", total_reward)

# --- Infinite horizon simulation ---
print("\nInfinite horizon simulation (with discount):")
state = (0, 0)
total_reward = 0
discount = 1
for t in range(10):  # simulate 10 steps
    a = policy(state)
    next_state = sample_next_state(state, a, P)
    reward = expected_reward(state, a, r)
    total_reward += discount * reward
    print(
        f"Step {t+1}: state={state}, action={a}, reward={reward}, next_state={next_state}, discounted reward={discount*reward}"
    )
    discount *= gamma
    state = next_state
print("Total discounted reward (infinite horizon):", total_reward)
