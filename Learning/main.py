from utils import is_valid_mdp, sample_next_state, expected_reward

S = {0, 1}
A = {"left", "right"}
H = 10
gamma = 0.9

valid, msg = is_valid_mdp(S, A, H, gamma)
print(valid, msg)

P0 = {0: 0.5, 1: 0.5}
P = {
    (0, "left"): {0: 1.0, 1: 0.0},
    (0, "right"): {0: 0.0, 1: 1.0},
    (1, "left"): {0: 0.5, 1: 0.5},
    (1, "right"): {0: 0.2, 1: 0.8},
}
r = {(0, "left"): 1, (0, "right"): 0, (1, "left"): 2, (1, "right"): 3}

# check if markov decision process is fulfilled.
valid, msg = is_valid_mdp(S, A, H, gamma, P0=P0, P=P, r=r)
print(valid, msg)

# Sample next states
current_state = 1
action = "right"
next_state = sample_next_state(current_state, action, P)
print(f"From state {current_state} taking action '{action}' -> next state {next_state}")

# get the reward for an action
reward = expected_reward(current_state, action, r)
print(f"Expected reward for state {current_state} and action '{action}': {reward}")
