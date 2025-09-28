import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rl.dp import finite_horizon_value_iteration, value_iteration
from examples.toy_mdp import S, A, P, r, gamma


def run_horizon_experiment(H_values=(1, 2, 3, 5, 10)):
    for H in H_values:
        V_list, pi_list = finite_horizon_value_iteration(S, A, P, r, H)
        print(f"\nH = {H}")
        print("Policies (time indexed t=0..H-1):")
        for t, pi in enumerate(pi_list):
            print(f" t={t}: {pi}")
        print("Terminal V_H:", V_list[H])
    # compare with infinite-horizon discounted solution
    V_inf, pi_inf = value_iteration(S, A, P, r, gamma)
    print("\nInfinite-horizon (discounted) greedy policy:", pi_inf)
    print("Infinite-horizon V:", V_inf)


if __name__ == "__main__":
    run_horizon_experiment()
