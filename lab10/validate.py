"""
Correctness checks for Lab 10.

Run with:
    python validate.py

The checks are deterministic and do not generate figures.  They verify:

* the grid-world transition probabilities sum to one;
* value iteration reproduces the textbook values for r(s) = -0.04;
* value iteration and policy iteration agree on the grid world;
* the factorised Gbike transition model agrees with brute-force enumeration;
* policy iteration satisfies the Bellman optimality equation for both the
  original and modified Gbike MDPs.
"""

import numpy as np
from scipy.stats import poisson

from gridworld import (
    ACTIONS,
    TERMINALS,
    TRANSITIONS,
    nonterminal_cells,
    policy_iteration,
    value_iteration,
)
from gbike import (
    MAX_BIKES,
    EXPECTED_SERVED_1,
    EXPECTED_SERVED_2,
    GbikeMDP,
    NEXT_DIST_1,
    NEXT_DIST_2,
)


def check(condition, message):
    if not condition:
        raise AssertionError(message)
    print(f"  ok: {message}")


def validate_gridworld():
    print("\nGrid world")

    for state in nonterminal_cells():
        for action in ACTIONS:
            total = sum(TRANSITIONS[state][action].values())
            check(abs(total - 1.0) < 1e-12,
                  f"P(.|{state},{action}) sums to 1")

    expected_textbook = np.array([
        0.811558219, 0.867808219, 0.917808219, 1.0,
        0.761558219, 0.660273973, -1.0,
        0.705308219, 0.655308219, 0.611415525, 0.387924911,
    ])
    V, policy, iterations, converged, _ = value_iteration(
        -0.04, gamma=1.0, theta=1e-12, max_iter=10000
    )
    check(converged and iterations < 100,
          f"value iteration converges ({iterations} sweeps)")
    check(np.allclose(V[~np.isnan(V)], expected_textbook, atol=1e-9),
          "canonical values match Russell & Norvig")

    V_pi, policy_pi, iterations_pi, converged_pi = policy_iteration(
        -0.04, gamma=1.0
    )
    check(converged_pi, f"policy iteration converges ({iterations_pi} iterations)")
    check(np.nanmax(np.abs(V - V_pi)) < 1e-8,
          "value iteration and policy iteration agree")
    check(all(policy[s] == policy_pi[s] for s in nonterminal_cells()),
          "value iteration and policy iteration produce the same greedy policy")

    # Terminal values are fixed, not updated.
    for state, value in TERMINALS.items():
        check(V[state] == value, f"terminal value at {state} is {value}")


def validate_gbike():
    print("\nGbike")

    check(np.allclose(NEXT_DIST_1.sum(axis=1), 1.0),
          "location-1 next-state distributions sum to 1")
    check(np.allclose(NEXT_DIST_2.sum(axis=1), 1.0),
          "location-2 next-state distributions sum to 1")
    check(np.isclose(EXPECTED_SERVED_1[20], 3.0, atol=1e-9),
          "expected rentals at location 1 with 20 bikes equals mean demand 3")
    check(np.isclose(EXPECTED_SERVED_2[20], 4.0, atol=1e-9),
          "expected rentals at location 2 with 20 bikes equals mean demand 4")

    # Brute-force check of the factorised transition model for one case.
    mdp = GbikeMDP(modified=True)
    rng = np.random.default_rng(0)
    V = rng.normal(size=(MAX_BIKES + 1, MAX_BIKES + 1))
    c1, c2, action = 7, 12, 2
    after_1, after_2 = c1 - action, c2 + action

    request_1 = poisson.pmf(np.arange(31), 3.0)
    request_2 = poisson.pmf(np.arange(31), 4.0)
    returns_1 = poisson.pmf(np.arange(31), 3.0)
    returns_2 = poisson.pmf(np.arange(31), 2.0)

    brute_value = 0.0
    brute_revenue = 0.0
    for q1 in range(31):
        for q2 in range(31):
            served_1 = min(q1, after_1)
            served_2 = min(q2, after_2)
            brute_revenue += (request_1[q1] * request_2[q2]
                              * 10.0 * (served_1 + served_2))
            for t1 in range(31):
                for t2 in range(31):
                    nxt = (min(after_1 - served_1 + t1, 20),
                           min(after_2 - served_2 + t2, 20))
                    brute_value += (request_1[q1] * request_2[q2]
                                    * returns_1[t1] * returns_2[t2] * V[nxt])

    factor_revenue = 10.0 * (EXPECTED_SERVED_1[after_1]
                             + EXPECTED_SERVED_2[after_2])
    factor_value = mdp.expected_next_value(V, c1, c2, action)
    check(abs(factor_revenue - brute_revenue) < 1e-9,
          "factorised expected revenue matches brute-force enumeration")
    check(abs(factor_value - brute_value) < 1e-9,
          "factorised expected next value matches brute-force enumeration")

    # Bellman optimality equation and policy iteration for both variants.
    for modified in (False, True):
        label = "modified" if modified else "original"
        model = GbikeMDP(modified=modified)
        result = model.policy_iteration(verbose=False)
        check(result["converged"],
              f"{label} policy iteration converges "
              f"({result['iterations']} iterations)")

        residual = 0.0
        for c1 in range(MAX_BIKES + 1):
            for c2 in range(MAX_BIKES + 1):
                best = max(model.q_value(result["values"], c1, c2, action)
                           for action in model.feasible_actions(c1, c2))
                residual = max(residual, abs(best - result["values"][c1, c2]))
        check(residual < 1e-8,
              f"{label} policy satisfies the Bellman optimality equation "
              f"(residual {residual:.2e})")

        # Every action chosen by the policy must be feasible.
        feasible = all(
            result["policy"][c1, c2] in model.feasible_actions(c1, c2)
            for c1 in range(MAX_BIKES + 1)
            for c2 in range(MAX_BIKES + 1)
        )
        check(feasible, f"{label} policy only uses feasible actions")


if __name__ == "__main__":
    print("=" * 60)
    print("LAB 10 VALIDATION")
    print("=" * 60)
    validate_gridworld()
    validate_gbike()
    print("\nAll checks passed.")
