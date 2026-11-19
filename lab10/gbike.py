"""
Lab 10 - Part 2/3: Gbike bicycle rental (Jack's car rental)
===========================================================
A continuing, finite, discounted MDP solved with policy iteration.

State
-----
s = (c1, c2): number of bicycles parked at location 1 and location 2 at
the end of the day (after the day's rentals and returns), with
0 <= c1, c2 <= 20.

Action
------
a in {-5, ..., +5}: net number of bicycles moved overnight.
    a > 0 : move a bicycles from location 1 to location 2
    a < 0 : move -a bicycles from location 2 to location 1
    a = 0 : move nothing
An action is feasible only if it respects the 20-bike capacity of both
locations.  At most five bicycles can be moved per night.

Dynamics (one day)
------------------
1. Bicycles are moved overnight; the move is paid for immediately.
2. Rental requests arrive at location i ~ Poisson(lambda_req[i]).
   served_i = min(requests_i, available_i), earning INR 10 per rental.
3. Returned bicycles arrive at location i ~ Poisson(lambda_ret[i]) and
   are added to the inventory (capacity 20).  They are available for
   rental the following day.
4. The resulting inventory is the next state.

The original problem uses INR 2 per bicycle moved.  The modified problem
(the one requested for submission) changes the cost model:
  * one bicycle moved from location 1 to location 2 is free (the employee
    shuttles it home);
  * every additional bicycle moved 1 -> 2 costs INR 2;
  * every bicycle moved 2 -> 1 costs INR 2;
  * if more than 10 bicycles are kept overnight at a location, a fixed
    INR 4 is paid for a second parking lot at that location.
"""

import json
import os
from collections import OrderedDict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import poisson


# ---------------------------------------------------------------------------
# Problem constants
# ---------------------------------------------------------------------------
MAX_BIKES = 20                 # capacity of each location
MAX_MOVE = 5                   # maximum bicycles moved in one night

REQUESTS_MEAN = (3.0, 4.0)     # location 1, location 2
RETURNS_MEAN = (3.0, 2.0)      # location 1, location 2
RENTAL_INCOME = 10.0           # INR per bicycle rented
MOVE_COST = 2.0                # INR per bicycle moved
DISCOUNT = 0.9

PARKING_THRESHOLD = 10         # "more than 10" means > 10
PARKING_COST = 4.0
FREE_MOVES_1_TO_2 = 1          # employee's free shuttle (modified problem)

ACTIONS = tuple(range(-MAX_MOVE, MAX_MOVE + 1))


# ---------------------------------------------------------------------------
# Poisson building blocks
# ---------------------------------------------------------------------------
def expected_served(capacity, request_mean):
    """
    E[min(R, capacity)] for R ~ Poisson(request_mean).

    Rentals beyond the number of available bicycles are lost, so this is
    exactly the expected number of bicycles rented out.
    """
    if capacity <= 0:
        return 0.0
    served = sum(k * poisson.pmf(k, request_mean) for k in range(capacity))
    served += capacity * poisson.sf(capacity - 1, request_mean)
    return float(served)


def next_day_distribution(request_mean, return_mean):
    """
    For every number of available bicycles c (0..20), return a vector of
    length 21 giving the distribution of the number of bicycles at the
    end of the day (after rentals and returns).

    Because requests and returns are independent Poisson variables and the
    capacity is applied per location, the two locations' next-state
    distributions factorise.  This is what makes policy iteration cheap:
    P(s' | s, a) = q1[c1'] * q2[c2'].
    """
    table = np.zeros((MAX_BIKES + 1, MAX_BIKES + 1))

    for c in range(MAX_BIKES + 1):
        # Distribution of bicycles left after serving the requests.
        # If k < c requests arrive, c - k bicycles remain; if k >= c
        # requests arrive, every bicycle is rented and none remain.
        remaining = np.zeros(c + 1)
        for k in range(c):
            remaining[c - k] = poisson.pmf(k, request_mean)
        remaining[0] = poisson.sf(c - 1, request_mean)

        # Add the Poisson returns, capping at the parking capacity.
        for k, probability in enumerate(remaining):
            if probability == 0.0:
                continue
            last = MAX_BIKES - k
            for n in range(last):
                table[c, k + n] += probability * poisson.pmf(n, return_mean)
            # Any number of returns >= last fills the location to capacity.
            table[c, MAX_BIKES] += probability * poisson.sf(last - 1, return_mean)

    # Guard against tiny floating-point drift in the Poisson tails.
    table /= table.sum(axis=1, keepdims=True)
    return table


# Precompute the per-location rental revenue and next-state distribution.
EXPECTED_SERVED_1 = np.array(
    [expected_served(c, REQUESTS_MEAN[0]) for c in range(MAX_BIKES + 1)]
)
EXPECTED_SERVED_2 = np.array(
    [expected_served(c, REQUESTS_MEAN[1]) for c in range(MAX_BIKES + 1)]
)
NEXT_DIST_1 = next_day_distribution(REQUESTS_MEAN[0], RETURNS_MEAN[0])
NEXT_DIST_2 = next_day_distribution(REQUESTS_MEAN[1], RETURNS_MEAN[1])


# ---------------------------------------------------------------------------
# The MDP
# ---------------------------------------------------------------------------
class GbikeMDP:
    """
    Gbike MDP.  Set modified=False for the original Jack's car rental
    formulation, modified=True for the submission problem (free shuttle
    and second-parking-lot cost).
    """

    def __init__(self, modified=True, gamma=DISCOUNT):
        self.modified = bool(modified)
        self.gamma = float(gamma)
        self.states = [(c1, c2)
                       for c1 in range(MAX_BIKES + 1)
                       for c2 in range(MAX_BIKES + 1)]
        self.state_index = {s: i for i, s in enumerate(self.states)}
        self._action_cache = {}

    # -- actions and costs ---------------------------------------------------
    def feasible_actions(self, c1, c2):
        """All moves that respect the five-bike limit and both capacities."""
        cached = self._action_cache.get((c1, c2))
        if cached is not None:
            return cached

        actions = []
        for action in ACTIONS:
            after_1 = c1 - action
            after_2 = c2 + action
            if 0 <= after_1 <= MAX_BIKES and 0 <= after_2 <= MAX_BIKES:
                actions.append(action)
        self._action_cache[(c1, c2)] = actions
        return actions

    def move_cost(self, action):
        """Cost in INR of moving `action` bicycles (positive = 1 -> 2)."""
        if not self.modified:
            return MOVE_COST * abs(action)
        if action <= 0:
            return MOVE_COST * abs(action)
        # The first bicycle from location 1 to location 2 is free.
        return MOVE_COST * max(0, action - FREE_MOVES_1_TO_2)

    def parking_cost(self, after_1, after_2):
        """Fixed cost of the second parking lot, per location, if needed."""
        if not self.modified:
            return 0.0
        cost = 0.0
        if after_1 > PARKING_THRESHOLD:
            cost += PARKING_COST
        if after_2 > PARKING_THRESHOLD:
            cost += PARKING_COST
        return cost

    # -- reward and transitions ---------------------------------------------
    def reward(self, c1, c2, action):
        """
        Expected immediate reward of taking `action` in state (c1, c2):
        rental income minus moving cost minus overnight parking cost.
        """
        after_1 = c1 - action
        after_2 = c2 + action

        income = RENTAL_INCOME * (
            EXPECTED_SERVED_1[after_1] + EXPECTED_SERVED_2[after_2]
        )
        return income - self.move_cost(action) - self.parking_cost(after_1, after_2)

    def transition_components(self, c1, c2, action):
        """Return (q1, q2), the next-state distributions of the two locations."""
        after_1 = c1 - action
        after_2 = c2 + action
        return NEXT_DIST_1[after_1], NEXT_DIST_2[after_2]

    def expected_next_value(self, value, c1, c2, action):
        """sum_{s'} P(s'|s,a) V(s') using the factorised distribution."""
        q1, q2 = self.transition_components(c1, c2, action)
        return float(q1 @ value @ q2)

    def q_value(self, value, c1, c2, action):
        return self.reward(c1, c2, action) + self.gamma * self.expected_next_value(
            value, c1, c2, action
        )

    # -- policy iteration ----------------------------------------------------
    def evaluate_policy(self, policy):
        """
        Exact policy evaluation: solve (I - gamma P_pi) V = r_pi.

        `policy` is a 21x21 integer array of actions.  Returns a 21x21
        array of state values.
        """
        n = len(self.states)
        transition = np.zeros((n, n))
        reward = np.zeros(n)

        for state, i in self.state_index.items():
            c1, c2 = state
            action = int(policy[c1, c2])
            reward[i] = self.reward(c1, c2, action)
            q1, q2 = self.transition_components(c1, c2, action)
            transition[i] = np.outer(q1, q2).ravel()

        system = np.eye(n) - self.gamma * transition
        values = np.linalg.solve(system, reward)
        return values.reshape(MAX_BIKES + 1, MAX_BIKES + 1)

    def improve_policy(self, policy, values, tol=1e-9):
        """
        Greedy policy improvement.  Ties keep the existing action, which
        prevents cycling between equally good moves.
        """
        new_policy = policy.copy()
        changed = False

        for c1 in range(MAX_BIKES + 1):
            for c2 in range(MAX_BIKES + 1):
                current_action = int(policy[c1, c2])
                current_q = self.q_value(values, c1, c2, current_action)
                best_action = current_action
                best_q = current_q

                for action in self.feasible_actions(c1, c2):
                    q = self.q_value(values, c1, c2, action)
                    if q > best_q + tol:
                        best_q = q
                        best_action = action

                if best_action != current_action:
                    new_policy[c1, c2] = best_action
                    changed = True

        return new_policy, changed

    def policy_iteration(self, initial_policy=None, max_iter=100, verbose=True):
        """
        Run policy iteration until the policy is stable.

        Returns a dict with the value function, policy, iteration count and
        per-iteration diagnostics.
        """
        if initial_policy is None:
            # Move nothing everywhere: a simple, always-feasible policy.
            policy = np.zeros((MAX_BIKES + 1, MAX_BIKES + 1), dtype=int)
        else:
            policy = np.array(initial_policy, dtype=int, copy=True)

        history = []
        converged = False
        values = None

        for iteration in range(1, max_iter + 1):
            values = self.evaluate_policy(policy)
            new_policy, changed = self.improve_policy(policy, values)
            improvement = int(np.sum(new_policy != policy))

            history.append({
                "iteration": iteration,
                "changed_states": improvement,
                "max_value": float(values.max()),
                "min_value": float(values.min()),
                "value_at_10_10": float(values[10, 10]),
            })

            if verbose:
                print(f"  iteration {iteration:2d}: changed {improvement:3d} "
                      f"states, V(10,10) = {values[10, 10]:+9.4f}")

            if not changed:
                converged = True
                break
            policy = new_policy

        if not converged:
            # The iteration limit was reached after installing a new policy;
            # report values that actually belong to the returned policy.
            values = self.evaluate_policy(policy)

        return {
            "values": values,
            "policy": policy,
            "iterations": iteration,
            "converged": converged,
            "history": history,
        }


# ---------------------------------------------------------------------------
# Console / file output
# ---------------------------------------------------------------------------
def print_policy(policy, title):
    print(f"\n{title}")
    print("rows = bicycles at location 1, columns = bicycles at location 2")
    print("positive = move 1 -> 2, negative = move 2 -> 1")
    header = "    " + " ".join(f"{c:>3d}" for c in range(MAX_BIKES + 1))
    print(header)
    for c1 in range(MAX_BIKES + 1):
        row = " ".join(f"{int(policy[c1, c2]):>3d}"
                       for c2 in range(MAX_BIKES + 1))
        print(f"{c1:>3d} {row}")


def save_policy_csv(policy, path):
    np.savetxt(path, policy, fmt="%d", delimiter=",")


def save_values_csv(values, path):
    np.savetxt(path, values, fmt="%.6f", delimiter=",")


def _draw_value(ax, values, title):
    image = ax.imshow(values, origin="lower", cmap="viridis", aspect="auto")
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel("bicycles at location 2")
    ax.set_ylabel("bicycles at location 1")
    ax.set_xticks(range(0, MAX_BIKES + 1, 2))
    ax.set_yticks(range(0, MAX_BIKES + 1, 2))
    ax.set_title(title)


def _draw_policy(ax, policy, title):
    image = ax.imshow(policy, origin="lower", cmap="coolwarm",
                      vmin=-MAX_MOVE, vmax=MAX_MOVE, aspect="auto")
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    for c1 in range(MAX_BIKES + 1):
        for c2 in range(MAX_BIKES + 1):
            action = int(policy[c1, c2])
            if action != 0:
                colour = "white" if abs(action) >= 3 else "black"
                ax.text(c2, c1, str(action), ha="center", va="center",
                        fontsize=5.2, color=colour)

    ax.set_xlabel("bicycles at location 2")
    ax.set_ylabel("bicycles at location 1")
    ax.set_xticks(range(0, MAX_BIKES + 1, 2))
    ax.set_yticks(range(0, MAX_BIKES + 1, 2))
    ax.set_title(title)


def plot_solution(values, policy, title, path):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))
    _draw_value(axes[0], values, f"{title}\nstate values (INR)")
    _draw_policy(axes[1], policy, f"{title}\noptimal overnight moves")
    fig.tight_layout()
    fig.savefig(path, dpi=165, bbox_inches="tight")
    plt.close(fig)


def plot_comparison(original, modified, path):
    """Plot original and modified value functions / policies side by side."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    _draw_value(axes[0, 0], original["values"],
                "Original Gbike: V(s)")
    _draw_value(axes[0, 1], modified["values"],
                "Modified Gbike: V(s)")
    _draw_policy(axes[1, 0], original["policy"],
                 "Original Gbike: policy")
    _draw_policy(axes[1, 1], modified["policy"],
                 "Modified Gbike: policy")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_differences(original, modified, path):
    """Value difference and policy difference maps."""
    value_difference = modified["values"] - original["values"]
    policy_difference = modified["policy"] != original["policy"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))

    limit = float(np.max(np.abs(value_difference)))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    image = axes[0].imshow(value_difference, origin="lower", cmap="RdBu_r",
                           norm=norm, aspect="auto")
    plt.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)
    axes[0].set_title("V(modified) - V(original)")
    axes[0].set_xlabel("bicycles at location 2")
    axes[0].set_ylabel("bicycles at location 1")
    axes[0].set_xticks(range(0, MAX_BIKES + 1, 2))
    axes[0].set_yticks(range(0, MAX_BIKES + 1, 2))

    image = axes[1].imshow(policy_difference, origin="lower", cmap="Greys",
                           aspect="auto", vmin=0, vmax=1)
    plt.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04,
                 ticks=[0, 1])
    axes[1].set_title(
        f"Policy differs in {int(policy_difference.sum())} of "
        f"{policy_difference.size} states"
    )
    axes[1].set_xlabel("bicycles at location 2")
    axes[1].set_ylabel("bicycles at location 1")
    axes[1].set_xticks(range(0, MAX_BIKES + 1, 2))
    axes[1].set_yticks(range(0, MAX_BIKES + 1, 2))

    fig.tight_layout()
    fig.savefig(path, dpi=165, bbox_inches="tight")
    plt.close(fig)


def _json_solution(solution):
    return {
        "converged": bool(solution["converged"]),
        "iterations": int(solution["iterations"]),
        "values": solution["values"].tolist(),
        "policy": solution["policy"].astype(int).tolist(),
        "history": solution["history"],
    }


# ---------------------------------------------------------------------------
# Experiment driver
# ---------------------------------------------------------------------------
def run_gbike_experiments(output_dir):
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 72)
    print("GBIKE 1/2: original formulation")
    print("=" * 72)
    original_mdp = GbikeMDP(modified=False)
    original = original_mdp.policy_iteration()
    print(f"Converged: {original['converged']} after "
          f"{original['iterations']} policy iterations")
    print(f"V(10,10) = {original['values'][10, 10]:.4f} INR")
    print_policy(original["policy"], "Original optimal policy")

    print("\n" + "=" * 72)
    print("GBIKE 2/2: modified formulation")
    print("(free 1->2 shuttle + INR 4 second parking lot when >10 bikes)")
    print("=" * 72)
    modified_mdp = GbikeMDP(modified=True)
    modified = modified_mdp.policy_iteration()
    print(f"Converged: {modified['converged']} after "
          f"{modified['iterations']} policy iterations")
    print(f"V(10,10) = {modified['values'][10, 10]:.4f} INR")
    print_policy(modified["policy"], "Modified optimal policy")

    # -- summary statistics -------------------------------------------------
    policy_difference = modified["policy"] != original["policy"]
    value_difference = modified["values"] - original["values"]

    print("\n" + "=" * 72)
    print("COMPARISON")
    print("=" * 72)
    print(f"States where the optimal move changes: "
          f"{int(policy_difference.sum())} / {policy_difference.size}")
    print(f"Mean value change: {value_difference.mean():+.4f} INR")
    print(f"Max value gain:    {value_difference.max():+.4f} INR")
    print(f"Min value change:  {value_difference.min():+.4f} INR")

    selected_states = [(10, 10), (0, 0), (20, 0), (0, 20), (20, 20),
                       (5, 15), (15, 5), (12, 8), (8, 12)]
    print("\nValue at selected states (INR):")
    print(f"{'state':>10} {'original':>12} {'modified':>12} {'change':>12}")
    for state in selected_states:
        c1, c2 = state
        print(f"{str(state):>10} {original['values'][c1, c2]:>12.3f} "
              f"{modified['values'][c1, c2]:>12.3f} "
              f"{value_difference[c1, c2]:>+12.3f}")

    # -- save artefacts -----------------------------------------------------
    save_policy_csv(original["policy"],
                    os.path.join(output_dir, "gbike_original_policy.csv"))
    save_policy_csv(modified["policy"],
                    os.path.join(output_dir, "gbike_modified_policy.csv"))
    save_values_csv(original["values"],
                    os.path.join(output_dir, "gbike_original_values.csv"))
    save_values_csv(modified["values"],
                    os.path.join(output_dir, "gbike_modified_values.csv"))

    plot_solution(original["values"], original["policy"],
                  "Original Gbike (INR 2 per move)",
                  os.path.join(output_dir, "gbike_original.png"))
    plot_solution(modified["values"], modified["policy"],
                  "Modified Gbike (free 1->2 shuttle, second-lot cost)",
                  os.path.join(output_dir, "gbike_modified.png"))
    plot_comparison(original, modified,
                    os.path.join(output_dir, "gbike_comparison.png"))
    plot_differences(original, modified,
                     os.path.join(output_dir, "gbike_differences.png"))

    results = {
        "parameters": {
            "max_bikes": MAX_BIKES,
            "max_move": MAX_MOVE,
            "requests_mean": REQUESTS_MEAN,
            "returns_mean": RETURNS_MEAN,
            "rental_income": RENTAL_INCOME,
            "move_cost": MOVE_COST,
            "discount": DISCOUNT,
            "modified_parking_threshold": PARKING_THRESHOLD,
            "modified_parking_cost": PARKING_COST,
        },
        "original": _json_solution(original),
        "modified": _json_solution(modified),
        "comparison": {
            "changed_policy_states": int(policy_difference.sum()),
            "mean_value_change": float(value_difference.mean()),
            "max_value_change": float(value_difference.max()),
            "min_value_change": float(value_difference.min()),
        },
    }
    with open(os.path.join(output_dir, "gbike_results.json"), "w",
              encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print(f"\nGbike outputs written to: {output_dir}")

    # Keep a convenient handle for run_all/main.
    return {
        "original": original,
        "modified": modified,
        "json": results,
    }


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    run_gbike_experiments(os.path.join(here, "outputs"))
