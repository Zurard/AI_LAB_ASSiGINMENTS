"""
Lab 10 - Part 1: Markov Decision Process and Value Iteration
================================================================
The classic 4x3 grid world from Russell & Norvig, Artificial Intelligence:
A Modern Approach (3rd ed.), Chapter 17.

Grid (row, column), row 0 is the top row:

        (0,0)   (0,1)   (0,2)   (0,3)=+1
        (1,0)   WALL    (1,2)   (1,3)=-1
        (2,0)   (2,1)   (2,2)   (2,3)
         START

Actions: Up, Down, Left, Right.  The intended action succeeds with
probability 0.8; with probability 0.1 each the agent slips to one of the
two squares at right angles to the intended direction.  Bumping into an
outer wall or the internal wall leaves the agent in the same square.

Reward: the living reward r(s) is collected on every step from a
non-terminal state (this is the convention used by Russell & Norvig, and
is what reproduces the textbook values for r(s) = -0.04).  Entering a
terminal state ends the episode and is worth +1 or -1.

The module implements both value iteration (the method requested in the
assignment) and policy iteration (used to cross-check the answer).
"""

import json
import os
from collections import OrderedDict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


# ---------------------------------------------------------------------------
# Environment definition
# ---------------------------------------------------------------------------
ROWS, COLS = 3, 4
WALLS = {(1, 1)}
TERMINALS = {(0, 3): +1.0, (1, 3): -1.0}
START = (2, 0)

ACTIONS = ("Up", "Down", "Left", "Right")
ACTION_DELTAS = {
    "Up": (-1, 0),
    "Down": (1, 0),
    "Left": (0, -1),
    "Right": (0, 1),
}
ARROWS = {"Up": "^", "Down": "v", "Left": "<", "Right": ">"}

# The two directions perpendicular to each action.  These are used for the
# 0.1 / 0.1 slip probabilities.
PERPENDICULARS = {
    "Up": ("Left", "Right"),
    "Down": ("Left", "Right"),
    "Left": ("Up", "Down"),
    "Right": ("Up", "Down"),
}


def all_cells():
    """Return every non-wall cell, including terminal cells."""
    return [(r, c) for r in range(ROWS) for c in range(COLS) if (r, c) not in WALLS]


def nonterminal_cells():
    return [s for s in all_cells() if s not in TERMINALS]


def is_free(cell):
    """True if the cell is inside the grid and is not a wall."""
    r, c = cell
    return 0 <= r < ROWS and 0 <= c < COLS and cell not in WALLS


def transition_model(state):
    """
    Build the transition model for every action in a state.

    Returns
    -------
    dict: action -> OrderedDict(next_state -> probability)

    Outcomes that bump into a wall are merged into the "stay" outcome, so
    the probabilities for an action always sum to 1.
    """
    transitions = OrderedDict()
    for action in ACTIONS:
        outcomes = OrderedDict()
        dr, dc = ACTION_DELTAS[action]
        perp_1, perp_2 = PERPENDICULARS[action]

        # (probability, direction) triples: intended move + two slips
        for prob, direction in ((0.8, action), (0.1, perp_1), (0.1, perp_2)):
            mr, mc = ACTION_DELTAS[direction]
            nxt = (state[0] + mr, state[1] + mc)
            if not is_free(nxt):
                nxt = state  # bumped into a wall / edge: stay put
            outcomes[nxt] = outcomes.get(nxt, 0.0) + prob

        transitions[action] = outcomes
    return transitions


# Precompute the model once; the environment is stationary.
TRANSITIONS = {s: transition_model(s) for s in nonterminal_cells()}


# ---------------------------------------------------------------------------
# Value iteration
# ---------------------------------------------------------------------------
def q_values(state, V, living_reward, gamma):
    """Q(s, a) for every action in a non-terminal state."""
    q = OrderedDict()
    for action in ACTIONS:
        expected_future = sum(
            prob * V[nxt] for nxt, prob in TRANSITIONS[state][action].items()
        )
        q[action] = living_reward + gamma * expected_future
    return q


def greedy_policy(V, living_reward, gamma):
    """Greedy policy; ties are broken in the order Up, Down, Left, Right."""
    policy = {}
    for state in nonterminal_cells():
        q = q_values(state, V, living_reward, gamma)
        policy[state] = max(ACTIONS, key=lambda a: q[a])
    return policy


def value_iteration(living_reward, gamma=1.0, theta=1e-10, max_iter=10000,
                    track_history=False):
    """
    Run value iteration.

    V(s) <- R(s) + gamma * max_a sum_{s'} P(s'|s,a) V(s')

    Terminal states keep their terminal payoff and are never updated.

    Returns
    -------
    V : 3x4 numpy array (np.nan for the wall)
    policy : dict state -> action
    iterations : number of sweeps performed
    converged : bool
    history : list of V snapshots (only if track_history=True)
    """
    V = np.full((ROWS, COLS), np.nan)
    for state, value in TERMINALS.items():
        V[state] = value
    for state in nonterminal_cells():
        V[state] = 0.0

    history = [V.copy()] if track_history else None
    converged = False
    iteration = 0

    for iteration in range(1, max_iter + 1):
        delta = 0.0
        V_new = V.copy()
        for state in nonterminal_cells():
            q = q_values(state, V, living_reward, gamma)
            best = max(q.values())
            delta = max(delta, abs(best - V[state]))
            V_new[state] = best
        V = V_new
        if track_history:
            history.append(V.copy())
        if delta < theta:
            converged = True
            break

    if gamma == 1.0 and living_reward > 0:
        # Positive living reward + no discount + an avoidable terminal state
        # gives an improper MDP: the agent can collect reward forever.
        # Reaching max_iter without convergence is expected in that case.
        pass

    return V, greedy_policy(V, living_reward, gamma), iteration, converged, history


# ---------------------------------------------------------------------------
# Policy iteration (cross-check)
# ---------------------------------------------------------------------------
def policy_evaluation(policy, living_reward, gamma):
    """
    Solve the linear system V = R + gamma P V exactly.

    Only non-terminal states are unknowns; terminal values enter the
    right-hand side.  Raises np.linalg.LinAlgError for an improper policy
    when gamma == 1.
    """
    states = nonterminal_cells()
    index = {s: i for i, s in enumerate(states)}
    n = len(states)
    A = np.eye(n)
    b = np.full(n, living_reward, dtype=float)

    for state in states:
        i = index[state]
        for nxt, prob in TRANSITIONS[state][policy[state]].items():
            if nxt in TERMINALS:
                b[i] += gamma * prob * TERMINALS[nxt]
            else:
                A[i, index[nxt]] -= gamma * prob

    solution = np.linalg.solve(A, b)
    V = np.full((ROWS, COLS), np.nan)
    for state, value in TERMINALS.items():
        V[state] = value
    for state in states:
        V[state] = solution[index[state]]
    return V


def policy_iteration(living_reward, gamma=1.0, max_iter=100, tol=1e-10):
    """
    Standard policy iteration.

    Returns V, policy, iterations, converged.
    """
    policy = {s: "Up" for s in nonterminal_cells()}

    for iteration in range(1, max_iter + 1):
        V = policy_evaluation(policy, living_reward, gamma)

        changed = False
        for state in nonterminal_cells():
            q = q_values(state, V, living_reward, gamma)
            best_action = max(ACTIONS, key=lambda a: q[a])
            if q[best_action] > q[policy[state]] + tol:
                policy[state] = best_action
                changed = True

        if not changed:
            return V, policy, iteration, True

    return policy_evaluation(policy, living_reward, gamma), policy, max_iter, False


# ---------------------------------------------------------------------------
# Pretty-printing / plotting helpers
# ---------------------------------------------------------------------------
def format_policy(policy):
    """Return a 3x4 list of arrow strings (terminal cells marked T)."""
    grid = []
    for r in range(ROWS):
        row = []
        for c in range(COLS):
            cell = (r, c)
            if cell in WALLS:
                row.append("##")
            elif cell in TERMINALS:
                row.append("T")
            else:
                row.append(ARROWS[policy[cell]])
        grid.append(row)
    return grid


def print_value_function(V, policy, title):
    print(f"\n{title}")
    print("-" * len(title))
    for r in range(ROWS):
        cells = []
        for c in range(COLS):
            cell = (r, c)
            if cell in WALLS:
                cells.append("  WALL ")
            elif cell in TERMINALS:
                cells.append(f" {V[cell]:+.3f}*")
            else:
                cells.append(f" {V[cell]:+.3f}{ARROWS[policy[cell]]}")
        print(" | ".join(cells))


def _draw_grid(ax, V, policy, title):
    """Draw a value heat-map with policy arrows."""
    masked = np.ma.array(V, mask=np.isnan(V))
    finite = V[~np.isnan(V)]
    vmin, vmax = float(finite.min()), float(finite.max())
    if vmin == vmax:
        vmin, vmax = vmin - 1.0, vmax + 1.0

    if vmin < 0 < vmax:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
        cmap = plt.get_cmap("RdYlGn")
    else:
        norm = None
        cmap = plt.get_cmap("RdYlGn")

    image = ax.imshow(masked, cmap=cmap, norm=norm, origin="upper")
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    for r in range(ROWS):
        for c in range(COLS):
            cell = (r, c)
            if cell in WALLS:
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                           facecolor="0.25", edgecolor="0.1"))
                ax.text(c, r, "WALL", ha="center", va="center",
                        color="white", fontsize=9, fontweight="bold")
            elif cell in TERMINALS:
                ax.text(c, r, f"{V[cell]:+.2f}\nT", ha="center", va="center",
                        fontsize=11, fontweight="bold")
            else:
                ax.text(c, r, f"{V[cell]:+.3f}\n{ARROWS[policy[cell]]}",
                        ha="center", va="center", fontsize=10)
            ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                       fill=False, edgecolor="0.4", linewidth=1))

    ax.set_xticks(range(COLS))
    ax.set_yticks(range(ROWS))
    ax.set_xticklabels([f"col {c}" for c in range(COLS)])
    ax.set_yticklabels([f"row {r}" for r in range(ROWS)])
    ax.set_title(title)
    ax.set_xlim(-0.5, COLS - 0.5)
    ax.set_ylim(ROWS - 0.5, -0.5)


def plot_value_and_policy(V, policy, title, path):
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    _draw_grid(ax, V, policy, title)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_reward_sweep(sweep, path):
    """
    Plot one column of value/policy panels per living reward.
    `sweep` maps reward -> (V, policy).
    """
    rewards = list(sweep.keys())
    n = len(rewards)
    fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 4.6))
    if n == 1:
        axes = [axes]
    for ax, reward in zip(axes, rewards):
        V, policy = sweep[reward]
        _draw_grid(ax, V, policy, f"r(s) = {reward:g}")
    fig.suptitle("4x3 grid world - value iteration with gamma = 0.9", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_gamma1_behaviour(histories, path, max_iterations=120):
    """
    Show why positive living rewards diverge when gamma = 1.
    `histories` maps living reward -> value history at START.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for reward, history in histories.items():
        values = [snap[START] for snap in history[:max_iterations + 1]]
        ax.plot(range(len(values)), values, linewidth=2,
                label=f"r(s) = {reward:g}")
    ax.set_xlabel("Value-iteration sweep")
    ax.set_ylabel(f"V({START})")
    ax.set_title("gamma = 1: negative living rewards converge, "
                 "positive ones diverge")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Experiment driver
# ---------------------------------------------------------------------------
def _serialise_result(V, policy, iterations, converged, gamma, living_reward):
    values = np.where(np.isnan(V), None, V).tolist()
    return {
        "gamma": gamma,
        "living_reward": living_reward,
        "converged": bool(converged),
        "iterations": int(iterations),
        "values": values,
        "policy": {
            f"{r},{c}": action
            for (r, c), action in sorted(policy.items())
        },
    }


def run_gridworld_experiments(output_dir):
    """Run all grid-world experiments and write plots/JSON into output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    # ---- 1. Canonical textbook experiment: r(s) = -0.04, gamma = 1 ----
    print("\n" + "=" * 72)
    print("GRID WORLD 1/3: canonical experiment  r(s) = -0.04, gamma = 1.0")
    print("=" * 72)
    V_canon, pol_canon, it_canon, conv_canon, _ = value_iteration(
        -0.04, gamma=1.0, theta=1e-10, max_iter=10000
    )
    print(f"Value iteration: {it_canon} sweeps, converged = {conv_canon}")
    print_value_function(V_canon, pol_canon, "Value function (r = -0.04, gamma = 1)")
    print("Policy:")
    for row in format_policy(pol_canon):
        print("   " + "  ".join(row))
    plot_value_and_policy(
        V_canon, pol_canon,
        "Value iteration: r(s) = -0.04, gamma = 1.0",
        os.path.join(output_dir, "gridworld_canonical.png"),
    )
    results["canonical"] = _serialise_result(
        V_canon, pol_canon, it_canon, conv_canon, 1.0, -0.04
    )

    # ---- 2. Cross-check with policy iteration ----
    print("\nCross-check with policy iteration ...")
    try:
        V_pi, pol_pi, it_pi, conv_pi = policy_iteration(-0.04, gamma=1.0)
        max_diff = float(np.nanmax(np.abs(V_canon - V_pi)))
        same_policy = all(pol_canon[s] == pol_pi[s] for s in nonterminal_cells())
        print(f"Policy iteration: {it_pi} iterations, converged = {conv_pi}")
        print(f"max |V_value_iteration - V_policy_iteration| = {max_diff:.3e}")
        print(f"policies identical: {same_policy}")
        results["policy_iteration_check"] = {
            "iterations": int(it_pi),
            "converged": bool(conv_pi),
            "max_value_difference": max_diff,
            "policies_identical": bool(same_policy),
        }
    except np.linalg.LinAlgError as exc:  # pragma: no cover - defensive
        print(f"Policy iteration could not evaluate the initial policy: {exc}")
        results["policy_iteration_check"] = {"error": str(exc)}

    # ---- 3. Reward sweep with gamma = 0.9 ----
    # With gamma = 1 every positive living reward makes the MDP improper
    # (the agent can loop forever collecting reward), so a discount is
    # required to obtain finite value functions.  gamma = 0.9 is used for
    # all five rewards so that the comparison is consistent.
    print("\n" + "=" * 72)
    print("GRID WORLD 2/3: reward sweep with gamma = 0.9")
    print("=" * 72)
    sweep_rewards = [-0.04, -2.0, 0.1, 0.02, 1.0]
    sweep = {}
    sweep_json = {}
    for reward in sweep_rewards:
        V, policy, iterations, converged, _ = value_iteration(
            reward, gamma=0.9, theta=1e-12, max_iter=10000
        )
        sweep[reward] = (V, policy)
        sweep_json[f"{reward:g}"] = _serialise_result(
            V, policy, iterations, converged, 0.9, reward
        )
        print(f"\nr(s) = {reward:g}: {iterations} sweeps, "
              f"converged = {converged}")
        print_value_function(V, policy, f"Value function (r = {reward:g}, gamma = 0.9)")
        print("Policy:")
        for row in format_policy(policy):
            print("   " + "  ".join(row))

    plot_reward_sweep(sweep, os.path.join(output_dir, "gridworld_reward_sweep.png"))
    results["reward_sweep_gamma_0.9"] = sweep_json

    # ---- 4. gamma = 1 behaviour for the same rewards ----
    print("\n" + "=" * 72)
    print("GRID WORLD 3/3: gamma = 1.0 behaviour (divergence check)")
    print("=" * 72)
    histories = {}
    gamma1_json = {}
    for reward in sweep_rewards:
        # 300 sweeps is enough to expose the linear growth of positive
        # living rewards; negative rewards converge in far fewer sweeps.
        V, policy, iterations, converged, history = value_iteration(
            reward, gamma=1.0, theta=1e-10, max_iter=300, track_history=True
        )
        histories[reward] = history
        entry = {
            "gamma": 1.0,
            "living_reward": reward,
            "converged": bool(converged),
            "iterations": int(iterations),
            "V_start_after_cap": float(V[START]),
            "V_start_history": [float(snap[START]) for snap in history],
        }
        if converged:
            entry["values"] = np.where(np.isnan(V), None, V).tolist()
            entry["policy"] = {
                f"{r},{c}": action
                for (r, c), action in sorted(policy.items())
            }
        gamma1_json[f"{reward:g}"] = entry
        status = "converged" if converged else "NOT converged (diverging)"
        print(f"r(s) = {reward:>5g}: {iterations:3d} sweeps -> {status}; "
              f"V(start) = {V[START]:.4f}")

    plot_gamma1_behaviour(
        histories,
        os.path.join(output_dir, "gridworld_gamma1_divergence.png"),
    )
    results["gamma_1_behaviour"] = gamma1_json

    with open(os.path.join(output_dir, "gridworld_results.json"), "w",
              encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print(f"\nGrid-world outputs written to: {output_dir}")
    return results


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    run_gridworld_experiments(os.path.join(here, "outputs"))
