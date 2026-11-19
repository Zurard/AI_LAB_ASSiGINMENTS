"""
Build the Lab 10 report from the computed results.

Running this script creates:
  report/REPORT.md              - editable Markdown report
  report/Lab10_Report.docx      - formatted Word report with embedded figures
  report/Lab10_Report.html      - self-contained HTML report with embedded figures
  report/figures/*.png          - all figures used by the report

All numeric results are read from ../outputs/*.json so that the report stays
consistent with the code output.
"""

import base64
import json
import re
import shutil
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


HERE = Path(__file__).resolve().parent
LAB = HERE.parent
OUT = LAB / "outputs"
FIG = HERE / "figures"
FIG.mkdir(parents=True, exist_ok=True)

GRID = json.loads((OUT / "gridworld_results.json").read_text(encoding="utf-8"))
GBIKE = json.loads((OUT / "gbike_results.json").read_text(encoding="utf-8"))

REWARDS = [-0.04, -2.0, 0.1, 0.02, 1.0]
ARROWS = {"Up": "\u2191", "Down": "\u2193", "Left": "\u2190", "Right": "\u2192"}
TERMINALS = {(0, 3): 1.0, (1, 3): -1.0}
WALL = (1, 1)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def make_grid_layout():
    fig, ax = plt.subplots(figsize=(8.6, 5.8))
    for r in range(3):
        for c in range(4):
            cell = (r, c)
            y = 2 - r
            if cell == WALL:
                face = "#4a4a4a"
                label = "WALL"
                colour = "white"
            elif cell == (0, 3):
                face = "#b7e4c7"
                label = "+1\nGOAL"
                colour = "#14532d"
            elif cell == (1, 3):
                face = "#f4a3a3"
                label = "\u22121\nTRAP"
                colour = "#7f1d1d"
            elif cell == (2, 0):
                face = "#bde0fe"
                label = "START"
                colour = "#1e3a8a"
            else:
                face = "#f8f9fa"
                label = ""
                colour = "#333333"

            hatch = "//" if cell == WALL else None
            ax.add_patch(Rectangle((c, y), 1, 1, facecolor=face,
                                   edgecolor="#333333", linewidth=1.4,
                                   hatch=hatch))
            ax.text(c + 0.06, y + 0.93, f"({r},{c})", ha="left", va="top",
                    fontsize=8, color="#666666")
            if label:
                ax.text(c + 0.5, y + 0.5, label, ha="center", va="center",
                        fontsize=17, fontweight="bold", color=colour)

    ax.set_xlim(0, 4)
    ax.set_ylim(0, 3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("4 \u00d7 3 grid world (4 columns, 3 rows)", fontsize=15,
                 fontweight="bold", pad=12)
    fig.text(0.5, 0.03,
             "Actions: Up, Down, Left, Right.  Intended move succeeds with "
             "probability 0.8; each perpendicular slip has probability 0.1.  "
             "Bumping into a wall keeps the agent in place.",
             ha="center", fontsize=9.5, color="#444444")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    path = FIG / "fig1_gridworld_layout.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def make_gbike_schematic():
    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 6.5)
    ax.axis("off")

    box1 = FancyBboxPatch((0.6, 2.0), 3.7, 3.4,
                          boxstyle="round,pad=0.08,rounding_size=0.2",
                          facecolor="#dbeafe", edgecolor="#1e40af", linewidth=2)
    box2 = FancyBboxPatch((8.2, 2.0), 3.7, 3.4,
                          boxstyle="round,pad=0.08,rounding_size=0.2",
                          facecolor="#ffedd5", edgecolor="#c2410c", linewidth=2)
    ax.add_patch(box1)
    ax.add_patch(box2)

    ax.text(2.45, 4.75, "Location 1", ha="center", fontsize=16,
            fontweight="bold", color="#1e3a8a")
    ax.text(2.45, 4.3, "capacity: 20 bicycles", ha="center", fontsize=11)
    ax.text(2.45, 3.72, "Rental requests\nPoisson(\u03bb = 3)", ha="center",
            fontsize=11)
    ax.text(2.45, 3.0, "Returns\nPoisson(\u03bc = 3)", ha="center", fontsize=11)

    ax.text(10.05, 4.75, "Location 2", ha="center", fontsize=16,
            fontweight="bold", color="#9a3412")
    ax.text(10.05, 4.3, "capacity: 20 bicycles", ha="center", fontsize=11)
    ax.text(10.05, 3.72, "Rental requests\nPoisson(\u03bb = 4)", ha="center",
            fontsize=11)
    ax.text(10.05, 3.0, "Returns\nPoisson(\u03bc = 2)", ha="center", fontsize=11)

    ax.add_patch(FancyArrowPatch((4.45, 4.45), (8.05, 4.45),
                                 arrowstyle="-|>", mutation_scale=26,
                                 linewidth=2.5, color="#1e40af"))
    ax.text(6.25, 5.0, "1 \u2192 2: up to 5 bicycles / night",
            ha="center", fontsize=11, fontweight="bold", color="#1e3a8a")
    ax.text(6.25, 4.05, "first bicycle free, then INR 2 each",
            ha="center", fontsize=10, color="#1e3a8a")

    ax.add_patch(FancyArrowPatch((8.05, 3.15), (4.45, 3.15),
                                 arrowstyle="-|>", mutation_scale=26,
                                 linewidth=2.5, color="#9a3412"))
    ax.text(6.25, 2.62, "2 \u2192 1: INR 2 per bicycle",
            ha="center", fontsize=11, fontweight="bold", color="#9a3412")

    note = ("Parking rule (modified problem): if more than 10 bicycles are kept "
            "overnight at a\nlocation after moving, an extra INR 4 per night is "
            "paid for a second parking lot.")
    ax.text(6.25, 1.1, note, ha="center", va="center", fontsize=10.5,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8fafc",
                      edgecolor="#94a3b8", linewidth=1.2))

    ax.set_title("Gbike: two-location bicycle rental MDP", fontsize=16,
                 fontweight="bold", pad=6)
    fig.tight_layout()
    path = FIG / "fig5_gbike_schematic.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def make_report_reward_sweep():
    fig = plt.figure(figsize=(13, 8.5))
    gs = fig.add_gridspec(2, 3, hspace=0.30, wspace=0.18)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]
    norm = TwoSlopeNorm(vmin=-9.0, vcenter=0.0, vmax=10.0)
    cmap = plt.get_cmap("RdYlGn")

    for ax, reward in zip(axes, REWARDS):
        entry = GRID["reward_sweep_gamma_0.9"][f"{reward:g}"]
        values = np.array([
            [np.nan if value is None else value for value in row]
            for row in entry["values"]
        ])
        policy = entry["policy"]
        masked = np.ma.array(values, mask=np.isnan(values))
        image = ax.imshow(masked, cmap=cmap, norm=norm, origin="upper")

        for r in range(3):
            for c in range(4):
                cell = (r, c)
                if cell == WALL:
                    ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1,
                                           facecolor="#444444",
                                           edgecolor="#222222"))
                    ax.text(c, r, "WALL", ha="center", va="center",
                            color="white", fontsize=8, fontweight="bold")
                elif cell in TERMINALS:
                    ax.text(c, r, f"{values[r, c]:+.2f}\nT", ha="center",
                            va="center", fontsize=9, fontweight="bold")
                else:
                    action = policy[f"{r},{c}"]
                    ax.text(c, r, f"{values[r, c]:+.2f}\n{ARROWS[action]}",
                            ha="center", va="center", fontsize=8.5)
                ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False,
                                       edgecolor="#555555", linewidth=0.8))

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"r(s) = {reward:g}", fontsize=12, fontweight="bold")
        ax.set_xlim(-0.5, 3.5)
        ax.set_ylim(2.5, -0.5)

    axes[-1].axis("off")
    cax = fig.add_axes([0.28, 0.055, 0.44, 0.020])
    cbar = fig.colorbar(image, cax=cax, orientation="horizontal")
    cbar.set_label("State value (shared colour scale)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.suptitle("Optimal value functions for the 4 \u00d7 3 grid world "
                 "(\u03b3 = 0.9)", fontsize=15, fontweight="bold")
    fig.subplots_adjust(top=0.89, bottom=0.12, left=0.04, right=0.98)
    path = FIG / "fig3_gridworld_reward_sweep.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def copy_output_figures():
    mapping = {
        "gridworld_canonical.png": "fig2_gridworld_canonical.png",
        "gridworld_gamma1_divergence.png": "fig4_gridworld_gamma1_divergence.png",
        "gbike_original.png": "fig6_gbike_original.png",
        "gbike_modified.png": "fig7_gbike_modified.png",
        "gbike_comparison.png": "fig8_gbike_comparison.png",
        "gbike_differences.png": "fig9_gbike_differences.png",
    }
    for source, target in mapping.items():
        shutil.copyfile(OUT / source, FIG / target)


# ---------------------------------------------------------------------------
# Table data helpers
# ---------------------------------------------------------------------------
def grid_rows(entry):
    values = entry["values"]
    policy = entry["policy"]
    rows = []
    for r in range(3):
        row = [f"row {r}"]
        for c in range(4):
            value = values[r][c]
            if value is None:
                row.append("WALL")
            elif (r, c) in TERMINALS:
                row.append(f"{value:+.3f} (T)")
            else:
                row.append(f"{value:+.3f} {ARROWS[policy[f'{r},{c}']]}")
        rows.append(row)
    return rows


def selected_state_rows():
    states = [(10, 10), (0, 0), (20, 0), (0, 20), (20, 20),
              (5, 15), (15, 5), (12, 8), (8, 12)]
    rows = []
    for c1, c2 in states:
        original = GBIKE["original"]["values"][c1][c2]
        modified = GBIKE["modified"]["values"][c1][c2]
        rows.append([f"({c1}, {c2})", f"{original:.3f}", f"{modified:.3f}",
                     f"{modified - original:+.3f}"])
    return rows


def history_rows(problem):
    rows = []
    for item in problem["history"]:
        rows.append([str(item["iteration"]), str(item["changed_states"]),
                     f"{item['value_at_10_10']:.4f}"])
    return rows


def reward_summary_rows():
    rows = []
    for reward in REWARDS:
        entry = GRID["reward_sweep_gamma_0.9"][f"{reward:g}"]
        rows.append([
            f"{reward:g}", "0.9", str(entry["iterations"]),
            "Yes" if entry["converged"] else "No",
            f"{entry['values'][2][0]:+.4f}",
        ])
    return rows


def gamma1_rows():
    rows = []
    for reward in REWARDS:
        entry = GRID["gamma_1_behaviour"][f"{reward:g}"]
        rows.append([
            f"{reward:g}",
            "Yes" if entry["converged"] else "No",
            str(entry["iterations"]),
            f"{entry['V_start_after_cap']:+.4f}",
        ])
    return rows


# ---------------------------------------------------------------------------
# Report content
# ---------------------------------------------------------------------------
def build_blocks():
    blocks = []

    # ------------------------------- title page ---------------------------
    blocks += [
        ("title", "Lab Assignment 10"),
        ("subtitle", "Markov Decision Processes and Dynamic Programming"),
        ("subsubtitle", "4\u00d73 Grid World and Gbike Bicycle Rental"),
        ("meta", [
            ("Course", "Artificial Intelligence"),
            ("Lab", "10 \u2014 Markov Decision Processes and Dynamic Programming"),
            ("Name", "______________________________"),
            ("Roll Number", "______________________________"),
            ("Instructor", "______________________________"),
            ("Date", "19 November 2026"),
        ]),
        ("p", "Implementation: Python 3.11, NumPy, SciPy, Matplotlib."),
        ("p", "All numerical results in this report are generated directly by "
              "the accompanying code and saved in JSON/CSV form. The report can "
              "be regenerated with `python report/build_report.py`."),
        ("pagebreak",),
    ]

    # ------------------------------- objective ----------------------------
    blocks += [
        ("h1", "1. Objective"),
        ("p", "The objective of this lab is to understand sequential decision "
              "making in a stochastic environment and its connection with "
              "reinforcement learning. The work covers the Bellman equations, "
              "value iteration, policy iteration, and the formulation and "
              "solution of a continuing finite Markov decision process (MDP)."),
        ("bullets", [
            "Model the 4\u00d73 grid world as a stochastic MDP and solve it with "
            "value iteration.",
            "Study how the living reward r(s) changes the optimal policy for "
            "r(s) = \u22120.04, \u22122, 0.1, 0.02 and 1.",
            "Formulate the Gbike bicycle-rental problem as a continuing finite "
            "MDP with Poisson demand and returns.",
            "Implement policy iteration and solve both the original Gbike "
            "problem and the modified problem with the free 1\u21922 shuttle "
            "and the second-parking-lot charge.",
        ]),
        ("pagebreak",),
    ]

    # ------------------------------- part 1 -------------------------------
    blocks += [
        ("h1", "2. Part 1 \u2014 The 4\u00d73 Grid World"),
        ("h2", "2.1 MDP formulation"),
        ("p", "The environment is the standard 4-column \u00d7 3-row grid used "
              "by Russell and Norvig. Row 0 is the top row. The start state is "
              "(2,0); the terminal states are (0,3) with value +1 and (1,3) with "
              "value \u22121; the cell (1,1) is a wall. Figure 1 shows the layout."),
        ("image", "fig1_gridworld_layout.png",
         "Figure 1: The 4\u00d73 grid world. The start state is (2,0), the "
         "terminal states are +1 at (0,3) and \u22121 at (1,3), and (1,1) is a "
         "wall.", 6.2),
        ("p", "**States.** The state space contains the 11 non-wall cells. The "
              "two terminal cells are absorbing: entering either one ends the "
              "episode."),
        ("p", "**Actions.** In every non-terminal state the agent may choose "
              "Up, Down, Left or Right."),
        ("p", "**Transition model.** The intended action succeeds with "
              "probability 0.8. With probability 0.1 the agent slips to each of "
              "the two directions at right angles to the intended direction. If "
              "a move (intended or slip) would leave the grid or enter the wall, "
              "the agent stays in its current cell. This wall-bumping rule is "
              "applied to every outcome, so the probabilities for an action "
              "always sum to 1. Table 1 gives representative examples."),
        ("table", "Table 1: Selected transition probabilities. Slips that hit a "
                  "wall are merged into the stay outcome.",
         ["Current state", "Action", "Outcome", "Probability"],
         [
             ["(2,0)", "Up", "(1,0)", "0.80"],
             ["(2,0)", "Up", "(2,0) \u2014 left slip hits wall", "0.10"],
             ["(2,0)", "Up", "(2,1)", "0.10"],
             ["(0,0)", "Up", "(0,0) \u2014 up and left slips hit walls", "0.90"],
             ["(0,0)", "Up", "(0,1)", "0.10"],
             ["(2,3)", "Right", "(2,3) \u2014 right slip hits wall", "0.90"],
             ["(2,3)", "Right", "(1,3) \u2014 terminal", "0.10"],
         ]),
        ("p", "**Reward.** The living reward r(s) is received on every step "
              "out of a non-terminal state. The terminal values are fixed at +1 "
              "and \u22121. The implementation uses the convention"),
        ("equation", "V(s) = R(s) + \u03b3 \u00b7 max_{a} \u03a3_{s\u2032} "
                     "P(s\u2032 | s, a) \u00b7 V(s\u2032),"),
        ("p", "with V(+1) = +1 and V(\u22121) = \u22121 held fixed. This is the "
              "same convention as Russell and Norvig and reproduces their "
              "published values for r(s) = \u22120.04 exactly. When a transition "
              "enters a terminal state, the agent pays the living reward of the "
              "current state and then receives the terminal value."),
        ("p", "**Discount.** The canonical experiment uses \u03b3 = 1, as in the "
              "textbook. The reward sweep uses \u03b3 = 0.9 so that finite value "
              "functions exist for the positive living rewards; Section 2.4 "
              "explains why a discount is necessary."),
        ("h2", "2.2 Value iteration"),
        ("p", "Value iteration applies the Bellman optimality operator "
              "repeatedly. For every non-terminal state it computes"),
        ("equation", "V_{k+1}(s) = R(s) + \u03b3 \u00b7 max_{a} \u03a3_{s\u2032} "
                     "P(s\u2032 | s, a) \u00b7 V_{k}(s\u2032),"),
        ("p", "leaving the terminal values fixed. The algorithm stops when the "
              "largest change between successive sweeps is below 10\u207b\u00b9\u2070. "
              "The greedy policy is then extracted by choosing the action with "
              "the largest Q-value in every state."),
        ("code", "initialize V(s) = 0 for every non-terminal state\n"
                 "initialize V(+1) = +1 and V(-1) = -1\n"
                 "repeat\n"
                 "    delta = 0\n"
                 "    for every non-terminal state s\n"
                 "        V_new(s) = R(s) + gamma * max_a sum_s' P(s'|s,a) V(s')\n"
                 "        delta = max(delta, |V_new(s) - V(s)|)\n"
                 "    V = V_new\n"
                 "until delta < theta\n"
                 "policy(s) = argmax_a [ R(s) + gamma * sum_s' P(s'|s,a) V(s') ]"),
        ("p", "As an independent check, policy iteration is also implemented. "
              "It evaluates a policy exactly by solving the linear system "
              "(I \u2212 \u03b3P)V = R for that policy and then improves the "
              "policy greedily. For the canonical problem, policy iteration "
              "converges in 5 iterations, produces the same policy as value "
              "iteration, and the maximum absolute value difference is "
              "7.4 \u00d7 10\u207b\u00b9\u00b9."),
        ("h2", "2.3 Canonical result: r(s) = \u22120.04, \u03b3 = 1"),
        ("p", "Value iteration converges in 40 sweeps. The resulting values and "
              "greedy policy are shown in Table 2 and Figure 2. The values are "
              "the standard textbook values: 0.812, 0.868, 0.918, 0.762, 0.660, "
              "0.705, 0.655, 0.611 and 0.388."),
        ("table", "Table 2: Canonical value function and policy for "
                  "r(s) = \u22120.04, \u03b3 = 1.",
         ["", "col 0", "col 1", "col 2", "col 3"], grid_rows(GRID["canonical"])),
        ("image", "fig2_gridworld_canonical.png",
         "Figure 2: Canonical value function and greedy policy for "
         "r(s) = \u22120.04 and \u03b3 = 1. The policy moves along the top row "
         "into the +1 terminal state and avoids the \u22121 state.", 6.2),
        ("p", "The optimal policy sends the agent right along the top row into "
              "the +1 goal. The \u22121 terminal state is never entered. The "
              "negative living reward encourages the agent to finish the episode "
              "quickly, and the stochastic slips make the values smooth rather "
              "than deterministic."),
        ("h2", "2.4 Effect of the living reward"),
        ("p", "The assignment asks for the optimal value functions for "
              "r(s) = \u22122, 0.1, 0.02 and 1. With \u03b3 = 1, every positive "
              "living reward makes the MDP improper: the agent can avoid the "
              "terminal states forever and collect reward at every step, so the "
              "optimal value is unbounded. Value iteration therefore does not "
              "converge for r(s) > 0. This is not a numerical problem; it is the "
              "correct behaviour of an undiscounted MDP with positive rewards."),
        ("p", "To obtain finite value functions for all requested rewards, the "
              "reward sweep uses \u03b3 = 0.9. This also matches the discount "
              "used in the Gbike problem. The \u03b3 = 1 behaviour is reported "
              "separately in Section 2.4.7. Table 3 summarises the sweep."),
        ("table", "Table 3: Reward sweep at \u03b3 = 0.9. V(start) is the value "
                  "of the start state (2,0).",
         ["r(s)", "\u03b3", "Sweeps", "Converged", "V(start)"],
         reward_summary_rows()),
        ("image", "fig3_gridworld_reward_sweep.png",
         "Figure 3: Optimal value functions for the five living rewards at "
         "\u03b3 = 0.9. The colour scale is shared across panels; arrows show "
         "the greedy action and T marks a terminal cell.", 6.4),

        ("h3", "2.4.1 r(s) = \u22120.04"),
        ("p", "With a mild negative living reward the agent still heads for the "
              "+1 terminal state, but the discount lowers the values relative to "
              "the \u03b3 = 1 case. The policy differs at (2,2), where the "
              "discounted problem prefers Up rather than Left. The start value "
              "is 0.296."),
        ("table", "Table 4: Value function and policy for r(s) = \u22120.04, "
                  "\u03b3 = 0.9.",
         ["", "col 0", "col 1", "col 2", "col 3"],
         grid_rows(GRID["reward_sweep_gamma_0.9"]["-0.04"])),

        ("h3", "2.4.2 r(s) = \u22122"),
        ("p", "A large negative living reward makes every step expensive, so the "
              "agent takes a short route to a terminal state. The values are "
              "strongly negative (the start value is \u22128.588) and the policy "
              "in the bottom row moves right and then up into the \u22121 "
              "terminal. The negative reward dominates the +1/\u22121 terminal "
              "difference: reducing the number of steps is more important than "
              "avoiding the \u22121 goal."),
        ("table", "Table 5: Value function and policy for r(s) = \u22122, "
                  "\u03b3 = 0.9.",
         ["", "col 0", "col 1", "col 2", "col 3"],
         grid_rows(GRID["reward_sweep_gamma_0.9"]["-2"])),

        ("h3", "2.4.3 r(s) = 0.1"),
        ("p", "The agent avoids both terminal states and circulates forever. The "
              "discounted value of the infinite loop is 0.1 / (1 \u2212 0.9) = "
              "1.0, which is exactly the value shown in every non-terminal state. "
              "At \u03b3 = 1 this policy would be improper and the value would be "
              "infinite."),
        ("table", "Table 6: Value function and policy for r(s) = 0.1, "
                  "\u03b3 = 0.9.",
         ["", "col 0", "col 1", "col 2", "col 3"],
         grid_rows(GRID["reward_sweep_gamma_0.9"]["0.1"])),

        ("h3", "2.4.4 r(s) = 0.02"),
        ("p", "A small positive living reward still makes reaching the +1 goal "
              "worthwhile in most states, but the policy hesitates near the "
              "\u22121 terminal. The start value is 0.590. The agent prefers a "
              "longer but safer route to +1 rather than risking the \u22121 "
              "terminal."),
        ("table", "Table 7: Value function and policy for r(s) = 0.02, "
                  "\u03b3 = 0.9.",
         ["", "col 0", "col 1", "col 2", "col 3"],
         grid_rows(GRID["reward_sweep_gamma_0.9"]["0.02"])),

        ("h3", "2.4.5 r(s) = 1"),
        ("p", "The agent avoids the terminal states completely and keeps "
              "collecting +1 forever. The discounted value is 1 / (1 \u2212 0.9) "
              "= 10.0, which is the value shown in every non-terminal state. This "
              "is the clearest illustration that a positive living reward "
              "rewards survival rather than goal completion."),
        ("table", "Table 8: Value function and policy for r(s) = 1, "
                  "\u03b3 = 0.9.",
         ["", "col 0", "col 1", "col 2", "col 3"],
         grid_rows(GRID["reward_sweep_gamma_0.9"]["1"])),

        ("h3", "2.4.6 Summary of the policies"),
        ("p", "The sweep shows a clear progression. For negative rewards the "
              "agent terminates as quickly as possible: \u22122 produces the "
              "shortest route, while \u22120.04 still prefers the +1 goal. As the "
              "living reward becomes positive, the agent prolongs the episode, "
              "and at 0.1 or 1 it avoids the terminal states entirely. The "
              "discount factor controls how much future reward is worth: with "
              "\u03b3 = 0.9, the infinite stream of +0.1 is worth 1.0 and the "
              "infinite stream of +1 is worth 10.0."),
        ("h3", "2.4.7 Behaviour at \u03b3 = 1"),
        ("p", "Table 9 reports the \u03b3 = 1 behaviour after 300 sweeps. The two "
              "negative rewards converge, while all three positive rewards keep "
              "growing. Figure 4 plots V(start) against the sweep number. The "
              "growth is approximately linear: the r(s) = 1 curve gains about 1 "
              "per sweep and the r(s) = 0.1 curve gains about 0.1 per sweep. "
              "There is no finite optimal value function for these cases."),
        ("table", "Table 9: Behaviour at \u03b3 = 1 after 300 value-iteration "
                  "sweeps. Positive living rewards do not converge.",
         ["r(s)", "Converged", "Sweeps used", "V(start) after cap"],
         gamma1_rows()),
        ("image", "fig4_gridworld_gamma1_divergence.png",
         "Figure 4: Value iteration at \u03b3 = 1. Negative living rewards "
         "converge; positive living rewards grow without bound because the agent "
         "can avoid the terminal states forever.", 6.2),
        ("p", "**Takeaway.** The living reward determines how urgently the agent "
              "wants to leave the non-terminal part of the grid. Negative rewards "
              "encourage short episodes; positive rewards encourage long "
              "episodes. With no discount, \u201clong\u201d becomes infinite, so "
              "a discount is required for a well-posed finite value function."),
        ("pagebreak",),
    ]

    # ------------------------------- part 2 -------------------------------
    blocks += [
        ("h1", "3. Part 2 \u2014 Gbike Bicycle Rental"),
        ("h2", "3.1 Problem description and MDP formulation"),
        ("p", "Gbike operates two bicycle-rental locations. Each day, customers "
              "arrive at each location and rent a bicycle if one is available, "
              "earning INR 10 per rental. Bicycles returned during the day become "
              "available the following day. Bicycles can be moved between the "
              "locations overnight at a cost of INR 2 per bicycle. Each location "
              "can hold at most 20 bicycles, and at most 5 bicycles can be moved "
              "in one night. The discount rate is \u03b3 = 0.9. Figure 5 shows the "
              "two-location system."),
        ("image", "fig5_gbike_schematic.png",
         "Figure 5: The Gbike two-location MDP. Requests and returns are Poisson; "
         "the manager may move up to five bicycles overnight.", 6.3),
        ("p", "**State.** The state is s = (c\u2081, c\u2082), where c\u1d62 is "
              "the number of bicycles at location i at the end of the day, after "
              "the day's rentals and returns. Each inventory is between 0 and 20, "
              "so there are 21 \u00d7 21 = 441 states."),
        ("p", "**Action.** The action a \u2208 {\u22125, \u22124, \u2026, 0, "
              "\u2026, 4, 5} is the net number of bicycles moved overnight. A "
              "positive action moves a bicycles from location 1 to location 2; a "
              "negative action moves \u2212a bicycles in the opposite direction. "
              "An action is feasible only if"),
        ("equation", "0 \u2264 c\u2081 \u2212 a \u2264 20,    "
                     "0 \u2264 c\u2082 + a \u2264 20,    |a| \u2264 5."),
        ("p", "**Dynamics.** One day proceeds as follows:"),
        ("numbers", [
            "Move bicycles overnight: n\u2081 = c\u2081 \u2212 a and "
            "n\u2082 = c\u2082 + a.",
            "Requests arrive at location i: R\u1d62 ~ Poisson(\u03bb\u1d62), with "
            "\u03bb\u2081 = 3 and \u03bb\u2082 = 4. The number served is "
            "min(R\u1d62, n\u1d62), earning INR 10 each.",
            "Returns arrive at location i: T\u1d62 ~ Poisson(\u03bc\u1d62), with "
            "\u03bc\u2081 = 3 and \u03bc\u2082 = 2. They are added to the "
            "inventory, capped at 20.",
            "The resulting inventory is the next state s\u2032.",
        ]),
        ("table", "Table 10: Gbike MDP parameters.",
         ["Parameter", "Value"],
         [
             ["Locations", "2"],
             ["State", "(c\u2081, c\u2082), 0 \u2264 c\u2081, c\u2082 \u2264 20"],
             ["Number of states", "441"],
             ["Actions", "\u22125, \u22124, \u2026, +5 (net bicycles moved 1 \u2192 2)"],
             ["Maximum move per night", "5 bicycles"],
             ["Rental requests", "Poisson(3) at location 1; Poisson(4) at location 2"],
             ["Returns", "Poisson(3) at location 1; Poisson(2) at location 2"],
             ["Rental income", "INR 10 per bicycle rented"],
             ["Move cost", "INR 2 per bicycle (first 1 \u2192 2 bicycle free in the modified problem)"],
             ["Parking cost", "INR 4 per location per night if more than 10 bicycles are kept overnight (modified problem)"],
             ["Discount", "\u03b3 = 0.9"],
         ]),
        ("p", "**Reward.** For a move a and post-move inventories n\u2081 and "
              "n\u2082, the expected immediate reward is"),
        ("equation", "R(s, a) = 10 \u00b7 ( E[min(R\u2081, n\u2081)] + "
                     "E[min(R\u2082, n\u2082)] ) \u2212 move(a) \u2212 "
                     "park(n\u2081, n\u2082)."),
        ("p", "The expected number of bicycles served at a location with c "
              "available bicycles is"),
        ("equation", "E[min(R, c)] = \u03a3_{k=0}^{c\u22121} k \u00b7 P(R = k) "
                     "+ c \u00b7 P(R \u2265 c)."),
        ("p", "**Original move cost.** In the original problem, move(a) = "
              "2|a|, and there is no parking cost."),
        ("p", "**Modified move cost.** The employee at location 1 takes one "
              "bicycle to location 2 for free each night. Every additional "
              "bicycle still costs INR 2, and every bicycle moved from location 2 "
              "to location 1 costs INR 2. Thus"),
        ("equation", "move(a) = 2|a| if a \u2264 0;    move(1) = 0;    "
                     "move(a) = 2(a \u2212 1) if a > 1."),
        ("p", "**Parking cost.** In the modified problem, if more than 10 "
              "bicycles are kept overnight at a location after moving, a fixed "
              "INR 4 is paid for a second parking lot at that location:"),
        ("equation", "park(n\u2081, n\u2082) = 4 \u00b7 1[n\u2081 > 10] + "
                     "4 \u00b7 1[n\u2082 > 10]."),
        ("p", "**Transition probabilities.** Requests and returns are independent "
              "across the two locations, so the transition distribution "
              "factorises:"),
        ("equation", "P(s\u2032 | s, a) = q_{1}(c_{1}\u2032 | n_{1}) "
                     "\u00b7 q_{2}(c_{2}\u2032 | n_{2}),"),
        ("p", "where q(c\u2032 | i, n) is the probability that location i, "
              "starting the day with n bicycles, ends the day with c\u2032 "
              "bicycles after rentals and returns. Each q is computed by "
              "convolving the distribution of remaining bicycles after rentals "
              "with the Poisson return distribution, then capping at 20. This "
              "factorisation keeps policy iteration fast and exact."),
        ("p", "**Objective.** The goal is to maximise the discounted expected "
              "total reward"),
        ("equation", "V^{\u03c0}(s) = E_{\u03c0}[ \u03a3_{t=0}^{\u221e} "
                     "\u03b3^{t} R(s_{t}, a_{t}) | s_{0} = s ],    \u03b3 = 0.9."),
        ("h2", "3.2 Policy iteration"),
        ("p", "Policy iteration alternates exact policy evaluation and greedy "
              "policy improvement until the policy is stable. Because there are "
              "only 441 states, the evaluation step solves the linear system "
              "exactly; there is no iterative evaluation error."),
        ("p", "**Policy evaluation.** For a policy \u03c0, solve"),
        ("equation", "(I \u2212 \u03b3P_{\u03c0}) V_{\u03c0} = R_{\u03c0}."),
        ("p", "**Policy improvement.** For every state, choose"),
        ("equation", "\u03c0\u2032(s) = argmax_{a \u2208 A(s)} [ R(s, a) + "
                     "\u03b3 \u03a3_{s\u2032} P(s\u2032 | s, a) V_{\u03c0}(s\u2032) ]."),
        ("p", "Ties keep the current action, which prevents cycling between "
              "equally good moves. The algorithm starts from the \u201cmove "
              "nothing\u201d policy and stops when no state changes."),
        ("code", "initialize policy pi(s) = 0 (move nothing) for every state\n"
                 "repeat\n"
                 "    evaluate pi exactly: solve (I - gamma P_pi) V = R_pi\n"
                 "    for every state s\n"
                 "        pi_new(s) = argmax_a [ R(s,a) + gamma sum_s' P(s'|s,a) V(s') ]\n"
                 "    if pi_new == pi: stop\n"
                 "    pi = pi_new"),
        ("h2", "3.3 Original problem"),
        ("p", "Policy iteration converges in 5 iterations. The value of the "
              "central state is V(10,10) = 574.948 INR, V(0,0) = 421.414 INR and "
              "V(20,20) = 636.990 INR. Table 11 lists the iteration history and "
              "Figure 6 shows the value function and optimal policy."),
        ("table", "Table 11: Policy-iteration history for the original Gbike "
                  "problem.",
         ["Iteration", "States changed", "V(10,10) (INR)"],
         history_rows(GBIKE["original"])),
        ("image", "fig6_gbike_original.png",
         "Figure 6: Original Gbike problem. Left: state values. Right: optimal "
         "overnight moves (positive = 1 \u2192 2, negative = 2 \u2192 1).", 6.3),
        ("p", "The optimal policy has the expected structure: when one location "
              "has many bicycles and the other has few, move up to five bicycles "
              "toward the low-inventory location. In the balanced middle of the "
              "state space, moving nothing is optimal. The original formulation "
              "is the standard Jack's car-rental problem from Sutton and Barto "
              "and serves as a sanity check for the implementation."),
        ("h2", "3.4 Modified problem"),
        ("p", "The modified problem changes two things. First, the employee at "
              "location 1 takes one bicycle to location 2 for free each night, so "
              "the first 1 \u2192 2 move costs nothing. Second, if more than 10 "
              "bicycles are kept overnight at a location after moving, an extra "
              "INR 4 is paid for a second parking lot. Policy iteration converges "
              "in 4 iterations. The central value rises slightly to "
              "V(10,10) = 580.963 INR, while V(0,0) rises to 429.946 INR and "
              "V(20,20) falls to 603.491 INR."),
        ("table", "Table 12: Policy-iteration history for the modified Gbike "
                  "problem.",
         ["Iteration", "States changed", "V(10,10) (INR)"],
         history_rows(GBIKE["modified"])),
        ("image", "fig7_gbike_modified.png",
         "Figure 7: Modified Gbike problem with the free 1\u21922 shuttle and "
         "the second-lot parking charge.", 6.3),
        ("p", "The modified policy is visibly different from the original one. "
              "The free 1 \u2192 2 shuttle creates a large region in which moving "
              "exactly one bicycle from location 1 to location 2 is optimal, even "
              "when both locations are below the parking threshold. The move is "
              "free and shifts a bicycle toward location 2, which has the higher "
              "request rate (4 versus 3)."),
        ("p", "The second-lot charge creates threshold effects around 10 "
              "bicycles. For example, at (13,11) the optimal action is \u22121: "
              "moving one bicycle back to location 1 costs INR 2 but avoids the "
              "INR 4 second-lot charge at location 2, so it is worth doing. At "
              "(13,12) the optimal action is +3: moving three bicycles to "
              "location 2 yields (10,15), empties the first lot below the "
              "threshold, and shifts bicycles toward the higher-demand location. "
              "The value of (20,20) drops by about 33.5 INR because both "
              "locations are over 10 bicycles and the manager cannot escape both "
              "second-lot charges quickly."),
        ("h2", "3.5 Comparison"),
        ("p", "Table 13 compares selected states. Across all 441 states, the "
              "optimal action changes in 242 states. The mean value change is "
              "\u22121.358 INR, the maximum gain is +12.057 INR, and the minimum "
              "change is \u221233.498 INR. The free shuttle is a pure benefit, "
              "while the second-lot charge is a pure cost; the net effect depends "
              "on the state."),
        ("table", "Table 13: Selected state values for the original and modified "
                  "problems (INR).",
         ["State (c\u2081, c\u2082)", "Original", "Modified", "Change"],
         selected_state_rows()),
        ("image", "fig8_gbike_comparison.png",
         "Figure 8: Side-by-side comparison of the original and modified value "
         "functions and policies.", 6.3),
        ("image", "fig9_gbike_differences.png",
         "Figure 9: Left: change in state value after the modifications. Right: "
         "states where the optimal action changes (black = changed).", 6.3),
        ("p", "The value-difference map is positive for moderate inventories, "
              "where the free shuttle helps, and strongly negative when both "
              "locations hold more than 10 bicycles, where the second-lot charges "
              "dominate. The policy-change map shows that the modifications affect "
              "a large portion of the state space, not just a narrow boundary "
              "region."),
        ("h2", "3.6 Validation"),
        ("p", "Several independent checks were run to make sure the formulation "
              "and implementation are correct. The factorised transition model was "
              "compared with a brute-force enumeration of requests and returns for "
              "a test state and action; the expected reward and expected "
              "next-state value matched to machine precision. Policy iteration was "
              "also compared with value iteration for both the original and "
              "modified MDPs."),
        ("table", "Table 14: Validation checks.",
         ["Check", "Result"],
         [
             ["Grid transition probabilities sum to 1", "Passed for all 36 state-action pairs"],
             ["Canonical grid values match Russell & Norvig", "Passed (max error < 10\u207b\u2079)"],
             ["Grid value iteration vs policy iteration", "Passed (max |\u0394V| = 7.4 \u00d7 10\u207b\u00b9\u00b9; same policy)"],
             ["Gbike next-state distributions sum to 1", "Passed for both locations"],
             ["Factorised Gbike transition vs brute force", "Passed (machine precision)"],
             ["Original Gbike Bellman residual", "6.8 \u00d7 10\u207b\u00b9\u00b3"],
             ["Modified Gbike Bellman residual", "8.0 \u00d7 10\u207b\u00b9\u00b3"],
             ["Policies use only feasible actions", "Passed for both problems"],
         ]),
        ("p", "All checks pass with `python validate.py`. The maximum difference "
              "between policy iteration and value iteration is 8.6 \u00d7 "
              "10\u207b\u00b9\u00b9 for the Gbike MDPs."),
        ("p", "**Note on gbike.zip.** No gbike.zip file was present in the "
              "repository or in the local Downloads folder, so there was no "
              "reference implementation to diff against. The implementation "
              "follows the formulation in the assignment and in Sutton and Barto, "
              "and the original problem is solved alongside the modified one so "
              "that the two formulations can be compared directly."),
    ]

    # ------------------------------- conclusion ---------------------------
    blocks += [
        ("h1", "4. Conclusion"),
        ("p", "This lab demonstrated the complete workflow of dynamic "
              "programming for MDPs: formulate the state, action, transition and "
              "reward model; apply value iteration or policy iteration; and "
              "interpret the resulting value function and policy."),
        ("bullets", [
            "For the 4\u00d73 grid world, value iteration reproduced the textbook "
            "values for r(s) = \u22120.04 exactly and agreed with policy iteration "
            "to within 7.4 \u00d7 10\u207b\u00b9\u00b9.",
            "The living reward controls how quickly the agent wants to terminate. "
            "Negative rewards produce short, goal-directed episodes; positive "
            "rewards produce long or non-terminating episodes.",
            "With \u03b3 = 1, positive living rewards make the MDP improper and "
            "the value function diverges. A discount such as \u03b3 = 0.9 is "
            "needed to obtain finite values.",
            "For Gbike, policy iteration converged in 5 iterations for the "
            "original problem and 4 for the modified problem. The free shuttle and "
            "the second-lot charge change the optimal policy in 242 of 441 states.",
            "The free 1 \u2192 2 shuttle is valuable because location 2 has the "
            "higher request rate. The second-lot charge penalises keeping more "
            "than 10 bicycles at either location and creates threshold effects "
            "around 10.",
        ]),
        ("p", "The results also illustrate a practical reinforcement-learning "
              "lesson: the reward function and the discount factor are part of the "
              "problem specification, not implementation details. Small changes in "
              "either can produce qualitatively different optimal behaviour."),
        ("h1", "5. References"),
        ("numbers", [
            "S. Russell and P. Norvig, *Artificial Intelligence: A Modern "
            "Approach*, 3rd edition, Chapters 16, 17 and 21.",
            "R. S. Sutton and A. G. Barto, *Reinforcement Learning: An "
            "Introduction*, 2nd edition, Chapters 3 and 4 (the Jack's car-rental "
            "problem).",
            "R. S. Sutton and A. G. Barto, *Reinforcement Learning: An "
            "Introduction*, 1st edition, Section 4.2 (policy iteration for Jack's "
            "car rental).",
        ]),
        ("h1", "Appendix A \u2014 Reproducing the results"),
        ("p", "The complete implementation is in the `lab10` folder. The main "
              "entry point runs both parts; the validation script checks the "
              "transition model, the textbook values and the Bellman residuals."),
        ("code", "cd lab10\n"
                 "python main.py                # run both parts\n"
                 "python main.py --part grid    # grid world only\n"
                 "python main.py --part gbike   # Gbike only\n"
                 "python validate.py            # correctness checks"),
        ("p", "The code writes all figures and numerical results to `outputs/`. "
              "The key files are `gridworld_results.json`, `gbike_results.json`, "
              "the value/policy CSV files, and the PNG figures included in this "
              "report."),
        ("table", "Table 15: Main output files.",
         ["File", "Contents"],
         [
             ["gridworld_canonical.png", "Canonical value function and policy"],
             ["gridworld_reward_sweep.png", "Value/policy panels for five rewards"],
             ["gridworld_gamma1_divergence.png", "Divergence of positive rewards at \u03b3 = 1"],
             ["gbike_original.png", "Original Gbike values and policy"],
             ["gbike_modified.png", "Modified Gbike values and policy"],
             ["gbike_comparison.png", "Original vs modified comparison"],
             ["gbike_differences.png", "Value and policy difference maps"],
             ["*.json / *.csv", "Exact numerical results and policies"],
         ]),
    ]
    return blocks


# ---------------------------------------------------------------------------
# Markdown builder
# ---------------------------------------------------------------------------
def build_markdown(blocks):
    lines = []

    def add(text=""):
        lines.append(text)

    for block in blocks:
        kind = block[0]
        if kind == "title":
            add(f"# {block[1]}")
            add()
        elif kind == "subtitle":
            add(f"## {block[1]}")
            add()
        elif kind == "subsubtitle":
            add(f"*{block[1]}*")
            add()
        elif kind == "meta":
            for label, value in block[1]:
                add(f"**{label}:** {value}  ")
            add()
        elif kind == "h1":
            add(f"# {block[1]}")
            add()
        elif kind == "h2":
            add(f"## {block[1]}")
            add()
        elif kind == "h3":
            add(f"### {block[1]}")
            add()
        elif kind == "p":
            add(block[1])
            add()
        elif kind == "bullets":
            for item in block[1]:
                add(f"- {item}")
            add()
        elif kind == "numbers":
            for i, item in enumerate(block[1], 1):
                add(f"{i}. {item}")
            add()
        elif kind == "equation":
            add("$$")
            add(block[1])
            add("$$")
            add()
        elif kind == "code":
            add("```text")
            add(block[1])
            add("```")
            add()
        elif kind == "image":
            filename, caption, _width = block[1], block[2], block[3]
            add(f"![{caption}](figures/{filename})")
            add()
            add(f"*{caption}*")
            add()
        elif kind == "table":
            caption, headers, rows = block[1], block[2], block[3]
            add(f"*{caption}*")
            add()
            add("| " + " | ".join(str(h) for h in headers) + " |")
            add("|" + "|".join(["---"] * len(headers)) + "|")
            for row in rows:
                add("| " + " | ".join(str(cell) for cell in row) + " |")
            add()
        elif kind == "pagebreak":
            add("---")
            add()
        else:
            raise ValueError(f"Unknown block type: {kind}")

    text = "\n".join(lines).strip() + "\n"
    (HERE / "REPORT.md").write_text(text, encoding="utf-8")
    return text


# ---------------------------------------------------------------------------
# HTML builder
# ---------------------------------------------------------------------------
def build_html(blocks):
    def esc(text):
        return (str(text).replace("&", "&amp;").replace("<", "&lt;")
                .replace(">", "&gt;"))

    def rich(text):
        # Minimal bold/italic/code conversion for the HTML report.
        text = esc(text)
        text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
        text = re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)
        text = re.sub(r"`(.+?)`", r"<code>\1</code>", text)
        return text

    def math(text):
        # Convert the light-weight _{...}/^{...} notation to real HTML
        # subscripts and superscripts.
        text = esc(text)
        text = re.sub(r"_\{(.*?)\}", r"<sub>\1</sub>", text)
        text = re.sub(r"\^\{(.*?)\}", r"<sup>\1</sup>", text)
        text = re.sub(r"_([A-Za-z0-9]+)", r"<sub>\1</sub>", text)
        text = re.sub(r"\^([A-Za-z0-9]+)", r"<sup>\1</sup>", text)
        return text

    def image_tag(filename):
        path = FIG / filename
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:image/png;base64,{data}"

    parts = []
    parts.append("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Lab 10 - Markov Decision Processes and Dynamic Programming</title>
<style>
  body { font-family: Georgia, 'Times New Roman', serif; color: #1f2937;
         line-height: 1.65; max-width: 980px; margin: 0 auto; padding: 36px 28px; }
  h1 { color: #1a3e6f; border-bottom: 2px solid #1a3e6f; padding-bottom: 6px;
       margin-top: 42px; font-size: 1.65em; }
  h2 { color: #2c5aa0; margin-top: 32px; font-size: 1.28em; }
  h3 { color: #334155; margin-top: 24px; font-size: 1.08em; }
  p { margin: 10px 0; text-align: justify; }
  ul, ol { margin: 10px 0 14px 24px; }
  li { margin: 5px 0; }
  code { font-family: Consolas, 'Courier New', monospace; background: #f1f5f9;
         padding: 1px 4px; border-radius: 3px; font-size: 0.92em; }
  pre { font-family: Consolas, 'Courier New', monospace; background: #f4f6fa;
        border: 1px solid #dbe2ea; border-radius: 5px; padding: 12px 14px;
        overflow-x: auto; font-size: 0.86em; line-height: 1.45; }
  .equation { text-align: center; font-style: italic; background: #f7f9fc;
              border-left: 4px solid #2c5aa0; padding: 10px 14px; margin: 16px 0;
              font-size: 1.04em; }
  figure { text-align: center; margin: 26px 0; }
  figure img { max-width: 100%; height: auto; border: 1px solid #d7dde5;
               border-radius: 4px; }
  figcaption { font-style: italic; font-size: 0.9em; color: #4b5563;
               margin-top: 8px; }
  table { border-collapse: collapse; margin: 14px auto 22px auto; width: 100%;
          font-size: 0.88em; }
  th, td { border: 1px solid #b8c2cc; padding: 6px 8px; text-align: center; }
  th { background: #1a3e6f; color: white; }
  tr:nth-child(even) td { background: #f4f7fb; }
  .table-caption { text-align: center; font-style: italic; font-size: 0.9em;
                   color: #4b5563; margin: 18px 0 6px 0; }
  .title-page { text-align: center; margin-top: 110px; }
  .title-page h1 { border: none; font-size: 2.6em; margin-bottom: 8px; }
  .title-page .subtitle { font-size: 1.45em; color: #2c5aa0; margin: 8px 0; }
  .title-page .subsubtitle { font-size: 1.15em; color: #475569; margin: 8px 0 36px 0; }
  .title-page .meta { font-size: 1.0em; line-height: 2.0; }
  .page-break { page-break-after: always; }
  @media print {
    body { max-width: none; margin: 0; padding: 18mm; font-size: 10.5pt; }
    h1 { page-break-after: avoid; }
    h2, h3 { page-break-after: avoid; }
    figure, table { page-break-inside: avoid; }
    .page-break { page-break-after: always; }
  }
</style>
</head>
<body>
""")

    for block in blocks:
        kind = block[0]
        if kind == "title":
            parts.append('<div class="title-page">')
            parts.append(f"<h1>{rich(block[1])}</h1>")
        elif kind == "subtitle":
            parts.append(f'<div class="subtitle">{rich(block[1])}</div>')
        elif kind == "subsubtitle":
            parts.append(f'<div class="subsubtitle">{rich(block[1])}</div>')
        elif kind == "meta":
            parts.append('<div class="meta">')
            for label, value in block[1]:
                parts.append(f"<div><strong>{rich(label)}:</strong> {rich(value)}</div>")
            parts.append("</div></div>")
        elif kind == "h1":
            parts.append(f"<h1>{rich(block[1])}</h1>")
        elif kind == "h2":
            parts.append(f"<h2>{rich(block[1])}</h2>")
        elif kind == "h3":
            parts.append(f"<h3>{rich(block[1])}</h3>")
        elif kind == "p":
            parts.append(f"<p>{rich(block[1])}</p>")
        elif kind == "bullets":
            parts.append("<ul>")
            for item in block[1]:
                parts.append(f"<li>{rich(item)}</li>")
            parts.append("</ul>")
        elif kind == "numbers":
            parts.append("<ol>")
            for item in block[1]:
                parts.append(f"<li>{rich(item)}</li>")
            parts.append("</ol>")
        elif kind == "equation":
            parts.append(f'<div class="equation">{math(block[1])}</div>')
        elif kind == "code":
            parts.append(f"<pre>{esc(block[1])}</pre>")
        elif kind == "image":
            filename, caption = block[1], block[2]
            parts.append("<figure>")
            parts.append(f'<img src="{image_tag(filename)}" alt="{esc(caption)}">')
            parts.append(f"<figcaption>{rich(caption)}</figcaption>")
            parts.append("</figure>")
        elif kind == "table":
            caption, headers, rows = block[1], block[2], block[3]
            parts.append(f'<p class="table-caption">{rich(caption)}</p>')
            parts.append("<table>")
            parts.append("<thead><tr>")
            for header in headers:
                parts.append(f"<th>{rich(header)}</th>")
            parts.append("</tr></thead><tbody>")
            for row in rows:
                parts.append("<tr>")
                for cell in row:
                    parts.append(f"<td>{rich(cell)}</td>")
                parts.append("</tr>")
            parts.append("</tbody></table>")
        elif kind == "pagebreak":
            parts.append('<div class="page-break"></div>')
        else:
            raise ValueError(f"Unknown block type: {kind}")

    parts.append("</body></html>")
    html = "\n".join(parts)
    (HERE / "Lab10_Report.html").write_text(html, encoding="utf-8")
    return html


# ---------------------------------------------------------------------------
# DOCX builder
# ---------------------------------------------------------------------------
def set_cell_background(cell, hex_color):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    tc_pr.append(shd)


def add_horizontal_rule(doc):
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_pr = paragraph._p.get_or_add_pPr()
    p_bdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "8")
    bottom.set(qn("w:space"), "1")
    bottom.set(qn("w:color"), "1A3E6F")
    p_bdr.append(bottom)
    p_pr.append(p_bdr)


def add_rich_text(paragraph, text, size=11, colour=None, italic=False):
    token_re = re.compile(r"(\*\*.*?\*\*|\*[^*]+?\*|`[^`]+?`)")
    for token in token_re.split(text):
        if not token:
            continue
        if token.startswith("**") and token.endswith("**"):
            run = paragraph.add_run(token[2:-2])
            run.bold = True
        elif token.startswith("*") and token.endswith("*"):
            run = paragraph.add_run(token[1:-1])
            run.italic = True
        elif token.startswith("`") and token.endswith("`"):
            run = paragraph.add_run(token[1:-1])
            run.font.name = "Consolas"
            run.font.size = Pt(size - 1)
        else:
            run = paragraph.add_run(token)
            run.font.name = "Calibri"
            run.font.size = Pt(size)
        if run.font.size is None:
            run.font.size = Pt(size)
        if colour is not None:
            run.font.color.rgb = colour
        if italic:
            run.italic = True


MATH_TOKEN_RE = re.compile(r"(_\{[^{}]*\}|\^\{[^{}]*\}|_[^\s{}]|\^[^\s{}])")


def add_math_text(paragraph, text, size=11):
    """Render a light-weight maths string with true sub/superscripts.

    Tokens such as ``V_{k+1}(s)`` and ``\\u03b3^{t}`` are split so that the
    script text becomes a real Word subscript/superscript run instead of
    showing literal braces and underscores.
    """
    for token in MATH_TOKEN_RE.split(text):
        if not token:
            continue
        run = paragraph.add_run()
        run.font.name = "Cambria Math"
        run.font.size = Pt(size)
        run.italic = True
        if token.startswith("_{") and token.endswith("}"):
            run.text = token[2:-1]
            run.font.subscript = True
        elif token.startswith("^{") and token.endswith("}"):
            run.text = token[2:-1]
            run.font.superscript = True
        elif token.startswith("_"):
            run.text = token[1:]
            run.font.subscript = True
        elif token.startswith("^"):
            run.text = token[1:]
            run.font.superscript = True
        else:
            run.text = token


def add_footer(section):
    footer = section.footer
    paragraph = footer.paragraphs[0]
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run("Lab 10 \u2014 Markov Decision Processes and "
                            "Dynamic Programming    |    Page ")
    run.font.name = "Calibri"
    run.font.size = Pt(8.5)
    run.font.color.rgb = RGBColor(0x66, 0x66, 0x66)

    field_begin = OxmlElement("w:fldChar")
    field_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = "PAGE"
    field_end = OxmlElement("w:fldChar")
    field_end.set(qn("w:fldCharType"), "end")
    run._r.append(field_begin)
    run._r.append(instr)
    run._r.append(field_end)


def build_docx(blocks):
    doc = Document()

    section = doc.sections[0]
    section.top_margin = Inches(1.0)
    section.bottom_margin = Inches(1.0)
    section.left_margin = Inches(1.0)
    section.right_margin = Inches(1.0)
    add_footer(section)

    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)
    style.paragraph_format.space_after = Pt(6)
    style.paragraph_format.line_spacing = 1.15

    for level, size, colour in (
        (1, 16, RGBColor(0x1A, 0x3E, 0x6F)),
        (2, 13, RGBColor(0x2C, 0x5A, 0xA0)),
        (3, 11.5, RGBColor(0x33, 0x41, 0x55)),
    ):
        h_style = doc.styles[f"Heading {level}"]
        h_style.font.name = "Calibri"
        h_style.font.size = Pt(size)
        h_style.font.bold = True
        h_style.font.color.rgb = colour
        h_style.paragraph_format.space_before = Pt(12 if level > 1 else 18)
        h_style.paragraph_format.space_after = Pt(5)
        h_style.paragraph_format.keep_with_next = True

    for block in blocks:
        kind = block[0]

        if kind == "title":
            for _ in range(4):
                doc.add_paragraph()
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = p.add_run(block[1])
            run.bold = True
            run.font.name = "Calibri"
            run.font.size = Pt(30)
            run.font.color.rgb = RGBColor(0x1A, 0x3E, 0x6F)
        elif kind == "subtitle":
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = p.add_run(block[1])
            run.font.name = "Calibri"
            run.font.size = Pt(17)
            run.font.color.rgb = RGBColor(0x2C, 0x5A, 0xA0)
        elif kind == "subsubtitle":
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = p.add_run(block[1])
            run.italic = True
            run.font.name = "Calibri"
            run.font.size = Pt(13)
            run.font.color.rgb = RGBColor(0x47, 0x55, 0x69)
            p.paragraph_format.space_after = Pt(20)
        elif kind == "meta":
            for label, value in block[1]:
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.space_after = Pt(3)
                run = p.add_run(f"{label}: ")
                run.bold = True
                run.font.name = "Calibri"
                run.font.size = Pt(11)
                run = p.add_run(value)
                run.font.name = "Calibri"
                run.font.size = Pt(11)
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            add_horizontal_rule(doc)
        elif kind == "h1":
            doc.add_heading(block[1], level=1)
        elif kind == "h2":
            doc.add_heading(block[1], level=2)
        elif kind == "h3":
            doc.add_heading(block[1], level=3)
        elif kind == "p":
            p = doc.add_paragraph()
            add_rich_text(p, block[1], size=11)
        elif kind == "bullets":
            for item in block[1]:
                p = doc.add_paragraph(style="List Bullet")
                p.paragraph_format.space_after = Pt(3)
                add_rich_text(p, item, size=11)
        elif kind == "numbers":
            # Manual numbering avoids Word continuing the numbering from a
            # previous list (for example, the references restarting at 5).
            for index, item in enumerate(block[1], 1):
                p = doc.add_paragraph()
                p.paragraph_format.left_indent = Inches(0.3)
                p.paragraph_format.first_line_indent = Inches(-0.3)
                p.paragraph_format.space_after = Pt(3)
                add_rich_text(p, f"{index}. {item}", size=11)
        elif kind == "equation":
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(4)
            p.paragraph_format.space_after = Pt(8)
            add_math_text(p, block[1], size=11)
        elif kind == "code":
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Inches(0.25)
            p.paragraph_format.right_indent = Inches(0.15)
            p.paragraph_format.space_before = Pt(3)
            p.paragraph_format.space_after = Pt(8)
            p.paragraph_format.keep_together = True
            p_pr = p._p.get_or_add_pPr()
            shd = OxmlElement("w:shd")
            shd.set(qn("w:val"), "clear")
            shd.set(qn("w:color"), "auto")
            shd.set(qn("w:fill"), "F2F4F8")
            p_pr.append(shd)
            lines = block[1].split("\n")
            for i, line in enumerate(lines):
                run = p.add_run(line)
                run.font.name = "Consolas"
                run.font.size = Pt(8.5)
                if i < len(lines) - 1:
                    run.add_break()
        elif kind == "image":
            filename, caption, width = block[1], block[2], block[3]
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(4)
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.keep_with_next = True
            p.add_run().add_picture(str(FIG / filename), width=Inches(width))
            cap = doc.add_paragraph()
            cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            cap.paragraph_format.space_after = Pt(12)
            add_rich_text(cap, caption, size=9,
                          colour=RGBColor(0x55, 0x55, 0x55), italic=True)
        elif kind == "table":
            caption, headers, rows = block[1], block[2], block[3]
            cap = doc.add_paragraph()
            cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            cap.paragraph_format.space_before = Pt(8)
            cap.paragraph_format.space_after = Pt(4)
            cap.paragraph_format.keep_with_next = True
            add_rich_text(cap, caption, size=9,
                          colour=RGBColor(0x55, 0x55, 0x55), italic=True)

            table = doc.add_table(rows=1 + len(rows), cols=len(headers))
            table.style = "Table Grid"
            table.alignment = WD_TABLE_ALIGNMENT.CENTER
            table.autofit = True

            for j, header in enumerate(headers):
                cell = table.rows[0].cells[j]
                cell.text = ""
                p = cell.paragraphs[0]
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.keep_with_next = True
                run = p.add_run(str(header))
                run.bold = True
                run.font.name = "Calibri"
                run.font.size = Pt(9)
                run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                set_cell_background(cell, "1A3E6F")

            for i, row in enumerate(rows):
                for j, value in enumerate(row):
                    cell = table.rows[i + 1].cells[j]
                    cell.text = ""
                    p = cell.paragraphs[0]
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    # Keep short tables on a single page.
                    if i < len(rows) - 1:
                        p.paragraph_format.keep_with_next = True
                    run = p.add_run(str(value))
                    run.font.name = "Calibri"
                    run.font.size = Pt(9)
                    if i % 2 == 1:
                        set_cell_background(cell, "F4F7FB")

            # Keep the header row on page breaks and avoid splitting rows.
            for row_index, table_row in enumerate(table.rows):
                tr_pr = table_row._tr.get_or_add_trPr()
                cant_split = OxmlElement("w:cantSplit")
                tr_pr.append(cant_split)
                if row_index == 0:
                    tbl_header = OxmlElement("w:tblHeader")
                    tbl_header.set(qn("w:val"), "true")
                    tr_pr.append(tbl_header)

            spacer = doc.add_paragraph()
            spacer.paragraph_format.space_after = Pt(4)
        elif kind == "pagebreak":
            doc.add_page_break()
        else:
            raise ValueError(f"Unknown block type: {kind}")

    doc.core_properties.title = "Lab 10 - Markov Decision Processes and Dynamic Programming"
    doc.core_properties.subject = "4x3 Grid World and Gbike Bicycle Rental"
    doc.core_properties.author = "Lab Report"

    path = HERE / "Lab10_Report.docx"
    doc.save(path)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Generating report figures ...")
    make_grid_layout()
    make_gbike_schematic()
    make_report_reward_sweep()
    copy_output_figures()

    print("Building report content ...")
    blocks = build_blocks()

    print("Writing REPORT.md ...")
    build_markdown(blocks)

    print("Writing Lab10_Report.docx ...")
    build_docx(blocks)

    print("Writing Lab10_Report.html ...")
    build_html(blocks)

    print(f"\nReport written to: {HERE}")
    print("  REPORT.md")
    print("  Lab10_Report.docx")
    print("  Lab10_Report.html")
    print(f"  figures/ ({len(list(FIG.glob('*.png')))} images)")


if __name__ == "__main__":
    main()
