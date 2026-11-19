"""
Lab 10 - Markov Decision Processes and Dynamic Programming
===========================================================
Run both parts of the assignment:

  1. 4x3 grid world  -> value iteration (and a policy-iteration check)
  2. Gbike           -> policy iteration (original + modified problem)

Usage
-----
    python main.py                # run everything
    python main.py --part grid    # only the grid world
    python main.py --part gbike   # only Gbike

All figures and JSON/CSV results are written to ./outputs.
"""

import argparse
import os
import sys

from gbike import run_gbike_experiments
from gridworld import run_gridworld_experiments


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--part",
        choices=("all", "grid", "gbike"),
        default="all",
        help="which part of the lab to run (default: all)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="output directory (default: <this folder>/outputs)",
    )
    args = parser.parse_args(argv)

    here = os.path.dirname(os.path.abspath(__file__))
    output_dir = args.output or os.path.join(here, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 72)
    print("LAB 10: MARKOV DECISION PROCESSES AND DYNAMIC PROGRAMMING")
    print("=" * 72)

    grid_results = None
    gbike_results = None

    if args.part in ("all", "grid"):
        grid_results = run_gridworld_experiments(output_dir)

    if args.part in ("all", "gbike"):
        gbike_results = run_gbike_experiments(output_dir)

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)

    if grid_results is not None:
        canonical = grid_results["canonical"]
        start_value = canonical["values"][2][0]
        print(f"Grid world (r = -0.04, gamma = 1): converged in "
              f"{canonical['iterations']} sweeps, V(start) = {start_value:.4f}")
        check = grid_results.get("policy_iteration_check", {})
        if "max_value_difference" in check:
            print(f"  value iteration vs policy iteration: "
                  f"max difference = {check['max_value_difference']:.2e}, "
                  f"policies identical = {check['policies_identical']}")

    if gbike_results is not None:
        original = gbike_results["original"]
        modified = gbike_results["modified"]
        comparison = gbike_results["json"]["comparison"]
        print(f"Gbike original: converged in {original['iterations']} policy "
              f"iterations, V(10,10) = {original['values'][10, 10]:.4f} INR")
        print(f"Gbike modified: converged in {modified['iterations']} policy "
              f"iterations, V(10,10) = {modified['values'][10, 10]:.4f} INR")
        print(f"  optimal policy differs in "
              f"{comparison['changed_policy_states']} / 441 states")

    print(f"\nAll outputs are in: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
