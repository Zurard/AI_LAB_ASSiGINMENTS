import importlib.util
from pathlib import Path


def load_module(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


lab3_dir = Path(__file__).resolve().parent
alpha_beta = load_module("alpha_beta", lab3_dir / "alpha_beta.py")
max_n = load_module("n_max", lab3_dir / "n_max.py")


def run_alpha_beta_case():
    state = {
        "remaining_time": 24,
        "selected_activities": [],
        "turn": "A"
    }

    alpha_beta.minimax_nodes = 0
    alpha_beta.minimax_evaluated = 0
    alpha_beta.alpha_beta_nodes = 0
    alpha_beta.alpha_beta_pruned = 0
    alpha_beta.alpha_beta_evaluated = 0

    minimax_value, minimax_path = alpha_beta.minimax(state, [])
    alpha_beta_value, alpha_beta_path = alpha_beta.alphaBetaPruning(
        state,
        [],
        float("-inf"),
        float("inf")
    )

    return {
        "state": state,
        "minimax": {
            "utility": minimax_value,
            "path": minimax_path,
            "nodes": alpha_beta.minimax_nodes,
            "evaluated": alpha_beta.minimax_evaluated,
        },
        "alpha_beta": {
            "utility": alpha_beta_value,
            "path": alpha_beta_path,
            "nodes": alpha_beta.alpha_beta_nodes,
            "evaluated": alpha_beta.alpha_beta_evaluated,
            "pruned": alpha_beta.alpha_beta_pruned,
        },
    }


def run_max_n_case():
    state = {
        "remaining_time": 16,
        "selected_activities": [],
        "turn": 0,
        "players": 4
    }

    max_n.maxN_nodes = 0
    max_n.maxN_evaluated = 0

    utility, path = max_n.maxN(state, [])

    return {
        "state": state,
        "utility": utility,
        "path": path,
        "nodes": max_n.maxN_nodes,
        "evaluated": max_n.maxN_evaluated,
    }


def compare_methods():
    print("========================================")
    print("Comparing Alpha-Beta and Max-N search")
    print("========================================")

    alpha_beta_result = run_alpha_beta_case()
    print("\n2-player game case for Alpha-Beta:")
    print(alpha_beta_result["state"])
    print("Minimax utility:", alpha_beta_result["minimax"]["utility"])
    print("Minimax nodes:", alpha_beta_result["minimax"]["nodes"])
    print("Minimax terminal states:", alpha_beta_result["minimax"]["evaluated"])
    print("Alpha-Beta utility:", alpha_beta_result["alpha_beta"]["utility"])
    print("Alpha-Beta nodes:", alpha_beta_result["alpha_beta"]["nodes"])
    print("Alpha-Beta terminal states:", alpha_beta_result["alpha_beta"]["evaluated"])
    print("Alpha-Beta pruning events:", alpha_beta_result["alpha_beta"]["pruned"])

    max_n_result = run_max_n_case()
    print("\nN-player game case for Max-N:")
    print(max_n_result["state"])
    print("Max-N utility:", max_n_result["utility"])
    print("Max-N path:", max_n_result["path"])
    print("Max-N nodes:", max_n_result["nodes"])
    print("Max-N terminal states:", max_n_result["evaluated"])

    print("\nEvaluation:")
    print("- Alpha-Beta is better for 2-player zero-sum games because it prunes branches that cannot improve the current optimum.")
    print("- Max-N is better when the game has more than 2 players and each player chooses based on their own utility vector.")
    print("- If the branching factor is large and the tree is deep, Alpha-Beta usually wins on efficiency.")
    print("- If the number of players is greater than 2, Max-N is the appropriate choice even though it may explore more nodes.")


if __name__ == "__main__":
    compare_methods()
