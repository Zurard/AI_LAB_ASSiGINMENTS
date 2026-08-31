# Run all search algorithms on both problems and print a comparison table

from Lab1_mnc_8puzzle.eight_puzzle import EightPuzzle, print_board, is_solvable, GOAL
from Lab1_mnc_8puzzle.missionaries import MissionariesCannibals, print_state
from Lab1_mnc_8puzzle.search import bfs, dfs, ucs, dls, ids, greedy, astar
 

# a solvable start that is not too easy (manhattan is 9)
PUZZLE_START = (1, 3, 4, 8, 0, 6, 7, 2, 5)

# DLS needs a depth cap. if this is smaller than the real solution, DLS fails
PUZZLE_DLS_LIMIT = 20
MC_DLS_LIMIT = 15


def print_table(rows):
    header = ["Algorithm", "Found", "Cost", "Depth", "Expanded", "Generated", "MaxFront", "Time(s)"]
    widths = [22, 7, 6, 6, 10, 10, 9, 10]
    line = ""
    for i in range(len(header)):
        line += header[i].ljust(widths[i])
    print(line)
    print("-" * sum(widths))
    for r in rows:
        found = "Yes" if r.found else "No"
        cost = str(r.path_cost) if r.found else "-"
        depth = str(r.depth) if r.found else "-"
        vals = [
            r.name,
            found,
            cost,
            depth,
            str(r.nodes_expanded),
            str(r.nodes_generated),
            str(r.max_frontier),
            str(round(r.time_sec, 5)),
        ]
        line = ""
        for i in range(len(vals)):
            line += vals[i].ljust(widths[i])
        print(line)
        if r.extra:
            print("   note:", r.extra)


def show_puzzle_path(res, max_steps=30):
    if not res.found:
        print("No path")
        return
    print("Moves:", " -> ".join(res.actions))
    print("Start:")
    print_board(res.states[0])
    n = len(res.states)
    if n <= max_steps + 1:
        for i in range(1, n):
            print("after", res.actions[i - 1])
            print_board(res.states[i])
    else:
        print("... path has", n - 1, "moves, not printing every board")
        print("Goal:")
        print_board(res.states[-1])


def show_mc_path(res):
    if not res.found:
        print("No path")
        return
    for i in range(len(res.states)):
        print_state(res.states[i])
        if i < len(res.actions):
            print("   ", res.actions[i])


def run_puzzle():
    print("=" * 70)
    print("8-PUZZLE")
    print("=" * 70)
    print("Start:")
    print_board(PUZZLE_START)
    print("Goal:")
    print_board(GOAL)
    print("Solvable:", is_solvable(PUZZLE_START))
    print()

    p_man = EightPuzzle(PUZZLE_START, "manhattan")
    p_mis = EightPuzzle(PUZZLE_START, "misplaced")

    rows = []
    rows.append(bfs(p_man))
    rows.append(dfs(p_man))
    rows.append(ucs(p_man))
    rows.append(dls(p_man, PUZZLE_DLS_LIMIT))
    rows.append(ids(p_man))
    rows.append(greedy(p_man, "manhattan"))
    rows.append(greedy(p_mis, "misplaced"))
    rows.append(astar(p_man, "manhattan"))
    rows.append(astar(p_mis, "misplaced"))

    print_table(rows)
    print()
    print("Example solution from A* (manhattan):")
    for r in rows:
        if r.name == "A* (manhattan)":
            show_puzzle_path(r)
            break
    return rows


def run_mc():
    print()
    print("=" * 70)
    print("MISSIONARIES AND CANNIBALS")
    print("=" * 70)
    print("Start: 3M 3C on left, boat on left")
    print("Goal : everyone on the right")
    print()

    p = MissionariesCannibals()

    rows = []
    rows.append(bfs(p))
    rows.append(dfs(p))
    rows.append(ucs(p))
    rows.append(dls(p, MC_DLS_LIMIT))
    rows.append(ids(p))
    rows.append(greedy(p, "people/2"))
    rows.append(astar(p, "people/2"))

    print_table(rows)
    print()
    print("Example solution from A*:")
    for r in rows:
        if r.name.startswith("A*"):
            show_mc_path(r)
            break
    return rows


if __name__ == "__main__":
    run_puzzle()
    run_mc()
