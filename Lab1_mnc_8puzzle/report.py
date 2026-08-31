# Prints the lab report page by page in the terminal so you can screenshot.
#   python report.py          -> one page at a time (press Enter after each SS)
#   python report.py --all    -> dump everything at once

import os
import sys
 
from Lab1_mnc_8puzzle.eight_puzzle import EightPuzzle, print_board, is_solvable, GOAL
from Lab1_mnc_8puzzle.missionaries import MissionariesCannibals, print_state
from Lab1_mnc_8puzzle.search import bfs, dfs, ucs, dls, ids, greedy, astar
from Lab1_mnc_8puzzle.main import PUZZLE_START, PUZZLE_DLS_LIMIT, MC_DLS_LIMIT, print_table, show_mc_path


PAUSE = "--all" not in sys.argv
W = 78


def cls():
    if PAUSE:
        os.system("cls" if os.name == "nt" else "clear")


def line(ch="-"):
    print(ch * W)


def box(title):
    line("=")
    print(title.center(W))
    line("=")
    print()


def wait(n, total):
    print()
    line()
    msg = "  page %d / %d" % (n, total)
    if PAUSE:
        print(msg + "   |   take a screenshot, then press Enter")
        try:
            input()
        except EOFError:
            pass
    else:
        print(msg)
        print()
        print()


def page(n, total, title, fn):
    cls()
    box(title)
    fn()
    wait(n, total)


def print_row(cols, widths):
    s = ""
    for i in range(len(cols)):
        s += str(cols[i]).ljust(widths[i])
    print(s)


# ---------- pages ----------

def p_title():
    print()
    print("ARTIFICIAL INTELLIGENCE LABORATORY".center(W))
    print()
    print("Implementation and Comparison of Search Algorithms".center(W))
    print("8-Puzzle  and  Missionaries & Cannibals".center(W))
    print()
    line()
    print()
    print("  Algorithms:")
    print("    1. BFS          2. DFS          3. UCS")
    print("    4. DLS          5. IDS          6. Greedy Best-First")
    print("    7. A*")
    print()
    print("  Heuristics:")
    print("    8-Puzzle  : misplaced tiles , Manhattan distance")
    print("    M & C     : people remaining / boat capacity")
    print()
    print("  Language : Python 3   |   Libraries : none (stdlib only)")


def p_aim():
    print("AIM")
    print("  Implement uninformed and informed search strategies and compare")
    print("  them on two classic AI problems using the same metrics.")
    print()
    print("OBJECTIVES")
    print("  1. Formulate 8-Puzzle and Missionaries & Cannibals as search problems.")
    print("  2. Implement BFS, DFS, UCS, DLS, IDS, Greedy Best-First and A*.")
    print("  3. Use misplaced tiles and Manhattan distance for informed search.")
    print("  4. Compare algorithms on cost, nodes, memory and time.")
    print()
    print("METRICS USED")
    print("  Found        : whether a goal state was reached")
    print("  Path cost    : number of moves (every action has cost 1)")
    print("  Depth        : length of the solution path")
    print("  Expanded     : nodes taken from the frontier and processed")
    print("  Generated    : child nodes created")
    print("  Max frontier : peak size of the open list  (memory)")
    print("  Time (s)     : wall-clock time of that run")


def p_puzzle_form():
    print("PROBLEM 1 : 8-PUZZLE")
    print()
    print("  State  : 3x3 board stored as a 9-tuple.  0 is the blank.")
    print("  Action : move the blank  Up / Down / Left / Right")
    print("  Cost   : 1 per move")
    print("  Test   : even inversion count  =>  solvable")
    print()
    print("  Start state                      Goal state")
    print("  -----------                      ----------")
    # print both boards side by side
    def rows_of(state):
        out = []
        for r in range(3):
            parts = []
            for c in range(3):
                v = state[r * 3 + c]
                parts.append(" " if v == 0 else str(v))
            out.append(" ".join(parts))
        return out
    a = rows_of(PUZZLE_START)
    b = rows_of(GOAL)
    for i in range(3):
        print("    " + a[i] + "                         " + b[i])
    print()
    print("  Start tuple :", PUZZLE_START)
    print("  Goal  tuple :", GOAL)
    print("  Solvable    :", is_solvable(PUZZLE_START))
    print()
    print("  Branching factor ~ 2 to 4.  State space = 9!/2 = 181440 reachable.")


def p_mc_form():
    print("PROBLEM 2 : MISSIONARIES AND CANNIBALS")
    print()
    print("  State  : (M_left, C_left, boat)")
    print("           boat = 1  means boat is on the LEFT bank")
    print("           boat = 0  means boat is on the RIGHT bank")
    print("  Start  : (3, 3, 1)   all 3M 3C and the boat on the left")
    print("  Goal   : (0, 0, 0)   everyone on the right")
    print("  Action : boat carries 1 or 2 people")
    print("           moves = 1M, 2M, 1C, 2C, 1M+1C")
    print("  Cost   : 1 per crossing")
    print()
    print("  Constraint (both banks):")
    print("    if missionaries > 0  then  missionaries >= cannibals")
    print("    otherwise the cannibals eat the missionaries  (illegal state)")
    print()
    print("  Start picture:")
    print("    3M 3C [boat] ~~~~ 0M 0C")
    print("  Goal picture:")
    print("    0M 0C ~~~~ [boat] 3M 3C")
    print()
    print("  State space is tiny (about 16 legal states).")
    print("  Optimal solution length is 11 crossings.")


def p_algos():
    print("SEARCH ALGORITHMS  (theory)")
    print()
    w = [10, 16, 14, 16, 20]
    print_row(["Algo", "Frontier", "Complete?", "Optimal?", "Notes"], w)
    line()
    print_row(["BFS", "FIFO queue", "Yes", "Yes (unit cost)", "high memory"], w)
    print_row(["DFS", "LIFO stack", "Yes (graph)", "No", "can be very long"], w)
    print_row(["UCS", "PQ on g(n)", "Yes", "Yes", "same as BFS here"], w)
    print_row(["DLS", "DFS + limit L", "No if L small", "No", "cutoff if L low"], w)
    print_row(["IDS", "DLS L=0,1,2..", "Yes", "Yes (unit cost)", "low memory"], w)
    print_row(["Greedy", "PQ on h(n)", "Not always", "No", "fast, may wander"], w)
    print_row(["A*", "PQ on g+h", "Yes", "Yes if h ok", "best tradeoff"], w)
    print()
    print("  Implementation notes")
    print("  - BFS / DFS : graph search (visited set) so 8-puzzle cannot loop.")
    print("  - DLS / IDS : tree search + cycle check on the current path.")
    print("  - UCS / A*  : keep the cheapest g(n) seen for a state.")
    print("  - All step costs are 1, so UCS returns the same cost as BFS.")


def p_heur():
    print("HEURISTICS")
    print()
    print("  8-PUZZLE  :  Misplaced tiles")
    print("    h(n) = number of tiles not in their goal position")
    print("    (blank is not counted)")
    print("    Admissible, but weak.")
    print()
    print("  8-PUZZLE  :  Manhattan distance")
    print("    h(n) = sum of |row-row_goal| + |col-col_goal|  for every tile")
    print("    Admissible AND consistent.")
    print("    Manhattan >= misplaced tiles, so it is the stronger heuristic.")
    print()
    print("  M & C  :  people remaining / 2")
    print("    boat holds at most 2 people, so")
    print("    h(n) = ceil( (M_left + C_left) / 2 )")
    print("    + 1 extra if people are still on the left and boat is on the right")
    print("    Admissible, but weak.  Start h = 3, real cost = 11")
    print("    (boat must keep coming back, and many moves are illegal).")
    print()
    print("  A* is optimal when h is admissible.")
    print("  Both 8-puzzle heuristics and the M&C heuristic are admissible.")


def p_setup():
    print("EXPERIMENTAL SETUP")
    print()
    print("  Machine     : same run for every algorithm")
    print("  Language    : Python 3")
    print("  Packages    : none  (time, heapq, collections only)")
    print()
    print("  8-Puzzle start : (1, 3, 4, 8, 0, 6, 7, 2, 5)")
    print("  8-Puzzle DLS L : 20     (optimal depth is 18)")
    print("  M & C start    : (3, 3, 1)")
    print("  M & C DLS L    : 15     (optimal depth is 11)")
    print()
    print("  Extra DLS checks (to show cutoff):")
    print("    8-Puzzle with L = 10  ->  cutoff  (needs 18)")
    print("    8-Puzzle with L = 17  ->  cutoff  (still 1 short)")
    print("    M & C    with L = 10  ->  cutoff  (needs 11)")
    print()
    print("  Informed 8-puzzle is run twice:")
    print("    Greedy + A*  with Manhattan")
    print("    Greedy + A*  with misplaced tiles")


def make_puzzle_rows():
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
    cut10 = dls(p_man, 10)
    return rows, cut10


def make_mc_rows():
    p = MissionariesCannibals()
    rows = []
    rows.append(bfs(p))
    rows.append(dfs(p))
    rows.append(ucs(p))
    rows.append(dls(p, MC_DLS_LIMIT))
    rows.append(ids(p))
    rows.append(greedy(p, "people/2"))
    rows.append(astar(p, "people/2"))
    cut10 = dls(p, 10)
    return rows, cut10


def p_puzzle_results(rows, cut10):
    print("RESULTS  :  8-PUZZLE")
    print("  Optimal cost = 18")
    print()
    print_table(rows)
    print()
    print("  DLS with limit 10 : found =", cut10.found,
          " | extra =", cut10.extra or "-",
          " | expanded =", cut10.nodes_expanded)
    print()
    print("  Quick read:")
    print("    A* (manhattan)  ->  optimal, fewest useful expansions")
    print("    Greedy          ->  fast but cost 28 (not optimal)")
    print("    DFS             ->  finds a path, but it is huge")
    print("    IDS             ->  optimal, high node count, tiny memory")


def p_puzzle_path(rows):
    print("8-PUZZLE  :  A* (Manhattan) solution   [18 moves]")
    print()
    astar_row = None
    for r in rows:
        if r.name == "A* (manhattan)":
            astar_row = r
            break
    if astar_row is None:
        print("  (no A* result)")
        return
    print("  Moves:")
    acts = astar_row.actions
    # print 6 per line
    i = 0
    while i < len(acts):
        chunk = acts[i:i + 6]
        bits = []
        for j in range(len(chunk)):
            bits.append(str(i + j + 1) + "." + chunk[j])
        print("    " + "  ".join(bits))
        i += 6
    print()
    print("  Start                         Goal")
    def row3(state, r):
        parts = []
        for c in range(3):
            v = state[r * 3 + c]
            parts.append(" " if v == 0 else str(v))
        return " ".join(parts)
    for r in range(3):
        print("    " + row3(astar_row.states[0], r) + "                      " + row3(astar_row.states[-1], r))
    print()
    print("  DFS path cost on the same start was 6536  (not printed).")
    print("  That shows DFS is complete as graph-search, but not optimal.")


def p_mc_results(rows, cut10):
    print("RESULTS  :  MISSIONARIES AND CANNIBALS")
    print("  Optimal cost = 11")
    print()
    print_table(rows)
    print()
    print("  DLS with limit 10 : found =", cut10.found,
          " | extra =", cut10.extra or "-",
          " | expanded =", cut10.nodes_expanded)
    print()
    print("  All algorithms find cost 11.  The graph is tiny, so this")
    print("  problem is for correctness, not for speed ranking.")
    print("  IDS expands more only because it repeats depths 0 .. 11.")


def p_mc_path(rows):
    print("M & C  :  A* solution   [11 crossings]")
    print()
    astar_row = None
    for r in rows:
        if r.name.startswith("A*"):
            astar_row = r
            break
    if astar_row is None:
        print("  (no A* result)")
        return
    show_mc_path(astar_row)


def p_discuss():
    print("DISCUSSION")
    print()
    print("  1. A* + Manhattan is the winner on 8-puzzle.")
    print("     Same cost as BFS (18) but ~369 expansions vs ~17649.")
    print()
    print("  2. Manhattan beats misplaced tiles.")
    print("     Both A* runs are optimal.  Stronger h => fewer nodes.")
    print()
    print("  3. Greedy is fast but not optimal (28 vs 18).")
    print("     It only looks at h(n), so it can take extra detours.")
    print()
    print("  4. DFS on 8-puzzle returned 6536 moves.  Graph search makes")
    print("     it stop, but the path quality is useless.")
    print()
    print("  5. IDS is optimal and memory-light, but it redoes every")
    print("     shallower depth, so it expands the most nodes.")
    print()
    print("  6. DLS fails when L is below the solution depth (cutoff).")
    print("     That is why IDS exists: it raises L until it works.")
    print()
    print("  7. UCS ~= BFS here because every action costs 1.")
    print()
    print("  8. M&C does not separate the algorithms.  Use it to show")
    print("     constraints / legality.  Use 8-puzzle for performance.")


def p_end():
    print("CONCLUSION")
    print("  Uninformed search is enough for Missionaries & Cannibals.")
    print("  For 8-puzzle, BFS wastes nodes and DFS returns a bad path.")
    print("  A* with Manhattan is the practical choice: optimal, few")
    print("  expansions, little time.  Greedy is only for a quick")
    print("  non-optimal path.  IDS is the pick when memory is tight.")
    print()
    print("VIVA  (short answers)")
    print("  Q. Why is Manhattan better than misplaced tiles?")
    print("  A. It dominates: Manhattan >= misplaced, still admissible.")
    print()
    print("  Q. Is A* always optimal?")
    print("  A. Yes if h is admissible.  Both of ours are.")
    print()
    print("  Q. Why is DFS complete here?")
    print("  A. Finite state space + visited set (graph search).")
    print()
    print("  Q. Why can DLS fail?")
    print("  A. If the depth limit is smaller than the solution depth.")
    print()
    print("  Q. Why does IDS expand more than BFS?")
    print("  A. It repeats levels 0, 1, 2, ... up to the solution.")
    print()
    print("  Q. How do we know the 8-puzzle start is solvable?")
    print("  A. Inversion count is even.")


def main():
    print("Running searches for the report tables...")
    print("(8-puzzle IDS can take a couple of seconds)")
    puzzle_rows, puzzle_cut = make_puzzle_rows()
    mc_rows, mc_cut = make_mc_rows()

    pages = [
        ("AI LAB  |  TITLE", p_title),
        ("AI LAB  |  AIM AND METRICS", p_aim),
        ("AI LAB  |  PROBLEM FORMULATION  (8-PUZZLE)", p_puzzle_form),
        ("AI LAB  |  PROBLEM FORMULATION  (M & C)", p_mc_form),
        ("AI LAB  |  ALGORITHMS", p_algos),
        ("AI LAB  |  HEURISTICS", p_heur),
        ("AI LAB  |  EXPERIMENTAL SETUP", p_setup),
        ("AI LAB  |  RESULTS  (8-PUZZLE)", lambda: p_puzzle_results(puzzle_rows, puzzle_cut)),
        ("AI LAB  |  SOLUTION PATH  (8-PUZZLE)", lambda: p_puzzle_path(puzzle_rows)),
        ("AI LAB  |  RESULTS  (M & C)", lambda: p_mc_results(mc_rows, mc_cut)),
        ("AI LAB  |  SOLUTION PATH  (M & C)", lambda: p_mc_path(mc_rows)),
        ("AI LAB  |  DISCUSSION", p_discuss),
        ("AI LAB  |  CONCLUSION AND VIVA", p_end),
    ]

    total = len(pages)
    for i, (title, fn) in enumerate(pages, start=1):
        page(i, total, title, fn)

    if PAUSE:
        cls()
        box("DONE")
        print("  All pages printed.  You can close this window.")
        print()


if __name__ == "__main__":
    main()
