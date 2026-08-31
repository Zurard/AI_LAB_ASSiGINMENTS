# 8 puzzle
# state is a tuple of 9 numbers, 0 is the blank

GOAL = (1, 2, 3, 4, 5, 6, 7, 8, 0)
 
# positions of each tile in the goal (for manhattan)
GOAL_POS = {}
for i in range(9):
    GOAL_POS[GOAL[i]] = (i // 3, i % 3)


class EightPuzzle:
    def __init__(self, start, heuristic="manhattan"):
        self.start = start
        self.heuristic_name = heuristic

    def is_goal(self, state):
        return state == GOAL

    def successors(self, state):
        # find blank
        i = state.index(0)
        r = i // 3
        c = i % 3
        moves = []
        # (dr, dc, name)
        dirs = [(-1, 0, "Up"), (1, 0, "Down"), (0, -1, "Left"), (0, 1, "Right")]
        for dr, dc, name in dirs:
            nr = r + dr
            nc = c + dc
            if nr < 0 or nr > 2 or nc < 0 or nc > 2:
                continue
            j = nr * 3 + nc
            lst = list(state)
            lst[i], lst[j] = lst[j], lst[i]
            moves.append((name, tuple(lst), 1))
        return moves

    def misplaced(self, state):
        count = 0
        for k in range(9):
            if state[k] != 0 and state[k] != GOAL[k]:
                count += 1
        return count

    def manhattan(self, state):
        dist = 0
        for k in range(9):
            tile = state[k]
            if tile == 0:
                continue
            r = k // 3
            c = k % 3
            gr, gc = GOAL_POS[tile]
            dist += abs(r - gr) + abs(c - gc)
        return dist

    def heuristic(self, state):
        if self.heuristic_name == "misplaced":
            return self.misplaced(state)
        return self.manhattan(state)


def inversions(state):
    arr = []
    for x in state:
        if x != 0:
            arr.append(x)
    inv = 0
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if arr[i] > arr[j]:
                inv += 1
    return inv


def is_solvable(state):
    # for 3x3, even inversion count means solvable (goal has 0 inversions)
    return inversions(state) % 2 == 0


def print_board(state):
    for r in range(3):
        row = []
        for c in range(3):
            v = state[r * 3 + c]
            if v == 0:
                row.append(" ")
            else:
                row.append(str(v))
        print(" ".join(row))
