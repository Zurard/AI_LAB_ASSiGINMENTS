# Missionaries and Cannibals
# state = (missionaries_left, cannibals_left, boat)
# boat = 1 means boat is on the left bank, 0 means right bank
# everyone starts on the left, we want everyone on the right

TOTAL_M = 3
TOTAL_C = 3 


class MissionariesCannibals:
    def __init__(self, start=(3, 3, 1)):
        self.start = start
        self.heuristic_name = "people/2"

    def is_goal(self, state):
        return state == (0, 0, 0)

    def _ok(self, m, c):
        # a bank is ok if no missionaries, or missionaries are not outnumbered
        if m < 0 or c < 0 or m > TOTAL_M or c > TOTAL_C:
            return False
        if m > 0 and m < c:
            return False
        return True

    def valid(self, state):
        ml, cl, boat = state
        mr = TOTAL_M - ml
        cr = TOTAL_C - cl
        if boat not in (0, 1):
            return False
        return self._ok(ml, cl) and self._ok(mr, cr)

    def successors(self, state):
        ml, cl, boat = state
        # boat carries 1 or 2 people
        possible = [(1, 0), (2, 0), (0, 1), (0, 2), (1, 1)]
        moves = []
        for dm, dc in possible:
            if boat == 1:
                ns = (ml - dm, cl - dc, 0)
                action = "L->R " + str(dm) + "M " + str(dc) + "C"
            else:
                ns = (ml + dm, cl + dc, 1)
                action = "R->L " + str(dm) + "M " + str(dc) + "C"
            if self.valid(ns):
                moves.append((action, ns, 1))
        return moves

    def heuristic(self, state):
        # each trip can take at most 2 people to the other side
        # so this is admissible (never overestimates)
        ml, cl, boat = state
        left = ml + cl
        if left == 0:
            return 0
        # if boat is on the right we need at least one extra trip back
        h = (left + 1) // 2
        if boat == 0 and left > 0:
            h += 1
        return h


def print_state(state):
    ml, cl, boat = state
    mr = TOTAL_M - ml
    cr = TOTAL_C - cl
    left = str(ml) + "M " + str(cl) + "C"
    right = str(mr) + "M " + str(cr) + "C"
    if boat == 1:
        mid = " [boat] ~~~~ "
    else:
        mid = " ~~~~ [boat] "
    print(left + mid + right)
