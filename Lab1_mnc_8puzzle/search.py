# Uninformed and informed search algorithms
# Only uses stuff from the python standard library

import time
import heapq
from collections import deque

 
class Node:
    def __init__(self, state, parent=None, action=None, cost=0, depth=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.cost = cost      # g(n) path cost from start
        self.depth = depth

    def path(self):
        # walk back to the root and reverse
        nodes = []
        cur = self
        while cur is not None:
            nodes.append(cur)
            cur = cur.parent
        nodes.reverse()
        return nodes


class Result:
    def __init__(self, name):
        self.name = name
        self.found = False
        self.path_cost = None
        self.depth = None
        self.actions = []
        self.states = []
        self.nodes_expanded = 0
        self.nodes_generated = 0
        self.max_frontier = 0
        self.time_sec = 0.0
        self.extra = ""


def _fill_path(res, node):
    res.found = True
    res.path_cost = node.cost
    res.depth = node.depth
    p = node.path()
    res.states = [n.state for n in p]
    res.actions = [n.action for n in p if n.action is not None]


def bfs(problem):
    res = Result("BFS")
    t0 = time.time()

    start = Node(problem.start)
    res.nodes_generated = 1

    if problem.is_goal(start.state):
        _fill_path(res, start)
        res.time_sec = time.time() - t0
        return res

    q = deque([start])
    visited = set([start.state])
    res.max_frontier = 1

    while q:
        node = q.popleft()
        res.nodes_expanded += 1

        for action, nxt, step_cost in problem.successors(node.state):
            if nxt in visited:
                continue
            visited.add(nxt)
            child = Node(nxt, node, action, node.cost + step_cost, node.depth + 1)
            res.nodes_generated += 1

            if problem.is_goal(nxt):
                _fill_path(res, child)
                res.max_frontier = max(res.max_frontier, len(q))
                res.time_sec = time.time() - t0
                return res

            q.append(child)

        if len(q) > res.max_frontier:
            res.max_frontier = len(q)

    res.time_sec = time.time() - t0
    return res


def dfs(problem):
    # graph search so it does not loop forever
    res = Result("DFS")
    t0 = time.time()

    start = Node(problem.start)
    res.nodes_generated = 1

    if problem.is_goal(start.state):
        _fill_path(res, start)
        res.time_sec = time.time() - t0
        return res

    stack = [start]
    visited = set([start.state])
    res.max_frontier = 1

    while stack:
        node = stack.pop()
        res.nodes_expanded += 1

        kids = problem.successors(node.state)
        # reverse so the first successor is tried first (stack)
        for action, nxt, step_cost in reversed(kids):
            if nxt in visited:
                continue
            visited.add(nxt)
            child = Node(nxt, node, action, node.cost + step_cost, node.depth + 1)
            res.nodes_generated += 1

            if problem.is_goal(nxt):
                _fill_path(res, child)
                res.max_frontier = max(res.max_frontier, len(stack))
                res.time_sec = time.time() - t0
                return res

            stack.append(child)

        if len(stack) > res.max_frontier:
            res.max_frontier = len(stack)

    res.time_sec = time.time() - t0
    return res


def ucs(problem):
    res = Result("UCS")
    t0 = time.time()

    start = Node(problem.start)
    res.nodes_generated = 1
    counter = 0
    pq = [(0, counter, start)]
    best_cost = {start.state: 0}
    res.max_frontier = 1

    while pq:
        g, _, node = heapq.heappop(pq)

        # skip if we already found a cheaper way to this state
        if g > best_cost.get(node.state, g):
            continue

        res.nodes_expanded += 1

        if problem.is_goal(node.state):
            _fill_path(res, node)
            res.max_frontier = max(res.max_frontier, len(pq))
            res.time_sec = time.time() - t0
            return res

        for action, nxt, step_cost in problem.successors(node.state):
            new_cost = node.cost + step_cost
            if nxt not in best_cost or new_cost < best_cost[nxt]:
                best_cost[nxt] = new_cost
                child = Node(nxt, node, action, new_cost, node.depth + 1)
                res.nodes_generated += 1
                counter += 1
                heapq.heappush(pq, (new_cost, counter, child))

        if len(pq) > res.max_frontier:
            res.max_frontier = len(pq)

    res.time_sec = time.time() - t0
    return res


def dls(problem, limit):
    res = Result("DLS (limit=" + str(limit) + ")")
    t0 = time.time()

    def rec(node, path_set):
        res.nodes_expanded += 1
        if problem.is_goal(node.state):
            return node
        if node.depth >= limit:
            return "cutoff"

        cutoff = False
        for action, nxt, step_cost in problem.successors(node.state):
            if nxt in path_set:
                continue
            child = Node(nxt, node, action, node.cost + step_cost, node.depth + 1)
            res.nodes_generated += 1
            path_set.add(nxt)
            # recursion depth is the current path, so memory ~ depth
            if len(path_set) > res.max_frontier:
                res.max_frontier = len(path_set)
            out = rec(child, path_set)
            path_set.remove(nxt)
            if out == "cutoff":
                cutoff = True
            elif out is not None:
                return out

        if cutoff:
            return "cutoff"
        return None

    start = Node(problem.start)
    res.nodes_generated = 1
    res.max_frontier = 1
    out = rec(start, set([start.state]))
    if out is not None and out != "cutoff":
        _fill_path(res, out)
    elif out == "cutoff":
        res.extra = "cutoff"
    res.time_sec = time.time() - t0
    return res


def ids(problem, max_limit=50):
    res = Result("IDS")
    t0 = time.time()
    total_exp = 0
    total_gen = 0
    max_front = 0

    for limit in range(0, max_limit + 1):
        one = dls(problem, limit)
        total_exp += one.nodes_expanded
        total_gen += one.nodes_generated
        if one.max_frontier > max_front:
            max_front = one.max_frontier
        if one.found:
            res.found = True
            res.path_cost = one.path_cost
            res.depth = one.depth
            res.actions = one.actions
            res.states = one.states
            res.extra = "solved at depth " + str(limit)
            break
        if one.extra != "cutoff":
            # no more nodes at all
            break

    res.nodes_expanded = total_exp
    res.nodes_generated = total_gen
    res.max_frontier = max_front
    res.time_sec = time.time() - t0
    return res


def greedy(problem, h_name=None):
    title = "Greedy"
    if h_name:
        title = "Greedy (" + h_name + ")"
    res = Result(title)
    t0 = time.time()

    start = Node(problem.start)
    res.nodes_generated = 1
    counter = 0
    h0 = problem.heuristic(start.state)
    pq = [(h0, counter, start)]
    visited = set()
    res.max_frontier = 1

    while pq:
        _, _, node = heapq.heappop(pq)
        if node.state in visited:
            continue
        visited.add(node.state)
        res.nodes_expanded += 1

        if problem.is_goal(node.state):
            _fill_path(res, node)
            res.max_frontier = max(res.max_frontier, len(pq))
            res.time_sec = time.time() - t0
            return res

        for action, nxt, step_cost in problem.successors(node.state):
            if nxt in visited:
                continue
            child = Node(nxt, node, action, node.cost + step_cost, node.depth + 1)
            res.nodes_generated += 1
            counter += 1
            heapq.heappush(pq, (problem.heuristic(nxt), counter, child))

        if len(pq) > res.max_frontier:
            res.max_frontier = len(pq)

    res.time_sec = time.time() - t0
    return res


def astar(problem, h_name=None):
    title = "A*"
    if h_name:
        title = "A* (" + h_name + ")"
    res = Result(title)
    t0 = time.time()

    start = Node(problem.start)
    res.nodes_generated = 1
    counter = 0
    f0 = start.cost + problem.heuristic(start.state)
    pq = [(f0, counter, start)]
    best_cost = {start.state: 0}
    res.max_frontier = 1

    while pq:
        _, _, node = heapq.heappop(pq)
        if node.cost > best_cost.get(node.state, node.cost):
            continue

        res.nodes_expanded += 1

        if problem.is_goal(node.state):
            _fill_path(res, node)
            res.max_frontier = max(res.max_frontier, len(pq))
            res.time_sec = time.time() - t0
            return res

        for action, nxt, step_cost in problem.successors(node.state):
            new_cost = node.cost + step_cost
            if nxt not in best_cost or new_cost < best_cost[nxt]:
                best_cost[nxt] = new_cost
                child = Node(nxt, node, action, new_cost, node.depth + 1)
                res.nodes_generated += 1
                counter += 1
                f = new_cost + problem.heuristic(nxt)
                heapq.heappush(pq, (f, counter, child))

        if len(pq) > res.max_frontier:
            res.max_frontier = len(pq)

    res.time_sec = time.time() - t0
    return res
