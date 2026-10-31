"""Verified oracle for the maze-strategies dataset.

Unoptimised reference implementation of the maze
generator, the unique path, the five tracers, the difficulty filter and the
formatter. Any faster implementation must reproduce these traces exactly
(tests/test_maze_matches_reference.py is the gate). Do not optimise this file.
"""

from collections import deque

FORMAT_VERSION = 1
N = 8
CELLS = N * N
MIN_PATH = 12
BFS_EXTRA = 3

# Neighbour order everywhere: N, E, S, W. It defines the reference traces.
DIRECTIONS = (("N", -1, 0), ("E", 0, 1), ("S", 1, 0), ("W", 0, -1))
CELL_NAMES = [f"r{i // N + 1}c{i % N + 1}" for i in range(CELLS)]
CELL_INDEX = {name: i for i, name in enumerate(CELL_NAMES)}

# Router-only labels: six cell tokens each, distinct even as unordered bags.
LABELS = {
    1: "r1c1 r1c2 r2c1 r1c3 r2c2 r3c1",
    2: "r1c1 r1c2 r1c3 r1c4 r1c3 r1c2",
    3: "r1c1 r1c2 r2c2 r2c3 r3c3 r3c4",
    4: "r1c1 r8c8 r1c2 r8c7 r1c3 r8c6",
    5: "r8c1 r8c2 r8c3 r1c8 r2c8 r3c8",
}


def neighbours(i):
    """(side letter, cell) for the in-grid neighbours of ``i``, N E S W."""
    r, c = divmod(i, N)
    out = []
    for letter, dr, dc in DIRECTIONS:
        rr, cc = r + dr, c + dc
        if 0 <= rr < N and 0 <= cc < N:
            out.append((letter, rr * N + cc))
    return out


def open_neighbours(adj, i):
    return [j for _, j in neighbours(i) if j in adj[i]]


def open_letters(adj, i):
    return [letter for letter, j in neighbours(i) if j in adj[i]]


def manhattan(i, j):
    return abs(i // N - j // N) + abs(i % N - j % N)


# ---------------------------------------------------------------- generation

def random_maze(rng):
    """Recursive backtracker from a random cell: step to a random unvisited
    neighbour (N E S W candidates, ``rng.choice``), back up at dead ends."""
    adj = [set() for _ in range(CELLS)]
    visited = [False] * CELLS
    start = rng.randrange(CELLS)
    visited[start] = True
    stack = [start]
    while stack:
        cur = stack[-1]
        options = [j for _, j in neighbours(cur) if not visited[j]]
        if options:
            nxt = rng.choice(options)
            adj[cur].add(nxt)
            adj[nxt].add(cur)
            visited[nxt] = True
            stack.append(nxt)
        else:
            stack.pop()
    return adj


def maze_key(adj):
    """One hex digit per cell holding its open sides as N E S W bits."""
    digits = []
    for i in range(CELLS):
        mask = 0
        for letter, j in neighbours(i):
            if j in adj[i]:
                mask |= {"N": 8, "E": 4, "S": 2, "W": 1}[letter]
        digits.append(f"{mask:x}")
    return "".join(digits)


def unique_path(adj, s, g):
    """The only start-to-goal walk of a perfect maze, from BFS parents."""
    parent = {s: None}
    queue = deque([s])
    while queue:
        cur = queue.popleft()
        if cur == g:
            break
        for j in open_neighbours(adj, cur):
            if j not in parent:
                parent[j] = cur
                queue.append(j)
    path = []
    cur = g
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    return path[::-1]


# ------------------------------------------------------------------- tracers

def trace_bfs(adj, s, g):
    """S1: expand the oldest discovered cell; trace = dequeue order; stop when
    the goal is dequeued."""
    discovered = {s}
    queue = deque([s])
    order = []
    while True:
        cur = queue.popleft()
        order.append(cur)
        if cur == g:
            return order
        for j in open_neighbours(adj, cur):
            if j not in discovered:
                discovered.add(j)
                queue.append(j)


def trace_dfs_walk(adj, s, g):
    """S2: walk to the first unvisited open neighbour; at a dead end step back
    one cell, writing the cell stepped back to; stop standing on the goal."""
    visited = {s}
    stack = [s]
    order = [s]
    while stack[-1] != g:
        cur = stack[-1]
        nxt = None
        for j in open_neighbours(adj, cur):
            if j not in visited:
                nxt = j
                break
        if nxt is None:
            stack.pop()
            order.append(stack[-1])
        else:
            visited.add(nxt)
            stack.append(nxt)
            order.append(nxt)
    return order


def trace_astar(adj, s, g):
    """S3: pop the open cell with the smallest f = steps + Manhattan, ties to
    the first discovered; trace = pop order; stop when the goal is popped."""
    steps = {s: 0}
    discovered = {s: 0}
    open_cells = [s]
    order = []
    while True:
        best = min(
            open_cells,
            key=lambda i: (steps[i] + manhattan(i, g), discovered[i]),
        )
        open_cells.remove(best)
        order.append(best)
        if best == g:
            return order
        for j in open_neighbours(adj, best):
            # A perfect maze has no second route to a cell, so the first
            # discovery is the only one and there is nothing to re-open.
            if j not in discovered:
                discovered[j] = len(discovered)
                steps[j] = steps[best] + 1
                open_cells.append(j)


def trace_bidirectional(adj, s, g):
    """S4: two BFS frontiers, start side first, one expansion each in turn;
    an expansion that discovers a cell already known to the other side writes
    that cell and stops."""
    known = [{s}, {g}]
    queues = [deque([s]), deque([g])]
    order = []
    side = 0
    while True:
        cur = queues[side].popleft()
        order.append(cur)
        for j in open_neighbours(adj, cur):
            if j in known[1 - side]:
                order.append(j)
                return order
            if j not in known[side]:
                known[side].add(j)
                queues[side].append(j)
        side = 1 - side


def trace_dead_end(adj, s, g):
    """S5: row-major passes filling every cell that is not start/goal and has
    at most one unfilled open neighbour, fills applying immediately; stop
    after a pass that fills nothing."""
    filled = [False] * CELLS
    order = []
    while True:
        changed = False
        for i in range(CELLS):
            if i == s or i == g or filled[i]:
                continue
            unfilled = [j for j in open_neighbours(adj, i) if not filled[j]]
            if len(unfilled) <= 1:
                filled[i] = True
                order.append(i)
                changed = True
        if not changed:
            return order


STRATEGIES = {
    1: trace_bfs,
    2: trace_dfs_walk,
    3: trace_astar,
    4: trace_bidirectional,
    5: trace_dead_end,
}
PAIRS = [(a, b) for a in range(1, 6) for b in range(a + 1, 6)]


# ---------------------------------------------------------------- difficulty

def rejection(traces, path, min_path=MIN_PATH, bfs_extra=BFS_EXTRA):
    """Why the maze is rejected, or None when it passes the four rules."""
    if len(path) < min_path:
        return "short_path"
    if len(traces[2]) <= len(path):
        return "dfs_no_retrace"
    if any(traces[a] == traces[b] for a, b in PAIRS):
        return "identical_pair"
    # BFS dequeues every path cell before the goal, so the rest is off-path.
    if len(traces[1]) - len(path) < bfs_extra:
        return "bfs_on_path"
    return None


def is_difficult(traces, path, min_path=MIN_PATH, bfs_extra=BFS_EXTRA):
    return rejection(traces, path, min_path, bfs_extra) is None


def first_divergence(a, b):
    """Index of the first differing cell; the shorter length for a prefix."""
    for n, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return n
    return min(len(a), len(b))


# ---------------------------------------------------------------- formatting

def question(adj, s, g, k):
    tokens = ["Q", ":"]
    for i in range(CELLS):
        tokens.append(CELL_NAMES[i])
        tokens += open_letters(adj, i)
    tokens += ["start", CELL_NAMES[s], "goal", CELL_NAMES[g], "["]
    tokens += LABELS[k].split(" ")
    tokens.append("]")
    return tokens


def build_sequence(adj, s, g, k, trace, path):
    tokens = question(adj, s, g, k) + ["R", ":"]
    tokens += [CELL_NAMES[i] for i in trace]
    tokens += ["A", ":"]
    tokens += [CELL_NAMES[i] for i in path]
    tokens.append("<eos>")
    return " ".join(tokens)


def parse_question(tokens):
    """(adj, start, goal) from the tokens of a question, label or not."""
    adj = [set() for _ in range(CELLS)]
    n = 2
    for i in range(CELLS):
        assert tokens[n] == CELL_NAMES[i], tokens[n]
        n += 1
        while tokens[n] in ("N", "E", "S", "W"):
            j = dict(neighbours(i))[tokens[n]]
            adj[i].add(j)
            adj[j].add(i)
            n += 1
    assert tokens[n] == "start" and tokens[n + 2] == "goal"
    return adj, CELL_INDEX[tokens[n + 1]], CELL_INDEX[tokens[n + 3]]


def generate(rng, min_path=MIN_PATH, bfs_extra=BFS_EXTRA):
    """One attempt: (adj, s, g, path, traces) or the rejection reason."""
    adj = random_maze(rng)
    s, g = rng.sample(range(CELLS), 2)
    path = unique_path(adj, s, g)
    if len(path) < min_path:
        return "short_path"
    traces = {k: STRATEGIES[k](adj, s, g) for k in STRATEGIES}
    reason = rejection(traces, path, min_path, bfs_extra)
    if reason is not None:
        return reason
    return adj, s, g, path, traces


def generate_maze(rng, min_path=MIN_PATH, bfs_extra=BFS_EXTRA):
    """Resample until an attempt is accepted."""
    while True:
        result = generate(rng, min_path, bfs_extra)
        if not isinstance(result, str):
            return result


# Worked example: start r3c6, goal r8c2.
EXAMPLE = {
    "seed": 0,
    "start": "r3c6",
    "goal": "r8c2",
    "question": (
        "Q : r1c1 E S r1c2 E W r1c3 E W r1c4 E S W r1c5 W r1c6 E S r1c7 E W "
        "r1c8 S W r2c1 N E r2c2 E S W r2c3 W r2c4 N E r2c5 S W r2c6 N S "
        "r2c7 E S r2c8 N W r3c1 S r3c2 N S r3c3 E S r3c4 E W r3c5 N W "
        "r3c6 N S r3c7 N E r3c8 S W r4c1 N E r4c2 N W r4c3 N S r4c4 E S "
        "r4c5 E W r4c6 N E W r4c7 W r4c8 N S r5c1 E S r5c2 S W r5c3 N S "
        "r5c4 N E r5c5 S W r5c6 S r5c7 S r5c8 N S r6c1 N S r6c2 N E "
        "r6c3 N S W r6c4 E S r6c5 N W r6c6 N S r6c7 N S r6c8 N S r7c1 N S "
        "r7c2 E r7c3 N W r7c4 N E r7c5 S W r7c6 N S r7c7 N E r7c8 N W "
        "r8c1 N E r8c2 E W r8c3 E W r8c4 E W r8c5 N E W r8c6 N E W r8c7 E W "
        "r8c8 W start r3c6 goal r8c2"
    ),
    "traces": {
        1: "r3c6 r2c6 r4c6 r1c6 r4c7 r4c5 r1c7 r4c4 r1c8 r5c4 r2c8 r5c5 "
           "r2c7 r6c5 r3c7 r6c4 r3c8 r7c4 r4c8 r7c5 r5c8 r8c5 r6c8 r8c6 "
           "r8c4 r7c8 r7c6 r8c7 r8c3 r7c7 r6c6 r8c8 r8c2",
        2: "r3c6 r2c6 r1c6 r1c7 r1c8 r2c8 r2c7 r3c7 r3c8 r4c8 r5c8 r6c8 "
           "r7c8 r7c7 r6c7 r5c7 r6c7 r7c7 r7c8 r6c8 r5c8 r4c8 r3c8 r3c7 "
           "r2c7 r2c8 r1c8 r1c7 r1c6 r2c6 r3c6 r4c6 r4c7 r4c6 r4c5 r4c4 "
           "r5c4 r5c5 r6c5 r6c4 r7c4 r7c5 r8c5 r8c6 r7c6 r6c6 r5c6 r6c6 "
           "r7c6 r8c6 r8c7 r8c8 r8c7 r8c6 r8c5 r8c4 r8c3 r8c2",
        3: "r3c6 r4c6 r4c5 r4c4 r5c4 r2c6 r4c7 r5c5 r6c5 r6c4 r7c4 r1c6 "
           "r7c5 r8c5 r8c4 r8c3 r8c2",
        4: "r3c6 r8c2 r2c6 r8c3 r4c6 r8c1 r1c6 r8c4 r4c7 r7c1 r4c5 r8c5 "
           "r1c7 r6c1 r4c4 r7c5 r1c8 r8c6 r5c4 r5c1 r2c8 r7c4 r5c5 r7c6 "
           "r2c7 r8c7 r6c5 r6c4",
        5: "r1c5 r2c3 r3c1 r4c1 r4c2 r4c7 r5c6 r5c7 r6c6 r6c7 r7c2 r7c3 "
           "r7c6 r7c7 r7c8 r8c8 r3c2 r6c8 r8c7 r2c2 r5c8 r8c6 r2c1 r4c8 "
           "r1c1 r1c2 r1c3 r1c4 r2c4 r2c5 r3c5 r3c8 r3c4 r3c7 r2c7 r2c8 "
           "r3c3 r4c3 r5c3 r6c3 r1c8 r6c2 r1c7 r5c2 r1c6 r2c6 r5c1 r6c1 "
           "r7c1 r8c1",
    },
    "answer": "r3c6 r4c6 r4c5 r4c4 r5c4 r5c5 r6c5 r6c4 r7c4 r7c5 r8c5 r8c4 "
              "r8c3 r8c2",
}
