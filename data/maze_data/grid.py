"""Grid geometry, the maze generator, keys, the unique path and ASCII art.

A maze is a list of 64 tuples: ``maze[i]`` holds the open neighbours of cell
``i`` in N E S W order, so a perfect maze lists each of its 63 open walls
twice. Cells are row-major indices, named ``r<row>c<col>`` with 1-indexed
rows and columns. ``random_maze`` makes exactly the same rng calls as the
oracle (``data/maze_reference.py``).
"""

from collections import deque

N = 8
CELLS = N * N
CELL_NAMES = [f"r{i // N + 1}c{i % N + 1}" for i in range(CELLS)]
CELL_INDEX = {name: i for i, name in enumerate(CELL_NAMES)}
SIDES = ("N", "E", "S", "W")
BIT = {"N": 8, "E": 4, "S": 2, "W": 1}

# In-grid neighbours of every cell as (side letter, cell), N E S W order.
NEIGHBOURS = []
for _i in range(CELLS):
    _r, _c = divmod(_i, N)
    NEIGHBOURS.append(tuple(
        (letter, (_r + dr) * N + _c + dc)
        for letter, dr, dc in (("N", -1, 0), ("E", 0, 1), ("S", 1, 0), ("W", 0, -1))
        if 0 <= _r + dr < N and 0 <= _c + dc < N
    ))
SIDE_OF = [{j: letter for letter, j in NEIGHBOURS[i]} for i in range(CELLS)]
NEIGHBOUR_ON = [dict(NEIGHBOURS[i]) for i in range(CELLS)]


def manhattan(i, j):
    return abs(i // N - j // N) + abs(i % N - j % N)


def open_letters(maze, i):
    return [SIDE_OF[i][j] for j in maze[i]]


def random_maze(rng):
    """Recursive backtracker from a random cell: step to a random unvisited
    neighbour (N E S W candidates, ``rng.choice``), back up at dead ends."""
    links = [set() for _ in range(CELLS)]
    visited = [False] * CELLS
    start = rng.randrange(CELLS)
    visited[start] = True
    stack = [start]
    while stack:
        cur = stack[-1]
        options = [j for _, j in NEIGHBOURS[cur] if not visited[j]]
        if options:
            nxt = rng.choice(options)
            links[cur].add(nxt)
            links[nxt].add(cur)
            visited[nxt] = True
            stack.append(nxt)
        else:
            stack.pop()
    return _from_links(links)


def _from_links(links):
    return [tuple(j for _, j in NEIGHBOURS[i] if j in links[i]) for i in range(CELLS)]


def from_sides(sides):
    """Maze from per-cell open-side letters (as written in a question).

    Raises ``ValueError`` for a side that leaves the grid or a wall that is
    open from one side only.
    """
    links = [set() for _ in range(CELLS)]
    for i, letters in enumerate(sides):
        for letter in letters:
            if letter not in NEIGHBOUR_ON[i]:
                raise ValueError(f"{CELL_NAMES[i]} has no {letter} neighbour")
            links[i].add(NEIGHBOUR_ON[i][letter])
    for i in range(CELLS):
        for j in links[i]:
            if i not in links[j]:
                raise ValueError(
                    f"wall {CELL_NAMES[i]}-{CELL_NAMES[j]} is open from one side only"
                )
    return _from_links(links)


def maze_key(maze):
    """One hex digit per cell holding its open sides as N E S W bits."""
    return "".join(
        f"{sum(BIT[SIDE_OF[i][j]] for j in maze[i]):x}" for i in range(CELLS)
    )


def unique_path(maze, s, g):
    """The only start-to-goal walk of a perfect maze, from BFS parents."""
    parent = {s: None}
    queue = deque([s])
    while queue:
        cur = queue.popleft()
        if cur == g:
            break
        for j in maze[cur]:
            if j not in parent:
                parent[j] = cur
                queue.append(j)
    path = []
    cur = g
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    return path[::-1]


def ascii_draw(maze, s, g):
    """Draw the maze with 2-wide cells, S and G marked."""
    lines = ["+" + "--+" * N]
    for r in range(N):
        row = "|"
        below = "+"
        for c in range(N):
            i = r * N + c
            row += " S" if i == s else " G" if i == g else "  "
            row += " " if i + 1 in maze[i] else "|"
            below += "  +" if i + N in maze[i] else "--+"
        lines.append(row)
        lines.append(below)
    return "\n".join(lines)
