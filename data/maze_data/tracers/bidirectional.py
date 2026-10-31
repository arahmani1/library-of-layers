"""S4 bidirectional BFS: two frontiers taking turns, start side first."""

from collections import deque

from ..grid import CELLS


def trace_bidirectional(maze, s, g):
    """Expanded cells in turn order, then the meeting cell: the first
    neighbour an expansion finds already known to the other side."""
    known = ([False] * CELLS, [False] * CELLS)
    known[0][s] = True
    known[1][g] = True
    queues = (deque([s]), deque([g]))
    order = []
    side = 0
    while True:
        cur = queues[side].popleft()
        order.append(cur)
        own, other = known[side], known[1 - side]
        for j in maze[cur]:
            if other[j]:
                order.append(j)
                return order
            if not own[j]:
                own[j] = True
                queues[side].append(j)
        side = 1 - side
