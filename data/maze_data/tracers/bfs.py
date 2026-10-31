"""S1 breadth-first search: expand the oldest discovered cell."""

from collections import deque

from ..grid import CELLS


def trace_bfs(maze, s, g):
    """Dequeue order, N E S W neighbours; stops when the goal is dequeued."""
    discovered = [False] * CELLS
    discovered[s] = True
    queue = deque([s])
    order = []
    while True:
        cur = queue.popleft()
        order.append(cur)
        if cur == g:
            return order
        for j in maze[cur]:
            if not discovered[j]:
                discovered[j] = True
                queue.append(j)
