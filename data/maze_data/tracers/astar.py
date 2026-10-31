"""S3 A*: pop the open cell with the smallest f = steps + Manhattan-to-goal."""

import heapq

from ..grid import manhattan


def trace_astar(maze, s, g):
    """Pop order; ties go to the first discovered cell; stops when the goal
    is popped. A perfect maze has one route to every cell, so the first
    discovery is the only one and nothing is ever re-opened."""
    steps = {s: 0}
    heap = [(manhattan(s, g), 0, s)]
    discovered = 1
    order = []
    while True:
        _f, _n, cur = heapq.heappop(heap)
        order.append(cur)
        if cur == g:
            return order
        for j in maze[cur]:
            if j not in steps:
                steps[j] = steps[cur] + 1
                heapq.heappush(heap, (steps[j] + manhattan(j, g), discovered, j))
                discovered += 1
