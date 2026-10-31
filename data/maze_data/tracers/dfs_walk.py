"""S2 depth-first walk: a walker that steps back one cell at dead ends."""

from ..grid import CELLS


def trace_dfs_walk(maze, s, g):
    """Every cell the walker stands on, from the start until it stands on
    the goal. It moves to the first unvisited open neighbour (N E S W);
    at a dead end it steps back to the previous cell, which is written."""
    visited = [False] * CELLS
    visited[s] = True
    stack = [s]
    order = [s]
    while stack[-1] != g:
        cur = stack[-1]
        for j in maze[cur]:
            if not visited[j]:
                visited[j] = True
                stack.append(j)
                order.append(j)
                break
        else:
            stack.pop()
            order.append(stack[-1])
    return order
