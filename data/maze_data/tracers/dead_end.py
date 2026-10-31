"""S5 dead-end filling: row-major passes that fill every dead end."""

from ..grid import CELLS


def trace_dead_end(maze, s, g):
    """Filled cells in fill order. A cell that is not the start or the goal
    and has at most one unfilled open neighbour is filled, immediately, so
    later cells of the same pass see it; stops after a pass fills nothing.
    The unfilled cells that remain are exactly the start-to-goal path."""
    filled = [False] * CELLS
    order = []
    while True:
        changed = False
        for i in range(CELLS):
            if filled[i] or i == s or i == g:
                continue
            if sum(not filled[j] for j in maze[i]) <= 1:
                filled[i] = True
                order.append(i)
                changed = True
        if not changed:
            return order
