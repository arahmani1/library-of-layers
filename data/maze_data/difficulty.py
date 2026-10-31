"""Difficulty filter and trace divergence."""

MIN_PATH = 12
BFS_EXTRA = 3
PAIRS = tuple((a, b) for a in range(1, 6) for b in range(a + 1, 6))


def rejection(traces, path, min_path=MIN_PATH, bfs_extra=BFS_EXTRA):
    """Why the maze is rejected, or ``None`` when it passes all four rules.

    Rules are checked cheapest first, so the reason is the first that fails:
    the path is too short; the DFS walker never steps back; two strategies
    write the same trace; BFS stays (almost) on the path. BFS dequeues every
    path cell before the goal, so its off-path expansions are the rest.
    """
    if len(path) < min_path:
        return "short_path"
    if len(traces[2]) <= len(path):
        return "dfs_no_retrace"
    if any(traces[a] == traces[b] for a, b in PAIRS):
        return "identical_pair"
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


def divergences(traces):
    """``{"S1-S2": index, ...}`` for the ten strategy pairs."""
    return {
        f"S{a}-S{b}": first_divergence(traces[a], traces[b]) for a, b in PAIRS
    }
