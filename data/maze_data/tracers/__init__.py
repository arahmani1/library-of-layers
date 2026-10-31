from .bfs import trace_bfs
from .dfs_walk import trace_dfs_walk
from .astar import trace_astar
from .bidirectional import trace_bidirectional
from .dead_end import trace_dead_end

STRATEGIES = {
    1: trace_bfs,
    2: trace_dfs_walk,
    3: trace_astar,
    4: trace_bidirectional,
    5: trace_dead_end,
}
