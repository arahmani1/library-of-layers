from .naked_singles import trace_naked_singles
from .hidden_singles import trace_hidden_singles
from .crosshatching import trace_crosshatching
from .propagation import trace_propagation
from .backtracking import trace_backtracking

STRATEGIES = {
    1: trace_naked_singles,
    2: trace_hidden_singles,
    3: trace_crosshatching,
    4: trace_propagation,
    5: trace_backtracking,
}
