"""Which strategy a trace follows, and how far it follows each one."""

from .difficulty import first_divergence
from .tracers import STRATEGIES


def identify(trace, maze, s, g, references=None):
    """``(prefix lengths by strategy, the strategy the trace equals or None)``.

    ``references`` are the five reference traces of the maze when the caller
    already has them; otherwise they are recomputed. The prefix length is
    the number of leading cells the trace shares with each reference.
    """
    if references is None:
        references = {k: STRATEGIES[k](maze, s, g) for k in STRATEGIES}
    prefixes = {k: first_divergence(trace, references[k]) for k in STRATEGIES}
    matches = [k for k in STRATEGIES if trace == references[k]]
    return prefixes, matches[0] if len(matches) == 1 else None
