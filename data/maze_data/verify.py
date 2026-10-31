"""Per-maze checks; run on every generated maze."""

from .format import LABEL_SPAN, Q_LEN, parse_sequence
from .grid import CELLS, unique_path
from .identify import identify
from .vocab import UNK, encode


class VerificationError(AssertionError):
    pass


def valid_walk(maze, cells, s, g):
    """Consecutive cells adjacent and open, from ``s`` to ``g``."""
    return (
        bool(cells) and cells[0] == s and cells[-1] == g
        and all(b in maze[a] for a, b in zip(cells, cells[1:]))
    )


def verify_sequence(sequence, references=None):
    """Check one formatted sequence against the maze it encodes.

    Returns ``(maze, start, goal, strategy, trace, path)``. With
    ``references`` (the maze's five traces) the trace is compared to them;
    without, the five tracers are rerun on the parsed question, which is the
    check a stored sequence gets when it is re-read.
    """
    tokens = sequence.split(" ")
    if UNK in encode(sequence):
        raise VerificationError("sequence contains <unk>")
    if (
        tokens[Q_LEN] != "R"
        or tokens[LABEL_SPAN[0]] != "["
        or tokens[LABEL_SPAN[1] - 1] != "]"
    ):
        raise VerificationError("label or R is not at its fixed position")
    try:
        maze, s, g, k, trace, path = parse_sequence(sequence)
    except ValueError as error:
        raise VerificationError(str(error)) from None
    if path != unique_path(maze, s, g) or not valid_walk(maze, path, s, g):
        raise VerificationError("answer is not the start-to-goal path")
    if identify(trace, maze, s, g, references)[1] != k:
        raise VerificationError(f"trace is not S{k}'s")
    return maze, s, g, k, trace, path


def verify_maze(maze, s, g, traces, path, sequences):
    """``traces``: {strategy: cells}; ``sequences``: {strategy: str}."""
    if path != unique_path(maze, s, g) or not valid_walk(maze, path, s, g):
        raise VerificationError("path is not the unique start-to-goal walk")
    if sorted(traces[5]) != sorted(set(range(CELLS)) - set(path)):
        raise VerificationError("dead-end filling did not fill the complement")
    for k, sequence in sequences.items():
        # The tracers are deterministic in (maze, s, g), so a sequence whose
        # question parses back to this maze and whose trace equals the one
        # computed from it is token-for-token what recomputation would give.
        parsed = verify_sequence(sequence, references=traces)
        if parsed != (maze, s, g, k, traces[k], path):
            raise VerificationError(f"S{k} sequence does not encode its maze")
