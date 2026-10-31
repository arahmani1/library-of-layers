"""Per-puzzle checks; run on every generated puzzle."""

from .format import classify, placements
from .grid import CELL_INDEX, CELL_NAMES, PEERS, UNITS, candidates
from .vocab import UNK, encode

UNIT_CELLS = dict(UNITS)


class VerificationError(AssertionError):
    pass


def replay(puzzle, strategy, steps, solution):
    g = list(puzzle)
    stack = []
    for op in placements(strategy, steps):
        if op[0] == "undo":
            if not stack:
                raise VerificationError("undo with no placement to undo")
            g[stack.pop()] = 0
        else:
            _, i, d = op
            if g[i] != 0 or d not in candidates(g, i):
                raise VerificationError(f"illegal placement {d} at cell {i}")
            g[i] = d
            stack.append(i)
    if g != list(solution):
        raise VerificationError("replayed grid does not match the answer")


def _hidden_single(g, unit, d, i, step):
    cells = UNIT_CELLS[unit]
    if i not in cells:
        raise VerificationError(f"{step}: cell is not in {unit}")
    if any(g[c] == d for c in cells):
        raise VerificationError(f"{step}: {unit} already holds {d}")
    spots = [c for c in cells if g[c] == 0 and d in candidates(g, c)]
    if spots != [i]:
        raise VerificationError(f"{step}: {d} fits {len(spots)} cells of {unit}")


def faithful(puzzle, strategy, steps, solution):
    """Check that every step is the deduction its syntax claims.

    ``replay`` only checks that placements are legal and reach the answer;
    this also checks the reasoning: a naked single really has one candidate
    (S1, S4, with S4's listed peers exactly the ones the placement reduces to
    one candidate), a hidden single is the only spot for the digit in the
    named unit (S2, S3), and a backtracking trace is the deterministic
    first-empty-cell, digits-ascending search (S5). Step order is not
    checked for S1-S4.
    """

    replay(puzzle, strategy, steps, solution)
    if strategy == 5:
        from .tracers.backtracking import trace_backtracking

        if steps != trace_backtracking(puzzle)[0]:
            raise VerificationError("S5 steps differ from the search order")
        return
    g = list(puzzle)
    for step in steps:
        t = step.split(" ")
        if strategy == 1:
            i, d = CELL_INDEX[t[0]], int(t[3])
            if candidates(g, i) != {d}:
                raise VerificationError(f"{step}: not a naked single")
        elif strategy == 2:
            i, d = CELL_INDEX[t[3]], int(t[1])
            _hidden_single(g, t[0], d, i, step)
        elif strategy == 3:
            i, d = CELL_INDEX[t[3]], int(t[0])
            _hidden_single(g, t[1], d, i, step)
        else:
            i, d = CELL_INDEX[t[1]], int(t[3])
            if candidates(g, i) != {d}:
                raise VerificationError(f"{step}: not a naked single")
            listed = " ".join(t[5:])
            expected = " ".join(
                f"{CELL_NAMES[p]} -> {{ {next(iter(candidates(g, p) - {d}))} }}"
                for p in PEERS[i]
                if g[p] == 0 and d in candidates(g, p)
                and len(candidates(g, p)) == 2
            )
            if listed != expected:
                raise VerificationError(f"{step}: listed peers are not the ones forced")
        g[i] = d


def verify_puzzle(puzzle, full, traces, sequences=None):
    """traces: {strategy: (steps, solution)}; sequences: {strategy: str}."""
    for k, (steps, solution) in traces.items():
        if solution != list(full):
            raise VerificationError(f"S{k} solution differs from the full grid")
        if classify(steps) != k:
            raise VerificationError(f"S{k} steps misclassified")
        replay(puzzle, k, steps, solution)
    if sequences is not None:
        for k, seq in sequences.items():
            if UNK in encode(seq):
                raise VerificationError(f"S{k} sequence contains <unk>")
