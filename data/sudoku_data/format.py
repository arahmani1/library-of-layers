"""Sequence building, step parsing for replay, and the strategy classifier."""

import re

from .grid import CELL_INDEX

Q_LEN = 86  # Q : <81 cells> [ S<k> ] ; token 86 is R

STEP_RES = {
    1: re.compile(r"^r[1-9]c[1-9] = \{ [1-9] \}$"),
    2: re.compile(r"^(row|col|box)[1-9] [1-9] -> r[1-9]c[1-9]$"),
    3: re.compile(r"^[1-9] box[1-9] -> r[1-9]c[1-9]$"),
    4: re.compile(
        r"^set r[1-9]c[1-9] = [1-9]"
        r"( : r[1-9]c[1-9] -> \{ [1-9] \}( r[1-9]c[1-9] -> \{ [1-9] \})*)?$"
    ),
    5: re.compile(r"^r[1-9]c[1-9] (= [1-9]( x)?|back)$"),
}


def build_sequence(puzzle, strategy, steps, solution):
    parts = ["Q", ":"]
    parts += ["." if v == 0 else str(v) for v in puzzle]
    parts += ["[", f"S{strategy}", "]", "R", ":"]
    for n, step in enumerate(steps):
        if n:
            parts.append(";")
        parts += step.split(" ")
    parts += ["A", ":"]
    parts += [str(v) for v in solution]
    parts.append("<eos>")
    return " ".join(parts)


def split_sequence(sequence):
    """Return (puzzle_cells, strategy, steps, answer_digits) from a decoded
    sequence string."""
    tokens = sequence.split(" ")
    strategy = int(tokens[84][1:])
    a = len(tokens) - 84  # index of "A"; tail is A : <81 digits> <eos>
    body = tokens[Q_LEN + 2 : a]
    steps = " ".join(body).split(" ; ") if body else []
    return tokens[2:83], strategy, steps, tokens[a + 2 : a + 83]


def classify(steps):
    matches = [
        k for k, rx in STEP_RES.items() if steps and all(rx.match(s) for s in steps)
    ]
    return matches[0] if len(matches) == 1 else None


def placements(strategy, steps):
    """Parse steps into ("place", cell, digit) / ("undo",) ops for replay."""
    ops = []
    for step in steps:
        t = step.split(" ")
        if strategy == 1:
            ops.append(("place", CELL_INDEX[t[0]], int(t[3])))
        elif strategy == 2:
            ops.append(("place", CELL_INDEX[t[3]], int(t[1])))
        elif strategy == 3:
            ops.append(("place", CELL_INDEX[t[3]], int(t[0])))
        elif strategy == 4:
            ops.append(("place", CELL_INDEX[t[1]], int(t[3])))
        elif strategy == 5:
            if t[1] == "back":
                ops.append(("undo",))
            elif t[-1] != "x":
                ops.append(("place", CELL_INDEX[t[0]], int(t[2])))
        else:
            raise ValueError(f"unknown strategy {strategy}")
    return ops
