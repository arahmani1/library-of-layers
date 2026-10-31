"""Verified oracle for the sudoku-strategies dataset.

Unoptimised, set-based implementation of the generator, the five tracers and
the formatter. Any faster implementation must reproduce these step strings
exactly (tests/test_matches_reference.py is the gate). Do not optimise this
file.
"""

import random

FORMAT_VERSION = 1
GIVENS_MIN = 30
GIVENS_MAX = 36
MAX_TOKENS = 1000

DIGITS = tuple(range(1, 10))
CELL_NAMES = [f"r{i // 9 + 1}c{i % 9 + 1}" for i in range(81)]

ROWS = [[r * 9 + c for c in range(9)] for r in range(9)]
COLS = [[r * 9 + c for r in range(9)] for c in range(9)]
BOXES = [
    [(br * 3 + r) * 9 + bc * 3 + c for r in range(3) for c in range(3)]
    for br in range(3)
    for bc in range(3)
]
UNITS = (
    [(f"row{r + 1}", ROWS[r]) for r in range(9)]
    + [(f"col{c + 1}", COLS[c]) for c in range(9)]
    + [(f"box{b + 1}", BOXES[b]) for b in range(9)]
)

PEERS = []
for _i in range(81):
    _r, _c = divmod(_i, 9)
    _b = (_r // 3) * 3 + _c // 3
    PEERS.append(sorted(set(ROWS[_r] + COLS[_c] + BOXES[_b]) - {_i}))


def parse(text):
    return [0 if ch in ".0" else int(ch) for ch in text]


def to_str(grid):
    return "".join("." if v == 0 else str(v) for v in grid)


def candidates(grid, i):
    used = {grid[p] for p in PEERS[i]}
    return {d for d in DIGITS if d not in used}


# ---------------------------------------------------------------- generation

def random_full_grid(rng):
    grid = [0] * 81

    def fill(i):
        if i == 81:
            return True
        digits = list(DIGITS)
        rng.shuffle(digits)
        for d in digits:
            if all(grid[p] != d for p in PEERS[i]):
                grid[i] = d
                if fill(i + 1):
                    return True
        grid[i] = 0
        return False

    fill(0)
    return grid


def transform(grid, rng):
    relabel = list(DIGITS)
    rng.shuffle(relabel)
    bands = [0, 1, 2]
    rng.shuffle(bands)
    rows = [[0, 1, 2] for _ in range(3)]
    for group in rows:
        rng.shuffle(group)
    stacks = [0, 1, 2]
    rng.shuffle(stacks)
    cols = [[0, 1, 2] for _ in range(3)]
    for group in cols:
        rng.shuffle(group)
    transpose = rng.random() < 0.5

    row_map = [bands[b] * 3 + rows[b][k] for b in range(3) for k in range(3)]
    col_map = [stacks[s] * 3 + cols[s][k] for s in range(3) for k in range(3)]
    out = [
        relabel[grid[row_map[r] * 9 + col_map[c]] - 1]
        for r in range(9)
        for c in range(9)
    ]
    if transpose:
        out = [out[c * 9 + r] for r in range(9) for c in range(9)]
    return out


def _solves_naked(puzzle):
    g = list(puzzle)
    progress = True
    while progress:
        progress = False
        for i in range(81):
            if g[i] == 0:
                cs = candidates(g, i)
                if len(cs) == 1:
                    g[i] = cs.pop()
                    progress = True
    return 0 not in g


def _solves_hidden(puzzle):
    g = list(puzzle)
    progress = True
    while progress:
        progress = False
        for _name, cells in UNITS:
            for d in DIGITS:
                if any(g[i] == d for i in cells):
                    continue
                spots = [i for i in cells if g[i] == 0 and d in candidates(g, i)]
                if len(spots) == 1:
                    g[spots[0]] = d
                    progress = True
    return 0 not in g


def _solves_box(puzzle):
    g = list(puzzle)
    progress = True
    while progress:
        progress = False
        for d in DIGITS:
            for cells in BOXES:
                if any(g[i] == d for i in cells):
                    continue
                spots = [i for i in cells if g[i] == 0 and d in candidates(g, i)]
                if len(spots) == 1:
                    g[spots[0]] = d
                    progress = True
    return 0 not in g


def dig_to(full, target, rng):
    grid = list(full)
    order = list(range(81))
    rng.shuffle(order)
    givens = 81
    for i in order:
        if givens == target:
            break
        removed = grid[i]
        grid[i] = 0
        if _solves_naked(grid) and _solves_box(grid) and _solves_hidden(grid):
            givens -= 1
        else:
            grid[i] = removed
    return grid


def generate_puzzle(rng, givens_min=GIVENS_MIN, givens_max=GIVENS_MAX):
    full = transform(random_full_grid(rng), rng)
    target = rng.randint(givens_min, givens_max)
    return dig_to(full, target, rng), full


# ------------------------------------------------------------------ tracers

def trace_naked_singles(puzzle):
    g = list(puzzle)
    steps = []
    progress = True
    while progress:
        progress = False
        for i in range(81):
            if g[i] == 0:
                cs = candidates(g, i)
                if len(cs) == 1:
                    d = cs.pop()
                    steps.append(f"{CELL_NAMES[i]} = {{ {d} }}")
                    g[i] = d
                    progress = True
    return steps, g


def trace_hidden_singles(puzzle):
    g = list(puzzle)
    steps = []
    progress = True
    while progress:
        progress = False
        for name, cells in UNITS:
            for d in DIGITS:
                if any(g[i] == d for i in cells):
                    continue
                spots = [i for i in cells if g[i] == 0 and d in candidates(g, i)]
                if len(spots) == 1:
                    i = spots[0]
                    steps.append(f"{name} {d} -> {CELL_NAMES[i]}")
                    g[i] = d
                    progress = True
    return steps, g


def trace_crosshatching(puzzle):
    g = list(puzzle)
    steps = []
    progress = True
    while progress:
        progress = False
        for d in DIGITS:
            for b, cells in enumerate(BOXES):
                if any(g[i] == d for i in cells):
                    continue
                spots = [i for i in cells if g[i] == 0 and d in candidates(g, i)]
                if len(spots) == 1:
                    i = spots[0]
                    steps.append(f"{d} box{b + 1} -> {CELL_NAMES[i]}")
                    g[i] = d
                    progress = True
    return steps, g


def trace_propagation(puzzle):
    g = list(puzzle)
    cands = {i: candidates(g, i) for i in range(81) if g[i] == 0}
    queue = [i for i in sorted(cands) if len(cands[i]) == 1]
    queued = set(queue)
    steps = []
    head = 0
    while head < len(queue):
        i = queue[head]
        head += 1
        d = next(iter(cands.pop(i)))
        g[i] = d
        listed = []
        for p in PEERS[i]:
            cs = cands.get(p)
            if cs is not None and d in cs:
                cs.discard(d)
                if len(cs) == 1 and p not in queued:
                    queued.add(p)
                    queue.append(p)
                    listed.append(p)
        step = f"set {CELL_NAMES[i]} = {d}"
        if listed:
            step += " : " + " ".join(
                f"{CELL_NAMES[p]} -> {{ {next(iter(cands[p]))} }}" for p in listed
            )
        steps.append(step)
    return steps, g


def trace_backtracking(puzzle):
    g = list(puzzle)
    steps = []

    def prune_ok():
        cands = {}
        for i in range(81):
            if g[i] == 0:
                cs = candidates(g, i)
                if not cs:
                    return False
                cands[i] = cs
        for _name, cells in UNITS:
            present = {g[i] for i in cells if g[i]}
            for d in DIGITS:
                if d not in present and not any(
                    i in cands and d in cands[i] for i in cells
                ):
                    return False
        return True

    def search():
        try:
            i = g.index(0)
        except ValueError:
            return True
        for d in sorted(candidates(g, i)):
            g[i] = d
            if prune_ok():
                steps.append(f"{CELL_NAMES[i]} = {d}")
                if search():
                    return True
                g[i] = 0
            else:
                steps.append(f"{CELL_NAMES[i]} = {d} x")
                g[i] = 0
        steps.append(f"{CELL_NAMES[i]} back")
        return False

    search()
    return steps, g


STRATEGIES = {
    1: trace_naked_singles,
    2: trace_hidden_singles,
    3: trace_crosshatching,
    4: trace_propagation,
    5: trace_backtracking,
}


# ---------------------------------------------------------------- formatter

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


# Generated by generate_puzzle(random.Random(1)); counts measured once and
# frozen. See tests/test_oracle_example.py.
EXAMPLE = {
    "seed": 1,
    "puzzle": "....3.194.1.4.7.3....1..876.5.9183.7.78.46..1"
              "...752...2.6.94..8.8......3..1...6.2",
    "full": "827635194619487235345129876452918367978346521"
            "163752489236594718784261953591873642",
    "steps": (45, 45, 45, 45, 145),
    "tokens": (441, 396, 396, 615, 737),
}
