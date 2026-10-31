"""Grid geometry, candidate logic and the puzzle generator.

Grids are flat lists of 81 ints, 0 for blank, row-major, rows/cols/boxes
1-indexed in names. Candidate sets are bitmasks with bit d for digit d.
random_full_grid and transform make exactly the same rng calls as the oracle.
"""

DIGITS = tuple(range(1, 10))
CELL_NAMES = [f"r{i // 9 + 1}c{i % 9 + 1}" for i in range(81)]
CELL_INDEX = {name: i for i, name in enumerate(CELL_NAMES)}

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

ROW_OF = [i // 9 for i in range(81)]
COL_OF = [i % 9 for i in range(81)]
BOX_OF = [(i // 9 // 3) * 3 + (i % 9) // 3 for i in range(81)]

PEERS = []
for _i in range(81):
    PEERS.append(
        sorted(set(ROWS[ROW_OF[_i]] + COLS[COL_OF[_i]] + BOXES[BOX_OF[_i]]) - {_i})
    )

FULL_MASK = 0b1111111110


def parse(text):
    return [0 if ch in ".0" else int(ch) for ch in text]


def to_str(grid):
    return "".join("." if v == 0 else str(v) for v in grid)


def unit_masks(grid):
    rows, cols, boxes = [0] * 9, [0] * 9, [0] * 9
    for i, v in enumerate(grid):
        if v:
            bit = 1 << v
            rows[ROW_OF[i]] |= bit
            cols[COL_OF[i]] |= bit
            boxes[BOX_OF[i]] |= bit
    return rows, cols, boxes


def candidates(grid, i):
    used = {grid[p] for p in PEERS[i]}
    return {d for d in DIGITS if d not in used}


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
