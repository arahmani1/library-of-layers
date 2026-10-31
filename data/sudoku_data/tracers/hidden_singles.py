"""S2: unit-major hidden singles; units scanned rows, cols, boxes."""

from ..grid import BOX_OF, CELL_NAMES, COL_OF, DIGITS, ROW_OF, UNITS, unit_masks


def trace_hidden_singles(puzzle):
    g = list(puzzle)
    rows, cols, boxes = unit_masks(g)
    steps = []
    progress = True
    while progress:
        progress = False
        for name, cells in UNITS:
            for d in DIGITS:
                bit = 1 << d
                if any(g[i] == d for i in cells):
                    continue
                spots = [
                    i
                    for i in cells
                    if g[i] == 0
                    and not (rows[ROW_OF[i]] | cols[COL_OF[i]] | boxes[BOX_OF[i]]) & bit
                ]
                if len(spots) == 1:
                    i = spots[0]
                    steps.append(f"{name} {d} -> {CELL_NAMES[i]}")
                    g[i] = d
                    rows[ROW_OF[i]] |= bit
                    cols[COL_OF[i]] |= bit
                    boxes[BOX_OF[i]] |= bit
                    progress = True
    return steps, g
