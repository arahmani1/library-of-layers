"""S3: digit-major crosshatching over boxes."""

from ..grid import BOXES, CELL_NAMES, COL_OF, DIGITS, ROW_OF, unit_masks


def trace_crosshatching(puzzle):
    g = list(puzzle)
    rows, cols, boxes = unit_masks(g)
    steps = []
    progress = True
    while progress:
        progress = False
        for d in DIGITS:
            bit = 1 << d
            for b, cells in enumerate(BOXES):
                if boxes[b] & bit:
                    continue
                spots = [
                    i
                    for i in cells
                    if g[i] == 0 and not (rows[ROW_OF[i]] | cols[COL_OF[i]]) & bit
                ]
                if len(spots) == 1:
                    i = spots[0]
                    steps.append(f"{d} box{b + 1} -> {CELL_NAMES[i]}")
                    g[i] = d
                    rows[ROW_OF[i]] |= bit
                    cols[COL_OF[i]] |= bit
                    boxes[b] |= bit
                    progress = True
    return steps, g
