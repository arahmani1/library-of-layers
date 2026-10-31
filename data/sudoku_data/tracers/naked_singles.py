"""S1: repeated row-major sweeps placing cells with a single candidate."""

from ..grid import BOX_OF, CELL_NAMES, COL_OF, FULL_MASK, ROW_OF, unit_masks


def trace_naked_singles(puzzle):
    g = list(puzzle)
    rows, cols, boxes = unit_masks(g)
    steps = []
    progress = True
    while progress:
        progress = False
        for i in range(81):
            if g[i] == 0:
                r, c, b = ROW_OF[i], COL_OF[i], BOX_OF[i]
                mask = FULL_MASK & ~(rows[r] | cols[c] | boxes[b])
                if mask and mask & (mask - 1) == 0:
                    d = mask.bit_length() - 1
                    steps.append(f"{CELL_NAMES[i]} = {{ {d} }}")
                    g[i] = d
                    rows[r] |= mask
                    cols[c] |= mask
                    boxes[b] |= mask
                    progress = True
    return steps, g
