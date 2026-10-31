"""S5: DFS on the first empty cell, digits ascending, rejecting placements
that leave a cell or unit with no option."""

from ..grid import (
    BOX_OF,
    CELL_NAMES,
    COL_OF,
    DIGITS,
    FULL_MASK,
    ROW_OF,
    UNITS,
    unit_masks,
)


def trace_backtracking(puzzle):
    g = list(puzzle)
    rows, cols, boxes = unit_masks(g)
    steps = []

    def prune_ok():
        cand = {}
        for i in range(81):
            if g[i] == 0:
                mask = FULL_MASK & ~(
                    rows[ROW_OF[i]] | cols[COL_OF[i]] | boxes[BOX_OF[i]]
                )
                if not mask:
                    return False
                cand[i] = mask
        for _name, cells in UNITS:
            present = 0
            for i in cells:
                if g[i]:
                    present |= 1 << g[i]
            missing = FULL_MASK & ~present
            while missing:
                bit = missing & -missing
                missing ^= bit
                for i in cells:
                    if g[i] == 0 and cand[i] & bit:
                        break
                else:
                    return False
        return True

    def search():
        try:
            i = g.index(0)
        except ValueError:
            return True
        r, c, b = ROW_OF[i], COL_OF[i], BOX_OF[i]
        for d in DIGITS:
            bit = 1 << d
            if (rows[r] | cols[c] | boxes[b]) & bit:
                continue
            g[i] = d
            rows[r] |= bit
            cols[c] |= bit
            boxes[b] |= bit
            accepted = prune_ok()
            if accepted:
                steps.append(f"{CELL_NAMES[i]} = {d}")
                if search():
                    return True
            else:
                steps.append(f"{CELL_NAMES[i]} = {d} x")
            g[i] = 0
            rows[r] ^= bit
            cols[c] ^= bit
            boxes[b] ^= bit
        steps.append(f"{CELL_NAMES[i]} back")
        return False

    search()
    return steps, g
