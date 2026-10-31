"""Digging: remove clues while pure S1, S2 and S3 each still solve.

The removal check runs S1 and S3 only: S2's rule set contains S3's, and
strategy closures are confluent, so S3-solvable implies S2-solvable.
test_matches_reference.py checks dig_to against the oracle, which tests all
three.
"""

from .grid import (
    BOXES,
    BOX_OF,
    COL_OF,
    DIGITS,
    FULL_MASK,
    ROW_OF,
    unit_masks,
)

GIVENS_MIN = 30
GIVENS_MAX = 36


def _solves_naked(grid):
    g = list(grid)
    rows, cols, boxes = unit_masks(g)
    empties = [i for i in range(81) if g[i] == 0]
    progress = True
    while empties and progress:
        progress = False
        remaining = []
        for i in empties:
            r, c, b = ROW_OF[i], COL_OF[i], BOX_OF[i]
            mask = FULL_MASK & ~(rows[r] | cols[c] | boxes[b])
            if mask and mask & (mask - 1) == 0:
                rows[r] |= mask
                cols[c] |= mask
                boxes[b] |= mask
                g[i] = mask.bit_length() - 1
                progress = True
            else:
                remaining.append(i)
        empties = remaining
    return not empties


def _solves_box(grid):
    g = list(grid)
    rows, cols, boxes = unit_masks(g)
    empty = sum(v == 0 for v in g)
    progress = True
    while empty and progress:
        progress = False
        for d in DIGITS:
            bit = 1 << d
            for b in range(9):
                if boxes[b] & bit:
                    continue
                spot = -1
                for i in BOXES[b]:
                    if g[i] == 0 and not (rows[ROW_OF[i]] | cols[COL_OF[i]]) & bit:
                        if spot >= 0:
                            spot = -2
                            break
                        spot = i
                if spot >= 0:
                    g[spot] = d
                    rows[ROW_OF[spot]] |= bit
                    cols[COL_OF[spot]] |= bit
                    boxes[b] |= bit
                    empty -= 1
                    progress = True
    return empty == 0


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
        if _solves_naked(grid) and _solves_box(grid):
            givens -= 1
        else:
            grid[i] = removed
    return grid


def generate_puzzle(rng, givens_min=GIVENS_MIN, givens_max=GIVENS_MAX):
    from .grid import random_full_grid, transform

    full = transform(random_full_grid(rng), rng)
    target = rng.randint(givens_min, givens_max)
    return dig_to(full, target, rng), full
