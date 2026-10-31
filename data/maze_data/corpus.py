"""Maze policy for ``data.strategy_corpus``: what differs from sudoku.

One perfect 8x8 maze, five search strategies, five sequences sharing the
question and the answer; only difficult mazes are kept.
The label that names the strategy is part of the
question, at positions 196..203, for the router alone.
"""

from . import FORMAT_VERSION  # noqa: F401  (part of the policy contract)
from .difficulty import BFS_EXTRA, MIN_PATH, divergences, rejection
from .format import build_sequence, parse_sequence
from .grid import CELL_NAMES, CELLS, ascii_draw, maze_key, random_maze, unique_path
from .tracers import STRATEGIES
from .verify import verify_maze, verify_sequence
from .vocab import SYMBOLS, UNK, decode, encode  # noqa: F401  (policy contract)

NAME = "maze"
UNIT = "mazes"
ID_FIELD = "maze_id"
SPLITS = ("train", "val", "test", "test_heldout_pairs", "test_swap")
VERIFICATION = (
    "every maze passed path, dead-end-complement, question round-trip, "
    "trace, identify, layout and <unk> checks at build time"
)


def add_arguments(parser):
    parser.add_argument("--min-path", type=int, default=MIN_PATH,
                        help="Shortest accepted path, in cells.")
    parser.add_argument("--bfs-extra", type=int, default=BFS_EXTRA,
                        help="Fewest cells BFS must expand off the path.")
    parser.add_argument("--n-swap", type=int, default=20_000,
                        help="Test mazes moved into test_swap.")


def params(args):
    return {
        "min_path": args.min_path,
        "bfs_extra": args.bfs_extra,
        "n_swap": args.n_swap,
    }


def generate(rng, params):
    maze = random_maze(rng)
    s, g = rng.sample(range(CELLS), 2)
    path = unique_path(maze, s, g)
    if len(path) < params["min_path"]:
        return "short_path"
    traces = {k: STRATEGIES[k](maze, s, g) for k in STRATEGIES}
    reason = rejection(traces, path, params["min_path"], params["bfs_extra"])
    if reason is not None:
        return reason
    sequences = {
        k: build_sequence(maze, s, g, k, traces[k], path) for k in STRATEGIES
    }
    verify_maze(maze, s, g, traces, path, sequences)
    ids = {k: encode(seq) for k, seq in sequences.items()}
    divergence = divergences(traces)
    return {
        "key": f"{maze_key(maze)}:{s}:{g}",
        "sequences": [
            (k, bytes(ids[k]), {
                "strategy": f"S{k}",
                "start": CELL_NAMES[s],
                "goal": CELL_NAMES[g],
                "path_len": len(path),
                "n_trace": len(traces[k]),
                "n_tokens": len(ids[k]),
                "first_divergence": divergence,
            })
            for k in STRATEGIES
        ],
    }


def describe(meta):
    return (
        f"{meta['maze_id']} {meta['strategy']} start={meta['start']} "
        f"goal={meta['goal']} path={meta['path_len']} trace={meta['n_trace']} "
        f"tokens={meta['n_tokens']}"
    )


def render(ids):
    maze, s, g, _k, _trace, _path = parse_sequence(decode(ids))
    return ascii_draw(maze, s, g)


def sample_entry(meta, ids):
    return f"# {describe(meta)}\n{render(ids)}\n{decode(ids)}\n"


def recheck(ids):
    verify_sequence(decode(ids))
