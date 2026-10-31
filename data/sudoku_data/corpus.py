"""Sudoku policy for ``data.strategy_corpus``: what differs from the maze.

One 9x9 puzzle dug to be solvable by singles, five solving strategies, five
sequences sharing the question and the answer. A puzzle
is rejected only when a sequence would exceed ``max_tokens``.
"""

from . import FORMAT_VERSION  # noqa: F401  (part of the policy contract)
from .dig import GIVENS_MAX, GIVENS_MIN, generate_puzzle
from .format import build_sequence
from .grid import to_str
from .tracers import STRATEGIES
from .verify import verify_puzzle
from .vocab import SYMBOLS, UNK, decode, encode  # noqa: F401  (policy contract)

NAME = "sudoku"
UNIT = "puzzles"
ID_FIELD = "puzzle_id"
SPLITS = ("train", "val", "test", "test_heldout_pairs")
VERIFICATION = (
    "all puzzles passed replay, classifier, solution-agreement and <unk> "
    "checks at build time"
)


def add_arguments(parser):
    parser.add_argument("--givens", type=int, nargs=2,
                        default=[GIVENS_MIN, GIVENS_MAX],
                        help="Inclusive range of clue counts.")
    parser.add_argument("--max-tokens", type=int, default=1000,
                        help="Reject puzzles with a longer sequence.")


def params(args):
    return {"givens": list(args.givens), "max_tokens": args.max_tokens}


def generate(rng, params):
    givens_min, givens_max = params["givens"]
    puzzle, full = generate_puzzle(rng, givens_min, givens_max)
    traces = {k: STRATEGIES[k](puzzle) for k in STRATEGIES}
    sequences = {
        k: build_sequence(puzzle, k, steps, sol)
        for k, (steps, sol) in traces.items()
    }
    verify_puzzle(puzzle, full, traces, sequences)
    ids = {k: encode(seq) for k, seq in sequences.items()}
    if max(len(v) for v in ids.values()) > params["max_tokens"]:
        return "max_tokens"
    givens = sum(v != 0 for v in puzzle)
    s5_steps = len(traces[5][0])
    return {
        "key": to_str(puzzle),
        "sequences": [
            (k, bytes(ids[k]), {
                "strategy": f"S{k}",
                "givens": givens,
                "n_steps": len(traces[k][0]),
                "n_tokens": len(ids[k]),
                "s5_steps": s5_steps,
            })
            for k in sorted(ids)
        ],
    }


def sample_entry(meta, ids):
    return decode(ids)
